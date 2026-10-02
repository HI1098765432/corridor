"""Headless entry point.

The graphical application is the product, but every analysis it can run must
also run without a display: for batches, for regression tests, and for anyone
who wants to script it.

Exit codes, so a batch script can tell the refusals apart:

    0  the analysis ran and its results were written
    1  any other failure (the message is printed, not a traceback)
    2  a bad command line, or the input file does not exist
    3  the validated segmentation model is missing or fails its checksum
    4  the file's axis order is ambiguous; pass --axes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import _version, app_meta
from .core import pipeline
from .core.config import (
    CHANNEL_CONSTRAINT_OFF,
    CalibrationConfig,
    ImportConfig,
    MeasurementConfig,
    RunConfig,
    SegmentationConfig,
    TrackingConfig,
)
from .core.imaging import AmbiguousAxes
from .core.model_registry import ModelUnavailable

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_MODEL_UNAVAILABLE = 3
EXIT_AMBIGUOUS_AXES = 4


class ConsoleProgress:
    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self._stage = ""

    def stage(self, name: str, detail: str = "") -> None:
        self._stage = name
        if not self.quiet:
            suffix = f" - {detail}" if detail else ""
            print(f"[{name}]{suffix}", flush=True)

    def step(self, done: int, total: int) -> None:
        if self.quiet or not total:
            return
        end = "\n" if done == total else "\r"
        print(f"  {self._stage}: {done}/{total}", end=end, flush=True)

    def cancelled(self) -> bool:
        return False


def reference_point(text: str) -> tuple[float, ...]:
    """``X,Y`` or ``X,Y,Z``: pixels (and slices), the frame of ``tracks.csv``'s x_px/y_px."""
    parts = [p.strip() for p in str(text).split(",")]
    try:
        values = tuple(float(p) for p in parts)
    except ValueError:
        values = ()
    if len(values) not in (2, 3):
        raise argparse.ArgumentTypeError(
            f"expected X,Y or X,Y,Z in pixels (Z in slices), got {text!r}"
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="corridor",
        description=f"{app_meta.APP_NAME} - {app_meta.APP_TAGLINE}",
        epilog=(
            "Exit codes: 0 done, 1 failed, 2 bad command line or missing file, "
            "3 validated model missing or checksum mismatch, 4 ambiguous axis order "
            "(pass --axes)."
        ),
    )
    p.add_argument("input", nargs="?", help="time-lapse TIFF to analyse")
    p.add_argument("-o", "--output", help="directory for the results")
    p.add_argument("--gpu", action="store_true", help="use the GPU if one is available")

    seg = p.add_argument_group("segmentation (always the validated model)")
    seg.add_argument("--cellprob", type=float, default=None)
    seg.add_argument("--flow", type=float, default=None)
    seg.add_argument("--diameter", type=float, default=None, help="pixels")
    seg.add_argument("--min-extent", type=int, default=None, help="pixels")
    seg.add_argument(
        "--labels",
        default=None,
        metavar="LABELS.tif",
        help="measure and track this integer label image instead of segmenting "
        "(the only route for 3-D stacks)",
    )

    trk = p.add_argument_group("tracking")
    trk.add_argument("--max-gap", type=int, default=None, help="frames")
    trk.add_argument("--max-speed", type=float, default=None, help="um per minute")
    trk.add_argument(
        "--position-sigma", type=float, default=None,
        help="isotropic centroid noise floor, um",
    )
    trk.add_argument("--unmatched", type=float, default=None, help="chi-square units")
    trk.add_argument(
        "--no-channel-constraint",
        action="store_true",
        help="never refuse a link between two lanes, even when walls were detected",
    )

    imp = p.add_argument_group("import")
    imp.add_argument(
        "--axes",
        default=None,
        help="axis order for a file whose metadata does not establish it, e.g. TYX or ZYX",
    )
    imp.add_argument(
        "--channel-index", "--channel",
        dest="channel_index", type=int, default=None,
        help="channel to analyse in a multichannel file, numbered from 0",
    )

    cal = p.add_argument_group("calibration and measurement")
    cal.add_argument("--pixel-size", type=float, default=None, help="um per pixel")
    cal.add_argument("--frame-interval", type=float, default=None, help="minutes")
    cal.add_argument("--z-step", type=float, default=None, help="um between Z planes (3-D)")
    cal.add_argument(
        "--reference-point",
        type=reference_point,
        default=None,
        metavar="X,Y[,Z]",
        help="point for distance_from_reference (MTrackJ D2R), in pixels (Z in slices)",
    )

    # Removed 1.x options. Parsed only so that they are refused with a reason
    # instead of argparse's bare "unrecognized arguments", and never shown.
    for flag in REMOVED_OPTIONS:
        p.add_argument(flag, dest=_dest(flag), default=None, help=argparse.SUPPRESS)

    p.add_argument("--napari", action="store_true", help="open Napari after saving")
    p.add_argument("--gui", action="store_true", help="force the desktop application")
    p.add_argument(
        "--headless",
        action="store_true",
        help="analyse without opening the application (implied by --output)",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="check that this installation is complete and working",
    )
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--version", action="version", version=f"{app_meta.APP_NAME} {_version.__version__}"
    )
    return p


#: Why ``--model`` / ``--builtin-model`` are refused rather than ignored: the
#: service would run the validated model regardless, and a run that silently
#: ignored the model its command line named would be a different analysis
#: from the one the user believes they ran.
REMOVED_MODEL_OPTIONS = (
    "--model and --builtin-model were removed in Corridor 2.0: it runs only the "
    "validated segmentation model, verified by checksum. For research with another "
    "model, set CORRIDOR_DEVELOPER=1 and CORRIDOR_DEVELOPER_MODEL=<file>; the run is "
    "then recorded as a developer override."
)

#: The axis options configured a migration direction, which 2.0 does not have.
REMOVED_AXIS_OPTIONS = (
    "{flags} were removed in Corridor 2.0: tracking no longer uses a migration axis. "
    "Each cell's own shape sets its positional uncertainty (--position-sigma sets the "
    "isotropic floor), and lanes measured from the channel walls keep cells in their "
    "own channel (--no-channel-constraint switches that off)."
)

REMOVED_OPTIONS = (
    "--model", "--builtin-model",
    "--axis", "--axis-angle", "--sigma-along", "--sigma-across",
)


def _dest(flag: str) -> str:
    return "removed_" + flag.lstrip("-").replace("-", "_")


def removed_options(args: argparse.Namespace) -> str | None:
    """The refusal for 1.x options that would otherwise be silently ignored."""
    if getattr(args, _dest("--model"), None) or getattr(args, _dest("--builtin-model"), None):
        return REMOVED_MODEL_OPTIONS
    axis = [
        flag for flag in ("--axis", "--axis-angle", "--sigma-along", "--sigma-across")
        if getattr(args, _dest(flag), None) is not None
    ]
    if axis:
        return REMOVED_AXIS_OPTIONS.format(flags=", ".join(axis))
    return None


def config_from_args(args: argparse.Namespace) -> RunConfig:
    refusal = removed_options(args)
    if refusal:
        raise ValueError(refusal)
    # No model fields: the registry resolves the one validated model.
    seg = SegmentationConfig(use_gpu=bool(args.gpu))
    if args.cellprob is not None:
        seg.cellprob_threshold = args.cellprob
    if args.flow is not None:
        seg.flow_threshold = args.flow
    if args.diameter is not None:
        seg.diameter = args.diameter
    if args.min_extent is not None:
        seg.min_extent_px = args.min_extent

    trk = TrackingConfig()
    if args.max_gap is not None:
        trk.max_gap = args.max_gap
    if args.max_speed is not None:
        trk.max_speed_um_per_min = args.max_speed
    if args.position_sigma is not None:
        trk.position_sigma_um = args.position_sigma
    if args.unmatched is not None:
        trk.unmatched_chi2 = args.unmatched
    if args.no_channel_constraint:
        trk.channel_constraint = CHANNEL_CONSTRAINT_OFF

    cal = CalibrationConfig(
        pixel_size_um=args.pixel_size,
        frame_interval_min=args.frame_interval,
        z_step_um=args.z_step,
    )
    measurement = MeasurementConfig(reference_point_px=args.reference_point)

    imp = ImportConfig()
    if args.axes:
        imp.axes = args.axes.strip().upper()
    if args.channel_index is not None:
        imp.channel_index = args.channel_index
    if args.labels:
        imp.labels_path = str(Path(args.labels))

    src = Path(args.input)
    out = Path(args.output) if args.output else src.parent / f"{src.stem}_corridor"
    return RunConfig(
        input_path=str(src), output_dir=str(out),
        segmentation=seg, tracking=trk, calibration=cal,
        measurement=measurement, import_=imp,
    )


def wants_interface(args: argparse.Namespace) -> bool:
    """Decide between opening the application and running an analysis.

    Opening a TIFF is what a double-click and a file association do, so a bare
    file argument must show the application. Running silently is the unusual
    request, and is asked for explicitly -- by naming an output directory, or
    with --headless.
    """
    if args.gui:
        return True
    if args.self_test:
        return False
    if not args.input:
        return True
    return not (args.headless or args.output)


def describe_model(result: pipeline.AnalysisResult) -> str:
    """One line naming what produced the masks: the model by id, version and checksum."""
    seg = result.segmentation
    if result.model is None:
        sha = (seg.labels_sha256 or "")[:12]
        name = Path(seg.labels_path).name if seg.labels_path else "a label file"
        return f"imported from {name} (sha256 {sha}...), no model ran"
    spec = result.model.spec
    text = f"{spec.model_id} {spec.model_version} (sha256 {result.model.sha256[:12]}...)"
    if result.model.developer_override:
        text += " -- DEVELOPER OVERRIDE, not the validated model"
    return text


def describe_lanes(result: pipeline.AnalysisResult) -> str:
    geometry = result.geometry
    if not geometry.lanes:
        return "no lanes (no channel walls found); tracking not constrained to any"
    gate = "links between lanes refused" if geometry.applied else "lanes do not gate links"
    pitch = f", pitch {geometry.pitch_px:.0f} px" if geometry.pitch_px else ""
    return f"{geometry.n_lanes} lane(s) from {geometry.source}{pitch}; {gate}"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    refusal = removed_options(args)
    if refusal:
        parser.error(refusal)

    if args.self_test:
        from .selftest import run as run_self_test

        return run_self_test(verbose=not args.quiet)

    if wants_interface(args):
        from .ui.app import run_app

        return run_app([] if not args.input else [args.input])

    config = config_from_args(args)
    progress = ConsoleProgress(args.quiet)
    try:
        result = pipeline.run_analysis(config, progress)
    except ModelUnavailable as exc:
        # The contract's sentence and every path tried, verbatim.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_MODEL_UNAVAILABLE
    except AmbiguousAxes as exc:
        choices = ", ".join(exc.choices) or "none found"
        print(
            f"error: {exc}\n"
            f"The axis order of this file is ambiguous. Possible orders: {choices}.\n"
            f"Run again with --axes ORDER, e.g. --axes {exc.choices[0] if exc.choices else 'TYX'}.",
            file=sys.stderr,
        )
        return EXIT_AMBIGUOUS_AXES
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:  # noqa: BLE001 - a CLI should print, not traceback
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED

    m = result.metadata
    shape = " x ".join(str(n) for n in m.axes_shape)
    print()
    print(f"{m.path.name}: {shape} ({m.axes}, {result.dimensionality})")
    print(
        f"  pixel size     {m.pixel_size_um.describe(' um/px')}\n"
        f"  frame interval {m.frame_interval_min.describe(' min')}"
    )
    if result.dimensionality == "3D":
        print(f"  z step         {m.z_step_um.describe(' um')}")
    print(f"  model          {describe_model(result)}")
    print(f"  lanes          {describe_lanes(result)}")
    raw = sum(d.raw_count for d in result.segmentation.diagnostics)
    kept = sum(d.kept_count for d in result.segmentation.diagnostics)
    recovered = result.n_detections - result.n_primary_detections
    print(
        f"  segmentation   {raw} raw instances, {kept} kept, {raw - kept} filtered out; "
        f"{recovered} recovered"
    )
    print(f"  tracking       {result.n_tracks} tracks, {len(result.usable_tracks)} with velocity")
    for issue in result.issues[:8]:
        where = f" (frame {issue.frame})" if issue.frame is not None else ""
        print(f"  ! {issue.severity:8s} {issue.title}{where}")
    print(f"  results        {result.output_dir}")

    if args.napari:
        from .viz.napari_qc import open_in_napari

        open_in_napari(result)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
