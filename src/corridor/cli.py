"""Headless entry point.

The graphical application is the product, but every analysis it can run must
also run without a display: for batches, for regression tests, and for anyone
who wants to script it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import app_meta
from .core import pipeline
from .core.config import (
    AXIS_MODES,
    CalibrationConfig,
    ConfinementConfig,
    ImportConfig,
    RunConfig,
    SegmentationConfig,
    TrackingConfig,
)


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="corridor",
        description=f"{app_meta.APP_NAME} - {app_meta.APP_TAGLINE}",
    )
    p.add_argument("input", nargs="?", help="time-lapse TIFF to analyse")
    p.add_argument("-o", "--output", help="directory for the results")
    # Kept only so that 1.x command lines fail loudly instead of running a
    # different model from the one they name: Corridor 2.0 runs only the
    # registered, checksum-verified model (contract section 2).
    p.add_argument("--model", help=argparse.SUPPRESS)
    p.add_argument("--builtin-model", default=None, help=argparse.SUPPRESS)
    p.add_argument("--gpu", action="store_true", help="use the GPU if one is available")

    seg = p.add_argument_group("segmentation")
    seg.add_argument("--cellprob", type=float, default=None)
    seg.add_argument("--flow", type=float, default=None)
    seg.add_argument("--diameter", type=float, default=None)
    seg.add_argument("--min-extent", type=int, default=None)

    trk = p.add_argument_group("tracking")
    trk.add_argument("--max-gap", type=int, default=None)
    trk.add_argument("--max-speed", type=float, default=None, help="um per minute")
    trk.add_argument("--sigma-along", type=float, default=None, help="um")
    trk.add_argument("--sigma-across", type=float, default=None, help="um")
    trk.add_argument("--unmatched", type=float, default=None, help="chi-square units")

    imp = p.add_argument_group("import")
    imp.add_argument(
        "--axes",
        default=None,
        help="axis order for a file whose metadata does not establish it, e.g. TYX or ZYX",
    )
    imp.add_argument(
        "--channel", type=int, default=None, help="channel to analyse, numbered from 0"
    )

    geo = p.add_argument_group("geometry and calibration")
    geo.add_argument("--axis", choices=AXIS_MODES, default=None)
    geo.add_argument("--axis-angle", type=float, default=None, help="degrees")
    geo.add_argument("--pixel-size", type=float, default=None, help="um per pixel")
    geo.add_argument("--frame-interval", type=float, default=None, help="minutes")

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
    p.add_argument("--version", action="version", version=f"{app_meta.APP_NAME} {app_meta.APP_VERSION}")
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


def removed_options(args: argparse.Namespace) -> str | None:
    """The refusal for 1.x options that would otherwise be silently ignored."""
    if getattr(args, "model", None) or getattr(args, "builtin_model", None):
        return REMOVED_MODEL_OPTIONS
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
    if args.sigma_along is not None:
        trk.sigma_along_um = args.sigma_along
    if args.sigma_across is not None:
        trk.sigma_perp_um = args.sigma_across
    if args.unmatched is not None:
        trk.unmatched_chi2 = args.unmatched

    conf = ConfinementConfig()
    if args.axis:
        conf.mode = args.axis
    if args.axis_angle is not None:
        conf.angle_deg = args.axis_angle

    cal = CalibrationConfig(
        pixel_size_um=args.pixel_size, frame_interval_min=args.frame_interval
    )

    imp = ImportConfig()
    if getattr(args, "axes", None):
        imp.axes = args.axes
    if getattr(args, "channel", None) is not None:
        imp.channel_index = args.channel

    src = Path(args.input)
    out = Path(args.output) if args.output else src.parent / f"{src.stem}_corridor"
    return RunConfig(
        input_path=str(src), output_dir=str(out),
        segmentation=seg, tracking=trk, confinement=conf, calibration=cal,
        import_=imp,
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
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - a CLI should print, not traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1

    m = result.metadata
    print()
    print(f"{m.path.name}: {m.n_frames} frames of {m.height}x{m.width}, axes {m.axes_raw}")
    print(
        f"  pixel size     {m.pixel_size_um.describe(' um/px')}\n"
        f"  frame interval {m.frame_interval_min.describe(' min')}"
    )
    print(
        f"  migration axis {result.axis.tilt_from_vertical_deg:+.1f} deg from vertical "
        f"({result.axis.describe()}), {len(result.axis.channels)} channel(s)"
    )
    raw = sum(d.raw_count for d in result.segmentation.diagnostics)
    kept = sum(d.kept_count for d in result.segmentation.diagnostics)
    print(f"  segmentation   {raw} raw instances, {kept} kept, {raw - kept} filtered out")
    print(f"  tracking       {result.n_tracks} tracks, {len(result.usable_tracks)} with velocity")
    for issue in result.issues[:8]:
        where = f" (frame {issue.frame})" if issue.frame is not None else ""
        print(f"  ! {issue.severity:8s} {issue.title}{where}")
    print(f"  results        {result.output_dir}")

    if args.napari:
        from .viz.napari_qc import open_in_napari

        open_in_napari(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
