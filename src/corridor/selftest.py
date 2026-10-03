"""Check that an installed copy is complete and working.

A packaged application can be missing a module that only a particular code
path touches, and the usual way that surfaces is a researcher pressing Analyse
and getting an error. This walks the paths instead: it imports every subsystem,
resolves the validated model through the registry (hashing it before anything
loads it), runs the full torch/Cellpose path on a synthetic frame, tracks
known trajectories and checks the velocity and MSD arithmetic, round-trips
every output format the 2.0 schema writes, and exercises Napari's layer
construction and plugin discovery.

Run it with ``corridor --self-test``.
"""

from __future__ import annotations

import math
import platform
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import app_meta, resources


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    error: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, fn: Callable[[], str]) -> bool:
        try:
            detail = fn() or ""
        except Exception as exc:  # noqa: BLE001 - the point is to catch everything
            self.checks.append(
                Check(name, False, error=f"{type(exc).__name__}: {exc}")
            )
            return False
        self.checks.append(Check(name, True, detail))
        return True

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]


def _check_imports() -> str:
    import numpy
    import scipy
    import skimage
    import tifffile
    import torch

    import cellpose

    version = str(cellpose.version)
    if not version.startswith("3."):
        raise RuntimeError(f"Cellpose 3.x is required; found {version}")
    return (
        f"cellpose {version}, torch {torch.__version__}, numpy {numpy.__version__}, "
        f"scipy {scipy.__version__}, skimage {skimage.__version__}, "
        f"tifffile {tifffile.__version__}"
    )


def _check_qt() -> str:
    from PySide6 import QtCore, QtGui, QtWidgets  # noqa: F401

    return f"PySide6 {QtCore.__version__}"


def _resolve_validated_model():
    """The production model, verified by the registry -- never an override.

    A developer override is a deliberate research setting, and a self-test
    that passed under one would certify a model nobody validated.
    """
    from .core import model_registry

    resolved = model_registry.resolve_model("2D")
    if resolved.developer_override:
        raise RuntimeError(
            f"{model_registry.ENV_DEVELOPER} and {model_registry.ENV_DEVELOPER_MODEL} are set, "
            f"so {resolved.path} would run instead of the validated model. Unset them to "
            "check this installation."
        )
    return resolved


def _check_model() -> str:
    """Resolve through the registry: every candidate is hashed before use."""
    resolved = _resolve_validated_model()
    spec = resolved.spec
    if resolved.sha256 != spec.sha256 or spec.sha256 != resources.BUNDLED_MODEL_SHA256:
        raise RuntimeError(
            f"checksum {resolved.sha256[:16]}... does not match the registry "
            f"({spec.sha256[:16]}...) and the application ({resources.BUNDLED_MODEL_SHA256[:16]}...)"
        )
    return f"{spec.model_id} {spec.model_version} ({resolved.sha256[:12]}...) at {resolved.path}"


def _check_segmentation() -> str:
    """Load the model and run it. This is the path torch pruning could break."""
    import numpy as np

    from .core.config import SegmentationConfig
    from .core.segmentation import SegmentationService

    # The service re-hashes the file immediately before Cellpose reads it.
    service = SegmentationService(SegmentationConfig(), model=_resolve_validated_model())
    rng = np.random.default_rng(0)
    frame = rng.normal(1000, 30, size=(160, 96)).astype(np.uint16)
    frame[40:120, 44:56] += 2500  # something cell-shaped to find
    mask, _, _ = service.segment_frame(frame)
    if mask.shape != frame.shape:
        raise RuntimeError(f"mask shape {mask.shape} does not match input {frame.shape}")
    # What matters here is that the whole torch/cellpose path executed and
    # returned a valid label image. A synthetic frame is not real microscopy,
    # so finding nothing in it is not a failure.
    return (
        f"the full torch/Cellpose path ran and returned a {mask.shape} label image "
        f"({int(mask.max())} instance(s) in synthetic noise)"
    )


#: The supplied KK2 calibration (0.467 um/px, 20.0 min frames), so the check
#: exercises the unit conversions real runs use.
_PIXEL_UM = 0.467060342995564
_FRAME_MIN = 20.006894938151042
_N_FRAMES = 6


#: Body width of the synthetic cells (minor axis, px).
_BODY_WIDTH_PX = 11.0


def _synthetic_cells():
    """Two elongated cells moving at constant velocity, in different directions.

    One runs down the image and one diagonally, so nothing about the check
    depends on a migration direction: the tracker is given none, and each
    cell's uncertainty comes from its own body.  Steps (px/frame) are well
    inside the 5 um/min speed gate (20 px/frame = 0.47 um/min).

    Their paths converge and diverge again: the centroids are 38.8, 26.0,
    16.1, 16.1, 26.0 and 38.8 px apart at frames 0-5, so at frames 2 and 3
    they are 1.5 body widths (11 px) apart.  That is close enough for identity
    to be a real question -- a linker that matches by position alone swaps
    them between frames 2 and 3 (cell 2 at frame 2 is 4.5 px from cell 1 at
    frame 3, and the swapped assignment is also the cheaper one in total
    distance, 35.8 against 37.0 px); see tests/test_selftest.py.
    """
    from .core.detections import Detection

    cells = {
        1: {"start": (40.0, 30.0), "step": (0.0, 20.0), "orientation": 0.0},
        # orientation_rad is the major axis angle with (x, y) = (sin, cos):
        # pi/4 lies along this cell's own (12, 12) step.
        2: {"start": (18.0, 62.0), "step": (12.0, 12.0), "orientation": math.pi / 4},
    }
    detections = []
    for label, cell in cells.items():
        (x0, y0), (sx, sy) = cell["start"], cell["step"]
        for t in range(_N_FRAMES):
            x, y = x0 + sx * t, y0 + sy * t
            detections.append(
                Detection(
                    frame=t, label=label, x=x, y=y, area_px=800.0,
                    bbox=(int(y - 45), int(x - 6), int(y + 45), int(x + 6)),
                    extent_px=90, eccentricity=0.99,
                    orientation_rad=cell["orientation"], major_axis_px=90.0,
                    minor_axis_px=_BODY_WIDTH_PX, solidity=0.95, touches_border=False,
                )
            )
    return cells, detections


def _track_synthetic():
    from .core.config import Scale, TrackingConfig
    from .core.measurements import frame_rows, msd_rows, summarise
    from .core.tracking import track_detections

    scale = Scale.from_values(_PIXEL_UM, _FRAME_MIN)
    cells, detections = _synthetic_cells()
    tracks, events = track_detections(detections, _N_FRAMES, scale, TrackingConfig())
    msd = msd_rows(tracks, scale)
    rows = frame_rows(tracks, scale)
    summaries = summarise(tracks, scale, msd_rows=msd)
    return scale, cells, detections, tracks, events, rows, msd, summaries


def _close(a: float | None, b: float, rel: float = 1e-9) -> bool:
    return a is not None and math.isclose(float(a), b, rel_tol=rel, abs_tol=1e-12)


def _check_tracking() -> str:
    """Identity, speed in um/min and um/hr, and MSD = v^2 tau^2, with no axis anywhere."""
    scale, cells, _, tracks, _, rows, msd, _ = _track_synthetic()
    if len(tracks) != len(cells):
        raise RuntimeError(f"expected {len(cells)} tracks, got {len(tracks)}")
    for tr in tracks:
        labels = {o.det_label for o in tr.observations}
        if len(labels) != 1 or tr.n_obs != _N_FRAMES:
            raise RuntimeError(
                f"track {tr.id} mixes cells {sorted(labels)} or is incomplete ({tr.n_obs} obs)"
            )

    checked = []
    for tr in tracks:
        (label,) = {o.det_label for o in tr.observations}
        sx, sy = cells[label]["step"]
        step_um = math.hypot(sx, sy) * _PIXEL_UM
        v_min = step_um / _FRAME_MIN
        for row in (r for r in rows if r["track_id"] == tr.id and r["observation_index"] > 0):
            if not _close(row["speed_um_per_min"], v_min):
                raise RuntimeError(f"speed {row['speed_um_per_min']} um/min, expected {v_min}")
            if not _close(row["speed_um_per_hr"], v_min * 60.0):
                raise RuntimeError(f"speed {row['speed_um_per_hr']} um/hr, expected {v_min * 60}")
        # A constant-velocity track: every pair k frames apart is k steps
        # apart, so MSD(tau) = (v tau)^2 exactly, in um^2.  Every lag must be
        # there, each averaged over every pair it has: a loop over rows that
        # do not exist would pass on an msd_rows that returned nothing.
        own = sorted((r for r in msd if r["track_id"] == tr.id), key=lambda r: r["lag_frames"])
        lags = [(r["lag_frames"], r["n_pairs"]) for r in own]
        expected_lags = [(k, _N_FRAMES - k) for k in range(1, _N_FRAMES)]
        if lags != expected_lags:
            raise RuntimeError(
                f"MSD of track {tr.id} has (lag, pairs) {lags}, expected {expected_lags}"
            )
        for r in own:
            tau_min = r["lag_frames"] * _FRAME_MIN
            if not _close(r["msd_um2"], (v_min * tau_min) ** 2, rel=1e-9):
                raise RuntimeError(
                    f"MSD {r['msd_um2']} um^2 at lag {r['lag_frames']}, "
                    f"expected {(v_min * tau_min) ** 2}"
                )
        checked.append(f"{v_min:.4f} um/min = {v_min * 60:.2f} um/hr")
    return (
        f"{len(tracks)} identities kept apart at 1.5 body widths; speeds {', '.join(checked)}; "
        f"MSD = v^2 tau^2 at all {_N_FRAMES - 1} lags"
    )


def _synthetic_result(directory: Path):
    """A complete AnalysisResult of the synthetic cells, built by the real code."""
    import numpy as np

    from .core import pipeline
    from .core.config import RunConfig
    from .core.geometry import ChannelGeometry
    from .core.imaging import SOURCE_USER, Calibrated, StackMetadata
    from .core.segmentation import SegmentationOutput

    scale, _, detections, tracks, events, rows, msd, summaries = _track_synthetic()
    metadata = StackMetadata(
        path=directory / "synthetic.tif", n_frames=_N_FRAMES, height=200, width=240,
        dtype="uint16", axes_raw="TYX", axes_interpretation="TYX",
        pixel_size_um=Calibrated(_PIXEL_UM, SOURCE_USER),
        frame_interval_min=Calibrated(_FRAME_MIN, SOURCE_USER),
        axes="TYX", axes_used="TYX",
    )
    segmentation = SegmentationOutput(
        masks=np.zeros((_N_FRAMES, 200, 240), np.int32),
        raw_masks=np.zeros((_N_FRAMES, 200, 240), np.int32),
        detections=detections, diagnostics=[], model_path="", model_sha256=None,
        cellpose_version="", used_gpu=False,
    )
    config = RunConfig(input_path=str(metadata.path), output_dir=str(directory / "run"))
    geometry = ChannelGeometry()
    manifest = pipeline.build_manifest(
        config, metadata, scale, geometry, segmentation, tracks, summaries,
        elapsed_s=0.0, output_dir=directory / "run",
    )
    return pipeline.AnalysisResult(
        config=config, metadata=metadata, scale=scale, geometry=geometry,
        segmentation=segmentation, tracks=tracks, events=events, rows=rows,
        summaries=summaries, issues=[], msd=msd, manifest=manifest,
        output_dir=directory / "run",
    )


def _check_outputs() -> str:
    """Write a whole result, read it back, export a track to CSV and XLSX."""
    import json
    import zipfile
    from xml.etree import ElementTree

    from .core import export, pipeline
    from .store.project import export_track, load_analysis, read_table

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        result = _synthetic_result(directory)
        out = directory / "run"
        pipeline.save_result(result, out)

        manifest = json.loads((out / pipeline.F_MANIFEST).read_text(encoding="utf-8"))
        if manifest.get("schema_version") != export.SCHEMA_VERSION:
            raise RuntimeError(f"run.json schema_version is {manifest.get('schema_version')}")
        saved = load_analysis(out)
        if len(saved.tracks) != len(result.rows):
            raise RuntimeError(f"tracks.csv: wrote {len(result.rows)} rows, read {len(saved.tracks)}")
        msd = read_table(out / pipeline.F_MSD)
        if len(msd) != len(result.msd) or not msd:
            raise RuntimeError(f"track_msd.csv: wrote {len(result.msd)} rows, read {len(msd)}")
        for written, read in zip(result.msd, msd):
            if not _close(read["msd_um2"], written["msd_um2"], rel=1e-8):
                raise RuntimeError("track_msd.csv did not round-trip its MSD values")
        if {int(s["track_id"]) for s in saved.summaries} != {s.track_id for s in result.summaries}:
            raise RuntimeError("track_summary.csv did not round-trip its track ids")
        masks = export.load_masks(out / pipeline.F_MASKS)
        if masks.shape != result.segmentation.masks.shape:
            raise RuntimeError(f"masks.npz came back {masks.shape}")

        track_id = saved.track_ids()[0]
        csv_rows = read_table(export_track(saved, track_id, directory / "one.csv"))
        if len(csv_rows) != len(saved.rows_for_track(track_id)):
            raise RuntimeError("the per-track CSV export lost rows")
        xlsx = export_track(saved, track_id, directory / "one.xlsx", fmt="xlsx")
        with zipfile.ZipFile(xlsx) as zf:
            sheets = [n for n in zf.namelist() if n.startswith("xl/worksheets/sheet")]
            for name in sheets:
                ElementTree.fromstring(zf.read(name))  # well-formed XML or raises
        if len(sheets) != 4:
            raise RuntimeError(f"the XLSX export has {len(sheets)} sheets, expected 4")

        export.write_json(directory / "nan.json", {"x": float("nan")})
        if json.loads((directory / "nan.json").read_text(encoding="utf-8"))["x"] is not None:
            raise RuntimeError("NaN was not sanitised out of the manifest")
    return (
        "run.json (schema 2), tracks, summaries, track_msd, masks, per-track CSV and "
        "XLSX all round-trip"
    )


def _check_store() -> str:
    from .store import db

    directory = db.app_data_dir()
    store = db.Store()
    try:
        count = len(store.recent_projects())
    finally:
        store.close()
    return f"{count} project(s) in {directory}"


def _check_napari() -> str:
    """Check what packaging can actually break, without opening a window.

    Constructing a real Viewer needs a display and an OpenGL context, and
    tearing one down outside a running Qt event loop crashes the process. What
    a frozen build genuinely risks losing is different: Napari finds its own
    components through package metadata and npe2 manifests rather than through
    imports, and those are exactly what a bundler drops. So this verifies
    discovery, not rendering.
    """
    from .viz.napari_qc import NapariUnavailable

    try:
        import napari
        from napari.components import ViewerModel
    except ImportError as exc:
        raise NapariUnavailable("napari is not present in this build") from exc

    # The viewer *model* holds the layer logic and needs no display, so the
    # layer construction in napari_qc can still be exercised for real.
    import numpy as np

    model = ViewerModel()
    stack = np.zeros((3, 32, 16), dtype=np.uint16)
    masks = np.zeros((3, 32, 16), dtype=np.int32)
    masks[:, 8:20, 6:10] = 1
    model.add_image(stack, name="microscopy")
    model.add_labels(masks, name="cell masks")
    model.add_points(np.array([[0, 10.0, 8.0]]), name="detections", size=6)
    model.add_tracks(
        np.array([[1, t, 10.0 + t, 8.0] for t in range(3)], dtype=float),
        name="trajectories",
    )
    names = [layer.name for layer in model.layers]
    if len(names) != 4:
        raise RuntimeError(f"expected 4 layers, built {names}")

    # Plugin discovery is the part a frozen build loses silently.
    try:
        from npe2 import PluginManager

        manager = PluginManager.instance()
        manager.discover()
        plugins = sorted(manager._manifests)
        discovery = f", plugins: {len(plugins)}"
    except Exception as exc:  # noqa: BLE001
        discovery = f", plugin discovery unavailable ({type(exc).__name__})"

    return f"napari {napari.__version__}, layers build: {', '.join(names)}{discovery}"


def _check_gpu() -> str:
    from .core.segmentation import gpu_available

    return "GPU available" if gpu_available() else "no GPU, will use the CPU"


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("scientific stack", _check_imports),
    ("interface toolkit", _check_qt),
    ("validated model", _check_model),
    ("segmentation", _check_segmentation),
    ("tracking, speed, MSD", _check_tracking),
    ("output files", _check_outputs),
    ("project store", _check_store),
    ("processing device", _check_gpu),
    ("napari (optional)", _check_napari),
]

#: A failure here does not make the application unusable.
OPTIONAL = {"napari (optional)"}


def run(verbose: bool = True, checks: list[tuple[str, Callable[[], str]]] | None = None) -> int:
    report = Report()
    if verbose:
        print(f"{app_meta.APP_NAME} {app_meta.APP_VERSION} self-test")
        print(f"  python {sys.version.split()[0]} on {platform.platform()}")
        print(f"  frozen: {resources.is_frozen()}")
        print()

    for name, fn in checks if checks is not None else CHECKS:
        ok = report.add(name, fn)
        if verbose:
            check = report.checks[-1]
            mark = "ok  " if ok else ("warn" if name in OPTIONAL else "FAIL")
            print(f"  [{mark}] {name:22s} {check.detail or check.error}")

    hard_failures = [c for c in report.failures if c.name not in OPTIONAL]
    if verbose:
        print()
        if hard_failures:
            print(f"{len(hard_failures)} check(s) failed.")
        else:
            soft = [c for c in report.failures if c.name in OPTIONAL]
            print(
                "All required checks passed."
                + (f" {len(soft)} optional feature unavailable." if soft else "")
            )
    return 1 if hard_failures else 0

