"""Check that an installed copy is complete and working.

A packaged application can be missing a module that only a particular code
path touches, and the usual way that surfaces is a researcher pressing Analyse
and getting an error. This walks the paths instead: it imports every subsystem,
verifies the bundled model's checksum, runs the full torch/Cellpose path on a
synthetic frame, tracks a known trajectory and checks the velocity arithmetic,
round-trips every output format, and exercises Napari's layer construction and
plugin discovery.

Run it with ``corridor --self-test``.
"""

from __future__ import annotations

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


def _check_model() -> str:
    path = resources.bundled_model_path()
    if path is None:
        raise RuntimeError("No Cellpose model is bundled or configured.")
    from .core.segmentation import file_sha256

    digest = file_sha256(path)
    expected = resources.BUNDLED_MODEL_SHA256
    if digest != expected:
        raise RuntimeError(
            f"The bundled model has checksum {digest[:16]}..., expected {expected[:16]}..."
        )
    return f"{path.name} ({digest[:12]}...)"


def _check_segmentation() -> str:
    """Load the model and run it. This is the path torch pruning could break."""
    import numpy as np

    from .core.config import SegmentationConfig
    from .core.segmentation import SegmentationService

    model = resources.bundled_model_path()
    service = SegmentationService(
        SegmentationConfig(model_path=str(model) if model else None,
                           use_custom_model=bool(model))
    )
    rng = np.random.default_rng(0)
    frame = rng.normal(1000, 30, size=(160, 96)).astype(np.uint16)
    frame[40:120, 44:56] += 2500  # something cell-shaped to find
    mask, _ = service.segment_frame(frame)
    if mask.shape != frame.shape:
        raise RuntimeError(f"mask shape {mask.shape} does not match input {frame.shape}")
    # What matters here is that the whole torch/cellpose path executed and
    # returned a valid label image. A synthetic frame is not real microscopy,
    # so finding nothing in it is not a failure.
    return (
        f"the full torch/Cellpose path ran and returned a {mask.shape} label image "
        f"({int(mask.max())} instance(s) in synthetic noise)"
    )


def _check_tracking() -> str:
    import numpy as np

    from .core.confinement import Channel, ConfinementAxis
    from .core.config import Scale, TrackingConfig
    from .core.detections import Detection
    from .core.tracking import track_detections

    axis = ConfinementAxis(
        ux=0.0, uy=1.0, source="configured", confidence=1.0,
        channels=[Channel(0, (45.0, 0.0), 45.0)],
    )
    scale = Scale.from_values(0.467060342995564, 20.006894938151042)
    detections = [
        Detection(
            frame=t, label=1, x=45.0, y=20.0 + 20.0 * t, area_px=800.0,
            bbox=(0, 0, 90, 11), extent_px=90, eccentricity=0.99,
            orientation_rad=0.0, major_axis_px=90.0, minor_axis_px=11.0,
            solidity=0.95, touches_border=False, channel=0,
        )
        for t in range(5)
    ]
    tracks, _ = track_detections(detections, 5, axis, scale, TrackingConfig())
    if len(tracks) != 1 or tracks[0].n_obs != 5:
        raise RuntimeError(f"expected one 5-point track, got {len(tracks)}")

    from .core.measurements import frame_rows

    rows = frame_rows(tracks, axis, scale)
    speed = rows[1]["speed_um_per_min"]
    expected = 20.0 * 0.467060342995564 / 20.006894938151042
    if abs(speed - expected) > 1e-9:
        raise RuntimeError(f"velocity is wrong: {speed} vs {expected}")
    return f"one track, speed {speed:.4f} um/min as expected"


def _check_outputs() -> str:
    import numpy as np

    from .core import export

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        export.write_csv(directory / "t.csv", export.TRACK_COLUMNS, [])
        export.write_json(directory / "m.json", {"x": float("nan"), "y": 1})
        export.save_masks(directory / "k.npz", np.zeros((2, 4, 4), np.int32))
        loaded = export.load_masks(directory / "k.npz")
        if loaded.shape != (2, 4, 4):
            raise RuntimeError("mask round-trip failed")
        import json

        data = json.loads((directory / "m.json").read_text(encoding="utf-8"))
        if data["x"] is not None:
            raise RuntimeError("NaN was not sanitised out of the manifest")
    return "CSV, JSON and mask round-trips all clean"


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
    ("bundled model", _check_model),
    ("segmentation", _check_segmentation),
    ("tracking and velocity", _check_tracking),
    ("output files", _check_outputs),
    ("project store", _check_store),
    ("processing device", _check_gpu),
    ("napari (optional)", _check_napari),
]

#: A failure here does not make the application unusable.
OPTIONAL = {"napari (optional)"}


def run(verbose: bool = True) -> int:
    report = Report()
    if verbose:
        print(f"{app_meta.APP_NAME} {app_meta.APP_VERSION} self-test")
        print(f"  python {sys.version.split()[0]} on {platform.platform()}")
        print(f"  frozen: {resources.is_frozen()}")
        print()

    for name, fn in CHECKS:
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
