"""Optional Napari inspection.

Napari is a supplementary viewer here, not the product's interface.  Two rules
follow from the defects in the original notebook:

*   Results are already on disk before anything opens.  ``napari.run()`` blocks
    until the window closes; in the notebook the final CSV was written *after*
    that call, so closing the viewer was required for the results to exist.
*   The viewer is returned rather than left in a module-level global.  The
    notebook's helper created ``viewer`` inside a function, and a later cell
    that referenced ``viewer`` failed with a NameError.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.imaging import load_stack, read_metadata


class NapariUnavailable(ImportError):
    """Raised when Napari is not installed in this environment."""


def _require_napari():
    try:
        import napari  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise NapariUnavailable(
            "Napari is not installed. Corridor's own viewer shows the same layers."
        ) from exc
    return napari


def build_viewer(
    stack: np.ndarray,
    masks: np.ndarray | None,
    track_rows: list[dict[str, Any]],
    detection_rows: list[dict[str, Any]],
    *,
    title: str = "Corridor",
    axis: tuple[float, float] | None = None,
):
    """Create and return a Napari viewer with every analysis layer.

    The viewer is returned, never stored in a global, so a caller always has a
    handle to add to it.
    """
    napari = _require_napari()
    viewer = napari.Viewer(title=title)
    viewer.add_image(stack, name="microscopy", colormap="gray", interpolation2d="nearest")

    if masks is not None and masks.size:
        viewer.add_labels(masks.astype(np.int32), name="cell masks", opacity=0.35)

    if detection_rows:
        points = np.array(
            [[r["frame"], r["y"], r["x"]] for r in detection_rows if r.get("x") is not None],
            dtype=float,
        )
        if points.size:
            viewer.add_points(
                points, name="detections", size=6, opacity=0.85,
                face_color="white", border_color="black",
            )

    usable = [
        r for r in track_rows
        if r.get("track_id") is not None and r.get("x_px") is not None
    ]
    if usable:
        # Napari wants [track_id, t, y, x], sorted by track then time.
        usable.sort(key=lambda r: (int(r["track_id"]), int(r["frame"])))
        data = np.array(
            [[int(r["track_id"]), int(r["frame"]), float(r["y_px"]), float(r["x_px"])]
             for r in usable],
            dtype=float,
        )
        speeds = [
            float(r["speed_um_per_min"]) if r.get("speed_um_per_min") is not None else 0.0
            for r in usable
        ]
        viewer.add_tracks(
            data,
            name="trajectories",
            properties={"speed": np.array(speeds, dtype=float)},
            tail_length=len(stack),
        )

    if axis is not None:
        height, width = stack.shape[-2], stack.shape[-1]
        ux, uy = axis
        length = min(height, width) * 0.35
        cx, cy = width / 2.0, height / 2.0
        line = np.array(
            [[cy - uy * length, cx - ux * length], [cy + uy * length, cx + ux * length]]
        )
        viewer.add_shapes(
            [line], shape_type="line", name="migration axis",
            edge_color="#22B8CF", edge_width=2, opacity=0.9,
        )

    viewer.dims.set_point(0, 0)
    return viewer


def open_saved_in_napari(analysis, block: bool = False):
    """Open a completed analysis. Everything shown is already saved."""
    napari = _require_napari()
    source = analysis.source_path
    if source is None or not Path(source).exists():
        raise FileNotFoundError(
            f"The original image is no longer at {source}; Napari needs it to show the overlays."
        )
    metadata = read_metadata(source)
    stack = load_stack(source, metadata)
    confinement = analysis.manifest.get("confinement", {})
    viewer = build_viewer(
        stack,
        analysis.masks,
        analysis.tracks,
        analysis.detections,
        title=f"Corridor — {Path(source).name}",
        axis=(float(confinement.get("ux", 0.0)), float(confinement.get("uy", 1.0))),
    )
    if block:
        napari.run()
    return viewer


def open_in_napari(result, block: bool = True):
    """Open a freshly computed result. Called only after it has been saved."""
    from ..core.detections import detections_to_rows

    confinement = result.axis
    viewer = build_viewer(
        load_stack(result.metadata.path, result.metadata),
        result.segmentation.masks,
        result.rows,
        detections_to_rows(result.segmentation.detections),
        title=f"Corridor — {result.metadata.path.name}",
        axis=(confinement.ux, confinement.uy),
    )
    if block:
        _require_napari().run()
    return viewer
