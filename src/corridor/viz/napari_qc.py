"""Optional Napari inspection.

Napari is a supplementary viewer here, not the product's interface.  Two rules
follow from the defects in the original notebook:

*   Results are already on disk before anything opens.  ``napari.run()`` blocks
    until the window closes; in the notebook the final CSV was written *after*
    that call, so closing the viewer was required for the results to exist.
*   The viewer is returned rather than left in a module-level global.  The
    notebook's helper created ``viewer`` inside a function, and a later cell
    that referenced ``viewer`` failed with a NameError.

2.0 adds two more:

*   **No migration-axis layer.** There is no axis (contract §5); 1.x drew one
    even when the run had recorded none, defaulting to vertical.
*   **3-D is native.** A ``TZYX`` stack is shown with its labels, its points
    as ``[t, z, y, x]`` and its tracks as ``[track_id, t, z, y, x]``, with the
    Z axis scaled by the anisotropy when the Z step is known -- and drawn one
    slice per pixel, not guessed, when it is not.

The layer data is built by pure functions (``track_layer_data``,
``point_layer_data``) so it can be tested without Napari installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


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


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _z_of(row: dict[str, Any]) -> float | None:
    for key in ("z_px", "z", "z_slice"):
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def track_layer_data(
    track_rows: Sequence[dict[str, Any]], *, three_d: bool
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """``[track_id, t, (z,) y, x]`` rows sorted by track then time, and properties.

    In 3-D a row without a Z position cannot be placed and is left out rather
    than put on slice 0. The ``speed`` property is µm/h, the unit results are
    reported in; an unknown speed is NaN, never 0 (0 would read as "stopped").
    """
    usable = []
    for r in track_rows:
        if r.get("track_id") is None or r.get("frame") is None:
            continue
        x, y = _number(r.get("x_px")), _number(r.get("y_px"))
        if x is None or y is None:
            continue
        z = _z_of(r) if three_d else None
        if three_d and z is None:
            continue
        usable.append((int(r["track_id"]), int(r["frame"]), z, y, x, r))
    usable.sort(key=lambda item: (item[0], item[1]))
    columns = 5 if three_d else 4
    data = np.array(
        [
            [tid, t, z, y, x] if three_d else [tid, t, y, x]
            for tid, t, z, y, x, _ in usable
        ],
        dtype=float,
    ).reshape(-1, columns)
    speeds = []
    for *_, r in usable:
        per_hr = _number(r.get("speed_um_per_hr"))
        if per_hr is None:
            per_min = _number(r.get("speed_um_per_min"))
            per_hr = per_min * 60.0 if per_min is not None else float("nan")
        speeds.append(per_hr)
    return data, {"speed_um_per_hr": np.array(speeds, dtype=float)}


def point_layer_data(detection_rows: Sequence[dict[str, Any]], *, three_d: bool) -> np.ndarray:
    """``[t, (z,) y, x]`` per detection (``detections.csv`` names them x, y, z)."""
    points = []
    for r in detection_rows:
        t, x, y = _number(r.get("frame")), _number(r.get("x")), _number(r.get("y"))
        if t is None or x is None or y is None:
            continue
        if three_d:
            z = _number(r.get("z"))
            if z is None:
                continue
            points.append([t, z, y, x])
        else:
            points.append([t, y, x])
    return np.array(points, dtype=float).reshape(-1, 4 if three_d else 3)


def layer_scale(stack_ndim: int, anisotropy: float | None) -> tuple[float, ...] | None:
    """Napari ``scale`` for a TZYX stack: Z stretched by the anisotropy."""
    if stack_ndim != 4 or not anisotropy or anisotropy <= 0:
        return None
    return (1.0, float(anisotropy), 1.0, 1.0)


def build_viewer(
    stack: np.ndarray,
    masks: np.ndarray | None,
    track_rows: list[dict[str, Any]],
    detection_rows: list[dict[str, Any]],
    *,
    title: str = "Corridor",
    anisotropy: float | None = None,
):
    """Create and return a Napari viewer with every analysis layer.

    ``stack`` is ``TYX`` or ``TZYX``; ``masks`` the same shape. The viewer is
    returned, never stored in a global, so a caller always has a handle to
    add to it.
    """
    napari = _require_napari()
    three_d = stack.ndim == 4
    scale = layer_scale(stack.ndim, anisotropy)
    extra = {"scale": scale} if scale else {}

    viewer = napari.Viewer(title=title)
    viewer.add_image(
        stack, name="microscopy", colormap="gray", interpolation2d="nearest", **extra
    )

    if masks is not None and masks.size:
        viewer.add_labels(masks.astype(np.int32), name="cell masks", opacity=0.35, **extra)

    points = point_layer_data(detection_rows, three_d=three_d)
    if points.size:
        viewer.add_points(
            points, name="detections", size=6, opacity=0.85,
            face_color="white", border_color="black", **extra,
        )

    data, properties = track_layer_data(track_rows, three_d=three_d)
    if data.size:
        viewer.add_tracks(
            data,
            name="trajectories",
            properties=properties,
            tail_length=len(stack),
            **extra,
        )

    viewer.dims.set_point(0, 0)
    return viewer


def _anisotropy(manifest: dict[str, Any]) -> float | None:
    cal = (manifest or {}).get("calibration") or {}
    z_step, pixel = _number(cal.get("z_step_um")), _number(cal.get("pixel_size_um"))
    if z_step and pixel and z_step > 0 and pixel > 0:
        return z_step / pixel
    return None


def open_saved_in_napari(analysis, block: bool = False, stack: np.ndarray | None = None):
    """Open a completed analysis. Everything shown is already saved.

    Pass ``stack`` when the pixels are already in memory (the window holds
    them); otherwise the source is read again, with the axis order the run
    recorded so an ambiguous file is not asked about twice.
    """
    napari = _require_napari()
    source = analysis.source_path
    if stack is None:
        if source is None or not Path(source).exists():
            raise FileNotFoundError(
                f"The original image is no longer at {source}; Napari needs it to show the overlays."
            )
        from ..core.imaging import load_stack  # noqa: PLC0415
        from ..ui.workers import import_config_for, read_metadata_for  # noqa: PLC0415

        metadata = read_metadata_for(source, import_config_for(analysis.manifest))
        stack = load_stack(source, metadata)
    name = Path(source).name if source else Path(analysis.directory).name
    viewer = build_viewer(
        stack,
        analysis.masks,
        analysis.tracks,
        analysis.detections,
        title=f"Corridor — {name}",
        anisotropy=_anisotropy(analysis.manifest),
    )
    if block:
        napari.run()
    return viewer


def open_in_napari(result, block: bool = True):
    """Open a freshly computed result. Called only after it has been saved."""
    from ..core.detections import detections_to_rows  # noqa: PLC0415
    from ..core.imaging import load_stack  # noqa: PLC0415

    scale = getattr(result, "scale", None)
    anisotropy = getattr(scale, "anisotropy", None) if scale is not None else None
    viewer = build_viewer(
        load_stack(result.metadata.path, result.metadata),
        result.segmentation.masks,
        result.rows,
        detections_to_rows(result.segmentation.detections),
        title=f"Corridor — {result.metadata.path.name}",
        anisotropy=anisotropy,
    )
    if block:
        _require_napari().run()
    return viewer
