"""Reading a saved analysis for display, across both output schemas.

The results screen shows v2 quantities (per-hour speeds, Len, D2S, MSD) for
every analysis, including ones written by 1.x. The store (work package C)
upgrades a v1 run in memory, so normally the v2 columns are simply there.
These helpers read them first and fall back, explicitly and in the same
units, only where a value is genuinely absent -- so a screen never shows a
number in a unit the file did not support.

Two conventions are kept everywhere:

*   **Per-hour speeds derive from the canonical µm/min value** (contract §6):
    ``x 60``, never recomputed from geometry.
*   **Absent is None**, never 0. A track that did not move has Len 0.0; a
    track whose length is unknown has Len None and shows an em dash.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

#: The sidecar holding the reference point the user set on the results
#: screen. It lives beside run.json rather than inside it: run.json records
#: what the run did, and a reference point picked afterwards is not part of
#: that record.
REFERENCE_FILE = "reference_point.json"

#: The marker every formatted value uses for "not known".
DASH = "—"


def first_present(mapping: dict[str, Any] | None, *keys: str) -> Any:
    """The first value among ``keys`` that is present and not None."""
    if not mapping:
        return None
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def speed_um_per_hr(row: dict[str, Any] | None, base: str = "speed") -> float | None:
    """``<base>_um_per_hr``, or the µm/min value x 60 when only that exists."""
    if not row:
        return None
    per_hr = finite(row.get(f"{base}_um_per_hr"))
    if per_hr is not None:
        return per_hr
    per_min = finite(row.get(f"{base}_um_per_min"))
    return per_min * 60.0 if per_min is not None else None


def row_z(row: dict[str, Any]) -> float | None:
    """The Z position of a track row in slices, if the run was 3-D."""
    return finite(first_present(row, "z_px", "z", "z_slice"))


def row_xy(row: dict[str, Any]) -> tuple[float, float] | None:
    x, y = finite(row.get("x_px")), finite(row.get("y_px"))
    if x is None or y is None:
        return None
    return (x, y)


# --------------------------------------------------------------------------
# Per-track path metrics (MTrackJ Len and D2S)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PathMetrics:
    """Total path length (Len) and net distance from the start (D2S)."""

    length: float | None
    from_start: float | None
    #: "µm" when calibrated, "px" when the run had no pixel size.
    unit: str


def path_metrics(
    rows: Sequence[dict[str, Any]],
    summary: dict[str, Any] | None,
    pixel_size_um: float | None,
    z_step_um: float | None = None,
) -> PathMetrics:
    """Len and D2S at the track's last observation.

    Preference order: the v2 per-observation columns at the last row
    (``cumulative_path_um``, ``distance_from_start_um``), then the summary's
    ``path_length_um`` / ``net_displacement_um``, then a sum over the rows'
    positions (see :func:`_computed`). The last is in µm only if the run was
    calibrated, and in px otherwise -- a px distance is never labelled µm.
    """
    ordered = sorted(rows, key=lambda r: (finite(r.get("frame")) or 0.0))
    last = ordered[-1] if ordered else {}
    length = finite(last.get("cumulative_path_um"))
    start = finite(last.get("distance_from_start_um"))
    if length is None:
        length = finite((summary or {}).get("path_length_um"))
    if start is None:
        start = finite((summary or {}).get("net_displacement_um"))
    if length is not None or start is not None:
        if length is None or start is None:
            computed = _computed(ordered, pixel_size_um, z_step_um)
            if computed.unit == "µm":
                length = length if length is not None else computed.length
                start = start if start is not None else computed.from_start
        return PathMetrics(length, start, "µm")
    return _computed(ordered, pixel_size_um, z_step_um)


def _computed(
    rows: Sequence[dict[str, Any]],
    pixel_size_um: float | None,
    z_step_um: float | None = None,
) -> PathMetrics:
    """Len and D2S summed from the rows' own positions, when that is honest.

    A 3-D track moves in Z too. Summing only x and y would report a cell
    that moved purely between slices as Len 0.0 µm, and the µm label would
    be a lie whenever the Z step is unknown -- which is exactly when the
    store leaves the µm path columns empty, because Z spacing is never
    assumed (contract §4). So a 3-D track is measured in µm with the real Z
    step, or not at all (None, shown as an em dash). Without an XY pixel size
    there is no common unit for slices and pixels either, so an uncalibrated
    3-D track is not measured in px.
    """
    calibrated = finite(pixel_size_um) is not None and float(pixel_size_um) > 0
    unit = "µm" if calibrated else "px"
    three_d = any(row_z(r) is not None for r in rows)
    if three_d:
        z_step = finite(z_step_um)
        if not calibrated or z_step is None or z_step <= 0:
            return PathMetrics(None, None, unit)
        pixel = float(pixel_size_um)
        points3: list[tuple[float, float, float]] = []
        for r in rows:
            xy, z = row_xy(r), row_z(r)
            if xy is None:
                continue
            if z is None:
                # A 3-D track with a row that has no Z cannot be summed in 3-D.
                return PathMetrics(None, None, unit)
            points3.append((xy[0] * pixel, xy[1] * pixel, z * z_step))
        if not points3:
            return PathMetrics(None, None, unit)
        length = sum(math.dist(a, b) for a, b in zip(points3, points3[1:]))
        return PathMetrics(length, math.dist(points3[0], points3[-1]), unit)

    points = [p for p in (row_xy(r) for r in rows) if p is not None]
    if not points:
        return PathMetrics(None, None, unit)
    factor = float(pixel_size_um) if calibrated else 1.0
    length = sum(
        math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(points, points[1:])
    )
    start = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
    return PathMetrics(length * factor, start * factor, unit)


def reference_distance_um(
    row: dict[str, Any],
    point: Sequence[float],
    pixel_size_um: float | None,
    z_step_um: float | None = None,
) -> float | None:
    """D2R for one track row: its distance to ``point``, in µm, or None.

    ``point`` is (x, y[, z]) in the frame of ``x_px``/``y_px`` and slices.
    The column is ``_um``, so without a pixel size there is no value; a 3-D
    row needs the point's Z and the real Z step for the same reason Len does
    (:func:`_computed`).
    """
    pixel = finite(pixel_size_um)
    xy = row_xy(row)
    if pixel is None or pixel <= 0 or xy is None or len(point) < 2:
        return None
    dx, dy = (xy[0] - float(point[0])) * pixel, (xy[1] - float(point[1])) * pixel
    z = row_z(row)
    if z is None:
        return math.hypot(dx, dy)
    z_step = finite(z_step_um)
    if len(point) < 3 or z_step is None or z_step <= 0:
        return None
    return math.hypot(dx, dy, (z - float(point[2])) * z_step)


def z_step_um_of(analysis: Any) -> float | None:
    """The run's Z step in µm, or None when it was 2-D or never established."""
    value = finite(getattr(analysis, "z_step_um", None))
    if value is None:
        calibration = (getattr(analysis, "manifest", None) or {}).get("calibration") or {}
        value = finite(calibration.get("z_step_um"))
    return value if value is not None and value > 0 else None


# --------------------------------------------------------------------------
# MSD
# --------------------------------------------------------------------------


def has_msd(analysis: Any) -> bool:
    """Whether this analysis carries MSD curves at all (schema 2, or upgraded)."""
    rows = getattr(analysis, "msd", None)
    return bool(rows)


def msd_rows_for(analysis: Any, track_id: int) -> list[dict[str, Any]]:
    """The MSD rows of one track, through the store's accessor when it has one."""
    accessor = getattr(analysis, "msd_for_track", None)
    if callable(accessor):
        try:
            return list(accessor(track_id) or [])
        except Exception:  # noqa: BLE001 - a display helper must not break the screen
            return []
    rows = getattr(analysis, "msd", None) or []
    return [r for r in rows if _same_id(r.get("track_id"), track_id)]


def _same_id(value: Any, track_id: int) -> bool:
    number = finite(value)
    return number is not None and int(number) == int(track_id)


@dataclass(frozen=True)
class MsdPoint:
    lag: float
    msd: float
    n_pairs: int


@dataclass(frozen=True)
class MsdSeries:
    points: tuple[MsdPoint, ...]
    lag_unit: str  # "h" or "frames"
    msd_unit: str  # "µm²" or "px²"


def msd_series(rows: Iterable[dict[str, Any]], frame_interval_min: float | None = None) -> MsdSeries:
    """Plottable MSD points: lag in hours, MSD in µm², when the run allows it.

    Falls back to frames and px² as a pair-wise decision over the whole
    curve, so one curve never mixes units. Non-positive values are dropped:
    a log-log plot cannot place them, and an MSD of exactly zero at a non-zero
    lag means the positions did not change, not a point on a power law.
    """
    rows = list(rows)
    interval = finite(frame_interval_min)
    in_hours = bool(rows) and all(
        finite(r.get("lag_time_hr")) is not None
        or finite(r.get("lag_time_min")) is not None
        or (interval is not None and finite(r.get("lag_frames")) is not None)
        for r in rows
    )
    in_um = bool(rows) and all(finite(r.get("msd_um2")) is not None for r in rows)
    points: list[MsdPoint] = []
    for r in rows:
        if in_hours:
            lag = finite(r.get("lag_time_hr"))
            if lag is None and finite(r.get("lag_time_min")) is not None:
                lag = float(r["lag_time_min"]) / 60.0
            if lag is None and interval is not None and finite(r.get("lag_frames")) is not None:
                lag = float(r["lag_frames"]) * interval / 60.0
        else:
            lag = finite(r.get("lag_frames"))
        msd = finite(r.get("msd_um2")) if in_um else finite(r.get("msd_px2"))
        n_pairs = finite(r.get("n_pairs"))
        if lag is None or msd is None or lag <= 0 or msd <= 0:
            continue
        points.append(MsdPoint(lag, msd, int(n_pairs) if n_pairs is not None else 1))
    points.sort(key=lambda p: p.lag)
    return MsdSeries(tuple(points), "h" if in_hours else "frames", "µm²" if in_um else "px²")


# --------------------------------------------------------------------------
# Whole-analysis facts
# --------------------------------------------------------------------------


def schema_version_of(analysis: Any) -> int | None:
    """2 for a v2 run, 1 for a 1.x run, None when nothing was loaded."""
    version = getattr(analysis, "schema_version", None)
    if version is None:
        manifest = getattr(analysis, "manifest", None) or {}
        version = manifest.get("schema_version")
        if version is None and manifest:
            version = 1  # 1.x wrote no schema_version at all
    try:
        return int(version) if version is not None else None
    except (TypeError, ValueError):
        return None


def dimensionality_of(analysis: Any, stack: Any = None) -> str:
    """'3D' or '2D'. Reads the analysis first; the stack only breaks a tie."""
    value = getattr(analysis, "dimensionality", None)
    if not value:
        manifest = getattr(analysis, "manifest", None) or {}
        value = manifest.get("dimensionality")
        if not value:
            axes = str((manifest.get("input") or {}).get("axes") or "")
            if "Z" in axes.upper():
                value = "3D"
    if not value and stack is not None and getattr(stack, "ndim", 0) == 4:
        value = "3D"
    return "3D" if str(value or "").upper() == "3D" else "2D"


def load_reference_point(directory: Path | str | None) -> tuple[float, ...] | None:
    """The reference point saved for one analysis, in px (x, y[, z])."""
    if directory is None:
        return None
    path = Path(directory) / REFERENCE_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    point = data.get("reference_point_px") if isinstance(data, dict) else None
    if not isinstance(point, (list, tuple)) or len(point) not in (2, 3):
        return None
    values = tuple(finite(v) for v in point)
    if any(v is None for v in values):
        return None
    return values  # type: ignore[return-value]


def save_reference_point(directory: Path | str, point: Sequence[float] | None) -> Path:
    """Persist (or, with None, clear) the reference point of one analysis."""
    path = Path(directory) / REFERENCE_FILE
    if point is None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return path
    payload = {
        "reference_point_px": [float(v) for v in point],
        "units": "pixels (x, y) and slices (z), the frame of tracks.csv x_px/y_px",
        "used_for": "distance_from_reference_um (D2R) in exports",
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def fmt_number(value: Any, digits: int = 2, suffix: str = "") -> str:
    """A number with a unit, or the em dash. Never the text 'None'."""
    number = finite(value)
    if number is None:
        return DASH
    return f"{number:.{digits}f}{suffix}"
