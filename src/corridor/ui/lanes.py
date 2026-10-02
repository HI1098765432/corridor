"""Channel lanes as drawable overlays, from either schema.

2.0 has no migration axis (contract §5): the device walls are described as
*lanes*, each with its own centre line and half-width, in
``run.json["channel_geometry"]``. A 1.x run recorded a single shared axis and
one origin per channel in ``run.json["confinement"]``; those channel lines are
still worth drawing on an old result, so they are converted into lanes here
-- and only here -- rather than keeping an axis anywhere in the viewer.

Rules this module keeps:

*   **Nothing is fabricated.** A missing ``confinement`` block used to default
    to a vertical axis ``(0, 1)``; here a missing or unreadable block yields
    no lanes at all, and the overlay simply has nothing to draw.
*   **Reads what the run wrote.** The geometry module owns the v2 format:
    ``ChannelGeometry.to_dict`` writes each lane as ``origin_x``/``origin_y``
    plus a unit ``direction_x``/``direction_y``. Lanes are read from the
    saved dict itself, not through ``ChannelGeometry.from_dict``, whose
    defaults (origin (0, 0), direction (0, 1)) would turn a lane it cannot
    read into a confident line down the left edge. The other plausible
    spellings (a polyline, two end points, ``origin``/``direction`` pairs, a
    ``ChannelGeometry`` object) are still accepted. A lane that cannot be read
    is skipped, never guessed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

Point = tuple[float, float]

SOURCE_V2 = "channel_geometry"
SOURCE_LEGACY = "legacy_confinement"


@dataclass(frozen=True)
class LaneOverlay:
    """One lane in image pixel coordinates (x right, y down)."""

    index: int
    centre: tuple[Point, ...]
    half_width_px: float | None
    source: str


def _get(obj: Any, *names: str) -> Any:
    """The first present, non-None attribute or key among ``names``."""
    for name in names:
        if isinstance(obj, dict):
            value = obj.get(name)
        else:
            value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _as_point(value: Any) -> Point | None:
    if value is None:
        return None
    if isinstance(value, dict):
        x, y = value.get("x"), value.get("y")
    else:
        try:
            x, y = value[0], value[1]
        except (TypeError, IndexError, KeyError):
            return None
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (x, y)


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clip_line(origin: Point, direction: Point, width: float, height: float) -> tuple[Point, Point] | None:
    """The segment of the infinite line through ``origin`` inside the image.

    Liang-Barsky against ``[0, width] x [0, height]``. None when the line
    misses the image or the direction is degenerate.
    """
    ox, oy = origin
    dx, dy = direction
    if math.hypot(dx, dy) < 1e-12 or width <= 0 or height <= 0:
        return None
    t0, t1 = -math.inf, math.inf
    for p, q in ((-dx, ox), (dx, width - ox), (-dy, oy), (dy, height - oy)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
    if t0 > t1 or not math.isfinite(t0) or not math.isfinite(t1):
        return None
    return ((ox + t0 * dx, oy + t0 * dy), (ox + t1 * dx, oy + t1 * dy))


def lane_from_object(lane: Any, index: int, size: tuple[float, float], source: str) -> LaneOverlay | None:
    """Read one lane (dict or object) into an overlay, or None if unreadable."""
    width, height = size
    half = _as_float(_get(lane, "half_width_px", "half_width", "halfwidth_px"))
    if half is None:
        full = _as_float(_get(lane, "width_px"))
        half = full / 2.0 if full is not None else None
    lane_index = _get(lane, "index", "lane", "id")
    try:
        lane_index = int(lane_index) if lane_index is not None else index
    except (TypeError, ValueError):
        lane_index = index

    polyline = _get(lane, "centre_line", "center_line", "centreline", "centerline", "points", "polyline")
    if polyline is not None and not isinstance(polyline, (str, bytes)):
        try:
            points = tuple(p for p in (_as_point(v) for v in polyline) if p is not None)
        except TypeError:
            points = ()
        if len(points) >= 2:
            return LaneOverlay(lane_index, points, half, source)

    start = _as_point(_get(lane, "start", "p0"))
    end = _as_point(_get(lane, "end", "p1"))
    if start is None:
        x0, y0 = _as_float(_get(lane, "x0")), _as_float(_get(lane, "y0"))
        x1, y1 = _as_float(_get(lane, "x1")), _as_float(_get(lane, "y1"))
        if None not in (x0, y0, x1, y1):
            start, end = (x0, y0), (x1, y1)
    if start is not None and end is not None:
        return LaneOverlay(lane_index, (start, end), half, source)

    origin = _as_point(_get(lane, "origin"))
    if origin is None:
        ox, oy = _as_float(_get(lane, "origin_x")), _as_float(_get(lane, "origin_y"))
        origin = (ox, oy) if ox is not None and oy is not None else None
    direction = _as_point(_get(lane, "direction", "unit", "tangent"))
    if direction is None:
        # direction_x/_y is what ChannelGeometry.to_dict writes; ux/uy is v1's.
        ux = _as_float(_get(lane, "direction_x", "ux"))
        uy = _as_float(_get(lane, "direction_y", "uy"))
        direction = (ux, uy) if ux is not None and uy is not None else None
    if origin is not None and direction is not None:
        segment = clip_line(origin, direction, width, height)
        if segment is not None:
            return LaneOverlay(lane_index, segment, half, source)
    return None


def lanes_from_geometry(geometry: Any, size: tuple[float, float]) -> list[LaneOverlay]:
    """Overlays from a v2 ``channel_geometry`` dict (or a ChannelGeometry).

    A dict is read as saved (``ChannelGeometry.to_dict``'s keys, or another
    spelling), never through ``from_dict``: its defaults would invent a lane.
    """
    if geometry is None:
        return []
    return _read_lanes(_get(geometry, "lanes"), size, SOURCE_V2)


def lanes_from_legacy(confinement: Any, size: tuple[float, float]) -> list[LaneOverlay]:
    """Overlays from a v1 ``confinement`` block, drawn along its recorded axis.

    The axis is used only to draw the channel lines that run *recorded*; no
    axis is created when the block does not state one.
    """
    if not isinstance(confinement, dict):
        return []
    ux, uy = _as_float(confinement.get("ux")), _as_float(confinement.get("uy"))
    if ux is None or uy is None or math.hypot(ux, uy) < 1e-12:
        return []
    out: list[LaneOverlay] = []
    for i, channel in enumerate(confinement.get("channels") or []):
        if not isinstance(channel, dict):
            continue
        lane = dict(channel)
        lane.setdefault("ux", ux)
        lane.setdefault("uy", uy)
        overlay = lane_from_object(lane, i, size, SOURCE_LEGACY)
        if overlay is not None:
            out.append(overlay)
    return out


def _read_lanes(lanes: Any, size: tuple[float, float], source: str) -> list[LaneOverlay]:
    if lanes is None or isinstance(lanes, (str, bytes)):
        return []
    out: list[LaneOverlay] = []
    try:
        iterator: Iterable[Any] = iter(lanes)
    except TypeError:
        return []
    for i, lane in enumerate(iterator):
        overlay = lane_from_object(lane, i, size, source)
        if overlay is not None:
            out.append(overlay)
    return out


def lanes_for_manifest(manifest: dict[str, Any], size: tuple[float, float]) -> list[LaneOverlay]:
    """The lanes to draw for a saved run: v2 geometry first, else v1 lines."""
    manifest = manifest or {}
    geometry = manifest.get("channel_geometry")
    if geometry:
        return lanes_from_geometry(geometry, size)
    return lanes_from_legacy(manifest.get("confinement"), size)


def offset_polyline(points: tuple[Point, ...], distance: float) -> list[Point]:
    """``points`` shifted sideways by ``distance`` along each segment normal.

    Used to draw a lane's walls at +/- half-width. Joints average the normals
    of the two segments they join, which is exact for a straight lane and
    close enough to show a gently curved one.
    """
    n = len(points)
    if n < 2:
        return list(points)
    normals: list[Point] = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy) or 1.0
        normals.append((-dy / length, dx / length))
    out: list[Point] = []
    for i, (x, y) in enumerate(points):
        if i == 0:
            nx, ny = normals[0]
        elif i == n - 1:
            nx, ny = normals[-1]
        else:
            nx = normals[i - 1][0] + normals[i][0]
            ny = normals[i - 1][1] + normals[i][1]
            length = math.hypot(nx, ny) or 1.0
            nx, ny = nx / length, ny / length
        out.append((x + nx * distance, y + ny * distance))
    return out
