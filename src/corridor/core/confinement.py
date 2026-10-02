"""DEPRECATED: the v1 stack-level migration axis, kept for legacy callers only.

Corridor 2.0 tracks without a migration axis (``tracking.py``) and measures
the device as lanes (``geometry.py``).  This module survives one more round
because ``pipeline.py``, ``qc.py``, ``recovery.py`` and ``measurements.py``
still take a ``ConfinementAxis`` until the integration package rewrites them,
and because v1 runs carry a ``confinement`` block that must stay readable
(``geometry.ChannelGeometry.from_legacy_manifest`` draws it as lanes).

**Nothing new may import this module.**  The wall-ridge detector now lives in
``geometry.py``; the names re-exported below are the same objects, so v1
behaviour is unchanged.

What v1 did, for the record: the research notebook used each detection's own
``regionprops.orientation`` as the direction of migration; v1 replaced that
with one axis measured from the device walls (tilted about 1.2-2.2 degrees
from vertical on the supplied data), channel centre lines sharing that one
direction, and an angle uncertainty that widened the tracker's lateral
tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .config import ConfinementConfig
from .detections import Detection

# The ridge detector moved to geometry.py; these are the same functions.
from .geometry import (  # noqa: F401  (re-exported for legacy callers)
    MIN_RIDGE_CONTRAST,
    _coarse_slope,
    _complete_lattice,
    _consensus_slope_filter,
    _group_ridges,
    _measure_ridges,
    _ridge_contrast,
    _RidgeEvidence,
    _shear_profile,
    _typical_cell_width,
    find_channel_centres,
    robust_line_fit,
    static_projection,
    trace_ridge,
)

AXIS_FROM_CONFIG = "configured"
AXIS_FROM_RIDGES = "channel_ridges"
AXIS_FROM_GEOMETRY = "image_aspect"
AXIS_FROM_MOTION = "detection_motion"
AXIS_FALLBACK = "assumed_vertical"

_AXIS_LABELS = {
    AXIS_FROM_CONFIG: "set by you",
    AXIS_FROM_RIDGES: "measured from the channel walls",
    AXIS_FROM_GEOMETRY: "from the shape of the field",
    AXIS_FROM_MOTION: "from how the cells moved",
    AXIS_FALLBACK: "assumed vertical",
}

#: Angular uncertainty (radians) attached to each way of getting the axis.
#: The tracker widens its lateral tolerance in proportion to this, so a poorly
#: known axis cannot manufacture a confinement violation.
_AXIS_SIGMA = {
    AXIS_FROM_CONFIG: math.radians(0.5),
    AXIS_FROM_RIDGES: math.radians(0.5),
    AXIS_FROM_GEOMETRY: math.radians(3.5),
    AXIS_FROM_MOTION: math.radians(5.0),
    AXIS_FALLBACK: math.radians(6.0),
}


@dataclass
class Channel:
    """One confinement channel: a centre line plus a half width.

    ``detected`` distinguishes a channel whose wall was actually measured from
    one inferred from the lattice because its wall was too faint to see. Both
    are used for gating, but a reviewer is entitled to know which is which:
    an inferred boundary is a assumption about the device, not an observation.
    """

    index: int
    origin: tuple[float, float]  # a point on the centre line, (x, y)
    half_width_px: float
    detected: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "origin_x": self.origin[0],
            "origin_y": self.origin[1],
            "half_width_px": self.half_width_px,
            "detected": self.detected,
        }


@dataclass
class ConfinementAxis:
    """Deprecated v1 axis. Read v1 manifests with ``ChannelGeometry.from_legacy_manifest``."""

    ux: float
    uy: float
    source: str
    confidence: float
    angle_sigma_rad: float = math.radians(3.5)
    channels: list[Channel] = field(default_factory=list)
    pitch_px: float | None = None
    notes: list[str] = field(default_factory=list)

    # -- vectors -----------------------------------------------------------
    @property
    def unit(self) -> np.ndarray:
        return np.array([self.ux, self.uy], dtype=float)

    @property
    def normal(self) -> np.ndarray:
        """Unit vector across the channel: the direction cells should not move."""
        return np.array([-self.uy, self.ux], dtype=float)

    @property
    def angle_deg(self) -> float:
        """Angle of the migration axis measured from the +x image axis."""
        return math.degrees(math.atan2(self.uy, self.ux))

    @property
    def tilt_from_vertical_deg(self) -> float:
        """Signed tilt away from straight down the image, for display."""
        return math.degrees(math.atan2(self.ux, self.uy))

    @property
    def is_multichannel(self) -> bool:
        return len(self.channels) > 1

    # -- geometry ----------------------------------------------------------
    def project(self, vector: np.ndarray) -> tuple[float, float]:
        v = np.asarray(vector, dtype=float)
        return float(v @ self.unit), float(v @ self.normal)

    def offset_from(self, channel: Channel, x: float, y: float) -> float:
        d = np.array([x - channel.origin[0], y - channel.origin[1]], dtype=float)
        return float(d @ self.normal)

    def channel_of(self, x: float, y: float) -> int:
        if not self.channels:
            return 0
        return min(
            self.channels, key=lambda c: abs(self.offset_from(c, x, y))
        ).index

    def distance_to_own_channel(self, x: float, y: float) -> float:
        if not self.channels:
            return 0.0
        return min(abs(self.offset_from(c, x, y)) for c in self.channels)

    def describe(self) -> str:
        return _AXIS_LABELS.get(self.source, self.source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ux": self.ux,
            "uy": self.uy,
            "angle_deg": self.angle_deg,
            "tilt_from_vertical_deg": self.tilt_from_vertical_deg,
            "angle_sigma_deg": math.degrees(self.angle_sigma_rad),
            "source": self.source,
            "confidence": self.confidence,
            "n_channels": len(self.channels),
            "pitch_px": self.pitch_px,
            "channels": [c.to_dict() for c in self.channels],
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def resolve_axis(
    stack: np.ndarray,
    config: ConfinementConfig,
    detections: Sequence[Detection] | None = None,
    *,
    pixel_size_um: float | None = None,
) -> ConfinementAxis:
    """Measure one stack-level migration axis and the channel centre lines.

    Deprecated: use ``geometry.detect_channels``, which returns lanes with
    their own directions and no axis.
    """
    notes: list[str] = []
    projection = static_projection(stack)
    height, width = projection.shape

    forced = config.unit_vector()
    typical = _typical_cell_width(detections)
    min_sep = max(8, int(typical * 0.9))

    # Read the image both ways and let the evidence decide. A profile-sharpness
    # heuristic is not enough: one bright horizontal reservoir edge can outvote
    # the channels it feeds.
    # Ridges closer than one plausible channel are the two walls of the same
    # channel; that pitch also sets how wide the ridge-following window may be
    # without wandering onto a neighbouring channel.
    min_pitch = (
        config.min_channel_pitch_um / pixel_size_um
        if pixel_size_um
        else config.min_channel_pitch_px
    )
    window = int(max(10, min(24, min_pitch / 3.0)))

    evidence: dict[bool, _RidgeEvidence] = {True: _RidgeEvidence(), False: _RidgeEvidence()}
    if config.detect_walls:
        evidence[True] = _measure_ridges(projection, min_sep, window)
        evidence[False] = _measure_ridges(
            np.ascontiguousarray(projection.T), min_sep, window
        )

    if forced is not None:
        vertical = abs(forced[1]) >= abs(forced[0])
    else:
        vertical = evidence[True].support >= evidence[False].support

    best = evidence[vertical]

    # A channel is bounded by two bright walls, and both show up as ridges.
    # Group them and combine the fits that were already measured on each wall.
    # Re-tracing from the midpoint of a group would start the search in the
    # dark lumen between two walls, where there is no ridge to follow.
    groups = _group_ridges(best.ridges, min_pitch)
    if len(groups) < len(best.ridges):
        notes.append(
            f"{len(best.ridges)} bright lines were grouped into {len(groups)} "
            f"channel(s); lines closer than {min_pitch:.0f} px are the two walls of "
            "one channel."
        )
    groups = _complete_lattice(groups)
    centres = [g[1] for g in groups]
    per_channel_slopes = [g[0] for g in groups]
    inlier_fraction = best.inlier_fraction
    slope = float(np.median(per_channel_slopes)) if per_channel_slopes else 0.0
    ambiguous = (
        min(evidence[True].support, evidence[False].support)
        > 0.8 * max(evidence[True].support, evidence[False].support, 1.0)
    )

    # -- decide the axis ----------------------------------------------------
    if forced is not None:
        ux, uy = forced
        source = AXIS_FROM_CONFIG
        confidence = 1.0
        if centres and abs(math.degrees(math.atan(slope))) > 1.0:
            notes.append(
                f"The channels in this image are tilted about "
                f"{abs(math.degrees(math.atan(slope))):.1f} degrees, but the axis was set manually."
            )
    elif centres:
        direction = np.array([slope, 1.0], dtype=float)
        direction /= np.linalg.norm(direction)
        ux, uy = (float(direction[0]), float(direction[1])) if vertical else (
            float(direction[1]), float(direction[0])
        )
        source = AXIS_FROM_RIDGES
        confidence = float(min(1.0, 0.55 + 0.45 * inlier_fraction))
    elif max(height, width) >= min(height, width) * config.multichannel_warn_ratio:
        ux, uy = (0.0, 1.0) if height >= width else (1.0, 0.0)
        source = AXIS_FROM_GEOMETRY
        confidence = 0.5
    else:
        motion = _axis_from_motion(detections or [])
        if motion is not None:
            ux, uy = motion
            source, confidence = AXIS_FROM_MOTION, 0.4
        else:
            ux, uy = 0.0, 1.0
            source, confidence = AXIS_FALLBACK, 0.2
            notes.append(
                "No channel walls, field shape or cell motion gave a reliable migration "
                "axis, so vertical was assumed. Set it explicitly if that is wrong."
            )

    angle_sigma = _AXIS_SIGMA.get(source, math.radians(3.5))
    if source is AXIS_FROM_RIDGES and len(per_channel_slopes) > 2:
        spread = float(np.std([math.atan(s) for s in per_channel_slopes]))
        angle_sigma = max(angle_sigma, spread)

    axis = ConfinementAxis(
        ux=float(ux), uy=float(uy), source=source, confidence=confidence,
        angle_sigma_rad=angle_sigma, notes=notes,
    )

    # -- channel centre lines ----------------------------------------------
    pitch = None
    if len(centres) > 1:
        gaps = np.diff(sorted(centres))
        gaps = gaps[gaps > min_sep * 0.5]
        if gaps.size:
            pitch = float(np.median(gaps))
    half_width = (pitch / 2.0) if pitch else float(max(min_sep, 0.5 * (width if vertical else height)))

    inferred = {round(g[1], 3) for g in groups if g[2] == 0}
    channels: list[Channel] = []
    for i, c in enumerate(sorted(centres)):
        origin = (float(c), 0.0) if vertical else (0.0, float(c))
        channels.append(
            Channel(
                index=i, origin=origin, half_width_px=half_width,
                detected=round(float(c), 3) not in inferred,
            )
        )
    if not channels:
        origin = (width / 2.0, 0.0) if vertical else (0.0, height / 2.0)
        channels = [
            Channel(index=0, origin=origin, half_width_px=float(max(width, height)))
        ]
    axis.channels = channels
    axis.pitch_px = pitch

    n_inferred = sum(1 for c in axis.channels if not c.detected)
    if n_inferred:
        axis.notes.append(
            f"{n_inferred} channel boundary/boundaries could not be seen directly and "
            "were placed from the spacing of the others. Check them if a cell appears "
            "to change channel."
        )
    if axis.is_multichannel:
        axis.notes.append(
            f"{len(axis.channels)} confinement channels were measured in this field"
            + (f" (spacing {pitch:.0f} px)" if pitch else "")
            + ". Cells are not tracked from one channel into another."
        )
    if source == AXIS_FROM_RIDGES:
        axis.notes.append(
            f"The channels run {abs(axis.tilt_from_vertical_deg):.1f} degrees "
            f"{'left' if axis.tilt_from_vertical_deg < 0 else 'right'} of vertical."
        )
    if ambiguous and forced is None and source == AXIS_FROM_RIDGES:
        axis.notes.append(
            "Straight structure runs both across and down this image; check that the "
            "migration axis overlay follows the channels."
        )
    return axis


def _axis_from_motion(detections: Sequence[Detection]) -> tuple[float, float] | None:
    """Principal direction of frame-to-frame nearest-neighbour displacement."""
    by_frame: dict[int, list[Detection]] = {}
    for d in detections:
        by_frame.setdefault(d.frame, []).append(d)
    steps: list[np.ndarray] = []
    frames = sorted(by_frame)
    for a, b in zip(frames[:-1], frames[1:]):
        if b - a != 1:
            continue
        for da in by_frame[a]:
            nearest = min(
                by_frame[b],
                key=lambda db: float(np.hypot(db.x - da.x, db.y - da.y)),
                default=None,
            )
            if nearest is not None:
                steps.append(np.array([nearest.x - da.x, nearest.y - da.y], dtype=float))
    if len(steps) < 3:
        return None
    m = np.vstack(steps)
    m = m[np.linalg.norm(m, axis=1) > 1e-6]
    if m.shape[0] < 3:
        return None
    vals, vecs = np.linalg.eigh(m.T @ m)  # direction is sign-ambiguous
    principal = vecs[:, int(np.argmax(vals))]
    norm = float(np.hypot(*principal))
    if norm < 1e-9:
        return None
    return (float(principal[0] / norm), float(principal[1] / norm))


def assign_channels(detections: Sequence[Detection], axis: ConfinementAxis) -> None:
    """Stamp every detection with the channel it sits in (in place)."""
    for d in detections:
        d.channel = axis.channel_of(d.x, d.y)
