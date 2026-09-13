"""Establishing the confinement axis and the channel layout.

The research notebook used each detection's own ``regionprops.orientation`` as
the direction of migration.  That is the cell's morphological major axis: it
is unstable for round cells, it is redefined by every detection, and it has
nothing to do with the geometry of the device.

The device itself is the right answer.  In these phase-contrast images the
microfluidic channels appear as high-contrast straight ridges that are static
for the whole time-lapse, so they can be measured once from a temporal median
and then used for every frame.  That gives three things the notebook lacked:

*   a stack-level migration axis, including the device's small tilt,
*   channel centre lines, so associations can be forbidden from crossing a
    wall in a multi-channel field,
*   an honest uncertainty on the axis angle, which the tracker widens its
    confinement prior by when the geometry could not be measured well.

Measured on the supplied data, the device is tilted about 1.2-2.2 degrees from
vertical.  That is small but not negligible: treating the axis as exactly
vertical converts 5 % of every along-channel step into fake lateral motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .config import ConfinementConfig
from .detections import Detection

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

#: A ridge must stand this far above the image beside it, in units of its own
#: row-to-row variability, before it is accepted as a channel wall.
#: Measured on the supplied device, genuine walls score 0.9 to 7; a pure-noise
#: image scores below 0.3. The threshold sits in the gap, with margin on both
#: sides, so a featureless field yields no axis rather than a confident wrong one.
MIN_RIDGE_CONTRAST = 0.5

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
    """One confinement channel: a centre line plus a half width."""

    index: int
    origin: tuple[float, float]  # a point on the centre line, (x, y)
    half_width_px: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "origin_x": self.origin[0],
            "origin_y": self.origin[1],
            "half_width_px": self.half_width_px,
        }


@dataclass
class ConfinementAxis:
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
# Image preparation
# --------------------------------------------------------------------------


def static_projection(stack: np.ndarray, max_frames: int = 32) -> np.ndarray:
    """Temporal median: the device without the cells."""
    arr = np.asarray(stack)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.shape[0] == 0:
        raise ValueError("Cannot measure device geometry from an empty stack.")
    if arr.shape[0] > max_frames:
        arr = arr[np.linspace(0, arr.shape[0] - 1, max_frames).astype(int)]
    return np.median(arr.astype(np.float32), axis=0)


@dataclass
class _RidgeEvidence:
    """What was found when the image is read as channels in one orientation."""

    #: One (slope, intercept, inlier_rows) triple per accepted ridge.
    ridges: list[tuple[float, float, int]] = field(default_factory=list)
    support: float = 0.0  # total inlier rows across all accepted ridges
    inlier_fraction: float = 0.0

    @property
    def slope(self) -> float:
        return float(np.median([r[0] for r in self.ridges])) if self.ridges else 0.0


def _shear_profile(projection: np.ndarray, slope: float) -> tuple[np.ndarray, int]:
    """Collapse along the channel after removing a tilt of ``slope`` (dx per dy)."""
    h, w = projection.shape
    xs = np.arange(w, dtype=np.float64)
    rows = np.empty((h, w), dtype=np.float64)
    centre = (h - 1) / 2.0
    for r in range(h):
        rows[r] = np.interp(xs + slope * (r - centre), xs, projection[r])
    margin = int(math.ceil(abs(slope) * (h - 1) / 2.0)) + 3
    if w - 2 * margin < 16:
        return np.zeros(0), 0
    return rows[:, margin : w - margin].mean(axis=0), w - 2 * margin


def _coarse_slope(projection: np.ndarray) -> float:
    """Rough tilt from a shear sweep; 0 when the field is too narrow to tell.

    A 90-pixel-wide crop leaves only ~20 usable columns after shearing, and the
    sweep then locks onto noise.  Returning 0 there is correct: the ridge trace
    that follows measures the tilt properly.
    """
    best_slope, best_score, best_cols = 0.0, -1.0, 0
    baseline: list[float] = []
    for deg in np.arange(-10.0, 10.001, 0.5):
        slope = math.tan(math.radians(deg))
        profile, cols = _shear_profile(projection, slope)
        if cols < 100:
            continue
        p = (profile - profile.mean()) / (profile.std() + 1e-9)
        score = float(np.mean(np.diff(p) ** 2))
        baseline.append(score)
        if score > best_score:
            best_slope, best_score, best_cols = slope, score, cols
    if not baseline or best_cols < 100:
        return 0.0
    if best_score < 2.5 * float(np.median(baseline)):
        return 0.0
    return best_slope


# --------------------------------------------------------------------------
# Ridge detection
# --------------------------------------------------------------------------


def find_channel_centres(
    projection: np.ndarray,
    slope: float,
    *,
    min_separation_px: int,
    relative_threshold: float = 0.35,
) -> list[float]:
    """Locate channel centres as prominent ridges in the perpendicular profile."""
    profile, _ = _shear_profile(projection, slope)
    if profile.size < 8:
        profile = projection.mean(axis=0)
        offset = 0
    else:
        offset = (projection.shape[1] - profile.size) // 2

    # Remove slow shading so a bright corner cannot outvote a real ridge.
    win = max(9, (profile.size // 6) | 1)
    pad = win // 2
    padded = np.pad(profile, pad, mode="edge")
    background = np.array(
        [np.median(padded[i : i + win]) for i in range(profile.size)], dtype=np.float64
    )
    ridge = profile - background
    if ridge.max() <= 0:
        return []

    # Ignore the outermost few columns: the field edge is not a channel.
    edge = max(2, int(0.02 * ridge.size))
    ridge[:edge] = 0.0
    ridge[-edge:] = 0.0

    threshold = relative_threshold * float(ridge.max())
    order = np.argsort(ridge)[::-1]
    centres: list[float] = []
    for i in order:
        if ridge[i] < threshold:
            break
        if all(abs(i - c) >= min_separation_px for c in centres):
            centres.append(float(i))
    return sorted(c + offset for c in centres)


def trace_ridge(
    projection: np.ndarray,
    centre_x: float,
    half_window: int = 12,
    *,
    slope: float = 0.0,
    iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Follow one ridge down the image, returning (rows, sub-pixel columns).

    The search window re-centres on the currently fitted line at each pass.
    A fixed vertical band would bias the measurement: a ridge tilted 2 degrees
    drifts 11 px across a 324-row image, so it walks out of a narrow fixed
    window and the apparent slope collapses towards zero.  Re-centring makes
    the result insensitive to the window size, which is how we know it is
    measuring the ridge and not the window.
    """
    h, w = projection.shape
    line_slope, line_intercept = float(slope), float(centre_x)
    rows_arr = np.zeros(0)
    cols_arr = np.zeros(0)

    for _ in range(max(1, iterations)):
        rows: list[int] = []
        cols: list[float] = []
        for r in range(h):
            xc = line_intercept + line_slope * r
            lo = max(0, int(round(xc)) - half_window)
            hi = min(w, int(round(xc)) + half_window + 1)
            if hi - lo < 3:
                continue
            seg = projection[r, lo:hi].astype(np.float64)
            seg = seg - seg.min()
            if seg.max() <= 0:
                continue
            weight = seg**3  # concentrate on the ridge crest
            total = weight.sum()
            if total <= 0:
                continue
            cols.append(float((weight * np.arange(lo, hi)).sum() / total))
            rows.append(r)
        rows_arr = np.asarray(rows, dtype=float)
        cols_arr = np.asarray(cols, dtype=float)
        if rows_arr.size < 8:
            break
        line_slope, line_intercept, _ = robust_line_fit(rows_arr, cols_arr)
        if abs(line_slope) >= math.tan(math.radians(20)):
            break
    return rows_arr, cols_arr


def robust_line_fit(
    t: np.ndarray, v: np.ndarray, iterations: int = 5
) -> tuple[float, float, int]:
    """Least squares with iterative outlier rejection. Returns (slope, intercept, n_inliers)."""
    if t.size < 8:
        return 0.0, float(np.mean(v)) if v.size else 0.0, int(t.size)
    design = np.vstack([t, np.ones_like(t)]).T
    keep = np.ones(t.size, dtype=bool)
    slope, intercept = 0.0, float(np.mean(v))
    for _ in range(iterations):
        if keep.sum() < 8:
            break
        sol, *_ = np.linalg.lstsq(design[keep], v[keep], rcond=None)
        slope, intercept = float(sol[0]), float(sol[1])
        residual = v - (design @ sol)
        centre = float(np.median(residual))
        spread = 1.4826 * float(np.median(np.abs(residual - centre))) + 1e-9
        keep = np.abs(residual - centre) < 3.0 * spread
    return slope, intercept, int(keep.sum())


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def _ridge_contrast(
    work: np.ndarray, rows: np.ndarray, cols: np.ndarray, offset: int
) -> float:
    """How much brighter a traced ridge is than the image beside it, in noise units.

    A relative peak threshold alone is not enough: in an image with no channels
    at all, the brightest noise excursion is still the brightest thing present,
    so ridges get "found" and the axis comes back confident and wrong. Comparing
    the ridge against its own flanks, scaled by the image noise, is an absolute
    test that featureless images fail.
    """
    if rows.size < 8:
        return 0.0
    _, w = work.shape
    r = rows.astype(int)
    c = np.clip(np.round(cols).astype(int), 0, w - 1)
    left = work[r, np.clip(c - offset, 0, w - 1)]
    right = work[r, np.clip(c + offset, 0, w - 1)]

    # Compare each row against its own flanks. Comparing pooled values instead
    # would divide by the image's shading from top to bottom, which is real
    # structure rather than noise, and would hide a genuine wall.
    delta = work[r, c] - 0.5 * (left + right)
    centre = float(np.median(delta))
    spread = 1.4826 * float(np.median(np.abs(delta - centre)))
    if spread <= 1e-9:
        spread = float(np.std(delta)) or 1e-9
    return centre / spread


def _measure_ridges(work: np.ndarray, min_sep: int, window: int) -> _RidgeEvidence:
    """Find and fit the straight ridges of an image read as vertical channels.

    ``work`` is oriented so channels run down the rows.  The returned support
    is the total number of inlier rows across accepted ridges, which is what
    makes the two candidate orientations comparable: a real set of confinement
    channels explains far more of the image than the one bright band at the
    edge of a reservoir.
    """
    coarse = _coarse_slope(work)
    candidates = find_channel_centres(work, coarse, min_separation_px=min_sep)
    span = work.shape[0]
    traced: list[tuple[float, float, int]] = []
    for cx in candidates:
        rows, cols = trace_ridge(work, cx, half_window=window)
        if rows.size < max(16, int(0.4 * span)):
            continue
        slope, intercept, n_inliers = robust_line_fit(rows, cols)
        straight = abs(slope) < math.tan(math.radians(15))
        supported = n_inliers >= 0.6 * rows.size
        real = _ridge_contrast(work, rows, cols, window) >= MIN_RIDGE_CONTRAST
        if straight and supported and real:
            traced.append((slope, intercept, n_inliers))
    traced = _consensus_slope_filter(traced)
    if not traced:
        return _RidgeEvidence()
    return _RidgeEvidence(
        ridges=traced,
        support=float(sum(t[2] for t in traced)),
        inlier_fraction=float(np.mean([t[2] for t in traced]) / max(span, 1)),
    )


def resolve_axis(
    stack: np.ndarray,
    config: ConfinementConfig,
    detections: Sequence[Detection] | None = None,
    *,
    pixel_size_um: float | None = None,
) -> ConfinementAxis:
    """Measure one stack-level migration axis and the channel centre lines."""
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

    channels: list[Channel] = []
    for i, c in enumerate(sorted(centres)):
        origin = (float(c), 0.0) if vertical else (0.0, float(c))
        channels.append(Channel(index=i, origin=origin, half_width_px=half_width))
    if not channels:
        origin = (width / 2.0, 0.0) if vertical else (0.0, height / 2.0)
        channels = [
            Channel(index=0, origin=origin, half_width_px=float(max(width, height)))
        ]
    axis.channels = channels
    axis.pitch_px = pitch

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


def _complete_lattice(
    groups: Sequence[tuple[float, float, int]], tolerance: float = 0.25
) -> list[tuple[float, float, int]]:
    """Fill in channels the ridge detector missed.

    A microfluidic array is manufactured at a fixed pitch, so a gap that is
    close to an integer multiple of the median pitch means a wall was too faint
    to detect, not that the device has a wide blank there. Inserting it matters:
    two channels merged into one would let the tracker link a cell across a wall
    it never crossed.
    """
    if len(groups) < 3:
        return list(groups)
    ordered = sorted(groups, key=lambda g: g[1])
    gaps = np.diff([g[1] for g in ordered])
    pitch = float(np.median(gaps))
    if pitch <= 0:
        return ordered
    slope = float(np.median([g[0] for g in ordered]))

    out: list[tuple[float, float, int]] = [ordered[0]]
    for previous, current in zip(ordered[:-1], ordered[1:]):
        gap = current[1] - previous[1]
        multiple = gap / pitch
        rounded = int(round(multiple))
        if rounded >= 2 and abs(multiple - rounded) <= tolerance:
            for k in range(1, rounded):
                out.append((slope, previous[1] + k * gap / rounded, 0))
        out.append(current)
    return out


def _consensus_slope_filter(
    ridges: Sequence[tuple[float, float, int]],
    min_spread: float = 0.006,
    n_sigma: float = 3.0,
) -> list[tuple[float, float, int]]:
    """Drop ridges whose tilt disagrees with the rest.

    The device is rigid: every channel in one field has the same tilt. A ridge
    that fits a visibly different angle has wandered onto something that is not
    a channel wall, so it is discarded rather than allowed to drag the
    consensus. ``min_spread`` (0.006 ~ 0.34 degrees) keeps the filter from
    becoming arbitrarily strict when several ridges happen to agree exactly.
    """
    if len(ridges) < 3:
        return list(ridges)
    slopes = np.array([r[0] for r in ridges], dtype=float)
    centre = float(np.median(slopes))
    spread = max(1.4826 * float(np.median(np.abs(slopes - centre))), min_spread)
    keep = [r for r in ridges if abs(r[0] - centre) <= n_sigma * spread]
    return keep or list(ridges)


def _group_ridges(
    ridges: Sequence[tuple[float, float, int]], min_pitch: float
) -> list[tuple[float, float, int]]:
    """Combine the walls of one channel into a single centre line.

    Each input is a fitted ridge ``(slope, intercept, inlier_rows)``.  Ridges
    whose intercepts are closer than one plausible channel belong to the same
    channel, and are merged by an inlier-weighted mean: the centre line of a
    channel is the average of the two walls that bound it, and the tilt is the
    better-supported of the two measurements.
    """
    if not ridges:
        return []
    ordered = sorted(ridges, key=lambda r: r[1])
    groups: list[list[tuple[float, float, int]]] = [[ordered[0]]]
    for ridge in ordered[1:]:
        # Compare against the group's centre, not its last member: chaining
        # from member to member would merge an entire evenly spaced array of
        # channels into one.
        centre = float(np.mean([g[1] for g in groups[-1]]))
        if ridge[1] - centre < min_pitch:
            groups[-1].append(ridge)
        else:
            groups.append([ridge])

    out: list[tuple[float, float, int]] = []
    for group in groups:
        weights = np.array([max(g[2], 1) for g in group], dtype=float)
        weights /= weights.sum()
        slope = float(np.sum(weights * np.array([g[0] for g in group])))
        intercept = float(np.sum(weights * np.array([g[1] for g in group])))
        out.append((slope, intercept, int(sum(g[2] for g in group))))
    return out


def _merge_close_ridges(centres: Sequence[float], min_pitch: float) -> list[float]:
    """Collapse ridges nearer than ``min_pitch`` onto their midpoint."""
    if not centres:
        return []
    ordered = sorted(float(c) for c in centres)
    groups: list[list[float]] = [[ordered[0]]]
    for c in ordered[1:]:
        if c - groups[-1][-1] < min_pitch:
            groups[-1].append(c)
        else:
            groups.append([c])
    return [float(np.mean(g)) for g in groups]


def _typical_cell_width(detections: Sequence[Detection] | None) -> float:
    """Width of a cell across the channel, which sets the ridge search scale.

    A confined cell is long and thin, so its bounding-box extent (its length)
    is the wrong scale here: using it would search for channel walls a hundred
    pixels apart. The minor axis is the relevant dimension.
    """
    if detections:
        widths = [d.minor_axis_px for d in detections if d.minor_axis_px > 0]
        if widths:
            return float(np.median(widths))
    return 12.0  # median width of the labelled cells in the supplied training set


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
