"""Where the device walls are -- never which way the cells go.

In these phase-contrast movies the microfluidic channel walls are straight,
bright ridges that do not move for the whole time-lapse, so they can be
measured once, from a temporal median, and used for every frame.  This module
measures them and returns *lanes*: one centre line and one half-width per
channel, each fitted from that channel's own walls.

What it deliberately does not return is a migration axis.  v1 folded every
fitted wall into one stack-level direction and then charged each cell for
moving "across" it.  That direction was a property of the device, not of the
cell, and a cell that turns, a field with no walls or a 3-D stack has no such
direction at all.  The tracker now takes its anisotropy from each cell's own
body (``tracking.py``); the only thing the walls still contribute is the
*lane gate* -- a link from one lane into another is refused -- and that is
applied only when the lanes really were measured from walls
(``ChannelGeometry.applied``).

The ridge detector itself is unchanged from v1 (it moved here from
``confinement.py``): the same projection, shear sweep, re-centring trace,
robust line fit, contrast test, wall grouping and lattice completion, with
the same thresholds.  What changed is what is kept: every lane keeps its own
fitted slope instead of the median of all of them, and its own half-width,
measured as half the distance to its nearest neighbouring lane.

Why not the lumen width between a lane's two walls: measured on the five
supplied stacks, the 2-3 ridge traces that v1 grouped into one channel span
0.15-4.0 px in the wide fields (repeated traces of one bright line, not two
walls), and in the 052924_t1 crop the cell runs along one grouped line while
the third lies 38.49 px from the other two -- grouped only because that is
0.05 px inside the 38.54 px grouping distance.
Nothing in this data establishes which ridges bound a lumen, so a lumen width
is not reported; the lane spacing is measured, and that is what a lane's
territory -- and the lane gate -- needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .config import GeometryConfig
from .detections import Detection

#: Lanes measured from the bright channel walls. The only source for which the
#: lane gate is applied.
GEOMETRY_FROM_RIDGES = "channel_ridges"
#: No walls were found (or wall detection is switched off): there are no lanes.
GEOMETRY_NONE = "none"
#: Lanes rebuilt from a v1 run's ``confinement`` block, for drawing only.
GEOMETRY_FROM_V1 = "v1_manifest"

#: A ridge must stand this far above the image beside it, in units of its own
#: row-to-row variability, before it is accepted as a channel wall.
#: Measured on the supplied device, genuine walls score 0.9 to 7; a pure-noise
#: image scores below 0.3. The threshold sits in the gap, with margin on both
#: sides, so a featureless field yields no lanes rather than confident wrong ones.
MIN_RIDGE_CONTRAST = 0.5

#: How a lane's half-width was obtained. ``lane_spacing``: half the measured
#: distance to its nearest neighbouring lane. ``pitch``: half the stated
#: pitch (lanes read back from a v1 manifest, which stored only that).
#: ``field``: half the field, because there is only one lane and no spacing
#: to measure -- not a measurement, and said so in the notes.
HALF_WIDTH_FROM_SPACING = "lane_spacing"
HALF_WIDTH_FROM_PITCH = "pitch"
HALF_WIDTH_FROM_FIELD = "field"


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class Lane:
    """One confinement channel: a centre line with its own direction.

    ``origin`` is a point on the centre line and ``direction`` the unit vector
    along it, both in image coordinates (x = column, y = row).  ``detected``
    is False for a lane inferred from the lattice spacing because its walls
    were too faint to see: both are gated, but a reviewer is entitled to know
    which boundary was observed and which was assumed.
    """

    index: int
    origin: tuple[float, float]
    direction: tuple[float, float]
    half_width_px: float
    half_width_source: str = HALF_WIDTH_FROM_PITCH
    detected: bool = True
    #: Inlier rows behind this lane's fit, summed over its walls (0 = inferred).
    support_rows: int = 0

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]], dtype=float)

    @property
    def tilt_from_vertical_deg(self) -> float:
        """Signed tilt of this lane away from straight down the image, for display."""
        dx, dy = self.direction
        return math.degrees(math.atan2(dx, dy))

    def offset(self, x: float, y: float) -> float:
        """Signed perpendicular distance of (x, y) from the centre line, in px."""
        d = np.array([x - self.origin[0], y - self.origin[1]], dtype=float)
        return float(d @ self.normal)

    def endpoints(self, shape: Sequence[int]) -> tuple[tuple[float, float], tuple[float, float]]:
        """Two points of the centre line spanning an image of ``shape`` (H, W), for drawing."""
        height, width = float(shape[0]), float(shape[1])
        dx, dy = self.direction
        reach = height + width
        x0, y0 = self.origin
        # Parameterise from the point nearest the image centre so both ends
        # reach past the frame whatever the origin is.
        t_centre = (width / 2.0 - x0) * dx + (height / 2.0 - y0) * dy
        cx, cy = x0 + t_centre * dx, y0 + t_centre * dy
        return (cx - reach * dx, cy - reach * dy), (cx + reach * dx, cy + reach * dy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "origin_x": self.origin[0],
            "origin_y": self.origin[1],
            "direction_x": self.direction[0],
            "direction_y": self.direction[1],
            "tilt_from_vertical_deg": self.tilt_from_vertical_deg,
            "half_width_px": self.half_width_px,
            "half_width_source": self.half_width_source,
            "detected": self.detected,
            "support_rows": self.support_rows,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Lane":
        dx = float(data.get("direction_x", 0.0))
        dy = float(data.get("direction_y", 1.0))
        norm = math.hypot(dx, dy)
        if norm > 0 and abs(norm - 1.0) > 1e-9:  # leave a stored unit vector bit-exact
            dx, dy = dx / norm, dy / norm
        return cls(
            index=int(data.get("index", 0)),
            origin=(float(data.get("origin_x", 0.0)), float(data.get("origin_y", 0.0))),
            direction=(dx, dy),
            half_width_px=float(data.get("half_width_px", 0.0)),
            half_width_source=str(data.get("half_width_source", HALF_WIDTH_FROM_PITCH)),
            detected=bool(data.get("detected", True)),
            support_rows=int(data.get("support_rows", 0)),
        )


@dataclass
class ChannelGeometry:
    """The lanes of one field, how they were found, and whether they gate links.

    ``applied`` is the run-level fact ``run.json`` reports: True only when the
    lanes came from walls (``source == "channel_ridges"``) and the channel
    constraint was not switched off.  A geometry with no lanes, or one rebuilt
    from a v1 manifest for drawing, never gates anything.
    """

    lanes: list[Lane] = field(default_factory=list)
    source: str = GEOMETRY_NONE
    confidence: float = 0.0
    pitch_px: float | None = None
    notes: list[str] = field(default_factory=list)
    applied: bool = False
    #: (height, width) of the field the lanes were measured on, when known.
    image_shape: tuple[int, int] | None = None

    @property
    def n_lanes(self) -> int:
        return len(self.lanes)

    @property
    def is_multilane(self) -> bool:
        return len(self.lanes) > 1

    def lane_of(self, x: float, y: float) -> int:
        """Index of the lane containing (x, y), or -1 when it is in none.

        Between two lanes the nearest centre line wins (a Voronoi split at
        the midline), whatever either lane's half-width says: with irregular
        spacing -- lanes at x = 0, 40 and 120 -- a half-width of half the
        *nearest* spacing (20 px for the middle lane) left x = 65 in no lane,
        and -1 is never gated, so a cell there could have been linked across
        a wall.  Only beyond the outermost lane on a side, where there is no
        neighbour to split with, must the point lie within that lane's
        half-width.  -1 is not "lane 0": a point outside the lattice (or with
        no lanes at all) is never gated, because there is no lane it could be
        refused from.
        """
        if not self.lanes:
            return -1
        best = min(self.lanes, key=lambda lane: abs(lane.offset(x, y)))
        side = best.offset(x, y)
        if side == 0.0:
            return best.index
        for other in self.lanes:
            if other is best:
                continue
            # Where ``other``'s centre line passes near the point, measured
            # from ``best``: a neighbour on the point's side means the point
            # is between two lanes, and the nearer one owns it.
            foot = np.array([x, y], dtype=float) - other.offset(x, y) * other.normal
            if best.offset(float(foot[0]), float(foot[1])) * side > 0:
                return best.index
        if abs(side) <= best.half_width_px:
            return best.index
        return -1

    def with_constraint(self, channel_constraint: str) -> "ChannelGeometry":
        """The same lanes, with ``applied`` set for one tracking constraint.

        ``"off"`` never applies the gate; ``"auto"`` applies it only to lanes
        measured from walls.  The reason is recorded in ``notes`` either way,
        because a reader of ``run.json`` should not have to know the rule.
        """
        from .config import CHANNEL_CONSTRAINT_OFF

        out = ChannelGeometry(
            lanes=list(self.lanes), source=self.source, confidence=self.confidence,
            pitch_px=self.pitch_px, notes=list(self.notes), applied=False,
            image_shape=self.image_shape,
        )
        if channel_constraint == CHANNEL_CONSTRAINT_OFF:
            if self.lanes:
                out.notes.append("The lane constraint is switched off: links between lanes are not refused.")
            return out
        if self.source == GEOMETRY_FROM_RIDGES and self.lanes:
            out.applied = True
            if self.is_multilane:
                out.notes.append(
                    f"{len(self.lanes)} lanes were measured from the channel walls; "
                    "a cell is never linked from one lane into another."
                )
        elif self.lanes:
            out.notes.append(
                f"These lanes come from '{self.source}', not from measured walls, "
                "so they do not constrain tracking."
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "confidence": self.confidence,
            "applied": self.applied,
            "n_lanes": len(self.lanes),
            "pitch_px": self.pitch_px,
            "image_shape": list(self.image_shape) if self.image_shape else None,
            "lanes": [lane.to_dict() for lane in self.lanes],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ChannelGeometry":
        data = data or {}
        shape = data.get("image_shape")
        return cls(
            lanes=[Lane.from_dict(d) for d in data.get("lanes") or []],
            source=str(data.get("source", GEOMETRY_NONE)),
            confidence=float(data.get("confidence") or 0.0),
            pitch_px=(float(data["pitch_px"]) if data.get("pitch_px") is not None else None),
            notes=[str(n) for n in data.get("notes") or []],
            applied=bool(data.get("applied", False)),
            image_shape=(int(shape[0]), int(shape[1])) if shape else None,
        )

    @classmethod
    def from_legacy_manifest(cls, confinement: dict[str, Any] | None) -> "ChannelGeometry":
        """Lanes for *drawing* a v1 run, from its ``run.json["confinement"]``.

        v1 stored one shared direction ``(ux, uy)`` and a centre-line origin
        per channel, so every lane rebuilt here has that one direction -- it
        is what v1 measured, and redrawing it differently would misreport the
        old result.  ``applied`` is False: a v1 run is never re-tracked from
        these lines, and v2's gate rule (walls only) is not v1's.
        """
        confinement = confinement or {}
        ux = float(confinement.get("ux", 0.0))
        uy = float(confinement.get("uy", 1.0))
        norm = math.hypot(ux, uy) or 1.0
        direction = (ux / norm, uy / norm)
        lanes = [
            Lane(
                index=int(ch.get("index", i)),
                origin=(float(ch.get("origin_x", 0.0)), float(ch.get("origin_y", 0.0))),
                direction=direction,
                half_width_px=float(ch.get("half_width_px", 0.0)),
                half_width_source=HALF_WIDTH_FROM_PITCH,
                detected=bool(ch.get("detected", True)),
            )
            for i, ch in enumerate(confinement.get("channels") or [])
        ]
        notes = [
            "Drawn from a v1 run: these lines share the single migration axis v1 "
            f"measured ({confinement.get('source', 'unknown source')}), and do not "
            "constrain anything."
        ]
        return cls(
            lanes=lanes, source=GEOMETRY_FROM_V1,
            confidence=float(confinement.get("confidence") or 0.0),
            pitch_px=(
                float(confinement["pitch_px"]) if confinement.get("pitch_px") is not None else None
            ),
            notes=notes + [str(n) for n in confinement.get("notes") or []],
            applied=False,
        )

    @classmethod
    def from_legacy_axis(cls, axis: Any) -> "ChannelGeometry":
        """Transition only: lanes from a v1 ``ConfinementAxis`` object.

        Lets ``track_detections(dets, n, axis, scale, cfg)`` keep v1's
        behaviour for the pipeline until the integration package passes a
        real ``ChannelGeometry``: v1 refused cross-channel links whenever the
        axis had more than one channel, whatever its source.
        """
        channels = list(getattr(axis, "channels", []) or [])
        direction = (float(axis.ux), float(axis.uy))
        lanes = [
            Lane(
                index=int(ch.index), origin=(float(ch.origin[0]), float(ch.origin[1])),
                direction=direction, half_width_px=float(ch.half_width_px),
                half_width_source=HALF_WIDTH_FROM_PITCH, detected=bool(ch.detected),
            )
            for ch in channels
        ]
        return cls(
            lanes=lanes, source=str(getattr(axis, "source", GEOMETRY_FROM_V1)),
            confidence=float(getattr(axis, "confidence", 0.0)),
            pitch_px=getattr(axis, "pitch_px", None),
            notes=list(getattr(axis, "notes", []) or []),
            applied=len(lanes) > 1,
        )


# --------------------------------------------------------------------------
# Image preparation
# --------------------------------------------------------------------------


def static_projection(stack: np.ndarray, max_frames: int = 32) -> np.ndarray:
    """Temporal median of a ``(T, Y, X)`` stack: the device without the cells.

    A 2-D image is returned as it is.  Callers with a Z axis collapse it
    first (``detect_channels`` does): the walls are the same in every plane.
    """
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
    """Locate candidate walls as prominent ridges in the sheared column profile."""
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


def _ridge_contrast(
    work: np.ndarray, rows: np.ndarray, cols: np.ndarray, offset: int
) -> float:
    """How much brighter a traced ridge is than the image beside it, in noise units.

    A relative peak threshold alone is not enough: in an image with no channels
    at all, the brightest noise excursion is still the brightest thing present,
    so ridges get "found" and come back confident and wrong. Comparing the
    ridge against its own flanks, scaled by the image noise, is an absolute
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


def _consensus_slope_filter(
    ridges: Sequence[tuple[float, float, int]],
    min_spread: float = 0.006,
    n_sigma: float = 3.0,
) -> list[tuple[float, float, int]]:
    """Drop ridges whose tilt disagrees with the rest.

    The device is rigid: its walls are parallel to within the fit error. A
    ridge that fits a visibly different angle has wandered onto something
    that is not a wall, so it is discarded rather than kept as a lane. This is
    a sanity filter on wall *detections*; the lanes that survive still keep
    their own fitted slopes. ``min_spread`` (0.006 ~ 0.34 degrees) keeps the
    filter from becoming arbitrarily strict when several ridges agree exactly.
    """
    if len(ridges) < 3:
        return list(ridges)
    slopes = np.array([r[0] for r in ridges], dtype=float)
    centre = float(np.median(slopes))
    spread = max(1.4826 * float(np.median(np.abs(slopes - centre))), min_spread)
    keep = [r for r in ridges if abs(r[0] - centre) <= n_sigma * spread]
    return keep or list(ridges)


@dataclass
class _WallGroup:
    """The walls (ridges) that bound one channel, and the lane they imply."""

    members: list[tuple[float, float, int]]
    #: True for a lane inserted by lattice completion: no wall was seen.
    inferred: bool = False
    slope_override: float | None = None
    intercept_override: float | None = None

    @property
    def support(self) -> int:
        return int(sum(m[2] for m in self.members))

    @property
    def slope(self) -> float:
        if self.slope_override is not None:
            return self.slope_override
        w = np.array([max(m[2], 1) for m in self.members], dtype=float)
        return float(np.sum(w * np.array([m[0] for m in self.members])) / w.sum())

    @property
    def intercept(self) -> float:
        if self.intercept_override is not None:
            return self.intercept_override
        w = np.array([max(m[2], 1) for m in self.members], dtype=float)
        return float(np.sum(w * np.array([m[1] for m in self.members])) / w.sum())



def _group_walls(
    ridges: Sequence[tuple[float, float, int]], min_pitch: float
) -> list[_WallGroup]:
    """Combine the walls of one channel into one group.

    Ridges whose intercepts are closer than one plausible channel belong to
    the same channel. Compare against the group's centre, not its last
    member: chaining from member to member would merge an entire evenly
    spaced array of channels into one.
    """
    if not ridges:
        return []
    ordered = sorted(ridges, key=lambda r: r[1])
    groups: list[list[tuple[float, float, int]]] = [[ordered[0]]]
    for ridge in ordered[1:]:
        centre = float(np.mean([g[1] for g in groups[-1]]))
        if ridge[1] - centre < min_pitch:
            groups[-1].append(ridge)
        else:
            groups.append([ridge])
    return [_WallGroup(members=g) for g in groups]


def _group_ridges(
    ridges: Sequence[tuple[float, float, int]], min_pitch: float
) -> list[tuple[float, float, int]]:
    """v1 form of :func:`_group_walls`: ``(slope, intercept, support)`` per group."""
    return [(g.slope, g.intercept, g.support) for g in _group_walls(ridges, min_pitch)]


def _complete_lattice(
    groups: Sequence[tuple[float, float, int]], tolerance: float = 0.25
) -> list[tuple[float, float, int]]:
    """Fill in channels the ridge detector missed (v1 tuple form).

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
                # Support 0 marks this centre line as inferred, not measured.
                out.append((slope, previous[1] + k * gap / rounded, 0))
        out.append(current)
    return out


def _complete_wall_lattice(
    groups: Sequence[_WallGroup], tolerance: float = 0.25
) -> list[_WallGroup]:
    """:func:`_complete_lattice` for wall groups.

    An inferred lane gets the slope and intercept interpolated between its two
    measured neighbours: it has no walls of its own to fit, and borrowing the
    neighbours' local tilt is closer to the device than a field-wide median.
    """
    if len(groups) < 3:
        return list(groups)
    ordered = sorted(groups, key=lambda g: g.intercept)
    gaps = np.diff([g.intercept for g in ordered])
    pitch = float(np.median(gaps))
    if pitch <= 0:
        return ordered
    out: list[_WallGroup] = [ordered[0]]
    for previous, current in zip(ordered[:-1], ordered[1:]):
        gap = current.intercept - previous.intercept
        multiple = gap / pitch
        rounded = int(round(multiple))
        if rounded >= 2 and abs(multiple - rounded) <= tolerance:
            for k in range(1, rounded):
                f = k / rounded
                out.append(
                    _WallGroup(
                        members=[], inferred=True,
                        slope_override=(1 - f) * previous.slope + f * current.slope,
                        intercept_override=previous.intercept + f * gap,
                    )
                )
        out.append(current)
    return out


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


def _wall_search_scales(
    config: GeometryConfig,
    detections: Sequence[Detection] | None,
    pixel_size_um: float | None,
) -> tuple[int, float, int]:
    """``(min_sep, min_pitch, window)``, exactly as v1 derived them.

    Ridges closer than one plausible channel are the two walls of the same
    channel; that pitch also sets how wide the ridge-following window may be
    without wandering onto a neighbouring channel.
    """
    typical = _typical_cell_width(detections)
    min_sep = max(8, int(typical * 0.9))
    min_pitch = (
        config.min_channel_pitch_um / pixel_size_um
        if pixel_size_um
        else config.min_channel_pitch_px
    )
    window = int(max(10, min(24, min_pitch / 3.0)))
    return min_sep, float(min_pitch), window


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------


def _as_time_stack(stack: np.ndarray, axes: str | None) -> np.ndarray:
    """Reduce ``stack`` to (T, Y, X) or (Y, X): the walls are the same in every plane."""
    arr = np.asarray(stack)
    if axes:
        axes = axes.upper()
        if "Z" in axes and len(axes) == arr.ndim:
            arr = arr.mean(axis=axes.index("Z"))
        return arr
    if arr.ndim == 4:  # canonical TZYX
        return arr.mean(axis=1)
    return arr


def detect_channels(
    stack: np.ndarray,
    config: GeometryConfig | None = None,
    detections: Sequence[Detection] | None = None,
    *,
    pixel_size_um: float | None = None,
    channel_constraint: str = "auto",
    axes: str | None = None,
) -> ChannelGeometry:
    """Measure the lanes of one field from its static walls.

    ``stack`` is ``(T, Y, X)``, ``(T, Z, Y, X)`` or a single ``(Y, X)``
    image; pass ``axes`` for a ``ZYX`` volume so that Z is not read as time.
    ``channel_constraint`` decides ``applied`` (see
    :meth:`ChannelGeometry.with_constraint`): the gate is applied only to
    lanes measured from walls.
    """
    config = config or GeometryConfig()
    notes: list[str] = []
    projection = static_projection(_as_time_stack(stack, axes))
    height, width = projection.shape
    min_sep, min_pitch, window = _wall_search_scales(config, detections, pixel_size_um)

    if not config.detect_walls:
        return ChannelGeometry(
            source=GEOMETRY_NONE, notes=["Wall detection is switched off; there are no lanes."],
            image_shape=(height, width),
        ).with_constraint(channel_constraint)

    # Read the image both ways and let the evidence decide. A profile-sharpness
    # heuristic is not enough: one bright horizontal reservoir edge can outvote
    # the channels it feeds.
    down = _measure_ridges(projection, min_sep, window)
    across = _measure_ridges(np.ascontiguousarray(projection.T), min_sep, window)
    vertical = down.support >= across.support
    best = down if vertical else across

    if not best.ridges:
        return ChannelGeometry(
            source=GEOMETRY_NONE,
            notes=[
                "No channel walls were found in this field, so there are no lanes and "
                "tracking is not constrained to any."
            ],
            image_shape=(height, width),
        ).with_constraint(channel_constraint)

    groups = _group_walls(best.ridges, min_pitch)
    if len(groups) < len(best.ridges):
        notes.append(
            f"{len(best.ridges)} ridge traces were grouped into {len(groups)} "
            f"lane(s); traces closer than {min_pitch:.0f} px belong to one channel."
        )
    groups = _complete_wall_lattice(groups)
    groups.sort(key=lambda g: g.intercept)

    pitch = None
    if len(groups) > 1:
        gaps = np.diff([g.intercept for g in groups])
        gaps = gaps[gaps > min_sep * 0.5]
        if gaps.size:
            pitch = float(np.median(gaps))

    lanes: list[Lane] = []
    for i, g in enumerate(groups):
        slope, intercept = g.slope, g.intercept
        # A ridge is fitted as column = intercept + slope * row in the
        # oriented image, so its direction there is (slope, 1).
        norm = math.hypot(slope, 1.0)
        if vertical:
            origin = (float(intercept), 0.0)
            direction = (slope / norm, 1.0 / norm)
        else:
            origin = (0.0, float(intercept))
            direction = (1.0 / norm, slope / norm)
        lanes.append(
            Lane(
                index=i, origin=origin, direction=direction,
                half_width_px=0.5 * float(width if vertical else height),
                half_width_source=HALF_WIDTH_FROM_FIELD,
                detected=not g.inferred, support_rows=g.support,
            )
        )
    _set_half_widths(lanes, (height, width))

    n_inferred = sum(1 for lane in lanes if not lane.detected)
    if n_inferred:
        notes.append(
            f"{n_inferred} lane(s) could not be seen directly and were placed from the "
            "spacing of the others. Check them if a cell appears to change lane."
        )
    if any(lane.half_width_source == HALF_WIDTH_FROM_FIELD for lane in lanes):
        notes.append(
            "Only one lane was found, so there is no lane spacing to measure; its "
            "half-width is half the field, not a measurement."
        )
    tilts = [lane.tilt_from_vertical_deg if vertical else lane.tilt_from_vertical_deg - 90.0
             for lane in lanes if lane.detected]
    if tilts:
        notes.append(
            "Measured lane tilts: "
            + ", ".join(f"{t:+.1f}" for t in tilts)
            + (" degrees from vertical." if vertical else " degrees from horizontal.")
        )
    ambiguous = min(down.support, across.support) > 0.8 * max(down.support, across.support, 1.0)
    if ambiguous:
        notes.append(
            "Straight structure runs both across and down this image; check that the "
            "lanes follow the channels."
        )

    geometry = ChannelGeometry(
        lanes=lanes,
        source=GEOMETRY_FROM_RIDGES,
        confidence=float(min(1.0, 0.55 + 0.45 * best.inlier_fraction)),
        pitch_px=pitch,
        notes=notes,
        image_shape=(height, width),
    )
    return geometry.with_constraint(channel_constraint)


def _set_half_widths(lanes: list[Lane], shape: tuple[int, int]) -> None:
    """Half the distance from each lane to its nearest neighbour, at mid-field.

    Measured per lane because lanes are not exactly parallel (their own
    slopes differ by up to 3 degrees on the supplied fields), so one shared
    pitch would be wrong by a pixel or two at the edges of the field.
    """
    if len(lanes) < 2:
        return
    mid = (shape[1] / 2.0, shape[0] / 2.0)
    for lane in lanes:
        t = (mid[0] - lane.origin[0]) * lane.direction[0] + (mid[1] - lane.origin[1]) * lane.direction[1]
        px = lane.origin[0] + t * lane.direction[0]
        py = lane.origin[1] + t * lane.direction[1]
        distances = [abs(other.offset(px, py)) for other in lanes if other is not lane]
        lane.half_width_px = 0.5 * float(min(distances))
        lane.half_width_source = HALF_WIDTH_FROM_SPACING


def assign_lanes(detections: Sequence[Detection], geometry: ChannelGeometry) -> None:
    """Stamp every detection with the lane it sits in (in place; -1 = none)."""
    for d in detections:
        d.channel = geometry.lane_of(d.x, d.y)
