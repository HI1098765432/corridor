"""Confinement-aware identity tracking by linear assignment.

The cost model
--------------
Every term is a squared residual divided by the spread that residual is
allowed to have, so the total is in chi-square units and a value of 1.0 means
"one standard deviation off".  That is what makes the thresholds readable:
``unmatched_chi2 = 15`` is a plausibility floor of roughly 3.9 sigma, not a
magic number.

    cost(track, detection) =
          ((r_along) / (sigma_along * sqrt(dt))) ^ 2     prediction residual
        + ((s_across) / (sigma_across * sqrt(dt))) ^ 2   confinement prior
        + (ln(a_det / a_track) / sigma_ln_area) ^ 2      appearance
        + w_reversal * (1 - cos(step, velocity)) / 2     direction consistency
        + w_orientation * (1 - |cos(axis_det, axis_track)|)   morphology (optional)

where

    dt        = frames since the track was last observed (>= 1)
    predicted = last_xy + velocity_per_frame * dt
    r_along   = (detection - predicted) . u        u = unit vector along the channel
    s_across  = (detection - last_xy)   . n        n = unit vector across the channel

Two deliberate asymmetries:

*   The **along-channel** term is measured against the motion *prediction*,
    because constant velocity is a good model in the direction a cell is
    actually travelling.
*   The **across-channel** term is measured against the last observed
    position, not the prediction.  The confinement prior is a statement about
    physical lateral movement ("a cell in a 4 um channel does not move
    sideways"), so extrapolating a spurious lateral velocity and then
    forgiving deviation from it would defeat the prior.

Both sigmas grow as sqrt(dt): uncertainty after a gap accumulates like a
random walk, so a 4-frame gap is allowed twice the deviation of a 1-frame gap,
not four times.

Unmatched assignments
---------------------
The assignment problem is solved over a square block matrix

        columns:   detections (n_d)      track-dummies (n_t)
    rows:
      tracks (n_t)     [  C  ]           [ diag(U) , BIG ]
      det-dummies(n_d) [ diag(U) , BIG ] [     0         ]

so "this track has no detection" and "this detection starts a new track" are
choices the optimiser makes, weighed against every real pairing, rather than
pairings forced first and undone afterwards.

This has a consequence worth stating plainly: linking a track to a detection
removes *two* unmatched decisions, so a pairing is accepted when its cost is
below ``2 * unmatched_chi2``, not below ``unmatched_chi2``.  With the defaults
that effective linking threshold is 30, which is also ``gate_chi2`` -- the two
are deliberately the same number so no pairing can be accepted by the
optimiser that the hard gate would have refused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import Scale, TrackingConfig
from .confinement import ConfinementAxis
from .detections import Detection

#: Cost used for structurally forbidden pairings. Large enough that a solution
#: containing one always loses to the all-unmatched solution, finite so that
#: scipy never reports an infeasible problem.
FORBIDDEN = 1.0e6

# Reasons a pair was refused before costing, kept for diagnostics.
GATE_CHANNEL = "different_channel"
GATE_SPEED = "implausible_speed"
GATE_PERP = "lateral_jump"
GATE_AREA = "area_discontinuity"
GATE_GAP = "gap_too_long"
GATE_COST = "above_cost_gate"


class TrackState(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    TERMINATED = "terminated"


@dataclass
class Observation:
    frame: int
    x: float
    y: float
    area_px: float
    det_label: int
    cost: float | None  # None for the observation that created the track
    gap_frames: int  # frames since the previous observation of this track
    eccentricity: float = 0.0
    orientation_rad: float = 0.0
    minor_axis_px: float = 1.0
    channel: int = -1

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass
class CostBreakdown:
    """Every term of one pairing, for inspection and testing."""

    total: float
    along: float = 0.0
    across: float = 0.0
    area: float = 0.0
    direction: float = 0.0
    orientation: float = 0.0
    gated: str | None = None

    @property
    def allowed(self) -> bool:
        return self.gated is None


@dataclass
class Track:
    id: int
    observations: list[Observation] = field(default_factory=list)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    state: TrackState = TrackState.ACTIVE
    channel: int = -1
    flags: set[str] = field(default_factory=set)

    # -- read-only views ---------------------------------------------------
    @property
    def last(self) -> Observation:
        return self.observations[-1]

    @property
    def last_frame(self) -> int:
        return self.observations[-1].frame

    @property
    def last_xy(self) -> np.ndarray:
        return self.observations[-1].xy

    @property
    def last_area(self) -> float:
        return self.observations[-1].area_px

    @property
    def n_obs(self) -> int:
        return len(self.observations)

    @property
    def has_velocity(self) -> bool:
        return self.n_obs >= 2 and bool(np.linalg.norm(self.velocity) > 0)

    def predict(self, frame: int) -> np.ndarray:
        """Constant-velocity prediction that respects how long the track has been missing."""
        dt = int(frame) - self.last_frame
        if dt < 1:
            raise ValueError(
                f"Track {self.id} was last seen at frame {self.last_frame}; "
                f"cannot predict backwards to {frame}."
            )
        return self.last_xy + self.velocity * float(dt)

    def gap_to(self, frame: int) -> int:
        return int(frame) - self.last_frame

    # -- mutation ----------------------------------------------------------
    def observe(
        self,
        detection: Detection,
        *,
        cost: float | None,
        smoothing: float = 0.7,
    ) -> None:
        """Record a detection, updating the per-frame velocity estimate.

        The displacement is divided by the number of elapsed frames before it
        touches the velocity state, so a cell reacquired after a three-frame
        absence does not acquire four times its real speed.
        """
        dt = detection.frame - self.last_frame if self.observations else 0
        if self.observations:
            if dt < 1:
                raise ValueError(
                    f"Track {self.id}: detection at frame {detection.frame} is not after "
                    f"its last observation at frame {self.last_frame}."
                )
            per_frame = (detection.xy - self.last_xy) / float(dt)
            if self.n_obs == 1:
                self.velocity = per_frame
            else:
                self.velocity = smoothing * self.velocity + (1.0 - smoothing) * per_frame

        self.observations.append(
            Observation(
                frame=int(detection.frame),
                x=float(detection.x),
                y=float(detection.y),
                area_px=float(detection.area_px),
                det_label=int(detection.label),
                cost=cost,
                gap_frames=int(dt),
                eccentricity=float(detection.eccentricity),
                orientation_rad=float(detection.orientation_rad),
                minor_axis_px=float(detection.minor_axis_px),
                channel=int(detection.channel),
            )
        )
        if self.channel < 0:
            self.channel = int(detection.channel)
        self.state = TrackState.ACTIVE

    def miss(self, frame: int, max_gap: int) -> None:
        """Age the track after a frame in which it was not matched."""
        if self.gap_to(frame) >= max_gap + 1:
            self.state = TrackState.TERMINATED
        else:
            self.state = TrackState.DORMANT


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def pair_cost(
    track: Track,
    detection: Detection,
    axis: ConfinementAxis,
    scale: Scale,
    cfg: TrackingConfig,
) -> CostBreakdown:
    """Cost of explaining ``detection`` as the next observation of ``track``."""
    frame = detection.frame
    dt = track.gap_to(frame)
    if dt < 1 or dt > cfg.max_delta_frames():
        return CostBreakdown(total=FORBIDDEN, gated=GATE_GAP)

    if (
        cfg.enforce_channel_identity
        and axis.is_multichannel
        and track.channel >= 0
        and detection.channel >= 0
        and track.channel != detection.channel
    ):
        return CostBreakdown(total=FORBIDDEN, gated=GATE_CHANNEL)

    u, n = axis.unit, axis.normal
    step = detection.xy - track.last_xy
    step_len_px = float(np.linalg.norm(step))
    across_px = float(step @ n)

    # -- hard physical gates ------------------------------------------------
    elapsed_min = scale.frames_to_min(dt)
    speed_um_min = scale.px_to_um(step_len_px) / max(elapsed_min, 1e-9)
    if speed_um_min > cfg.max_speed_um_per_min:
        return CostBreakdown(total=FORBIDDEN, gated=GATE_SPEED)

    # An imperfectly known axis angle turns part of an along-channel step into
    # apparent lateral motion. Allow for that explicitly rather than letting a
    # 2-degree device tilt look like a cell jumping sideways.
    along_step_px = abs(float(step @ u))
    axis_slop_px = axis.angle_sigma_rad * along_step_px

    # The cell's own width sets the scale on which its centroid can wander.
    width_px = max(detection.minor_axis_px, track.last.minor_axis_px, 1.0)

    across_um = abs(scale.px_to_um(across_px))
    perp_budget = (
        max(cfg.max_perp_um, scale.px_to_um(cfg.max_perp_widths * width_px))
        * math.sqrt(dt)
        + scale.px_to_um(axis_slop_px)
    )
    if across_um > perp_budget:
        return CostBreakdown(total=FORBIDDEN, gated=GATE_PERP)

    ratio = detection.area_px / max(track.last_area, 1.0)
    if ratio < cfg.area_ratio_min or ratio > cfg.area_ratio_max:
        return CostBreakdown(total=FORBIDDEN, gated=GATE_AREA)

    # -- soft terms ---------------------------------------------------------
    spread = math.sqrt(dt)
    # Along the channel, two uncertainties add: a floor for a stationary cell,
    # and a term proportional to how far this track was predicted to travel,
    # because a moving cell may stall or surge.
    if track.has_velocity:
        predicted_travel_px = abs(float(track.velocity @ u)) * dt
        speed_term_px = cfg.speed_uncertainty_fraction * predicted_travel_px
    else:
        # A track seen only once has no speed estimate at all, and predicting
        # that it stays put is not a measurement. Treat the speed limit as a
        # 3-sigma bound on how far it could have gone, so the very first link
        # of a fast cell is not charged for the tracker's own ignorance.
        max_travel_px = scale.um_to_px(cfg.max_speed_um_per_min * elapsed_min)
        speed_term_px = max_travel_px / 3.0
    sigma_along_px = max(
        math.hypot(scale.um_to_px(cfg.sigma_along_um) * spread, speed_term_px), 1e-6
    )
    # Three independent sources of lateral spread add in quadrature: genuine
    # lateral wander of the cell, centroid instability proportional to the
    # cell's own width, and the projection error of an axis whose angle is
    # only known to +/- angle_sigma.
    sigma_across_px = max(
        math.sqrt(
            (scale.um_to_px(cfg.sigma_perp_um) * spread) ** 2
            + (cfg.perp_width_fraction * width_px) ** 2
            + axis_slop_px**2
        ),
        1e-6,
    )

    residual = detection.xy - track.predict(frame)
    along_px = float(residual @ u)

    c_along = (along_px / sigma_along_px) ** 2
    c_across = (across_px / sigma_across_px) ** 2
    c_area = (math.log(ratio) / max(cfg.sigma_ln_area, 1e-6)) ** 2

    c_dir = 0.0
    noise_px = scale.um_to_px(cfg.direction_noise_floor_um)
    v_len = float(np.linalg.norm(track.velocity))
    if track.has_velocity and step_len_px > noise_px and v_len > noise_px / max(dt, 1):
        cos_ang = float(step @ track.velocity) / (step_len_px * v_len)
        cos_ang = max(-1.0, min(1.0, cos_ang))
        c_dir = cfg.w_reversal * (1.0 - cos_ang) / 2.0

    c_or = 0.0
    if (
        cfg.w_orientation > 0
        and detection.eccentricity >= cfg.orientation_min_eccentricity
        and track.last.eccentricity >= cfg.orientation_min_eccentricity
    ):
        prev_axis = np.array(
            [math.sin(track.last.orientation_rad), math.cos(track.last.orientation_rad)]
        )
        align = abs(float(detection.axis_unit @ prev_axis))
        c_or = cfg.w_orientation * (1.0 - min(1.0, align))

    total = c_along + c_across + c_area + c_dir + c_or
    breakdown = CostBreakdown(
        total=total, along=c_along, across=c_across, area=c_area,
        direction=c_dir, orientation=c_or,
    )
    if total > cfg.gate_chi2:
        breakdown.total = FORBIDDEN
        breakdown.gated = GATE_COST
    return breakdown


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------


def build_assignment_matrix(
    costs: np.ndarray, unmatched_cost: float
) -> np.ndarray:
    """Square block matrix that lets the optimiser choose 'no match'.

    ``costs`` is (n_tracks, n_detections).  The result is
    (n_tracks + n_detections, n_detections + n_tracks).
    """
    n_t, n_d = costs.shape
    size = n_t + n_d
    m = np.full((size, size), FORBIDDEN, dtype=float)
    if n_t and n_d:
        m[:n_t, :n_d] = costs
    # Track i may go unmatched at its own dummy column.
    for i in range(n_t):
        m[i, n_d + i] = unmatched_cost
    # Detection j may start a new track at its own dummy row.
    for j in range(n_d):
        m[n_t + j, j] = unmatched_cost
    # Dummy-to-dummy pairings are free; they only pad the matrix to square.
    m[n_t:, n_d:] = 0.0
    return m


def solve_assignment(
    costs: np.ndarray, unmatched_cost: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Return (matches, unmatched_track_indices, unmatched_detection_indices)."""
    n_t, n_d = costs.shape
    if n_t == 0 and n_d == 0:
        return [], [], []
    if n_t == 0:
        return [], [], list(range(n_d))
    if n_d == 0:
        return [], list(range(n_t)), []

    matrix = build_assignment_matrix(costs, unmatched_cost)
    rows, cols = linear_sum_assignment(matrix)

    matches: list[tuple[int, int]] = []
    matched_t: set[int] = set()
    matched_d: set[int] = set()
    for r, c in zip(rows, cols):
        if r < n_t and c < n_d:
            # Safety net: a forbidden pairing must never survive. It can only
            # appear if the matrix was degenerate, which would be a bug.
            if matrix[r, c] >= FORBIDDEN:
                continue
            matches.append((int(r), int(c)))
            matched_t.add(int(r))
            matched_d.add(int(c))
    unmatched_t = [i for i in range(n_t) if i not in matched_t]
    unmatched_d = [j for j in range(n_d) if j not in matched_d]
    return matches, unmatched_t, unmatched_d


# --------------------------------------------------------------------------
# Tracker
# --------------------------------------------------------------------------


@dataclass
class FrameEvent:
    """What the tracker decided in one frame, for QC."""

    frame: int
    n_detections: int
    n_candidates: int
    n_matched: int
    n_new: int
    n_dormant: int
    n_terminated: int
    merge_suspected: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "detections": self.n_detections,
            "candidate_tracks": self.n_candidates,
            "matched": self.n_matched,
            "new_tracks": self.n_new,
            "dormant": self.n_dormant,
            "terminated": self.n_terminated,
            "merge_suspected_tracks": ";".join(str(t) for t in self.merge_suspected),
            "notes": "; ".join(self.notes),
        }


class ConfinementTracker:
    """Frame-by-frame identity assignment under a confinement prior."""

    def __init__(
        self,
        axis: ConfinementAxis,
        scale: Scale,
        config: TrackingConfig,
        *,
        merge_area_factor: float = 1.4,
    ) -> None:
        self.axis = axis
        self.scale = scale
        self.cfg = config
        self.merge_area_factor = merge_area_factor
        self.tracks: list[Track] = []
        self.events: list[FrameEvent] = []
        self._next_id = 1

    # -- helpers -----------------------------------------------------------
    def _new_track(self, detection: Detection) -> Track:
        track = Track(id=self._next_id)
        self._next_id += 1
        track.observe(detection, cost=None)
        self.tracks.append(track)
        return track

    def _candidates(self, frame: int) -> list[Track]:
        """Tracks that may legally be matched in this frame."""
        limit = self.cfg.max_delta_frames()
        out = []
        for tr in self.tracks:
            if tr.state is TrackState.TERMINATED:
                continue
            gap = tr.gap_to(frame)
            if 1 <= gap <= limit:
                out.append(tr)
            elif gap > limit:
                tr.state = TrackState.TERMINATED
        return out

    # -- main loop ---------------------------------------------------------
    def run(self, detections_by_frame: dict[int, list[Detection]], n_frames: int) -> list[Track]:
        for frame in range(n_frames):
            dets = list(detections_by_frame.get(frame, []))
            candidates = self._candidates(frame)

            if not dets:
                dormant = 0
                terminated = 0
                for tr in candidates:
                    tr.miss(frame, self.cfg.max_gap)
                    dormant += tr.state is TrackState.DORMANT
                    terminated += tr.state is TrackState.TERMINATED
                self.events.append(
                    FrameEvent(frame, 0, len(candidates), 0, 0, dormant, terminated)
                )
                continue

            if not candidates:
                for d in dets:
                    self._new_track(d)
                self.events.append(
                    FrameEvent(frame, len(dets), 0, 0, len(dets), 0, 0)
                )
                continue

            costs = np.full((len(candidates), len(dets)), FORBIDDEN, dtype=float)
            breakdowns: dict[tuple[int, int], CostBreakdown] = {}
            for i, tr in enumerate(candidates):
                for j, d in enumerate(dets):
                    b = pair_cost(tr, d, self.axis, self.scale, self.cfg)
                    breakdowns[(i, j)] = b
                    costs[i, j] = b.total

            matches, unmatched_t, unmatched_d = solve_assignment(
                costs, self.cfg.unmatched_chi2
            )

            for i, j in matches:
                candidates[i].observe(dets[j], cost=float(costs[i, j]))

            dormant = terminated = 0
            for i in unmatched_t:
                tr = candidates[i]
                tr.miss(frame, self.cfg.max_gap)
                dormant += tr.state is TrackState.DORMANT
                terminated += tr.state is TrackState.TERMINATED

            for j in unmatched_d:
                self._new_track(dets[j])

            merged = self._flag_merges(
                frame, candidates, unmatched_t, matches, dets
            )

            self.events.append(
                FrameEvent(
                    frame=frame,
                    n_detections=len(dets),
                    n_candidates=len(candidates),
                    n_matched=len(matches),
                    n_new=len(unmatched_d),
                    n_dormant=dormant,
                    n_terminated=terminated,
                    merge_suspected=merged,
                )
            )

        # Anything still unmatched at the end of the stack simply ends.
        for tr in self.tracks:
            if tr.state is not TrackState.TERMINATED:
                tr.state = TrackState.TERMINATED
        return self.tracks

    # -- merged-mask evidence ---------------------------------------------
    def _flag_merges(
        self,
        frame: int,
        candidates: list[Track],
        unmatched_t: Sequence[int],
        matches: Sequence[tuple[int, int]],
        dets: Sequence[Detection],
    ) -> list[int]:
        """Note when a track's disappearance looks like two cells merging into one mask.

        No centroid is invented.  The tracker records that the identity became
        ambiguous and leaves the track dormant, so the CSV shows a gap rather
        than a fabricated position.
        """
        if not unmatched_t or not matches:
            return []
        winners = [dets[j] for _, j in matches]
        flagged: list[int] = []
        for i in unmatched_t:
            tr = candidates[i]
            try:
                pred = tr.predict(frame)
            except ValueError:
                continue
            for d in winners:
                min_r, min_c, max_r, max_c = d.bbox
                inside = (min_c <= pred[0] <= max_c) and (min_r <= pred[1] <= max_r)
                bulky = d.area_px >= self.merge_area_factor * tr.last_area
                if inside and bulky:
                    tr.flags.add("merge_suspected")
                    flagged.append(tr.id)
                    break
        return flagged


@dataclass
class UnlinkedStart:
    """A track that began mid-stack, and the best case for it being a continuation.

    When a cell disappears and something appears later, the tracker's refusal
    to join them is a judgement. Recording what that judgement was based on
    turns it from an opaque verdict into evidence a reviewer can weigh: how far
    apart, how long a gap, what the match would have cost, and which rule
    refused it.
    """

    track_id: int
    frame: int
    candidate_track_id: int | None = None
    candidate_last_frame: int | None = None
    gap_frames: int | None = None
    distance_px: float | None = None
    along_px: float | None = None
    across_px: float | None = None
    speed_um_per_min: float | None = None
    cost_chi2: float | None = None
    refused_because: str | None = None

    def describe(self) -> str:
        if self.candidate_track_id is None:
            return "No earlier track was a plausible source."
        reason = {
            GATE_GAP: (
                f"the gap of {self.gap_frames} frames is longer than the "
                "tracker is allowed to bridge"
            ),
            GATE_SPEED: "it would have had to move implausibly fast",
            GATE_PERP: "it would have had to jump sideways out of the channel",
            GATE_AREA: "the size change is too large",
            GATE_CHANNEL: "it is in a different channel",
            GATE_COST: "the overall fit was too poor",
        }.get(self.refused_because or "", "the fit was not good enough")
        detail = (
            f"Track {self.candidate_track_id} ended at frame "
            f"{self.candidate_last_frame}, {self.gap_frames} frames earlier"
        )
        if self.distance_px is not None:
            detail += f" and {self.distance_px:.0f} px away"
        if self.speed_um_per_min is not None:
            detail += f" ({self.speed_um_per_min:.2f} um/min if joined)"
        return f"{detail}. Not joined because {reason}."


def explain_unlinked_starts(
    tracks: Sequence[Track],
    axis: ConfinementAxis,
    scale: Scale,
    cfg: TrackingConfig,
) -> list[UnlinkedStart]:
    """For every track that began mid-stack, why it was not joined to an earlier one.

    The gap limit is deliberately ignored while costing, so that a pairing
    refused *only* because it was too far apart in time is reported as exactly
    that, rather than silently vanishing.
    """
    out: list[UnlinkedStart] = []
    ordered = sorted(tracks, key=lambda t: t.observations[0].frame if t.observations else 0)
    for track in ordered:
        if not track.observations:
            continue
        start = track.observations[0]
        if start.frame == 0:
            continue

        # Rebuild the first observation as a detection so the real cost model
        # can be applied to it.
        probe = Detection(
            frame=start.frame, label=start.det_label, x=start.x, y=start.y,
            area_px=start.area_px, bbox=(0, 0, 1, 1), extent_px=1,
            eccentricity=start.eccentricity, orientation_rad=start.orientation_rad,
            major_axis_px=1.0, minor_axis_px=start.minor_axis_px,
            solidity=1.0, touches_border=False, channel=start.channel,
        )

        best: UnlinkedStart | None = None
        for other in tracks:
            if other is track or not other.observations:
                continue
            if other.last_frame >= start.frame:
                continue
            gap = start.frame - other.last_frame
            relaxed = replace(cfg, max_gap=max(cfg.max_gap, gap))
            breakdown = pair_cost(other, probe, axis, scale, relaxed)
            step = probe.xy - other.last_xy
            distance = float(np.linalg.norm(step))
            elapsed = scale.frames_to_min(gap)
            candidate = UnlinkedStart(
                track_id=track.id,
                frame=start.frame,
                candidate_track_id=other.id,
                candidate_last_frame=other.last_frame,
                gap_frames=gap,
                distance_px=distance,
                along_px=float(step @ axis.unit),
                across_px=float(step @ axis.normal),
                speed_um_per_min=(
                    scale.px_to_um(distance) / elapsed
                    if scale.calibrated and elapsed > 0 else None
                ),
                cost_chi2=(None if breakdown.gated else breakdown.total),
                refused_because=breakdown.gated or (
                    GATE_GAP if gap > cfg.max_delta_frames() else None
                ),
            )
            # Prefer the nearest in time, then the cheapest.
            key = (candidate.gap_frames, candidate.cost_chi2 or FORBIDDEN)
            if best is None or key < (best.gap_frames, best.cost_chi2 or FORBIDDEN):
                best = candidate
        out.append(best or UnlinkedStart(track_id=track.id, frame=start.frame))
    return out


def track_detections(
    detections: Iterable[Detection],
    n_frames: int,
    axis: ConfinementAxis,
    scale: Scale,
    config: TrackingConfig,
) -> tuple[list[Track], list[FrameEvent]]:
    """Convenience wrapper: group detections by frame and run the tracker."""
    by_frame: dict[int, list[Detection]] = {}
    for d in detections:
        by_frame.setdefault(int(d.frame), []).append(d)
    tracker = ConfinementTracker(axis, scale, config)
    tracks = tracker.run(by_frame, n_frames)
    return tracks, tracker.events
