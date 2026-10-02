"""Axis-free identity tracking: per-cell Kalman prediction, block LAP, global gap closing.

Positions
---------
Everything is tracked in isotropic pixel units: ``(x, y)`` in 2-D and
``(x, y, z * anisotropy)`` in 3-D, where ``anisotropy = z_step_um /
pixel_size_um`` (``Scale.anisotropy``).  A 3-D stack whose Z step is unknown
is tracked with anisotropy 1.0 -- one slice counted as one pixel -- and every
track says so (flag ``z_uncalibrated``), because that is an assumption, not a
measurement.  Nothing anywhere receives a migration direction.

Stage 1, frame to frame
-----------------------
Each track carries a constant-velocity Kalman state ``[r, v]`` (pixels and
pixels per frame) with covariance ``P``.  Prediction over ``dt`` frames uses
``F = [[I, dt I], [0, I]]`` and isotropic white-noise-acceleration process
noise of spectral density ``q`` (px^2 per frame^3):

    Q(dt) = q * [[dt^3/3 I, dt^2/2 I], [dt^2/2 I, dt I]]

so the predicted position widens like dt^3 over a gap, not like sqrt(dt).
``velocity_sigma_um_per_min`` is the 1-sigma velocity change over one
*20.0 min* frame, the interval it was measured on (``TrackingConfig``), so
``q`` is a physical rate, ``sigma^2 / 20 min`` in um^2/min^3, converted per run
to ``q * frame_min^3 / pixel_size^2``: the same cells imaged every 5 min get
the same velocity diffusion per hour as every 20 min.  (An uncalibrated
time axis reads the value per frame, like every other physical default.)

``Q`` and the fresh-track prior are isotropic, as the contract (§5) says.  A
body-shaped ``Q`` (across-body acceleration scaled by the body's aspect, the
0.11 ratio measured on the baseline) was tried and withdrawn: that ratio was
measured in channels, where the walls stop sideways motion -- it describes the
device, not the cell -- and in an open field it broke elongated cells that
turn (a 60x8 px cell turning 20 deg per frame at 0.93 um/min became four
tracks).  It survives only as an opt-in for confined fields
(``body_shaped_noise_in_lanes``), applied only where the lane gate is.

Measurement noise is shaped by each cell's own body, never by an axis:

    R = sigma_p^2 I + (shape_position_fraction * major)^2 u u^T
                    + (width_position_fraction * minor)^2 n n^T

with u, n the detection's own major and minor directions (in 3-D the
principal axes of its mask, when it has one).  A long thin confined cell has an
uncertain centroid along its length and a precise one across it -- the
physical reason v1's along/across model worked, derived per cell.

A fresh track starts with zero velocity and an isotropic speed variance
``initial_speed^2`` (``effective_initial_speed_sigma_um_per_min``, 71 px per
frame at the KK2 calibration).  That prior is so wide that a fresh cell's
motion term barely distinguishes 10 px from 30 px; what makes a 10 px
sideways jump of an elongated fresh cell cost more than 30 px along its body
is its own mask: moved sideways by a body width it no longer overlaps itself.
Once a track has moved, ``P`` itself becomes elongated along its learned
motion.

Pair cost, in chi-square units (``CostBreakdown`` reports every term):

    motion      = d^2 = (z - H x_pred)^T S^-1 (z - H x_pred),  S = H P H^T + R
    size        = (ln(size_j / size_i) / sigma_ln_area)^2            area or volume
    shape       = w_shape * [(d ln aspect / 0.35)^2 + (d solidity / 0.10)^2]
    orientation = w_orientation * (1 - |cos angle between major axes|)  ecc >= 0.6
    direction   = w_reversal * (1 - cos(step, velocity)) / 2           moving tracks
    overlap     = w_overlap * (1 - IoU(previous mask shifted by prediction, candidate))
    gap         = gap_penalty_chi2 * (dt - 1)

The overlap term needs both masks (recovered detections and v1 CSV rows have
none).  It is charged only when every pair in the same competition -- the
connected group of tracks and detections that could be linked to each other
in that frame -- has both masks; otherwise it is withheld from the whole group
(``CostBreakdown.overlap_withheld``).  Charging only the pairs that have masks
made a mask-less detection up to ``w_overlap`` cheaper than a primary one:
measured on 052924_2 frame 11 (with the earlier body-shaped noise), a
recovered duplicate came within 0.16 chi2 of taking a track from its primary
detection, which carried a 0.87 overlap charge the duplicate was spared.  With
the term withheld there the primary wins by 0.99
(``docs/tracking_v2_vs_v1_baseline.json``).

Hard gates, checked in this order: gap (``dt > max_gap + 1``), lane, physical
speed, size ratio, the motion gate (``d^2`` above the chi-square 0.999
quantile for the dimension: 13.8 in 2-D, 16.3 in 3-D) and finally
``total > gate_chi2`` where ``gate_chi2 = 2 U`` is derived, never stored.

Assignment
----------
One square block matrix per frame,

        columns:   detections (n_d)      track-dummies (n_t)
    rows:
      tracks (n_t)     [  C  ]           [ diag(U) , BIG ]
      det-dummies(n_d) [ diag(U) , BIG ] [     0         ]

so "this track has no detection" and "this detection starts a new track" are
choices the optimiser weighs against every real pairing.  Linking removes
two unmatched decisions, so a pairing is accepted below ``2 U`` -- which is
why the gate is ``2 U`` and not an independent number that can drift.

Every accepted link carries two margins (chi-square units; margins, not
probabilities):

*   ``link_margin`` -- the contract's: the cheapest other explanation for its
    track or its detection (another detection, another track, or both
    unmatched at ``2 U``) minus the chosen cost.  It can be negative: the
    global optimum gave this track a detection that another track wanted
    more, which is a locally contested link (``link_margins``).
*   ``link_margin_global`` -- how much the whole frame's best explanation
    worsens if this link is forbidden, knock-on effects included
    (``resolve_link_margins``).  Never negative.  It measures how much the
    *solution* depends on the link, not how contested the link is: for costs
    ``[[1, 4], [12, 2]]`` and ``U = 10`` track 0's next-best detection is
    only 3 worse (local margin 3), but the re-solve reports 13 because the
    swap also costs track 1 its detection.  A QC rule for ambiguity reads the
    contract's ``link_margin``.

Stage 2, global gap closing
---------------------------
Every track end is matched against every later track start within
``max_gap + 1`` frames in one assignment.  Evidence comes from both sides:
the end's state predicted forward to the start (``d2_f``), and the later
track filtered backwards over its own observations and predicted back to the
end (``d2_b``).  A side counts as evidence only if its track has at least two
observations -- a one-observation track's "prediction" is just the fresh
prior, and averaging it in would dilute the informative side.  With both
sides, the motion gate is on the joint statistic ``d2_f + d2_b`` against the
chi-square 0.999 quantile for ``2 * ndim`` degrees of freedom (18.5 in 2-D,
22.5 in 3-D) and the motion term is half of it, on the stage-1 per-``ndim``
scale so the same ``U`` and cost gate apply; with one side it is that side's
``d^2`` against the stage-1 gate.  (The two sides share the two endpoint
measurements, so the sum is not exactly chi-square with ``2 * ndim`` degrees
of freedom; the quantile is the convention, stated as one.)  Track ids are
then renumbered by first appearance; ``TrackList.id_map`` maps every stage-1
id of *this run* to its final id.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import CHANNEL_CONSTRAINT_OFF, Scale, TrackingConfig
from .detections import Detection
from .geometry import ChannelGeometry

#: Cost used for structurally forbidden pairings. Large enough that a solution
#: containing one always loses to the all-unmatched solution, finite so that
#: scipy never reports an infeasible problem.
FORBIDDEN = 1.0e6

# Reasons a pair was refused before or after costing, kept for diagnostics.
GATE_CHANNEL = "different_channel"
#: v2 name for the same refusal: the two are in different lanes.
GATE_LANE = GATE_CHANNEL
GATE_SPEED = "implausible_speed"
#: Size ratio outside [area_ratio_min, area_ratio_max]; volume in 3-D.
GATE_AREA = "area_discontinuity"
GATE_SIZE = GATE_AREA
GATE_GAP = "gap_too_long"
GATE_MOTION = "motion_outlier"
GATE_COST = "above_cost_gate"
#: v1 only (a lateral jump across the migration axis). v2 never produces it;
#: kept so a v1 ``unlinked_starts.csv`` row can still be described.
GATE_PERP = "lateral_jump"

#: Quantile of the chi-square distribution used as the motion gate.
MOTION_GATE_QUANTILE = 0.999

#: The frame interval ``velocity_sigma_um_per_min`` was measured on (the KK2
#: baseline, 20.0 min frames; ``TrackingConfig`` docstring). The value is the
#: velocity change over *that* interval, so it is converted to a physical
#: diffusion rate with this, not read per frame of whatever movie is tracked.
#: TrackingConfig has no field for it; see the integration notes.
VELOCITY_SIGMA_REFERENCE_INTERVAL_MIN = 20.0

#: 1-sigma of the shape term's components (contract §5): a change of ln(aspect)
#: by 0.35 (aspect x1.42) or of solidity by 0.10 costs one chi-square unit each
#: before ``w_shape``.
SHAPE_SIGMA_LN_ASPECT = 0.35
SHAPE_SIGMA_SOLIDITY = 0.10

#: A matched detection at least this much larger than a vanished track's last
#: size, sitting where that track was predicted, is evidence of two cells in
#: one mask. v1's value; the same factor, inverted, flags a split.
MERGE_AREA_FACTOR = 1.4

FLAG_MERGE = "merge_suspected"
FLAG_SPLIT = "split_suspected"
FLAG_ENTERS_BORDER = "enters_at_border"
FLAG_EXITS_BORDER = "exits_at_border"
FLAG_GAP_CLOSED = "gap_closed"
FLAG_Z_UNCALIBRATED = "z_uncalibrated"


@functools.lru_cache(maxsize=None)
def motion_gate_chi2(ndim: int) -> float:
    """The chi-square 0.999 quantile for ``ndim`` degrees of freedom.

    13.82 for 2, 16.27 for 3; stage 2's two-sided gate uses ``2 * ndim``
    (18.47 in 2-D, 22.46 in 3-D).
    """
    from scipy.stats import chi2  # imported lazily: scipy.stats is slow to import

    return float(chi2.ppf(MOTION_GATE_QUANTILE, int(ndim)))


class TrackState(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    TERMINATED = "terminated"


# --------------------------------------------------------------------------
# Motion model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MotionModel:
    """Everything the Kalman filter needs, converted once to pixels and frames."""

    ndim: int
    #: XY pixels per Z slice; 1.0 when the Z step is unknown (and then assumed).
    anisotropy: float
    z_assumed: bool
    #: White-noise-acceleration spectral density, px^2 per frame^3.
    accel_var_px2_per_frame3: float
    position_sigma_px: float
    shape_position_fraction: float
    width_position_fraction: float
    initial_speed_px_per_frame: float
    orientation_min_eccentricity: float
    #: Opt-in only (see the module docstring): shape Q and the fresh-track
    #: prior by the cell body instead of isotropically. False is the contract.
    body_shaped_noise: bool = False

    @classmethod
    def from_config(
        cls, scale: Scale, cfg: TrackingConfig, ndim: int = 2, *, body_shaped_noise: bool = False
    ) -> "MotionModel":
        # velocity_sigma is the 1-sigma velocity change over one 20.0 min
        # frame, where it was measured. White-noise acceleration makes the
        # velocity variance grow linearly in time, so the physical density is
        # sigma^2 / 20 min (um^2/min^3), and over one frame of this movie the
        # position-noise density is that times frame_min^3, in px^2. Read per
        # frame instead, 5 min frames would get twice the measured velocity
        # sigma per 20 min and 80 min frames half of it.
        frame_min = scale.frame_interval_min
        sigma_px_per_min = scale.um_to_px(cfg.velocity_sigma_um_per_min)
        if scale.calibrated_time:
            accel_var = sigma_px_per_min**2 * frame_min**3 / VELOCITY_SIGMA_REFERENCE_INTERVAL_MIN
        else:
            # No time calibration: physical defaults are read as frames
            # (Scale's rule), so the value is the change per frame.
            accel_var = (sigma_px_per_min * frame_min) ** 2
        initial = scale.um_to_px(cfg.effective_initial_speed_sigma_um_per_min) * frame_min
        anisotropy = scale.anisotropy
        return cls(
            ndim=int(ndim),
            anisotropy=float(anisotropy) if anisotropy else 1.0,
            z_assumed=bool(ndim == 3 and not anisotropy),
            accel_var_px2_per_frame3=float(accel_var),
            position_sigma_px=float(scale.um_to_px(cfg.position_sigma_um)),
            shape_position_fraction=float(cfg.shape_position_fraction),
            width_position_fraction=float(cfg.width_position_fraction),
            initial_speed_px_per_frame=float(initial),
            orientation_min_eccentricity=float(cfg.orientation_min_eccentricity),
            body_shaped_noise=bool(body_shaped_noise),
        )

    @classmethod
    def default(cls, ndim: int = 2) -> "MotionModel":
        """The model of an uncalibrated dataset: physical defaults read as px and frames."""
        return cls.from_config(Scale.from_values(None, None), TrackingConfig(), ndim)

    # -- coordinates -------------------------------------------------------
    def position(self, det: Detection) -> np.ndarray:
        """Isotropic pixel position of a detection."""
        if self.ndim == 3:
            z = float(det.z) if det.z is not None else 0.0
            return np.array([det.x, det.y, z * self.anisotropy], dtype=float)
        return np.array([det.x, det.y], dtype=float)

    def to_image(self, p: np.ndarray) -> np.ndarray:
        """Isotropic position (or velocity) back to image units (z in slices)."""
        out = np.array(p, dtype=float)
        if self.ndim == 3:
            out[2] = out[2] / self.anisotropy
        return out

    # -- the cell body -----------------------------------------------------
    def body_axes(self, det: Detection) -> tuple[np.ndarray, np.ndarray]:
        """``(lengths, axes)``: full lengths (longest first, px) and unit vectors (rows).

        2-D: the detection's own major/minor axes.  3-D: the principal axes of
        its mask in isotropic pixels when it carries one (full lengths
        ``sqrt(20 * eigenvalue)``, exact for a solid ellipsoid); otherwise the
        XY footprint's axes plus Z with the minor length.
        """
        major = max(float(det.major_axis_px), 0.0)
        minor = max(float(det.minor_axis_px), 0.0)
        u = np.array([math.sin(det.orientation_rad), math.cos(det.orientation_rad)], dtype=float)
        n = np.array([-u[1], u[0]], dtype=float)
        if self.ndim == 2:
            return np.array([major, minor]), np.vstack([u, n])
        crop = det.mask_crop
        if crop is not None and np.asarray(crop).ndim == 3 and np.count_nonzero(crop) > 1:
            zyx = np.argwhere(np.asarray(crop)).astype(float)
            pts = np.column_stack([zyx[:, 2], zyx[:, 1], zyx[:, 0] * self.anisotropy])
            cov = np.cov(pts, rowvar=False, bias=True)
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            lengths = np.sqrt(20.0 * np.clip(vals[order], 0.0, None))
            return lengths, vecs[:, order].T
        axes = np.zeros((3, 3))
        axes[0, :2] = u
        axes[1, :2] = n
        axes[2, 2] = 1.0
        return np.array([major, minor, minor]), axes

    def measurement_cov(self, det: Detection) -> np.ndarray:
        """R for one detection: the isotropic floor plus its own body shape."""
        d = self.ndim
        lengths, axes = self.body_axes(det)
        R = (self.position_sigma_px**2) * np.eye(d)
        for k in range(d):
            fraction = self.shape_position_fraction if k == 0 else self.width_position_fraction
            e = axes[k]
            R += (fraction * lengths[k]) ** 2 * np.outer(e, e)
        return R

    def motion_shape(self, det: Detection | None) -> np.ndarray:
        """How a cell's velocity may change, by direction.

        The identity -- isotropic, the contract -- unless ``body_shaped_noise``
        is set, and then ``u u^T + (minor/major)^2 n n^T``: unit along the
        body, the body's own aspect across it (still the identity for a round
        cell or when there is no body to read).  The shaped form rests on the
        baseline's across/along acceleration ratio, 0.11, which was measured
        in channels whose walls stop sideways motion: a property of the
        device.  In an open field it broke an elongated cell turning 20 deg
        per frame at 0.93 um/min into four tracks, so it is only ever used
        where the lane gate says the walls are real.
        """
        d = self.ndim
        if det is None or not self.body_shaped_noise:
            return np.eye(d)
        lengths, axes = self.body_axes(det)
        if float(det.eccentricity) < self.orientation_min_eccentricity or not lengths[0] > 0:
            return np.eye(d)
        B = np.zeros((d, d))
        for k in range(d):
            ratio = 1.0 if k == 0 else min(1.0, float(lengths[k]) / float(lengths[0]))
            B += ratio**2 * np.outer(axes[k], axes[k])
        return B

    def initial_velocity_cov(self, det: Detection) -> np.ndarray:
        """Velocity prior of a fresh track: unknown speed, ``initial_speed^2`` per component."""
        return self.initial_speed_px_per_frame**2 * self.motion_shape(det)

    # -- Kalman primitives -------------------------------------------------
    def transition(self, dt: float, det: Detection | None = None) -> tuple[np.ndarray, np.ndarray]:
        """``(F, Q)`` for ``dt`` frames (``dt`` > 0); ``det`` matters only with body-shaped noise."""
        d = self.ndim
        F = np.eye(2 * d)
        F[:d, d:] = dt * np.eye(d)
        qB = self.accel_var_px2_per_frame3 * self.motion_shape(det)
        Q = np.zeros((2 * d, 2 * d))
        Q[:d, :d] = qB * dt**3 / 3.0
        Q[:d, d:] = Q[d:, :d] = qB * dt**2 / 2.0
        Q[d:, d:] = qB * dt
        return F, Q

    def predict(
        self, mean: np.ndarray, cov: np.ndarray, dt: float, det: Detection | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict ``dt`` frames ahead; ``det`` (the last detection) shapes Q only when opted in."""
        F, Q = self.transition(dt, det)
        return F @ mean, F @ cov @ F.T + Q

    def initial(self, det: Detection, R: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        d = self.ndim
        R = self.measurement_cov(det) if R is None else R
        mean = np.zeros(2 * d)
        mean[:d] = self.position(det)
        cov = np.zeros((2 * d, 2 * d))
        cov[:d, :d] = R
        cov[d:, d:] = self.initial_velocity_cov(det)
        return mean, cov

    def update(
        self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray, R: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Joseph-form update: stays symmetric positive definite when R is tiny."""
        d = self.ndim
        S = cov[:d, :d] + R
        K = np.linalg.solve(S, cov[:d, :]).T  # (2d, d) == P H^T S^-1
        mean = mean + K @ (z - mean[:d])
        IKH = np.eye(2 * d)
        IKH[:, :d] -= K
        cov = IKH @ cov @ IKH.T + K @ R @ K.T
        return mean, 0.5 * (cov + cov.T)


def _mahalanobis(innovation: np.ndarray, S: np.ndarray) -> float:
    try:
        return float(innovation @ np.linalg.solve(S, innovation))
    except np.linalg.LinAlgError:
        return float("inf")


# --------------------------------------------------------------------------
# Observations and tracks
# --------------------------------------------------------------------------


@dataclass
class CostBreakdown:
    """Every term of one pairing, for inspection, testing and the audit."""

    total: float
    motion: float = 0.0
    size: float = 0.0
    shape: float = 0.0
    orientation: float = 0.0
    direction: float = 0.0
    overlap: float = 0.0
    gap: float = 0.0
    gated: str | None = None
    #: ``d^2`` of the motion term even when a gate fired before it was added
    #: (None when it could not be computed, e.g. a gap gate).
    mahalanobis: float | None = None
    #: Stage 2 only: the two one-sided ``d^2`` (always computed, for the audit)
    #: and which of them counted as evidence: "forward+backward" (motion is
    #: half their sum, gated at the 2*ndim quantile), "forward" or "backward"
    #: (that side alone, gated like stage 1). A side counts only when its
    #: track has at least two observations.
    motion_forward: float | None = None
    motion_backward: float | None = None
    motion_evidence: str | None = None
    #: The mask IoU behind ``overlap``; None when a mask was missing.
    iou: float | None = None
    #: True when ``iou`` was measured but not charged, because another pair in
    #: the same competition had no mask (charging only the masked pairs makes
    #: mask-less detections cheaper; see the module docstring).
    overlap_withheld: bool = False

    @property
    def allowed(self) -> bool:
        return self.gated is None

    @property
    def area(self) -> float:
        """v1 name of :attr:`size`."""
        return self.size

    def terms_sum(self) -> float:
        return (
            self.motion + self.size + self.shape + self.orientation
            + self.direction + self.overlap + self.gap
        )


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
    source: str = "primary"
    confidence: float = 1.0
    #: Centroid plane in slices; None in 2-D.
    z: float | None = None
    #: Chi-square margin of the link that produced this observation, the
    #: contract's local definition (``link_margins``; may be negative); None
    #: for a track's first observation.
    link_margin: float | None = None
    #: The full measured detection, mask included (None for a synthetic one).
    detection: Detection | None = field(default=None, repr=False, compare=False)
    breakdown: CostBreakdown | None = field(default=None, repr=False, compare=False)
    major_axis_px: float = 0.0
    #: How much the frame's optimal assignment worsens if this link is
    #: forbidden (``resolve_link_margins``; never negative). Not the
    #: contract's margin: see the module docstring for the difference.
    link_margin_global: float | None = None

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @property
    def position(self) -> np.ndarray:
        if self.z is None:
            return self.xy
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def size(self) -> float:
        if self.detection is not None:
            return self.detection.size
        return float(self.area_px)


@dataclass
class Track:
    id: int
    observations: list[Observation] = field(default_factory=list)
    state: TrackState = TrackState.ACTIVE
    #: Lane index (-1 none): the first lane this track was seen in.
    channel: int = -1
    flags: set[str] = field(default_factory=set)
    model: MotionModel | None = field(default=None, repr=False, compare=False)
    #: Stage-1 ids joined into this track by gap closing, in time order.
    source_ids: list[int] = field(default_factory=list)
    _mean: np.ndarray | None = field(default=None, repr=False, compare=False)
    _cov: np.ndarray | None = field(default=None, repr=False, compare=False)

    # -- read-only views ---------------------------------------------------
    @property
    def last(self) -> Observation:
        return self.observations[-1]

    @property
    def last_frame(self) -> int:
        return self.observations[-1].frame

    @property
    def first_frame(self) -> int:
        return self.observations[0].frame

    @property
    def last_xy(self) -> np.ndarray:
        return self.observations[-1].xy

    @property
    def last_area(self) -> float:
        return self.observations[-1].area_px

    @property
    def last_size(self) -> float:
        return self.observations[-1].size

    @property
    def n_obs(self) -> int:
        return len(self.observations)

    @property
    def has_velocity(self) -> bool:
        return self.n_obs >= 2

    @property
    def kalman_state(self) -> tuple[np.ndarray, np.ndarray]:
        """``(mean, cov)`` after the last observation, in isotropic px and px/frame."""
        if self._mean is None or self._cov is None:
            raise ValueError(f"Track {self.id} has no observations.")
        return self._mean, self._cov

    @property
    def velocity(self) -> np.ndarray:
        """Filtered velocity in image units per frame (z in slices per frame)."""
        if self._mean is None or self.model is None:
            return np.zeros(2, dtype=float)
        d = self.model.ndim
        return self.model.to_image(self._mean[d:])

    def predict_state(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        dt = int(frame) - self.last_frame
        if dt < 1:
            raise ValueError(
                f"Track {self.id} was last seen at frame {self.last_frame}; "
                f"cannot predict backwards to {frame}."
            )
        mean, cov = self.kalman_state
        assert self.model is not None
        return self.model.predict(mean, cov, float(dt), self.last.detection)

    def predict(self, frame: int) -> np.ndarray:
        """Constant-velocity prediction in image units (``(x, y)`` or ``(x, y, z)``)."""
        mean, _ = self.predict_state(frame)
        assert self.model is not None
        return self.model.to_image(mean[: self.model.ndim])

    def gap_to(self, frame: int) -> int:
        return int(frame) - self.last_frame

    # -- mutation ----------------------------------------------------------
    def observe(
        self,
        detection: Detection,
        *,
        cost: float | None = None,
        link_margin: float | None = None,
        breakdown: CostBreakdown | None = None,
        lane: int | None = None,
        R: np.ndarray | None = None,
        link_margin_global: float | None = None,
        prediction: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Record a detection and run the Kalman update.

        The elapsed frames enter the transition matrix, so the velocity state
        is per frame however long the track was missing.
        """
        if self.model is None:
            self.model = MotionModel.default(detection.ndim)
        model = self.model
        R = model.measurement_cov(detection) if R is None else R
        if not self.observations:
            dt = 0
            self._mean, self._cov = model.initial(detection, R)
        else:
            dt = int(detection.frame) - self.last_frame
            if dt < 1:
                raise ValueError(
                    f"Track {self.id}: detection at frame {detection.frame} is not after "
                    f"its last observation at frame {self.last_frame}."
                )
            mean, cov = self.predict_state(detection.frame) if prediction is None else prediction
            self._mean, self._cov = model.update(mean, cov, model.position(detection), R)

        lane = int(detection.channel) if lane is None else int(lane)
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
                channel=lane,
                source=str(getattr(detection, "source", "primary")),
                confidence=float(getattr(detection, "confidence", 1.0)),
                z=None if detection.z is None else float(detection.z),
                link_margin=link_margin,
                detection=detection,
                breakdown=breakdown,
                major_axis_px=float(detection.major_axis_px),
                link_margin_global=link_margin_global,
            )
        )
        if self.channel < 0 and lane >= 0:
            self.channel = lane
        self.state = TrackState.ACTIVE

    def miss(self, frame: int, max_gap: int) -> None:
        """Age the track after a frame in which it was not matched."""
        if self.gap_to(frame) >= max_gap + 1:
            self.state = TrackState.TERMINATED
        else:
            self.state = TrackState.DORMANT


class TrackList(list):
    """``list[Track]`` that also carries what gap closing did to the ids.

    ``id_map`` maps every stage-1 track id *of this run* to the final,
    renumbered id.  It cannot re-key anything produced by a different run:
    the pipeline recovers detections against a first pass's final tracks and
    then tracks again from scratch, so recovery provenance must be re-keyed
    by detection membership -- the ``(frame, det_label)`` of the recovered
    detection in the final tracks -- not through this map.
    """

    def __init__(self, tracks: Iterable[Track] = (), id_map: Mapping[int, int] | None = None,
                 notes: Sequence[str] = ()):
        super().__init__(tracks)
        self.id_map: dict[int, int] = dict(id_map or {})
        self.notes: list[str] = list(notes)


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def _is_legacy_axis(obj: Any) -> bool:
    """A v1 ``ConfinementAxis`` passed where v2 expects a Scale (transition only)."""
    return obj is not None and hasattr(obj, "ux") and hasattr(obj, "uy") and hasattr(obj, "channels")


def lane_gate_applies(geometry: ChannelGeometry | None, cfg: TrackingConfig) -> bool:
    """Whether links between lanes are refused for this geometry and configuration."""
    if geometry is None or not geometry.applied or not geometry.lanes:
        return False
    if cfg.channel_constraint == CHANNEL_CONSTRAINT_OFF:
        return False
    return bool(cfg.enforce_channel_identity)


def _aspect(det: Detection | None, obs: Observation | None = None) -> float | None:
    if det is not None:
        if det.ndim == 3 and det.elongation:
            return float(det.elongation)
        if det.minor_axis_px > 0 and det.major_axis_px > 0:
            return float(det.major_axis_px) / float(det.minor_axis_px)
        return None
    if obs is not None and obs.minor_axis_px > 0 and obs.major_axis_px > 0:
        return obs.major_axis_px / obs.minor_axis_px
    return None


def _shape_cost(prev: Detection | None, prev_obs: Observation, det: Detection, cfg: TrackingConfig) -> float:
    if cfg.w_shape <= 0:
        return 0.0
    total = 0.0
    a0, a1 = _aspect(prev, prev_obs), _aspect(det)
    if a0 and a1 and a0 > 0 and a1 > 0:
        total += (math.log(a1 / a0) / SHAPE_SIGMA_LN_ASPECT) ** 2
    if prev is not None:
        s0, s1 = float(prev.solidity), float(det.solidity)
        if math.isfinite(s0) and math.isfinite(s1):
            total += ((s1 - s0) / SHAPE_SIGMA_SOLIDITY) ** 2
    return cfg.w_shape * total


def shifted_iou(prev: Detection | None, cand: Detection | None, shift: Sequence[float]) -> float | None:
    """IoU of ``prev``'s mask moved by ``shift`` and ``cand``'s mask, in a common frame.

    ``shift`` is in image units, ordered like the bbox: (dy, dx) in 2-D,
    (dz, dy, dx) in 3-D, and rounded to whole pixels/slices.  None when either
    detection carries no mask (a recovered detection, a v1 CSV row), so the
    term is skipped rather than charged.
    """
    if prev is None or cand is None:
        return None
    a, b = prev.mask_crop, cand.mask_crop
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    nd = a.ndim
    if b.ndim != nd or len(prev.bbox) != 2 * nd or len(cand.bbox) != 2 * nd:
        return None
    a_lo = np.asarray(prev.bbox[:nd], dtype=int) + np.rint(np.asarray(shift, dtype=float)).astype(int)
    b_lo = np.asarray(cand.bbox[:nd], dtype=int)
    if tuple(np.asarray(prev.bbox[nd:]) - np.asarray(prev.bbox[:nd])) != a.shape:
        return None
    if tuple(np.asarray(cand.bbox[nd:]) - np.asarray(cand.bbox[:nd])) != b.shape:
        return None
    lo = np.maximum(a_lo, b_lo)
    hi = np.minimum(a_lo + np.asarray(a.shape), b_lo + np.asarray(b.shape))
    inter = 0
    if np.all(hi > lo):
        sa = tuple(slice(int(l - o), int(h - o)) for l, h, o in zip(lo, hi, a_lo))
        sb = tuple(slice(int(l - o), int(h - o)) for l, h, o in zip(lo, hi, b_lo))
        inter = int(np.count_nonzero(a[sa] & b[sb]))
    union = int(a.sum()) + int(b.sum()) - inter
    return inter / union if union > 0 else None


@dataclass
class _Context:
    """What one tracking run needs on every pair: scale, config, model, lanes."""

    scale: Scale
    cfg: TrackingConfig
    model: MotionModel
    geometry: ChannelGeometry | None = None

    @property
    def lane_gate(self) -> bool:
        return lane_gate_applies(self.geometry, self.cfg)

    def lane_of(self, det: Detection) -> int:
        if self.geometry is not None:
            return self.geometry.lane_of(det.x, det.y)
        return int(det.channel)


def _hard_gates(
    ctx: _Context,
    track_lane: int,
    last_obs: Observation,
    p_last: np.ndarray,
    det: Detection,
    p_det: np.ndarray,
    det_lane: int,
    dt: int,
) -> str | None:
    cfg, scale = ctx.cfg, ctx.scale
    if dt < 1 or dt > cfg.max_delta_frames():
        return GATE_GAP
    if ctx.lane_gate and track_lane >= 0 and det_lane >= 0 and track_lane != det_lane:
        return GATE_LANE
    step_px = float(np.linalg.norm(p_det - p_last))
    speed = scale.px_to_um(step_px) / max(scale.frames_to_min(dt), 1e-9)
    if speed > cfg.max_speed_um_per_min:
        return GATE_SPEED
    ratio = det.size / max(last_obs.size, 1.0)
    if ratio < cfg.area_ratio_min or ratio > cfg.area_ratio_max:
        return GATE_AREA
    return None


def _soft_terms(
    ctx: _Context,
    b: CostBreakdown,
    last_obs: Observation,
    p_last: np.ndarray,
    velocity_px: np.ndarray | None,
    det: Detection,
    p_det: np.ndarray,
    predicted_shift_image: np.ndarray,
    dt: int,
) -> None:
    """Fill every non-motion term of ``b`` in place."""
    cfg, scale = ctx.cfg, ctx.scale
    ratio = det.size / max(last_obs.size, 1.0)
    b.size = (math.log(ratio) / max(cfg.sigma_ln_area, 1e-6)) ** 2
    prev = last_obs.detection
    b.shape = _shape_cost(prev, last_obs, det, cfg)

    if (
        cfg.w_orientation > 0
        and det.eccentricity >= cfg.orientation_min_eccentricity
        and last_obs.eccentricity >= cfg.orientation_min_eccentricity
    ):
        prev_axis = np.array(
            [math.sin(last_obs.orientation_rad), math.cos(last_obs.orientation_rad)]
        )
        align = abs(float(det.axis_unit @ prev_axis))
        b.orientation = cfg.w_orientation * (1.0 - min(1.0, align))

    if velocity_px is not None and cfg.w_reversal > 0:
        step = p_det - p_last
        step_len = float(np.linalg.norm(step))
        v_len = float(np.linalg.norm(velocity_px))
        noise_px = scale.um_to_px(cfg.direction_noise_floor_um)
        if step_len > noise_px and v_len > noise_px / max(dt, 1):
            cos_ang = max(-1.0, min(1.0, float(step @ velocity_px) / (step_len * v_len)))
            b.direction = cfg.w_reversal * (1.0 - cos_ang) / 2.0

    if cfg.w_overlap > 0:
        # bbox order is (row, col) / (z, row, col); the shift is (x, y[, z]).
        shift = predicted_shift_image
        ordered = [shift[1], shift[0]] if len(shift) == 2 else [shift[2], shift[1], shift[0]]
        iou = shifted_iou(prev, det, ordered)
        b.iou = iou
        if iou is not None:
            b.overlap = cfg.w_overlap * (1.0 - iou)

    b.gap = cfg.gap_penalty_chi2 * max(dt - 1, 0)


def _finish(ctx: _Context, b: CostBreakdown) -> CostBreakdown:
    """Sum the terms and apply the cost gate (``total > 2U`` is forbidden)."""
    b.total = b.terms_sum()
    if b.total > ctx.cfg.effective_gate_chi2:
        b.total = FORBIDDEN
        b.gated = GATE_COST
    return b


def _withhold_partial_overlap(cfg: TrackingConfig, breakdowns: Mapping[tuple[int, int], CostBreakdown]) -> None:
    """Charge the overlap term to a competition only if every pair in it has masks.

    ``breakdowns`` maps (row, column) to an unfinished breakdown; pairs with
    ``gated`` set are not competitors.  Rows and columns joined by an ungated
    pair form a connected group, and the assignment compares costs only
    within such a group (and against the unmatched cost).  If any pair of a
    group lacks an IoU, the overlap term is removed from every pair of the
    group, so a mask-less detection is never cheaper merely for having no
    mask.  The IoU stays recorded; ``overlap_withheld`` says why it was not
    charged.
    """
    if cfg.w_overlap <= 0:
        return
    live = [(key, b) for key, b in breakdowns.items() if b.gated is None]
    if not live:
        return
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(node: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for (i, j), _ in live:
        ra, rb = find(("r", i)), find(("c", j))
        if ra != rb:
            parent[ra] = rb
    incomplete = {find(("r", i)) for (i, _j), b in live if b.iou is None}
    for (i, _j), b in live:
        if find(("r", i)) in incomplete:
            b.overlap_withheld = b.iou is not None
            b.overlap = 0.0


def _pair_cost(
    ctx: _Context,
    track: Track,
    det: Detection,
    *,
    R: np.ndarray | None = None,
    det_lane: int | None = None,
    prediction: tuple[np.ndarray, np.ndarray] | None = None,
    finish: bool = True,
    diagnose: bool = True,
) -> CostBreakdown:
    """Cost of one pair.

    ``prediction`` is ``track.predict_state(det.frame)`` when the caller has
    it already (one per track per frame, not one per pair).  ``finish=False``
    leaves ``total`` unsummed and the cost gate unapplied, so the frame can
    first make the overlap term consistent (``_withhold_partial_overlap``).
    ``diagnose=False`` skips the Mahalanobis distance of a pair a cheaper gate
    already refused -- it is reported only for the audit.
    """
    model = ctx.model
    dt = track.gap_to(det.frame)
    last = track.last
    p_last = model.position(last.detection or _probe_detection(last))
    p_det = model.position(det)
    det_lane = ctx.lane_of(det) if det_lane is None else det_lane
    gated = _hard_gates(ctx, track.channel, last, p_last, det, p_det, det_lane, dt)
    if gated == GATE_GAP or (gated is not None and not diagnose):
        return CostBreakdown(total=FORBIDDEN, gated=gated)

    R = model.measurement_cov(det) if R is None else R
    mean, cov = track.predict_state(det.frame) if prediction is None else prediction
    d = model.ndim
    d2 = _mahalanobis(p_det - mean[:d], cov[:d, :d] + R)
    if gated is not None:
        return CostBreakdown(total=FORBIDDEN, gated=gated, mahalanobis=d2)
    if d2 > motion_gate_chi2(d):
        return CostBreakdown(total=FORBIDDEN, gated=GATE_MOTION, mahalanobis=d2, motion=d2)

    b = CostBreakdown(total=0.0, motion=d2, mahalanobis=d2)
    velocity = track._mean[d:] if track.has_velocity else None
    # Move the last mask from where it was segmented to where the cell is predicted.
    shift = model.to_image(mean[:d] - p_last)
    _soft_terms(ctx, b, last, p_last, velocity, det, p_det, shift, dt)
    return _finish(ctx, b) if finish else b


def pair_cost(
    track: Track,
    detection: Detection,
    scale: Scale,
    cfg: TrackingConfig,
    legacy_cfg: TrackingConfig | None = None,
    *,
    geometry: ChannelGeometry | None = None,
) -> CostBreakdown:
    """Cost of explaining ``detection`` as the next observation of ``track``.

    Transition: ``pair_cost(track, det, axis, scale, cfg)`` (v1 order) still
    works; the axis only supplies lanes, through
    :meth:`ChannelGeometry.from_legacy_axis`.
    """
    if _is_legacy_axis(scale):
        axis, scale, cfg = scale, cfg, legacy_cfg  # type: ignore[assignment]
        if geometry is None:
            geometry = ChannelGeometry.from_legacy_axis(axis)
    assert isinstance(cfg, TrackingConfig)
    if track.model is None:
        raise ValueError(
            f"Track {track.id} has no motion model; start it with start_track() or observe()."
        )
    return _pair_cost(_Context(scale, cfg, track.model, geometry), track, detection)


def start_track(
    detection: Detection, scale: Scale, cfg: TrackingConfig, *, track_id: int = 1, lane: int | None = None
) -> Track:
    """A one-observation track under the motion model of ``scale`` and ``cfg``."""
    track = Track(id=track_id, model=MotionModel.from_config(scale, cfg, detection.ndim))
    track.source_ids = [track_id]
    track.observe(detection, cost=None, lane=lane)
    return track


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------


def build_assignment_matrix(costs: np.ndarray, unmatched_cost: float) -> np.ndarray:
    """Square block matrix that lets the optimiser choose 'no match'.

    ``costs`` is (n_tracks, n_detections).  The result is
    (n_tracks + n_detections, n_detections + n_tracks).
    """
    n_t, n_d = costs.shape
    size = n_t + n_d
    m = np.full((size, size), FORBIDDEN, dtype=float)
    if n_t and n_d:
        m[:n_t, :n_d] = costs
    for i in range(n_t):
        m[i, n_d + i] = unmatched_cost
    for j in range(n_d):
        m[n_t + j, j] = unmatched_cost
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
            # Safety net: a forbidden pairing must never survive.
            if matrix[r, c] >= FORBIDDEN:
                continue
            matches.append((int(r), int(c)))
            matched_t.add(int(r))
            matched_d.add(int(c))
    unmatched_t = [i for i in range(n_t) if i not in matched_t]
    unmatched_d = [j for j in range(n_d) if j not in matched_d]
    return matches, unmatched_t, unmatched_d


def link_margins(
    costs: np.ndarray, matches: Sequence[tuple[int, int]], unmatched_cost: float
) -> dict[tuple[int, int], float]:
    """The contract's ``link_margin`` of every accepted link, in chi-square units.

    For link (i, j): the cheapest other explanation for track i or detection
    j -- another detection for i, another track for j, or leaving both
    unmatched (``2 U``, the cap) -- minus the chosen cost.  Small means
    ambiguous.  It is negative when the global optimum gave track i a
    detection another track explains more cheaply (measured: -13.5 on
    052924_1 frame 9): the link is the best *joint* answer but locally
    contested, which is what an ambiguity flag should catch, so it is
    reported as is, not clamped.
    """
    out: dict[tuple[int, int], float] = {}
    cap = 2.0 * float(unmatched_cost)
    for i, j in matches:
        row = np.delete(costs[i, :], j)
        col = np.delete(costs[:, j], i)
        alternative = min(
            cap,
            float(row.min()) if row.size else math.inf,
            float(col.min()) if col.size else math.inf,
        )
        out[(i, j)] = alternative - float(costs[i, j])
    return out


def _components(allowed: np.ndarray) -> list[tuple[list[int], list[int]]]:
    """Connected groups of rows and columns joined by allowed pairs."""
    n_r, n_c = allowed.shape
    parent = list(range(n_r + n_c))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in zip(*np.nonzero(allowed)):
        ra, rb = find(int(i)), find(n_r + int(j))
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, tuple[list[int], list[int]]] = {}
    for i in range(n_r):
        groups.setdefault(find(i), ([], []))[0].append(i)
    for j in range(n_c):
        groups.setdefault(find(n_r + j), ([], []))[1].append(j)
    return list(groups.values())


def _assignment_total(costs: np.ndarray, unmatched_cost: float) -> float:
    matrix = build_assignment_matrix(costs, unmatched_cost)
    rows, cols = linear_sum_assignment(matrix)
    return float(matrix[rows, cols].sum())


def resolve_link_margins(
    costs: np.ndarray, matches: Sequence[tuple[int, int]], unmatched_cost: float
) -> dict[tuple[int, int], float]:
    """``link_margin_global``: how much the optimum worsens if each link is forbidden.

    The assignment is solved again without the link and the increase in
    total cost is the margin: never negative, at most ``2 U - cost``, and 0
    for a tie.  Only the link's connected group is re-solved -- tracks and
    detections joined by no allowed pair interact only through their own
    unmatched costs, so the block problem separates exactly by group -- which
    keeps the cost per link independent of how many cells are in the field.
    """
    out: dict[tuple[int, int], float] = {}
    if not matches:
        return out
    group_of_row: dict[int, tuple[list[int], list[int]]] = {}
    for rows, cols in _components(costs < FORBIDDEN):
        for r in rows:
            group_of_row[r] = (rows, cols)
    base_cache: dict[int, float] = {}
    for i, j in matches:
        rows, cols = group_of_row[i]
        sub = costs[np.ix_(rows, cols)]
        key = id(rows)
        if key not in base_cache:
            base_cache[key] = _assignment_total(sub, unmatched_cost)
        trial = sub.copy()
        trial[rows.index(i), cols.index(j)] = FORBIDDEN
        out[(i, j)] = max(0.0, _assignment_total(trial, unmatched_cost) - base_cache[key])
    return out


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


@dataclass
class FrameEvent:
    """What the tracker decided in one frame, for QC and ``tracking_events.csv``."""

    frame: int
    n_detections: int
    n_candidates: int
    #: Links that end in this frame: stage-1 matches plus stage-2 closures
    #: (``gap_closed``), so ``n_matched + n_new == n_detections``.
    n_matched: int
    n_new: int
    n_dormant: int
    n_terminated: int
    merge_suspected: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    split_suspected: list[int] = field(default_factory=list)
    #: Final ids of tracks whose gap ending at this frame was closed in stage 2.
    gap_closed: list[int] = field(default_factory=list)

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
            "split_suspected_tracks": ";".join(str(t) for t in self.split_suspected),
            "gap_closed_tracks": ";".join(str(t) for t in self.gap_closed),
            "notes": "; ".join(self.notes),
        }


# --------------------------------------------------------------------------
# Tracker
# --------------------------------------------------------------------------


def _ndim_of(detections: Sequence[Detection]) -> int:
    dims = {d.ndim for d in detections}
    if len(dims) > 1:
        raise ValueError("Cannot track 2-D and 3-D detections together.")
    return dims.pop() if dims else 2


class KalmanTracker:
    """Frame-by-frame Kalman/LAP association, then global gap closing.

    Conforms to ``interfaces.Tracker``: ``track(detections_by_frame,
    n_frames) -> (tracks, events)``.  Every call starts from nothing, so one
    instance can track any number of movies in turn.

    ``body_shaped_noise_in_lanes`` (default False, the contract) shapes the
    process noise and fresh-track prior by each cell's body, and only when
    the lane gate applies -- the confined case its measurement came from.
    """

    def __init__(
        self,
        scale: Scale,
        config: TrackingConfig,
        legacy_config: TrackingConfig | None = None,
        *,
        geometry: ChannelGeometry | None = None,
        merge_area_factor: float = MERGE_AREA_FACTOR,
        body_shaped_noise_in_lanes: bool = False,
    ) -> None:
        if _is_legacy_axis(scale):  # v1 order: (axis, scale, config)
            axis, scale, config = scale, config, legacy_config  # type: ignore[assignment]
            if geometry is None:
                geometry = ChannelGeometry.from_legacy_axis(axis)
        self.scale: Scale = scale  # type: ignore[assignment]
        self.cfg: TrackingConfig = config  # type: ignore[assignment]
        self.geometry = geometry
        self.merge_area_factor = merge_area_factor
        self.body_shaped_noise_in_lanes = bool(body_shaped_noise_in_lanes)
        self._reset()

    def _reset(self) -> None:
        """Forget everything from a previous call (tracks, events, ids, notes)."""
        self.tracks: list[Track] = []
        self.events: list[FrameEvent] = []
        self.id_map: dict[int, int] = {}
        self.notes: list[str] = []
        self._next_id = 1
        self._ctx: _Context | None = None

    # -- helpers -----------------------------------------------------------
    def _new_track(self, detection: Detection, lane: int, R: np.ndarray) -> Track:
        assert self._ctx is not None
        track = Track(id=self._next_id, model=self._ctx.model)
        track.source_ids = [track.id]
        self._next_id += 1
        track.observe(detection, cost=None, lane=lane, R=R)
        self.tracks.append(track)
        return track

    def _candidates(self, frame: int) -> tuple[list[Track], int]:
        """Tracks that may legally be matched in this frame, and how many just expired."""
        limit = self.cfg.max_delta_frames()
        out, expired = [], 0
        for tr in self.tracks:
            if tr.state is TrackState.TERMINATED:
                continue
            gap = tr.gap_to(frame)
            if 1 <= gap <= limit:
                out.append(tr)
            elif gap > limit:
                tr.state = TrackState.TERMINATED
                expired += 1
        return out, expired

    # -- main loop ---------------------------------------------------------
    def track(
        self, detections_by_frame: Mapping[int, Sequence[Detection]], n_frames: int
    ) -> tuple[TrackList, list[FrameEvent]]:
        tracks = self.run(detections_by_frame, n_frames)
        return tracks, self.events

    def run(
        self, detections_by_frame: Mapping[int, Sequence[Detection]], n_frames: int
    ) -> TrackList:
        self._reset()
        all_dets = [d for dets in detections_by_frame.values() for d in dets]
        ndim = _ndim_of(all_dets)
        shaped = self.body_shaped_noise_in_lanes and lane_gate_applies(self.geometry, self.cfg)
        model = MotionModel.from_config(self.scale, self.cfg, ndim, body_shaped_noise=shaped)
        self._ctx = _Context(self.scale, self.cfg, model, self.geometry)
        ignored = sum(1 for d in all_dets if not 0 <= int(d.frame) < n_frames)
        if shaped:
            self.notes.append(
                "Process noise and the fresh-track prior were shaped by each cell's body "
                "(opt-in, confined lanes only)."
            )

        for frame in range(n_frames):
            self._step(frame, list(detections_by_frame.get(frame, [])))

        for tr in self.tracks:
            tr.state = TrackState.TERMINATED

        if self.cfg.global_gap_closing:
            self._close_gaps()
        final = self._renumber()
        self._flag_borders(final, n_frames)
        if model.z_assumed:
            for tr in final:
                tr.flags.add(FLAG_Z_UNCALIBRATED)
            self.notes.append(
                "This stack has no calibrated Z step, so one slice was tracked as one "
                "pixel (anisotropy 1.0). Distances in Z are an assumption."
            )
        if ignored:
            self.notes.append(
                f"{ignored} detection(s) at frames outside 0..{n_frames - 1} were ignored."
            )
        self._render_notes()
        self.tracks = final
        return TrackList(final, self.id_map, self.notes)

    def _step(self, frame: int, dets: list[Detection]) -> None:
        ctx = self._ctx
        assert ctx is not None
        candidates, expired = self._candidates(frame)
        lanes = [ctx.lane_of(d) for d in dets]
        Rs = [ctx.model.measurement_cov(d) for d in dets]
        event = FrameEvent(frame, len(dets), len(candidates), 0, 0, 0, expired)

        if not dets:
            for tr in candidates:
                tr.miss(frame, self.cfg.max_gap)
                event.n_dormant += tr.state is TrackState.DORMANT
                event.n_terminated += tr.state is TrackState.TERMINATED
            self.events.append(event)
            return

        predictions = [tr.predict_state(frame) for tr in candidates]
        reach = self._within_speed(candidates, dets, frame)
        costs = np.full((len(candidates), len(dets)), FORBIDDEN, dtype=float)
        breakdowns: dict[tuple[int, int], CostBreakdown] = {}
        for i, j in zip(*np.nonzero(reach)):
            i, j = int(i), int(j)
            breakdowns[(i, j)] = _pair_cost(
                ctx, candidates[i], dets[j], R=Rs[j], det_lane=lanes[j],
                prediction=predictions[i], finish=False, diagnose=False,
            )
        _withhold_partial_overlap(self.cfg, breakdowns)
        for (i, j), b in breakdowns.items():
            if b.gated is None:
                _finish(ctx, b)
            costs[i, j] = b.total

        matches, unmatched_t, unmatched_d = solve_assignment(costs, self.cfg.unmatched_chi2)
        local = link_margins(costs, matches, self.cfg.unmatched_chi2)
        joint = resolve_link_margins(costs, matches, self.cfg.unmatched_chi2)
        previous = {i: candidates[i].last for i, _ in matches}
        for i, j in matches:
            candidates[i].observe(
                dets[j], cost=float(costs[i, j]), link_margin=local[(i, j)],
                link_margin_global=joint[(i, j)], breakdown=breakdowns[(i, j)],
                lane=lanes[j], R=Rs[j], prediction=predictions[i],
            )
        for i in unmatched_t:
            tr = candidates[i]
            tr.miss(frame, self.cfg.max_gap)
            event.n_dormant += tr.state is TrackState.DORMANT
            event.n_terminated += tr.state is TrackState.TERMINATED
        newborn = {j: self._new_track(dets[j], lanes[j], Rs[j]) for j in unmatched_d}

        event.n_matched = len(matches)
        event.n_new = len(unmatched_d)
        event.merge_suspected = self._flag_merges(frame, candidates, unmatched_t, matches, dets)
        event.split_suspected = self._flag_splits(candidates, matches, previous, newborn, dets)
        self.events.append(event)

    def _within_speed(self, candidates: Sequence[Track], dets: Sequence[Detection], frame: int) -> np.ndarray:
        """(n_tracks, n_dets) mask of pairs the speed gate does not refuse.

        The same test as ``_hard_gates``, vectorised, so that the Kalman
        algebra runs only on pairs that could possibly be linked: in a dense
        open field most pairs are hundreds of pixels apart.
        """
        ctx = self._ctx
        assert ctx is not None
        if not candidates or not dets:
            return np.zeros((len(candidates), len(dets)), dtype=bool)
        model, scale = ctx.model, ctx.scale
        p_last = np.array([
            model.position(tr.last.detection or _probe_detection(tr.last)) for tr in candidates
        ])
        p_det = np.array([model.position(d) for d in dets])
        dt = np.array([max(tr.gap_to(frame), 1) for tr in candidates], dtype=float)
        step_px = np.linalg.norm(p_det[None, :, :] - p_last[:, None, :], axis=2)
        speed = scale.px_to_um(step_px) / np.maximum(scale.frames_to_min(dt), 1e-9)[:, None]
        return speed <= self.cfg.max_speed_um_per_min

    # -- merged and split masks --------------------------------------------
    @staticmethod
    def _inside_bbox(det: Detection, point: np.ndarray) -> bool:
        if len(det.bbox) == 6:
            min_z, min_r, min_c, max_z, max_r, max_c = det.bbox
            if len(point) < 3 or not (min_z <= point[2] <= max_z):
                return False
        else:
            min_r, min_c, max_r, max_c = det.bbox
        return bool(min_c <= point[0] <= max_c and min_r <= point[1] <= max_r)

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
                if self._inside_bbox(d, pred) and d.size >= self.merge_area_factor * tr.last_size:
                    tr.flags.add(FLAG_MERGE)
                    flagged.append(tr.id)
                    break
        return flagged

    def _flag_splits(
        self,
        candidates: list[Track],
        matches: Sequence[tuple[int, int]],
        previous: Mapping[int, Observation],
        newborn: Mapping[int, Track],
        dets: Sequence[Detection],
    ) -> list[int]:
        """Note when a new track appears out of a mask its neighbour just shrank from.

        The mirror image of a merge: a matched track's previous mask was much
        larger than what it kept, and the new detection sits inside that
        previous mask.  Both tracks are flagged; neither is changed.
        """
        if not matches or not newborn:
            return []
        flagged: list[int] = []
        for i, j_kept in matches:
            prev = previous[i].detection
            if prev is None:
                continue
            kept = dets[j_kept]
            if prev.size < self.merge_area_factor * kept.size:
                continue
            for j_new, child in newborn.items():
                if self._inside_bbox(prev, dets[j_new].position):
                    candidates[i].flags.add(FLAG_SPLIT)
                    child.flags.add(FLAG_SPLIT)
                    for tid in (candidates[i].id, child.id):
                        if tid not in flagged:
                            flagged.append(tid)
        return flagged

    # -- stage 2 -----------------------------------------------------------
    def _close_gaps(self) -> None:
        """Match every track end against every later start in one assignment."""
        ctx = self._ctx
        assert ctx is not None
        limit = self.cfg.max_delta_frames()
        tracks = [t for t in self.tracks if t.observations]
        ends = [t for t in tracks if any(1 <= s.first_frame - t.last_frame <= limit for s in tracks)]
        starts = [s for s in tracks if any(1 <= s.first_frame - t.last_frame <= limit for t in tracks)]
        if not ends or not starts:
            return
        backward = {s.id: _backward_state(ctx.model, s) for s in starts}
        costs = np.full((len(ends), len(starts)), FORBIDDEN, dtype=float)
        breakdowns: dict[tuple[int, int], CostBreakdown] = {}
        for i, a in enumerate(ends):
            for j, b in enumerate(starts):
                if a is b or not 1 <= b.first_frame - a.last_frame <= limit:
                    continue
                breakdowns[(i, j)] = closing_cost(ctx, a, b, backward[b.id], finish=False)
        _withhold_partial_overlap(self.cfg, breakdowns)
        for (i, j), bd in breakdowns.items():
            if bd.gated is None:
                _finish(ctx, bd)
            costs[i, j] = bd.total
        matches, _, _ = solve_assignment(costs, self.cfg.unmatched_chi2)
        if not matches:
            return
        local = link_margins(costs, matches, self.cfg.unmatched_chi2)
        joint = resolve_link_margins(costs, matches, self.cfg.unmatched_chi2)

        next_of: dict[int, tuple[Track, float, float, float, CostBreakdown]] = {}
        has_prev: set[int] = set()
        for i, j in matches:
            a, b = ends[i], starts[j]
            next_of[a.id] = (
                b, float(costs[i, j]), local[(i, j)], joint[(i, j)], breakdowns[(i, j)]
            )
            has_prev.add(b.id)

        merged: list[Track] = []
        for head in tracks:
            if head.id in has_prev:
                continue
            current = head
            while current.id in next_of:
                nxt, cost, margin, margin_global, bd = next_of[current.id]
                first = nxt.observations[0]
                first.cost = cost
                first.link_margin = margin
                first.link_margin_global = margin_global
                first.breakdown = bd
                first.gap_frames = first.frame - head.last_frame
                frame_event = self._event_at(first.frame)
                if frame_event is not None:
                    frame_event.gap_closed.append(head.id)
                    # Stage 1 counted this start as a new track; it is a
                    # continuation now, so matched + new == detections holds.
                    frame_event.n_new -= 1
                    frame_event.n_matched += 1
                head.observations.extend(nxt.observations)
                head.flags |= nxt.flags
                head.flags.add(FLAG_GAP_CLOSED)
                head.source_ids.extend(nxt.source_ids)
                head._mean, head._cov = nxt._mean, nxt._cov
                if head.channel < 0:
                    head.channel = nxt.channel
                current = nxt
            merged.append(head)
        self.tracks = merged

    def _event_at(self, frame: int) -> FrameEvent | None:
        if 0 <= frame < len(self.events) and self.events[frame].frame == frame:
            return self.events[frame]
        return next((e for e in self.events if e.frame == frame), None)

    # -- finishing ---------------------------------------------------------
    def _renumber(self) -> list[Track]:
        """Final ids by first appearance; every stage-1 id maps to one of them."""
        ordered = sorted(self.tracks, key=lambda t: (t.first_frame, t.source_ids[0] if t.source_ids else t.id))
        head_to_final: dict[int, int] = {}
        for new_id, tr in enumerate(ordered, start=1):
            head_to_final[tr.id] = new_id
        stage1_to_head: dict[int, int] = {}
        for tr in ordered:
            for sid in tr.source_ids or [tr.id]:
                stage1_to_head[sid] = tr.id
        self.id_map = {sid: head_to_final[h] for sid, h in stage1_to_head.items()}
        for tr in ordered:
            tr.id = head_to_final[tr.id]

        def remap(ids: Sequence[int]) -> list[int]:
            out: list[int] = []
            for tid in ids:
                new = self.id_map.get(tid, tid)
                if new not in out:
                    out.append(new)
            return out

        for ev in self.events:
            ev.merge_suspected = remap(ev.merge_suspected)
            ev.split_suspected = remap(ev.split_suspected)
            ev.gap_closed = remap(ev.gap_closed)
        return ordered

    @staticmethod
    def _flag_borders(tracks: Sequence[Track], n_frames: int) -> None:
        """A track that starts (ends) mid-movie on a mask cut by the image edge.

        At frame 0 or the last frame touching the border is just where the
        cell was when filming began or stopped, so only mid-movie starts and
        ends are flagged.
        """
        for tr in tracks:
            first, last = tr.observations[0], tr.observations[-1]
            if first.frame > 0 and first.detection is not None and first.detection.touches_border:
                tr.flags.add(FLAG_ENTERS_BORDER)
            if (
                last.frame < n_frames - 1
                and last.detection is not None
                and last.detection.touches_border
            ):
                tr.flags.add(FLAG_EXITS_BORDER)

    def _render_notes(self) -> None:
        by_id = {t.id: t for t in self.tracks}
        for ev in self.events:
            for tid in ev.merge_suspected:
                ev.notes.append(
                    f"track {tid} vanished inside a larger mask matched to another track "
                    "(merge suspected; no position invented)"
                )
            if ev.split_suspected:
                ev.notes.append(
                    "split suspected between tracks "
                    + " and ".join(str(t) for t in ev.split_suspected)
                )
            for tid in ev.gap_closed:
                tr = by_id.get(tid)
                obs = next((o for o in tr.observations if o.frame == ev.frame), None) if tr else None
                gap = obs.gap_frames if obs is not None else None
                if gap == 1:
                    # No frame was missing: stage 1 refused the link, stage 2 restored it.
                    ev.notes.append(f"track {tid} rejoined by global gap closing")
                else:
                    ev.notes.append(
                        f"track {tid} continued after {gap - 1 if gap else '?'} missing frame(s) "
                        "by global gap closing"
                    )
            if ev.n_terminated:
                ev.notes.append(f"{ev.n_terminated} track(s) ended")
        if self.notes and self.events:
            self.events[0].notes.extend(self.notes)


#: v1 name; the v1 argument order ``(axis, scale, config)`` is still accepted.
ConfinementTracker = KalmanTracker


# --------------------------------------------------------------------------
# Stage 2 cost
# --------------------------------------------------------------------------


def _backward_state(model: MotionModel, track: Track) -> tuple[np.ndarray, np.ndarray]:
    """Run ``track`` backwards in time from its last observation to its first.

    Returns the filtered state at its first observation in *reversed* time
    (velocity points back along the trajectory), ready to be predicted to an
    earlier frame with a positive ``dt``.
    """
    obs = list(reversed(track.observations))
    det0 = obs[0].detection or _probe_detection(obs[0])
    mean, cov = model.initial(det0)
    for prev, cur in zip(obs[:-1], obs[1:]):
        dt = prev.frame - cur.frame
        mean, cov = model.predict(mean, cov, float(dt), prev.detection)
        det = cur.detection or _probe_detection(cur)
        mean, cov = model.update(mean, cov, model.position(det), model.measurement_cov(det))
    return mean, cov


#: The evidence labels of ``CostBreakdown.motion_evidence``.
EVIDENCE_BOTH = "forward+backward"
EVIDENCE_FORWARD = "forward"
EVIDENCE_BACKWARD = "backward"


def closing_cost(
    ctx: _Context,
    end: Track,
    start: Track,
    backward: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    finish: bool = True,
) -> CostBreakdown:
    """Stage-2 cost of continuing ``end`` with ``start``, with evidence from both sides.

    A side is evidence only when its track has at least two observations: a
    one-observation track predicts with the fresh-track prior alone (71 px
    per frame of velocity spread at the KK2 calibration), so its ``d^2`` is
    small whatever happened.  Averaged in, it halved the informative side:
    measured, a vertical cell moving 20 px/frame was joined to a fragment
    18 px to its side (forward 23.3, backward 3.96, mean 13.7 -- under the
    13.8 gate the forward side alone fails).  With both sides informative the
    gate is on ``d2_f + d2_b`` at the ``2 * ndim`` quantile, and the motion
    term is half the sum.
    """
    model = ctx.model
    d = model.ndim
    last = end.last
    first = start.observations[0]
    det = first.detection or _probe_detection(first)
    prev_det = last.detection or _probe_detection(last)
    dt = first.frame - last.frame
    p_last = model.position(prev_det)
    p_det = model.position(det)
    gated = _hard_gates(ctx, end.channel, last, p_last, det, p_det, first.channel, dt)
    if gated == GATE_GAP:
        return CostBreakdown(total=FORBIDDEN, gated=GATE_GAP)

    mean_f, cov_f = end.predict_state(first.frame)
    d2_f = _mahalanobis(p_det - mean_f[:d], cov_f[:d, :d] + model.measurement_cov(det))
    if backward is None:
        backward = _backward_state(model, start)
    mean_b, cov_b = model.predict(backward[0], backward[1], float(dt), det)
    d2_b = _mahalanobis(p_last - mean_b[:d], cov_b[:d, :d] + model.measurement_cov(prev_det))

    forward_informs, backward_informs = end.n_obs >= 2, start.n_obs >= 2
    if forward_informs and backward_informs:
        evidence, motion = EVIDENCE_BOTH, 0.5 * (d2_f + d2_b)
        refused = d2_f + d2_b > motion_gate_chi2(2 * d)
    elif backward_informs:
        evidence, motion = EVIDENCE_BACKWARD, d2_b
        refused = d2_b > motion_gate_chi2(d)
    else:
        # Forward only -- or neither side informative, where this is exactly
        # the judgement stage 1 made.
        evidence, motion = EVIDENCE_FORWARD, d2_f
        refused = d2_f > motion_gate_chi2(d)
    sides = dict(motion_forward=d2_f, motion_backward=d2_b, motion_evidence=evidence)
    if gated is not None:
        return CostBreakdown(total=FORBIDDEN, gated=gated, mahalanobis=motion, **sides)
    if refused:
        return CostBreakdown(
            total=FORBIDDEN, gated=GATE_MOTION, motion=motion, mahalanobis=motion, **sides
        )
    b = CostBreakdown(total=0.0, motion=motion, mahalanobis=motion, **sides)
    end_mean, _ = end.kalman_state
    velocity = end_mean[d:] if end.has_velocity else None
    shift = model.to_image(mean_f[:d] - p_last)
    _soft_terms(ctx, b, last, p_last, velocity, det, p_det, shift, dt)
    return _finish(ctx, b) if finish else b


def _probe_detection(obs: Observation) -> Detection:
    """A stand-in Detection for an observation that carries none (hand-built tracks)."""
    major = obs.major_axis_px or obs.minor_axis_px
    return Detection(
        frame=obs.frame, label=obs.det_label, x=obs.x, y=obs.y, z=obs.z,
        area_px=obs.area_px, bbox=(0, 0, 1, 1), extent_px=1,
        eccentricity=obs.eccentricity, orientation_rad=obs.orientation_rad,
        major_axis_px=major, minor_axis_px=obs.minor_axis_px,
        solidity=float("nan"), touches_border=False, channel=obs.channel,
    )


# --------------------------------------------------------------------------
# Unlinked-start audit
# --------------------------------------------------------------------------


@dataclass
class UnlinkedStart:
    """A track that began mid-stack, and the best case for it being a continuation.

    When a cell disappears and something appears later, the tracker's refusal
    to join them is a judgement. Recording what that judgement was based on
    turns it from an opaque verdict into evidence a reviewer can weigh: how far
    apart (``dx_px``, ``dy_px``, ``dz_px`` -- Z in isotropic pixels, i.e.
    slices times the anisotropy), how surprising to the motion model
    (``mahalanobis``, the stage-2 motion term), what the join would have cost,
    and which rule refused it.
    """

    track_id: int
    frame: int
    candidate_track_id: int | None = None
    candidate_last_frame: int | None = None
    gap_frames: int | None = None
    distance_px: float | None = None
    dx_px: float | None = None
    dy_px: float | None = None
    dz_px: float | None = None
    mahalanobis: float | None = None
    speed_um_per_min: float | None = None
    cost_chi2: float | None = None
    refused_because: str | None = None

    # v1 columns, kept only so the v1 pipeline writer still runs until the
    # integration package removes ``along_channel_px``/``across_channel_px``.
    @property
    def along_px(self) -> None:
        return None

    @property
    def across_px(self) -> None:
        return None

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
            GATE_LANE: "it is in a different channel lane",
            GATE_MOTION: "it is too far from where either track's motion predicts",
            GATE_COST: "the overall fit was too poor",
            None: "joining them was plausible, but a cheaper explanation won the assignment",
        }.get(self.refused_because, "the fit was not good enough")
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
    scale: Scale,
    cfg: TrackingConfig,
    legacy_cfg: TrackingConfig | None = None,
    *,
    geometry: ChannelGeometry | None = None,
) -> list[UnlinkedStart]:
    """For every track that began mid-stack, why it was not joined to an earlier one.

    The gap limit is deliberately relaxed while costing, so that a pairing
    refused *only* because it was too far apart in time is reported as exactly
    that -- a policy -- rather than silently vanishing.  The cost is the
    stage-2 closing cost, the last judgement the tracker made.

    Transition: ``explain_unlinked_starts(tracks, axis, scale, cfg)`` (v1
    order) still works; the axis only supplies lanes.
    """
    if _is_legacy_axis(scale):
        axis, scale, cfg = scale, cfg, legacy_cfg  # type: ignore[assignment]
        if geometry is None:
            geometry = ChannelGeometry.from_legacy_axis(axis)
    assert isinstance(cfg, TrackingConfig)
    live = [t for t in tracks if t.observations]
    if not live:
        return []
    ndim = 3 if live[0].observations[0].z is not None else 2
    model = live[0].model or MotionModel.from_config(scale, cfg, ndim)
    ctx_strict = _Context(scale, cfg, model, geometry)

    out: list[UnlinkedStart] = []
    ordered = sorted(live, key=lambda t: t.first_frame)
    backward_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for track in ordered:
        start = track.observations[0]
        if start.frame == 0:
            continue
        best: UnlinkedStart | None = None
        best_key: tuple[float, float] | None = None
        for other in live:
            if other is track or other.last_frame >= start.frame:
                continue
            if other.model is None or other._mean is None:
                continue
            gap = start.frame - other.last_frame
            relaxed = _Context(
                scale, replace(cfg, max_gap=max(cfg.max_gap, gap)), model, geometry
            )
            if id(track) not in backward_cache:
                backward_cache[id(track)] = _backward_state(model, track)
            breakdown = closing_cost(relaxed, other, track, backward_cache[id(track)])
            step = model.position(start.detection or _probe_detection(start)) - model.position(
                other.last.detection or _probe_detection(other.last)
            )
            distance = float(np.linalg.norm(step))
            elapsed = scale.frames_to_min(gap)
            candidate = UnlinkedStart(
                track_id=track.id,
                frame=start.frame,
                candidate_track_id=other.id,
                candidate_last_frame=other.last_frame,
                gap_frames=gap,
                distance_px=distance,
                dx_px=float(step[0]),
                dy_px=float(step[1]),
                dz_px=float(step[2]) if ndim == 3 else None,
                mahalanobis=breakdown.mahalanobis,
                speed_um_per_min=(
                    scale.px_to_um(distance) / elapsed
                    if scale.calibrated and elapsed > 0 else None
                ),
                cost_chi2=(None if breakdown.gated else breakdown.total),
                refused_because=breakdown.gated or (
                    GATE_GAP if gap > ctx_strict.cfg.max_delta_frames() else None
                ),
            )
            # Prefer the nearest in time, then the cheapest. A genuine cost of
            # 0.0 is the cheapest, not "forbidden" (v1 ranked it with ``or``).
            key = (
                float(gap),
                candidate.cost_chi2 if candidate.cost_chi2 is not None else math.inf,
            )
            if best_key is None or key < best_key:
                best, best_key = candidate, key
        out.append(best or UnlinkedStart(track_id=track.id, frame=start.frame))
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def track_detections(
    detections: Iterable[Detection],
    n_frames: int,
    scale: Scale,
    config: TrackingConfig,
    legacy_config: TrackingConfig | None = None,
    *,
    geometry: ChannelGeometry | None = None,
    body_shaped_noise_in_lanes: bool = False,
) -> tuple[TrackList, list[FrameEvent]]:
    """Group detections by frame and run the tracker.

    Transition: ``track_detections(dets, n, axis, scale, cfg)`` (v1 order)
    still works; the axis only supplies lanes, through
    :meth:`ChannelGeometry.from_legacy_axis`, so the v1 pipeline keeps refusing
    cross-channel links until it passes a real ``ChannelGeometry``.
    ``body_shaped_noise_in_lanes`` is the opt-in described on
    :class:`KalmanTracker`; the default is the contract's isotropic noise.
    """
    by_frame: dict[int, list[Detection]] = {}
    for d in detections:
        by_frame.setdefault(int(d.frame), []).append(d)
    tracker = KalmanTracker(
        scale, config, legacy_config, geometry=geometry,
        body_shaped_noise_in_lanes=body_shaped_noise_in_lanes,
    )
    return tracker.track(by_frame, n_frames)
