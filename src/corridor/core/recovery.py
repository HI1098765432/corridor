"""Finding cells the first pass missed.

The measured failure mode of this segmentation model is recall: held out it
finds 20-44 % of labelled cells, and even on its own training images it misses
12-20 %. Precision stays high (0.54-0.60 held out), so what it reports is
usually right -- it simply does not report enough.

That asymmetry is what makes recovery worth doing, and it dictates the shape of
it. A global sensitivity increase trades precision for recall everywhere, and
on the near-empty control stack it starts reporting a cell in every frame. What
is safe instead is to become more sensitive *only where a track already
predicts a cell*: a specific position, with a known expected size, shape and
orientation. Evidence that would be too weak to assert a cell on its own is
enough to confirm one that motion continuity already predicts.

Three tiers, each tried only when the one before it fails:

1.  **Re-segment a window.** Crop around the prediction and run the same model
    on it. This is not just a permissiveness change: Cellpose normalises per
    image, so a crop containing mostly one cell puts that cell at the top of
    the intensity range instead of buried under the brightest structure in the
    whole field.
2.  **Permissive re-segmentation.** The same crop at a much lower probability
    threshold. Only reachable inside a predicted window.
3.  **Intensity profile.** No network at all: these cells are bright elongated
    objects, so a local signal-to-noise test around the expected position finds
    them when the network will not.

There is no migration axis anywhere in this module (contract §5). Everything
that v1 measured against the channel direction is measured in **the cell's own
frame** instead: the major and minor axes of the observations that bracket the
missing frame. The crop is sized from that body (v1 sized it as if every
channel ran down the image, which cut a horizontal cell in half), the
intensity tier bounds the candidate's offset *across the body*, and lanes enter
only as the tracker's own lane gate does -- when that gate applies, a candidate
in a different real lane from the track's is refused, and lane -1 (outside
every lane) is never refused, on either side.

A recovered detection that is a second copy of a cell the first pass already
detected is dropped (see :func:`duplicate_of`). Measured before this rule
existed, recovery re-found primary cells -- 052924_2 lane 4 frames 11-14 at
20.7, 7.9, 1.2 and 0.6 px from the primary, 052924_1 frame 9 at 4.5 px -- and
re-tracking then ran two tracks along one cell. Applied to the 16 recoveries
of the frozen v1.3.0 baseline (primary masks from its masks.npz, recovered
bodies from its detections.csv; no recovered masks exist there, so the two
centroid tests only), the rule drops exactly those five and keeps the other
eleven, whose nearest primary is 84.5 px or more away.

Everything recovered is marked as recovered, carries the tier that found it, a
confidence, its own mask (``mask_crop``) and full morphology, and is written to
its own column in the output. A recovered position is a weaker claim than a
primary detection and must never be indistinguishable from one. Every attempt
carries the ``bracket`` it was interpolated from, so its first-pass track id
can be re-keyed to the final, re-tracked id (critique C4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy import ndimage

from .config import Scale, TrackingConfig
from .detections import SOURCE_PRIMARY, Detection, extract_detections
from .geometry import ChannelGeometry
from .tracking import (
    Observation,
    Track,
    _is_legacy_axis,
    _probe_detection,
    lane_gate_applies,
    shifted_iou,
)

SOURCE_WINDOW = "windowed"
SOURCE_PERMISSIVE = "permissive"
SOURCE_INTENSITY = "intensity"

#: Confidence attached to each tier. A primary detection is the network
#: asserting a cell unprompted; the later tiers are confirmations of something
#: motion already predicted, and are weaker in that order.
TIER_CONFIDENCE = {
    SOURCE_PRIMARY: 1.00,
    SOURCE_WINDOW: 0.80,
    SOURCE_PERMISSIVE: 0.60,
    SOURCE_INTENSITY: 0.40,
}

#: ``(frame_before, det_label_before, frame_after, det_label_after)`` of the
#: first-pass observations a missing frame was estimated from. ``after`` is
#: None for a trailing frame, which has no far side.
Bracket = tuple[int, int, "int | None", "int | None"]


@dataclass
class RecoveryConfig:
    """How hard to look for a cell a track says should be there.

    **On by default for interior gaps, off for trailing frames.** That split is
    not a compromise; it is the measurement.

    Scored against real ground truth -- deleting a primary detection from a
    frame where a cell is known present, and checking whether recovery puts it
    back, while also counting anything it reports in frames where a cell is
    known absent (scripts/experiment_recovery.py, 1.x):

        where it looks              recovered   median error   false positives
        interior gaps only            10 / 13        3.5 px           0
        trailing frames only           1 / 13        2.9 px           2
        both                          11 / 13        2.9 px           2

    An **interior gap** is a frame between two observations: the cell was there
    before and after, so a hole is a detection failure and nothing else. Those
    recover at 0.77 with a median error of about a quarter of a cell's width,
    and on the available ground truth they produced no false positives at all.

    A **trailing frame**, past a track's last observation, is a different claim
    entirely -- the cell may simply have left, died, or gone out of focus, and
    there is no evidence on the far side to say otherwise. Every false position
    in the measurement came from there, and it bought one true one. So it is
    off, and turning it on is a deliberate act.

    That table was measured with 1.x's window and channel test. 2.0 sizes the
    window from the cell's own body and drops duplicates of primary cells, so
    ``scripts/experiment_recovery.py`` must be re-run before the table is
    quoted for 2.0; the defaults are kept as measured until then.
    """

    #: Recovery runs by default, but only for the gaps it was measured to fill
    #: without inventing anything. Every recovered position is still marked with
    #: its tier and confidence, written to detections.csv, and flagged by
    #: quality control when a trajectory leans on it.
    enabled: bool = True

    #: Probe the frames *between* two observations. These are the safe ones:
    #: the cell is known to have been there before and after, so a hole is a
    #: detection failure and nothing else.
    interior: bool = True
    #: Probe the frames *after* a track's last observation. These are not the
    #: same question at all -- a track that ends may be a cell that left, died,
    #: or went out of focus, and there is no evidence on the far side to say
    #: otherwise. Measured on the supplied data this is where every false
    #: positive came from, so it is off by default. See docs/ACCURACY.md.
    trailing: bool = False

    #: Try re-segmenting a crop around the prediction.
    window: bool = True
    #: Then the same crop at a much lower probability threshold.
    permissive: bool = True
    #: Then a classical intensity test, with no network involved.
    intensity: bool = True

    #: Crop size relative to the cell's own body: ``window_scale`` half-lengths
    #: either side along its major axis and ``2 * window_scale`` widths either
    #: side across it -- 1.x's two ratios, now taken in the cell's own frame
    #: instead of "length along y, width along x". Large enough to give
    #: Cellpose context, small enough that normalisation actually helps.
    window_scale: float = 2.2
    #: Minimum crop half-size in pixels, so a small cell still gets context.
    window_min_px: int = 40

    #: Probability threshold for tier 2. Far below anything safe globally, and
    #: only ever applied inside a predicted window.
    permissive_cellprob: float = -4.0
    permissive_flow: float = 0.8

    #: Tier 3: how far above the local background, in robust standard
    #: deviations, a candidate must sit to count as a cell.
    intensity_snr: float = 6.0
    #: And how much of the expected cell area it must cover.
    intensity_min_area_fraction: float = 0.35
    #: Candidates whose centroid is within this many pixels of the image edge
    #: are refused. The brightest structures in these fields are the frame
    #: border and the reservoir band, and a threshold alone does not tell them
    #: apart from a cell -- their position does.
    intensity_edge_margin_px: int = 12
    #: A confined cell is long and thin. A round blob at the right brightness
    #: is not the thing being looked for, so a candidate must be at least this
    #: elongated.
    intensity_min_eccentricity: float = 0.85
    #: And must sit within this many of the cell's own widths of the expected
    #: position, measured *across* the cell's body (its minor axis). Replaces
    #: 1.x's "within one channel half-width of the channel centre line", which
    #: needed an axis. The across-body centroid noise measured on the v1.3.0
    #: baseline is 0.75 um against a median width of 4.7 um (TrackingConfig),
    #: so a true cell sits within about a sixth of a width of where its
    #: bracketing observations put it; one full width is roughly six of those
    #: sigmas, generous for the cell and still far narrower than a lane.
    intensity_max_across_widths: float = 1.0

    #: A recovered detection must still be a plausible continuation: it is
    #: rejected if it is further than this many multiples of the track's own
    #: length from the prediction.
    max_offset_lengths: float = 1.5

    #: Duplicate rule (see :func:`duplicate_of`): a recovered mask overlapping
    #: a primary mask of the same frame by more than this IoU is that primary
    #: cell counted twice. Primary masks come from one label image and never
    #: overlap each other at all, so a fifth of a union is a claim on pixels
    #: already owned; a smaller value would start catching the boundary pixel
    #: a re-segmented cell can share with a touching neighbour.
    duplicate_iou: float = 0.2
    #: ...or whose centroid lies within this many of the primary cell's
    #: lengths of the primary centroid along its own long axis (and inside its
    #: width across it). Half a length is the cell's own body; a different
    #: cell following in the same lane sits about a whole length away.
    duplicate_along_lengths: float = 0.5


@dataclass
class RecoveryAttempt:
    """One attempt to find a missing cell, successful or not.

    ``track_id`` is the **first-pass** id when :func:`recover` returns; the
    pipeline re-tracks and then calls :func:`assign_final_track_ids`, which
    sets it to the final id and keeps the first-pass one in
    ``first_pass_track_id``. ``bracket`` is what makes that possible without
    trusting the order of ids (critique C4).
    """

    track_id: int | None
    frame: int
    predicted_x: float
    predicted_y: float
    found: bool
    source: str = ""
    confidence: float = 0.0
    offset_px: float | None = None
    detail: str = ""
    bracket: Bracket | None = None
    first_pass_track_id: int | None = None
    #: Label (same frame) of the detection a dropped candidate duplicated, as
    #: it was when the candidate was dropped; None when nothing was dropped
    #: as a duplicate.  ``to_row`` prefers ``duplicate_of``'s label as it is
    #: when the row is written.
    duplicate_of_label: int | None = None
    #: The recovered Detection object itself (None when nothing was found), so
    #: the final track holding it can be found by identity after relabelling.
    detection: Detection | None = field(default=None, repr=False, compare=False)
    #: The detection a dropped candidate duplicated, held as the object: a
    #: label copied at drop time goes stale if anything relabels that
    #: detection afterwards (the review found it naming a different cell).
    duplicate_of: Detection | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.first_pass_track_id is None:
            self.first_pass_track_id = self.track_id

    @property
    def is_duplicate(self) -> bool:
        return self.duplicate_of is not None or self.duplicate_of_label is not None

    def to_row(self) -> dict[str, Any]:
        before_f, before_l, after_f, after_l = self.bracket or (None, None, None, None)
        return {
            "track_id": self.track_id,
            "first_pass_track_id": self.first_pass_track_id,
            "frame": self.frame,
            "predicted_x": self.predicted_x,
            "predicted_y": self.predicted_y,
            "recovered": self.found,
            "found_by": self.source,
            "confidence": self.confidence,
            "offset_from_prediction_px": self.offset_px,
            "detail": self.detail,
            "bracket_frame_before": before_f,
            "bracket_label_before": before_l,
            "bracket_frame_after": after_f,
            "bracket_label_after": after_l,
            "duplicate_of_label": (
                int(self.duplicate_of.label) if self.duplicate_of is not None
                else self.duplicate_of_label
            ),
        }


@dataclass
class RecoveryResult:
    detections: list[Detection] = field(default_factory=list)
    attempts: list[RecoveryAttempt] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n_recovered(self) -> int:
        return sum(1 for a in self.attempts if a.found)

    @property
    def n_attempted(self) -> int:
        return len(self.attempts)

    @property
    def n_duplicates_dropped(self) -> int:
        return sum(1 for a in self.attempts if a.is_duplicate)


# --------------------------------------------------------------------------
# What the track expects at a missing frame
# --------------------------------------------------------------------------


@dataclass
class Expectation:
    """Where, how big and which way round a track says its cell is at one frame.

    ``u`` is the unit vector along the body (x, y), ``n`` across it. Lengths
    are full axis lengths in pixels. ``lane`` is the bracketing observations'
    lane (-1 none).
    """

    frame: int
    position: np.ndarray
    major_px: float
    minor_px: float
    u: np.ndarray
    area_px: float
    bracket: Bracket
    lane: int = -1

    @property
    def n(self) -> np.ndarray:
        return np.array([-self.u[1], self.u[0]], dtype=float)

    def body_offset(self, x: float, y: float) -> tuple[float, float]:
        """``(along, across)`` of a point from the expected centroid, in px."""
        d = np.array([x, y], dtype=float) - self.position
        return float(abs(d @ self.u)), float(abs(d @ self.n))


def _body(obs: Observation) -> tuple[float, float, np.ndarray, float]:
    """``(major, minor, u, area)`` of one observation, from its detection when it has one."""
    det = obs.detection
    major = float(det.major_axis_px if det is not None else obs.major_axis_px)
    minor = float(det.minor_axis_px if det is not None else obs.minor_axis_px)
    theta = float(det.orientation_rad if det is not None else obs.orientation_rad)
    area = float(det.area_px if det is not None else obs.area_px)
    if not major > 0:
        # A hand-built observation records no length; 1.x's estimate from the
        # area stands in, and is said to be one.
        major = max(math.sqrt(max(area, 1.0) * 4.0), 3.0 * max(minor, 1.0))
    minor = max(minor, 1.0)
    u = np.array([math.sin(theta), math.cos(theta)], dtype=float)
    return major, minor, u, area


def expected_position(track: Track, frame: int) -> np.ndarray:
    """Where a track says its cell should be at one frame.

    ``Track.predict`` extrapolates forward from the last observation and
    refuses to look backwards, which is exactly right while tracking is running
    -- the future is all it can know. Here it is the wrong tool, and using it
    was a real defect: once tracking has finished, ``last_frame`` is the track's
    *final* observation, so every interior gap lies behind it and every single
    one raised ValueError. Recovery examined nothing but the frames past the end
    of each track, and said nothing about it.

    An interior gap deserves better than extrapolation anyway. The cell's
    position is known on *both* sides, so the estimate is an interpolation
    between them, which needs no velocity model and cannot drift.
    """
    return expected_body(track, frame).position


def expected_body(track: Track, frame: int) -> Expectation:
    """The expected position, body and bracket of ``track`` at ``frame``.

    Interior frames interpolate position by elapsed frames and orientation as
    an axis (a major axis has no head or tail, so the far side is flipped onto
    the near side before averaging). Lengths take the larger of the two
    bracketing observations, so the window holds the cell in either shape.
    """
    observations = track.observations
    if not observations:
        raise ValueError(f"Track {track.id} has no observations to predict from.")

    frame = int(frame)
    if frame > observations[-1].frame:
        last = observations[-1]
        major, minor, u, area = _body(last)
        position = np.asarray(track.predict(frame), dtype=float)[:2]
        return Expectation(
            frame, position, major, minor, u, area,
            (int(last.frame), int(last.det_label), None, None), int(last.channel),
        )
    if frame < observations[0].frame:
        raise ValueError(
            f"Track {track.id} starts at frame {observations[0].frame}; "
            f"frame {frame} is before it existed."
        )

    for before, after in zip(observations[:-1], observations[1:]):
        if before.frame < frame < after.frame:
            fraction = (frame - before.frame) / float(after.frame - before.frame)
            position = np.array(
                [
                    before.x + (after.x - before.x) * fraction,
                    before.y + (after.y - before.y) * fraction,
                ],
                dtype=float,
            )
            maj0, min0, u0, a0 = _body(before)
            maj1, min1, u1, a1 = _body(after)
            if float(u0 @ u1) < 0:
                u1 = -u1
            u = (1.0 - fraction) * u0 + fraction * u1
            norm = float(np.hypot(u[0], u[1]))
            u = u / norm if norm > 1e-9 else u0
            # With the lane gate applied the two cannot differ (a link across
            # lanes is refused); without it the lane is informational only.
            lane = int(before.channel)
            return Expectation(
                frame, position, max(maj0, maj1), max(min0, min1), u,
                0.5 * (a0 + a1),
                (int(before.frame), int(before.det_label), int(after.frame), int(after.det_label)),
                lane,
            )

    raise ValueError(
        f"Track {track.id} already has an observation at frame {frame}."
    )


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def window_for(
    expectation: Expectation, cfg: RecoveryConfig, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)``: a crop around the expected cell, sized from its own body.

    The crop is a rectangle in the cell's frame -- ``window_scale``
    half-lengths along the body, ``2 * window_scale`` widths across it --
    and the image-aligned box that contains it. 1.x took the length along y
    and the width along x whatever the cell's orientation, so a 140 px
    horizontal cell got a crop 97 px wide and came back cut at both ends.
    """
    height, width = shape
    half_along = cfg.window_scale * expectation.major_px / 2.0
    half_across = cfg.window_scale * 2.0 * expectation.minor_px
    u, n = expectation.u, expectation.n
    half_x = max(cfg.window_min_px, int(math.ceil(abs(u[0]) * half_along + abs(n[0]) * half_across)))
    half_y = max(cfg.window_min_px, int(math.ceil(abs(u[1]) * half_along + abs(n[1]) * half_across)))
    cx, cy = int(round(expectation.position[0])), int(round(expectation.position[1]))
    x0 = max(0, cx - half_x)
    x1 = min(width, cx + half_x + 1)
    y0 = max(0, cy - half_y)
    y1 = min(height, cy + half_y + 1)
    return x0, y0, x1, y1


def _closest(detections: Sequence[Detection], point: np.ndarray) -> Detection | None:
    if not detections:
        return None
    return min(
        detections,
        key=lambda d: float(np.hypot(d.x - point[0], d.y - point[1])),
    )


def _acceptable(
    candidate: Detection, expectation: Expectation, cfg: RecoveryConfig
) -> tuple[bool, float, str]:
    """1.x's plausibility test, unchanged in form: distance and size.

    The length is still 1.x's area-based estimate (floored at 30 px), so the
    measured offset limit means what it meant; the expected area is now the
    bracketing observations' rather than the track's final one, which may be
    many frames away from the gap.
    """
    offset = float(np.hypot(candidate.x - expectation.position[0],
                            candidate.y - expectation.position[1]))
    length = max(math.sqrt(max(expectation.area_px, 1.0) * 4.0), 30.0)
    limit = length * cfg.max_offset_lengths
    if offset > limit:
        return False, offset, f"{offset:.0f} px from the prediction, limit {limit:.0f}"
    ratio = candidate.area_px / max(expectation.area_px, 1.0)
    if not 0.25 <= ratio <= 4.0:
        return False, offset, f"area ratio {ratio:.2f} is implausible"
    return True, offset, ""


def _lane_check(
    candidate: Detection,
    track_lane: int,
    geometry: ChannelGeometry | None,
    tracking_cfg: TrackingConfig,
) -> str:
    """Empty when the candidate may belong to the track; the refusal otherwise.

    The same rule as the tracker's lane gate (``tracking._hard_gates``): it
    applies only when :func:`lane_gate_applies` says so (an applied geometry
    with lanes, and a tracking configuration that constrains to them), and it
    refuses only a candidate in a *different real lane* from the track's.  A
    lane of -1 -- a centroid just outside every measured half-width, or a
    track never seen inside a lane -- is never gated (``geometry.lane_of``):
    there is no lane it could be refused from.  A stricter rule here would
    throw away a cell the re-tracking pass would have linked.
    """
    if geometry is None:
        return ""
    candidate.channel = geometry.lane_of(candidate.x, candidate.y)
    if not lane_gate_applies(geometry, tracking_cfg):
        return ""
    if track_lane >= 0 and candidate.channel >= 0 and candidate.channel != track_lane:
        return f"in lane {candidate.channel}, but the track is in lane {track_lane}"
    return ""


def _contains(det: Detection, x: float, y: float) -> bool:
    """Whether the pixel at (x, y) belongs to ``det``'s own mask (2-D)."""
    if det.mask_crop is None or len(det.bbox) != 4:
        return False
    r = int(round(y)) - int(det.bbox[0])
    c = int(round(x)) - int(det.bbox[1])
    crop = np.asarray(det.mask_crop)
    return bool(0 <= r < crop.shape[0] and 0 <= c < crop.shape[1] and crop[r, c])


def _inside_body(det: Detection, x: float, y: float, along_lengths: float) -> bool:
    """Within ``along_lengths`` of det's length along its own axis, and inside its width."""
    major = float(det.major_axis_px)
    minor = float(det.minor_axis_px)
    if not major > 0:
        return False
    u = det.axis_unit
    n = np.array([-u[1], u[0]], dtype=float)
    d = np.array([x - det.x, y - det.y], dtype=float)
    return bool(abs(d @ u) <= along_lengths * major and abs(d @ n) <= 0.5 * max(minor, 1.0))


def duplicate_of(
    candidate: Detection, existing: Sequence[Detection], cfg: RecoveryConfig
) -> tuple[Detection, str] | None:
    """The detection ``candidate`` is a second copy of, and why; None if it is new.

    A candidate duplicates a detection of the same frame when any of these
    holds, tested both ways round so a fragment and the whole cell match
    whichever of them was found first:

    * their masks overlap with IoU above ``cfg.duplicate_iou``;
    * either centroid lies inside the other's mask;
    * either centroid lies within ``cfg.duplicate_along_lengths`` of the other
      cell's length from its centroid along that cell's own long axis, and
      inside its width across it (the only test left when a mask is missing).
    """
    for other in existing:
        if int(other.frame) != int(candidate.frame):
            continue
        iou = shifted_iou(other, candidate, (0.0, 0.0))
        if iou is not None and iou > cfg.duplicate_iou:
            return other, f"its mask overlaps it with IoU {iou:.2f}"
        if _contains(other, candidate.x, candidate.y) or _contains(candidate, other.x, other.y):
            return other, "one centroid lies inside the other's mask"
        a = cfg.duplicate_along_lengths
        if _inside_body(other, candidate.x, candidate.y, a) or _inside_body(candidate, other.x, other.y, a):
            gap = float(np.hypot(candidate.x - other.x, candidate.y - other.y))
            return other, (
                f"its centroid is {gap:.1f} px from it, inside the cell's own body "
                "(half a length along it, half a width across)"
            )
    return None


def _measure_crop(
    labels: np.ndarray, box: tuple[int, int, int, int], frame: np.ndarray, frame_index: int
) -> list[Detection]:
    """Measure a crop's labels as full-frame detections, mask and morphology included.

    Everything ``extract_detections`` measures is translation-invariant except
    the coordinates and the border flag, so measuring the crop and shifting
    is exact -- and avoids a frame-sized label image per attempt.
    """
    x0, y0, x1, y1 = box
    height, width = frame.shape[:2]
    found = extract_detections(labels, frame_index, intensity=frame[y0:y1, x0:x1])
    for det in found:
        det.x += x0
        det.y += y0
        r0, c0, r1, c1 = det.bbox
        det.bbox = (r0 + y0, c0 + x0, r1 + y0, c1 + x0)
        det.touches_border = bool(
            det.bbox[0] == 0 or det.bbox[1] == 0 or det.bbox[2] >= height or det.bbox[3] >= width
        )
    return found


def _cut_by_window(det: Detection, box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    """The mask reaches a crop edge that is not also the image edge."""
    x0, y0, x1, y1 = box
    height, width = shape
    r0, c0, r1, c1 = det.bbox
    return bool(
        (r0 <= y0 and y0 > 0) or (c0 <= x0 and x0 > 0)
        or (r1 >= y1 and y1 < height) or (c1 >= x1 and x1 < width)
    )


# --------------------------------------------------------------------------
# Tier 3: no network
# --------------------------------------------------------------------------


def intensity_candidate(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    expectation: Expectation,
    cfg: RecoveryConfig,
    frame_index: int,
    background: np.ndarray | None = None,
) -> tuple[Detection | None, str]:
    """Find a bright elongated object in the crop without using the network.

    The device is static, so subtracting a temporal median leaves only what
    moved. Returns ``(candidate, "")`` or ``(None, why)``.
    """
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1].astype(np.float32)
    if crop.size < 64:
        return None, "window too small for an intensity test"
    if background is not None:
        crop = crop - background[y0:y1, x0:x1].astype(np.float32)

    centre = float(np.median(crop))
    spread = 1.4826 * float(np.median(np.abs(crop - centre)))
    if spread <= 1e-6:
        return None, "flat window, no intensity contrast"
    hot = (crop - centre) / spread >= cfg.intensity_snr
    if not hot.any():
        return None, f"nothing {cfg.intensity_snr:.0f} sigma above the local background"

    # Keep the largest four-connected blob (ndimage's default structure in 2-D).
    labels, count = ndimage.label(hot)
    if count == 0:
        return None, "nothing above the local background"
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    best = int(np.argmax(sizes))
    area = int(sizes[best])
    expected = max(expectation.area_px, 1.0)
    if area < expected * cfg.intensity_min_area_fraction:
        return None, f"brightest object is {area} px, under {cfg.intensity_min_area_fraction:.0%} of the expected area"

    found = _measure_crop((labels == best).astype(np.int32), box, frame, frame_index)
    if not found:
        return None, "brightest object could not be measured"
    detection = found[0]

    # Brightness alone cannot distinguish a cell from the frame border or the
    # reservoir band, which are the brightest things in these images. Shape and
    # position can, so a candidate has to look like a confined cell and be
    # where this one could be.
    height, width = frame.shape[:2]
    margin = cfg.intensity_edge_margin_px
    if not (margin <= detection.x <= width - margin
            and margin <= detection.y <= height - margin):
        return None, "bright object at the image edge"
    if detection.touches_border:
        return None, "bright object touches the image border"
    if detection.eccentricity < cfg.intensity_min_eccentricity:
        return None, f"bright object too round (eccentricity {detection.eccentricity:.2f})"

    _, across = expectation.body_offset(detection.x, detection.y)
    limit = cfg.intensity_max_across_widths * expectation.minor_px
    if across > limit:
        return None, (
            f"bright object {across:.1f} px across the cell's body from where it was "
            f"expected, limit {limit:.1f} px"
        )
    return detection, ""


# --------------------------------------------------------------------------
# The recovery pass
# --------------------------------------------------------------------------


def _primaries_by_frame(tracks: Sequence[Track]) -> dict[int, list[Detection]]:
    """Every first-pass detection, by frame.

    The tracker turns every detection into an observation (an unmatched one
    starts its own track), so the observations of the first-pass tracks are
    exactly the primary detections.
    """
    out: dict[int, list[Detection]] = {}
    for track in tracks:
        for obs in track.observations:
            det = obs.detection if obs.detection is not None else _probe_detection(obs)
            out.setdefault(int(obs.frame), []).append(det)
    return out


def recover(
    stack: np.ndarray,
    tracks: Sequence[Track],
    service,
    scale: Scale,
    tracking_cfg: TrackingConfig,
    recovery_cfg: RecoveryConfig,
    legacy_recovery_cfg: RecoveryConfig | None = None,
    *,
    geometry: ChannelGeometry | None = None,
    background: np.ndarray | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> RecoveryResult:
    """Look for cells in the frames where a first-pass track went unobserved.

    ``stack`` is ``(T, Y, X)``. ``scale`` is part of the interface but unused:
    every quantity here is in pixels of the image being searched.
    ``geometry`` adds the lane test (when the tracker's lane gate applies) and
    stamps recovered detections with their lane.

    Transition: ``recover(stack, tracks, service, axis, scale, tracking_cfg,
    recovery_cfg, ...)`` (v1 order) still works, exactly as
    ``track_detections`` and ``explain_unlinked_starts`` do: the axis only
    supplies lanes, through :meth:`ChannelGeometry.from_legacy_axis`. Without
    it a v1 pipeline raised TypeError on every run with recovery on. It goes
    when the pipeline passes a real ``ChannelGeometry`` (E1).
    """
    if _is_legacy_axis(scale):  # v1 order: (axis, scale, tracking_cfg, recovery_cfg)
        axis = scale
        scale, tracking_cfg = tracking_cfg, recovery_cfg  # type: ignore[assignment]
        recovery_cfg = legacy_recovery_cfg  # type: ignore[assignment]
        if geometry is None:
            geometry = ChannelGeometry.from_legacy_axis(axis)
    elif legacy_recovery_cfg is not None:
        raise TypeError(
            "recover() takes 6 positional arguments but 7 were given; geometry, "
            "background and progress are keyword-only"
        )
    if not isinstance(recovery_cfg, RecoveryConfig):
        raise TypeError(
            "recover(stack, tracks, service, scale, tracking_cfg, recovery_cfg, *, "
            f"geometry=None, background=None): recovery_cfg is {type(recovery_cfg).__name__}"
        )
    cfg = recovery_cfg
    result = RecoveryResult()
    if not cfg.enabled or not tracks:
        return result
    if np.asarray(stack).ndim != 3:
        result.notes.append(
            "Recovery re-segments 2-D windows; this stack is not (T, Y, X), so it was not run."
        )
        return result

    n_frames = int(stack.shape[0])
    shape = (int(stack.shape[1]), int(stack.shape[2]))
    gaps = _dormant_frames(tracks, n_frames, tracking_cfg, cfg)
    total = len(gaps)
    primaries = _primaries_by_frame(tracks)
    recovered_by_frame: dict[int, list[Detection]] = {}

    for index, (track, frame_index) in enumerate(gaps):
        if progress is not None:
            progress(index + 1, total)
        try:
            expectation = expected_body(track, frame_index)
        except ValueError as exc:
            # Recorded, never skipped in silence. A gap that was enumerated and
            # then dropped without a row is invisible in recovery_attempts.csv
            # and in the manifest's attempt count, which is how a pass can look
            # like it ran while never having examined anything.
            result.attempts.append(
                RecoveryAttempt(
                    track.id, frame_index, float("nan"), float("nan"),
                    found=False, detail=f"no position could be estimated: {exc}",
                )
            )
            continue
        px, py = float(expectation.position[0]), float(expectation.position[1])
        if not (0 <= px < shape[1] and 0 <= py < shape[0]):
            result.attempts.append(
                RecoveryAttempt(
                    track.id, frame_index, px, py, found=False,
                    detail="predicted position is outside the image",
                    bracket=expectation.bracket,
                )
            )
            continue

        box = window_for(expectation, cfg, shape)
        existing = primaries.get(frame_index, []) + recovered_by_frame.get(frame_index, [])
        attempt = _try_tiers(
            stack[frame_index], box, track, expectation, cfg, frame_index, service,
            background, geometry, existing, tracking_cfg,
        )
        result.attempts.append(attempt)
        if attempt.detection is not None:
            # A crop's labels start at 1 and collide with the frame's primary
            # labels, so a recovered cell gets the next label free in its
            # frame before anything can refer to it -- the rule the pipeline
            # applies too (highest primary label, then one more per recovered
            # cell in this order), so relabelling there changes nothing.
            attempt.detection.label = 1 + max((int(d.label) for d in existing), default=0)
            result.detections.append(attempt.detection)
            recovered_by_frame.setdefault(frame_index, []).append(attempt.detection)
    return result


def _try_tiers(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    track: Track,
    expectation: Expectation,
    cfg: RecoveryConfig,
    frame_index: int,
    service,
    background: np.ndarray | None,
    geometry: ChannelGeometry | None,
    existing: Sequence[Detection],
    tracking_cfg: TrackingConfig,
) -> RecoveryAttempt:
    x0, y0, x1, y1 = box
    # The tracker gates on Track.channel (the first lane the track was seen
    # in); the bracketing observation's lane stands in for a hand-built track
    # that never had one set.
    track_lane = int(track.channel) if int(track.channel) >= 0 else int(expectation.lane)
    crop = frame[y0:y1, x0:x1]
    attempt = RecoveryAttempt(
        track.id, frame_index, float(expectation.position[0]), float(expectation.position[1]),
        found=False, bracket=expectation.bracket,
    )
    shape = (int(frame.shape[0]), int(frame.shape[1]))

    def accept(candidate: Detection, source: str, how: str) -> bool:
        """Run the shared checks; True when the attempt is settled either way."""
        ok, offset, why = _acceptable(candidate, expectation, cfg)
        if not ok:
            attempt.detail = why
            return False
        why = _lane_check(candidate, track_lane, geometry, tracking_cfg)
        if why:
            attempt.detail = why
            return False
        dup = duplicate_of(candidate, existing, cfg)
        if dup is not None:
            # The cell this track predicts is already a detection of this
            # frame -- assigned to another track. Adding it again is what ran
            # two tracks along one cell; a later tier would only find the same
            # pixels again, so the attempt ends here.
            other, reason = dup
            attempt.duplicate_of = other
            attempt.duplicate_of_label = int(other.label)
            if other.source == SOURCE_PRIMARY:
                # A primary label is the pixel value in masks.npz and is never
                # changed, so it is safe to name in prose.
                what = f"the primary detection labelled {other.label}"
            else:
                # Prose cannot follow a later relabelling; the column can.
                what = "a cell already recovered (see duplicate_of_label)"
            attempt.detail = (
                f"dropped as a duplicate: the {source} candidate is {what} "
                f"in this frame ({reason})"
            )
            return True
        candidate.source = source
        candidate.confidence = TIER_CONFIDENCE[source]
        attempt.found = True
        attempt.source = source
        attempt.confidence = candidate.confidence
        attempt.offset_px = offset
        attempt.detail = how + (
            "; the mask reaches the window edge, so its shape may be cut"
            if _cut_by_window(candidate, box, shape) else ""
        )
        attempt.detection = candidate
        return True

    tiers: list[tuple[str, Any]] = []
    if cfg.window:
        tiers.append((SOURCE_WINDOW, None))
    if cfg.permissive:
        tiers.append((SOURCE_PERMISSIVE, (cfg.permissive_cellprob, cfg.permissive_flow)))

    for source, override in tiers:
        try:
            mask = service.segment_crop(crop, override)
        except Exception:  # noqa: BLE001 - a failed tier falls through to the next
            continue
        if mask is None or int(np.max(mask)) == 0:
            continue
        mask = np.asarray(mask)
        if mask.shape != crop.shape[:2]:
            attempt.detail = f"the {source} pass returned a {mask.shape} mask for a {crop.shape[:2]} window"
            continue
        candidate = _closest(_measure_crop(mask.astype(np.int32), box, frame, frame_index),
                             expectation.position)
        if candidate is None:
            continue
        if accept(candidate, source, f"re-segmented a {x1 - x0}x{y1 - y0} px window"):
            return attempt

    if cfg.intensity:
        candidate, why = intensity_candidate(frame, box, expectation, cfg, frame_index, background)
        if candidate is None:
            attempt.detail = attempt.detail or why
        elif accept(
            candidate, SOURCE_INTENSITY,
            f"bright object {cfg.intensity_snr:.0f} sigma above the local background, "
            "no network involved",
        ):
            return attempt

    if not attempt.detail:
        attempt.detail = "nothing found where the track predicted a cell"
    return attempt


def assign_final_track_ids(
    attempts: Sequence[RecoveryAttempt], final_tracks: Sequence[Track]
) -> None:
    """Re-key attempts from first-pass track ids to the final, re-tracked ones.

    A found attempt belongs to whichever final track holds its recovered
    detection (found by identity, so the pipeline's relabelling does not
    matter). Any other attempt belongs to the final track holding the
    observation before the gap, then the one after it. ``first_pass_track_id``
    keeps the original. An attempt that none of them locates gets
    ``track_id = None`` and says why, rather than keeping a first-pass id that
    would join the wrong row of tracks.csv.
    """
    by_object: dict[int, int] = {}
    by_key: dict[tuple[int, int], int] = {}
    for tr in final_tracks:
        for obs in tr.observations:
            if obs.detection is not None:
                by_object[id(obs.detection)] = tr.id
            # (frame, label) is unique across primary and recovered detections:
            # the pipeline relabels recovered ones above the frame's highest
            # primary label, and a bracket only ever names a primary one.
            by_key[(int(obs.frame), int(obs.det_label))] = tr.id
    for attempt in attempts:
        final: int | None = None
        if attempt.detection is not None:
            final = by_object.get(id(attempt.detection))
        if final is None and attempt.bracket is not None:
            before_f, before_l, after_f, after_l = attempt.bracket
            final = by_key.get((int(before_f), int(before_l)))
            if final is None and after_f is not None and after_l is not None:
                final = by_key.get((int(after_f), int(after_l)))
        if final is None:
            attempt.detail = (attempt.detail + "; " if attempt.detail else "") + (
                "no final track holds this attempt's detection or bracket"
            )
        attempt.track_id = final


def _dormant_frames(
    tracks: Sequence[Track],
    n_frames: int,
    cfg: TrackingConfig,
    recovery: "RecoveryConfig | None" = None,
) -> list[tuple[Track, int]]:
    """Frames where a track was alive but unobserved.

    Two kinds, and they are different questions.

    **Interior gaps**, between two observations, are unambiguous: the cell was
    there before and after, so the hole is a detection failure and the only
    question is where exactly. These are the gaps the first-pass tracker
    already bridged (``gap <= max_gap + 1``, or longer only through global gap
    closing).

    **Trailing frames**, past a track's last observation, look like the more
    valuable case -- a cell that vanishes and never returns is where recall is
    lost -- and are in fact the dangerous one. There is no evidence on the far
    side, so "the cell is still there" is an assumption rather than a
    deduction. Measured on a channel known to be empty, every false position
    came from here. They are off by default for that reason.
    """
    out: list[tuple[Track, int]] = []
    limit = cfg.max_delta_frames()
    for track in tracks:
        frames = [o.frame for o in track.observations]
        if not frames:
            continue
        if recovery is None or recovery.interior:
            for a, b in zip(frames[:-1], frames[1:]):
                for missing in range(a + 1, b):
                    out.append((track, missing))
        if recovery is not None and recovery.trailing:
            last = frames[-1]
            for ahead in range(1, limit + 1):
                frame = last + ahead
                if frame < n_frames:
                    out.append((track, frame))
    return out
