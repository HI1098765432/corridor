"""Finding cells the first pass missed.

The measured failure mode of this segmentation model is recall: held out it
finds 20-44 % of labelled cells, and even on its own training images it misses
12-20 %. Precision stays high (0.54-0.60 held out), so what it reports is
usually right -- it simply does not report enough.

That asymmetry is what makes recovery worth doing, and it dictates the shape of
it. A global sensitivity increase trades precision for recall everywhere, and
on the near-empty control stack it starts reporting a cell in every frame. What
is safe instead is to become more sensitive *only where a track already
predicts a cell*: a specific position, inside one channel, with a known
expected size and appearance. Evidence that would be too weak to assert a cell
on its own is enough to confirm one that motion continuity already predicts.

Three tiers, each tried only when the one before it fails:

1.  **Re-segment a window.** Crop around the prediction and run the same model
    on it. This is not just a permissiveness change: Cellpose normalises per
    image, so a crop containing mostly one cell puts that cell at the top of
    the intensity range instead of buried under the brightest structure in the
    whole field.
2.  **Permissive re-segmentation.** The same crop at a much lower probability
    threshold. Only reachable inside a predicted window.
3.  **Intensity profile.** No network at all: these cells are bright elongated
    objects inside a narrow channel, so a local signal-to-noise test along the
    channel finds them when the network will not.

Everything recovered is marked as recovered, carries the tier that found it and
a confidence, and is written to its own column in the output. A recovered
position is a weaker claim than a primary detection and must never be
indistinguishable from one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .config import Scale, SegmentationConfig, TrackingConfig
from .confinement import ConfinementAxis
from .detections import Detection, extract_detections
from .tracking import Track, TrackState

SOURCE_PRIMARY = "primary"
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


@dataclass
class RecoveryConfig:
    """How hard to look for a cell a track says should be there.

    **On by default for interior gaps, off for trailing frames.** That split is
    not a compromise; it is the measurement.

    Scored against real ground truth -- deleting a primary detection from a
    frame where a cell is known present, and checking whether recovery puts it
    back, while also counting anything it reports in frames where a cell is
    known absent (scripts/experiment_recovery.py):

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

    An earlier version of this docstring said recovery "does not work" and
    reported 1 of 13. That measurement was taken while a defect skipped every
    interior gap in silence (see ``expected_position``), so it described the
    skip and not the algorithm. The number above replaces it.
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

    #: Crop size as a multiple of the track's own bounding box. Large enough to
    #: give Cellpose context, small enough that normalisation actually helps.
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
    #: And must sit within this many multiples of the channel half-width of the
    #: channel centre line. A cell is in the channel by definition.
    intensity_max_channel_offset: float = 1.0

    #: A recovered detection must still be a plausible continuation: it is
    #: rejected if it is further than this many multiples of the track's own
    #: length from the prediction.
    max_offset_lengths: float = 1.5


@dataclass
class RecoveryAttempt:
    """One attempt to find a missing cell, successful or not."""

    track_id: int
    frame: int
    predicted_x: float
    predicted_y: float
    found: bool
    source: str = ""
    confidence: float = 0.0
    offset_px: float | None = None
    detail: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "frame": self.frame,
            "predicted_x": self.predicted_x,
            "predicted_y": self.predicted_y,
            "recovered": self.found,
            "found_by": self.source,
            "confidence": self.confidence,
            "offset_from_prediction_px": self.offset_px,
            "detail": self.detail,
        }


@dataclass
class RecoveryResult:
    detections: list[Detection] = field(default_factory=list)
    attempts: list[RecoveryAttempt] = field(default_factory=list)

    @property
    def n_recovered(self) -> int:
        return sum(1 for a in self.attempts if a.found)

    @property
    def n_attempted(self) -> int:
        return len(self.attempts)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def _window_for(track: Track, cfg: RecoveryConfig, shape: tuple[int, int],
                centre: np.ndarray) -> tuple[int, int, int, int]:
    """A crop around the predicted position, sized from the cell itself."""
    height, width = shape
    last = track.last
    extent = max(last.minor_axis_px, 1.0)
    # Use the track's own recent area to estimate how long it is.
    length = max(math.sqrt(max(last.area_px, 1.0) * 4.0), extent * 3.0)
    half_y = max(cfg.window_min_px, int(length * cfg.window_scale / 2))
    half_x = max(cfg.window_min_px, int(extent * cfg.window_scale * 2))
    cx, cy = int(round(centre[0])), int(round(centre[1]))
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
    candidate: Detection, track: Track, prediction: np.ndarray, cfg: RecoveryConfig
) -> tuple[bool, float, str]:
    offset = float(np.hypot(candidate.x - prediction[0], candidate.y - prediction[1]))
    length = max(math.sqrt(max(track.last.area_px, 1.0) * 4.0), 30.0)
    limit = length * cfg.max_offset_lengths
    if offset > limit:
        return False, offset, f"{offset:.0f} px from the prediction, limit {limit:.0f}"
    ratio = candidate.area_px / max(track.last_area, 1.0)
    if not 0.25 <= ratio <= 4.0:
        return False, offset, f"area ratio {ratio:.2f} is implausible"
    return True, offset, ""


# --------------------------------------------------------------------------
# Tier 3: no network
# --------------------------------------------------------------------------


def intensity_candidate(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    track: Track,
    axis: ConfinementAxis,
    cfg: RecoveryConfig,
    frame_index: int,
    background: np.ndarray | None = None,
) -> Detection | None:
    """Find a bright elongated object in the crop without using the network.

    The device is static, so subtracting a temporal median leaves only what
    moved. Inside a confinement channel that is the cell.
    """
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1].astype(np.float32)
    if crop.size < 64:
        return None
    if background is not None:
        crop = crop - background[y0:y1, x0:x1].astype(np.float32)

    centre = float(np.median(crop))
    spread = 1.4826 * float(np.median(np.abs(crop - centre)))
    if spread <= 1e-6:
        return None
    z = (crop - centre) / spread
    hot = z >= cfg.intensity_snr
    if not hot.any():
        return None

    # Keep the largest connected blob. Labelling by flood fill avoids a SciPy
    # dependency here and the crops are small.
    labels = _label(hot)
    if labels.max() == 0:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    best = int(np.argmax(sizes))
    area = int(sizes[best])
    expected = max(track.last_area, 1.0)
    if area < expected * cfg.intensity_min_area_fraction:
        return None

    mask = np.zeros(frame.shape, dtype=np.int32)
    mask[y0:y1, x0:x1][labels == best] = 1
    found = extract_detections(mask, frame_index, intensity=frame)
    if not found:
        return None
    detection = found[0]

    # Brightness alone cannot distinguish a cell from the frame border or the
    # reservoir band, which are the brightest things in these images. Shape and
    # position can, so a candidate has to look like a confined cell and be
    # where one could be.
    height, width = frame.shape
    margin = cfg.intensity_edge_margin_px
    if not (margin <= detection.x <= width - margin
            and margin <= detection.y <= height - margin):
        return None
    if detection.touches_border:
        return None
    if detection.eccentricity < cfg.intensity_min_eccentricity:
        return None

    detection.channel = axis.channel_of(detection.x, detection.y)
    offset = axis.distance_to_own_channel(detection.x, detection.y)
    half_width = axis.channels[detection.channel].half_width_px if axis.channels else 1e9
    if offset > half_width * cfg.intensity_max_channel_offset:
        return None
    return detection


def _label(mask: np.ndarray) -> np.ndarray:
    """Four-connected labelling, iterative so a long thin cell cannot overflow."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    height, width = mask.shape
    for sy in range(height):
        for sx in range(width):
            if not mask[sy, sx] or labels[sy, sx]:
                continue
            current += 1
            stack = [(sy, sx)]
            labels[sy, sx] = current
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width:
                        if mask[ny, nx] and not labels[ny, nx]:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels


# --------------------------------------------------------------------------
# The recovery pass
# --------------------------------------------------------------------------


def recover(
    stack: np.ndarray,
    tracks: Sequence[Track],
    service,
    axis: ConfinementAxis,
    scale: Scale,
    tracking: TrackingConfig,
    cfg: RecoveryConfig,
    *,
    background: np.ndarray | None = None,
    progress=None,
) -> RecoveryResult:
    """Look for cells in the frames where a track went dormant."""
    result = RecoveryResult()
    if not cfg.enabled or not tracks:
        return result

    n_frames = int(stack.shape[0])
    shape = (int(stack.shape[1]), int(stack.shape[2]))
    gaps = _dormant_frames(tracks, n_frames, tracking, cfg)
    total = len(gaps)

    for index, (track, frame_index) in enumerate(gaps):
        try:
            prediction = expected_position(track, frame_index)
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
        if not (0 <= prediction[0] < shape[1] and 0 <= prediction[1] < shape[0]):
            result.attempts.append(
                RecoveryAttempt(
                    track.id, frame_index, float(prediction[0]), float(prediction[1]),
                    found=False, detail="predicted position is outside the image",
                )
            )
            continue

        box = _window_for(track, cfg, shape, prediction)
        frame = stack[frame_index]
        attempt = _try_tiers(
            frame, box, track, prediction, axis, cfg, frame_index, service, background
        )
        result.attempts.append(attempt[0])
        if attempt[1] is not None:
            result.detections.append(attempt[1])
        if progress is not None:
            progress(index + 1, total)
    return result


def _try_tiers(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    track: Track,
    prediction: np.ndarray,
    axis: ConfinementAxis,
    cfg: RecoveryConfig,
    frame_index: int,
    service,
    background: np.ndarray | None,
) -> tuple[RecoveryAttempt, Detection | None]:
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1]
    base = RecoveryAttempt(
        track.id, frame_index, float(prediction[0]), float(prediction[1]), found=False
    )

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
        if mask is None or int(mask.max()) == 0:
            continue
        full = np.zeros(frame.shape, dtype=np.int32)
        full[y0:y1, x0:x1] = mask
        candidates = extract_detections(full, frame_index, intensity=frame)
        candidate = _closest(candidates, prediction)
        if candidate is None:
            continue
        ok, offset, why = _acceptable(candidate, track, prediction, cfg)
        if not ok:
            base.detail = why
            continue
        candidate.channel = axis.channel_of(candidate.x, candidate.y)
        candidate.source = source
        candidate.confidence = TIER_CONFIDENCE[source]
        base.found = True
        base.source = source
        base.confidence = candidate.confidence
        base.offset_px = offset
        base.detail = f"re-segmented a {x1 - x0}x{y1 - y0} px window"
        return base, candidate

    if cfg.intensity:
        candidate = intensity_candidate(
            frame, box, track, axis, cfg, frame_index, background
        )
        if candidate is not None:
            ok, offset, why = _acceptable(candidate, track, prediction, cfg)
            if ok:
                candidate.source = SOURCE_INTENSITY
                candidate.confidence = TIER_CONFIDENCE[SOURCE_INTENSITY]
                base.found = True
                base.source = SOURCE_INTENSITY
                base.confidence = candidate.confidence
                base.offset_px = offset
                base.detail = (
                    f"bright object {cfg.intensity_snr:.0f} sigma above the local "
                    "background, no network involved"
                )
                return base, candidate
            base.detail = why

    if not base.detail:
        base.detail = "nothing found where the track predicted a cell"
    return base, None


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
    observations = track.observations
    if not observations:
        raise ValueError(f"Track {track.id} has no observations to predict from.")

    frame = int(frame)
    if frame > observations[-1].frame:
        return track.predict(frame)
    if frame < observations[0].frame:
        raise ValueError(
            f"Track {track.id} starts at frame {observations[0].frame}; "
            f"frame {frame} is before it existed."
        )

    for before, after in zip(observations[:-1], observations[1:]):
        if before.frame < frame < after.frame:
            span = after.frame - before.frame
            fraction = (frame - before.frame) / float(span)
            return np.array(
                [
                    before.x + (after.x - before.x) * fraction,
                    before.y + (after.y - before.y) * fraction,
                ],
                dtype=float,
            )

    raise ValueError(
        f"Track {track.id} already has an observation at frame {frame}."
    )


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
    question is where exactly. Measured against real holes, these are recovered
    at 0.85 with a median error under a quarter of a cell width.

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
