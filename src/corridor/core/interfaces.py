"""The seams between segmentation, tracking, measurement and prediction.

The directive (§52) asks for the four stages to stay separable: a
segmentation model, a tracker or a predictor must be replaceable, testable and
validatable on its own, and a result must be able to say which
implementation of each produced it.  In 1.x one object crossed every
boundary: a migration axis inferred from the image and the detections, which
the tracker's costs, the measurements' along/across columns and quality
control all read.

These Protocols are the boundaries.  They say what crosses each one and
nothing about how a stage works inside; implementations conform
structurally and need not inherit from anything.  Nothing here imports
Cellpose, torch or Qt, so a test, a research script or a future model pack
can depend on the seams without pulling in a stage::

    stack (T[Z]YX)        --Segmenter-->  SegmentationOutput: masks, detections,
                                          per-frame diagnostics
    detections by frame   --Tracker---->  (tracks, frame events)
    tracks + Scale        --measurement-> rows, summaries, MSD (pure functions)
    features              --Predictor-->  predictions, each with an interval

Measurement is deliberately not a Protocol: it is a set of pure functions of
tracks and a :class:`~corridor.core.config.Scale`, with one implementation,
and an interface with one implementation is ceremony.

The current implementations predate these seams and are adapted by the
packages that rewrite them: ``SegmentationService.run_stack`` is today's
segmenter, ``track_detections`` / ``ConfinementTracker.run`` today's tracker
(it still takes a migration axis, which v2 removes), and no predictor exists
in the application yet (``training/predict`` is research, outside it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps the seams import-light
    from .detections import Detection
    from .segmentation import SegmentationOutput
    from .tracking import FrameEvent, Track

#: ``progress(done, total)``; returning False asks the stage to stop.
ProgressCallback = Callable[[int, int], "bool | None"]


@runtime_checkable
class Segmenter(Protocol):
    """Turns an image stack into labelled masks and measured detections.

    ``stack`` is in a canonical order (``TYX`` or ``TZYX``) with any channel
    axis already reduced at import.  An implementation that cannot handle the
    stack's dimensionality must refuse it -- for 3-D, by raising
    ``model_registry.ModelUnavailable`` -- rather than segment it slice by
    slice as though Z were time.
    """

    def segment(
        self,
        stack: np.ndarray,
        *,
        progress: ProgressCallback | None = None,
    ) -> "SegmentationOutput": ...


@runtime_checkable
class Tracker(Protocol):
    """Links detections into tracks.

    It sees detections and nothing else: no image, no migration direction.
    ``n_frames`` is passed separately because the last frames may hold no
    detections and a track's dormancy still has to be judged against them.
    Frame events (merges, splits, gap closures, border entries) are returned
    beside the tracks so they reach ``tracking_events.csv`` and quality
    control without a second pass.
    """

    def track(
        self,
        detections_by_frame: Mapping[int, Sequence["Detection"]],
        n_frames: int,
    ) -> tuple[list["Track"], list["FrameEvent"]]: ...


@dataclass(frozen=True)
class Prediction:
    """One predicted value with the interval that makes it checkable.

    The interval is required, not optional: a point prediction cannot be
    tested for calibration, and the research contract (§9) accepts only
    calibrated intervals measured against the mandatory baselines.
    """

    value: float
    lower: float
    upper: float
    #: Nominal coverage of [lower, upper], e.g. 0.9.
    level: float

    def __post_init__(self) -> None:
        if not (0.0 < self.level < 1.0):
            raise ValueError(f"interval level must be in (0, 1), got {self.level}")
        if not (math.isnan(self.lower) or math.isnan(self.upper)) and self.lower > self.upper:
            raise ValueError(f"interval lower {self.lower} exceeds upper {self.upper}")


@runtime_checkable
class Predictor(Protocol):
    """Predicts a future-migration quantity from per-observation features.

    ``feature_names`` declares the columns of ``features``, so a
    strict-morphology predictor can be *shown* to use the mask only -- no
    position, velocity or history -- by reading its inputs rather than its
    code.  ``target`` names what is predicted, with its unit.
    """

    target: str
    feature_names: tuple[str, ...]

    def predict(self, features: np.ndarray) -> Sequence[Prediction]: ...
