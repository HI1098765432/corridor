"""Every tunable parameter in one place, with physical units and defaults.

Two rules keep this honest:

*   Parameters are stated in **physical units** (micrometres, minutes) wherever
    a physical meaning exists.  They are converted to pixels and frames exactly
    once, by :class:`Scale`, using the calibration that was actually resolved
    for the dataset being analysed.
*   Every default carries a comment saying where the number comes from.  A
    constant nobody can justify is a bug waiting to be tuned around.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Unit bridge
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scale:
    """Converts between physical units and image units for one dataset.

    When a dataset carries no calibration, ``calibrated`` is False and the
    conversion factors are 1.  Physical parameters are then interpreted as
    pixels and frames directly, and results are reported in pixel units only.
    """

    pixel_size_um: float
    frame_interval_min: float
    calibrated_space: bool
    calibrated_time: bool

    @property
    def calibrated(self) -> bool:
        return self.calibrated_space and self.calibrated_time

    def um_to_px(self, um: float) -> float:
        return um / self.pixel_size_um

    def px_to_um(self, px: float) -> float:
        return px * self.pixel_size_um

    def min_to_frames(self, minutes: float) -> float:
        return minutes / self.frame_interval_min

    def frames_to_min(self, frames: float) -> float:
        return frames * self.frame_interval_min

    @classmethod
    def from_values(
        cls, pixel_size_um: float | None, frame_interval_min: float | None
    ) -> "Scale":
        ok_space = bool(pixel_size_um and math.isfinite(pixel_size_um) and pixel_size_um > 0)
        ok_time = bool(
            frame_interval_min and math.isfinite(frame_interval_min) and frame_interval_min > 0
        )
        return cls(
            pixel_size_um=float(pixel_size_um) if ok_space else 1.0,
            frame_interval_min=float(frame_interval_min) if ok_time else 1.0,
            calibrated_space=ok_space,
            calibrated_time=ok_time,
        )


# --------------------------------------------------------------------------
# Confinement
# --------------------------------------------------------------------------

AXIS_AUTO = "auto"
AXIS_VERTICAL = "vertical"
AXIS_HORIZONTAL = "horizontal"
AXIS_ANGLE = "angle"

AXIS_MODES = (AXIS_AUTO, AXIS_VERTICAL, AXIS_HORIZONTAL, AXIS_ANGLE)


@dataclass
class ConfinementConfig:
    """How the migration axis of the confinement channel is established."""

    #: auto | vertical | horizontal | angle
    mode: str = AXIS_AUTO
    #: Used when mode == "angle". Degrees, measured from the +x image axis,
    #: increasing towards +y (screen-down), matching image coordinates.
    angle_deg: float = 90.0
    #: Detect the bright channel walls of the microfluidic device and use them
    #: to (a) confirm the axis and (b) forbid cross-channel associations.
    detect_walls: bool = True
    #: A field wider than this many channel widths is treated as multi-channel
    #: and requires per-channel association before tracking is considered valid.
    multichannel_warn_ratio: float = 1.6
    #: Two bright ridges closer together than this are the two walls of one
    #: channel, not two channels. 18 um is comfortably wider than the widest
    #: labelled cell in the supplied training data (15 px = 7 um) and far
    #: narrower than the measured device pitch (82 px = 38 um).
    min_channel_pitch_um: float = 18.0
    #: Used when the dataset carries no spatial calibration.
    min_channel_pitch_px: float = 38.0

    def unit_vector(self) -> tuple[float, float] | None:
        """Return the (ux, uy) unit vector, or None when it must be inferred."""
        if self.mode == AXIS_VERTICAL:
            return (0.0, 1.0)
        if self.mode == AXIS_HORIZONTAL:
            return (1.0, 0.0)
        if self.mode == AXIS_ANGLE:
            rad = math.radians(self.angle_deg)
            return (math.cos(rad), math.sin(rad))
        return None


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------


#: Detection fallback ladder.  Each rung adds segmentation passes whose extra
#: instances are merged into the primary result, with the primary pass always
#: authoritative.  The rungs are named after what they cost, and the numbers
#: below are measured on the 71 supplied hand-labelled images
#: (``scripts/experiment_recall.py``, ``docs/recall_experiment.json``):
#:
#:     rung         precision  recall     F1     cost
#:     off              0.832    0.846   0.839     1x
#:     thresholds       0.833    0.854   0.843     2x
#:     wide             0.793    0.874   0.832     4x
#:     models           0.772    0.882   0.824     3x
#:     max_recall       0.703    0.902   0.790     6x
#:
#: Reading that table as "off is nearly best, stop here" would be the wrong
#: conclusion, because F1 scores a single image and this software scores a
#: trajectory.  An extra false positive that appears in one frame and nowhere
#: near the track's predicted position in the next one is refused by the
#: assignment step, so the tracker acts as a temporal precision filter that the
#: per-image F1 cannot see.  A missed cell has no such second chance.  That is
#: why the higher rungs are offered at all -- and why the claim is measured
#: end to end in ``scripts/experiment_fallback.py`` rather than assumed.
#: Named normalisation settings, because "tile_norm_blocksize" is not a thing a
#: microscopist should have to know.  Each maps onto the three fields on
#: SegmentationConfig; the measured effect of each is in docs/ACCURACY.md.
NORM_WHOLE_FRAME = "whole_frame"
NORM_LOCAL = "local"
NORM_STRETCH = "stretch"
NORM_SHARPEN = "sharpen"
NORM_LOCAL_SHARPEN = "local_sharpen"

NORMALISATION_MODES = (
    NORM_WHOLE_FRAME,
    NORM_LOCAL,
    NORM_STRETCH,
    NORM_SHARPEN,
    NORM_LOCAL_SHARPEN,
)

NORMALISATION_LABELS = {
    NORM_WHOLE_FRAME: "Whole frame (default)",
    NORM_LOCAL: "Local contrast",
    NORM_STRETCH: "Stretch contrast",
    NORM_SHARPEN: "Sharpen edges",
    NORM_LOCAL_SHARPEN: "Local contrast and sharpen",
}

#: (percentiles, tile size px, sharpen radius px)
NORMALISATION_PRESETS: dict[str, tuple[tuple[float, float], int, int]] = {
    NORM_WHOLE_FRAME: ((1.0, 99.0), 0, 0),
    NORM_LOCAL: ((1.0, 99.0), 128, 0),
    NORM_STRETCH: ((3.0, 97.0), 0, 0),
    NORM_SHARPEN: ((1.0, 99.0), 0, 15),
    NORM_LOCAL_SHARPEN: ((1.0, 99.0), 128, 15),
}


ENSEMBLE_OFF = "off"
ENSEMBLE_THRESHOLDS = "thresholds"
ENSEMBLE_WIDE = "wide"
ENSEMBLE_MODELS = "models"
ENSEMBLE_MAX_RECALL = "max_recall"

ENSEMBLE_MODES = (
    ENSEMBLE_OFF,
    ENSEMBLE_THRESHOLDS,
    ENSEMBLE_WIDE,
    ENSEMBLE_MODELS,
    ENSEMBLE_MAX_RECALL,
)

#: Human-readable labels, used by the interface so the wording lives in one
#: place rather than being retyped next to a combo box.
ENSEMBLE_LABELS = {
    ENSEMBLE_OFF: "Single pass (fastest)",
    ENSEMBLE_THRESHOLDS: "Two thresholds (recommended)",
    ENSEMBLE_WIDE: "Four thresholds",
    ENSEMBLE_MODELS: "Every available model",
    ENSEMBLE_MAX_RECALL: "Every model, both thresholds (slowest)",
}

#: The extra (cellprob_threshold, flow_threshold) passes each rung adds *on top
#: of* the configured primary pass.
_ENSEMBLE_SETTINGS: dict[str, tuple[tuple[float, float], ...]] = {
    ENSEMBLE_OFF: (),
    ENSEMBLE_THRESHOLDS: ((-2.0, 0.4),),
    ENSEMBLE_WIDE: ((-2.0, 0.4), (0.0, 0.6), (-2.0, 0.6)),
    ENSEMBLE_MODELS: (),
    ENSEMBLE_MAX_RECALL: ((-2.0, 0.6),),
}

#: Which rungs also run the companion models, when they have been located.
_ENSEMBLE_USES_MODELS = frozenset({ENSEMBLE_MODELS, ENSEMBLE_MAX_RECALL})


@dataclass
class SegmentationConfig:
    """Cellpose v3 settings and the post-processing filter."""

    model_path: str | None = None
    builtin_model: str = "cyto3"
    use_custom_model: bool = True

    #: None lets Cellpose use the value baked into the custom model
    #: (diam_labels = 31.90 px for the supplied KK1+KK2 model).
    diameter: float | None = None
    cellprob_threshold: float = 0.0  # Cellpose default; the notebook's value.
    flow_threshold: float = 0.4  # Cellpose default; the notebook's value.
    channels: tuple[int, int] = (0, 0)  # grayscale phase contrast
    normalize: bool = True
    use_gpu: bool = False

    # -- how the image is normalised before the network sees it --------------
    # These matter more than they look. Measured across the two halves of the
    # supplied training data, the cells are the *same size* (12.1 vs 12.2 px
    # wide) but sit at very different contrast: 0.219 of the image range above
    # background in one half, 0.378 in the other. That difference, not shape
    # and not scale, is what a model trained on one half meets in the other.
    # See docs/ACCURACY.md and scripts/experiment_generalisation.py.
    #
    #: The percentile window mapped to [0, 1]. Cellpose's default is (1, 99).
    #: Narrowing it stretches contrast, which helps a faint image and clips a
    #: bright one -- measured, not assumed: below about (10, 90) it destroys
    #: detections rather than adding them.
    normalize_percentiles: tuple[float, float] = (1.0, 99.0)
    #: Normalise in tiles of this many pixels instead of over the whole frame.
    #: 0 uses the whole frame. Local normalisation is the setting that survives
    #: a contrast shift, because it rescales each region to its own background
    #: rather than to a global range one bright structure can dominate.
    normalize_tile_px: int = 0
    #: Unsharp-mask radius applied before normalising. 0 disables.
    normalize_sharpen_px: int = 0

    @property
    def normalisation_mode(self) -> str:
        """Which named preset the current three fields correspond to.

        Reported rather than stored, so that a configuration edited field by
        field -- from a saved project, or from the command line -- still names
        itself correctly instead of claiming to be whichever preset was last
        clicked.  A combination matching no preset reports ``custom``.
        """
        current = (
            tuple(float(p) for p in self.normalize_percentiles),
            int(self.normalize_tile_px),
            int(self.normalize_sharpen_px),
        )
        for name, preset in NORMALISATION_PRESETS.items():
            if (tuple(float(p) for p in preset[0]), preset[1], preset[2]) == current:
                return name
        return "custom"

    def apply_normalisation_preset(self, mode: str) -> None:
        """Set the three fields from a named preset, ignoring an unknown name."""
        preset = NORMALISATION_PRESETS.get(mode)
        if preset is None:
            return
        percentiles, tile, sharpen = preset
        self.normalize_percentiles = percentiles
        self.normalize_tile_px = tile
        self.normalize_sharpen_px = sharpen

    def normalize_argument(self) -> bool | dict[str, Any]:
        """Build the value Cellpose's ``normalize=`` parameter expects.

        Returns a plain ``True``/``False`` when nothing has been changed from
        Cellpose's own defaults, so that a default run is byte-identical to one
        made before these options existed and results stay comparable.
        """
        if not self.normalize:
            return False
        options: dict[str, Any] = {}
        if tuple(self.normalize_percentiles) != (1.0, 99.0):
            options["percentile"] = tuple(float(p) for p in self.normalize_percentiles)
        if self.normalize_tile_px > 0:
            options["tile_norm_blocksize"] = int(self.normalize_tile_px)
        if self.normalize_sharpen_px > 0:
            options["sharpen_radius"] = int(self.normalize_sharpen_px)
        return options or True

    #: Post-filter: discard instances whose bounding box is smaller than this
    #: in its larger dimension.  The research notebook hard-coded 20 px.
    #: Kept as the default so results stay comparable, but now measurable.
    min_extent_px: int = 20
    #: Secondary guard against single-pixel specks. 0 disables.
    min_area_px: int = 0
    #: Drop instances touching the image border (they are partly outside the
    #: field, so their centroid and area are biased). Off by default because
    #: cells entering a channel legitimately touch the border.
    drop_border_touching: bool = False

    # -- detection fallback --------------------------------------------------
    #: Which rung of the fallback ladder to use.  See ``ENSEMBLE_MODES`` above
    #: for the measured precision/recall of each.
    ensemble: str = ENSEMBLE_OFF
    #: Additional Cellpose model files to run when the chosen rung uses them.
    #: These are the sibling models shipped beside the combined one; when the
    #: list is empty a model-based rung quietly degrades to its threshold
    #: passes rather than failing, because a missing optional model is not a
    #: reason to refuse an analysis.
    ensemble_model_paths: tuple[str, ...] = ()
    #: A candidate instance from a later pass is discarded when more than this
    #: fraction of it is already claimed by an accepted instance.  0.3 means
    #: "if a third of you is already somebody else, you are that somebody".
    ensemble_merge_overlap: float = 0.3
    #: After subtracting the claimed part, a surviving sliver smaller than this
    #: is not a cell.  Matches the smallest plausible instance area.
    ensemble_min_fragment_px: int = 20

    def resolved_model(self) -> str:
        return self.model_path if (self.use_custom_model and self.model_path) else self.builtin_model

    def ensemble_passes(self) -> tuple[tuple[str | None, float, float], ...]:
        """The extra passes this configuration asks for.

        Each entry is ``(model_path_or_None, cellprob_threshold,
        flow_threshold)``; ``None`` means the primary model.  The primary pass
        itself is not included -- it always runs, and it always wins a merge.
        """
        mode = self.ensemble if self.ensemble in ENSEMBLE_MODES else ENSEMBLE_OFF
        if mode == ENSEMBLE_OFF:
            return ()

        settings = _ENSEMBLE_SETTINGS[mode]
        passes: list[tuple[str | None, float, float]] = [
            (None, cellprob, flow) for cellprob, flow in settings
        ]
        if mode in _ENSEMBLE_USES_MODELS:
            for path in self.ensemble_model_paths:
                passes.append((path, self.cellprob_threshold, self.flow_threshold))
                for cellprob, flow in settings:
                    passes.append((path, cellprob, flow))
        return tuple(passes)

    def ensemble_cost_factor(self) -> int:
        """How many segmentation passes per frame this configuration asks for.

        This is what was *requested*. What actually ran is reported by
        ``SegmentationOutput.passes_per_frame``, and the two differ whenever a
        companion model could not be loaded -- which is the normal case in the
        installed application, because it ships only the combined model.
        """
        return 1 + len(self.ensemble_passes())

    def uses_companion_models(self) -> bool:
        """Whether the chosen rung wants models other than the primary one."""
        mode = self.ensemble if self.ensemble in ENSEMBLE_MODES else ENSEMBLE_OFF
        return mode in _ENSEMBLE_USES_MODELS


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------


@dataclass
class TrackingConfig:
    """The confinement-aware assignment model.

    All costs are expressed in chi-square units: each term is a squared
    residual divided by the variance it is allowed to have.  A term equal to
    1.0 means "one standard deviation off".  That makes every threshold below
    readable as a number of sigmas rather than an arbitrary weight.
    """

    # -- hard physical gates -------------------------------------------------
    #: Nothing may move faster than this. The research notebook's 200 px per
    #: 20.01 min frame at 0.4671 um/px is 4.67 um/min; 5.0 keeps that intent
    #: while making it physical and gap-aware.
    max_speed_um_per_min: float = 5.0
    #: A cell in a channel may not jump sideways further than this within one
    #: frame interval. 4.0 um is roughly a tenth of the 42 um channel width.
    max_perp_um: float = 4.0
    #: Reject matches whose area changes by more than this factor either way.
    area_ratio_min: float = 0.3
    area_ratio_max: float = 3.0

    # -- soft cost scales ----------------------------------------------------
    #: Expected 1-sigma error of the constant-velocity prediction *along* the
    #: channel for a cell that is not moving, per sqrt(frame).
    sigma_along_um: float = 3.0
    #: Confined cells stall and surge: between two frames a cell's speed can
    #: change by a large fraction of itself. This is that fraction, and it is
    #: what stops a cell that brakes hard from being read as a different cell.
    #: Without it the prediction error of a fast cell that stops is as large as
    #: its own previous step.
    speed_uncertainty_fraction: float = 0.7
    #: Expected 1-sigma error *across* the channel, from genuine lateral
    #: wander of the cell. Small on purpose: this is the confinement prior.
    #: The ratio sigma_along/sigma_perp replaces the notebook's unexplained
    #: w_dir = 1000 with something readable.
    sigma_perp_um: float = 0.8
    #: The centroid of a long thin cell slides sideways whenever its mask
    #: gains or loses a tail, by a fraction of the cell's own width. This adds
    #: that measurement noise to the lateral tolerance, so the prior does not
    #: punish a cell for being segmented slightly differently.
    perp_width_fraction: float = 0.35
    #: Hard lateral gate, as a multiple of the cell's own width. Beyond this a
    #: step is a jump between objects, not a wobble of one object.
    max_perp_widths: float = 1.6
    #: 1-sigma of log(area ratio) between consecutive observations of a cell.
    #: ln(1.35) ~ 0.30 allows routine shape change without penalty.
    sigma_ln_area: float = 0.30

    #: Cost added for a full reversal of direction along the channel, in
    #: chi-square units (4.0 == a 2-sigma event). Applied only once a track
    #: has an established direction and the step exceeds the noise floor.
    w_reversal: float = 4.0
    #: Cost for a complete mismatch of cell body orientation, in chi-square
    #: units. Deliberately small: morphology is a hint, not evidence.
    w_orientation: float = 1.0
    #: Below this eccentricity a cell is too round for its orientation to mean
    #: anything, and the orientation term is skipped entirely.
    orientation_min_eccentricity: float = 0.6
    #: Steps shorter than this are dominated by centroid noise, so the
    #: direction term is skipped.
    direction_noise_floor_um: float = 1.0

    # -- assignment ----------------------------------------------------------
    #: The cost of leaving a track unmatched, in chi-square units. A detection
    #: is only accepted if it fits better than this, i.e. within ~3.9 sigma of
    #: the two-dimensional prediction. Real, not decorative: it forms the
    #: dummy blocks of the assignment matrix.
    unmatched_chi2: float = 15.0
    #: Hard rejection ceiling. Pairs above this can never be matched.
    gate_chi2: float = 30.0

    # -- lifecycle -----------------------------------------------------------
    #: A track may be absent for at most this many consecutive frames and
    #: still be reacquired; i.e. the largest allowed gap between successive
    #: observations is max_gap + 1 frames.
    max_gap: int = 3
    #: Tracks with fewer observations than this are reported but flagged as
    #: fragments rather than trajectories.
    min_observations: int = 2

    # -- multi-channel -------------------------------------------------------
    #: Forbid associations between detections assigned to different channels.
    enforce_channel_identity: bool = True

    def max_delta_frames(self) -> int:
        """Largest allowed frame separation between successive observations."""
        return int(self.max_gap) + 1


# --------------------------------------------------------------------------
# Calibration overrides
# --------------------------------------------------------------------------


@dataclass
class CalibrationConfig:
    """User overrides. ``None`` means 'use whatever the file says'."""

    pixel_size_um: float | None = None
    frame_interval_min: float | None = None


# --------------------------------------------------------------------------
# Whole run
# --------------------------------------------------------------------------


@dataclass
class RunConfig:
    input_path: str = ""
    output_dir: str = ""
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    confinement: ConfinementConfig = field(default_factory=ConfinementConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    recovery: "RecoveryConfig" = field(default_factory=lambda: _recovery_default())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        return _from_dict(cls, data or {})


def _from_dict(cls, data: dict[str, Any]):
    """Tolerant reconstruction: unknown keys are ignored, missing keys default.

    Tolerance matters because a project saved by version N must still open in
    version N+1 after a parameter is added or renamed.
    """
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        origin = f.type
        if hasattr(origin, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[f.name] = _from_dict(origin, value)
        elif isinstance(value, list) and f.name == "channels":
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    # Nested dataclasses referenced by string annotations need explicit handling.
    from .recovery import RecoveryConfig

    for name, sub in (
        ("segmentation", SegmentationConfig),
        ("tracking", TrackingConfig),
        ("confinement", ConfinementConfig),
        ("calibration", CalibrationConfig),
        ("recovery", RecoveryConfig),
    ):
        if name in kwargs and isinstance(kwargs[name], dict):
            kwargs[name] = _from_dict(sub, kwargs[name])
    return cls(**kwargs)


def _recovery_default():
    # Imported lazily: recovery depends on tracking, which depends on config.
    from .recovery import RecoveryConfig

    return RecoveryConfig()


def default_model_path() -> Path | None:
    """Locate the bundled custom Cellpose model, if one shipped with the app."""
    from corridor import resources

    return resources.bundled_model_path()
