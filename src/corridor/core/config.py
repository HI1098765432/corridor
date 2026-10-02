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

import dataclasses
import functools
import math
import types
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

# --------------------------------------------------------------------------
# Unit bridge
# --------------------------------------------------------------------------


def _positive(value: float | None) -> bool:
    return bool(value and math.isfinite(value) and value > 0)


@dataclass(frozen=True)
class Scale:
    """Converts between physical units and image units for one dataset.

    When a dataset carries no calibration, ``calibrated`` is False and the
    conversion factors are 1.  Physical parameters are then interpreted as
    pixels and frames directly, and results are reported in pixel units only.

    Z is different on purpose.  An unknown pixel size falls back to "1 unit
    per pixel" because every XY quantity is still self-consistent in pixels;
    an unknown Z step has no such fallback, because a slice is not a pixel.
    Optical stacks are routinely sampled more coarsely in Z than in XY, so
    ``z_step_um`` stays ``None`` and every quantity that needs it (µm³, µm²,
    ``anisotropy``) stays empty rather than assuming isotropy.
    """

    pixel_size_um: float
    frame_interval_min: float
    calibrated_space: bool
    calibrated_time: bool
    #: Distance between Z planes. None for a 2-D movie and for a stack whose
    #: Z step is unknown.
    z_step_um: float | None = None
    calibrated_z: bool = False

    @property
    def calibrated(self) -> bool:
        return self.calibrated_space and self.calibrated_time

    @property
    def anisotropy(self) -> float | None:
        """How many XY pixels one Z step spans (``z_step_um / pixel_size_um``).

        Positions are tracked as ``(x, y, z * anisotropy)`` so that a distance
        means the same thing along every axis.  None unless *both* the Z step
        and the pixel size are calibrated: a Z step in µm divided by a
        placeholder pixel size of 1.0 is a number with no meaning, and callers
        must not invent one.
        """
        if not (self.calibrated_z and self.calibrated_space and self.z_step_um):
            return None
        return float(self.z_step_um) / self.pixel_size_um

    @property
    def spacing_zyx_um(self) -> tuple[float, float, float] | None:
        """Voxel spacing for 3-D measurement, or None when it is not known."""
        if self.anisotropy is None:
            return None
        return (float(self.z_step_um), self.pixel_size_um, self.pixel_size_um)

    def um_to_px(self, um: float) -> float:
        return um / self.pixel_size_um

    def px_to_um(self, px: float) -> float:
        return px * self.pixel_size_um

    def min_to_frames(self, minutes: float) -> float:
        return minutes / self.frame_interval_min

    def frames_to_min(self, frames: float) -> float:
        return frames * self.frame_interval_min

    def frames_to_hr(self, frames: float) -> float:
        """Elapsed hours, from the same minutes every other time unit uses."""
        return frames * self.frame_interval_min / 60.0

    @classmethod
    def from_values(
        cls,
        pixel_size_um: float | None,
        frame_interval_min: float | None,
        z_step_um: float | None = None,
    ) -> "Scale":
        ok_space = _positive(pixel_size_um)
        ok_time = _positive(frame_interval_min)
        ok_z = _positive(z_step_um)
        return cls(
            pixel_size_um=float(pixel_size_um) if ok_space else 1.0,
            frame_interval_min=float(frame_interval_min) if ok_time else 1.0,
            calibrated_space=ok_space,
            calibrated_time=ok_time,
            z_step_um=float(z_step_um) if ok_z else None,
            calibrated_z=ok_z,
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
    """How the migration axis of the confinement channel is established.

    Legacy v1. Replaced by :class:`GeometryConfig` (there is no axis to
    configure in 2.0); kept while the pipeline, UI and v1 manifests still read
    it, and removed when nothing does.
    """

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


@dataclass
class GeometryConfig:
    """Where the device walls are -- never which way the cells go.

    The wall-ridge detector finds lanes (each with its own centre line and
    half-width) and nothing else; with
    ``TrackingConfig.channel_constraint == "auto"`` a lane gate is applied
    only when lanes really were detected from walls.
    """

    #: Detect the bright channel walls of the microfluidic device.
    detect_walls: bool = True
    #: Two bright ridges closer together than this are the two walls of one
    #: channel, not two channels. 18 um is comfortably wider than the widest
    #: labelled cell in the supplied training data (15 px = 7 um) and far
    #: narrower than the measured device pitch (82 px = 38 um).
    min_channel_pitch_um: float = 18.0
    #: Used when the dataset carries no spatial calibration.
    min_channel_pitch_px: float = 38.0


#: The ``confinement`` keys that carry over into ``geometry``. ``mode``,
#: ``angle_deg`` and ``multichannel_warn_ratio`` describe an axis and are
#: dropped.
_GEOMETRY_FROM_CONFINEMENT = ("detect_walls", "min_channel_pitch_um", "min_channel_pitch_px")


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
    """Cellpose v3 settings and the post-processing filter.

    The model-selection fields (``model_path``, ``builtin_model``,
    ``use_custom_model``, ``ensemble_model_paths`` and ``resolved_model()``)
    are legacy v1.  In 2.0 the model is not a setting: production resolves
    exactly one SHA-256-verified file through
    :mod:`corridor.core.model_registry`, and never falls back to ``cyto3`` or
    any other built-in.  They stay only until the segmentation service stops
    reading them; :meth:`RunConfig.for_new_project` already refuses to carry
    them from one project into the next.
    """

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


#: ``TrackingConfig.channel_constraint`` values. ``auto`` applies the lane gate
#: only when lanes were detected from walls; ``off`` never applies it.
CHANNEL_CONSTRAINT_AUTO = "auto"
CHANNEL_CONSTRAINT_OFF = "off"
CHANNEL_CONSTRAINTS = (CHANNEL_CONSTRAINT_AUTO, CHANNEL_CONSTRAINT_OFF)


@dataclass
class TrackingConfig:
    """The assignment model, v1 (axis) and v2 (axis-free Kalman) side by side.

    All costs are expressed in chi-square units: each term is a squared
    residual divided by the variance it is allowed to have.  A term equal to
    1.0 means "one standard deviation off".  That makes every threshold below
    readable as a number of sigmas rather than an arbitrary weight.

    Fields marked *legacy v1* belong to the migration-axis model and are read
    only by the v1 tracker; they are removed when nothing reads them.  Fields
    without that mark are shared by both, or are new in v2 (the block at the
    end).  ``gate_chi2`` is legacy: v2 uses :attr:`effective_gate_chi2`,
    derived from ``unmatched_chi2`` so the ``gate == 2U`` invariant cannot
    drift.

    Where the v2 defaults come from: the frozen v1.3.0 baseline of the five
    supplied sample stacks (``build/baseline_v1.3.0``; KK2 instrument,
    0.467 µm/px, 20.0 min frames), measured when these fields were added --
    155 detections with median major/minor axes 48.2/4.7 µm (100/9.8 px), and
    128 linked steps with step speeds of median 0.32, p90 1.19 and maximum
    1.98 µm/min.  The second differences ``r[t+2] - 2 r[t+1] + r[t]`` of 87
    pairs of consecutive one-frame triples, projected on each cell's own
    major and minor axes, separate centroid noise from real acceleration
    (``Var = q + 6 sm^2`` and lag-1 ``Cov = -4 sm^2``): along the body
    sm = 2.58 µm and sqrt(q) = 7.96 µm per frame squared (0.40 µm/min of
    velocity change per 20 min frame); across it sm = 0.72 µm and 0.043
    µm/min.  That is one experiment and one instrument, from tracks the v1
    gates accepted, so these are starting values for the tracker's own tests
    to tune, not fitted constants.
    """

    # -- hard physical gates -------------------------------------------------
    #: Nothing may move faster than this. The research notebook's 200 px per
    #: 20.01 min frame at 0.4671 um/px is 4.67 um/min; 5.0 keeps that intent
    #: while making it physical and gap-aware. The fastest baseline step is
    #: 1.98 um/min, so the gate refuses only what no observed cell did.
    max_speed_um_per_min: float = 5.0
    #: Legacy v1. A cell in a channel may not jump sideways further than this
    #: within one frame interval. 4.0 um is roughly a tenth of the 42 um
    #: channel width.
    max_perp_um: float = 4.0
    #: Reject matches whose area changes by more than this factor either way.
    #: In 3-D the same ratio applies to volume.
    area_ratio_min: float = 0.3
    area_ratio_max: float = 3.0

    # -- soft cost scales ----------------------------------------------------
    #: Legacy v1. Expected 1-sigma error of the constant-velocity prediction
    #: *along* the channel for a cell that is not moving, per sqrt(frame).
    sigma_along_um: float = 3.0
    #: Legacy v1. Confined cells stall and surge: between two frames a cell's
    #: speed can change by a large fraction of itself. This is that fraction,
    #: and it is what stops a cell that brakes hard from being read as a
    #: different cell. Without it the prediction error of a fast cell that
    #: stops is as large as its own previous step.
    speed_uncertainty_fraction: float = 0.7
    #: Legacy v1. Expected 1-sigma error *across* the channel, from genuine
    #: lateral wander of the cell. Small on purpose: this is the confinement
    #: prior. The ratio sigma_along/sigma_perp replaces the notebook's
    #: unexplained w_dir = 1000 with something readable.
    sigma_perp_um: float = 0.8
    #: Legacy v1. The centroid of a long thin cell slides sideways whenever its
    #: mask gains or loses a tail, by a fraction of the cell's own width. This
    #: adds that measurement noise to the lateral tolerance, so the prior does
    #: not punish a cell for being segmented slightly differently.
    perp_width_fraction: float = 0.35
    #: Legacy v1. Hard lateral gate, as a multiple of the cell's own width.
    #: Beyond this a step is a jump between objects, not a wobble of one object.
    max_perp_widths: float = 1.6
    #: 1-sigma of log(area ratio) between consecutive observations of a cell.
    #: ln(1.35) ~ 0.30 allows routine shape change without penalty.
    sigma_ln_area: float = 0.30

    #: Cost added for a full reversal of direction, in chi-square units
    #: (4.0 == a 2-sigma event). Applied only once a track has an established
    #: direction and the step exceeds the noise floor. v1 measured "direction"
    #: along the channel axis; v2 measures it against the track's own velocity.
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
    #: Legacy v1. Hard rejection ceiling. Pairs above this can never be
    #: matched. Stored independently of ``unmatched_chi2``, which is how the
    #: GUI's max-ratchet and the CLI that ignored it let the two drift apart;
    #: v2 reads :attr:`effective_gate_chi2` instead.
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
    #: Legacy v1. Forbid associations between detections assigned to different
    #: channels. Superseded by ``channel_constraint``; a saved ``False`` loads
    #: as ``channel_constraint="off"``.
    enforce_channel_identity: bool = True

    # -- v2: axis-free Kalman model (contract §5) -----------------------------
    # Measurement noise is shaped by each cell's own body:
    #   R = position_sigma^2 I + (shape_position_fraction * major)^2 u u^T
    #       + (width_position_fraction * minor)^2 n n^T
    # with u, n the detection's major and minor directions. A long thin cell
    # has an uncertain centroid along its length and a precise one across it,
    # which is the physical reason the v1 along/across model worked.

    #: Isotropic floor of the centroid noise, whatever the cell's shape. With
    #: the width term below it reproduces the measured across-body noise
    #: (sqrt(0.5^2 + (0.12 * 4.7)^2) = 0.75 um against 0.72 measured), and it
    #: is about one pixel of the 0.467 um/px instrument.
    position_sigma_um: float = 0.5
    #: Centroid noise along the body, as a fraction of the major-axis length:
    #: a mask that gains or loses a tail moves its centroid along the cell.
    #: 0.06 x 48.2 um = 2.89 um (2.94 um with the floor) against the measured
    #: 2.58 um, rounded up because a tracker that trusts a centroid too much
    #: splits tracks, while one that trusts it too little only links slower.
    shape_position_fraction: float = 0.06
    #: Centroid noise across the body, as a fraction of the minor-axis length.
    #: Measured 0.72 um across bodies of median width 4.7 um; after the 0.5 um
    #: floor that leaves 0.52 um, 0.11 of the width, rounded up. v1's
    #: ``perp_width_fraction`` plays the same role, but its 0.35 cites no
    #: measurement, so it is not carried over.
    width_position_fraction: float = 0.12
    #: White-noise-acceleration process noise: the 1-sigma change of each
    #: velocity component over one frame interval, so the predicted position
    #: widens with every frame of a gap. 0.40 um/min is the along-body value
    #: measured on 20 min frames (0.043 um/min across). One isotropic value
    #: has to carry the larger, because underestimating it is what splits a
    #: cell that brakes hard -- the problem v1's speed_uncertainty_fraction
    #: existed to solve. Per frame, not per minute: on 10 min frames the same
    #: number allows twice the velocity diffusion per minute, which the
    #: tracker package must either accept or rescale.
    velocity_sigma_um_per_min: float = 0.4
    #: Speed uncertainty of a track seen once, whose velocity is unknown and
    #: starts at zero. None means ``max_speed_um_per_min / 3`` (a "3-sigma"
    #: bound): 1.67 um/min at the default, above the p90 (1.19 um/min) of the
    #: baseline step speeds, and equal to v1's fresh-track spread.
    initial_speed_sigma_um_per_min: float | None = None
    #: Weight of the shape term, (dln aspect / 0.35)^2 + (dsolidity / 0.10)^2.
    #: 0.5 makes a one-sigma change of both cost one chi-square unit:
    #: morphology is a hint, not evidence (as for w_orientation).
    w_shape: float = 0.5
    #: Weight of 1 - IoU(previous mask shifted by the prediction, candidate).
    #: Overlap is largely the same evidence as the motion term -- both measure
    #: displacement from the prediction -- so a large weight would count it
    #: twice. 1.0 lets it break ties between cells whose centroids fit equally
    #: well but whose bodies do not overlap.
    w_overlap: float = 1.0
    #: Cost per missed frame, (dt - 1) * this. At equal motion fit a direct
    #: link is preferred to one across a gap; at max_gap = 3 the largest
    #: penalty, 3.0, is a fifth of the unmatched cost.
    gap_penalty_chi2: float = 1.0
    #: Run stage 2: match every track end against every later track start in
    #: one global assignment, using evidence from both sides of the gap.
    global_gap_closing: bool = True
    #: "auto" applies the lane gate only when lanes were detected from walls;
    #: "off" never applies it. See CHANNEL_CONSTRAINTS.
    channel_constraint: str = CHANNEL_CONSTRAINT_AUTO

    def max_delta_frames(self) -> int:
        """Largest allowed frame separation between successive observations."""
        return int(self.max_gap) + 1

    @property
    def effective_gate_chi2(self) -> float:
        """The v2 hard rejection ceiling, always ``2 * unmatched_chi2``.

        Derived, never stored: the link margin is capped at ``2U`` and the
        unlinked-start audit compares with the gate, so the two must move
        together.
        """
        return 2.0 * float(self.unmatched_chi2)

    @property
    def effective_initial_speed_sigma_um_per_min(self) -> float:
        """``initial_speed_sigma_um_per_min``, resolving None to max_speed / 3."""
        if self.initial_speed_sigma_um_per_min is not None:
            return float(self.initial_speed_sigma_um_per_min)
        return float(self.max_speed_um_per_min) / 3.0


# --------------------------------------------------------------------------
# Calibration overrides
# --------------------------------------------------------------------------


@dataclass
class CalibrationConfig:
    """User overrides. ``None`` means 'use whatever the file says'."""

    pixel_size_um: float | None = None
    frame_interval_min: float | None = None
    #: Distance between Z planes, for a stack whose file does not record it.
    #: Without one, 3-D results are reported in voxels only.
    z_step_um: float | None = None


# --------------------------------------------------------------------------
# Measurement and import (contract §3, §4, §6)
# --------------------------------------------------------------------------


@dataclass
class MeasurementConfig:
    """Per-observation and per-track measurement settings."""

    #: Reference point for the distance-to-reference column (MTrackJ's D2R),
    #: as (x, y) or (x, y, z) in pixel/slice units -- the same frame as
    #: ``Detection.position``. None leaves that column empty.
    reference_point_px: tuple[float, ...] | None = None
    #: A lag enters the MSD fit only if this many observation pairs are
    #: separated by it. Fewer pairs make a point whose own error dominates
    #: the fit.
    msd_min_pairs: int = 3
    #: ``msd_alpha`` is reported only when at least this many lags qualify;
    #: a power law through two points is a line through two points.
    msd_min_lags_for_fit: int = 3
    #: Fit only lags up to this fraction of the track's span. The longest lags
    #: are averaged over the fewest pairs and are the least reliable.
    msd_max_lag_fraction: float = 0.5


@dataclass
class ImportConfig:
    """How to read a file whose dimensions or contents need saying explicitly."""

    #: Explicit axis order (e.g. "TYX", "ZYX", "TZCYX") for a file whose
    #: metadata cannot establish it. None trusts the metadata, and ambiguous
    #: metadata is refused rather than guessed -- a Z axis read as time is a
    #: wrong answer, not a degraded one.
    axes: str | None = None
    #: The channel to analyse in a multichannel file. Recorded in run.json.
    channel_index: int = 0
    #: A label image to measure and track instead of segmenting (the only
    #: route for 3-D data, since no 3-D segmentation model is validated).
    labels_path: str | None = None


# --------------------------------------------------------------------------
# Whole run
# --------------------------------------------------------------------------


@dataclass
class RunConfig:
    input_path: str = ""
    output_dir: str = ""
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    #: Legacy v1, superseded by ``geometry``. While both exist, a dict that
    #: carries only one of the two fills the other's shared keys, so legacy
    #: readers and v2 readers see the same walls setting.
    confinement: ConfinementConfig = field(default_factory=ConfinementConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    recovery: "RecoveryConfig" = field(default_factory=lambda: _recovery_default())
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    #: Serialised under the key "import" (a keyword in Python, not in JSON).
    import_: ImportConfig = field(default_factory=ImportConfig)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["import"] = data.pop("import_")
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        return _from_dict(cls, _upgrade_legacy(data or {}))

    @classmethod
    def for_new_project(cls, saved: dict[str, Any] | None) -> "RunConfig":
        """A configuration for a *new* project, seeded from saved preferences.

        ``saved`` is typically the ``default_config`` the application stores
        after each run.  Tuning carries over; anything that describes one
        particular file or one particular model does not:

        *   ``input_path`` and ``output_dir`` -- they name the last dataset.
        *   every calibration override.  KK1 and KK2 are 0.639 and
            0.467 µm/px; an override typed for one and silently applied to
            the other is a wrong answer with nothing on screen to show it.
        *   model selection (``model_path``, ``builtin_model``,
            ``use_custom_model``, ``ensemble_model_paths``).  The model is
            resolved and hash-verified per run, never inherited.
        *   the whole ``import`` block (axis order, channel index, label
            image) and ``measurement.reference_point_px``: each is a fact
            about one file's layout or one field of view.

        A project's *own* saved configuration must still be loaded with
        :meth:`from_dict`; reopening a project is not starting one.
        """
        config = cls.from_dict(saved or {})
        seg_defaults = SegmentationConfig()
        config.input_path = ""
        config.output_dir = ""
        config.calibration = CalibrationConfig()
        config.segmentation.model_path = seg_defaults.model_path
        config.segmentation.builtin_model = seg_defaults.builtin_model
        config.segmentation.use_custom_model = seg_defaults.use_custom_model
        config.segmentation.ensemble_model_paths = seg_defaults.ensemble_model_paths
        config.import_ = ImportConfig()
        config.measurement.reference_point_px = None
        return config


def _upgrade_legacy(data: dict[str, Any]) -> dict[str, Any]:
    """Map v1 keys onto v2 ones without touching the caller's dict.

    *   ``import`` -> ``import_`` (the field name).
    *   ``confinement`` -> ``geometry``: ``detect_walls`` and the two pitches
        carry over, ``mode`` and ``angle_deg`` are dropped (there is no axis to
        configure).  The reverse fill keeps the legacy pipeline, which still
        reads ``confinement``, in step with a dict saved with only
        ``geometry``.
    *   ``tracking.enforce_channel_identity = False`` -> ``channel_constraint
        = "off"``, unless the dict already says which constraint it wants.
    """
    data = dict(data)
    if "import" in data and "import_" not in data:
        data["import_"] = data.pop("import")

    confinement = data.get("confinement")
    geometry = data.get("geometry")
    if isinstance(confinement, dict) and not isinstance(geometry, dict):
        data["geometry"] = {
            k: confinement[k] for k in _GEOMETRY_FROM_CONFINEMENT if k in confinement
        }
    elif isinstance(geometry, dict) and not isinstance(confinement, dict):
        data["confinement"] = {
            k: geometry[k] for k in _GEOMETRY_FROM_CONFINEMENT if k in geometry
        }

    tracking = data.get("tracking")
    if (
        isinstance(tracking, dict)
        and "channel_constraint" not in tracking
        and tracking.get("enforce_channel_identity") is False
    ):
        data["tracking"] = {**tracking, "channel_constraint": CHANNEL_CONSTRAINT_OFF}
    return data


@functools.lru_cache(maxsize=None)
def _field_types(cls) -> dict[str, Any]:
    """Resolved annotations of a config dataclass.

    Under ``from __future__ import annotations`` every ``Field.type`` is a
    *string*, so a test like ``hasattr(f.type, "__dataclass_fields__")`` is
    never true.  That is how v1 came to restore nested dataclasses only
    through a hand-written list, and tuples (``normalize_percentiles``,
    ``ensemble_model_paths``) as lists.  ``RecoveryConfig`` is supplied here
    because ``config`` cannot import ``recovery`` at module level (recovery
    depends on tracking, which depends on config).
    """
    from .recovery import RecoveryConfig

    return get_type_hints(cls, localns={"RecoveryConfig": RecoveryConfig})


def _coerce(hint: Any, value: Any) -> Any:
    """Turn a JSON value back into what the annotation asks for.

    Only the shapes JSON destroys are rebuilt: a dict back into a nested
    config dataclass, a list back into a tuple.  Scalars pass through
    untouched, so a saved value of an unexpected type is kept as saved rather
    than silently replaced by a default.
    """
    if value is None:
        return None
    origin = get_origin(hint)
    if origin is Union or origin is types.UnionType:
        for arg in get_args(hint):
            if arg is type(None):
                continue
            coerced = _coerce(arg, value)
            if coerced is not value:
                return coerced
        return value
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        return _from_dict(hint, value) if isinstance(value, dict) else value
    if (origin is tuple or hint is tuple) and isinstance(value, list):
        return tuple(value)
    return value


def _from_dict(cls, data: dict[str, Any]):
    """Tolerant reconstruction: unknown keys are ignored, missing keys default.

    Tolerance matters because a project saved by version N must still open in
    version N+1 after a parameter is added or renamed.
    """
    hints = _field_types(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if not f.init or f.name not in data:
            continue
        kwargs[f.name] = _coerce(hints.get(f.name, Any), data[f.name])
    return cls(**kwargs)


def _recovery_default():
    # Imported lazily: recovery depends on tracking, which depends on config.
    from .recovery import RecoveryConfig

    return RecoveryConfig()


def default_model_path() -> Path | None:
    """Locate the bundled custom Cellpose model, if one shipped with the app."""
    from corridor import resources

    return resources.bundled_model_path()
