"""End-to-end analysis: metadata -> segmentation -> lanes -> tracking -> measurement -> disk.

Three ordering rules are structural, not stylistic:

*   Results are written **before** any viewer opens.  Visualisation is a way to
    check a result, never a precondition for the result existing.
*   Each stage persists as soon as it finishes.  If tracking fails, the
    segmentation that took minutes is still on disk and still valid.
*   ``run.json`` is written **last**, and a stale one is removed first.  It is
    what marks a directory as a complete result
    (``store.project.analysis_is_complete``), so a run interrupted half-way
    through rewriting an old directory must not leave the old marker beside
    new, partial CSVs.

Nothing here receives or infers a migration direction (contract §5).  The
device contributes lanes (``geometry.detect_channels``), and the lane gate is
applied only when the lanes were measured from walls.
"""

from __future__ import annotations

import inspect
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from .. import app_meta
from . import export, qc, recovery as recovery_mod
from .config import RECONSTRUCTOR_OVERLAP, CalibrationConfig, RunConfig, Scale
from .detections import SOURCE_ENSEMBLE, Detection, detections_to_rows
from .geometry import ChannelGeometry, assign_lanes, detect_channels, static_projection
from .imaging import (
    SOURCE_USER,
    Calibrated,
    StackMetadata,
    effective_z_step,
    load_stack,
    read_metadata,
)
from .measurements import TrackSummary, frame_rows, msd_rows, summarise
from .model_registry import ResolvedModel
from .recovery import RecoveryResult
from .segmentation import (
    PROVENANCE_IMPORTED,
    PROVENANCE_MODEL,
    SegmentationOutput,
    SegmentationService,
    gpu_available,
    load_label_stack,
)
from .tracking import (
    FrameEvent,
    Track,
    UnlinkedStart,
    explain_unlinked_starts,
    lane_gate_applies,
    track_detections,
)


def _track(detections, metadata, scale, config, geometry):
    """Dispatch to the configured tracking backend, ``(TrackList, events)``.

    ``kalman`` (default) is the axis-free motion-model tracker; ``overlap`` is
    the mask-overlap reconstructor in :mod:`corridor.engine.reconstruct`, which
    needs the per-frame image shape (``metadata.shape`` is ``T[Z]YX``, so
    ``shape[1:]`` is the ``(Y, X)`` or ``(Z, Y, X)`` frame) that the motion
    tracker does not. The two were measured equal on all eye-verified real data
    (``docs/ENGINE_ACCURACY.md``); the choice is a config flag, not a default
    change, so a run reproduces the shipped behaviour unless asked otherwise.
    """
    if config.tracking.reconstructor == RECONSTRUCTOR_OVERLAP:
        from ..engine.reconstruct import reconstruct_tracks

        return reconstruct_tracks(
            detections, metadata.shape[1:], metadata.n_frames, scale,
            config.tracking, geometry=geometry,
        )
    return track_detections(
        detections, metadata.n_frames, scale, config.tracking, geometry=geometry,
    )

# File names inside a result directory. Stable: other tools may rely on them.
F_DETECTIONS = "detections.csv"
F_TRACKS = "tracks.csv"
F_SUMMARY = "track_summary.csv"
F_MSD = "track_msd.csv"
F_DIAGNOSTICS = "segmentation_diagnostics.csv"
F_EVENTS = "tracking_events.csv"
F_QC = "qc_issues.csv"
F_UNLINKED = "unlinked_starts.csv"
F_RECOVERY = "recovery_attempts.csv"
F_MANIFEST = "run.json"
F_MASKS = "masks.npz"
F_RAW_MASKS = "masks_raw.npz"

#: Columns ``recovery_attempts.csv`` gains in schema 2 (critique C4).
#: ``track_id`` in that file is the FINAL track id -- the one ``tracks.csv``
#: uses -- and ``first_pass_track_id`` the id the attempt was made for, which
#: no other file mentions: recovery runs on the first-pass tracks, and the
#: second tracking pass that follows renumbers every track.  The bracket is
#: the pair of first-pass observations either side of the probed frame, by
#: ``(frame, det_label)``, so an attempt stays attributable whatever the ids
#: became.  The bracket names are ``RecoveryAttempt.to_row``'s own (E2), so
#: one attempt never writes the same fact under two headers.  ``det_label`` is
#: the recovered detection's label in ``detections.csv`` (after relabelling),
#: which makes a found attempt join ``tracks.csv`` on ``(frame, det_label)``.
RECOVERY_V2_COLUMNS = (
    "first_pass_track_id",
    "bracket_frame_before", "bracket_label_before",
    "bracket_frame_after", "bracket_label_after",
    "det_label", "duplicate_of_label",
)

#: TrackingConfig fields only the 1.x along/across tracker read.  run.json
#: records them apart from the settings that were applied, so a manifest never
#: implies the axis-free tracker used ``sigma_along_um``.  ``gate_chi2`` is
#: here because v2 derives the gate (``effective_gate_chi2 = 2U``); the value
#: applied is written under ``gate_chi2`` in the tracking block instead.
_TRACKING_V1_ONLY = (
    "max_perp_um", "sigma_along_um", "speed_uncertainty_fraction", "sigma_perp_um",
    "perp_width_fraction", "max_perp_widths", "gate_chi2",
)

#: What the run-level speed figures in ``run.json["results"]`` are.  Named in
#: the file because "mean speed" alone is ambiguous, and two different kinds
#: of robustness are at stake:
#:
#: *   *Within* a track, the net speed reads only the first and last
#:     observation, so a frame missed in between costs it nothing; the mean
#:     step speed does not have that property.
#: *   *Across* tracks, an unweighted mean is not robust at all: one spurious
#:     2-observation fragment moves it.  Measured on 052924_1 (2.0 defaults):
#:     one such track (3.80 µm/min) lifts the mean net speed of the other 12
#:     tracks from 0.525 to 0.777 µm/min (+48 %); their median moves from
#:     0.428 to 0.485 (+13 %): one extra value shifts a median by at most
#:     half a rank, however extreme it is.
#:     The median over tracks is therefore the run-level figure called
#:     robust; the means stay, named for what they are.
SPEED_ESTIMATORS = {
    "mean_speed": (
        "unweighted mean over tracks (>= 2 observations) of each track's mean step "
        "speed (step distance / elapsed time between its observations); sensitive "
        "both to missed frames and to short fragment tracks"
    ),
    "mean_net_speed": (
        "unweighted mean over tracks (>= 2 observations) of each track's net speed "
        "(net displacement / elapsed time of the track); per track it is insensitive "
        "to frames missed inside the track, but the mean over tracks is NOT robust: "
        "one short outlier track moves it"
    ),
    "median_speed": "median over tracks (>= 2 observations) of each track's mean step speed",
    "median_net_speed": (
        "median over tracks (>= 2 observations) of each track's net speed -- the robust "
        "run-level estimator: insensitive to missed frames inside a track (net speed) "
        "and to a minority of outlier or fragment tracks (median)"
    ),
    "robust_estimator": "median_net_speed",
}


class Cancelled(RuntimeError):
    """Raised when the caller asked for the run to stop."""


class Progress(Protocol):
    def stage(self, name: str, detail: str = "") -> None: ...
    def step(self, done: int, total: int) -> None: ...
    def cancelled(self) -> bool: ...


class NullProgress:
    """A progress sink for headless runs and tests."""

    def stage(self, name: str, detail: str = "") -> None:  # noqa: D102
        return None

    def step(self, done: int, total: int) -> None:  # noqa: D102
        return None

    def cancelled(self) -> bool:  # noqa: D102
        return False


@dataclass
class AnalysisResult:
    config: RunConfig
    metadata: StackMetadata
    scale: Scale
    #: The lanes of the field and whether they gated links (``applied``).
    geometry: ChannelGeometry
    segmentation: SegmentationOutput
    tracks: list[Track]
    events: list[FrameEvent]
    rows: list[dict[str, Any]]
    summaries: list[TrackSummary]
    issues: list[Any]
    unlinked: list[UnlinkedStart] = field(default_factory=list)
    recovery: RecoveryResult | None = None
    #: ``recovery_attempts.csv`` rows, keyed to FINAL track ids.
    recovery_rows: list[dict[str, Any]] = field(default_factory=list)
    #: ``track_msd.csv`` rows (µm², actual frame lags).
    msd: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    output_dir: Path | None = None
    #: The verified model that produced the masks; None for imported labels.
    model: ResolvedModel | None = None
    #: "2D" or "3D".
    dimensionality: str = "2D"

    @property
    def n_detections(self) -> int:
        """Every detection in ``detections.csv``: primary plus recovered.

        1.x reported the primary count here and in ``run.json`` while
        ``detections.csv`` held both, so the number on screen and the rows in
        the file disagreed whenever recovery found anything.
        """
        return len(self.all_detections)

    @property
    def n_primary_detections(self) -> int:
        """Detections the first segmentation pass produced (or the label file held)."""
        return len(self.segmentation.detections)

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)

    @property
    def all_detections(self) -> list[Detection]:
        """Every detection the analysis used, primary and recovered together.

        ``segmentation.detections`` is the first pass alone. Writing that as
        detections.csv while tracks.csv contains recovered positions leaves
        rows in one file with no counterpart in the other, which breaks the
        obvious join and gives no hint that it has.
        """
        out = list(self.segmentation.detections)
        if self.recovery is not None:
            out.extend(self.recovery.detections)
        return out

    @property
    def usable_tracks(self) -> list[Track]:
        need = self.config.tracking.min_observations
        return [t for t in self.tracks if t.n_obs >= need]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def effective_calibration(
    metadata: StackMetadata, override: CalibrationConfig
) -> tuple[Calibrated, Calibrated, Scale]:
    """Combine what the file says with what the user asked for.

    Returns ``(pixel size, frame interval, scale)``.  The scale carries the Z
    step too (``imaging.effective_z_step``: an override wins for a 3-D file,
    a 2-D file never has one), so ``Scale.anisotropy`` is defined exactly
    when both the Z step and the pixel size are known -- never assumed 1.
    """
    pixel = metadata.pixel_size_um
    interval = metadata.frame_interval_min
    if override.pixel_size_um:
        pixel = Calibrated(float(override.pixel_size_um), SOURCE_USER)
    if override.frame_interval_min:
        interval = Calibrated(float(override.frame_interval_min), SOURCE_USER)
    z_step = effective_z_step(metadata, override)
    scale = Scale.from_values(pixel.value, interval.value, z_step.value)
    return pixel, interval, scale


@dataclass(frozen=True)
class ReportedCalibration:
    """What the file itself says, captured before any user override replaces it.

    ``run_analysis`` writes the values it *used* into the metadata (every
    later stage reads them there).  Without this record an override would
    leave run.json half overwritten -- the review found ``pixel_size_um``
    0.639 (user) beside ``pixel_size_y_um`` 0.467 (file) and
    ``anisotropic_pixels`` false, a manifest contradicting itself.
    """

    pixel_size_x_um: Calibrated
    pixel_size_y_um: Calibrated
    frame_interval_min: Calibrated
    z_step_um: Calibrated

    @classmethod
    def of(cls, metadata: StackMetadata) -> "ReportedCalibration":
        return cls(
            metadata.pixel_size_um, metadata.pixel_size_y_um,
            metadata.frame_interval_min, metadata.z_step_um,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in ("pixel_size_x_um", "pixel_size_y_um", "frame_interval_min", "z_step_um"):
            value: Calibrated = getattr(self, name)
            out[name] = value.value
            out[f"{name}_source"] = value.source
        return out


def _check(progress: Progress) -> None:
    if progress.cancelled():
        raise Cancelled("Analysis cancelled.")


def _accepts(fn: Callable[..., Any], name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins only
        return False


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def run_analysis(
    config: RunConfig,
    progress: Progress | None = None,
    *,
    save: bool = True,
    keep_raw_masks: bool = True,
    model: ResolvedModel | None = None,
) -> AnalysisResult:
    """Analyse one file end to end.

    Raises, never degrades, on the two refusals the contract defines:
    ``imaging.AmbiguousAxes`` when the file cannot say which axis is time and
    ``config.import_.axes`` does not either, and
    ``model_registry.ModelUnavailable`` when the validated model is missing
    or does not match its checksum -- there is no fallback model.

    ``model`` is a :class:`ResolvedModel` the caller already verified (the
    desktop window does, to show it before the run); the service re-hashes it
    immediately before Cellpose reads it.  ``None`` lets the registry decide.
    It is ignored when ``config.import_.labels_path`` supplies the
    segmentation.
    """
    progress = progress or NullProgress()
    started = time.time()
    out_dir = Path(config.output_dir) if config.output_dir else None
    # The output directory is touched only once there is something to write
    # in it -- after the metadata and the model were accepted and the
    # segmentation ran (step 2).  A refusal (exit 3 or 4 on the command line)
    # or a cancel during segmentation therefore leaves no empty directory
    # behind, and leaves a previous complete result in it untouched.
    writes = bool(save and out_dir is not None)

    # -- 1. metadata --------------------------------------------------------
    progress.stage("Reading", "interpreting the image and its metadata")
    # The import settings name the axis order (and channel) of a file whose
    # metadata cannot: without them an ambiguous file is refused, not guessed.
    metadata = read_metadata(config.input_path, config.import_)
    reported = ReportedCalibration.of(metadata)
    pixel, interval, scale = effective_calibration(metadata, config.calibration)
    metadata.pixel_size_um = pixel
    metadata.frame_interval_min = interval
    metadata.z_step_um = effective_z_step(metadata, config.calibration)
    stack = load_stack(config.input_path, metadata)
    dimensionality = metadata.dimensionality
    _check(progress)

    # -- 2. segmentation (or the user's own labels) -------------------------
    def seg_progress(done: int, total: int) -> bool:
        progress.step(done, total)
        return not progress.cancelled()

    service: SegmentationService | None = None
    try:
        if config.import_.labels_path:
            progress.stage("Importing labels", Path(config.import_.labels_path).name)
            segmentation = load_label_stack(
                config.import_.labels_path, metadata.axes,
                image=stack, scale=scale, progress=seg_progress,
            )
        else:
            progress.stage("Segmenting", f"{metadata.n_frames} frames")
            service = SegmentationService(config.segmentation, model=model, scale=scale)
            segmentation = service.run_stack(stack, progress=seg_progress)
    except KeyboardInterrupt as exc:
        # The segmentation stages signal a cancel this way; anything else
        # that raises it is a real interrupt and must keep propagating.
        if progress.cancelled():
            raise Cancelled("Analysis cancelled.") from exc
        raise
    _check(progress)

    if writes:
        out_dir.mkdir(parents=True, exist_ok=True)
        # The completeness marker of a previous run goes before anything new
        # is written: until the new one exists, the directory is not a result.
        (out_dir / F_MANIFEST).unlink(missing_ok=True)
        export.save_masks(out_dir / F_MASKS, segmentation.masks)
        if keep_raw_masks and segmentation.provenance == PROVENANCE_MODEL:
            # An imported label file is its own raw record; a copy of it adds
            # nothing a reader could not get from the file itself.
            export.save_masks(out_dir / F_RAW_MASKS, segmentation.raw_masks)
        export.write_csv(
            out_dir / F_DIAGNOSTICS,
            export.DIAGNOSTIC_COLUMNS,
            [d.to_row() for d in segmentation.diagnostics],
        )

    # -- 3. device geometry: lanes, never an axis ----------------------------
    progress.stage("Measuring the device", "locating the channel walls")
    geometry = detect_channels(
        stack,
        config.geometry,
        segmentation.detections,
        pixel_size_um=scale.pixel_size_um if scale.calibrated_space else None,
        channel_constraint=config.tracking.channel_constraint,
        axes=metadata.stack_axes,
    )
    assign_lanes(segmentation.detections, geometry)
    _check(progress)

    if writes:
        export.write_csv(
            out_dir / F_DETECTIONS,
            export.DETECTION_COLUMNS,
            _detection_rows(segmentation.detections, scale, metadata),
        )

    # -- 4. tracking --------------------------------------------------------
    progress.stage("Tracking", "linking cells between frames")
    tracks, events = _track(segmentation.detections, metadata, scale, config, geometry)
    _check(progress)

    # -- 4b. recovery -------------------------------------------------------
    # Segmentation on this data is recall-limited, so a second pass looks
    # specifically where the first pass's tracks say a cell should be. Anything
    # found is re-tracked together with the primary detections, so a recovered
    # position has to survive the same cost model as everything else.
    recovery_result: RecoveryResult | None = None
    recovery_rows: list[dict[str, Any]] = []
    recovery_skipped = _recovery_skip_reason(config, segmentation, dimensionality, service)
    if config.recovery.enabled and tracks and recovery_skipped is None:
        progress.stage("Recovering", "looking where tracks predict a missing cell")
        first_pass = list(tracks)
        recovery_result = _recover(
            stack, first_pass, service, scale, config, geometry, progress
        )
        if recovery_result.detections:
            assign_lanes(recovery_result.detections, geometry)
            _relabel_recovered(segmentation.detections, recovery_result.detections)
            combined = list(segmentation.detections) + list(recovery_result.detections)
            tracks, events = _track(combined, metadata, scale, config, geometry)
        recovery_rows = recovery_attempt_rows(recovery_result, tracks)
        _check(progress)

    # -- 5. measurement -----------------------------------------------------
    progress.stage("Measuring", "velocities, MSD and track statistics")
    min_obs = config.tracking.min_observations
    msd = msd_rows(tracks, scale)
    rows = frame_rows(
        tracks, scale,
        source_frames=metadata.source_frames,
        min_observations=min_obs,
        reference_point_px=config.measurement.reference_point_px,
    )
    summaries = summarise(
        tracks, scale,
        source_frames=metadata.source_frames,
        min_observations=min_obs,
        msd_rows=msd,
        measurement=config.measurement,
    )
    unlinked = explain_unlinked_starts(tracks, scale, config.tracking, geometry=geometry)

    all_detections = list(segmentation.detections) + (
        list(recovery_result.detections) if recovery_result else []
    )
    issues = _collect_issues(
        metadata, scale, segmentation, events, tracks, summaries, config,
        geometry=geometry, unlinked=unlinked, dimensionality=dimensionality,
        count_series=_count_series(all_detections, metadata.n_frames),
    )

    manifest = build_manifest(
        config, metadata, scale, geometry, segmentation, tracks, summaries,
        elapsed_s=time.time() - started, output_dir=out_dir,
        recovery=recovery_result, recovery_skipped=recovery_skipped,
        tracking_notes=getattr(tracks, "notes", ()),
        issues=issues,
        reported=reported,
    )

    result = AnalysisResult(
        config=config, metadata=metadata, scale=scale, geometry=geometry,
        # The TrackList itself: it carries the tracker's id_map and notes.
        segmentation=segmentation, tracks=tracks, events=events, rows=rows,
        summaries=summaries, issues=issues, unlinked=unlinked,
        recovery=recovery_result, recovery_rows=recovery_rows, msd=msd,
        manifest=manifest, output_dir=out_dir,
        model=segmentation.model, dimensionality=dimensionality,
    )

    if writes:
        progress.stage("Saving", str(out_dir))
        save_result(result, out_dir)
    return result


def _detection_rows(
    detections: Sequence[Detection], scale: Scale, metadata: Any
) -> list[dict[str, Any]]:
    return detections_to_rows(
        detections,
        pixel_size_um=scale.pixel_size_um if scale.calibrated_space else None,
        frame_interval_min=scale.frame_interval_min if scale.calibrated_time else None,
        source_frames=getattr(metadata, "source_frames", None),
        z_step_um=scale.z_step_um if scale.calibrated_z else None,
    )


def _count_series(detections: Sequence[Detection], n_frames: int) -> list[int]:
    """Objects per frame among every detection the tracker saw (detections.csv)."""
    counts = [0] * int(n_frames)
    for d in detections:
        if 0 <= int(d.frame) < n_frames:
            counts[int(d.frame)] += 1
    return counts


# --------------------------------------------------------------------------
# Recovery
# --------------------------------------------------------------------------


def _recovery_skip_reason(
    config: RunConfig,
    segmentation: SegmentationOutput,
    dimensionality: str,
    service: SegmentationService | None,
) -> str | None:
    """Why recovery cannot run on this result, or None when it can.

    Recovery re-runs the validated model on 2-D crops around a prediction.
    With imported labels no model ran (and running one would put model
    output beside the user's own objects), and no model is validated for
    3-D; either way it is skipped and the manifest says why, rather than the
    attempt count silently reading zero.
    """
    if not config.recovery.enabled:
        return None
    if segmentation.provenance == PROVENANCE_IMPORTED or service is None:
        return (
            "the segmentation was imported from a label file; recovery re-runs the "
            "segmentation model and was not attempted"
        )
    if dimensionality != "2D":
        return "recovery searches 2-D frames and was not attempted on a 3-D stack"
    return None


def _recover(
    stack: np.ndarray,
    first_pass: list[Track],
    service: SegmentationService | None,
    scale: Scale,
    config: RunConfig,
    geometry: ChannelGeometry,
    progress: Progress,
) -> RecoveryResult:
    """``recovery.recover`` with the 2.0 interface (E1/E2, binding).

    ``recover(stack, tracks, service, scale, tracking_cfg, recovery_cfg, *,
    geometry=None, background=None)``.  The background is the static
    projection of the whole movie: the intensity tier measures a candidate
    against what the field looks like without cells, not against one frame.
    ``progress`` is passed only to a ``recover`` that takes it (E2's does; the
    binding interface does not require it).
    """
    background = static_projection(stack)
    fn = recovery_mod.recover
    extra: dict[str, Any] = {}
    if _accepts(fn, "progress"):
        extra["progress"] = lambda done, total: progress.step(done, total)
    return fn(
        stack, first_pass, service, scale, config.tracking, config.recovery,
        geometry=geometry, background=background, **extra,
    )


def _relabel_recovered(primary: Sequence[Detection], recovered: Sequence[Detection]) -> None:
    """Give every recovered detection a label no primary detection of its frame uses.

    A recovered detection carries a label from the crop it was found in,
    which routinely collides with a primary label in the same frame.
    tracks.csv records that label, so a reader joining the two files on
    (frame, det_label) would silently pick up a different cell.  Primary
    labels are never touched: they are the pixel values in ``masks.npz`` and
    what every attempt's bracket names.

    Since E2 09925b8, ``recovery.recover`` assigns labels by this same rule
    as it finds each cell (highest label among the frame's first-pass
    observations, then one more per recovered cell in order), so on its
    output this is a no-op whenever every primary detection is held by a
    first-pass track.  It stays as the safety net for a ``recover`` that
    does not label, and for a primary detection ``recover`` never saw.
    """
    highest: dict[int, int] = {}
    for det in primary:
        highest[det.frame] = max(highest.get(det.frame, 0), int(det.label))
    for det in recovered:
        highest[det.frame] = highest.get(det.frame, 0) + 1
        det.label = highest[det.frame]


def recovery_attempt_rows(
    result: RecoveryResult | None,
    final_tracks: Sequence[Track],
) -> list[dict[str, Any]]:
    """``recovery_attempts.csv`` rows keyed to FINAL track ids (critique C4).

    An attempt is made for a first-pass track.  The second tracking pass
    renumbers every track, and its ``TrackList.id_map`` maps only that pass's
    own stage-1 ids -- a first-pass id is not among them, so it cannot be
    translated through it.  What survives both passes is the detections:
    ``recovery.assign_final_track_ids`` (E2) gives each attempt the final
    track holding its recovered Detection (by identity), else the one holding
    its bracket's observation before the gap, else the one after; None, and
    a note in ``detail``, when none does -- never the first-pass id, which
    would join the wrong row of ``tracks.csv``.  The attempts themselves end
    up holding the final ids, so ``AnalysisResult.recovery`` agrees with the
    file.  ``det_label`` is the recovered detection's label after
    relabelling, which makes a found attempt join ``tracks.csv`` and
    ``detections.csv`` on ``(frame, det_label)``.
    """
    if result is None:
        return []
    recovery_mod.assign_final_track_ids(result.attempts, final_tracks)
    rows: list[dict[str, Any]] = []
    for attempt in result.attempts:
        row = dict(attempt.to_row())
        detection = attempt.detection
        row["det_label"] = None if detection is None else int(detection.label)
        rows.append(row)
    return rows


def recovery_columns(rows: Sequence[dict[str, Any]] = ()) -> list[str]:
    """``export.RECOVERY_COLUMNS``, the schema-2 provenance columns, then any
    other key the attempts report, once each.

    The tail exists because ``write_csv`` drops keys not in the header: a
    field ``RecoveryAttempt.to_row`` gains later must reach the file rather
    than vanish without a trace.
    """
    columns = list(export.RECOVERY_COLUMNS)
    columns += [c for c in RECOVERY_V2_COLUMNS if c not in columns]
    for row in rows:
        columns += [c for c in row if c not in columns]
    return columns


# --------------------------------------------------------------------------
# Quality control
# --------------------------------------------------------------------------


def _collect_issues(
    metadata: StackMetadata,
    scale: Scale,
    segmentation: SegmentationOutput,
    events: Sequence[FrameEvent],
    tracks: Sequence[Track],
    summaries: Sequence[TrackSummary],
    config: RunConfig,
    *,
    geometry: ChannelGeometry,
    unlinked: Sequence[UnlinkedStart],
    dimensionality: str,
    count_series: list[int],
) -> list[Any]:
    """``qc.collect_issues`` with the 2.0 interface (E1/E2, binding; no axis)."""
    return qc.collect_issues(
        metadata, scale, segmentation.diagnostics, events, tracks, summaries,
        config.tracking,
        geometry=geometry, unlinked=unlinked, model=segmentation.model,
        dimensionality=dimensionality, provenance=segmentation.provenance,
        count_series=count_series,
    )


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def save_result(result: AnalysisResult, out_dir: Path) -> None:
    """Write every output file, ``run.json`` last. Safe to call again to refresh a directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / F_MANIFEST).unlink(missing_ok=True)
    scale = result.scale
    metadata = result.metadata

    export.write_csv(
        out_dir / F_DETECTIONS,
        export.DETECTION_COLUMNS,
        _detection_rows(result.all_detections, scale, metadata),
    )
    export.write_csv(out_dir / F_TRACKS, export.TRACK_COLUMNS, result.rows)
    export.write_csv(
        out_dir / F_SUMMARY, export.SUMMARY_COLUMNS, [s.to_row() for s in result.summaries]
    )
    # getattr: a result assembled by a caller (tests, research scripts) may
    # predate the 2.0 fields; it still writes every file, empty where it has nothing.
    export.write_csv(out_dir / F_MSD, export.MSD_COLUMNS, getattr(result, "msd", None) or [])
    export.write_csv(
        out_dir / F_DIAGNOSTICS,
        export.DIAGNOSTIC_COLUMNS,
        [d.to_row() for d in result.segmentation.diagnostics],
    )
    export.write_csv(
        out_dir / F_EVENTS, export.EVENT_COLUMNS, [e.to_row() for e in result.events]
    )
    export.write_csv(
        out_dir / F_QC, export.QC_COLUMNS, [i.to_row() for i in result.issues]
    )
    rec_rows = list(getattr(result, "recovery_rows", None) or [])
    if not rec_rows and result.recovery is not None and result.recovery.attempts:
        # A result built by hand, without the pipeline's re-keying: key its
        # attempts to the tracks it holds now, never write first-pass ids.
        rec_rows = recovery_attempt_rows(result.recovery, result.tracks)
    export.write_csv(out_dir / F_RECOVERY, recovery_columns(rec_rows), rec_rows)
    export.write_csv(
        out_dir / F_UNLINKED,
        export.UNLINKED_COLUMNS,
        [export.unlinked_row(u) for u in result.unlinked],
    )
    export.save_masks(out_dir / F_MASKS, result.segmentation.masks)
    # Last: its presence is what says the directory holds a complete result.
    export.write_json(out_dir / F_MANIFEST, result.manifest)


# --------------------------------------------------------------------------
# run.json
# --------------------------------------------------------------------------


def _tier_counts(recovery: RecoveryResult | None) -> dict[str, int]:
    if recovery is None:
        return {}
    counts: dict[str, int] = {}
    for attempt in recovery.attempts:
        if attempt.found:
            counts[attempt.source] = counts.get(attempt.source, 0) + 1
    return counts


def _package_version(name: str) -> str | None:
    """An installed distribution's version from its metadata, importing nothing.

    ``import torch`` costs seconds and hundreds of megabytes; a manifest
    written after a label import, which never loads torch, must not pay that
    to print a version string.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return str(version(name))
    except PackageNotFoundError:
        return None


def _environment(segmentation: SegmentationOutput) -> dict[str, Any]:
    # gpu_available() imports torch. Once segmentation ran, torch is already
    # loaded and the probe is free; after a label import it is not, and the
    # question "could a GPU have been used" did not arise -- None, not False.
    probed = "torch" in sys.modules
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cellpose": _package_version("cellpose"),
        "torch": _package_version("torch"),
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
        "scikit-image": _package_version("scikit-image"),
        "tifffile": _package_version("tifffile"),
        "gpu_available": gpu_available() if probed else None,
        "gpu_used": bool(segmentation.used_gpu),
    }


def _finite(values: Sequence[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def _mean(values: Sequence[float | None]) -> float | None:
    clean = _finite(values)
    return float(np.mean(clean)) if clean else None


def _median(values: Sequence[float | None]) -> float | None:
    clean = _finite(values)
    return float(np.median(clean)) if clean else None


def _per_hr(per_min: float | None) -> float | None:
    return None if per_min is None else per_min * 60.0


def speed_results(summaries: Sequence[Any]) -> dict[str, Any]:
    """The run-level speed figures of ``run.json["results"]``, both units.

    Every figure is over the tracks that have one (a single observation has
    no speed), and ``speed_estimators`` names which is which and which is the
    robust one (see :data:`SPEED_ESTIMATORS`).
    """
    step = [getattr(s, "mean_speed_um_per_min", None) for s in summaries]
    net = [getattr(s, "net_speed_um_per_min", None) for s in summaries]
    out: dict[str, Any] = {"n_tracks_with_speed": len(_finite(net))}
    for name, value in (
        ("mean_speed", _mean(step)),
        ("mean_net_speed", _mean(net)),
        ("median_speed", _median(step)),
        ("median_net_speed", _median(net)),
    ):
        out[f"{name}_um_per_min"] = value
        out[f"{name}_um_per_hr"] = _per_hr(value)
    out["speed_estimators"] = dict(SPEED_ESTIMATORS)
    return out


def build_manifest(
    config: RunConfig,
    metadata: StackMetadata,
    scale: Scale,
    geometry: ChannelGeometry,
    segmentation: SegmentationOutput,
    tracks: Sequence[Track],
    summaries: Sequence[TrackSummary],
    *,
    elapsed_s: float,
    output_dir: Path | None,
    recovery: RecoveryResult | None = None,
    recovery_skipped: str | None = None,
    tracking_notes: Sequence[str] = (),
    issues: Sequence[Any] = (),
    reported: ReportedCalibration | None = None,
) -> dict[str, Any]:
    """Everything needed to reproduce or audit this run (``run.json``, schema 2).

    ``metadata`` holds the calibration that was *used*; ``reported`` what the
    file said before any override (None: nothing was overridden, so the two
    are the same).
    """
    if reported is None:
        reported = ReportedCalibration.of(metadata)
    diag = segmentation.diagnostics
    tracking = asdict(config.tracking)
    v1_only = {k: tracking.pop(k) for k in _TRACKING_V1_ONLY if k in tracking}
    n_primary = len(segmentation.detections)
    n_recovered = len(recovery.detections) if recovery is not None else 0
    dimensionality = metadata.dimensionality
    severities: dict[str, int] = {}
    for issue in issues:
        sev = str(getattr(issue, "severity", ""))
        severities[sev] = severities.get(sev, 0) + 1

    return {
        "schema_version": export.SCHEMA_VERSION,
        "application": {
            "name": app_meta.APP_NAME,
            "version": app_meta.APP_VERSION,
        },
        "environment": _environment(segmentation),
        "dimensionality": dimensionality,
        "input": {
            "path": str(metadata.path),
            "name": metadata.path.name,
            # ``shape`` is in the order of ``axes`` (store.project zips them);
            # ``stack_shape`` is the T[Z]YX array every result file indexes.
            "axes": metadata.axes,
            "shape": list(metadata.axes_shape),
            "stack_axes": metadata.stack_axes,
            "stack_shape": list(metadata.shape),
            "dimensionality": dimensionality,
            "axes_source": metadata.axes_source,
            "axes_reported": metadata.axes_raw,
            "axes_used": metadata.axes_used,
            "axes_interpretation": metadata.axes_interpretation,
            "channel_index": metadata.channel_index,
            "n_channels": metadata.n_channels,
            "dtype": metadata.dtype,
            "source_frames": metadata.source_frames,
            "source_frame_total": metadata.source_frame_total,
            "acquisition": metadata.acquisition,
            "notes": metadata.notes,
        },
        "import": asdict(config.import_),
        "calibration": {
            # Used.  Corridor measures X and Y with ONE pixel size, so the Y
            # size applied is the X size, whatever the file reports for Y.
            "pixel_size_um": metadata.pixel_size_um.value,
            "pixel_size_um_source": metadata.pixel_size_um.source,
            "pixel_size_y_um": metadata.pixel_size_um.value,
            "pixel_size_y_um_source": metadata.pixel_size_um.source,
            # The FILE's own X and Y sizes differ (imaging.read_metadata).  An
            # entered pixel size is one number and cannot make non-square
            # pixels square, so an override never clears this; it is what
            # QC's critical ``anisotropic_pixels`` issue reports.
            "anisotropic_pixels": metadata.anisotropic_pixels,
            "reported_by_file": reported.to_dict(),
            "frame_interval_min": metadata.frame_interval_min.value,
            "frame_interval_min_source": metadata.frame_interval_min.source,
            "z_step_um": metadata.z_step_um.value,
            "z_step_um_source": metadata.z_step_um.source,
            "anisotropy": scale.anisotropy,
            "spatially_calibrated": scale.calibrated_space,
            "temporally_calibrated": scale.calibrated_time,
            "z_calibrated": scale.calibrated_z,
        },
        # None when no model ran (imported labels).
        "model": segmentation.model_manifest(),
        "segmentation": {
            "provenance": segmentation.provenance,
            "dimensionality": segmentation.dimensionality,
            "model_path": segmentation.model_path or None,
            "model_sha256": segmentation.model_sha256,
            "labels_path": segmentation.labels_path,
            "labels_sha256": segmentation.labels_sha256,
            "diameter": config.segmentation.diameter,
            "cellprob_threshold": config.segmentation.cellprob_threshold,
            "flow_threshold": config.segmentation.flow_threshold,
            "channels": list(config.segmentation.channels),
            "normalize": config.segmentation.normalize,
            "normalisation_mode": config.segmentation.normalisation_mode,
            "normalize_percentiles": list(config.segmentation.normalize_percentiles),
            "normalize_tile_px": config.segmentation.normalize_tile_px,
            "normalize_sharpen_px": config.segmentation.normalize_sharpen_px,
            "ensemble": config.segmentation.ensemble,
            "ensemble_passes": segmentation.passes_per_frame,
            "min_extent_px": config.segmentation.min_extent_px,
            "min_area_px": config.segmentation.min_area_px,
            "drop_border_touching": config.segmentation.drop_border_touching,
            "raw_instances_per_frame": [d.raw_count for d in diag],
            "kept_instances_per_frame": [d.kept_count for d in diag],
            "removed_instances_total": sum(d.removed_count for d in diag),
            "detections_from_fallback": sum(
                1 for d in segmentation.detections if d.source == SOURCE_ENSEMBLE
            ),
            "notes": list(segmentation.notes),
        },
        "channel_geometry": {
            **geometry.to_dict(),
            "channel_constraint": config.tracking.channel_constraint,
            "lane_gate_applied": lane_gate_applies(geometry, config.tracking),
            "config": asdict(config.geometry),
        },
        "tracking": {
            "tracker": "axis-free Kalman/LAP with global gap closing (contract section 5)",
            **tracking,
            # Applied, derived: never an independent number that can drift.
            "gate_chi2": config.tracking.effective_gate_chi2,
            "initial_speed_sigma_um_per_min_applied": (
                config.tracking.effective_initial_speed_sigma_um_per_min
            ),
            "max_delta_frames": config.tracking.max_delta_frames(),
            "v1_fields_not_applied": v1_only,
            "notes": list(tracking_notes),
        },
        "measurement": asdict(config.measurement),
        "recovery": {
            **asdict(config.recovery),
            "skipped": recovery_skipped,
            "attempted": recovery.n_attempted if recovery else 0,
            "recovered": recovery.n_recovered if recovery else 0,
            # Candidates that were a second copy of a cell already detected in
            # that frame, and were therefore not added (recovery.duplicate_of).
            "duplicates_dropped": recovery.n_duplicates_dropped if recovery else 0,
            "by_tier": _tier_counts(recovery),
            "notes": list(recovery.notes) if recovery else [],
            # recovery_attempts.csv: track_id is the final id (tracks.csv's),
            # first_pass_track_id the id the attempt was made for.
            "attempt_track_ids": "final",
        },
        "results": {
            # n_detections is the row count of detections.csv, both kinds.
            "n_detections": n_primary + n_recovered,
            "n_detections_primary": n_primary,
            "n_detections_recovered": n_recovered,
            "n_tracks": len(tracks),
            "n_tracks_with_velocity": sum(
                1 for t in tracks if t.n_obs >= config.tracking.min_observations
            ),
            "n_observations": sum(t.n_obs for t in tracks),
            **speed_results(summaries),
            "qc_issues_by_severity": severities,
            "output_dir": str(output_dir) if output_dir else None,
            "elapsed_seconds": round(elapsed_s, 3),
        },
    }
