"""End-to-end analysis: metadata -> segmentation -> tracking -> measurement -> disk.

Two ordering rules are structural, not stylistic:

*   Results are written **before** any viewer opens.  Visualisation is a way to
    check a result, never a precondition for the result existing.
*   Each stage persists as soon as it finishes.  If tracking fails, the
    segmentation that took minutes is still on disk and still valid.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from .. import app_meta
from . import export
from .confinement import (
    ConfinementAxis,
    assign_channels,
    resolve_axis,
    static_projection,
)
from .config import CalibrationConfig, RunConfig, Scale, SegmentationConfig
from .detections import (
    SOURCE_ENSEMBLE,
    Detection,
    FrameDiagnostics,
    detections_to_rows,
)
from .imaging import (
    SOURCE_USER,
    Calibrated,
    StackMetadata,
    load_stack,
    read_metadata,
)
from .measurements import TrackSummary, frame_rows, summarise
from .qc import QCIssue, collect_issues
from .recovery import RecoveryResult, recover
from .segmentation import (
    SegmentationOutput,
    SegmentationService,
    cellpose_version,
    discover_companion_models,
    gpu_available,
)
from .tracking import (
    FrameEvent,
    Track,
    UnlinkedStart,
    explain_unlinked_starts,
    track_detections,
)

# File names inside a result directory. Stable: other tools may rely on them.
F_DETECTIONS = "detections.csv"
F_TRACKS = "tracks.csv"
F_SUMMARY = "track_summary.csv"
F_DIAGNOSTICS = "segmentation_diagnostics.csv"
F_EVENTS = "tracking_events.csv"
F_QC = "qc_issues.csv"
F_UNLINKED = "unlinked_starts.csv"
F_RECOVERY = "recovery_attempts.csv"
F_MANIFEST = "run.json"
F_MASKS = "masks.npz"
F_RAW_MASKS = "masks_raw.npz"


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
    axis: ConfinementAxis
    segmentation: SegmentationOutput
    tracks: list[Track]
    events: list[FrameEvent]
    rows: list[dict[str, Any]]
    summaries: list[TrackSummary]
    issues: list[QCIssue]
    unlinked: list[UnlinkedStart] = field(default_factory=list)
    recovery: RecoveryResult | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    output_dir: Path | None = None

    @property
    def n_detections(self) -> int:
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


def effective_calibration(
    metadata: StackMetadata, override: CalibrationConfig
) -> tuple[Calibrated, Calibrated, Scale]:
    """Combine what the file says with what the user asked for."""
    pixel = metadata.pixel_size_um
    interval = metadata.frame_interval_min
    if override.pixel_size_um:
        pixel = Calibrated(float(override.pixel_size_um), SOURCE_USER)
    if override.frame_interval_min:
        interval = Calibrated(float(override.frame_interval_min), SOURCE_USER)
    scale = Scale.from_values(pixel.value, interval.value)
    return pixel, interval, scale


def _check(progress: Progress) -> None:
    if progress.cancelled():
        raise Cancelled("Analysis cancelled.")


def run_analysis(
    config: RunConfig,
    progress: Progress | None = None,
    *,
    save: bool = True,
    keep_raw_masks: bool = True,
) -> AnalysisResult:
    progress = progress or NullProgress()
    started = time.time()
    out_dir = Path(config.output_dir) if config.output_dir else None
    if save and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. metadata --------------------------------------------------------
    progress.stage("Reading", "interpreting the image and its metadata")
    # The import settings name the axis order (and channel) of a file whose
    # metadata cannot: without them an ambiguous file is refused, not guessed.
    metadata = read_metadata(config.input_path, config.import_)
    pixel, interval, scale = effective_calibration(metadata, config.calibration)
    metadata.pixel_size_um = pixel
    metadata.frame_interval_min = interval
    stack = load_stack(config.input_path, metadata)
    _check(progress)

    # -- 2. segmentation ----------------------------------------------------
    # A model-based fallback rung means "every model on this machine", so the
    # companions are discovered at run time rather than configured.
    #
    # The result is deliberately NOT written back into the caller's config. It
    # would be saved with the project and reused as the application default, so
    # a later run against a different model would quietly load the *previous*
    # model's siblings -- a wrong analysis that nothing in the output would
    # reveal. Discovery belongs to one run, so it stays in a local copy.
    seg_config = config.segmentation
    if not seg_config.ensemble_model_paths and seg_config.uses_companion_models():
        seg_config = replace(
            seg_config,
            ensemble_model_paths=discover_companion_models(seg_config.resolved_model()),
        )
    requested = seg_config.ensemble_cost_factor()
    detail = f"{metadata.n_frames} frames"
    if requested > 1:
        detail += f", up to {requested} passes each"
    progress.stage("Segmenting", detail)
    service = SegmentationService(seg_config)

    def seg_progress(done: int, total: int) -> bool:
        progress.step(done, total)
        return not progress.cancelled()

    segmentation = service.run_stack(stack, progress=seg_progress)
    _check(progress)

    if save and out_dir is not None:
        export.save_masks(out_dir / F_MASKS, segmentation.masks)
        if keep_raw_masks:
            export.save_masks(out_dir / F_RAW_MASKS, segmentation.raw_masks)
        export.write_csv(
            out_dir / F_DIAGNOSTICS,
            export.DIAGNOSTIC_COLUMNS,
            [d.to_row() for d in segmentation.diagnostics],
        )

    # -- 3. device geometry -------------------------------------------------
    progress.stage("Measuring the device", "locating the confinement channels")
    axis = resolve_axis(
        stack,
        config.confinement,
        segmentation.detections,
        pixel_size_um=scale.pixel_size_um if scale.calibrated_space else None,
    )
    assign_channels(segmentation.detections, axis)
    _check(progress)

    if save and out_dir is not None:
        export.write_csv(
            out_dir / F_DETECTIONS,
            export.DETECTION_COLUMNS,
            detections_to_rows(
                segmentation.detections,
                pixel_size_um=scale.pixel_size_um if scale.calibrated_space else None,
                frame_interval_min=(
                    scale.frame_interval_min if scale.calibrated_time else None
                ),
                source_frames=metadata.source_frames,
            ),
        )

    # -- 4. tracking --------------------------------------------------------
    progress.stage("Tracking", "linking cells between frames")
    tracks, events = track_detections(
        segmentation.detections, metadata.n_frames, axis, scale, config.tracking
    )
    _check(progress)

    # -- 4b. recovery -------------------------------------------------------
    # Segmentation on this data is recall-limited, so a second pass looks
    # specifically where the first pass's tracks say a cell should be. Anything
    # found is re-tracked together with the primary detections, so a recovered
    # position has to survive the same cost model as everything else.
    recovery_result: RecoveryResult | None = None
    if config.recovery.enabled and tracks:
        progress.stage("Recovering", "looking where tracks predict a missing cell")

        def recovery_progress(done: int, total: int) -> None:
            progress.step(done, total)

        recovery_result = recover(
            stack, tracks, service, axis, scale, config.tracking, config.recovery,
            background=static_projection(stack),
            progress=recovery_progress,
        )
        if recovery_result.detections:
            assign_channels(recovery_result.detections, axis)

            # A recovered detection carries a label from the crop it was found
            # in, which routinely collides with a primary label in the same
            # frame. tracks.csv records that label, so a reader joining the two
            # files on (frame, det_label) would silently pick up a different
            # cell. Relabel above whatever the frame already uses.
            next_label: dict[int, int] = {}
            for det in segmentation.detections:
                next_label[det.frame] = max(next_label.get(det.frame, 0), det.label)
            for det in recovery_result.detections:
                next_label[det.frame] = next_label.get(det.frame, 0) + 1
                det.label = next_label[det.frame]

            combined = list(segmentation.detections) + list(recovery_result.detections)
            tracks, events = track_detections(
                combined, metadata.n_frames, axis, scale, config.tracking
            )

        _check(progress)

    # -- 5. measurement -----------------------------------------------------
    progress.stage("Measuring", "velocities and track statistics")
    rows = frame_rows(
        tracks, axis, scale,
        source_frames=metadata.source_frames,
        min_observations=config.tracking.min_observations,
    )
    summaries = summarise(
        tracks, axis, scale,
        source_frames=metadata.source_frames,
        min_observations=config.tracking.min_observations,
    )
    unlinked = explain_unlinked_starts(tracks, axis, scale, config.tracking)
    issues = collect_issues(
        metadata, axis, scale, segmentation.diagnostics, events, tracks,
        summaries, config.tracking, unlinked,
    )

    manifest = build_manifest(
        config, metadata, scale, axis, segmentation, tracks, summaries,
        elapsed_s=time.time() - started, output_dir=out_dir,
        recovery=recovery_result, seg_config=seg_config,
    )

    result = AnalysisResult(
        config=config, metadata=metadata, scale=scale, axis=axis,
        segmentation=segmentation, tracks=tracks, events=events, rows=rows,
        summaries=summaries, issues=issues, unlinked=unlinked,
        recovery=recovery_result, manifest=manifest, output_dir=out_dir,
    )

    if save and out_dir is not None:
        progress.stage("Saving", str(out_dir))
        save_result(result, out_dir)
    return result


def save_result(result: AnalysisResult, out_dir: Path) -> None:
    """Write every output file. Safe to call again to refresh a directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = result.scale
    metadata = result.metadata

    export.write_csv(
        out_dir / F_DETECTIONS,
        export.DETECTION_COLUMNS,
        detections_to_rows(
            result.all_detections,
            pixel_size_um=scale.pixel_size_um if scale.calibrated_space else None,
            frame_interval_min=scale.frame_interval_min if scale.calibrated_time else None,
            source_frames=metadata.source_frames,
        ),
    )
    export.write_csv(out_dir / F_TRACKS, export.TRACK_COLUMNS, result.rows)
    export.write_csv(
        out_dir / F_SUMMARY, export.SUMMARY_COLUMNS, [s.to_row() for s in result.summaries]
    )
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
    export.write_csv(
        out_dir / F_RECOVERY,
        export.RECOVERY_COLUMNS,
        [a.to_row() for a in (result.recovery.attempts if result.recovery else [])],
    )
    export.write_csv(
        out_dir / F_UNLINKED,
        export.UNLINKED_COLUMNS,
        [
            {
                "track_id": u.track_id,
                "starts_at_frame": u.frame,
                "nearest_earlier_track": u.candidate_track_id,
                "that_track_ended_at_frame": u.candidate_last_frame,
                "gap_frames": u.gap_frames,
                "distance_px": u.distance_px,
                "along_channel_px": u.along_px,
                "across_channel_px": u.across_px,
                "implied_speed_um_per_min": u.speed_um_per_min,
                "would_have_cost_chi2": u.cost_chi2,
                "refused_because": u.refused_because,
                "explanation": u.describe(),
            }
            for u in result.unlinked
        ],
    )
    export.save_masks(out_dir / F_MASKS, result.segmentation.masks)
    export.write_json(out_dir / F_MANIFEST, result.manifest)


def _tier_counts(recovery: RecoveryResult | None) -> dict[str, int]:
    if recovery is None:
        return {}
    counts: dict[str, int] = {}
    for attempt in recovery.attempts:
        if attempt.found:
            counts[attempt.source] = counts.get(attempt.source, 0) + 1
    return counts


def build_manifest(
    config: RunConfig,
    metadata: StackMetadata,
    scale: Scale,
    axis: ConfinementAxis,
    segmentation: SegmentationOutput,
    tracks: Sequence[Track],
    summaries: Sequence[TrackSummary],
    *,
    elapsed_s: float,
    output_dir: Path | None,
    recovery: RecoveryResult | None = None,
    seg_config: SegmentationConfig | None = None,
) -> dict[str, Any]:
    """Everything needed to reproduce or audit this run.

    ``seg_config`` is the segmentation configuration that was actually used,
    which is not always ``config.segmentation``: companion models are
    discovered per run into a local copy. Defaulting to the caller's config
    keeps this callable on its own, for a re-save of an older result.
    """
    seg_config = seg_config or config.segmentation
    diag = segmentation.diagnostics
    return {
        "application": {
            "name": app_meta.APP_NAME,
            "version": app_meta.APP_VERSION,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cellpose": cellpose_version(),
            "numpy": np.__version__,
            "gpu_available": gpu_available(),
            "gpu_used": segmentation.used_gpu,
        },
        "input": {
            "path": str(metadata.path),
            "name": metadata.path.name,
            "shape_tyx": list(metadata.shape),
            "dtype": metadata.dtype,
            "axes_reported": metadata.axes_raw,
            "axes_interpretation": metadata.axes_interpretation,
            "source_frames": metadata.source_frames,
            "source_frame_total": metadata.source_frame_total,
            "acquisition": metadata.acquisition,
            "notes": metadata.notes,
        },
        "calibration": {
            "pixel_size_um": metadata.pixel_size_um.value,
            "pixel_size_um_source": metadata.pixel_size_um.source,
            "frame_interval_min": metadata.frame_interval_min.value,
            "frame_interval_min_source": metadata.frame_interval_min.source,
            "spatially_calibrated": scale.calibrated_space,
            "temporally_calibrated": scale.calibrated_time,
        },
        "segmentation": {
            "model_path": segmentation.model_path,
            "model_sha256": segmentation.model_sha256,
            "use_custom_model": config.segmentation.use_custom_model,
            "diameter": config.segmentation.diameter,
            "cellprob_threshold": config.segmentation.cellprob_threshold,
            "flow_threshold": config.segmentation.flow_threshold,
            "channels": list(config.segmentation.channels),
            "normalize": config.segmentation.normalize,
            # Both of these change what actually ran, so a manifest without
            # them would describe a different analysis from the one that
            # produced the numbers beside it.
            "normalisation_mode": seg_config.normalisation_mode,
            "normalize_percentiles": list(seg_config.normalize_percentiles),
            "normalize_tile_px": seg_config.normalize_tile_px,
            "normalize_sharpen_px": seg_config.normalize_sharpen_px,
            "ensemble": seg_config.ensemble,
            # Both numbers, because they differ whenever an optional model is
            # absent, and the difference is exactly what a reader needs to see.
            "ensemble_passes_requested": seg_config.ensemble_cost_factor(),
            "ensemble_passes": segmentation.passes_per_frame,
            "ensemble_model_paths": list(seg_config.ensemble_model_paths),
            "ensemble_models_unavailable": list(segmentation.unavailable_models),
            "min_extent_px": config.segmentation.min_extent_px,
            "min_area_px": config.segmentation.min_area_px,
            "drop_border_touching": config.segmentation.drop_border_touching,
            "raw_instances_per_frame": [d.raw_count for d in diag],
            "kept_instances_per_frame": [d.kept_count for d in diag],
            "removed_instances_total": sum(d.removed_count for d in diag),
            "detections_from_fallback": sum(
                1 for d in segmentation.detections if d.source == SOURCE_ENSEMBLE
            ),
        },
        "confinement": axis.to_dict(),
        "tracking": {
            **config.tracking.__dict__,
            "max_delta_frames": config.tracking.max_delta_frames(),
        },
        "recovery": {
            **config.recovery.__dict__,
            "attempted": recovery.n_attempted if recovery else 0,
            "recovered": recovery.n_recovered if recovery else 0,
            "by_tier": _tier_counts(recovery),
        },
        "results": {
            "n_detections": len(segmentation.detections),
            "n_tracks": len(tracks),
            "n_tracks_with_velocity": sum(
                1 for t in tracks if t.n_obs >= config.tracking.min_observations
            ),
            "n_observations": sum(t.n_obs for t in tracks),
            "mean_speed_um_per_min": _mean(
                [s.mean_speed_um_per_min for s in summaries]
            ),
            "output_dir": str(output_dir) if output_dir else None,
            "elapsed_seconds": round(elapsed_s, 3),
        },
    }


def _mean(values: Sequence[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else None
