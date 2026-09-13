"""Quality-control findings: the things a reviewer must be shown, not buried.

Each issue carries somewhere to *go* -- a frame and optionally a track -- so
the interface can take the reviewer straight to the evidence instead of
printing a warning they cannot act on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .config import Scale, TrackingConfig
from .confinement import ConfinementAxis
from .detections import FrameDiagnostics
from .imaging import StackMetadata
from .measurements import TrackSummary
from .tracking import FrameEvent, Track

SEVERITY_INFO = "info"
SEVERITY_WARN = "warning"
SEVERITY_CRITICAL = "critical"

_ORDER = {SEVERITY_CRITICAL: 0, SEVERITY_WARN: 1, SEVERITY_INFO: 2}


@dataclass
class QCIssue:
    code: str
    severity: str
    title: str
    detail: str
    frame: int | None = None
    track_id: int | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "frame": self.frame,
            "track_id": self.track_id,
        }


def collect_issues(
    metadata: StackMetadata,
    axis: ConfinementAxis,
    scale: Scale,
    diagnostics: Sequence[FrameDiagnostics],
    events: Sequence[FrameEvent],
    tracks: Sequence[Track],
    summaries: Sequence[TrackSummary],
    cfg: TrackingConfig,
) -> list[QCIssue]:
    issues: list[QCIssue] = []

    # -- calibration -------------------------------------------------------
    if not scale.calibrated_space:
        issues.append(
            QCIssue(
                "no_pixel_size", SEVERITY_CRITICAL,
                "Pixel size unknown",
                "This file carries no spatial calibration, so distances are reported in "
                "pixels only. Set the pixel size to get micrometres.",
            )
        )
    if not scale.calibrated_time:
        issues.append(
            QCIssue(
                "no_frame_interval", SEVERITY_CRITICAL,
                "Frame interval unknown",
                "This file carries no timing, so speeds cannot be reported per minute. "
                "Set the frame interval to get physical velocities.",
            )
        )

    # -- geometry ----------------------------------------------------------
    if axis.is_multichannel:
        issues.append(
            QCIssue(
                "multichannel_field", SEVERITY_WARN,
                f"{len(axis.channels)} channels in this field",
                "Cells were kept inside their own channel; associations across channel "
                "walls were forbidden. Check the detected channel boundaries.",
            )
        )
    if axis.confidence < 0.5:
        issues.append(
            QCIssue(
                "weak_axis", SEVERITY_WARN,
                "Migration axis is uncertain",
                f"The direction of confinement was {axis.describe()} with low confidence. "
                "Set it explicitly if the overlay looks wrong.",
            )
        )

    # -- segmentation ------------------------------------------------------
    kept = [d.kept_count for d in diagnostics]
    for d in diagnostics:
        if d.removed_count > 0:
            biggest = max(d.removed_extents) if d.removed_extents else 0
            issues.append(
                QCIssue(
                    "filtered_instances", SEVERITY_INFO,
                    f"{d.removed_count} small object(s) removed",
                    f"Frame {d.frame}: Cellpose produced {d.raw_count} object(s); "
                    f"{d.removed_count} were below the minimum size "
                    f"(largest removed span {biggest} px).",
                    frame=d.frame,
                )
            )
    for i, d in enumerate(diagnostics):
        if d.kept_count == 0:
            before = any(k > 0 for k in kept[:i])
            after = any(k > 0 for k in kept[i + 1:])
            if before and after:
                cause = (
                    "Cellpose produced no objects at all in this frame"
                    if d.raw_count == 0
                    else f"Cellpose produced {d.raw_count} object(s), all removed by the size filter"
                )
                issues.append(
                    QCIssue(
                        "segmentation_gap", SEVERITY_WARN,
                        "No cells found in a frame between detections",
                        f"Frame {d.frame}: {cause}. Tracks spanning this frame rely on "
                        "prediction rather than measurement.",
                        frame=d.frame,
                    )
                )

    # -- tracking ----------------------------------------------------------
    for ev in events:
        for tid in ev.merge_suspected:
            issues.append(
                QCIssue(
                    "merge_suspected", SEVERITY_WARN,
                    f"Track {tid} may have merged with another cell",
                    f"Frame {ev.frame}: this track's predicted position falls inside a "
                    "noticeably larger object that was matched to a different track. "
                    "No position was invented; the track is left unmatched here.",
                    frame=ev.frame, track_id=tid,
                )
            )

    speed_ceiling = cfg.max_speed_um_per_min
    for tr in tracks:
        for obs in tr.observations[1:]:
            if obs.gap_frames > 1:
                issues.append(
                    QCIssue(
                        "gap_bridged", SEVERITY_INFO,
                        f"Track {tr.id} reacquired after {obs.gap_frames - 1} missing frame(s)",
                        f"Frame {obs.frame}: matched with cost {obs.cost:.1f} "
                        f"(the limit is {cfg.unmatched_chi2:.0f}).",
                        frame=obs.frame, track_id=tr.id,
                    )
                )

    for s in summaries:
        if s.max_speed_um_per_min and s.max_speed_um_per_min > 0.8 * speed_ceiling:
            issues.append(
                QCIssue(
                    "fast_track", SEVERITY_WARN,
                    f"Track {s.track_id} moves close to the speed limit",
                    f"Peak {s.max_speed_um_per_min:.2f} um/min against a ceiling of "
                    f"{speed_ceiling:.2f}. Confirm this is one cell and not two.",
                    frame=s.last_frame, track_id=s.track_id,
                )
            )
        if "fragment" in (s.flags or ""):
            issues.append(
                QCIssue(
                    "fragment", SEVERITY_INFO,
                    f"Track {s.track_id} has only {s.n_observations} observation(s)",
                    "Too short to give a velocity. Reported for completeness.",
                    frame=s.first_frame, track_id=s.track_id,
                )
            )
        if (
            s.net_across_um is not None
            and s.net_along_um is not None
            and abs(s.net_across_um) > abs(s.net_along_um)
            and s.n_observations >= 3
        ):
            issues.append(
                QCIssue(
                    "lateral_drift", SEVERITY_WARN,
                    f"Track {s.track_id} moved more across the channel than along it",
                    f"Across {s.net_across_um:.1f} um vs along {s.net_along_um:.1f} um. "
                    "Either the migration axis is wrong or this is not a confined cell.",
                    frame=s.last_frame, track_id=s.track_id,
                )
            )

    if not tracks:
        issues.append(
            QCIssue(
                "no_tracks", SEVERITY_INFO,
                "No trajectories in this dataset",
                "Segmentation produced too few detections to link. The result files are "
                "written and empty.",
            )
        )

    issues.sort(key=lambda i: (_ORDER.get(i.severity, 9), i.frame if i.frame is not None else -1))
    return issues
