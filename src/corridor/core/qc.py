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
from .detections import SOURCE_PRIMARY, FrameDiagnostics
from .imaging import StackMetadata
from .measurements import TrackSummary
from .tracking import FrameEvent, Track, UnlinkedStart

SEVERITY_INFO = "info"
SEVERITY_WARN = "warning"
SEVERITY_CRITICAL = "critical"

_ORDER = {SEVERITY_CRITICAL: 0, SEVERITY_WARN: 1, SEVERITY_INFO: 2}

#: Below this **net** displacement a trajectory has gone nowhere.  The cells in
#: the supplied labelled data are 9-15 px wide, which at 0.4671 µm/px is
#: 4.2-7.0 µm, so a track whose start and end are less than 4 µm apart has not
#: moved its own width.
#:
#: Net displacement, not path length: path length accumulates centroid jitter,
#: so a fixed object whose centroid wanders half a pixel a frame racks up
#: several µm of "path" over ten frames and escapes the very check meant to
#: catch it.  Start-to-end distance does not accumulate anything.
STATIONARY_NET_UM = 4.0
#: ...and only once enough *time* has passed to call it. A frame count is the
#: wrong unit here: four frames is an hour of the supplied data but two minutes
#: of a fast acquisition, and a cell that has not moved 4 µm in two minutes is
#: simply a cell. Half an hour is roughly 1.5 frames of the supplied data and
#: sixty frames of a 30-second acquisition; either way it is long enough that
#: going nowhere means something.
STATIONARY_MIN_MINUTES = 30.0
#: Fallback when the file carries no timing at all and minutes do not exist.
STATIONARY_MIN_SPAN = 3


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
    unlinked: Sequence[UnlinkedStart] = (),
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
    inferred = [c for c in axis.channels if not c.detected]
    if inferred:
        issues.append(
            QCIssue(
                "inferred_channel", SEVERITY_WARN,
                f"{len(inferred)} channel boundary/boundaries were inferred, not seen",
                "Their walls were too faint to detect, so they were placed from the "
                "spacing of the channels either side. That is an assumption about the "
                "device. If a cell looks like it changed channel, check these first.",
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

    # -- identities the tracker deliberately did not join --------------------
    for start in unlinked:
        if start.candidate_track_id is None:
            continue
        severity = SEVERITY_INFO
        # A pairing refused only by the gap limit, that would otherwise have
        # fitted well, is the case a reviewer most needs to see: the tracker is
        # declining to assert continuity, not ruling it out.
        if (
            start.refused_because == "gap_too_long"
            and start.cost_chi2 is not None
            and start.cost_chi2 <= cfg.gate_chi2
        ):
            severity = SEVERITY_WARN
        issues.append(
            QCIssue(
                "unlinked_start", severity,
                f"Track {start.track_id} starts at frame {start.frame} "
                f"and was not joined to track {start.candidate_track_id}",
                start.describe()
                + (
                    f" The fit itself would have cost {start.cost_chi2:.1f} "
                    f"against a limit of {cfg.gate_chi2:.0f}."
                    if start.cost_chi2 is not None else ""
                )
                + " Corridor does not assert continuity it cannot support; "
                "judge this one from the images.",
                frame=start.frame, track_id=start.track_id,
            )
        )

    # -- what the trajectory is actually made of ----------------------------
    # Measured on the supplied data (scripts/experiment_fallback.py): the
    # assignment step does *not* filter out a false detection that repeats in
    # the same place, because a stationary object is the most self-consistent
    # thing a cost model based on predicted position can see. Channel-wall
    # texture at a permissive flow threshold is exactly that. So the two
    # properties below are checked directly rather than assumed away.
    for summary in summaries:
        track = next((t for t in tracks if t.id == summary.track_id), None)
        if track is None or summary.n_observations < cfg.min_observations:
            continue

        borrowed = sum(1 for o in track.observations if o.source != SOURCE_PRIMARY)
        if borrowed and borrowed * 2 >= summary.n_observations:
            issues.append(
                QCIssue(
                    "fallback_dependent_track", SEVERITY_WARN,
                    f"Track {summary.track_id} depends on the extra detection passes",
                    f"{borrowed} of its {summary.n_observations} positions were found "
                    "only by a more permissive setting, not by the model's own "
                    "thresholds. Those passes raise recall and lower precision "
                    "together, so this trajectory deserves a look at the images "
                    "before it is used.",
                    frame=summary.first_frame, track_id=summary.track_id,
                )
            )

        long_enough = (
            summary.duration_min >= STATIONARY_MIN_MINUTES
            if summary.duration_min is not None
            else summary.span_frames >= STATIONARY_MIN_SPAN
        )
        if (
            summary.net_displacement_um is not None
            and summary.net_displacement_um < STATIONARY_NET_UM
            and long_enough
        ):
            elapsed = (
                f"{summary.duration_min:.0f} min"
                if summary.duration_min is not None
                else f"{summary.span_frames} frames"
            )
            issues.append(
                QCIssue(
                    "stationary_track", SEVERITY_WARN,
                    f"Track {summary.track_id} barely moves",
                    f"It ends {summary.net_displacement_um:.1f} µm from where it "
                    f"started, after {elapsed} — less than the width of a cell in "
                    "this data. That is either a cell that never migrated or a "
                    "fixed feature of the device being tracked as one; the images "
                    "distinguish them and this software cannot.",
                    frame=summary.first_frame, track_id=summary.track_id,
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
