"""Quality-control findings: the things a reviewer must be shown, not buried.

Each issue carries somewhere to *go* -- a frame and, where one is meant, a
track -- so the interface can take the reviewer straight to the evidence
instead of printing a warning they cannot act on.

v2 (contract §8): there is no migration axis, so ``weak_axis`` and
``lateral_drift`` are gone; the lane findings (``multichannel_field``,
``inferred_channel``) are informational notes about the measured lanes. The
added codes describe what a trajectory is made of -- ambiguous links, jumps of
shape and size, long reacquisitions, gaps, splits, border events, abrupt count
changes and likely missed detections -- and what the result rests on: an
unvalidated model, a missing Z step, non-square pixels, or a segmentation the
validated model did not produce.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .config import Scale, TrackingConfig
from .detections import SOURCE_PRIMARY, FrameDiagnostics
from .geometry import ChannelGeometry
from .imaging import StackMetadata
from .measurements import TrackSummary
from .tracking import (
    FLAG_ENTERS_BORDER,
    FLAG_EXITS_BORDER,
    SHAPE_SIGMA_LN_ASPECT,
    SHAPE_SIGMA_SOLIDITY,
    FrameEvent,
    Observation,
    Track,
    UnlinkedStart,
    _aspect,
)

SEVERITY_INFO = "info"
SEVERITY_WARN = "warning"
SEVERITY_CRITICAL = "critical"

_ORDER = {SEVERITY_CRITICAL: 0, SEVERITY_WARN: 1, SEVERITY_INFO: 2}

#: ``SegmentationOutput.provenance`` of a segmentation the validated model
#: produced (``segmentation.PROVENANCE_MODEL``; repeated here so QC does not
#: import the segmentation service to read one string).
PROVENANCE_MODEL = "model"

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

#: A track whose peak step speed exceeds this fraction of the tracker's speed
#: gate is close to the limit of what the tracker will link (1.x's value).
FAST_TRACK_FRACTION = 0.8

#: ``link_margin_chi2`` below this is an ambiguous link: the next-best
#: explanation of the frame costs less than 2 chi-square units more than the
#: chosen one.  2 is the mean of a 2-D chi-square -- the residual of one
#: perfectly ordinary 2-D link -- so an alternative within it is one the cost
#: model cannot honestly tell apart.  A margin, not a probability: it is not
#: calibrated against ground truth, which does not exist for this data.
LINK_AMBIGUOUS_MARGIN_CHI2 = 2.0

#: The shape change between two consecutive observations, in the tracker's own
#: units, ``(dln aspect / 0.35)^2 + (dsolidity / 0.10)^2`` (before ``w_shape``),
#: above which it is a discontinuity: the 0.99 quantile of a 2-dof
#: chi-square, ``-2 ln 0.01``.
MORPHOLOGY_JUMP_CHI2 = -2.0 * math.log(0.01)

#: A size ratio between consecutive observations beyond ``exp(3 sigma)``,
#: sigma being the tracker's own ``sigma_ln_area`` (2.46x at the default 0.30),
#: is a size jump.  Within the tracker's hard gate (0.3-3.0x) so it can fire
#: on a link the tracker accepted.
SIZE_JUMP_SIGMAS = 3.0

#: A link that skipped more than one frame (``gap_frames > 2``) is a long
#: reacquisition: two or more frames with no evidence that it is one cell.
LONG_REACQUISITION_GAP_FRAMES = 2

#: Object count change between consecutive frames that is a jump: at least
#: this many objects, and at least this fraction of the earlier frame's count.
COUNT_JUMP_MIN_OBJECTS = 2
COUNT_JUMP_FRACTION = 0.5


@dataclass
class Issue:
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


#: 1.x name; the pipeline and UI still import it.
QCIssue = Issue


def _developer_override(model: Any) -> bool:
    """True for a ResolvedModel, a SegmentationOutput or a run.json block that says so."""
    if model is None:
        return False
    if isinstance(model, Mapping):
        return bool(model.get("developer_override"))
    return bool(getattr(model, "developer_override", False))


def _model_label(model: Any) -> str:
    if isinstance(model, Mapping):
        return str(model.get("model_id") or "an unregistered model")
    spec = getattr(model, "spec", None)
    path = getattr(model, "path", None)
    label = getattr(spec, "model_id", None) or "an unregistered model"
    return f"{label} ({path})" if path else str(label)


def _counts(
    diagnostics: Sequence[FrameDiagnostics],
    events: Sequence[FrameEvent],
    count_series: Sequence[int] | None,
) -> list[tuple[int, int]]:
    """``(frame, objects)`` in frame order, from the best source available."""
    if count_series is not None:
        return [(i, int(c)) for i, c in enumerate(count_series)]
    if diagnostics:
        return sorted((int(d.frame), int(d.kept_count)) for d in diagnostics)
    return sorted((int(e.frame), int(e.n_detections)) for e in events)


def _shape_jump(prev: Observation, obs: Observation) -> float | None:
    """The tracker's unweighted shape term between two observations, or None."""
    total, terms = 0.0, 0
    a0, a1 = _aspect(prev.detection, prev), _aspect(obs.detection, obs)
    if a0 and a1 and a0 > 0 and a1 > 0:
        total += (math.log(a1 / a0) / SHAPE_SIGMA_LN_ASPECT) ** 2
        terms += 1
    if prev.detection is not None and obs.detection is not None:
        s0, s1 = float(prev.detection.solidity), float(obs.detection.solidity)
        if math.isfinite(s0) and math.isfinite(s1):
            total += ((s1 - s0) / SHAPE_SIGMA_SOLIDITY) ** 2
            terms += 1
    return total if terms else None


def collect_issues(
    metadata: StackMetadata,
    scale: Scale,
    diagnostics: Sequence[FrameDiagnostics],
    events: Sequence[FrameEvent],
    tracks: Sequence[Track],
    summaries: Sequence[TrackSummary],
    cfg: TrackingConfig,
    *,
    geometry: ChannelGeometry | None = None,
    unlinked: Sequence[UnlinkedStart] = (),
    model: Any = None,
    dimensionality: str = "2D",
    provenance: str = PROVENANCE_MODEL,
    count_series: Sequence[int] | None = None,
) -> list[Issue]:
    """Every finding for one analysis, most severe first, then by frame.

    ``model`` is the ResolvedModel (or anything with ``developer_override``,
    or run.json's ``model`` block) the segmentation ran with; None when no
    model ran (imported labels).  ``provenance`` is
    ``SegmentationOutput.provenance``.  ``count_series`` is the number of
    objects per frame (index = frame) for the count-jump check; when None it
    comes from the diagnostics, or from the tracking events when there are no
    diagnostics (imported labels).
    """
    issues: list[Issue] = []
    three_d = str(dimensionality).upper() == "3D"

    # -- what the result rests on --------------------------------------------
    if _developer_override(model):
        issues.append(
            Issue(
                "developer_model_override", SEVERITY_CRITICAL,
                "Segmented with a model that is not the validated one",
                f"This run used {_model_label(model)}, set by a developer override. "
                "Its results are not validated Corridor results and must not be "
                "reported as such.",
            )
        )
    if provenance != PROVENANCE_MODEL:
        issues.append(
            Issue(
                "imported_segmentation", SEVERITY_INFO,
                "Segmentation was imported, not produced by the validated model",
                "The cell masks came from a label image supplied with the data. "
                "Measurements and tracks are only as good as those labels; Corridor "
                "has not checked them.",
            )
        )

    # -- calibration ---------------------------------------------------------
    if not scale.calibrated_space:
        issues.append(
            Issue(
                "no_pixel_size", SEVERITY_CRITICAL,
                "Pixel size unknown",
                "This file carries no spatial calibration, so distances are reported in "
                "pixels only. Set the pixel size to get micrometres.",
            )
        )
    if not scale.calibrated_time:
        issues.append(
            Issue(
                "no_frame_interval", SEVERITY_CRITICAL,
                "Frame interval unknown",
                "This file carries no timing, so speeds cannot be reported per minute. "
                "Set the frame interval to get physical velocities.",
            )
        )
    if three_d and not scale.calibrated_z:
        issues.append(
            Issue(
                "no_z_step", SEVERITY_CRITICAL,
                "Z step unknown",
                "This is a 3-D stack without a Z spacing. Volumes, surface areas and "
                "every distance with a Z component are not reported, and tracking "
                "counted one slice as one pixel, which is an assumption. Set the Z step "
                "to get physical 3-D measurements.",
            )
        )
    if getattr(metadata, "anisotropic_pixels", False):
        px = getattr(metadata, "pixel_size_um", None)
        py = getattr(metadata, "pixel_size_y_um", None)
        sizes = ""
        if px is not None and py is not None and getattr(px, "known", False) and getattr(py, "known", False):
            sizes = f" ({px.value:.6g} µm in X, {py.value:.6g} µm in Y)"
        issues.append(
            Issue(
                "anisotropic_pixels", SEVERITY_CRITICAL,
                "Pixels are not square",
                f"The file records different pixel sizes in X and Y{sizes}. Corridor "
                "measures with one pixel size, so every distance, speed, area and shape "
                "that is not purely along X is wrong by up to their ratio.",
            )
        )

    # -- lanes -----------------------------------------------------------------
    if geometry is not None and geometry.applied and geometry.is_multilane:
        issues.append(
            Issue(
                "multichannel_field", SEVERITY_INFO,
                f"{geometry.n_lanes} lanes in this field",
                "The lanes were measured from the channel walls, and a cell was never "
                "linked from one lane into another. Check the drawn lanes if a "
                "trajectory looks cut where a cell crosses between them.",
            )
        )
    inferred = [lane for lane in (geometry.lanes if geometry is not None else []) if not lane.detected]
    if inferred:
        constrained = (
            "They constrain tracking like the others."
            if geometry is not None and geometry.applied
            else "They do not constrain tracking in this run."
        )
        issues.append(
            Issue(
                "inferred_channel", SEVERITY_INFO,
                f"{len(inferred)} lane(s) were inferred, not seen",
                "Their walls were too faint to detect, so they were placed from the "
                "spacing of the lanes either side. That is an assumption about the "
                f"device. {constrained}",
            )
        )

    # -- segmentation --------------------------------------------------------
    kept = [d.kept_count for d in diagnostics]
    for d in diagnostics:
        if d.removed_count > 0:
            biggest = max(d.removed_extents) if d.removed_extents else 0
            issues.append(
                Issue(
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
                    Issue(
                        "segmentation_gap", SEVERITY_WARN,
                        "No cells found in a frame between detections",
                        f"Frame {d.frame}: {cause}. Tracks spanning this frame rely on "
                        "prediction rather than measurement.",
                        frame=d.frame,
                    )
                )

    counts = _counts(diagnostics, events, count_series)
    for (f0, c0), (f1, c1) in zip(counts[:-1], counts[1:]):
        if f1 != f0 + 1:
            continue
        change = abs(c1 - c0)
        if change >= COUNT_JUMP_MIN_OBJECTS and change >= COUNT_JUMP_FRACTION * c0:
            issues.append(
                Issue(
                    "count_jump", SEVERITY_WARN,
                    f"Object count jumps from {c0} to {c1} at frame {f1}",
                    f"Frame {f0} has {c0} object(s) and frame {f1} has {c1}. Cells rarely "
                    "appear or vanish in such numbers between two frames; segmentation "
                    "changing its mind about the same cells is the likelier cause. "
                    "Compare the two frames.",
                    frame=f1,
                )
            )

    # -- tracking events -------------------------------------------------------
    for ev in events:
        for tid in ev.merge_suspected:
            issues.append(
                Issue(
                    "merge_suspected", SEVERITY_WARN,
                    f"Track {tid} may have merged with another cell",
                    f"Frame {ev.frame}: this track's predicted position falls inside a "
                    "noticeably larger object that was matched to a different track. "
                    "No position was invented; the track is left unmatched here.",
                    frame=ev.frame, track_id=tid,
                )
            )
        for tid in getattr(ev, "split_suspected", ()) or ():
            issues.append(
                Issue(
                    "split_suspected", SEVERITY_WARN,
                    f"Track {tid} may be part of a split mask",
                    f"Frame {ev.frame}: a new track appeared inside the previous mask of a "
                    "track whose mask shrank sharply at the same time. That is one cell "
                    "segmented as two, or a division. Neither track was changed.",
                    frame=ev.frame, track_id=tid,
                )
            )

    # -- per link --------------------------------------------------------------
    size_bound = math.exp(SIZE_JUMP_SIGMAS * float(cfg.sigma_ln_area))
    gate = cfg.effective_gate_chi2
    for tr in tracks:
        if FLAG_ENTERS_BORDER in tr.flags and tr.observations:
            issues.append(
                Issue(
                    "border_entry", SEVERITY_INFO,
                    f"Track {tr.id} enters at the image border",
                    f"Frame {tr.first_frame}: it starts mid-movie on a mask cut by the "
                    "image edge, so its first positions and shape are of a partial cell.",
                    frame=tr.first_frame, track_id=tr.id,
                )
            )
        if FLAG_EXITS_BORDER in tr.flags and tr.observations:
            issues.append(
                Issue(
                    "border_exit", SEVERITY_INFO,
                    f"Track {tr.id} leaves at the image border",
                    f"Frame {tr.last_frame}: it ends mid-movie on a mask cut by the image "
                    "edge; the cell most likely left the field rather than vanished.",
                    frame=tr.last_frame, track_id=tr.id,
                )
            )

        for prev, obs in zip(tr.observations[:-1], tr.observations[1:]):
            gap = int(obs.gap_frames)
            cost = f"{obs.cost:.1f}" if obs.cost is not None else "unknown"
            stage2 = obs.breakdown is not None and obs.breakdown.motion_forward is not None
            how = "by global gap closing" if stage2 else "frame to frame"
            if gap > 1:
                issues.append(
                    Issue(
                        "gap_bridged", SEVERITY_INFO,
                        f"Track {tr.id} reacquired after {gap - 1} missing frame(s)",
                        f"Frame {obs.frame}: linked {how} with cost {cost} chi-square "
                        f"against a gate of {gate:.0f} (twice the unmatched cost).",
                        frame=obs.frame, track_id=tr.id,
                    )
                )
            if gap == 2:
                missing = prev.frame + 1
                issues.append(
                    Issue(
                        "likely_missed_detection", SEVERITY_WARN,
                        f"Frame {missing} is a likely missed detection",
                        f"Track {tr.id} is present at frames {prev.frame} and {obs.frame} "
                        f"and absent at frame {missing}. The cell was most likely there "
                        "and not segmented; this frame is a review case.",
                        frame=missing, track_id=tr.id,
                    )
                )
            if gap > LONG_REACQUISITION_GAP_FRAMES:
                margin = (
                    f" Its link margin is {obs.link_margin:.1f} chi-square."
                    if obs.link_margin is not None else ""
                )
                issues.append(
                    Issue(
                        "long_reacquisition", SEVERITY_WARN,
                        f"Track {tr.id} was reacquired after {gap - 1} missing frames",
                        f"Frame {obs.frame}: frames {prev.frame + 1}-{obs.frame - 1} hold no "
                        f"observation of it, and it was linked {how} at cost {cost} against "
                        f"a gate of {gate:.0f}.{margin} The longer the gap, the more room for "
                        "a different cell to have taken its place; check the frames either "
                        "side.",
                        frame=obs.frame, track_id=tr.id,
                    )
                )
            if obs.link_margin is not None and obs.link_margin < LINK_AMBIGUOUS_MARGIN_CHI2:
                issues.append(
                    Issue(
                        "link_ambiguous", SEVERITY_WARN,
                        f"Track {tr.id}'s link into frame {obs.frame} is ambiguous",
                        f"Frame {obs.frame}: the next-best explanation of this link costs "
                        f"only {obs.link_margin:.2f} chi-square more than the one chosen "
                        f"(below {LINK_AMBIGUOUS_MARGIN_CHI2:.0f}). The identity here could "
                        "have gone either way. This is a margin, not a probability.",
                        frame=obs.frame, track_id=tr.id,
                    )
                )
            jump = _shape_jump(prev, obs)
            if jump is not None and jump > MORPHOLOGY_JUMP_CHI2:
                issues.append(
                    Issue(
                        "morphology_discontinuity", SEVERITY_WARN,
                        f"Track {tr.id} changes shape abruptly at frame {obs.frame}",
                        f"Frames {prev.frame} to {obs.frame}: the change of aspect ratio and "
                        f"solidity scores {jump:.1f} in the tracker's shape units, above "
                        f"{MORPHOLOGY_JUMP_CHI2:.1f} (the 99th percentile of the change "
                        "expected for one cell). A different cell, or a segmentation error, "
                        "is likelier than a real change of shape.",
                        frame=obs.frame, track_id=tr.id,
                    )
                )
            s0, s1 = float(prev.size), float(obs.size)
            if s0 > 0 and s1 > 0:
                ratio = s1 / s0
                if ratio > size_bound or ratio < 1.0 / size_bound:
                    # Observation.size is the voxel count only when the
                    # detection carries one (Detection.size).
                    det = obs.detection
                    what = (
                        "volume"
                        if det is not None and det.ndim == 3 and det.volume_vox is not None
                        else "area"
                    )
                    issues.append(
                        Issue(
                            "size_jump", SEVERITY_WARN,
                            f"Track {tr.id}'s {what} changes {ratio:.2f}x at frame {obs.frame}",
                            f"Frames {prev.frame} to {obs.frame}: the {what} goes from "
                            f"{s0:.0f} to {s1:.0f} (outside {1.0 / size_bound:.2f}-"
                            f"{size_bound:.2f}x, three of the tracker's own size sigmas). "
                            "Two cells in one mask, part of a cell, or a different cell.",
                            frame=obs.frame, track_id=tr.id,
                        )
                    )

    # -- per track -------------------------------------------------------------
    speed_ceiling = float(cfg.max_speed_um_per_min)
    for s in summaries:
        peak = s.max_speed_um_per_min
        if peak and peak > FAST_TRACK_FRACTION * speed_ceiling:
            peak_hr = getattr(s, "max_speed_um_per_hr", None) or peak * 60.0
            issues.append(
                Issue(
                    "fast_track", SEVERITY_WARN,
                    f"Track {s.track_id} moves close to the speed limit",
                    f"Peak {peak:.2f} µm/min ({peak_hr:.0f} µm/hr) against a ceiling of "
                    f"{speed_ceiling:.2f} µm/min ({speed_ceiling * 60.0:.0f} µm/hr). "
                    "Confirm this is one cell and not two.",
                    frame=s.last_frame, track_id=s.track_id,
                )
            )
        if "fragment" in (s.flags or ""):
            issues.append(
                Issue(
                    "fragment", SEVERITY_INFO,
                    f"Track {s.track_id} has only {s.n_observations} observation(s)",
                    "Too short to give a velocity. Reported for completeness.",
                    frame=s.first_frame, track_id=s.track_id,
                )
            )
        missing = int(getattr(s, "total_missing_frames", 0) or 0)
        if missing and missing >= s.n_observations:
            issues.append(
                Issue(
                    "gap_dominated_track", SEVERITY_WARN,
                    f"Track {s.track_id} is missing more often than it is seen",
                    f"{missing} missing frame(s) against {s.n_observations} observation(s) "
                    f"between frames {s.first_frame} and {s.last_frame}. Most of this "
                    "trajectory is the tracker's inference across gaps, not measurement.",
                    frame=s.first_frame, track_id=s.track_id,
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
            and start.cost_chi2 <= gate
        ):
            severity = SEVERITY_WARN
        issues.append(
            Issue(
                "unlinked_start", severity,
                f"Track {start.track_id} starts at frame {start.frame} "
                f"and was not joined to track {start.candidate_track_id}",
                start.describe()
                + (
                    f" The fit itself would have cost {start.cost_chi2:.1f} "
                    f"against a gate of {gate:.0f}."
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
    by_id = {t.id: t for t in tracks}
    for summary in summaries:
        track = by_id.get(summary.track_id)
        if track is None or summary.n_observations < cfg.min_observations:
            continue

        borrowed = sum(1 for o in track.observations if o.source != SOURCE_PRIMARY)
        if borrowed and borrowed * 2 >= summary.n_observations:
            issues.append(
                Issue(
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
                Issue(
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
            Issue(
                "no_tracks", SEVERITY_INFO,
                "No trajectories in this dataset",
                "Segmentation produced too few detections to link. The result files are "
                "written and empty.",
            )
        )

    issues.sort(key=lambda i: (_ORDER.get(i.severity, 9), i.frame if i.frame is not None else -1))
    return issues
