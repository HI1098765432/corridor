"""Stage 2: global gap closing with evidence from both sides of the gap.

Stage 1 decides frame by frame, with only the past.  When a cell reverses or
turns hard, its forward prediction misses and stage 1 starts a new track; but
the new track's own trajectory, run backwards, lands on where the old one
ended.  Stage 2 weighs both and rejoins them -- through the same U, gates and
margin.  The real case behind these tests: 052924_1, lane 3, frame 13, where
a cell moving up the channel at ~50 px per frame stopped and came back
(forward d^2 15.7, backward 0.08); stage 1 refused the link that v1 had made,
and stage 2 restored it.
"""

from __future__ import annotations

import math

import pytest

from corridor.core.config import TrackingConfig
from corridor.core.tracking import (
    EVIDENCE_BOTH,
    EVIDENCE_FORWARD,
    FLAG_GAP_CLOSED,
    GATE_GAP,
    GATE_MOTION,
    KalmanTracker,
    MotionModel,
    _Context,
    closing_cost,
    explain_unlinked_starts,
    motion_gate_chi2,
    track_detections,
)

from conftest import group_by_frame, identity_of, make_detection, straight_track

HORIZONTAL = math.pi / 2


def reversing_cell(speed: float = 40.0, missing: int = 0, label: int = 1, y: float = 100.0):
    """Moves right along its body for five frames, then straight back."""
    dets = [
        make_detection(t, 40 + speed * t, y, orientation_rad=HORIZONTAL, label=label)
        for t in range(5)
    ]
    x_end = 40 + speed * 4
    start = 5 + missing
    for k in range(5):
        dets.append(
            make_detection(start + k, x_end - speed * (k + 1), y, orientation_rad=HORIZONTAL, label=label)
        )
    return dets, start + 5


def turning_cell(speed: float = 30.0, y: float = 100.0):
    """Moves right along its body for five frames, then turns and moves down along its body."""
    dets = [make_detection(t, 40 + speed * t, y, orientation_rad=HORIZONTAL) for t in range(5)]
    x_end = 40 + speed * 4
    dets += [make_detection(5 + k, x_end, y + speed * (k + 1)) for k in range(5)]
    return dets, 10


@pytest.mark.parametrize("speed,missing", [(40.0, 0), (50.0, 1)])
def test_a_link_stage_1_refuses_is_closed_by_stage_2(scale, speed, missing):
    dets, n = reversing_cell(speed, missing)
    stage1, _ = track_detections(dets, n, scale, TrackingConfig(global_gap_closing=False))
    assert len(stage1) == 2, "the scenario must defeat stage 1, or this tests nothing"

    tracks, events = track_detections(dets, n, scale, TrackingConfig())
    assert len(tracks) == 1
    track = tracks[0]
    assert FLAG_GAP_CLOSED in track.flags
    joint = next(o for o in track.observations if o.frame == 5 + missing)
    b = joint.breakdown
    # The forward side alone would have refused it; the backward side is sure,
    # and the two together pass the two-sided gate.
    assert b.motion_evidence == EVIDENCE_BOTH
    assert b.motion_forward > motion_gate_chi2(2)
    assert b.motion_backward < 2.0
    assert b.motion_forward + b.motion_backward <= motion_gate_chi2(4)
    assert b.motion == pytest.approx(0.5 * (b.motion_forward + b.motion_backward))
    assert joint.gap_frames == 1 + missing
    assert joint.cost == pytest.approx(b.total)
    assert joint.link_margin is not None and joint.link_margin > 0
    assert joint.link_margin_global is not None and joint.link_margin_global > 0
    event = events[5 + missing]
    assert event.gap_closed == [track.id]
    assert "global gap closing" in event.to_row()["notes"]
    # The closed start is a continuation now, not a new track.
    assert event.n_new == 0 and event.n_matched == event.n_detections == 1


def test_a_rejoin_without_a_missing_frame_is_worded_as_one(scale):
    dets, n = reversing_cell(40.0, 0)
    _, events = track_detections(dets, n, scale, TrackingConfig())
    notes = events[5].to_row()["notes"]
    assert "rejoined by global gap closing" in notes
    assert "0 missing" not in notes


def test_gap_closing_can_be_switched_off(scale):
    dets, n = reversing_cell(40.0, 0)
    tracks, events = track_detections(dets, n, scale, TrackingConfig(global_gap_closing=False))
    assert len(tracks) == 2
    assert not any(e.gap_closed for e in events)


@pytest.mark.parametrize("speed", [44.0, 70.0])
def test_a_reversal_too_violent_for_both_sides_stays_split(scale, speed):
    """Straight back at 44 or 70 px/frame: the joint statistic fails the 2*ndim gate.

    At 44 px/frame (1.03 um/min) the forward side reads 21.3 and the backward
    side 0.0. Their mean, 10.7, is under the one-sided 13.8 -- the rule that
    used to be applied -- but a sum of two chi-square(2) statistics is
    judged against chi-square(4): 21.3 > 18.5.
    """
    dets, n = reversing_cell(speed, 0)
    tracks, _ = track_detections(dets, n, scale, TrackingConfig())
    assert len(tracks) == 2
    early, late = sorted(tracks, key=lambda t: t.first_frame)
    b = closing_cost(_context(scale), early, late)
    assert b.gated == GATE_MOTION
    assert b.motion_forward + b.motion_backward > motion_gate_chi2(4)
    if speed == 44.0:
        assert 0.5 * (b.motion_forward + b.motion_backward) < motion_gate_chi2(2)


def test_a_one_observation_fragment_is_judged_on_the_forward_side_alone(scale):
    """A fragment's 'backward prediction' is only the fresh prior; it is not evidence.

    A cell moving down at 20 px/frame, then one detection 60 px to its side.
    Forward d^2 17.4 fails the gate; the fragment's backward d^2 (0.8) says
    nothing, because a one-observation track predicts with a 71 px/frame
    spread. Averaging the two (9.1) used to join them.
    """
    dets = [make_detection(t, 45, 20 + 20 * t, label=1) for t in range(5)]
    dets.append(make_detection(5, 105, 120, label=2))
    stage1, _ = track_detections(dets, 6, scale, TrackingConfig(global_gap_closing=False))
    assert len(stage1) == 2
    tracks, events = track_detections(dets, 6, scale, TrackingConfig())
    assert len(tracks) == 2
    assert not events[5].gap_closed
    early, fragment = sorted(tracks, key=lambda t: t.first_frame)
    b = closing_cost(_context(scale), early, fragment)
    assert b.motion_evidence == EVIDENCE_FORWARD
    assert b.gated == GATE_MOTION
    assert b.motion == pytest.approx(b.motion_forward) and b.motion > motion_gate_chi2(2)
    assert 0.5 * (b.motion_forward + b.motion_backward) < motion_gate_chi2(2)


def _context(scale, cfg: TrackingConfig | None = None) -> _Context:
    cfg = cfg or TrackingConfig()
    return _Context(scale, cfg, MotionModel.from_config(scale, cfg), None)


def test_ids_are_renumbered_and_the_stage_1_ids_are_mapped(scale):
    """An early cell, a turning cell whose second half got a stage-1 id of its own, a late cell."""
    other = straight_track(10, x=400.0, label=2)
    turn, n = reversing_cell(40.0, 0)
    late = straight_track(3, x=600.0, start_frame=7, label=3)
    tracks, _ = track_detections(other + turn + late, n, scale, TrackingConfig())
    assert [t.id for t in tracks] == [1, 2, 3]
    assert [t.observations[0].frame for t in tracks] == [0, 0, 7]
    # Stage 1 made four tracks; three final ids remain, and every stage-1 id maps to one.
    assert len(tracks.id_map) == 4
    assert set(tracks.id_map.values()) == {1, 2, 3}
    turned = next(t for t in tracks if t.id == identity_of(tracks, 9, 1))
    assert len(turned.source_ids) == 2
    assert all(tracks.id_map[s] == turned.id for s in turned.source_ids)


def test_gap_closing_respects_the_gap_limit(scale):
    """A turn after max_gap + 1 missing frames is never bridged, by either stage."""
    cfg = TrackingConfig(max_gap=1)
    dets, n = reversing_cell(50.0, missing=2)
    tracks, _ = track_detections(dets, n, scale, cfg)
    assert len(tracks) == 2


def test_gap_closing_respects_lanes(scale):
    """A turn that would be closed in an open field is refused when it changes lane."""
    from corridor.core.geometry import GEOMETRY_FROM_RIDGES, ChannelGeometry, Lane

    # Horizontal lanes centred on y = 100 and y = 180: the boundary is y = 140.
    # The cell runs along y = 135 in lane 0 and turns down into lane 1.
    geometry = ChannelGeometry(
        lanes=[
            Lane(index=0, origin=(0.0, 100.0), direction=(1.0, 0.0), half_width_px=40.0),
            Lane(index=1, origin=(0.0, 180.0), direction=(1.0, 0.0), half_width_px=40.0),
        ],
        source=GEOMETRY_FROM_RIDGES, applied=True, pitch_px=80.0,
    )
    dets, n = turning_cell(30.0, y=135.0)
    free, _ = track_detections(dets, n, scale, TrackingConfig())
    walled, _ = track_detections(dets, n, scale, TrackingConfig(), geometry=geometry)
    assert len(free) == 1
    assert len(walled) == 2
    # (-1: past the last lane, where nothing is gated)
    assert all(len({o.channel for o in t.observations} - {-1}) == 1 for t in walled)


# --------------------------------------------------------------------------
# The unlinked-start audit
# --------------------------------------------------------------------------


def test_a_gap_only_refusal_is_reported_as_policy(scale):
    cfg = TrackingConfig(max_gap=2)
    dets = straight_track(12, skip={4, 5, 6, 7})
    tracks, _ = track_detections(dets, 12, scale, cfg)
    assert len(tracks) == 2
    (audit,) = explain_unlinked_starts(tracks, scale, cfg)
    assert audit.track_id == 2 and audit.candidate_track_id == 1
    assert audit.gap_frames == 5
    assert audit.refused_because == GATE_GAP
    assert audit.cost_chi2 is not None and audit.cost_chi2 < cfg.effective_gate_chi2
    assert audit.dx_px == pytest.approx(0.0)
    assert audit.dy_px == pytest.approx(100.0)
    assert audit.dz_px is None
    assert audit.distance_px == pytest.approx(100.0)
    assert audit.mahalanobis is not None and audit.mahalanobis < motion_gate_chi2(2)
    assert "gap" in audit.describe()


def test_a_motion_refusal_reports_the_mahalanobis_distance(scale):
    cfg = TrackingConfig()
    first = straight_track(5, label=1)
    # Appears 150 px to the side of where the first cell was heading.
    second = straight_track(4, x=195.0, y0=120.0, start_frame=5, label=2)
    tracks, _ = track_detections(first + second, 9, scale, cfg)
    assert len(tracks) == 2
    (audit,) = explain_unlinked_starts(tracks, scale, cfg)
    assert audit.refused_because == GATE_MOTION
    assert audit.mahalanobis > motion_gate_chi2(2)
    assert audit.dx_px == pytest.approx(150.0)


def test_the_audit_accepts_the_v1_argument_order(vertical_axis, scale):
    cfg = TrackingConfig(max_gap=2)
    tracks, _ = track_detections(straight_track(12, skip={4, 5, 6, 7}), 12, scale, cfg)
    (audit,) = explain_unlinked_starts(tracks, vertical_axis, scale, cfg)
    assert audit.refused_because == GATE_GAP
    assert audit.along_px is None and audit.across_px is None


def test_a_zero_cost_candidate_is_ranked_as_the_cheapest(scale):
    """v1 ranked candidates by ``cost or FORBIDDEN``, so a genuine 0.0 lost to anything.

    Two earlier tracks end at frame 1, both two frames before track 3 starts
    (``max_gap = 0``, so neither may join it). The one listed first sits
    15 px away (a positive closing cost); the other sits exactly where track
    3 appears, with the same round body and no gap penalty, so its relaxed
    closing cost is exactly 0.0. The audit must name the second.
    """
    cfg = TrackingConfig(max_gap=0, gap_penalty_chi2=0.0, global_gap_closing=False)
    round_cell = dict(eccentricity=0.3, major=32.0, minor=30.0, area=750.0)
    dets = [
        make_detection(0, 30, 300, label=1, **round_cell),
        make_detection(0, 45, 315, label=2, **round_cell),
        make_detection(1, 30, 300, label=1, **round_cell),
        make_detection(1, 45, 315, label=2, **round_cell),
        make_detection(3, 45, 315, label=3, **round_cell),
        make_detection(4, 45, 315, label=3, **round_cell),
    ]
    tracks, _ = track_detections(dets, 5, scale, cfg)
    assert len(tracks) == 3
    first, zero, late = sorted(tracks, key=lambda t: t.id)
    assert (first.observations[0].x, zero.observations[0].x) == (30, 45)
    assert late.first_frame == 3
    # The competitor is a real candidate with a positive, ungated cost.
    relaxed = _context(scale, TrackingConfig(max_gap=1, gap_penalty_chi2=0.0))
    competitor = closing_cost(relaxed, first, late)
    assert competitor.allowed and 0.0 < competitor.total < cfg.effective_gate_chi2
    (audit,) = explain_unlinked_starts(tracks, scale, cfg)
    assert audit.track_id == late.id
    assert audit.candidate_track_id == zero.id
    assert audit.cost_chi2 == 0.0
    assert audit.refused_because == GATE_GAP


def test_one_tracker_tracks_two_movies_independently(scale):
    """``track`` starts from nothing: no tracks, events, ids or notes carry over."""
    tracker = KalmanTracker(scale, TrackingConfig())
    first = group_by_frame(straight_track(5))
    second = group_by_frame(straight_track(5, x=200.0))
    tracks_a, events_a = tracker.track(first, 5)
    tracks_b, events_b = tracker.track(second, 5)
    assert [t.id for t in tracks_a] == [t.id for t in tracks_b] == [1]
    assert len(tracks_b) == 1 and tracks_b[0].observations[0].x == 200.0
    assert len(events_b) == 5 and not any(e.gap_closed for e in events_b)
    assert tracks_b.id_map == {1: 1}
    assert len(events_a) == 5 and events_a is not events_b


def test_the_tracker_object_reports_the_same_result(scale):
    dets, n = reversing_cell(40.0, 0)
    tracker = KalmanTracker(scale, TrackingConfig())
    tracks, events = tracker.track(group_by_frame(dets), n)
    assert len(tracks) == 1 and tracker.id_map == tracks.id_map
    assert len(events) == n
