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
    FLAG_GAP_CLOSED,
    GATE_GAP,
    GATE_MOTION,
    KalmanTracker,
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


@pytest.mark.parametrize(
    "scene", [("reverse", 40.0, 0), ("reverse", 50.0, 1), ("turn", 30.0, 0)]
)
def test_a_link_stage_1_refuses_is_closed_by_stage_2(scale, scene):
    kind, speed, missing = scene
    dets, n = reversing_cell(speed, missing) if kind == "reverse" else turning_cell(speed)
    stage1, _ = track_detections(dets, n, scale, TrackingConfig(global_gap_closing=False))
    assert len(stage1) == 2, "the scenario must defeat stage 1, or this tests nothing"

    tracks, events = track_detections(dets, n, scale, TrackingConfig())
    assert len(tracks) == 1
    track = tracks[0]
    assert FLAG_GAP_CLOSED in track.flags
    joint = next(o for o in track.observations if o.frame == 5 + missing)
    b = joint.breakdown
    # The forward side alone would have refused it; the backward side is sure.
    assert b.motion_forward > motion_gate_chi2(2)
    assert b.motion_backward < 2.0
    assert b.motion == pytest.approx(0.5 * (b.motion_forward + b.motion_backward))
    assert joint.gap_frames == 1 + missing
    assert joint.cost == pytest.approx(b.total)
    assert joint.link_margin is not None and joint.link_margin > 0
    assert events[5 + missing].gap_closed == [track.id]
    assert "global gap closing" in events[5 + missing].to_row()["notes"]


def test_gap_closing_can_be_switched_off(scale):
    dets, n = reversing_cell(40.0, 0)
    tracks, events = track_detections(dets, n, scale, TrackingConfig(global_gap_closing=False))
    assert len(tracks) == 2
    assert not any(e.gap_closed for e in events)


def test_a_reversal_too_violent_for_both_sides_stays_split(scale):
    """70 px/frame straight back: the mean of both sides is still above the motion gate."""
    dets = [make_detection(t, 40 + 70 * t, 100, orientation_rad=HORIZONTAL) for t in range(5)]
    dets += [
        make_detection(5 + k, 320 - 70 * (k + 1), 100, orientation_rad=HORIZONTAL)
        for k in range(5)
    ]
    tracks, _ = track_detections(dets, 10, scale, TrackingConfig())
    assert len(tracks) == 2


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
    """v1 ranked candidates by ``cost or FORBIDDEN``, so a genuine 0.0 lost to anything."""
    cfg = TrackingConfig(max_gap=0, global_gap_closing=False)
    # Track 1 stops at frame 1 exactly where track 2 appears at frame 3, with
    # the same size and shape (cost near 0 once the gap is relaxed).
    dets = [make_detection(0, 45, 100, label=1), make_detection(1, 45, 100, label=1)]
    dets += [make_detection(3, 45, 100, label=1), make_detection(4, 45, 100, label=1)]
    tracks, _ = track_detections(dets, 5, scale, cfg)
    (audit,) = explain_unlinked_starts(tracks, scale, cfg)
    assert audit.candidate_track_id == 1


def test_the_tracker_object_reports_the_same_result(scale):
    dets, n = reversing_cell(40.0, 0)
    tracker = KalmanTracker(scale, TrackingConfig())
    tracks, events = tracker.track(group_by_frame(dets), n)
    assert len(tracks) == 1 and tracker.id_map == tracks.id_map
    assert len(events) == n
