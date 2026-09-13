"""Tracking behaviour, tested without Cellpose.

Every scenario here is one the supplied datasets actually contain or that the
assignment model must survive: a cell that stalls, a cell that disappears for a
frame, two cells in one channel, a detection that cannot belong to anything.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from corridor.core.tracking import (
    FORBIDDEN,
    GATE_AREA,
    GATE_CHANNEL,
    GATE_GAP,
    GATE_PERP,
    GATE_SPEED,
    ConfinementTracker,
    Track,
    TrackState,
    build_assignment_matrix,
    pair_cost,
    solve_assignment,
    track_detections,
)

from conftest import group_by_frame, make_detection, straight_track


# --------------------------------------------------------------------------
# Whole-sequence behaviour
# --------------------------------------------------------------------------


def test_single_cell_moving_straight_is_one_track(vertical_axis, scale, tracking_config):
    dets = straight_track(10)
    tracks, _ = track_detections(dets, 10, vertical_axis, scale, tracking_config)
    assert len(tracks) == 1
    assert tracks[0].n_obs == 10
    assert [o.frame for o in tracks[0].observations] == list(range(10))


def test_one_frame_loss_keeps_identity(vertical_axis, scale, tracking_config):
    dets = straight_track(10, skip={4})
    tracks, _ = track_detections(dets, 10, vertical_axis, scale, tracking_config)
    assert len(tracks) == 1, "a single missing frame must not split the track"
    observed = [o.frame for o in tracks[0].observations]
    assert observed == [0, 1, 2, 3, 5, 6, 7, 8, 9]
    reacquired = next(o for o in tracks[0].observations if o.frame == 5)
    assert reacquired.gap_frames == 2


def test_multi_frame_loss_within_max_gap_keeps_identity(
    vertical_axis, scale, tracking_config
):
    tracking_config.max_gap = 3
    dets = straight_track(12, skip={4, 5, 6})
    tracks, _ = track_detections(dets, 12, vertical_axis, scale, tracking_config)
    assert len(tracks) == 1
    reacquired = next(o for o in tracks[0].observations if o.frame == 7)
    assert reacquired.gap_frames == 4


def test_loss_beyond_max_gap_starts_a_new_track(vertical_axis, scale, tracking_config):
    """Scientific honesty: a gap longer than allowed must not be bridged."""
    tracking_config.max_gap = 3
    dets = straight_track(14, skip={4, 5, 6, 7})
    tracks, _ = track_detections(dets, 14, vertical_axis, scale, tracking_config)
    assert len(tracks) == 2
    assert [o.frame for o in tracks[0].observations] == [0, 1, 2, 3]
    assert [o.frame for o in tracks[1].observations] == [8, 9, 10, 11, 12, 13]


def test_max_gap_semantics_are_exact(vertical_axis, scale, tracking_config):
    """``max_gap = N`` means N consecutive missing frames, i.e. dt <= N + 1."""
    for max_gap in (0, 1, 2, 3, 5):
        tracking_config.max_gap = max_gap
        assert tracking_config.max_delta_frames() == max_gap + 1

        # Exactly max_gap missing frames: must stay one track.
        missing = set(range(1, 1 + max_gap))
        dets = straight_track(2 + max_gap + 2, skip=missing)
        tracks, _ = track_detections(
            dets, 2 + max_gap + 2, vertical_axis, scale, tracking_config
        )
        assert len(tracks) == 1, f"max_gap={max_gap} should tolerate {max_gap} gaps"

        # One more missing frame: must split.
        missing = set(range(1, 2 + max_gap))
        dets = straight_track(3 + max_gap + 2, skip=missing)
        tracks, _ = track_detections(
            dets, 3 + max_gap + 2, vertical_axis, scale, tracking_config
        )
        assert len(tracks) == 2, f"max_gap={max_gap} must reject {max_gap + 1} gaps"


def test_two_cells_same_direction_stay_separate(vertical_axis, scale, tracking_config):
    a = straight_track(8, y0=20.0, step=18.0, label=1)
    b = straight_track(8, y0=180.0, step=18.0, label=2)
    for d in b:
        d.label = 2
    tracks, _ = track_detections(a + b, 8, vertical_axis, scale, tracking_config)
    assert len(tracks) == 2
    assert all(t.n_obs == 8 for t in tracks)
    starts = sorted(t.observations[0].y for t in tracks)
    ends = sorted(t.observations[-1].y for t in tracks)
    assert starts[0] < starts[1] and ends[0] < ends[1], "tracks must not swap"


def test_two_cells_passing_close_do_not_swap_identity(
    vertical_axis, scale, tracking_config
):
    """One cell moving down, one moving up, approaching and separating."""
    dets = []
    for t in range(9):
        dets.append(make_detection(t, 45.0, 40.0 + 16.0 * t, label=1))
        dets.append(make_detection(t, 45.0, 200.0 - 16.0 * t, label=2))
    tracks, _ = track_detections(dets, 9, vertical_axis, scale, tracking_config)
    assert len(tracks) == 2
    for tr in tracks:
        ys = [o.y for o in tr.observations]
        deltas = np.diff(ys)
        # Each identity must keep one consistent direction throughout.
        assert np.all(deltas > 0) or np.all(deltas < 0), (
            "an identity reversed direction, which means the two cells swapped"
        )


def test_one_detection_is_never_shared_by_two_tracks(
    vertical_axis, scale, tracking_config
):
    dets = [
        make_detection(0, 45.0, 100.0, label=1),
        make_detection(0, 45.0, 140.0, label=2),
        make_detection(1, 45.0, 120.0, label=1),  # only one detection now
    ]
    tracks, events = track_detections(dets, 2, vertical_axis, scale, tracking_config)
    matched = [t for t in tracks if t.n_obs == 2]
    assert len(matched) <= 1, "a single detection was given to more than one track"
    assert events[1].n_matched <= 1


def test_merged_detection_leaves_the_loser_dormant_not_fabricated(
    vertical_axis, scale, tracking_config
):
    """Two cells, then one big mask covering both, then two again."""
    dets = [
        make_detection(0, 45.0, 100.0, label=1, area=600),
        make_detection(0, 45.0, 130.0, label=2, area=600),
        # frame 1: a single large detection where both cells were
        make_detection(1, 45.0, 116.0, label=1, area=1400, major=130),
        make_detection(2, 45.0, 104.0, label=1, area=600),
        make_detection(2, 45.0, 134.0, label=2, area=600),
    ]
    tracks, events = track_detections(dets, 3, vertical_axis, scale, tracking_config)
    # Exactly one track may claim the merged mask; no position is invented.
    rows_at_1 = [o for t in tracks for o in t.observations if o.frame == 1]
    assert len(rows_at_1) == 1, "a centroid was fabricated for the merged frame"
    assert events[1].merge_suspected, "the ambiguity was not flagged"


def test_no_detections_anywhere_is_safe(vertical_axis, scale, tracking_config):
    tracks, events = track_detections([], 6, vertical_axis, scale, tracking_config)
    assert tracks == []
    assert len(events) == 6
    assert all(e.n_detections == 0 for e in events)


def test_new_cell_entering_starts_a_track(vertical_axis, scale, tracking_config):
    dets = straight_track(6) + straight_track(3, y0=250.0, start_frame=3, label=2)
    tracks, _ = track_detections(dets, 6, vertical_axis, scale, tracking_config)
    assert len(tracks) == 2
    late = min(tracks, key=lambda t: t.observations[0].frame * -1)
    assert late.observations[0].frame == 3


def test_departing_cell_terminates(vertical_axis, scale, tracking_config):
    tracking_config.max_gap = 2
    dets = straight_track(4)
    tracks, _ = track_detections(dets, 12, vertical_axis, scale, tracking_config)
    assert len(tracks) == 1
    assert tracks[0].state is TrackState.TERMINATED
    assert tracks[0].n_obs == 4


def test_stalling_cell_is_not_split(vertical_axis, scale, tracking_config):
    """A cell that brakes hard keeps its identity.

    This is the failure mode seen in 052924_t1: the cell decelerates from
    ~45 px per frame to nearly zero as it stops in the channel.
    """
    ys = [20, 65, 110, 155, 200, 212, 213, 213, 214, 224]
    dets = [make_detection(t, 45.0, y) for t, y in enumerate(ys)]
    tracks, _ = track_detections(dets, len(ys), vertical_axis, scale, tracking_config)
    assert len(tracks) == 1
    assert tracks[0].n_obs == len(ys)


# --------------------------------------------------------------------------
# Gating
# --------------------------------------------------------------------------


def _track_at(frame: int, x: float, y: float, **kw) -> Track:
    tr = Track(id=1)
    tr.observe(make_detection(frame, x, y, **kw), cost=None)
    return tr


def test_lateral_jump_is_refused(vertical_axis, scale, tracking_config):
    tr = _track_at(0, 45.0, 100.0)
    sideways = make_detection(1, 45.0 + 40.0, 120.0)
    result = pair_cost(tr, sideways, vertical_axis, scale, tracking_config)
    assert result.gated == GATE_PERP
    assert result.total >= FORBIDDEN


def test_implausible_speed_is_refused(vertical_axis, scale, tracking_config):
    tr = _track_at(0, 45.0, 20.0)
    # 5 um/min over 20.007 min is 100 um, i.e. 214 px. 400 px is far beyond.
    far = make_detection(1, 45.0, 420.0)
    result = pair_cost(tr, far, vertical_axis, scale, tracking_config)
    assert result.gated == GATE_SPEED


def test_large_area_discontinuity_is_refused(vertical_axis, scale, tracking_config):
    tr = _track_at(0, 45.0, 100.0, area=800)
    huge = make_detection(1, 45.0, 118.0, area=800 * 5)
    result = pair_cost(tr, huge, vertical_axis, scale, tracking_config)
    assert result.gated == GATE_AREA


def test_gap_longer_than_allowed_is_refused(vertical_axis, scale, tracking_config):
    tracking_config.max_gap = 2
    tr = _track_at(0, 45.0, 100.0)
    late = make_detection(4, 45.0, 130.0)
    result = pair_cost(tr, late, vertical_axis, scale, tracking_config)
    assert result.gated == GATE_GAP


def test_cross_channel_association_is_refused(scale, tracking_config):
    from corridor.core.confinement import Channel, ConfinementAxis

    axis = ConfinementAxis(
        ux=0.0, uy=1.0, source="configured", confidence=1.0,
        angle_sigma_rad=math.radians(0.5),
        channels=[
            Channel(index=0, origin=(45.0, 0.0), half_width_px=40.0),
            Channel(index=1, origin=(127.0, 0.0), half_width_px=40.0),
        ],
    )
    tr = _track_at(0, 45.0, 100.0, channel=0)
    other = make_detection(1, 127.0, 110.0, channel=1)
    result = pair_cost(tr, other, axis, scale, tracking_config)
    assert result.gated in (GATE_CHANNEL, GATE_PERP)


def test_tracks_never_cross_a_channel_wall(scale, tracking_config):
    from corridor.core.confinement import Channel, ConfinementAxis

    axis = ConfinementAxis(
        ux=0.0, uy=1.0, source="configured", confidence=1.0,
        angle_sigma_rad=math.radians(0.5),
        channels=[
            Channel(index=0, origin=(45.0, 0.0), half_width_px=40.0),
            Channel(index=1, origin=(127.0, 0.0), half_width_px=40.0),
        ],
    )
    dets = []
    for t in range(6):
        dets.append(make_detection(t, 45.0, 30.0 + 15.0 * t, label=1, channel=0))
        dets.append(make_detection(t, 127.0, 30.0 + 15.0 * t, label=2, channel=1))
    tracks, _ = track_detections(dets, 6, axis, scale, tracking_config)
    assert len(tracks) == 2
    for tr in tracks:
        channels = {o.channel for o in tr.observations}
        assert len(channels) == 1, "a track moved between confinement channels"


# --------------------------------------------------------------------------
# Prediction and velocity state
# --------------------------------------------------------------------------


def test_prediction_scales_with_elapsed_frames(vertical_axis, scale, tracking_config):
    tr = Track(id=1)
    tr.observe(make_detection(0, 45.0, 10.0), cost=None)
    tr.observe(make_detection(1, 45.0, 30.0), cost=1.0)
    assert tr.velocity[1] == pytest.approx(20.0)
    assert tr.predict(2)[1] == pytest.approx(50.0)
    assert tr.predict(4)[1] == pytest.approx(90.0), (
        "a three-frame gap must not predict the same place as a one-frame gap"
    )


def test_prediction_backwards_is_an_error(vertical_axis):
    tr = Track(id=1)
    tr.observe(make_detection(5, 45.0, 10.0), cost=None)
    with pytest.raises(ValueError):
        tr.predict(5)


def test_velocity_is_divided_by_elapsed_frames():
    """The defect this guards: reacquiring after a gap must not inflate speed."""
    tr = Track(id=1)
    tr.observe(make_detection(0, 45.0, 10.0), cost=None)
    tr.observe(make_detection(4, 45.0, 90.0), cost=1.0)  # 80 px over 4 frames
    assert tr.velocity[1] == pytest.approx(20.0), (
        "velocity must be per frame, not per observation"
    )


def test_velocity_smoothing_after_first_link():
    tr = Track(id=1)
    tr.observe(make_detection(0, 45.0, 0.0), cost=None)
    tr.observe(make_detection(1, 45.0, 10.0), cost=1.0)
    assert tr.velocity[1] == pytest.approx(10.0)
    tr.observe(make_detection(2, 45.0, 30.0), cost=1.0)
    # 0.7 * 10 + 0.3 * 20
    assert tr.velocity[1] == pytest.approx(13.0)


# --------------------------------------------------------------------------
# Assignment mechanics
# --------------------------------------------------------------------------


def test_assignment_matrix_shape_and_blocks():
    costs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    m = build_assignment_matrix(costs, unmatched_cost=10.0)
    assert m.shape == (5, 5)
    assert np.allclose(m[:3, :2], costs)
    assert m[0, 2] == 10.0 and m[1, 3] == 10.0 and m[2, 4] == 10.0
    assert m[0, 3] >= FORBIDDEN
    assert m[3, 0] == 10.0 and m[4, 1] == 10.0
    assert np.allclose(m[3:, 2:], 0.0)


def test_unmatched_is_chosen_when_every_pairing_is_expensive():
    costs = np.array([[100.0, 100.0], [100.0, 100.0]])
    matches, un_t, un_d = solve_assignment(costs, unmatched_cost=10.0)
    assert matches == []
    assert sorted(un_t) == [0, 1]
    assert sorted(un_d) == [0, 1]


def test_a_pairing_beats_two_unmatched_decisions():
    """Linking removes one birth and one death, so the threshold is 2 * U."""
    costs = np.array([[19.0]])
    matches, _, _ = solve_assignment(costs, unmatched_cost=10.0)
    assert matches == [(0, 0)], "cost below 2*U should be linked"

    costs = np.array([[21.0]])
    matches, un_t, un_d = solve_assignment(costs, unmatched_cost=10.0)
    assert matches == []
    assert un_t == [0] and un_d == [0]


def test_forbidden_pairings_are_never_selected():
    costs = np.array([[FORBIDDEN, 5.0], [FORBIDDEN, FORBIDDEN]])
    matches, un_t, un_d = solve_assignment(costs, unmatched_cost=100.0)
    assert matches == [(0, 1)]
    assert un_t == [1]
    assert un_d == [0]


def test_optimiser_prefers_the_globally_better_pairing():
    """A greedy nearest-neighbour tracker gets this wrong; the LAP must not."""
    costs = np.array([[1.0, 0.9], [12.0, 1.1]])
    matches, _, _ = solve_assignment(costs, unmatched_cost=50.0)
    assert sorted(matches) == [(0, 0), (1, 1)]


def test_empty_inputs_are_handled():
    assert solve_assignment(np.zeros((0, 0)), 10.0) == ([], [], [])
    assert solve_assignment(np.zeros((0, 3)), 10.0) == ([], [], [0, 1, 2])
    assert solve_assignment(np.zeros((2, 0)), 10.0) == ([], [0, 1], [])


# --------------------------------------------------------------------------
# Cost structure
# --------------------------------------------------------------------------


def test_cost_terms_are_reported_separately(vertical_axis, scale, tracking_config):
    tr = _track_at(0, 45.0, 100.0, area=800)
    det = make_detection(1, 46.0, 120.0, area=900)
    breakdown = pair_cost(tr, det, vertical_axis, scale, tracking_config)
    assert breakdown.allowed
    assert breakdown.total == pytest.approx(
        breakdown.along + breakdown.across + breakdown.area
        + breakdown.direction + breakdown.orientation
    )


def test_lateral_motion_costs_more_than_along_channel_motion(
    vertical_axis, scale, tracking_config
):
    """The confinement prior, stated as a test rather than as a weight of 1000."""
    tr_a = _track_at(0, 45.0, 100.0)
    along = pair_cost(
        tr_a, make_detection(1, 45.0, 110.0), vertical_axis, scale, tracking_config
    )
    tr_b = _track_at(0, 45.0, 100.0)
    across = pair_cost(
        tr_b, make_detection(1, 55.0, 100.0), vertical_axis, scale, tracking_config
    )
    assert across.total > along.total * 3, (
        "moving 10 px sideways must cost far more than 10 px along the channel"
    )
