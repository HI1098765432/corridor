"""Tracking behaviour, tested without Cellpose.

Every scenario here is one the supplied datasets actually contain or that the
assignment model must survive: a cell that stalls, a cell that disappears for a
frame, two cells in one channel, a detection that cannot belong to anything.
None of them passes a migration axis: the tracker takes its anisotropy from
each cell's own body.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from corridor.core.config import CHANNEL_CONSTRAINT_OFF, Scale, TrackingConfig
from corridor.core.geometry import GEOMETRY_FROM_RIDGES, ChannelGeometry, Lane
from corridor.core.tracking import (
    FORBIDDEN,
    GATE_AREA,
    GATE_COST,
    GATE_GAP,
    GATE_LANE,
    GATE_MOTION,
    GATE_SPEED,
    KalmanTracker,
    MotionModel,
    TrackState,
    build_assignment_matrix,
    link_margins,
    motion_gate_chi2,
    pair_cost,
    solve_assignment,
    start_track,
    track_detections,
)

from conftest import identity_of, make_detection, straight_track


# --------------------------------------------------------------------------
# Whole-sequence behaviour (v1 tests 39-200, axis removed)
# --------------------------------------------------------------------------


def test_single_cell_moving_straight_is_one_track(scale, tracking_config):
    dets = straight_track(10)
    tracks, _ = track_detections(dets, 10, scale, tracking_config)
    assert len(tracks) == 1
    assert tracks[0].n_obs == 10
    assert [o.frame for o in tracks[0].observations] == list(range(10))


def test_one_frame_loss_keeps_identity(scale, tracking_config):
    dets = straight_track(10, skip={4})
    tracks, _ = track_detections(dets, 10, scale, tracking_config)
    assert len(tracks) == 1, "a single missing frame must not split the track"
    observed = [o.frame for o in tracks[0].observations]
    assert observed == [0, 1, 2, 3, 5, 6, 7, 8, 9]
    reacquired = next(o for o in tracks[0].observations if o.frame == 5)
    assert reacquired.gap_frames == 2


def test_multi_frame_loss_within_max_gap_keeps_identity(scale, tracking_config):
    tracking_config.max_gap = 3
    dets = straight_track(12, skip={4, 5, 6})
    tracks, _ = track_detections(dets, 12, scale, tracking_config)
    assert len(tracks) == 1
    reacquired = next(o for o in tracks[0].observations if o.frame == 7)
    assert reacquired.gap_frames == 4


def test_loss_beyond_max_gap_starts_a_new_track(scale, tracking_config):
    """Scientific honesty: a gap longer than allowed must not be bridged."""
    tracking_config.max_gap = 3
    dets = straight_track(14, skip={4, 5, 6, 7})
    tracks, _ = track_detections(dets, 14, scale, tracking_config)
    assert len(tracks) == 2
    assert [o.frame for o in tracks[0].observations] == [0, 1, 2, 3]
    assert [o.frame for o in tracks[1].observations] == [8, 9, 10, 11, 12, 13]


@pytest.mark.parametrize("closing", [True, False])
def test_max_gap_semantics_are_exact(scale, tracking_config, closing):
    """``max_gap = N`` means N consecutive missing frames, i.e. dt <= N + 1.

    Both stages obey it: global gap closing never bridges further than the
    frame-to-frame step may.
    """
    tracking_config.global_gap_closing = closing
    for max_gap in (0, 1, 2, 3, 5):
        tracking_config.max_gap = max_gap
        assert tracking_config.max_delta_frames() == max_gap + 1

        missing = set(range(1, 1 + max_gap))
        dets = straight_track(2 + max_gap + 2, skip=missing)
        tracks, _ = track_detections(dets, 2 + max_gap + 2, scale, tracking_config)
        assert len(tracks) == 1, f"max_gap={max_gap} should tolerate {max_gap} gaps"

        missing = set(range(1, 2 + max_gap))
        dets = straight_track(3 + max_gap + 2, skip=missing)
        tracks, _ = track_detections(dets, 3 + max_gap + 2, scale, tracking_config)
        assert len(tracks) == 2, f"max_gap={max_gap} must reject {max_gap + 1} gaps"


def test_two_cells_same_direction_stay_separate(scale, tracking_config):
    a = straight_track(8, y0=20.0, step=18.0, label=1)
    b = straight_track(8, y0=180.0, step=18.0, label=2)
    tracks, _ = track_detections(a + b, 8, scale, tracking_config)
    assert len(tracks) == 2
    for frame in range(8):
        assert identity_of(tracks, frame, 1) == identity_of(tracks, 0, 1)
        assert identity_of(tracks, frame, 2) == identity_of(tracks, 0, 2)


def test_two_cells_passing_close_do_not_swap_identity(scale, tracking_config):
    """One cell moving down, one moving up, approaching and separating."""
    dets = []
    for t in range(9):
        dets.append(make_detection(t, 45.0, 40.0 + 16.0 * t, label=1))
        dets.append(make_detection(t, 45.0, 200.0 - 16.0 * t, label=2))
    tracks, _ = track_detections(dets, 9, scale, tracking_config)
    assert len(tracks) == 2
    for tr in tracks:
        ys = [o.y for o in tr.observations]
        deltas = np.diff(ys)
        # Each identity must keep one consistent direction throughout. (At
        # t = 5 the two detections are identical, so labels prove nothing
        # there; direction does.)
        assert np.all(deltas > 0) or np.all(deltas < 0), (
            "an identity reversed direction, which means the two cells swapped"
        )


def test_one_detection_is_never_shared_by_two_tracks(scale, tracking_config):
    dets = [
        make_detection(0, 45.0, 100.0, label=1),
        make_detection(0, 45.0, 140.0, label=2),
        make_detection(1, 45.0, 120.0, label=1),  # only one detection now
    ]
    tracks, events = track_detections(dets, 2, scale, tracking_config)
    matched = [t for t in tracks if t.n_obs == 2]
    assert len(matched) <= 1, "a single detection was given to more than one track"
    assert events[1].n_matched <= 1


def test_merged_detection_leaves_the_loser_dormant_not_fabricated(scale, tracking_config):
    """Two cells, then one big mask covering both, then two again."""
    dets = [
        make_detection(0, 45.0, 100.0, label=1, area=600),
        make_detection(0, 45.0, 130.0, label=2, area=600),
        make_detection(1, 45.0, 116.0, label=1, area=1400, major=130),
        make_detection(2, 45.0, 104.0, label=1, area=600),
        make_detection(2, 45.0, 134.0, label=2, area=600),
    ]
    tracks, events = track_detections(dets, 3, scale, tracking_config)
    rows_at_1 = [o for t in tracks for o in t.observations if o.frame == 1]
    assert len(rows_at_1) == 1, "a centroid was fabricated for the merged frame"
    assert events[1].merge_suspected, "the ambiguity was not flagged"
    flagged = [t for t in tracks if "merge_suspected" in t.flags]
    assert [t.id for t in flagged] == events[1].merge_suspected
    assert "merge suspected" in events[1].to_row()["notes"]


def test_no_detections_anywhere_is_safe(scale, tracking_config):
    tracks, events = track_detections([], 6, scale, tracking_config)
    assert list(tracks) == []
    assert len(events) == 6
    assert all(e.n_detections == 0 for e in events)


def test_new_cell_entering_starts_a_track(scale, tracking_config):
    dets = straight_track(6) + straight_track(3, y0=250.0, start_frame=3, label=2)
    tracks, _ = track_detections(dets, 6, scale, tracking_config)
    assert len(tracks) == 2
    assert identity_of(tracks, 3, 2) != identity_of(tracks, 3, 1)
    late = next(t for t in tracks if t.id == identity_of(tracks, 3, 2))
    assert late.observations[0].frame == 3


def test_departing_cell_terminates(scale, tracking_config):
    tracking_config.max_gap = 2
    dets = straight_track(4)
    tracks, _ = track_detections(dets, 12, scale, tracking_config)
    assert len(tracks) == 1
    assert tracks[0].state is TrackState.TERMINATED
    assert tracks[0].n_obs == 4


def test_stalling_cell_is_not_split(scale, tracking_config):
    """A cell that brakes hard keeps its identity.

    This is the failure mode seen in 052924_t1: the cell decelerates from
    ~45 px per frame to nearly zero as it stops in the channel.
    """
    ys = [20, 65, 110, 155, 200, 212, 213, 213, 214, 224]
    dets = [make_detection(t, 45.0, y) for t, y in enumerate(ys)]
    tracks, _ = track_detections(dets, len(ys), scale, tracking_config)
    assert len(tracks) == 1
    assert tracks[0].n_obs == len(ys)


def test_ids_are_numbered_by_first_appearance(scale, tracking_config):
    late = straight_track(3, x=200.0, start_frame=4, label=2)
    early = straight_track(8, label=1)
    tracks, _ = track_detections(late + early, 8, scale, tracking_config)
    assert [t.id for t in tracks] == [1, 2]
    assert tracks[0].observations[0].frame == 0
    assert tracks[1].observations[0].frame == 4


# --------------------------------------------------------------------------
# Gating
# --------------------------------------------------------------------------


def _track_at(scale, cfg, frame: int, x: float, y: float, **kw):
    return start_track(make_detection(frame, x, y, **kw), scale, cfg)


def test_implausible_speed_is_refused(scale, tracking_config):
    tr = _track_at(scale, tracking_config, 0, 45.0, 20.0)
    # 5 um/min over 20.007 min is 100 um, i.e. 214 px. 400 px is far beyond.
    result = pair_cost(tr, make_detection(1, 45.0, 420.0), scale, tracking_config)
    assert result.gated == GATE_SPEED
    assert result.total >= FORBIDDEN


def test_large_area_discontinuity_is_refused(scale, tracking_config):
    tr = _track_at(scale, tracking_config, 0, 45.0, 100.0, area=800)
    result = pair_cost(tr, make_detection(1, 45.0, 118.0, area=800 * 5), scale, tracking_config)
    assert result.gated == GATE_AREA


def test_gap_longer_than_allowed_is_refused(scale, tracking_config):
    tracking_config.max_gap = 2
    tr = _track_at(scale, tracking_config, 0, 45.0, 100.0)
    result = pair_cost(tr, make_detection(4, 45.0, 130.0), scale, tracking_config)
    assert result.gated == GATE_GAP


def test_motion_outlier_is_refused_for_a_moving_track(scale, tracking_config):
    """Once a track has a velocity, a jump far off its prediction is gated.

    The speed is legal (80 px in one frame is 1.9 um/min); what is not is
    landing 80 px sideways of a body that has been moving straight down.
    """
    tr = _track_at(scale, tracking_config, 0, 45.0, 20.0)
    for t in range(1, 5):
        tr.observe(make_detection(t, 45.0, 20.0 + 20.0 * t), cost=0.0)
    result = pair_cost(tr, make_detection(5, 125.0, 120.0), scale, tracking_config)
    assert result.gated == GATE_MOTION
    assert result.mahalanobis > motion_gate_chi2(2)


def test_motion_gate_is_the_chi_square_quantile():
    assert motion_gate_chi2(2) == pytest.approx(13.8155, abs=1e-3)
    assert motion_gate_chi2(3) == pytest.approx(16.2662, abs=1e-3)


def test_the_cost_gate_is_twice_the_unmatched_cost(scale):
    """``gate = 2U`` is derived, so lowering U tightens the gate with it."""
    cfg = TrackingConfig(unmatched_chi2=0.5)
    assert cfg.effective_gate_chi2 == 1.0
    tr = _track_at(scale, cfg, 0, 45.0, 100.0, area=800)
    # A legal but visible size change costs (ln 1.3 / 0.3)^2 = 0.77 alone.
    result = pair_cost(tr, make_detection(1, 45.0, 140.0, area=1040), scale, cfg)
    assert result.gated == GATE_COST


def test_a_link_across_lanes_is_refused(scale, tracking_config):
    geometry = _two_lanes()
    tr = start_track(make_detection(0, 45.0, 100.0), scale, tracking_config, lane=0)
    result = pair_cost(
        tr, make_detection(1, 127.0, 110.0), scale, tracking_config, geometry=geometry
    )
    assert result.gated == GATE_LANE


def test_the_lane_gate_can_be_switched_off(scale):
    cfg = TrackingConfig(channel_constraint=CHANNEL_CONSTRAINT_OFF, max_speed_um_per_min=50.0)
    geometry = _two_lanes()
    tr = start_track(make_detection(0, 45.0, 100.0), scale, cfg, lane=0)
    result = pair_cost(tr, make_detection(1, 127.0, 110.0), scale, cfg, geometry=geometry)
    assert result.gated != GATE_LANE


def test_tracks_never_cross_a_lane_wall(scale, tracking_config):
    dets = []
    for t in range(6):
        dets.append(make_detection(t, 45.0, 30.0 + 15.0 * t, label=1))
        dets.append(make_detection(t, 127.0, 30.0 + 15.0 * t, label=2))
    tracks, _ = track_detections(dets, 6, scale, tracking_config, geometry=_two_lanes())
    assert len(tracks) == 2
    for tr in tracks:
        assert len({o.channel for o in tr.observations}) == 1, "a track changed lane"


def test_lanes_that_were_not_measured_from_walls_never_gate(scale):
    cfg = TrackingConfig(max_speed_um_per_min=50.0)
    geometry = _two_lanes()
    geometry.applied = False
    tr = start_track(make_detection(0, 45.0, 100.0), scale, cfg, lane=0)
    result = pair_cost(tr, make_detection(1, 127.0, 110.0), scale, cfg, geometry=geometry)
    assert result.gated != GATE_LANE


def _two_lanes() -> ChannelGeometry:
    lanes = [
        Lane(index=0, origin=(45.0, 0.0), direction=(0.0, 1.0), half_width_px=41.0),
        Lane(index=1, origin=(127.0, 0.0), direction=(0.0, 1.0), half_width_px=41.0),
    ]
    return ChannelGeometry(
        lanes=lanes, source=GEOMETRY_FROM_RIDGES, confidence=1.0, pitch_px=82.0, applied=True
    )


# --------------------------------------------------------------------------
# The Kalman state
# --------------------------------------------------------------------------


def test_prediction_scales_with_elapsed_frames(scale, tracking_config):
    tr = _track_at(scale, tracking_config, 0, 45.0, 10.0)
    tr.observe(make_detection(1, 45.0, 30.0), cost=1.0)
    v = tr.velocity[1]
    assert v == pytest.approx(20.0, rel=0.05)
    filtered_y = tr.kalman_state[0][1]
    assert filtered_y == pytest.approx(30.0, abs=1.0)
    assert tr.predict(2)[1] == pytest.approx(filtered_y + v)
    assert tr.predict(4)[1] == pytest.approx(filtered_y + 3.0 * v), (
        "a three-frame gap must not predict the same place as a one-frame gap"
    )


def test_uncertainty_widens_with_the_gap(scale, tracking_config):
    tr = _track_at(scale, tracking_config, 0, 45.0, 10.0)
    tr.observe(make_detection(1, 45.0, 30.0), cost=1.0)
    _, p1 = tr.predict_state(2)
    _, p4 = tr.predict_state(5)
    assert np.trace(p4[:2, :2]) > 4.0 * np.trace(p1[:2, :2])


def test_prediction_backwards_is_an_error(scale, tracking_config):
    tr = _track_at(scale, tracking_config, 5, 45.0, 10.0)
    with pytest.raises(ValueError):
        tr.predict(5)


def test_velocity_is_divided_by_elapsed_frames(scale, tracking_config):
    """The defect this guards: reacquiring after a gap must not inflate speed."""
    tr = _track_at(scale, tracking_config, 0, 45.0, 10.0)
    tr.observe(make_detection(4, 45.0, 90.0), cost=1.0)  # 80 px over 4 frames
    assert tr.velocity[1] == pytest.approx(20.0, rel=0.05), (
        "velocity must be per frame, not per observation"
    )


def test_measurement_noise_follows_the_cell_body(scale, tracking_config):
    """R is long along the cell and short across it, whatever its orientation."""
    model = MotionModel.from_config(scale, tracking_config)
    for angle in (0.0, math.pi / 2, math.radians(30)):
        det = make_detection(0, 45.0, 45.0, orientation_rad=angle)
        R = model.measurement_cov(det)
        u = det.axis_unit
        n = np.array([-u[1], u[0]])
        along, across = float(u @ R @ u), float(n @ R @ n)
        assert along > 10.0 * across
        expected_along = model.position_sigma_px**2 + (0.06 * 90.0) ** 2
        assert along == pytest.approx(expected_along, rel=1e-9)


def test_sideways_jump_costs_more_than_moving_along_the_body(scale, tracking_config):
    """The confinement prior, now stated about the cell rather than the device.

    A vertically elongated fresh cell (no velocity yet): 10 px sideways must
    cost more than 30 px along its own long axis. v1 asserted the same thing
    about a migration axis.
    """
    tr_a = _track_at(scale, tracking_config, 0, 45.0, 100.0)
    along = pair_cost(tr_a, make_detection(1, 45.0, 130.0), scale, tracking_config)
    tr_b = _track_at(scale, tracking_config, 0, 45.0, 100.0)
    across = pair_cost(tr_b, make_detection(1, 55.0, 100.0), scale, tracking_config)
    assert along.allowed and across.allowed
    assert across.motion > along.motion
    assert across.total > along.total


def test_the_same_holds_for_a_horizontal_cell(scale, tracking_config):
    """No direction is privileged: rotate the cell and the cost rotates with it."""
    horizontal = dict(orientation_rad=math.pi / 2)
    tr_a = _track_at(scale, tracking_config, 0, 100.0, 45.0, **horizontal)
    along = pair_cost(
        tr_a, make_detection(1, 130.0, 45.0, **horizontal), scale, tracking_config
    )
    tr_b = _track_at(scale, tracking_config, 0, 100.0, 45.0, **horizontal)
    across = pair_cost(
        tr_b, make_detection(1, 100.0, 55.0, **horizontal), scale, tracking_config
    )
    assert across.total > along.total


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
    matches, _, _ = solve_assignment(np.array([[19.0]]), unmatched_cost=10.0)
    assert matches == [(0, 0)], "cost below 2*U should be linked"
    matches, un_t, un_d = solve_assignment(np.array([[21.0]]), unmatched_cost=10.0)
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


def test_link_margin_is_the_gap_to_the_next_best_explanation():
    # A lone link: the next-best explanation is leaving both unmatched (2U).
    assert link_margins(np.array([[5.0]]), [(0, 0)], 10.0)[(0, 0)] == pytest.approx(15.0)
    # Two links: forbidding either one makes the best alternative the swap,
    # 4 + 12 = 16 against the chosen 1 + 2 = 3. The local "next-best for its
    # row or column" (4 - 1 = 3) would ignore that the swap also costs the
    # other link its detection.
    costs = np.array([[1.0, 4.0], [12.0, 2.0]])
    margins = link_margins(costs, [(0, 0), (1, 1)], unmatched_cost=10.0)
    assert margins[(0, 0)] == pytest.approx(13.0)
    assert margins[(1, 1)] == pytest.approx(13.0)


def test_link_margin_is_never_negative_under_a_global_assignment():
    """The case that made the local definition read -13.5 on real data.

    Detection 0's cheapest track is track 1 (2.8), but the global optimum
    gives it to track 0, because track 1 is needed for detection 1.
    """
    costs = np.array([[16.4, FORBIDDEN], [2.8, 3.0]])
    matches, _, _ = solve_assignment(costs, unmatched_cost=15.0)
    assert sorted(matches) == [(0, 0), (1, 1)]
    margins = link_margins(costs, matches, 15.0)
    assert all(m >= 0.0 for m in margins.values())
    # Forbidding (0, 0): track 1 takes detection 0, detection 1 and track 0
    # go unmatched: 2.8 + 30 = 32.8 against 19.4.
    assert margins[(0, 0)] == pytest.approx(13.4)


def test_every_link_carries_its_margin_and_cost(scale, tracking_config):
    dets = straight_track(5)
    tracks, _ = track_detections(dets, 5, scale, tracking_config)
    first, *rest = tracks[0].observations
    assert first.link_margin is None and first.cost is None
    for obs in rest:
        assert obs.cost is not None and obs.link_margin is not None
        assert 0.0 < obs.link_margin <= tracking_config.effective_gate_chi2
        assert obs.detection is not None


def test_close_neighbours_make_a_small_margin(scale, tracking_config):
    """Two candidates 6 px apart is ambiguous; one candidate is not."""
    lone = straight_track(3)
    tracks, _ = track_detections(lone, 3, scale, tracking_config)
    clear = tracks[0].observations[-1].link_margin

    crowded = straight_track(3) + [make_detection(2, 45.0, 66.0, label=2)]
    tracks, _ = track_detections(crowded, 3, scale, tracking_config)
    holder = next(t for t in tracks if t.n_obs == 3)
    assert holder.observations[-1].link_margin < clear


# --------------------------------------------------------------------------
# Cost structure
# --------------------------------------------------------------------------


def test_cost_terms_are_reported_separately(scale, tracking_config):
    tr = _track_at(scale, tracking_config, 0, 45.0, 100.0, area=800)
    det = make_detection(1, 46.0, 120.0, area=900)
    breakdown = pair_cost(tr, det, scale, tracking_config)
    assert breakdown.allowed
    assert breakdown.total == pytest.approx(
        breakdown.motion + breakdown.size + breakdown.shape + breakdown.orientation
        + breakdown.direction + breakdown.overlap + breakdown.gap
    )
    assert breakdown.size == pytest.approx((math.log(900 / 800) / 0.30) ** 2)


def test_gap_penalty_is_charged_per_missing_frame(scale, tracking_config):
    tr = _track_at(scale, tracking_config, 0, 45.0, 100.0)
    b = pair_cost(tr, make_detection(3, 45.0, 100.0), scale, tracking_config)
    assert b.gap == pytest.approx(2.0 * tracking_config.gap_penalty_chi2)


def test_overlap_term_uses_the_masks_when_both_have_one(scale, tracking_config):
    tr = start_track(make_detection(0, 45.0, 100.0, with_mask=True), scale, tracking_config)
    same = pair_cost(tr, make_detection(1, 45.0, 100.0, with_mask=True), scale, tracking_config)
    assert same.iou == pytest.approx(1.0)
    assert same.overlap == pytest.approx(0.0)
    moved = pair_cost(tr, make_detection(1, 45.0, 140.0, with_mask=True), scale, tracking_config)
    assert 0.0 < moved.iou < 1.0
    assert moved.overlap > same.overlap
    bare = pair_cost(tr, make_detection(1, 45.0, 140.0), scale, tracking_config)
    assert bare.iou is None and bare.overlap == 0.0


def test_the_direction_term_charges_a_reversal(scale, tracking_config):
    tr = _track_at(scale, tracking_config, 0, 45.0, 20.0)
    for t in range(1, 4):
        tr.observe(make_detection(t, 45.0, 20.0 + 20.0 * t), cost=0.0)
    forward = pair_cost(tr, make_detection(4, 45.0, 95.0), scale, tracking_config)
    backward = pair_cost(tr, make_detection(4, 45.0, 70.0), scale, tracking_config)
    assert forward.direction == pytest.approx(0.0, abs=1e-6)
    assert backward.direction == pytest.approx(tracking_config.w_reversal)


# --------------------------------------------------------------------------
# Transition: the v1 argument order still runs
# --------------------------------------------------------------------------


def test_the_v1_signature_is_still_accepted(vertical_axis, scale, tracking_config):
    dets = straight_track(6)
    tracks, events = track_detections(dets, 6, vertical_axis, scale, tracking_config)
    assert len(tracks) == 1 and len(events) == 6
    tracker = KalmanTracker(vertical_axis, scale, tracking_config)
    assert tracker.scale is scale and tracker.cfg is tracking_config


def test_uncalibrated_defaults_refuse_rather_than_guess(tracking_config):
    """5 um/min read as 5 px/frame rejects a 20 px step, in both stages."""
    bare = Scale.from_values(None, None)
    tracks, _ = track_detections(straight_track(4, step=20.0), 4, bare, tracking_config)
    assert len(tracks) == 4
