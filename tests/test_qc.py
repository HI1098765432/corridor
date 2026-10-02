"""Quality-control findings: what a trajectory is made of, and what the result rests on.

The first group exists because of a measurement, not a hunch. Running the
fallback ladder over the supplied stacks (scripts/experiment_fallback.py) showed
that the assignment step does not reject a false detection which repeats in the
same place: a stationary object is the most self-consistent thing a
predicted-position cost model can be shown. So the software says so instead.

The rest pin the v2 codes (contract §8) on synthetic tracks and events whose
answer is known, and pin that the two axis codes can no longer be produced.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from corridor.core.config import Scale, TrackingConfig
from corridor.core.detections import FrameDiagnostics
from corridor.core.geometry import GEOMETRY_FROM_RIDGES, ChannelGeometry, Lane
from corridor.core.imaging import SOURCE_USER, Calibrated, StackMetadata
from corridor.core.measurements import summarise
from corridor.core.qc import (
    LINK_AMBIGUOUS_MARGIN_CHI2,
    MORPHOLOGY_JUMP_CHI2,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARN,
    SIZE_JUMP_SIGMAS,
    STATIONARY_MIN_MINUTES,
    STATIONARY_NET_UM,
    Issue,
    QCIssue,
    collect_issues,
)
from corridor.core.tracking import (
    CostBreakdown,
    FrameEvent,
    Track,
    UnlinkedStart,
    track_detections,
)

from conftest import (
    FRAME_INTERVAL_MIN,
    PIXEL_SIZE_UM,
    make_detection,
    straight_track,
)


@pytest.fixture
def metadata(tmp_path) -> StackMetadata:
    """A calibrated stack, so calibration warnings do not drown the findings."""
    return StackMetadata(
        path=tmp_path / "synthetic.tif",
        n_frames=8,
        height=324,
        width=90,
        dtype="uint16",
        axes_raw="TYX",
        axes_interpretation="TYX",
        pixel_size_um=Calibrated(PIXEL_SIZE_UM, SOURCE_USER),
        frame_interval_min=Calibrated(FRAME_INTERVAL_MIN, SOURCE_USER),
    )


def issues_for(detections, n_frames, scale, cfg, metadata, **kwargs):
    tracks, events = track_detections(detections, n_frames, scale, cfg)
    summaries = summarise(tracks, scale)
    return collect_issues(
        metadata, scale, [], events, tracks, summaries, cfg, **kwargs
    ), tracks


def issues_of_tracks(tracks, scale, cfg, metadata, *, events=(), diagnostics=(), **kwargs):
    return collect_issues(
        metadata, scale, list(diagnostics), list(events), tracks,
        summarise(tracks, scale), cfg, **kwargs
    )


def hand_track(track_id: int, detections, *, margins=None) -> Track:
    """A track built observation by observation, bypassing assignment.

    For findings about what a track *contains*, the assignment that produced it
    is beside the point: building it directly makes the content exact.
    """
    track = Track(id=track_id)
    for k, det in enumerate(detections):
        first = not track.observations
        track.observe(
            det,
            cost=None if first else 1.0,
            link_margin=None if first else (margins[k] if margins else 10.0),
        )
    return track


def codes(issues) -> set[str]:
    return {i.code for i in issues}


def only(issues, code):
    found = [i for i in issues if i.code == code]
    assert found, f"no {code} among {sorted(codes(issues))}"
    return found


# --------------------------------------------------------------------------
# What the trajectory is made of (1.x findings, axis removed)
# --------------------------------------------------------------------------


def test_a_cell_that_never_moves_is_flagged(scale, tracking_config, metadata):
    """A fixed feature of the device tracks perfectly and migrates nowhere."""
    dets = [make_detection(frame, 45.0, 100.0) for frame in range(6)]
    issues, _ = issues_for(dets, 6, scale, tracking_config, metadata)

    assert "stationary_track" in codes(issues)
    flagged = next(i for i in issues if i.code == "stationary_track")
    assert flagged.track_id == 1
    assert "less than the width of a cell" in flagged.detail


def test_a_migrating_cell_is_not_flagged(scale, tracking_config, metadata):
    dets = straight_track(6, step=20.0)
    issues, _ = issues_for(dets, 6, scale, tracking_config, metadata)

    assert "stationary_track" not in codes(issues)


def test_a_brief_pause_is_not_called_stationary(scale, tracking_config, metadata):
    """Two frames is not enough sequence to accuse a cell of standing still."""
    dets = [make_detection(frame, 45.0, 100.0) for frame in range(2)]
    issues, _ = issues_for(dets, 2, scale, tracking_config, metadata)

    assert "stationary_track" not in codes(issues)


def test_a_fixed_object_with_a_jittery_centroid_is_still_caught(
    scale, tracking_config, metadata
):
    """The reason this test measures net displacement rather than path length.

    A wall artefact does not sit on exactly the same pixel every frame: its
    centroid wanders by a fraction of a pixel. Summed over ten frames that
    wander is several µm of "path", which is enough to slip past a path-length
    threshold — while the object has, of course, gone nowhere at all.
    """
    wobble = [0.0, 0.6, -0.5, 0.7, -0.6, 0.5, -0.7, 0.6, -0.5, 0.4]
    dets = [
        make_detection(frame, 45.0 + dx, 100.0 - dx)
        for frame, dx in enumerate(wobble)
    ]
    issues, _ = issues_for(dets, len(wobble), scale, tracking_config, metadata)

    assert "stationary_track" in codes(issues)


def test_a_migrating_cell_in_a_fast_acquisition_is_not_accused(tracking_config, metadata):
    """The reason the minimum is in minutes rather than frames.

    At half a minute per frame, four frames is two minutes of experiment. A
    cell moving at a perfectly healthy 2 µm/min covers 3 µm in that time, which
    is under the 4 µm threshold — and calling that cell "a fixed feature of the
    device" would teach the reader to ignore this warning.
    """
    fast = Scale.from_values(0.25, 0.5)
    # 2 um/min at 0.25 um/px and 0.5 min/frame is 4 px per frame.
    dets = [make_detection(frame, 45.0, 100.0 + 4.0 * frame) for frame in range(4)]
    issues, _ = issues_for(dets, 4, fast, tracking_config, metadata)

    assert "stationary_track" not in codes(issues)


def test_the_stationary_threshold_is_under_one_cell_width():
    """The constant must stay below a real cell, or every track trips it.

    The labelled cells are 9-15 px wide at 0.467060343 µm/px, so the narrowest
    is about 4.2 µm. A threshold at or above that would flag cells that moved
    their own width, which is migration.
    """
    assert STATIONARY_NET_UM < 9 * 0.467060342995564
    # And the time bar must be long enough that "went nowhere" is a statement.
    assert STATIONARY_MIN_MINUTES >= 30.0


def test_a_track_built_mostly_from_fallback_detections_is_flagged(
    scale, tracking_config, metadata
):
    """The recall/precision trade has to be visible in the result, not just the docs."""
    dets = straight_track(6, step=20.0)
    for det in dets[2:]:  # four of six positions came from a permissive pass
        det.source = "ensemble"
        det.confidence = 0.75

    issues, _ = issues_for(dets, 6, scale, tracking_config, metadata)

    assert "fallback_dependent_track" in codes(issues)
    flagged = next(i for i in issues if i.code == "fallback_dependent_track")
    assert "4 of its 6 positions" in flagged.detail


def test_an_ordinary_track_is_not_flagged_as_fallback_dependent(
    scale, tracking_config, metadata
):
    dets = straight_track(6, step=20.0)
    issues, _ = issues_for(dets, 6, scale, tracking_config, metadata)

    assert "fallback_dependent_track" not in codes(issues)


def test_one_borrowed_position_in_a_long_track_is_not_flagged(
    scale, tracking_config, metadata
):
    """The warning is about a trajectory that *rests* on the fallback.

    A single recovered position in an otherwise ordinary track is the fallback
    working as intended, and warning about it would train the reader to ignore
    the warning.
    """
    dets = straight_track(8, step=20.0)
    dets[3].source = "ensemble"

    issues, _ = issues_for(dets, 8, scale, tracking_config, metadata)
    assert "fallback_dependent_track" not in codes(issues)


# --------------------------------------------------------------------------
# The axis is gone
# --------------------------------------------------------------------------


def test_the_axis_codes_can_no_longer_be_produced(scale, tracking_config, metadata):
    """``weak_axis`` and ``lateral_drift`` needed a migration direction.

    A cell moving sideways across the image was ``lateral_drift`` in 1.x when
    the axis was vertical. In 2.0 nothing knows which way is "along".
    """
    sideways = [make_detection(f, 20.0 + 20.0 * f, 100.0, orientation_rad=math.pi / 2)
                for f in range(6)]
    geometry = ChannelGeometry(source="none", confidence=0.0)
    issues, _ = issues_for(sideways, 6, scale, tracking_config, metadata, geometry=geometry)
    assert not codes(issues) & {"weak_axis", "lateral_drift"}


def test_collect_issues_takes_no_axis_argument(scale, tracking_config, metadata):
    """The interface (E1/E2): metadata, scale, diagnostics, events, tracks, summaries, cfg."""
    issues = collect_issues(metadata, scale, [], [], [], [], tracking_config)
    assert codes(issues) == {"no_tracks"}
    assert QCIssue is Issue
    with pytest.raises(TypeError, match="keyword-only"):
        collect_issues(metadata, scale, [], [], [], [], tracking_config, [])


def test_the_v1_argument_order_still_runs(scale, tracking_config, metadata):
    """Transition, like ``track_detections``: a v1 pipeline passes an axis second.

    Without it the v1 pipeline on this branch raised TypeError on every
    analysis (review of E2). The axis only supplies lanes, so a two-channel
    v1 axis gives the lane note a two-lane applied geometry gives.
    """
    channel = lambda i, x: SimpleNamespace(index=i, origin=(x, 0.0), half_width_px=20.0,
                                           detected=True)
    axis = SimpleNamespace(ux=0.0, uy=1.0, channels=[channel(0, 20.0), channel(1, 60.0)],
                           source="ridges", confidence=0.9, pitch_px=40.0, notes=[])
    track = hand_track(3, straight_track(5, skip={2}))
    tracks = [track]
    summaries = summarise(tracks, scale)
    unlinked = [UnlinkedStart(track_id=9, frame=4, candidate_track_id=3, candidate_last_frame=1,
                              gap_frames=3, distance_px=40.0, cost_chi2=12.0,
                              refused_because="gap_too_long")]

    v1 = collect_issues(metadata, axis, scale, [], [], tracks, summaries, tracking_config, unlinked)
    v2 = collect_issues(metadata, scale, [], [], tracks, summaries, tracking_config,
                        geometry=_two_lanes(applied=True), unlinked=unlinked)
    assert [(i.code, i.frame, i.track_id) for i in v1] == [(i.code, i.frame, i.track_id) for i in v2]
    assert {"multichannel_field", "likely_missed_detection", "unlinked_start"} <= codes(v1)
    with pytest.raises(TypeError, match="v1 order"):
        collect_issues(metadata, axis, scale, [], [], tracks, summaries)


# --------------------------------------------------------------------------
# Lanes are notes
# --------------------------------------------------------------------------


def _two_lanes(*, applied: bool, inferred: bool = False) -> ChannelGeometry:
    return ChannelGeometry(
        lanes=[
            Lane(index=0, origin=(20.0, 0.0), direction=(0.0, 1.0), half_width_px=20.0),
            Lane(index=1, origin=(60.0, 0.0), direction=(0.0, 1.0), half_width_px=20.0,
                 detected=not inferred),
        ],
        source=GEOMETRY_FROM_RIDGES, confidence=0.9, applied=applied,
    )


def test_an_applied_multilane_field_is_an_info_note(scale, tracking_config, metadata):
    issues = collect_issues(metadata, scale, [], [], [], [], tracking_config,
                            geometry=_two_lanes(applied=True))
    note = only(issues, "multichannel_field")[0]
    assert note.severity == SEVERITY_INFO
    assert "2 lanes" in note.title


def test_lanes_that_constrain_nothing_raise_no_lane_note(scale, tracking_config, metadata):
    issues = collect_issues(metadata, scale, [], [], [], [], tracking_config,
                            geometry=_two_lanes(applied=False))
    assert "multichannel_field" not in codes(issues)


def test_an_inferred_lane_is_an_info_note(scale, tracking_config, metadata):
    issues = collect_issues(metadata, scale, [], [], [], [], tracking_config,
                            geometry=_two_lanes(applied=True, inferred=True))
    note = only(issues, "inferred_channel")[0]
    assert note.severity == SEVERITY_INFO
    assert "constrain tracking like the others" in note.detail


# --------------------------------------------------------------------------
# What the result rests on
# --------------------------------------------------------------------------


def test_a_developer_model_override_is_critical(scale, tracking_config, metadata):
    model = SimpleNamespace(developer_override=True,
                            spec=SimpleNamespace(model_id="developer_override"),
                            path="C:/models/experiment")
    issues = collect_issues(metadata, scale, [], [], [], [], tracking_config, model=model)
    issue = only(issues, "developer_model_override")[0]
    assert issue.severity == SEVERITY_CRITICAL
    assert issues[0].code == "developer_model_override", "critical findings sort first"


def test_the_run_json_model_block_is_read_too(scale, tracking_config, metadata):
    issues = collect_issues(metadata, scale, [], [], [], [], tracking_config,
                            model={"model_id": "research:x", "developer_override": True})
    assert "developer_model_override" in codes(issues)


def test_the_validated_model_raises_no_override(scale, tracking_config, metadata):
    model = SimpleNamespace(developer_override=False, spec=SimpleNamespace(model_id="m"), path="p")
    issues = collect_issues(metadata, scale, [], [], [], [], tracking_config, model=model)
    assert "developer_model_override" not in codes(issues)


def test_a_3d_stack_without_a_z_step_is_critical(tracking_config, metadata):
    no_z = Scale.from_values(PIXEL_SIZE_UM, FRAME_INTERVAL_MIN)
    issues = collect_issues(metadata, no_z, [], [], [], [], tracking_config, dimensionality="3D")
    assert only(issues, "no_z_step")[0].severity == SEVERITY_CRITICAL

    with_z = Scale.from_values(PIXEL_SIZE_UM, FRAME_INTERVAL_MIN, z_step_um=2.0)
    issues = collect_issues(metadata, with_z, [], [], [], [], tracking_config, dimensionality="3D")
    assert "no_z_step" not in codes(issues)

    issues = collect_issues(metadata, no_z, [], [], [], [], tracking_config, dimensionality="2D")
    assert "no_z_step" not in codes(issues), "a 2-D movie has no Z step to miss"


def test_non_square_pixels_are_critical(scale, tracking_config, metadata):
    metadata.anisotropic_pixels = True
    metadata.pixel_size_y_um = Calibrated(0.5, SOURCE_USER)
    issue = only(collect_issues(metadata, scale, [], [], [], [], tracking_config),
                 "anisotropic_pixels")[0]
    assert issue.severity == SEVERITY_CRITICAL
    assert "0.5 µm in Y" in issue.detail


def test_imported_labels_are_said_to_be_imported(scale, tracking_config, metadata):
    issues = collect_issues(metadata, scale, [], [], [], [], tracking_config,
                            provenance="imported_labels")
    assert only(issues, "imported_segmentation")[0].severity == SEVERITY_INFO

    issues = collect_issues(metadata, scale, [], [], [], [], tracking_config, provenance="model")
    assert "imported_segmentation" not in codes(issues)


# --------------------------------------------------------------------------
# Per link
# --------------------------------------------------------------------------


def test_an_ambiguous_link_is_flagged_at_its_frame(scale, tracking_config, metadata):
    margins = [None, 10.0, 10.0, LINK_AMBIGUOUS_MARGIN_CHI2 / 4, 10.0]
    track = hand_track(4, straight_track(5), margins=margins)
    issues = issues_of_tracks([track], scale, tracking_config, metadata)

    flagged = only(issues, "link_ambiguous")
    assert [(i.frame, i.track_id) for i in flagged] == [(3, 4)]
    assert "not a probability" in flagged[0].detail


def test_a_clear_link_is_not_ambiguous(scale, tracking_config, metadata):
    track = hand_track(1, straight_track(5), margins=[None] + [LINK_AMBIGUOUS_MARGIN_CHI2 + 0.5] * 4)
    assert "link_ambiguous" not in codes(issues_of_tracks([track], scale, tracking_config, metadata))


def test_the_tracker_records_margins_qc_can_read(scale, tracking_config, metadata):
    """End to end: two cells pressed side by side, 8 px apart, moving together.

    Before either track has a velocity, swapping the two costs little more
    than keeping them (margin 1.56 chi-square into frame 1); once each has
    one, the swap is clearly worse (8.0 and up). The tight links, and only
    those, are ambiguous.
    """
    dets = []
    for f in range(5):
        dets.append(make_detection(f, 40.0, 20.0 + 20.0 * f, label=1))
        dets.append(make_detection(f, 48.0, 22.0 + 20.0 * f, label=2))
    tracks, events = track_detections(dets, 5, scale, tracking_config)
    assert len(tracks) == 2
    tight = {(o.frame, t.id) for t in tracks for o in t.observations[1:]
             if o.link_margin < LINK_AMBIGUOUS_MARGIN_CHI2}
    clear = {(o.frame, t.id) for t in tracks for o in t.observations[1:]
             if o.link_margin >= LINK_AMBIGUOUS_MARGIN_CHI2}
    assert tight == {(1, tracks[0].id), (1, tracks[1].id)}, "the scene must have tight links"
    assert clear, "and clear ones, or this tests nothing"

    issues = collect_issues(metadata, scale, [], events, tracks, summarise(tracks, scale),
                            tracking_config)
    assert {(i.frame, i.track_id) for i in only(issues, "link_ambiguous")} == tight


def test_an_abrupt_change_of_shape_is_flagged(scale, tracking_config, metadata):
    dets = straight_track(5)
    # Aspect 90/11 = 8.2 to 90/40 = 2.25: (ln 3.6 / 0.35)^2 = 13.6 > 9.2.
    dets[3] = make_detection(3, 45.0, 80.0, minor=40.0)
    track = hand_track(2, dets)
    flagged = only(issues_of_tracks([track], scale, tracking_config, metadata),
                   "morphology_discontinuity")
    assert {(i.frame, i.track_id) for i in flagged} == {(3, 2), (4, 2)}
    assert MORPHOLOGY_JUMP_CHI2 == pytest.approx(9.21, abs=0.01)
    # The sigmas behind the percentile are contract values, and it says so.
    assert "assumed shape noise" in flagged[0].detail and "not measured" in flagged[0].detail


def test_ordinary_shape_noise_is_not_a_discontinuity(scale, tracking_config, metadata):
    dets = straight_track(5)
    dets[3] = make_detection(3, 45.0, 80.0, minor=12.0, solidity=0.92)
    track = hand_track(1, dets)
    assert "morphology_discontinuity" not in codes(
        issues_of_tracks([track], scale, tracking_config, metadata)
    )


def test_a_size_jump_is_flagged(scale, tracking_config, metadata):
    dets = straight_track(5, area=800.0)
    dets[2] = make_detection(2, 45.0, 60.0, area=2400.0)  # 3.0x, beyond exp(0.9) = 2.46x
    track = hand_track(1, dets)
    flagged = only(issues_of_tracks([track], scale, tracking_config, metadata), "size_jump")
    assert {i.frame for i in flagged} == {2, 3}
    assert "area" in flagged[0].title
    assert "from 800 to 2400 pixels" in flagged[0].detail
    bound = math.exp(SIZE_JUMP_SIGMAS * tracking_config.sigma_ln_area)
    assert bound < tracking_config.area_ratio_max, "must be able to fire on an accepted link"


def test_a_moderate_size_change_is_not_a_jump(scale, tracking_config, metadata):
    dets = straight_track(5, area=800.0)
    dets[2] = make_detection(2, 45.0, 60.0, area=1400.0)  # 1.75x
    track = hand_track(1, dets)
    assert "size_jump" not in codes(issues_of_tracks([track], scale, tracking_config, metadata))


def test_one_missing_frame_is_a_likely_missed_detection(scale, tracking_config, metadata):
    """Directive §45: present at t-1 and t+1, absent at t."""
    track = hand_track(3, straight_track(5, skip={2}))
    issues = issues_of_tracks([track], scale, tracking_config, metadata)

    missed = only(issues, "likely_missed_detection")
    assert [(i.frame, i.track_id) for i in missed] == [(2, 3)]
    assert missed[0].title == "Frame 2 is a likely missed detection"
    assert "long_reacquisition" not in codes(issues), "one frame is not a long gap"
    bridged = only(issues, "gap_bridged")[0]
    assert bridged.frame == 3
    assert f"gate of {tracking_config.effective_gate_chi2:.0f}" in bridged.detail


def test_a_long_reacquisition_is_flagged(scale, tracking_config, metadata):
    track = hand_track(1, straight_track(7, skip={2, 3, 4}))
    issues = issues_of_tracks([track], scale, tracking_config, metadata)

    long = only(issues, "long_reacquisition")
    assert [(i.frame, i.track_id, i.severity) for i in long] == [(5, 1, SEVERITY_WARN)]
    assert "3 missing frames" in long[0].title
    assert "likely_missed_detection" not in codes(issues), "three frames is not one missed detection"


def test_a_track_seen_less_than_it_is_missed_is_gap_dominated(scale, tracking_config, metadata):
    track = hand_track(1, straight_track(7, skip={1, 2, 4, 5}))  # seen 0, 3, 6; missing 4
    issue = only(issues_of_tracks([track], scale, tracking_config, metadata),
                 "gap_dominated_track")[0]
    assert (issue.frame, issue.track_id) == (0, 1)
    assert "4 missing frame(s) against 3 observation(s)" in issue.detail


def test_a_mostly_observed_track_is_not_gap_dominated(scale, tracking_config, metadata):
    track = hand_track(1, straight_track(6, skip={3}))
    assert "gap_dominated_track" not in codes(
        issues_of_tracks([track], scale, tracking_config, metadata)
    )


def test_gap_closing_says_it_closed_the_gap(scale, tracking_config, metadata):
    """The detail names the stage that made the link, from the tracker's breakdown.

    A stage-2 link is one whose breakdown carries both directions
    (``motion_forward``/``motion_backward``). Built by hand: the end-to-end
    stage-2 scenes of tests/test_gap_closing.py do not close at the base this
    package builds on (the reversal's forward d^2 is 54.3, so the averaged
    motion 27.2 exceeds the 13.8 motion gate) -- a tracker matter, not QC's.
    """
    track = hand_track(4, straight_track(8, skip={3, 4}))
    link = next(o for o in track.observations if o.frame == 5)
    link.cost = 6.0
    link.breakdown = CostBreakdown(total=6.0, motion=5.0, mahalanobis=5.0,
                                   motion_forward=9.9, motion_backward=0.1)
    issues = issues_of_tracks([track], scale, tracking_config, metadata)
    bridged = only(issues, "gap_bridged")
    assert [(i.frame, i.track_id) for i in bridged] == [(5, 4)]
    assert "linked by global gap closing with cost 6.0" in bridged[0].detail
    long = only(issues, "long_reacquisition")[0]
    assert "linked by global gap closing" in long.detail


def test_a_frame_to_frame_reacquisition_says_so(scale, tracking_config, metadata):
    dets = straight_track(8, skip={3, 4})
    tracks, events = track_detections(dets, 8, scale, tracking_config)
    assert len(tracks) == 1
    link = next(o for o in tracks[0].observations if o.frame == 5)
    assert link.breakdown is None or link.breakdown.motion_forward is None, "stage 1 made it"
    issues = collect_issues(metadata, scale, [], events, tracks, summarise(tracks, scale),
                            tracking_config)
    bridged = only(issues, "gap_bridged")[0]
    assert bridged.frame == 5
    assert "frame to frame" in bridged.detail


# --------------------------------------------------------------------------
# Events, borders and counts
# --------------------------------------------------------------------------


def test_a_suspected_split_is_flagged_for_both_tracks(scale, tracking_config, metadata):
    event = FrameEvent(5, 3, 2, 2, 1, 0, 0, split_suspected=[2, 7])
    issues = collect_issues(metadata, scale, [], [event], [], [], tracking_config)
    flagged = only(issues, "split_suspected")
    assert {(i.frame, i.track_id) for i in flagged} == {(5, 2), (5, 7)}
    assert all(i.severity == SEVERITY_WARN for i in flagged)


def test_border_entry_and_exit_come_from_the_tracker_flags(scale, tracking_config, metadata):
    """Mid-movie starts and ends on a mask cut by the image edge, end to end."""
    dets = [make_detection(f, 45.0, 20.0 + 20.0 * f, touches_border=f in (2, 5))
            for f in range(2, 6)]
    tracks, events = track_detections(dets, 8, scale, tracking_config)
    issues = collect_issues(metadata, scale, [], events, tracks, summarise(tracks, scale),
                            tracking_config)
    entry = only(issues, "border_entry")[0]
    exit_ = only(issues, "border_exit")[0]
    assert (entry.frame, entry.track_id, entry.severity) == (2, 1, SEVERITY_INFO)
    assert (exit_.frame, exit_.track_id, exit_.severity) == (5, 1, SEVERITY_INFO)


def test_a_track_cut_by_the_edge_at_frame_zero_is_not_an_entry(scale, tracking_config, metadata):
    dets = [make_detection(f, 45.0, 20.0 + 20.0 * f, touches_border=f == 0) for f in range(4)]
    issues, _ = issues_for(dets, 4, scale, tracking_config, metadata)
    assert "border_entry" not in codes(issues)


def test_an_abrupt_change_of_object_count_is_flagged(scale, tracking_config, metadata):
    kept = [2, 2, 6, 6, 1, 1]
    diagnostics = [FrameDiagnostics(frame=f, raw_count=k, kept_count=k) for f, k in enumerate(kept)]
    issues = collect_issues(metadata, scale, diagnostics, [], [], [], tracking_config)
    jumps = only(issues, "count_jump")
    assert [i.frame for i in jumps] == [2, 4]
    assert jumps[0].title == "Object count jumps from 2 to 6 at frame 2"
    assert all(i.track_id is None for i in jumps), "a frame-level finding"


def test_count_noise_is_not_a_jump(scale, tracking_config, metadata):
    """Off by one, or by under half: ordinary segmentation wobble."""
    kept = [1, 0, 1, 10, 7, 9]
    diagnostics = [FrameDiagnostics(frame=f, raw_count=k, kept_count=k) for f, k in enumerate(kept)]
    issues = collect_issues(metadata, scale, diagnostics, [], [], [], tracking_config)
    # 1 -> 10 is a jump; 10 -> 7 (3 < 5) and 7 -> 9 (2 < 3.5) are not.
    assert [i.frame for i in issues if i.code == "count_jump"] == [3]


def test_count_series_overrides_and_events_stand_in(scale, tracking_config, metadata):
    issues = collect_issues(metadata, scale, [], [], [], [], tracking_config,
                            count_series=[4, 4, 0, 4])
    assert [i.frame for i in issues if i.code == "count_jump"] == [2, 3]

    events = [FrameEvent(f, n, 0, 0, 0, 0, 0) for f, n in enumerate([3, 3, 9])]
    issues = collect_issues(metadata, scale, [], events, [], [], tracking_config)
    assert [i.frame for i in issues if i.code == "count_jump"] == [2]


# --------------------------------------------------------------------------
# Kept codes, v2 wording
# --------------------------------------------------------------------------


def test_fast_track_reports_both_units(tracking_config, metadata):
    scale = Scale.from_values(1.0, 1.0)
    # 4.5 px/frame at 1 um/px and 1 min/frame is 4.5 um/min, above 0.8 x 5.0.
    dets = [make_detection(f, 45.0, 20.0 + 4.5 * f) for f in range(5)]
    issues, _ = issues_for(dets, 5, scale, tracking_config, metadata)
    fast = only(issues, "fast_track")[0]
    assert "4.50 µm/min (270 µm/hr)" in fast.detail
    assert "5.00 µm/min (300 µm/hr)" in fast.detail


def test_unlinked_start_compares_with_the_derived_gate(scale, metadata):
    """The legacy ``gate_chi2`` field no longer decides anything (critique C6)."""
    cfg = TrackingConfig(unmatched_chi2=15.0)
    cfg.gate_chi2 = 10.0  # a stale 1.x value; the v2 gate is 2U = 30
    start = UnlinkedStart(
        track_id=2, frame=9, candidate_track_id=1, candidate_last_frame=3, gap_frames=6,
        distance_px=40.0, cost_chi2=25.0, refused_because="gap_too_long",
    )
    issue = only(collect_issues(metadata, scale, [], [], [], [], cfg, unlinked=[start]),
                 "unlinked_start")[0]
    assert issue.severity == SEVERITY_WARN
    assert "against a gate of 30" in issue.detail
    assert (issue.frame, issue.track_id) == (9, 2)


def test_every_track_finding_is_clickable(scale, tracking_config, metadata):
    """Each finding about a track names the frame and the track to go to."""
    tracks = [
        hand_track(1, straight_track(7, skip={2, 3, 4}), margins=[None, 0.1, 10.0, 10.0]),
        hand_track(2, [make_detection(f, 80.0, 20.0, area=800.0 if f < 2 else 2400.0)
                       for f in range(4)]),
    ]
    issues = issues_of_tracks(tracks, scale, tracking_config, metadata,
                              events=[FrameEvent(1, 2, 2, 2, 0, 0, 0, split_suspected=[2])])
    per_track = {"link_ambiguous", "size_jump", "long_reacquisition", "gap_bridged",
                 "split_suspected", "stationary_track", "gap_dominated_track"}
    seen = [i for i in issues if i.code in per_track]
    assert {i.code for i in seen} >= {"link_ambiguous", "size_jump", "long_reacquisition"}
    for issue in seen:
        assert issue.frame is not None and issue.track_id is not None, issue
