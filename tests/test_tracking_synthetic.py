"""Synthetic scenes with a known answer, scored on identities, not counts.

There is no tracking ground truth in the supplied data (contract §0), so the
tracker's claims are tested here, where the truth is planted: every
detection's ``label`` is its true cell, and every test asserts that one track
holds exactly one cell's detections. A count of tracks can be right with the
identities swapped; these cannot.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from corridor.core.tracking import (
    FLAG_ENTERS_BORDER,
    FLAG_EXITS_BORDER,
    FLAG_MERGE,
    FLAG_SPLIT,
    track_detections,
)

from conftest import identity_of, make_detection

HORIZONTAL = math.pi / 2  # scikit-image orientation of a body lying along x
#: Round cells for the open field: orientation means nothing below ecc 0.6.
ROUND = dict(eccentricity=0.3, orientation_rad=0.0, major=32.0, minor=30.0, area=750.0)


def assert_identities(tracks, truth: dict[int, list[int]]) -> None:
    """``truth`` maps a cell label to the frames it was detected in.

    Every one of those detections must sit in one and the same track, and no
    two cells may share a track.
    """
    owners = {}
    for label, frames in truth.items():
        ids = {identity_of(tracks, f, label) for f in frames}
        assert len(ids) == 1 and None not in ids, f"cell {label} is split over tracks {ids}"
        owners[label] = ids.pop()
    assert len(set(owners.values())) == len(owners), f"two cells share a track: {owners}"


# --------------------------------------------------------------------------
# Crossing
# --------------------------------------------------------------------------


def test_head_on_crossing_in_an_open_field_keeps_identity(scale, tracking_config):
    """A moves left to right, B right to left, one body-width apart."""
    dets = []
    for t in range(11):
        dets.append(make_detection(t, 20 + 18 * t, 100, label=1, orientation_rad=HORIZONTAL))
        dets.append(make_detection(t, 200 - 18 * t, 112, label=2, orientation_rad=HORIZONTAL))
    tracks, _ = track_detections(dets, 11, scale, tracking_config)
    assert len(tracks) == 2
    assert_identities(tracks, {1: list(range(11)), 2: list(range(11))})


def test_x_crossing_of_round_cells_keeps_identity(scale, tracking_config):
    """Two round cells whose paths cross at 8 px separation in the crossing frame.

    Round cells give the tracker no body orientation to lean on, so this is
    carried by the velocity state alone.
    """
    dets = []
    for t in range(11):
        dets.append(make_detection(t, 20 + 16 * t, 40 + 12 * t, label=1, **ROUND))
        dets.append(make_detection(t, 28 + 16 * t, 160 - 12 * t, label=2, **ROUND))
    tracks, _ = track_detections(dets, 11, scale, tracking_config)
    assert len(tracks) == 2
    assert_identities(tracks, {1: list(range(11)), 2: list(range(11))})


def test_crossing_with_a_missing_frame_at_the_crossing(scale, tracking_config):
    """Both cells vanish in the frame where they meet; both must come back as themselves."""
    dets = []
    for t in range(11):
        if t == 5:
            continue
        dets.append(make_detection(t, 20 + 18 * t, 100, label=1, orientation_rad=HORIZONTAL))
        dets.append(make_detection(t, 200 - 18 * t, 112, label=2, orientation_rad=HORIZONTAL))
    tracks, _ = track_detections(dets, 11, scale, tracking_config)
    frames = [t for t in range(11) if t != 5]
    assert_identities(tracks, {1: frames, 2: frames})


# --------------------------------------------------------------------------
# Missing frames
# --------------------------------------------------------------------------


def test_single_missing_frame_with_a_neighbour(scale, tracking_config):
    """Two cells in neighbouring lanes; each misses a different frame."""
    dets = []
    for t in range(10):
        if t != 3:
            dets.append(make_detection(t, 45, 20 + 20 * t, label=1))
        if t != 6:
            dets.append(make_detection(t, 127, 200 - 15 * t, label=2))
    tracks, _ = track_detections(dets, 10, scale, tracking_config)
    assert len(tracks) == 2
    assert_identities(tracks, {1: [t for t in range(10) if t != 3], 2: [t for t in range(10) if t != 6]})
    one = next(t for t in tracks if t.id == identity_of(tracks, 4, 1))
    assert next(o for o in one.observations if o.frame == 4).gap_frames == 2


def test_three_missing_frames_with_a_neighbour(scale, tracking_config):
    tracking_config.max_gap = 3
    dets = []
    for t in range(12):
        if t not in (4, 5, 6):
            dets.append(make_detection(t, 45, 20 + 20 * t, label=1))
        dets.append(make_detection(t, 127, 40 + 18 * t, label=2))
    tracks, _ = track_detections(dets, 12, scale, tracking_config)
    assert len(tracks) == 2
    assert_identities(tracks, {1: [t for t in range(12) if t not in (4, 5, 6)], 2: list(range(12))})
    one = next(t for t in tracks if t.id == identity_of(tracks, 7, 1))
    assert next(o for o in one.observations if o.frame == 7).gap_frames == 4


# --------------------------------------------------------------------------
# Size
# --------------------------------------------------------------------------


def test_sudden_area_change_within_the_gate_keeps_identity(scale, tracking_config):
    """A cell that doubles its mask (spreading, or a segmentation change) is still itself."""
    dets = []
    for t in range(8):
        area = 800.0 if t < 4 else 1700.0
        dets.append(make_detection(t, 45, 20 + 20 * t, label=1, area=area))
        dets.append(make_detection(t, 127, 30 + 20 * t, label=2, area=800.0))
    tracks, _ = track_detections(dets, 8, scale, tracking_config)
    assert_identities(tracks, {1: list(range(8)), 2: list(range(8))})
    jump = next(o for t in tracks for o in t.observations if o.frame == 4 and o.det_label == 1)
    assert jump.breakdown.size == pytest.approx((math.log(1700 / 800) / 0.30) ** 2)


def test_area_change_beyond_the_gate_is_a_new_track(scale, tracking_config):
    dets = [make_detection(t, 45, 20 + 20 * t, label=1, area=800.0) for t in range(4)]
    dets += [make_detection(t, 45, 20 + 20 * t, label=2, area=800.0 * 3.5) for t in range(4, 8)]
    tracks, _ = track_detections(dets, 8, scale, tracking_config)
    assert len(tracks) == 2
    assert_identities(tracks, {1: [0, 1, 2, 3], 2: [4, 5, 6, 7]})


# --------------------------------------------------------------------------
# Merge and split
# --------------------------------------------------------------------------


def test_merge_then_separation_restores_both_identities(scale, tracking_config):
    """Two round cells moving side by side become one mask for two frames, then two again.

    Exactly one track claims each merged mask (no centroid is invented for the
    other), the vanished one is flagged, and after the separation each cell is
    picked up by its own track again -- the loser across a two-frame gap.
    """
    dets = []
    for t in range(9):
        if t in (3, 4):
            dets.append(make_detection(t, 40 + 15 * t, 110, label=9, **{**ROUND, "area": 1500.0, "major": 60.0}))
        else:
            dets.append(make_detection(t, 40 + 15 * t, 95, label=1, **ROUND))
            dets.append(make_detection(t, 40 + 15 * t, 125, label=2, **ROUND))
    tracks, events = track_detections(dets, 9, scale, tracking_config)
    apart = [0, 1, 2, 5, 6, 7, 8]
    assert_identities(tracks, {1: apart, 2: apart})
    assert len(tracks) == 2
    for frame in (3, 4):
        assert len([o for t in tracks for o in t.observations if o.frame == frame]) == 1
    loser = [t for t in tracks if FLAG_MERGE in t.flags]
    assert len(loser) == 1
    assert events[3].merge_suspected == [loser[0].id]
    assert [o.frame for o in loser[0].observations] == apart
    assert next(o for o in loser[0].observations if o.frame == 5).gap_frames == 3


def test_split_is_flagged_on_both_tracks(scale, tracking_config):
    """One long mask becomes two halves: one continues the track, one is new, both flagged."""
    dets = [make_detection(t, 45, 100 + 5 * t, label=1, area=1600, major=180) for t in range(4)]
    for t in range(4, 8):
        dets.append(make_detection(t, 45, 70 + 5 * t, label=1, area=800))
        dets.append(make_detection(t, 45, 150 + 5 * t, label=2, area=800))
    tracks, events = track_detections(dets, 8, scale, tracking_config)
    assert len(tracks) == 2
    parent = next(t for t in tracks if t.observations[0].frame == 0)
    child = next(t for t in tracks if t.observations[0].frame == 4)
    # Each half is one identity from the split onwards.
    assert_identities(
        tracks, {1: [4, 5, 6, 7], 2: [4, 5, 6, 7]}
    )
    assert FLAG_SPLIT in parent.flags and FLAG_SPLIT in child.flags
    assert sorted(events[4].split_suspected) == sorted([parent.id, child.id])
    assert "split suspected" in events[4].to_row()["notes"]


# --------------------------------------------------------------------------
# Border
# --------------------------------------------------------------------------


def test_border_entry_and_exit_are_flagged(scale, tracking_config):
    """Only a mid-movie start or end on a border-touching mask is an entry or exit."""
    dets = []
    # Present from frame 0 on the border: already there, not an entry.
    for t in range(8):
        dets.append(make_detection(t, 45, 10 + 10 * t, label=1, touches_border=(t == 0)))
    # Enters at frame 3 through the top edge.
    for t in range(3, 8):
        dets.append(make_detection(t, 127, 5 + 15 * (t - 3), label=2, touches_border=(t == 3)))
    # Leaves at frame 5 through the bottom edge.
    for t in range(0, 6):
        dets.append(make_detection(t, 209, 200 + 20 * t, label=3, touches_border=(t == 5)))
    tracks, _ = track_detections(dets, 8, scale, tracking_config)
    assert_identities(tracks, {1: list(range(8)), 2: list(range(3, 8)), 3: list(range(6))})
    flags = {label: tracks[[t.id for t in tracks].index(identity_of(tracks, 5, label))].flags
             for label in (1, 2, 3)}
    assert flags[1].isdisjoint({FLAG_ENTERS_BORDER, FLAG_EXITS_BORDER})
    assert FLAG_ENTERS_BORDER in flags[2] and FLAG_EXITS_BORDER not in flags[2]
    assert FLAG_EXITS_BORDER in flags[3] and FLAG_ENTERS_BORDER not in flags[3]


# --------------------------------------------------------------------------
# Many cells
# --------------------------------------------------------------------------


def test_a_dense_field_of_lanes_keeps_every_identity(scale, tracking_config):
    """Six lanes 82 px apart, cells at different speeds and directions, some frames lost."""
    rng = np.random.default_rng(7)
    dets = []
    truth: dict[int, list[int]] = {}
    for lane in range(6):
        x = 45 + 82 * lane
        speed = [12, -18, 25, -8, 30, 15][lane]
        y0 = 40 if speed > 0 else 300
        frames = []
        for t in range(14):
            if rng.random() < 0.12:
                continue
            jitter = rng.normal(0, 1.5, size=2)
            dets.append(make_detection(t, x + jitter[0], y0 + speed * t + jitter[1], label=lane + 1))
            frames.append(t)
        truth[lane + 1] = frames
    tracks, _ = track_detections(dets, 14, scale, tracking_config)
    assert_identities(tracks, truth)
    assert len(tracks) == 6
