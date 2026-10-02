"""Where recovery looks, and whether it admits when it did not look.

This file exists because of a defect that was invisible in every output the
software produced. ``recover()`` asked ``Track.predict`` where a dormant cell
should be. That method extrapolates forward and refuses to look backwards,
which is correct while tracking is running. But recovery runs *after* tracking,
when a track's ``last_frame`` is its final observation -- so every interior gap
lay behind it, every one raised ValueError, and a bare ``except: continue``
dropped them all without recording an attempt.

The consequence was worse than the missing feature: the measurement that
justified shipping recovery disabled was made against this bug, so it described
the skip rather than the algorithm. These tests pin the two properties that
would have caught it -- interior gaps are examined, and nothing is ever dropped
in silence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from corridor.core.config import Scale, TrackingConfig
from corridor.core.confinement import Channel, ConfinementAxis
from corridor.core.recovery import (
    RecoveryConfig,
    _dormant_frames,
    expected_position,
    recover,
)
from corridor.core.tracking import track_detections

from conftest import make_detection


@pytest.fixture
def straight_track_with_a_hole(vertical_axis, scale, tracking_config):
    """A cell moving steadily down the channel, unobserved at frame 3."""
    dets = [make_detection(f, 45.0, 20.0 + 20.0 * f) for f in range(7) if f != 3]
    tracks, _ = track_detections(dets, 7, vertical_axis, scale, tracking_config)
    assert len(tracks) == 1, "the gap must be bridged, or this tests nothing"
    return tracks[0], tracks


# --------------------------------------------------------------------------
# Where the cell is expected to be
# --------------------------------------------------------------------------


def test_an_interior_gap_is_interpolated_not_extrapolated(straight_track_with_a_hole):
    """The regression. Before the fix this raised and the frame was skipped."""
    track, _ = straight_track_with_a_hole
    assert track.last_frame == 6  # the gap at 3 is well behind the track's end

    position = expected_position(track, 3)

    # The cell really was at y = 80 on frame 3.
    assert position[0] == pytest.approx(45.0)
    assert position[1] == pytest.approx(80.0)


def test_interpolation_beats_extrapolation_for_a_cell_that_changed_speed(
    vertical_axis, scale, tracking_config
):
    """Interpolation needs no velocity model, which is why it is used here.

    A cell that accelerates has a velocity at the end of its track that says
    nothing useful about where it was in the middle.
    """
    # The cell creeps at 5 px per frame, then speeds up to 20. The acceleration
    # has to stay inside what the tracker will link, or this would be testing
    # the assignment gate rather than the estimate.
    positions = {0: 20.0, 1: 25.0, 2: 30.0, 4: 50.0, 5: 70.0}
    dets = [make_detection(f, 45.0, y) for f, y in positions.items()]
    tracks, _ = track_detections(dets, 6, vertical_axis, scale, tracking_config)
    assert len(tracks) == 1, "the gap must be bridged, or this tests nothing"
    track = tracks[0]

    estimate = expected_position(track, 3)
    # Halfway between frame 2 (y=30) and frame 4 (y=50), using both sides.
    assert estimate[1] == pytest.approx(40.0)
    # Extrapolating the earlier 5 px/frame velocity would have said 35, and
    # extrapolating the track's final velocity cannot look backwards at all.
    assert estimate[1] != pytest.approx(35.0)


def test_a_frame_past_the_end_still_extrapolates(straight_track_with_a_hole):
    """Trailing frames have nothing on the far side, so the velocity model stands."""
    track, _ = straight_track_with_a_hole
    position = expected_position(track, 7)
    assert position[1] > track.observations[-1].y


def test_a_frame_before_the_track_began_is_refused(straight_track_with_a_hole):
    """Recovery recovers a cell's past, it does not invent one before it existed."""
    dets = [make_detection(f, 45.0, 20.0 + 20.0 * f) for f in range(3, 7)]
    tracks, _ = track_detections(
        dets, 7, ConfinementAxis(
            ux=0.0, uy=1.0, source="configured", confidence=1.0,
            angle_sigma_rad=math.radians(0.5),
            channels=[Channel(index=0, origin=(45.0, 0.0), half_width_px=45.0)],
        ),
        Scale.from_values(0.467060342995564, 20.006894938151042),
        TrackingConfig(),
    )
    with pytest.raises(ValueError, match="before it existed"):
        expected_position(tracks[0], 0)


def test_a_frame_that_was_observed_is_refused(straight_track_with_a_hole):
    track, _ = straight_track_with_a_hole
    with pytest.raises(ValueError, match="already has an observation"):
        expected_position(track, 2)


def test_a_track_with_no_observations_is_refused(straight_track_with_a_hole):
    track, _ = straight_track_with_a_hole
    empty = type(track)(id=99, channel=0)
    with pytest.raises(ValueError, match="no observations"):
        expected_position(empty, 0)


# --------------------------------------------------------------------------
# Nothing is dropped in silence
# --------------------------------------------------------------------------


class _RefusingService:
    """A segmentation service that finds nothing, so only the bookkeeping shows."""

    def segment_crop(self, crop, override=None):
        return np.zeros(crop.shape[:2], dtype=np.int32)


def test_every_enumerated_gap_produces_an_attempt(
    straight_track_with_a_hole, vertical_axis, scale, tracking_config
):
    """The property whose absence hid the defect.

    Recovery may fail to find a cell -- that is an ordinary outcome. What it may
    not do is enumerate a frame, decline to look, and leave no trace, because
    then recovery_attempts.csv and the manifest's attempt count both describe a
    pass that never happened.
    """
    track, tracks = straight_track_with_a_hole
    stack = np.zeros((7, 324, 90), dtype=np.uint16)

    cfg = RecoveryConfig(enabled=True, intensity=False, trailing=True)
    enumerated = _dormant_frames(tracks, 7, tracking_config, cfg)
    result = recover(
        stack, tracks, _RefusingService(), vertical_axis, scale, tracking_config, cfg,
    )

    assert len(result.attempts) == len(enumerated)
    assert {a.frame for a in result.attempts} == {f for _, f in enumerated}
    assert 3 in {a.frame for a in result.attempts}, "the interior gap must be examined"
    assert not result.detections, "an empty image contains no cells to recover"


def test_recovery_disabled_examines_nothing(
    straight_track_with_a_hole, vertical_axis, scale, tracking_config
):
    track, tracks = straight_track_with_a_hole
    stack = np.zeros((7, 324, 90), dtype=np.uint16)

    result = recover(
        stack, tracks, _RefusingService(), vertical_axis, scale, tracking_config,
        RecoveryConfig(enabled=False),
    )
    assert result.attempts == []
    assert result.detections == []


# --------------------------------------------------------------------------
# Interior gaps and trailing frames are different questions
# --------------------------------------------------------------------------


def test_trailing_frames_are_not_probed_by_default(
    straight_track_with_a_hole, tracking_config
):
    """The measured reason recovery is safe to run at all.

    An interior gap is a cell known to have been present on both sides. A frame
    past the end of a track is a cell that may simply have left, and on a
    channel known to be empty every false position came from exactly there.
    """
    _, tracks = straight_track_with_a_hole
    gaps = _dormant_frames(tracks, 12, tracking_config, RecoveryConfig(enabled=True))

    assert {f for _, f in gaps} == {3}, "only the interior hole should be probed"


def test_trailing_frames_are_probed_when_asked_for(
    straight_track_with_a_hole, tracking_config
):
    _, tracks = straight_track_with_a_hole
    gaps = _dormant_frames(
        tracks, 12, tracking_config, RecoveryConfig(enabled=True, trailing=True)
    )
    frames = {f for _, f in gaps}

    assert 3 in frames
    assert frames & {7, 8, 9, 10}, "frames past the track's end should appear"


def test_interior_can_be_switched_off_independently(
    straight_track_with_a_hole, tracking_config
):
    _, tracks = straight_track_with_a_hole
    gaps = _dormant_frames(
        tracks, 12, tracking_config,
        RecoveryConfig(enabled=True, interior=False, trailing=True),
    )
    assert 3 not in {f for _, f in gaps}


def test_the_shipped_default_probes_interior_gaps_only():
    """Changing either of these changes what every published figure describes."""
    cfg = RecoveryConfig()
    assert cfg.interior is True
    assert cfg.trailing is False
