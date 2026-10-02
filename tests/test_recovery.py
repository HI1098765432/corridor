"""Where recovery looks, what it may add, and whether it admits when it did not look.

This file began with a defect that was invisible in every output the software
produced. ``recover()`` asked ``Track.predict`` where a dormant cell should be.
That method extrapolates forward and refuses to look backwards, which is
correct while tracking is running. But recovery runs *after* tracking, when a
track's ``last_frame`` is its final observation -- so every interior gap lay
behind it, every one raised ValueError, and a bare ``except: continue`` dropped
them all without recording an attempt.

2.0 adds three properties, each pinned here:

* **No axis.** The crop is sized from the cell's own body and the intensity
  tier measures offsets across that body. A horizontal channel -- which 1.x's
  "length along y, width along x" window cut in half -- is recovered whole.
* **No duplicates.** Measured on the supplied wide fields, recovery re-found
  cells the first pass had already detected (052924_2 lane 4 frames 11-14 at
  20.7, 7.9, 1.2 and 0.6 px from the primary) and re-tracking then ran two
  tracks along one cell. A candidate that is a primary cell is dropped.
* **Final ids.** Every attempt carries the bracket it was estimated from, so
  recovery_attempts.csv can be keyed to the re-tracked ids (critique C4).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from corridor.core.config import Scale, TrackingConfig
from corridor.core.geometry import GEOMETRY_FROM_RIDGES, ChannelGeometry, Lane
from corridor.core.recovery import (
    SOURCE_INTENSITY,
    SOURCE_WINDOW,
    RecoveryConfig,
    _dormant_frames,
    assign_final_track_ids,
    duplicate_of,
    expected_body,
    expected_position,
    recover,
    window_for,
)
from corridor.core.tracking import Track, track_detections

from conftest import FRAME_INTERVAL_MIN, PIXEL_SIZE_UM, make_detection

HORIZONTAL = math.pi / 2  # scikit-image orientation of a body along the image columns


@pytest.fixture
def straight_track_with_a_hole(scale, tracking_config):
    """A cell moving steadily down the image, unobserved at frame 3."""
    dets = [make_detection(f, 45.0, 20.0 + 20.0 * f) for f in range(7) if f != 3]
    tracks, _ = track_detections(dets, 7, scale, tracking_config)
    assert len(tracks) == 1, "the gap must be bridged, or this tests nothing"
    return tracks[0], tracks


# --------------------------------------------------------------------------
# Synthetic scenes: bright cells on a dark field, and a "network" that
# thresholds whatever crop it is given -- enough to exercise every tier
# without Cellpose.
# --------------------------------------------------------------------------

BACKGROUND, CELL = 100.0, 1000.0


class _ThresholdService:
    """Segments a crop by thresholding it; the permissive override changes nothing."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, int], object]] = []

    def segment_crop(self, crop, override=None):
        self.calls.append((crop.shape, override))
        labels, _ = ndimage.label(crop > 0.5 * (BACKGROUND + CELL))
        return labels.astype(np.int32)


class _RefusingService:
    """A segmentation service that finds nothing, so only the bookkeeping shows."""

    def segment_crop(self, crop, override=None):
        return np.zeros(crop.shape[:2], dtype=np.int32)


def horizontal_cells(frames, *, x0=100.0, step=20.0, y=60.0, major=140.0, minor=11.0):
    """One horizontal cell per frame, with masks, moving along x."""
    return [
        make_detection(f, x0 + step * f, y, major=major, minor=minor,
                       orientation_rad=HORIZONTAL, with_mask=True)
        for f in frames
    ]


def paint(cells, shape, *, noise_sigma: float = 0.0, seed: int = 0) -> np.ndarray:
    """A (T, Y, X) stack with each cell's own mask painted bright."""
    stack = np.full(shape, BACKGROUND, dtype=np.float32)
    if noise_sigma:
        stack += np.random.default_rng(seed).normal(0.0, noise_sigma, size=shape).astype(np.float32)
    for det in cells:
        stack[det.frame][det.full_mask(shape[1:])] = CELL
    return stack


def hand_track(track_id: int, detections) -> Track:
    track = Track(id=track_id)
    for det in detections:
        track.observe(det, cost=None if not track.observations else 1.0)
    return track


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
    scale, tracking_config
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
    tracks, _ = track_detections(dets, 6, scale, tracking_config)
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


def test_a_frame_before_the_track_began_is_refused():
    """Recovery recovers a cell's past, it does not invent one before it existed."""
    dets = [make_detection(f, 45.0, 20.0 + 20.0 * f) for f in range(3, 7)]
    tracks, _ = track_detections(
        dets, 7, Scale.from_values(PIXEL_SIZE_UM, FRAME_INTERVAL_MIN), TrackingConfig(),
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


def test_the_expected_body_comes_from_the_bracketing_cells(straight_track_with_a_hole):
    track, _ = straight_track_with_a_hole
    body = expected_body(track, 3)
    assert body.bracket == (2, 1, 4, 1)
    assert body.major_px == pytest.approx(90.0)
    assert body.minor_px == pytest.approx(11.0)
    # orientation 0 is a body along the image rows: u = (0, 1).
    assert abs(body.u[1]) == pytest.approx(1.0)

    trailing = expected_body(track, 7)
    assert trailing.bracket == (6, 1, None, None)


def test_orientation_is_averaged_as_an_axis_not_an_angle():
    """+89 and -89 degrees are the same horizontal body, not a vertical one."""
    before = make_detection(0, 50.0, 50.0, orientation_rad=math.radians(89.0))
    after = make_detection(2, 90.0, 50.0, orientation_rad=math.radians(-89.0))
    body = expected_body(hand_track(1, [before, after]), 1)
    assert abs(body.u[0]) == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------
# The window follows the cell, not an axis
# --------------------------------------------------------------------------


def test_the_window_holds_a_horizontal_cell_whole():
    """1.x sized the crop 'length along y, width along x' whatever the cell.

    For a 140 px horizontal cell (the supplied cells are 46-201 px long) that
    gave a crop max(40, 2.2 * 2 * 11) = 48 px either side in x -- 97 px for a
    140 px body, cut at both ends. The window now comes from the cell's own
    orientation and length.
    """
    cells = horizontal_cells([0, 2], x0=300.0)
    body = expected_body(hand_track(1, cells), 1)
    x0, y0, x1, y1 = window_for(body, RecoveryConfig(), (120, 800))

    old_half_x = max(RecoveryConfig().window_min_px, int(11.0 * 2.2 * 2))
    assert 2 * old_half_x + 1 < 140, "the 1.x window could not contain this cell"
    assert x1 - x0 >= 2.2 * 140 - 2
    assert x0 < body.position[0] - 70 and x1 > body.position[0] + 70
    assert y1 - y0 <= 120


def test_a_vertical_cell_keeps_the_1x_window_shape():
    """For the cells 1.x was measured on, the window is the same shape, longer along."""
    cells = [make_detection(f, 500.0, 400.0 + 20.0 * f, with_mask=True) for f in (0, 2)]
    body = expected_body(hand_track(1, cells), 1)
    x0, y0, x1, y1 = window_for(body, RecoveryConfig(), (1000, 1000))
    # 2.2 x 2 widths either side, as before (rounded up now, not down: 49 not 48).
    assert abs((x1 - x0) - (2 * 48 + 1)) <= 2
    assert (y1 - y0) > (x1 - x0)


def test_a_horizontal_hole_is_recovered_whole_with_its_mask(scale, tracking_config):
    """End to end: the window tier finds the whole cell, mask and v2 morphology included."""
    shape = (7, 120, 400)
    truth = horizontal_cells(range(7))
    stack = paint(truth, shape)
    primaries = [d for d in horizontal_cells(range(7)) if d.frame != 3]
    tracks, _ = track_detections(primaries, 7, scale, tracking_config)
    assert len(tracks) == 1

    result = recover(stack, tracks, _ThresholdService(), scale, tracking_config, RecoveryConfig())

    assert [a.frame for a in result.attempts] == [3]
    attempt = result.attempts[0]
    assert attempt.found and attempt.source == SOURCE_WINDOW
    assert "window edge" not in attempt.detail
    [det] = result.detections
    true = truth[3]
    assert det.x == pytest.approx(true.x, abs=1.0) and det.y == pytest.approx(true.y, abs=1.0)
    assert det.major_axis_px == pytest.approx(true.major_axis_px, rel=0.05)
    assert det.area_px == pytest.approx(true.area_px, rel=0.02)
    assert det.mask_crop is not None and det.mask_crop.sum() == det.area_px
    assert det.full_mask(shape[1:]).sum() == true.full_mask(shape[1:]).sum()
    for name in ("perimeter_px", "circularity", "aspect_ratio", "convex_area_px",
                 "equivalent_diameter_px", "mean_intensity"):
        assert getattr(det, name) is not None, name
    assert det.source == SOURCE_WINDOW and det.confidence == pytest.approx(0.80)
    assert attempt.bracket == (2, 1, 4, 1)


def test_the_intensity_tier_works_in_a_horizontal_channel(scale, tracking_config):
    """No network, no axis: the offset is measured across the cell's own body."""
    shape = (7, 120, 400)
    stack = paint(horizontal_cells(range(7)), shape, noise_sigma=5.0)
    primaries = [d for d in horizontal_cells(range(7)) if d.frame != 3]
    tracks, _ = track_detections(primaries, 7, scale, tracking_config)

    cfg = RecoveryConfig(window=False, permissive=False)
    result = recover(stack, tracks, _RefusingService(), scale, tracking_config, cfg)

    [attempt] = result.attempts
    assert attempt.found and attempt.source == SOURCE_INTENSITY, attempt.detail
    [det] = result.detections
    assert det.x == pytest.approx(160.0, abs=1.5) and det.y == pytest.approx(60.0, abs=1.0)
    assert det.mask_crop is not None and det.confidence == pytest.approx(0.40)


def test_the_intensity_tier_refuses_a_blob_off_the_cells_line(scale, tracking_config):
    """Bright, elongated and the right size -- but two widths across the body from the gap."""
    shape = (7, 120, 400)
    cells = horizontal_cells(range(7))
    painted = [d for d in cells if d.frame != 3] + [
        make_detection(3, 160.0, 60.0 + 25.0, major=140.0, minor=11.0,
                       orientation_rad=HORIZONTAL, with_mask=True)
    ]
    stack = paint(painted, shape, noise_sigma=5.0)
    tracks, _ = track_detections([d for d in cells if d.frame != 3], 7, scale, tracking_config)

    cfg = RecoveryConfig(window=False, permissive=False)
    result = recover(stack, tracks, _RefusingService(), scale, tracking_config, cfg)
    [attempt] = result.attempts
    assert not attempt.found
    assert "across the cell's body" in attempt.detail


def test_a_candidate_in_another_lane_is_refused_when_lanes_are_applied(scale, tracking_config):
    """Lanes enter only as the tracker's lane gate does: applied geometries refuse."""
    lanes = [
        Lane(index=0, origin=(0.0, 40.0), direction=(1.0, 0.0), half_width_px=20.0),
        Lane(index=1, origin=(0.0, 80.0), direction=(1.0, 0.0), half_width_px=20.0),
    ]
    geometry = ChannelGeometry(lanes=lanes, source=GEOMETRY_FROM_RIDGES, confidence=1.0,
                               applied=True)
    shape = (7, 120, 400)
    cells = horizontal_cells(range(7), y=57.0)
    stray = make_detection(3, 160.0, 63.0, major=140.0, minor=11.0,
                           orientation_rad=HORIZONTAL, with_mask=True)
    stack = paint([d for d in cells if d.frame != 3] + [stray], shape)
    primaries = [d for d in cells if d.frame != 3]
    tracks, _ = track_detections(primaries, 7, scale, tracking_config, geometry=geometry)
    assert len(tracks) == 1 and tracks[0].channel == 0

    cfg = RecoveryConfig(intensity=False)
    refused = recover(stack, tracks, _ThresholdService(), scale, tracking_config, cfg,
                      geometry=geometry)
    assert not refused.attempts[0].found
    assert "in lane 1, but the track is in lane 0" in refused.attempts[0].detail

    unconstrained = recover(stack, tracks, _ThresholdService(), scale, tracking_config, cfg)
    assert unconstrained.attempts[0].found, "without lanes the same cell is accepted"

    off = TrackingConfig(channel_constraint="off")
    assert recover(stack, tracks, _ThresholdService(), scale, off, cfg,
                   geometry=geometry).attempts[0].found, "constraint off: the tracker would link it"


def _two_narrow_lanes(y0: float = 40.0, y1: float = 80.0, half_width: float = 8.0):
    lanes = [
        Lane(index=0, origin=(0.0, y0), direction=(1.0, 0.0), half_width_px=half_width),
        Lane(index=1, origin=(0.0, y1), direction=(1.0, 0.0), half_width_px=half_width),
    ]
    return ChannelGeometry(lanes=lanes, source=GEOMETRY_FROM_RIDGES, confidence=1.0, applied=True)


def test_a_candidate_just_outside_every_lane_is_not_refused(scale, tracking_config):
    """Lane -1 is never gated -- by the tracker (``_hard_gates``) or here.

    The review's case: a track in lane 0 (centre y = 40, half-width 8 px) and
    a candidate at y = 49.5, 1.5 px past the measured half-width. The tracker
    would link it; recovery used to refuse it as "in lane -1".
    """
    geometry = _two_narrow_lanes()
    shape = (7, 120, 400)
    cells = horizontal_cells(range(7), y=40.0)
    stray = make_detection(3, 160.0, 49.5, major=140.0, minor=11.0,
                           orientation_rad=HORIZONTAL, with_mask=True)
    primaries = [d for d in cells if d.frame != 3]
    stack = paint(primaries + [stray], shape)
    tracks, _ = track_detections(primaries, 7, scale, tracking_config, geometry=geometry)
    assert len(tracks) == 1 and tracks[0].channel == 0
    assert geometry.lane_of(160.0, 49.5) == -1

    result = recover(stack, tracks, _ThresholdService(), scale, tracking_config,
                     RecoveryConfig(intensity=False), geometry=geometry)
    [attempt] = result.attempts
    assert attempt.found, attempt.detail
    assert result.detections[0].channel == -1


def test_a_track_with_no_lane_is_not_held_to_one(scale, tracking_config):
    """A track never seen inside a lane has no lane to refuse a candidate from."""
    geometry = _two_narrow_lanes(y0=20.0, y1=100.0)
    shape = (7, 120, 400)
    cells = horizontal_cells(range(7), y=60.0)  # between the two lanes
    stray = make_detection(3, 160.0, 96.0, major=140.0, minor=11.0,
                           orientation_rad=HORIZONTAL, with_mask=True)
    primaries = [d for d in cells if d.frame != 3]
    stack = paint(primaries + [stray], shape)
    tracks, _ = track_detections(primaries, 7, scale, tracking_config, geometry=geometry)
    assert len(tracks) == 1 and tracks[0].channel == -1
    assert geometry.lane_of(160.0, 96.0) == 1

    result = recover(stack, tracks, _ThresholdService(), scale, tracking_config,
                     RecoveryConfig(intensity=False), geometry=geometry)
    [attempt] = result.attempts
    assert attempt.found, attempt.detail
    assert result.detections[0].channel == 1


# --------------------------------------------------------------------------
# A cell already detected is not recovered again
# --------------------------------------------------------------------------


def _first_pass_with_the_cell_elsewhere():
    """Track 1 has a gap at frame 3 while the cell's primary detection there is track 2.

    That is the state the measured duplicates came from: the tracker gave the
    frame-3 detection to another track, so track 1's 'missing' frame holds a
    cell that was in fact detected.
    """
    cells = horizontal_cells(range(7), major=100.0)
    track1 = hand_track(1, [d for d in cells if d.frame != 3])
    track2 = hand_track(2, [d for d in cells if d.frame == 3])
    return cells, [track1, track2]


def test_a_duplicate_of_a_primary_cell_runs_two_tracks_along_one_cell(scale, tracking_config):
    """The defect, reproduced: a near-copy of a primary detection splits the track."""
    cells, _ = _first_pass_with_the_cell_elsewhere()
    alone, _ = track_detections(cells, 7, scale, tracking_config)
    assert len(alone) == 1

    copy = make_detection(3, cells[3].x + 1.2, cells[3].y, major=100.0, minor=11.0,
                          orientation_rad=HORIZONTAL, with_mask=True, label=2)
    doubled, _ = track_detections(cells + [copy], 7, scale, tracking_config)
    assert len(doubled) == 2


def test_recovery_drops_a_duplicate_of_a_primary_cell(scale, tracking_config):
    """The fix: the candidate is recognised as the primary cell and not added."""
    cells, first_pass = _first_pass_with_the_cell_elsewhere()
    stack = paint(cells, (7, 120, 400))

    result = recover(stack, first_pass, _ThresholdService(), scale, tracking_config,
                     RecoveryConfig())

    [attempt] = result.attempts
    assert attempt.frame == 3 and attempt.track_id == 1
    assert not attempt.found and result.detections == []
    assert attempt.duplicate_of_label == cells[3].label
    assert attempt.duplicate_of is cells[3]
    assert "dropped as a duplicate" in attempt.detail
    assert result.n_duplicates_dropped == 1

    retracked, _ = track_detections(cells + result.detections, 7, scale, tracking_config)
    assert len(retracked) == 1, "one cell, one track"


def test_two_tracks_cannot_recover_the_same_cell_twice(scale, tracking_config):
    """The second recovery of one cell in one frame is a duplicate of the first."""
    cells = horizontal_cells(range(7), major=100.0)
    stack = paint(cells, (7, 120, 400))
    without = [d for d in cells if d.frame != 3]
    twin = [make_detection(d.frame, d.x, d.y, major=100.0, minor=11.0,
                           orientation_rad=HORIZONTAL, with_mask=True, label=2)
            for d in without]
    first_pass = [hand_track(1, without), hand_track(2, twin)]

    result = recover(stack, first_pass, _ThresholdService(), scale, tracking_config,
                     RecoveryConfig())
    assert [a.found for a in result.attempts] == [True, False]
    assert "already recovered" in result.attempts[1].detail
    assert len(result.detections) == 1
    [kept] = result.detections
    assert result.attempts[1].duplicate_of is kept


def test_a_duplicate_of_a_recovered_cell_names_that_cell(scale, tracking_config):
    """The review's case: two tracks miss frame 3, and an unrelated primary owns label 1 there.

    The cell recovered first used to keep its crop-local label 1 until the
    pipeline relabelled it, so the second attempt's duplicate_of_label named
    the unrelated primary cell. A recovered cell now gets the frame's next
    free label at once, and the row reads the label from the object, so a
    later relabelling is followed too.
    """
    cells = horizontal_cells(range(7), major=100.0)
    stack = paint(cells, (7, 120, 400))
    without = [d for d in cells if d.frame != 3]
    twin = [make_detection(d.frame, d.x, d.y, major=100.0, minor=11.0,
                           orientation_rad=HORIZONTAL, with_mask=True, label=2)
            for d in without]
    bystander = make_detection(3, 330.0, 100.0, major=40.0, minor=11.0, label=1)
    first_pass = [hand_track(1, without), hand_track(2, twin), hand_track(3, [bystander])]

    result = recover(stack, first_pass, _ThresholdService(), scale, tracking_config,
                     RecoveryConfig())
    [kept] = result.detections
    dropped = next(a for a in result.attempts if a.duplicate_of is not None)
    assert dropped.duplicate_of is kept
    assert kept.label == 2, "the next label free in frame 3, not the crop's 1"
    assert dropped.duplicate_of_label == 2 and dropped.to_row()["duplicate_of_label"] == 2
    assert "labelled 1" not in dropped.detail
    assert result.n_duplicates_dropped == 1

    kept.label = 7  # any later relabelling is what the row reports
    assert dropped.to_row()["duplicate_of_label"] == 7


def test_recovered_labels_never_collide_with_the_frames_primaries(scale, tracking_config):
    """The pipeline's relabelling rule, applied where the cell is found."""
    shape = (7, 120, 400)
    cells = horizontal_cells(range(7))
    stack = paint(cells, shape)
    bystanders = [make_detection(3, 330.0, 100.0, major=40.0, minor=11.0, label=lab)
                  for lab in (1, 4)]
    primaries = [d for d in cells if d.frame != 3]
    first_pass = [hand_track(1, primaries)] + [
        hand_track(k + 2, [b]) for k, b in enumerate(bystanders)
    ]
    result = recover(stack, first_pass, _ThresholdService(), scale, tracking_config,
                     RecoveryConfig(intensity=False))
    [det] = result.detections
    assert det.frame == 3 and det.label == 5


@pytest.mark.parametrize(
    "dx, dy, masks, duplicate",
    [
        (0.6, 0.0, True, True),     # measured: 0.6 px from the primary
        (20.7, 0.0, True, True),    # measured: 20.7 px, along the body
        (20.7, 0.0, False, True),   # the same without masks: the body test alone
        (4.5, 1.0, False, True),    # measured: 4.5 px
        (100.0, 0.0, True, False),  # the next cell in the same lane, a length away
        (0.0, 40.0, True, False),   # a cell in the neighbouring lane
        (0.0, 8.0, False, False),   # beside it, outside its width
    ],
)
def test_what_counts_as_a_duplicate(dx, dy, masks, duplicate):
    primary = make_detection(3, 200.0, 60.0, major=100.0, minor=11.0,
                             orientation_rad=HORIZONTAL, with_mask=masks)
    candidate = make_detection(3, 200.0 + dx, 60.0 + dy, major=100.0, minor=11.0,
                               orientation_rad=HORIZONTAL, with_mask=masks)
    found = duplicate_of(candidate, [primary], RecoveryConfig())
    assert (found is not None) == duplicate, found


def test_a_tail_fragment_of_a_primary_cell_is_a_duplicate():
    """Re-segmenting can return part of the cell; its centroid sits inside the body."""
    primary = make_detection(3, 200.0, 60.0, major=100.0, minor=11.0,
                             orientation_rad=HORIZONTAL, with_mask=True)
    tail = make_detection(3, 225.0, 60.0, major=50.0, minor=11.0,
                          orientation_rad=HORIZONTAL, with_mask=True)
    found = duplicate_of(tail, [primary], RecoveryConfig())
    assert found is not None and found[0] is primary


def test_a_different_frame_is_never_a_duplicate():
    primary = make_detection(2, 200.0, 60.0, with_mask=True)
    candidate = make_detection(3, 200.0, 60.0, with_mask=True)
    assert duplicate_of(candidate, [primary], RecoveryConfig()) is None


# --------------------------------------------------------------------------
# Final track ids
# --------------------------------------------------------------------------


def test_attempts_are_rekeyed_to_the_final_track_ids(scale, tracking_config):
    """Critique C4: recovery_attempts.csv must join tracks.csv on track_id."""
    shape = (8, 120, 400)
    cells = horizontal_cells(range(8))
    stack = paint([d for d in cells if d.frame != 5], shape)  # frame 5 truly empty
    primaries = [d for d in cells if d.frame not in (3, 5)]
    tracks, _ = track_detections(primaries, 8, scale, tracking_config)
    assert len(tracks) == 1
    tracks[0].id = 7  # a first-pass id that re-tracking will not reproduce

    result = recover(stack, tracks, _ThresholdService(), scale, tracking_config,
                     RecoveryConfig(intensity=False))
    by_frame = {a.frame: a for a in result.attempts}
    assert by_frame[3].found and not by_frame[5].found
    assert {a.track_id for a in result.attempts} == {7}

    for det in result.detections:  # what the pipeline does: unique (frame, label)
        det.label = 100
    final, _ = track_detections(primaries + result.detections, 8, scale, tracking_config)
    assert len(final) == 1 and final[0].id == 1
    assign_final_track_ids(result.attempts, final)

    for attempt in result.attempts:
        assert attempt.track_id == 1
        assert attempt.first_pass_track_id == 7
    row = by_frame[5].to_row()
    assert row["track_id"] == 1 and row["first_pass_track_id"] == 7
    assert (row["bracket_frame_before"], row["bracket_label_before"],
            row["bracket_frame_after"], row["bracket_label_after"]) == (4, 1, 6, 1)


def test_an_attempt_no_final_track_holds_is_not_given_a_stale_id():
    attempt_track = hand_track(3, [make_detection(0, 45.0, 20.0), make_detection(2, 45.0, 60.0)])
    from corridor.core.recovery import RecoveryAttempt

    attempt = RecoveryAttempt(3, 1, 45.0, 40.0, found=False, bracket=(0, 9, 2, 9))
    assign_final_track_ids([attempt], [attempt_track])
    assert attempt.track_id is None and attempt.first_pass_track_id == 3
    assert "no final track" in attempt.detail


# --------------------------------------------------------------------------
# Nothing is dropped in silence
# --------------------------------------------------------------------------


def test_every_enumerated_gap_produces_an_attempt(
    straight_track_with_a_hole, scale, tracking_config
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
    result = recover(stack, tracks, _RefusingService(), scale, tracking_config, cfg)

    assert len(result.attempts) == len(enumerated)
    assert {a.frame for a in result.attempts} == {f for _, f in enumerated}
    assert 3 in {a.frame for a in result.attempts}, "the interior gap must be examined"
    assert not result.detections, "an empty image contains no cells to recover"
    assert all(a.bracket is not None for a in result.attempts)


def test_recovery_disabled_examines_nothing(straight_track_with_a_hole, scale, tracking_config):
    track, tracks = straight_track_with_a_hole
    stack = np.zeros((7, 324, 90), dtype=np.uint16)

    result = recover(stack, tracks, _RefusingService(), scale, tracking_config,
                     RecoveryConfig(enabled=False))
    assert result.attempts == []
    assert result.detections == []


def test_the_v1_argument_order_still_runs(scale, tracking_config):
    """Transition, like ``track_detections``: a v1 pipeline passes an axis fourth.

    Without it the v1 pipeline on this branch raised TypeError on every run
    with recovery on (review of E2). The axis only supplies lanes.
    """
    shape = (7, 120, 400)
    stack = paint(horizontal_cells(range(7)), shape)
    primaries = [d for d in horizontal_cells(range(7)) if d.frame != 3]
    tracks, _ = track_detections(primaries, 7, scale, tracking_config)
    axis = SimpleNamespace(ux=1.0, uy=0.0, channels=[], source="none", confidence=0.0,
                           pitch_px=None, notes=[])

    v1 = recover(stack, tracks, _ThresholdService(), axis, scale, tracking_config,
                 RecoveryConfig())
    v2 = recover(stack, tracks, _ThresholdService(), scale, tracking_config, RecoveryConfig())
    assert [a.to_row() for a in v1.attempts] == [a.to_row() for a in v2.attempts]
    assert v1.attempts[0].found
    with pytest.raises(TypeError, match="recovery_cfg"):
        recover(stack, tracks, _ThresholdService(), axis, scale, tracking_config)
    with pytest.raises(TypeError, match="keyword-only"):
        recover(stack, tracks, _ThresholdService(), scale, tracking_config, RecoveryConfig(),
                RecoveryConfig())


def test_a_3d_stack_is_not_searched_in_silence(straight_track_with_a_hole, scale, tracking_config):
    _, tracks = straight_track_with_a_hole
    result = recover(np.zeros((7, 3, 64, 64)), tracks, _RefusingService(), scale,
                     tracking_config, RecoveryConfig())
    assert result.attempts == [] and result.notes


# --------------------------------------------------------------------------
# Interior gaps and trailing frames are different questions
# --------------------------------------------------------------------------


def test_trailing_frames_are_not_probed_by_default(straight_track_with_a_hole, tracking_config):
    """The measured reason recovery is safe to run at all.

    An interior gap is a cell known to have been present on both sides. A frame
    past the end of a track is a cell that may simply have left, and on a
    channel known to be empty every false position came from exactly there.
    """
    _, tracks = straight_track_with_a_hole
    gaps = _dormant_frames(tracks, 12, tracking_config, RecoveryConfig(enabled=True))

    assert {f for _, f in gaps} == {3}, "only the interior hole should be probed"


def test_trailing_frames_are_probed_when_asked_for(straight_track_with_a_hole, tracking_config):
    _, tracks = straight_track_with_a_hole
    gaps = _dormant_frames(
        tracks, 12, tracking_config, RecoveryConfig(enabled=True, trailing=True)
    )
    frames = {f for _, f in gaps}

    assert 3 in frames
    assert frames & {7, 8, 9, 10}, "frames past the track's end should appear"


def test_interior_can_be_switched_off_independently(straight_track_with_a_hole, tracking_config):
    _, tracks = straight_track_with_a_hole
    gaps = _dormant_frames(
        tracks, 12, tracking_config,
        RecoveryConfig(enabled=True, interior=False, trailing=True),
    )
    assert 3 not in {f for _, f in gaps}


def test_the_shipped_default_probes_interior_gaps_only():
    """Changing either of these changes what every published figure describes."""
    cfg = RecoveryConfig()
    assert cfg.enabled is True
    assert cfg.interior is True
    assert cfg.trailing is False
    assert (cfg.window, cfg.permissive, cfg.intensity) == (True, True, True)
