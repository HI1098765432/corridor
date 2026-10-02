"""Quality-control findings that describe what a trajectory is made of.

These two checks exist because of a measurement, not a hunch. Running the
fallback ladder over the supplied stacks (scripts/experiment_fallback.py) showed
that the assignment step does not reject a false detection which repeats in the
same place: a stationary object is the most self-consistent thing a
predicted-position cost model can be shown. So the software says so instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from corridor.core.config import Scale, TrackingConfig
from corridor.core.imaging import SOURCE_USER, Calibrated, StackMetadata
from corridor.core.measurements import summarise
from corridor.core.qc import (
    STATIONARY_MIN_MINUTES,
    STATIONARY_NET_UM,
    collect_issues,
)
from corridor.core.tracking import track_detections

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


def issues_for(detections, n_frames, axis, scale, cfg, metadata):
    tracks, events = track_detections(detections, n_frames, axis, scale, cfg)
    summaries = summarise(tracks, axis, scale)
    return collect_issues(
        metadata, axis, scale, [], events, tracks, summaries, cfg
    ), tracks


def codes(issues) -> set[str]:
    return {i.code for i in issues}


def test_a_cell_that_never_moves_is_flagged(vertical_axis, scale, tracking_config, metadata):
    """A fixed feature of the device tracks perfectly and migrates nowhere."""
    dets = [make_detection(frame, 45.0, 100.0) for frame in range(6)]
    issues, _ = issues_for(dets, 6, vertical_axis, scale, tracking_config, metadata)

    assert "stationary_track" in codes(issues)
    flagged = next(i for i in issues if i.code == "stationary_track")
    assert flagged.track_id == 1
    assert "less than the width of a cell" in flagged.detail


def test_a_migrating_cell_is_not_flagged(vertical_axis, scale, tracking_config, metadata):
    dets = straight_track(6, step=20.0)
    issues, _ = issues_for(dets, 6, vertical_axis, scale, tracking_config, metadata)

    assert "stationary_track" not in codes(issues)


def test_a_brief_pause_is_not_called_stationary(
    vertical_axis, scale, tracking_config, metadata
):
    """Two frames is not enough sequence to accuse a cell of standing still."""
    dets = [make_detection(frame, 45.0, 100.0) for frame in range(2)]
    issues, _ = issues_for(dets, 2, vertical_axis, scale, tracking_config, metadata)

    assert "stationary_track" not in codes(issues)


def test_a_fixed_object_with_a_jittery_centroid_is_still_caught(
    vertical_axis, scale, tracking_config, metadata
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
    issues, _ = issues_for(dets, len(wobble), vertical_axis, scale, tracking_config, metadata)

    assert "stationary_track" in codes(issues)


def test_a_migrating_cell_in_a_fast_acquisition_is_not_accused(
    vertical_axis, tracking_config, metadata
):
    """The reason the minimum is in minutes rather than frames.

    At half a minute per frame, four frames is two minutes of experiment. A
    cell moving at a perfectly healthy 2 µm/min covers 3 µm in that time, which
    is under the 4 µm threshold — and calling that cell "a fixed feature of the
    device" would teach the reader to ignore this warning.
    """
    fast = Scale.from_values(0.25, 0.5)
    # 2 um/min at 0.25 um/px and 0.5 min/frame is 4 px per frame.
    dets = [make_detection(frame, 45.0, 100.0 + 4.0 * frame) for frame in range(4)]
    issues, _ = issues_for(dets, 4, vertical_axis, fast, tracking_config, metadata)

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
    vertical_axis, scale, tracking_config, metadata
):
    """The recall/precision trade has to be visible in the result, not just the docs."""
    dets = straight_track(6, step=20.0)
    for det in dets[2:]:  # four of six positions came from a permissive pass
        det.source = "ensemble"
        det.confidence = 0.75

    issues, _ = issues_for(dets, 6, vertical_axis, scale, tracking_config, metadata)

    assert "fallback_dependent_track" in codes(issues)
    flagged = next(i for i in issues if i.code == "fallback_dependent_track")
    assert "4 of its 6 positions" in flagged.detail


def test_an_ordinary_track_is_not_flagged_as_fallback_dependent(
    vertical_axis, scale, tracking_config, metadata
):
    dets = straight_track(6, step=20.0)
    issues, _ = issues_for(dets, 6, vertical_axis, scale, tracking_config, metadata)

    assert "fallback_dependent_track" not in codes(issues)


def test_one_borrowed_position_in_a_long_track_is_not_flagged(
    vertical_axis, scale, tracking_config, metadata
):
    """The warning is about a trajectory that *rests* on the fallback.

    A single recovered position in an otherwise ordinary track is the fallback
    working as intended, and warning about it would train the reader to ignore
    the warning.
    """
    dets = straight_track(8, step=20.0)
    dets[3].source = "ensemble"

    issues, _ = issues_for(dets, 8, vertical_axis, scale, tracking_config, metadata)
    assert "fallback_dependent_track" not in codes(issues)
