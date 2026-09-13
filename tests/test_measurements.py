"""Physical units, velocities and summaries."""

from __future__ import annotations

import math

import numpy as np
import pytest

from corridor.core.config import Scale
from corridor.core.measurements import frame_rows, summarise
from corridor.core.tracking import track_detections

from conftest import (
    FRAME_INTERVAL_MIN,
    PIXEL_SIZE_UM,
    make_detection,
    straight_track,
)


def test_velocity_is_exact_for_a_known_displacement(vertical_axis, scale, tracking_config):
    """Deterministic guard against the 10-minute timing error returning.

    A cell moving exactly 20 px per frame, at 0.467060343 um/px and
    20.006894938 min/frame, has a speed of

        20 * 0.467060343 / 20.006894938 = 0.46690... um/min

    Any silent reversion to dt = 10 min would double this.
    """
    dets = straight_track(5, step=20.0)
    tracks, _ = track_detections(dets, 5, vertical_axis, scale, tracking_config)
    rows = frame_rows(tracks, vertical_axis, scale)

    expected = 20.0 * PIXEL_SIZE_UM / FRAME_INTERVAL_MIN
    assert expected == pytest.approx(0.4668994, abs=1e-6)

    moving = [r for r in rows if r["speed_um_per_min"] is not None]
    assert len(moving) == 4
    for row in moving:
        assert row["speed_um_per_min"] == pytest.approx(expected, rel=1e-9)
        assert row["speed_px_per_frame"] == pytest.approx(20.0, rel=1e-9)
        assert row["v_along_um_per_min"] == pytest.approx(expected, rel=1e-9)
        assert row["v_across_um_per_min"] == pytest.approx(0.0, abs=1e-9)


def test_ten_minute_assumption_would_double_the_speed(vertical_axis, tracking_config):
    """Explicitly demonstrate the magnitude of the original defect."""
    correct = Scale.from_values(PIXEL_SIZE_UM, FRAME_INTERVAL_MIN)
    wrong = Scale.from_values(0.46, 10.0)  # the notebook's hard-coded values

    dets = straight_track(3, step=20.0)
    t_ok, _ = track_detections(dets, 3, vertical_axis, correct, tracking_config)
    t_bad, _ = track_detections(dets, 3, vertical_axis, wrong, tracking_config)

    ok = frame_rows(t_ok, vertical_axis, correct)[1]["speed_um_per_min"]
    bad = frame_rows(t_bad, vertical_axis, wrong)[1]["speed_um_per_min"]
    assert bad / ok == pytest.approx(1.9704, rel=1e-3)


def test_velocity_after_a_gap_uses_elapsed_time(vertical_axis, scale, tracking_config):
    """A cell seen at t=0 and t=3 moved over three frames, not one."""
    dets = straight_track(6, step=20.0, skip={1, 2})
    tracks, _ = track_detections(dets, 6, vertical_axis, scale, tracking_config)
    rows = frame_rows(tracks, vertical_axis, scale)
    bridged = next(r for r in rows if r["frame"] == 3)
    assert bridged["gap_frames"] == 3
    expected = 20.0 * PIXEL_SIZE_UM / FRAME_INTERVAL_MIN
    assert bridged["speed_um_per_min"] == pytest.approx(expected, rel=1e-9)


def test_first_observation_has_no_velocity(vertical_axis, scale, tracking_config):
    dets = straight_track(4)
    tracks, _ = track_detections(dets, 4, vertical_axis, scale, tracking_config)
    rows = frame_rows(tracks, vertical_axis, scale)
    assert rows[0]["speed_um_per_min"] is None
    assert rows[0]["vx_um_per_min"] is None


def test_uncalibrated_data_reports_pixels_only(vertical_axis, tracking_config):
    """With no calibration, parameters mean pixels and frames, and the
    micrometre columns stay empty rather than being silently wrong."""
    bare = Scale.from_values(None, None)
    assert not bare.calibrated
    assert bare.pixel_size_um == 1.0 and bare.frame_interval_min == 1.0

    # The gates are now read in px per frame, so they must be set in those units.
    tracking_config.max_speed_um_per_min = 50.0
    tracking_config.sigma_along_um = 6.0
    tracking_config.max_perp_um = 8.0

    dets = straight_track(4, step=20.0)
    tracks, _ = track_detections(dets, 4, vertical_axis, bare, tracking_config)
    assert len(tracks) == 1
    rows = frame_rows(tracks, vertical_axis, bare)
    assert rows[1]["speed_px_per_frame"] == pytest.approx(20.0)
    assert rows[1]["speed_um_per_min"] is None, (
        "micrometre columns must be empty rather than silently wrong"
    )
    assert "x_um" not in rows[1] or rows[1].get("x_um") is None


def test_uncalibrated_defaults_refuse_rather_than_guess(vertical_axis, tracking_config):
    """The physical defaults must not quietly become plausible pixel values.

    5 um/min read as 5 px/frame rejects a 20 px step. Producing fragments plus
    a loud calibration warning is the honest outcome; inventing a pixel size
    would not be.
    """
    bare = Scale.from_values(None, None)
    dets = straight_track(4, step=20.0)
    tracks, _ = track_detections(dets, 4, vertical_axis, bare, tracking_config)
    assert len(tracks) == 4, "each detection should stand alone under the px reading"


def test_source_frames_are_preserved(vertical_axis, scale, tracking_config):
    """Original acquisition frame numbers survive into the output."""
    source = list(range(42, 53))  # 052924_t3_dual carries t:42/54 .. t:52/54
    dets = straight_track(11)
    tracks, _ = track_detections(dets, 11, vertical_axis, scale, tracking_config)
    rows = frame_rows(tracks, vertical_axis, scale, source_frames=source)
    assert rows[0]["source_frame"] == 42
    assert rows[-1]["source_frame"] == 52


def test_elapsed_time_matches_frame_index(vertical_axis, scale, tracking_config):
    dets = straight_track(4)
    tracks, _ = track_detections(dets, 4, vertical_axis, scale, tracking_config)
    rows = frame_rows(tracks, vertical_axis, scale)
    for row in rows:
        assert row["elapsed_min"] == pytest.approx(row["frame"] * FRAME_INTERVAL_MIN)


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def test_summary_of_a_straight_track(vertical_axis, scale, tracking_config):
    dets = straight_track(6, step=20.0)
    tracks, _ = track_detections(dets, 6, vertical_axis, scale, tracking_config)
    (summary,) = summarise(tracks, vertical_axis, scale)

    assert summary.n_observations == 6
    assert summary.first_frame == 0 and summary.last_frame == 5
    assert summary.n_gaps == 0
    assert summary.span_frames == 5
    assert summary.duration_min == pytest.approx(5 * FRAME_INTERVAL_MIN)
    assert summary.net_displacement_um == pytest.approx(100.0 * PIXEL_SIZE_UM)
    assert summary.path_length_um == pytest.approx(100.0 * PIXEL_SIZE_UM)
    assert summary.straightness == pytest.approx(1.0)
    expected_speed = 20.0 * PIXEL_SIZE_UM / FRAME_INTERVAL_MIN
    assert summary.mean_speed_um_per_min == pytest.approx(expected_speed)
    assert summary.median_speed_um_per_min == pytest.approx(expected_speed)
    assert summary.max_speed_um_per_min == pytest.approx(expected_speed)
    assert summary.net_across_um == pytest.approx(0.0, abs=1e-9)


def test_summary_counts_gaps(vertical_axis, scale, tracking_config):
    dets = straight_track(8, skip={2, 5})
    tracks, _ = track_detections(dets, 8, vertical_axis, scale, tracking_config)
    (summary,) = summarise(tracks, vertical_axis, scale)
    assert summary.n_observations == 6
    assert summary.n_gaps == 2
    assert summary.total_missing_frames == 2


def test_straightness_detects_a_wandering_path(vertical_axis, scale, tracking_config):
    ys = [20, 60, 40, 80, 60, 100]
    dets = [make_detection(t, 45.0, y) for t, y in enumerate(ys)]
    tracks, _ = track_detections(dets, len(ys), vertical_axis, scale, tracking_config)
    summaries = summarise(tracks, vertical_axis, scale)
    long_track = max(summaries, key=lambda s: s.n_observations)
    if long_track.straightness is not None:
        assert long_track.straightness < 0.9


def test_short_track_is_flagged_as_a_fragment(vertical_axis, scale, tracking_config):
    dets = [make_detection(0, 45.0, 100.0)]
    tracks, _ = track_detections(dets, 1, vertical_axis, scale, tracking_config)
    (summary,) = summarise(tracks, vertical_axis, scale, min_observations=2)
    assert "fragment" in summary.flags
    assert summary.mean_speed_um_per_min is None
