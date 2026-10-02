"""Physical units, velocities and summaries.

Tracks are assembled from detections by hand, not by the tracker: these
tests pin measurement arithmetic, and a tracker in the loop would make them
fail for reasons that are not about measurement (the tracker has its own
tests).  The detections are the conftest's real-cell-shaped ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pytest

from corridor.core.config import Scale
from corridor.core.measurements import frame_rows, summarise

from conftest import (
    FRAME_INTERVAL_MIN,
    PIXEL_SIZE_UM,
    make_detection,
    straight_track,
)


@dataclass
class Obs:
    frame: int
    x: float
    y: float
    area_px: float
    det_label: int = 1
    channel: int = 0
    gap_frames: int = 0
    cost: float | None = None
    link_margin: float | None = None
    source: str = "primary"
    confidence: float = 1.0
    z: float | None = None
    detection: Any = None


@dataclass
class Trk:
    id: int
    observations: list[Obs]
    flags: set[str] = field(default_factory=set)
    channel: int = 0


def as_track(detections, tid: int = 1) -> Trk:
    obs = [
        Obs(frame=d.frame, x=d.x, y=d.y, area_px=d.area_px, det_label=d.label, detection=d)
        for d in sorted(detections, key=lambda d: d.frame)
    ]
    for k in range(1, len(obs)):
        obs[k].gap_frames = obs[k].frame - obs[k - 1].frame
    return Trk(tid, obs)


SPEED_20PX = 20.0 * PIXEL_SIZE_UM / FRAME_INTERVAL_MIN


def test_velocity_is_exact_for_a_known_displacement(scale):
    """Deterministic guard against the 10-minute timing error returning.

    A cell moving exactly 20 px per frame, at 0.467060343 um/px and
    20.006894938 min/frame, has a speed of

        20 * 0.467060343 / 20.006894938 = 0.46690... um/min

    Any silent reversion to dt = 10 min would double this.
    """
    rows = frame_rows([as_track(straight_track(5, step=20.0))], scale)

    assert SPEED_20PX == pytest.approx(0.4668994, abs=1e-6)
    moving = [r for r in rows if r["speed_um_per_min"] is not None]
    assert len(moving) == 4
    for row in moving:
        assert row["speed_um_per_min"] == pytest.approx(SPEED_20PX, rel=1e-9)
        assert row["speed_um_per_hr"] == pytest.approx(SPEED_20PX * 60.0, rel=1e-9)
        assert row["speed_px_per_frame"] == pytest.approx(20.0, rel=1e-9)
        assert row["vx_um_per_min"] == pytest.approx(0.0, abs=1e-12)
        assert row["vy_um_per_min"] == pytest.approx(SPEED_20PX, rel=1e-9)


def test_ten_minute_assumption_would_double_the_speed():
    """Explicitly demonstrate the magnitude of the original defect."""
    correct = Scale.from_values(PIXEL_SIZE_UM, FRAME_INTERVAL_MIN)
    wrong = Scale.from_values(0.46, 10.0)  # the notebook's hard-coded values
    track = as_track(straight_track(3, step=20.0))
    ok = frame_rows([track], correct)[1]["speed_um_per_min"]
    bad = frame_rows([track], wrong)[1]["speed_um_per_min"]
    assert bad / ok == pytest.approx(1.9704, rel=1e-3)


def test_velocity_after_a_gap_uses_elapsed_time(scale):
    """A cell seen at t=0 and t=3 moved over three frames, not one."""
    rows = frame_rows([as_track(straight_track(6, step=20.0, skip={1, 2}))], scale)
    bridged = next(r for r in rows if r["frame"] == 3)
    assert bridged["gap_frames"] == 3
    assert bridged["speed_um_per_min"] == pytest.approx(SPEED_20PX, rel=1e-9)
    assert bridged["distance_from_previous_um"] == pytest.approx(60.0 * PIXEL_SIZE_UM)


def test_first_observation_has_no_velocity(scale):
    rows = frame_rows([as_track(straight_track(4))], scale)
    for key in ("speed_um_per_min", "speed_um_per_hr", "vx_um_per_min", "distance_from_previous_um"):
        assert rows[0][key] is None
    assert rows[0]["cumulative_path_um"] == 0.0
    assert rows[0]["distance_from_start_um"] == 0.0


def test_uncalibrated_data_reports_pixels_only():
    """Without calibration the µm columns stay empty rather than silently wrong."""
    bare = Scale.from_values(None, None)
    assert not bare.calibrated
    assert bare.pixel_size_um == 1.0 and bare.frame_interval_min == 1.0

    rows = frame_rows([as_track(straight_track(4, step=20.0))], bare)
    assert rows[1]["speed_px_per_frame"] == pytest.approx(20.0)
    assert rows[1]["cumulative_path_px"] == pytest.approx(20.0)
    for key in ("speed_um_per_min", "speed_um_per_hr", "x_um", "area_um2",
                "major_axis_um", "cumulative_path_um"):
        assert rows[1][key] is None, f"{key} must be empty rather than silently wrong"


def test_source_frames_are_preserved(scale):
    """Original acquisition frame numbers survive into the output."""
    source = list(range(42, 53))  # 052924_t3_dual carries t:42/54 .. t:52/54
    rows = frame_rows([as_track(straight_track(11))], scale, source_frames=source)
    assert rows[0]["source_frame"] == 42
    assert rows[-1]["source_frame"] == 52


def test_elapsed_time_matches_frame_index(scale):
    for row in frame_rows([as_track(straight_track(4))], scale):
        assert row["elapsed_min"] == pytest.approx(row["frame"] * FRAME_INTERVAL_MIN)
        assert row["elapsed_hr"] == pytest.approx(row["frame"] * FRAME_INTERVAL_MIN / 60.0)


def test_morphology_columns_come_from_the_detection(scale):
    rows = frame_rows([as_track(straight_track(2))], scale)
    assert rows[0]["major_axis_px"] == 90.0 and rows[0]["minor_axis_px"] == 11.0
    assert rows[0]["major_axis_um"] == pytest.approx(90.0 * PIXEL_SIZE_UM)
    assert rows[0]["solidity"] == 0.95


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def test_summary_of_a_straight_track(scale):
    (summary,) = summarise([as_track(straight_track(6, step=20.0))], scale)

    assert summary.n_observations == 6
    assert summary.first_frame == 0 and summary.last_frame == 5
    assert summary.n_gaps == 0
    assert summary.span_frames == 5
    assert summary.dimensionality == "2D"
    assert summary.duration_min == pytest.approx(5 * FRAME_INTERVAL_MIN)
    assert summary.duration_hr == pytest.approx(5 * FRAME_INTERVAL_MIN / 60.0)
    assert summary.net_displacement_um == pytest.approx(100.0 * PIXEL_SIZE_UM)
    assert summary.path_length_um == pytest.approx(100.0 * PIXEL_SIZE_UM)
    assert summary.max_distance_from_start_um == pytest.approx(100.0 * PIXEL_SIZE_UM)
    assert summary.net_displacement_px == pytest.approx(100.0)
    assert summary.straightness == pytest.approx(1.0)
    assert summary.mean_speed_um_per_min == pytest.approx(SPEED_20PX)
    assert summary.median_speed_um_per_min == pytest.approx(SPEED_20PX)
    assert summary.max_speed_um_per_min == pytest.approx(SPEED_20PX)
    assert summary.mean_turning_angle_deg == pytest.approx(0.0)
    assert summary.mean_area_um2 == pytest.approx(800.0 * PIXEL_SIZE_UM**2)


def test_every_hourly_summary_value_is_sixty_times_the_per_minute_one(scale):
    dets = [make_detection(t, 45.0 + 3.0 * (t % 2), 20.0 + 17.0 * t) for t in range(8)]
    (s,) = summarise([as_track(dets)], scale)
    for name in ("mean_speed", "median_speed", "max_speed", "net_speed", "path_speed"):
        assert getattr(s, f"{name}_um_per_hr") == pytest.approx(
            getattr(s, f"{name}_um_per_min") * 60.0
        )


def test_summary_counts_gaps(scale):
    (summary,) = summarise([as_track(straight_track(8, skip={2, 5}))], scale)
    assert summary.n_observations == 6
    assert summary.n_gaps == 2
    assert summary.total_missing_frames == 2


def test_straightness_detects_a_wandering_path(scale):
    ys = [20, 60, 40, 80, 60, 100]
    (summary,) = summarise([as_track([make_detection(t, 45.0, y) for t, y in enumerate(ys)])], scale)
    assert summary.straightness < 0.9


def test_short_track_is_flagged_as_a_fragment(scale):
    (summary,) = summarise([as_track([make_detection(0, 45.0, 100.0)])], scale, min_observations=2)
    assert "fragment" in summary.flags
    assert summary.mean_speed_um_per_min is None


def test_the_weakest_link_is_reported(scale):
    track = as_track(straight_track(4))
    track.observations[1].link_margin = 9.0
    track.observations[2].link_margin = 2.5
    (summary,) = summarise([track], scale)
    assert summary.min_link_margin_chi2 == 2.5


# --------------------------------------------------------------------------
# The robust speed estimators
# --------------------------------------------------------------------------


def test_net_speed_divides_by_elapsed_time_not_by_observation_count(scale):
    """The arithmetic guard on the estimator the accuracy claim rests on.

    A cell moving 20 px per frame for 10 frames covers 200 px in 10 frame
    intervals. Dividing by the number of *observations* instead of the elapsed
    span is the same class of error as the notebook's 10-minute interval, and
    it would show up here as a value off by a factor of 11/10.
    """
    (summary,) = summarise([as_track(straight_track(11, step=20.0))], scale)
    expected = 200.0 * PIXEL_SIZE_UM / (10 * FRAME_INTERVAL_MIN)
    assert summary.net_speed_um_per_min == pytest.approx(expected, rel=1e-9)
    assert summary.net_speed_um_per_hr == pytest.approx(expected * 60.0, rel=1e-9)
    assert summary.span_frames == 10
    assert summary.n_observations == 11


def test_a_missing_interior_detection_does_not_move_the_net_speed(scale):
    """Why net displacement is quoted at a higher accuracy than mean speed.

    This is the measured result from scripts/experiment_velocity_robustness.py
    reduced to its cause: the net estimator reads only the first and last
    observation, so losing one in between is invisible to it.
    """
    complete = summarise([as_track(straight_track(11, step=20.0))], scale)[0]
    gappy = summarise([as_track(straight_track(11, step=20.0, skip={4, 7}))], scale)[0]
    assert gappy.net_speed_um_per_min == pytest.approx(complete.net_speed_um_per_min, rel=1e-9)


def test_net_speed_falls_below_path_speed_when_the_cell_turns_back(scale):
    """The two numbers must not be interchangeable, or reporting both is a lie."""
    positions = [20.0, 60.0, 100.0, 60.0, 20.0]
    dets = [make_detection(frame, 45.0, y) for frame, y in enumerate(positions)]
    (summary,) = summarise([as_track(dets)], scale)

    # It ends where it started, so net progress is zero and path speed is not.
    assert summary.net_speed_um_per_min == pytest.approx(0.0, abs=1e-9)
    assert summary.path_speed_um_per_min > 0.4
    assert summary.straightness == pytest.approx(0.0, abs=1e-9)
    # It went 80 px out before coming back.
    assert summary.max_distance_from_start_um == pytest.approx(80.0 * PIXEL_SIZE_UM)


def test_speeds_are_absent_rather_than_wrong_without_a_calibration():
    """An uncalibrated dataset must report no micrometres, not fake ones."""
    uncalibrated = Scale.from_values(None, None)
    (summary,) = summarise([as_track(straight_track(5, step=20.0))], uncalibrated)

    for name in ("net_speed_um_per_min", "net_speed_um_per_hr", "path_speed_um_per_min",
                 "mean_speed_um_per_hr", "net_displacement_um", "mean_area_um2",
                 "duration_min", "persistence_time_min"):
        assert getattr(summary, name) is None, name
    assert summary.net_speed_px_per_frame == pytest.approx(20.0)
    assert summary.mean_speed_px_per_frame == pytest.approx(20.0)
    assert summary.net_displacement_px == pytest.approx(80.0)


def test_a_track_seen_only_once_has_no_speed_at_all(scale):
    """Zero elapsed time must not become a division by zero or an infinity."""
    (summary,) = summarise([as_track([make_detection(0, 45.0, 20.0)])], scale)
    assert summary.span_frames == 0
    assert summary.net_speed_um_per_min is None
    assert summary.path_speed_um_per_min is None
    assert summary.net_speed_px_per_frame is None
    assert summary.msd_alpha is None


# --------------------------------------------------------------------------
# Three dimensions
# --------------------------------------------------------------------------


def _track_3d(points) -> Trk:
    return Trk(1, [Obs(frame=k, x=x, y=y, z=z, area_px=50.0) for k, (x, y, z) in enumerate(points)])


def test_three_d_summary_needs_a_z_step():
    points = [(0.0, 0.0, 0.0), (6.0, 8.0, 0.0), (6.0, 8.0, 3.0)]
    (with_z,) = summarise([_track_3d(points)], Scale.from_values(0.5, 60.0, z_step_um=2.0))
    assert with_z.dimensionality == "3D"
    assert with_z.path_length_um == pytest.approx(5.0 + 6.0)
    assert with_z.net_displacement_um == pytest.approx(math.sqrt(9 + 16 + 36))

    (without_z,) = summarise([_track_3d(points)], Scale.from_values(0.5, 60.0))
    for name in ("path_length_um", "net_displacement_um", "net_displacement_px",
                 "mean_speed_um_per_hr", "straightness", "msd_alpha"):
        assert getattr(without_z, name) is None, name


# --------------------------------------------------------------------------
# Transition
# --------------------------------------------------------------------------


def test_the_legacy_axis_argument_is_accepted_and_ignored(scale):
    track = as_track(straight_track(5))
    assert summarise([track], object(), scale) == summarise([track], scale)


def test_axis_fields_read_none_for_v1_readers(scale):
    """qc.lateral_drift still reads these until the integration rewrites it."""
    (summary,) = summarise([as_track(straight_track(5))], scale)
    assert summary.net_along_um is None and summary.net_across_um is None
    assert summary.along_speed_um_per_min is None
    assert "net_along_um" not in summary.to_row()
