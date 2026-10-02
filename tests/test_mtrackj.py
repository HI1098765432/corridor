"""MTrackJ-equivalent per-observation measurements (contract §6).

Tracks are built by hand: these tests pin the arithmetic of measurement, and
a tracker in the loop would make them fail for reasons that are not about
measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from corridor.core.config import Scale
from corridor.core.detections import Detection
from corridor.core.export import TRACK_COLUMNS
from corridor.core.measurements import add_reference_distance, frame_rows


@dataclass
class Obs:
    frame: int
    x: float
    y: float
    z: float | None = None
    area_px: float = 100.0
    det_label: int = 1
    channel: int = 0
    gap_frames: int = 0
    cost: float | None = None
    link_margin: float | None = None
    source: str = "primary"
    confidence: float = 1.0
    detection: Any = None


@dataclass
class Trk:
    id: int
    observations: list[Obs]
    flags: set[str] = field(default_factory=set)
    channel: int = 0


def track(points, frames=None, tid=1, **kw) -> Trk:
    frames = list(range(len(points))) if frames is None else list(frames)
    obs = []
    for f, p in zip(frames, points):
        z = p[2] if len(p) > 2 else None
        obs.append(Obs(frame=f, x=float(p[0]), y=float(p[1]), z=z, **kw))
    return Trk(tid, obs)


# 0.5 µm/px and one frame per hour: pixel positions (0,0), (6,8), (12,16) are
# the directive's (0,0), (3,4), (6,8) µm at t = 0, 1, 2 h.
HOURLY = Scale.from_values(0.5, 60.0)
DIRECTIVE = [(0, 0), (6, 8), (12, 16)]


def column(rows, key):
    return [r[key] for r in rows]


def test_the_directive_case_exactly():
    rows = frame_rows([track(DIRECTIVE)], HOURLY)

    assert column(rows, "distance_from_previous_um") == [None, 5.0, 5.0]  # D2P
    assert column(rows, "cumulative_path_um") == [0.0, 5.0, 10.0]  # Len
    assert column(rows, "distance_from_start_um") == [0.0, 5.0, 10.0]  # D2S
    assert column(rows, "speed_um_per_hr") == [None, 5.0, 5.0]
    assert rows[1]["speed_um_per_min"] == pytest.approx(5.0 / 60.0)
    # No reference point was set, so D2R is empty rather than zero.
    assert column(rows, "distance_from_reference_um") == [None, None, None]
    assert column(rows, "distance_from_reference_px") == [None, None, None]
    # Straight at constant speed: no acceleration, no turning.
    assert column(rows, "acceleration_um_per_hr2") == [None, None, 0.0]
    assert column(rows, "turning_angle_deg") == [None, None, 0.0]
    assert column(rows, "elapsed_hr") == [0.0, 1.0, 2.0]


def test_speed_across_a_gap_uses_the_elapsed_time():
    """Frames 0, 1, 4: the last 5 µm took three hours, not one."""
    rows = frame_rows([track(DIRECTIVE, frames=[0, 1, 4])], HOURLY)

    assert column(rows, "gap_frames") == [0, 1, 3]
    assert column(rows, "distance_from_previous_um") == [None, 5.0, 5.0]
    assert column(rows, "cumulative_path_um") == [0.0, 5.0, 10.0]
    assert rows[1]["speed_um_per_hr"] == pytest.approx(5.0)
    assert rows[2]["speed_um_per_hr"] == pytest.approx(5.0 / 3.0)
    # Speed fell from 5 to 5/3 µm/hr between step midpoints 0.5 h and 2.5 h.
    assert rows[2]["acceleration_um_per_hr2"] == pytest.approx((5.0 / 3.0 - 5.0) / 2.0)


def test_every_hourly_value_is_the_per_minute_value_times_sixty():
    rng = np.random.default_rng(3)
    points = np.cumsum(rng.normal(scale=5.0, size=(12, 2)), axis=0)
    scale = Scale.from_values(0.467060343, 20.006894938)
    rows = frame_rows([track(points, frames=[0, 1, 2, 4, 5, 6, 9, 10, 11, 12, 14, 15])], scale)
    for row in rows[1:]:
        for axis in ("vx", "vy", "speed"):
            assert row[f"{axis}_um_per_hr"] == pytest.approx(row[f"{axis}_um_per_min"] * 60.0)
        assert row["speed_um_per_min"] == pytest.approx(
            row["distance_from_previous_um"] / (row["gap_frames"] * scale.frame_interval_min)
        )


def test_reference_distance_only_when_a_reference_is_set():
    rows = frame_rows([track(DIRECTIVE)], HOURLY, reference_point_px=(0.0, 16.0))
    # Reference at (0, 8) µm.
    assert column(rows, "distance_from_reference_um") == pytest.approx(
        [8.0, math.hypot(3.0, 4.0), 6.0]
    )
    assert column(rows, "distance_from_reference_px") == pytest.approx(
        [16.0, math.hypot(6.0, 8.0), 12.0]
    )
    cleared = add_reference_distance(rows, None, 0.5)
    assert column(cleared, "distance_from_reference_um") == [None, None, None]
    assert rows[0]["distance_from_reference_um"] == 8.0, "the input rows are not modified"


def test_a_two_d_reference_is_not_applied_to_a_three_d_track():
    scale = Scale.from_values(0.5, 60.0, z_step_um=2.0)
    rows = frame_rows([track([(0, 0, 0), (6, 8, 1)])], scale, reference_point_px=(0.0, 0.0))
    assert column(rows, "distance_from_reference_um") == [None, None]
    rows = frame_rows([track([(0, 0, 0), (6, 8, 1)])], scale, reference_point_px=(0.0, 0.0, 0.0))
    assert rows[1]["distance_from_reference_um"] == pytest.approx(math.sqrt(9 + 16 + 4))


def test_turning_angles():
    right_angle = frame_rows([track([(0, 0), (10, 0), (10, 10)])], HOURLY)
    assert right_angle[2]["turning_angle_deg"] == pytest.approx(90.0)
    reversal = frame_rows([track([(0, 0), (10, 0), (0, 0)])], HOURLY)
    assert reversal[2]["turning_angle_deg"] == pytest.approx(180.0)
    # A step of zero length has no direction to compare.
    paused = frame_rows([track([(0, 0), (10, 0), (10, 0), (20, 0)])], HOURLY)
    assert paused[2]["turning_angle_deg"] is None
    assert paused[3]["turning_angle_deg"] is None


def test_uncalibrated_data_reports_pixels_only():
    bare = Scale.from_values(None, None)
    rows = frame_rows([track(DIRECTIVE)], bare)
    assert column(rows, "cumulative_path_px") == [0.0, 10.0, 20.0]
    assert column(rows, "distance_from_start_px") == [0.0, 10.0, 20.0]
    assert column(rows, "speed_px_per_frame") == [None, 10.0, 10.0]
    for key in (
        "x_um", "y_um", "area_um2", "cumulative_path_um", "distance_from_start_um",
        "distance_from_previous_um", "speed_um_per_min", "speed_um_per_hr",
        "acceleration_um_per_hr2", "elapsed_min", "elapsed_hr",
    ):
        assert column(rows, key) == [None] * 3, key
    # Direction needs no calibration.
    assert rows[2]["turning_angle_deg"] == pytest.approx(0.0)


def test_space_calibrated_but_untimed_data_has_distances_and_no_speeds():
    rows = frame_rows([track(DIRECTIVE)], Scale.from_values(0.5, None))
    assert column(rows, "cumulative_path_um") == [0.0, 5.0, 10.0]
    assert column(rows, "speed_um_per_hr") == [None, None, None]


def test_z_columns_are_filled_only_in_three_d():
    two_d = frame_rows([track(DIRECTIVE)], HOURLY)
    for key in ("z_slice", "z_um", "vz_um_per_min", "vz_um_per_hr", "volume_um3"):
        assert column(two_d, key) == [None] * 3, key

    scale = Scale.from_values(0.5, 60.0, z_step_um=2.0)
    three_d = frame_rows([track([(0, 0, 0), (6, 8, 1), (6, 8, 3)])], scale)
    assert column(three_d, "z_slice") == [0.0, 1.0, 3.0]
    assert column(three_d, "z_um") == [0.0, 2.0, 6.0]
    assert three_d[1]["distance_from_previous_um"] == pytest.approx(math.sqrt(9 + 16 + 4))
    assert three_d[2]["vz_um_per_hr"] == pytest.approx(4.0)
    assert three_d[2]["speed_um_per_hr"] == pytest.approx(4.0)
    # Isotropic pixels: z counts anisotropy (2.0 / 0.5 = 4) pixels per slice.
    assert three_d[2]["distance_from_previous_px"] == pytest.approx(8.0)


def test_three_d_without_a_z_step_reports_no_distance_at_all():
    """A slice is not a pixel: an in-plane distance must not pose as a 3-D one."""
    rows = frame_rows([track([(0, 0, 0), (6, 8, 1)])], HOURLY)
    assert rows[1]["z_slice"] == 1.0
    assert rows[1]["z_um"] is None
    for key in (
        "distance_from_previous_um", "distance_from_previous_px", "speed_um_per_hr",
        "speed_px_per_frame", "cumulative_path_um", "distance_from_start_px",
    ):
        assert rows[1][key] is None, key
    # The in-plane components are still facts.
    assert rows[1]["vx_px_per_frame"] == 6.0
    assert rows[1]["x_um"] == 3.0


def _detection(frame: int, **extra) -> Detection:
    return Detection(
        frame=frame, label=1, x=0.0, y=0.0, area_px=100.0, bbox=(0, 0, 10, 10),
        extent_px=10, eccentricity=0.9, orientation_rad=0.1, major_axis_px=20.0,
        minor_axis_px=6.0, solidity=0.95, touches_border=False,
        perimeter_px=50.0, circularity=0.5, aspect_ratio=20.0 / 6.0, **extra,
    )


def test_morphology_comes_from_the_detection_and_only_when_present():
    trk = track(DIRECTIVE)
    trk.observations[0].detection = _detection(0)
    rows = frame_rows([trk], HOURLY)

    assert rows[0]["perimeter_px"] == 50.0
    assert rows[0]["perimeter_um"] == 25.0
    assert rows[0]["major_axis_um"] == 10.0 and rows[0]["minor_axis_um"] == 3.0
    assert rows[0]["circularity"] == 0.5
    assert rows[0]["area_um2"] == pytest.approx(25.0)
    # No detection, no morphology: never a placeholder.
    assert rows[1]["eccentricity"] is None and rows[1]["perimeter_px"] is None

    bare = frame_rows([trk], Scale.from_values(None, 60.0))
    assert bare[0]["perimeter_px"] == 50.0 and bare[0]["perimeter_um"] is None


def test_volume_needs_a_calibrated_z_step():
    trk = track([(0, 0, 1), (6, 8, 1)])
    trk.observations[0].detection = _detection(0, z=1.0, volume_vox=40.0)
    with_z = frame_rows([trk], Scale.from_values(0.5, 60.0, z_step_um=2.0))
    assert with_z[0]["volume_vox"] == 40.0
    assert with_z[0]["volume_um3"] == pytest.approx(40.0 * 0.5 * 0.5 * 2.0)
    without_z = frame_rows([trk], HOURLY)
    assert without_z[0]["volume_vox"] == 40.0 and without_z[0]["volume_um3"] is None


def test_link_margin_provenance_and_flags():
    trk = track(DIRECTIVE, source="windowed", confidence=0.8)
    trk.observations[1].link_margin = 7.5
    trk.observations[1].cost = 1.25
    trk.flags = {"merge_suspected"}
    rows = frame_rows([trk], HOURLY, min_observations=5)
    assert column(rows, "link_margin_chi2") == [None, 7.5, None]
    assert rows[1]["match_cost_chi2"] == 1.25
    assert rows[0]["detection_source"] == "windowed"
    assert rows[0]["segmentation_confidence"] == 0.8
    assert rows[0]["track_flags"] == "fragment;merge_suspected"


def test_rows_carry_exactly_the_schema_columns():
    rows = frame_rows([track(DIRECTIVE)], HOURLY, reference_point_px=(0, 0))
    rows += frame_rows([track([(0, 0, 0), (1, 1, 1)])], Scale.from_values(0.5, 60, 2.0))
    for row in rows:
        assert set(row) == set(TRACK_COLUMNS)


def test_the_legacy_axis_argument_is_accepted_and_ignored():
    """pipeline.py and qc.py still pass an axis until the integration package."""
    new = frame_rows([track(DIRECTIVE)], HOURLY)
    legacy = frame_rows([track(DIRECTIVE)], object(), HOURLY)
    assert new == legacy
    with pytest.raises(TypeError):
        frame_rows([track(DIRECTIVE)], object())
