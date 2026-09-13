"""Measuring the device: migration axis and channel layout."""

from __future__ import annotations

import math

import numpy as np
import pytest
import tifffile

from corridor.core.config import (
    AXIS_ANGLE,
    AXIS_AUTO,
    AXIS_HORIZONTAL,
    AXIS_VERTICAL,
    ConfinementConfig,
)
from corridor.core.confinement import (
    AXIS_FROM_CONFIG,
    AXIS_FROM_RIDGES,
    ConfinementAxis,
    Channel,
    assign_channels,
    resolve_axis,
    robust_line_fit,
    static_projection,
    trace_ridge,
)

from conftest import PIXEL_SIZE_UM, SAMPLE_DIR, make_detection, requires_samples


def synthetic_device(
    height: int = 320,
    width: int = 300,
    pitch: int = 80,
    tilt_deg: float = 0.0,
    n_frames: int = 8,
    wall_width: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """A stack of straight bright channels at a known pitch and tilt."""
    rng = np.random.default_rng(seed)
    slope = math.tan(math.radians(tilt_deg))
    frames = []
    centres = list(range(pitch // 2, width - pitch // 4, pitch))
    for _ in range(n_frames):
        frame = rng.normal(1000, 25, size=(height, width)).astype(np.float32)
        for row in range(height):
            for centre in centres:
                x = int(round(centre + slope * row))
                lo, hi = max(0, x - wall_width), min(width, x + wall_width + 1)
                frame[row, lo:hi] += 2200.0
        frames.append(frame)
    return np.stack(frames).astype(np.uint16)


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def test_static_projection_removes_moving_objects():
    base = np.full((9, 40, 40), 100, dtype=np.uint16)
    for t in range(9):
        base[t, t * 4 : t * 4 + 3, 20] = 60000  # a bright object moving down
    projection = static_projection(base)
    assert projection.max() < 200, "a moving object should not survive the median"


def test_robust_line_fit_ignores_outliers():
    t = np.arange(60, dtype=float)
    v = 10.0 + 0.05 * t
    v[[5, 17, 33]] += 40.0  # gross outliers
    slope, intercept, inliers = robust_line_fit(t, v)
    assert slope == pytest.approx(0.05, abs=0.01)
    assert intercept == pytest.approx(10.0, abs=0.5)
    assert inliers >= 55


def test_ridge_trace_is_insensitive_to_the_window_size():
    """The trace must measure the ridge, not the box it is searched in."""
    stack = synthetic_device(tilt_deg=-2.5, width=200, pitch=90)
    projection = static_projection(stack)
    angles = []
    for window in (9, 12, 16, 20):
        rows, cols = trace_ridge(projection, 45.0, half_window=window)
        slope, _, _ = robust_line_fit(rows, cols)
        angles.append(math.degrees(math.atan(slope)))
    assert max(angles) - min(angles) < 0.5, f"window-dependent result: {angles}"
    assert all(abs(a - (-2.5)) < 0.6 for a in angles), angles


# --------------------------------------------------------------------------
# Axis resolution
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tilt", [0.0, -2.5, 1.8, -5.0])
def test_axis_recovers_a_known_tilt(tilt):
    stack = synthetic_device(tilt_deg=tilt)
    axis = resolve_axis(stack, ConfinementConfig(), pixel_size_um=PIXEL_SIZE_UM)
    assert axis.source == AXIS_FROM_RIDGES
    assert axis.tilt_from_vertical_deg == pytest.approx(tilt, abs=0.7)


def test_axis_finds_every_channel_at_the_right_pitch():
    stack = synthetic_device(width=340, pitch=80, tilt_deg=-2.0)
    axis = resolve_axis(stack, ConfinementConfig(), pixel_size_um=PIXEL_SIZE_UM)
    assert axis.is_multichannel
    assert len(axis.channels) == 4
    assert axis.pitch_px == pytest.approx(80, abs=4)


def test_a_single_channel_field_is_not_multichannel():
    stack = synthetic_device(width=90, pitch=200, tilt_deg=-2.0)
    axis = resolve_axis(stack, ConfinementConfig(), pixel_size_um=PIXEL_SIZE_UM)
    assert len(axis.channels) == 1
    assert not axis.is_multichannel


def test_horizontal_channels_are_detected():
    stack = np.transpose(synthetic_device(tilt_deg=0.0), (0, 2, 1)).copy()
    axis = resolve_axis(stack, ConfinementConfig(), pixel_size_um=PIXEL_SIZE_UM)
    assert abs(axis.ux) > abs(axis.uy), "channels running across should give a near-x axis"


def test_configured_axis_overrides_detection():
    stack = synthetic_device(tilt_deg=-2.5)
    config = ConfinementConfig(mode=AXIS_VERTICAL)
    axis = resolve_axis(stack, config, pixel_size_um=PIXEL_SIZE_UM)
    assert axis.source == AXIS_FROM_CONFIG
    assert (axis.ux, axis.uy) == (0.0, 1.0)
    assert any("tilted" in note for note in axis.notes), (
        "a manual axis that disagrees with the device should say so"
    )


def test_explicit_angle_is_used():
    stack = synthetic_device(tilt_deg=0.0)
    config = ConfinementConfig(mode=AXIS_ANGLE, angle_deg=80.0)
    axis = resolve_axis(stack, config, pixel_size_um=PIXEL_SIZE_UM)
    assert axis.angle_deg == pytest.approx(80.0, abs=0.01)


def test_featureless_image_falls_back_and_says_so():
    rng = np.random.default_rng(1)
    stack = rng.normal(1000, 20, size=(5, 200, 200)).astype(np.uint16)
    axis = resolve_axis(stack, ConfinementConfig(), pixel_size_um=PIXEL_SIZE_UM)
    assert axis.confidence < 0.7
    assert axis.notes, "a guessed axis must be reported as a guess"


def test_uncertain_axis_carries_a_wider_angle_sigma():
    rng = np.random.default_rng(2)
    blank = rng.normal(1000, 20, size=(5, 200, 200)).astype(np.uint16)
    weak = resolve_axis(blank, ConfinementConfig(), pixel_size_um=PIXEL_SIZE_UM)
    strong = resolve_axis(
        synthetic_device(tilt_deg=-2.0), ConfinementConfig(), pixel_size_um=PIXEL_SIZE_UM
    )
    assert weak.angle_sigma_rad > strong.angle_sigma_rad


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_normal_is_perpendicular_to_the_axis():
    axis = ConfinementAxis(ux=0.3, uy=0.954, source="configured", confidence=1.0)
    assert float(axis.unit @ axis.normal) == pytest.approx(0.0, abs=1e-12)


def test_projection_splits_a_vector_correctly():
    axis = ConfinementAxis(ux=0.0, uy=1.0, source="configured", confidence=1.0)
    along, across = axis.project(np.array([3.0, 4.0]))
    assert along == pytest.approx(4.0)
    assert abs(across) == pytest.approx(3.0)


def test_detections_are_assigned_to_the_nearest_channel():
    axis = ConfinementAxis(
        ux=0.0, uy=1.0, source="configured", confidence=1.0,
        channels=[
            Channel(0, (45.0, 0.0), 40.0),
            Channel(1, (127.0, 0.0), 40.0),
            Channel(2, (209.0, 0.0), 40.0),
        ],
    )
    detections = [
        make_detection(0, 44.0, 100.0),
        make_detection(0, 130.0, 100.0),
        make_detection(0, 205.0, 100.0),
    ]
    assign_channels(detections, axis)
    assert [d.channel for d in detections] == [0, 1, 2]


def test_channel_assignment_follows_the_tilt():
    """A tilted channel's membership must be judged along the tilt, not by x."""
    tilt = math.radians(-6.0)
    axis = ConfinementAxis(
        ux=math.sin(tilt), uy=math.cos(tilt), source="configured", confidence=1.0,
        channels=[Channel(0, (100.0, 0.0), 40.0), Channel(1, (180.0, 0.0), 40.0)],
    )
    # 300 rows down a -6 degree channel starting at x=100 lands near x=68.
    deep = make_detection(0, 100.0 + math.tan(tilt) * 300.0, 300.0)
    assign_channels([deep], axis)
    assert deep.channel == 0


# --------------------------------------------------------------------------
# The supplied device
# --------------------------------------------------------------------------

SUPPLIED = [
    ("052924_t1.tif", 1),
    ("052924_t2_empty.tif", 1),
    ("052924_t3_dual.tif", 1),
    ("052924_1.tif", 6),
    ("052924_2.tif", 6),
]


@requires_samples
@pytest.mark.parametrize("name,n_channels", SUPPLIED)
def test_supplied_device_geometry(name, n_channels):
    stack = tifffile.imread(SAMPLE_DIR / name)
    axis = resolve_axis(stack, ConfinementConfig(), pixel_size_um=PIXEL_SIZE_UM)
    assert axis.source == AXIS_FROM_RIDGES
    assert len(axis.channels) == n_channels
    # One rigid device: every crop and field must agree on the tilt.
    assert -4.0 < axis.tilt_from_vertical_deg < -1.0, axis.tilt_from_vertical_deg


@requires_samples
def test_supplied_wide_fields_share_one_pitch():
    pitches = []
    for name in ("052924_1.tif", "052924_2.tif"):
        stack = tifffile.imread(SAMPLE_DIR / name)
        axis = resolve_axis(stack, ConfinementConfig(), pixel_size_um=PIXEL_SIZE_UM)
        pitches.append(axis.pitch_px)
    assert all(78 < p < 90 for p in pitches), pitches
    assert abs(pitches[0] - pitches[1]) < 8
