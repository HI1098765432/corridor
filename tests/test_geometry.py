"""Measuring the device as lanes -- and, for v1 runs only, reading its old axis.

The wall-ridge detector is v1's, moved to ``geometry.py``; what is new is
what it returns: lanes, each with its own fitted direction and a half-width
measured from its neighbours, and an ``applied`` flag that says whether they
gate tracking.  There is no migration axis.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import tifffile

from corridor.core.config import AXIS_VERTICAL, ConfinementConfig, GeometryConfig
from corridor.core.geometry import (
    GEOMETRY_FROM_RIDGES,
    GEOMETRY_FROM_V1,
    GEOMETRY_NONE,
    HALF_WIDTH_FROM_FIELD,
    HALF_WIDTH_FROM_SPACING,
    ChannelGeometry,
    Lane,
    assign_lanes,
    detect_channels,
    robust_line_fit,
    static_projection,
    trace_ridge,
)

from conftest import PIXEL_SIZE_UM, SAMPLE_DIR, make_detection, requires_samples


def synthetic_device(
    height: int = 320,
    width: int = 300,
    pitch: int = 80,
    tilt_deg: float | list[float] = 0.0,
    n_frames: int = 8,
    wall_width: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """A stack of straight bright channels at a known pitch; one tilt, or one per channel."""
    rng = np.random.default_rng(seed)
    centres = list(range(pitch // 2, width - pitch // 4, pitch))
    tilts = tilt_deg if isinstance(tilt_deg, list) else [tilt_deg] * len(centres)
    slopes = [math.tan(math.radians(t)) for t in tilts]
    frames = []
    for _ in range(n_frames):
        frame = rng.normal(1000, 25, size=(height, width)).astype(np.float32)
        for row in range(height):
            for centre, slope in zip(centres, slopes):
                x = int(round(centre + slope * row))
                lo, hi = max(0, x - wall_width), min(width, x + wall_width + 1)
                frame[row, lo:hi] += 2200.0
        frames.append(frame)
    return np.stack(frames).astype(np.uint16)


def _detect(stack, **kw):
    return detect_channels(stack, GeometryConfig(), pixel_size_um=PIXEL_SIZE_UM, **kw)


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def test_static_projection_removes_moving_objects():
    base = np.full((9, 40, 40), 100, dtype=np.uint16)
    for t in range(9):
        base[t, t * 4 : t * 4 + 3, 20] = 60000  # a bright object moving down
    assert static_projection(base).max() < 200, "a moving object should not survive the median"


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
    projection = static_projection(synthetic_device(tilt_deg=-2.5, width=200, pitch=90))
    angles = []
    for window in (9, 12, 16, 20):
        rows, cols = trace_ridge(projection, 45.0, half_window=window)
        slope, _, _ = robust_line_fit(rows, cols)
        angles.append(math.degrees(math.atan(slope)))
    assert max(angles) - min(angles) < 0.5, f"window-dependent result: {angles}"
    assert all(abs(a - (-2.5)) < 0.6 for a in angles), angles


# --------------------------------------------------------------------------
# Lanes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tilt", [0.0, -2.5, 1.8, -5.0])
def test_every_lane_recovers_a_known_tilt(tilt):
    geometry = _detect(synthetic_device(tilt_deg=tilt))
    assert geometry.source == GEOMETRY_FROM_RIDGES
    assert geometry.applied
    assert geometry.lanes
    for lane in geometry.lanes:
        assert lane.tilt_from_vertical_deg == pytest.approx(tilt, abs=0.7)


def test_lanes_keep_their_own_slopes():
    """Four channels at four tilts: no shared axis may average them away."""
    tilts = [-1.5, -2.5, -2.0, -3.0]
    geometry = _detect(synthetic_device(width=340, pitch=80, tilt_deg=tilts))
    assert geometry.n_lanes == 4
    measured = [lane.tilt_from_vertical_deg for lane in geometry.lanes]
    for got, want in zip(measured, tilts):
        assert got == pytest.approx(want, abs=0.4)
    assert max(measured) - min(measured) > 1.0


def test_every_lane_is_found_at_the_right_pitch():
    geometry = _detect(synthetic_device(width=340, pitch=80, tilt_deg=-2.0))
    assert geometry.is_multilane
    assert geometry.n_lanes == 4
    assert geometry.pitch_px == pytest.approx(80, abs=4)
    for lane in geometry.lanes:
        assert lane.half_width_source == HALF_WIDTH_FROM_SPACING
        assert lane.half_width_px == pytest.approx(40, abs=3)


def test_a_single_lane_field_says_its_half_width_is_not_measured():
    geometry = _detect(synthetic_device(width=90, pitch=90, tilt_deg=-2.0))
    assert geometry.n_lanes == 1
    assert not geometry.is_multilane
    assert geometry.lanes[0].half_width_source == HALF_WIDTH_FROM_FIELD
    assert any("not a measurement" in n for n in geometry.notes)


def test_horizontal_lanes_are_detected():
    stack = np.transpose(synthetic_device(tilt_deg=0.0), (0, 2, 1)).copy()
    geometry = _detect(stack)
    assert geometry.lanes
    for lane in geometry.lanes:
        assert abs(lane.direction[0]) > abs(lane.direction[1])


def test_a_featureless_image_has_no_lanes_and_says_so():
    rng = np.random.default_rng(1)
    stack = rng.normal(1000, 20, size=(5, 200, 200)).astype(np.uint16)
    geometry = _detect(stack)
    assert geometry.source == GEOMETRY_NONE
    assert geometry.lanes == []
    assert not geometry.applied
    assert geometry.notes
    assert geometry.lane_of(100, 100) == -1


def test_switching_wall_detection_off_gives_no_lanes():
    geometry = detect_channels(
        synthetic_device(), GeometryConfig(detect_walls=False), pixel_size_um=PIXEL_SIZE_UM
    )
    assert geometry.lanes == [] and not geometry.applied


def test_the_off_constraint_keeps_the_lanes_but_does_not_apply_them():
    geometry = _detect(synthetic_device(width=340, pitch=80), channel_constraint="off")
    assert geometry.n_lanes == 4
    assert not geometry.applied
    assert any("switched off" in n for n in geometry.notes)


def test_a_z_stack_is_collapsed_not_read_as_time():
    tyx = synthetic_device(width=340, pitch=80, tilt_deg=-2.0, n_frames=4)
    tzyx = np.repeat(tyx[:, None], 3, axis=1)
    assert _detect(tzyx).n_lanes == _detect(tyx).n_lanes
    zyx = tyx[:3]
    assert _detect(zyx, axes="ZYX").n_lanes == 4


# --------------------------------------------------------------------------
# Membership
# --------------------------------------------------------------------------


def _three_lanes(tilt_deg: float = 0.0) -> ChannelGeometry:
    t = math.radians(tilt_deg)
    direction = (math.sin(t), math.cos(t))
    return ChannelGeometry(
        lanes=[
            Lane(i, (x, 0.0), direction, 41.0, HALF_WIDTH_FROM_SPACING)
            for i, x in enumerate((45.0, 127.0, 209.0))
        ],
        source=GEOMETRY_FROM_RIDGES, applied=True, pitch_px=82.0,
    )


def test_detections_are_assigned_to_their_lane():
    detections = [
        make_detection(0, 44.0, 100.0),
        make_detection(0, 130.0, 100.0),
        make_detection(0, 205.0, 100.0),
        make_detection(0, 300.0, 100.0),  # beyond the last lane
    ]
    assign_lanes(detections, _three_lanes())
    assert [d.channel for d in detections] == [0, 1, 2, -1]


def test_irregular_spacing_leaves_no_hole_between_lanes():
    """Lanes at x = 0, 40 and 120: x = 65 is between lanes, so it belongs to the nearer.

    The middle lane's half-width is 20 (half its nearest spacing). Judged by
    that alone, x = 65 was in no lane (-1) and never gated -- a cell there
    could be linked across a wall. Outside the lattice the half-width still
    bounds the outer lanes.
    """
    lanes = [
        Lane(0, (0.0, 0.0), (0.0, 1.0), 20.0, HALF_WIDTH_FROM_SPACING),
        Lane(1, (40.0, 0.0), (0.0, 1.0), 20.0, HALF_WIDTH_FROM_SPACING),
        Lane(2, (120.0, 0.0), (0.0, 1.0), 40.0, HALF_WIDTH_FROM_SPACING),
    ]
    geometry = ChannelGeometry(lanes=lanes, source=GEOMETRY_FROM_RIDGES, applied=True)
    assert geometry.lane_of(65.0, 50.0) == 1
    assert geometry.lane_of(85.0, 50.0) == 2
    assert geometry.lane_of(-15.0, 50.0) == 0
    assert geometry.lane_of(-25.0, 50.0) == -1
    assert geometry.lane_of(155.0, 50.0) == 2
    assert geometry.lane_of(165.0, 50.0) == -1


def test_lane_membership_follows_each_lane_s_own_tilt():
    """A tilted lane's membership is judged along its tilt, not by x."""
    geometry = _three_lanes(-6.0)
    deep = make_detection(0, 45.0 + math.tan(math.radians(-6.0)) * 300.0, 300.0)
    assert geometry.lane_of(deep.x, deep.y) == 0


def test_geometry_round_trips_through_json():
    geometry = _detect(synthetic_device(width=340, pitch=80, tilt_deg=-2.0))
    again = ChannelGeometry.from_dict(json.loads(json.dumps(geometry.to_dict())))
    assert again.to_dict() == geometry.to_dict()
    assert again.lane_of(45.0, 100.0) == geometry.lane_of(45.0, 100.0)


# --------------------------------------------------------------------------
# v1 runs: drawn as lanes, never re-applied
# --------------------------------------------------------------------------

#: The ``confinement`` block of build/baseline_v1.3.0/052924_1/run.json (v1.3.0),
#: abridged to three of its six channels.
V1_CONFINEMENT = {
    "ux": -0.04360834050821684, "uy": 0.9990487038368646,
    "angle_deg": 92.49936645887219, "tilt_from_vertical_deg": -2.4993664588721862,
    "angle_sigma_deg": 0.5, "source": "channel_ridges", "confidence": 0.9918489767113621,
    "n_channels": 3, "pitch_px": 81.74684368491182,
    "channels": [
        {"index": 0, "origin_x": 57.18978905715025, "origin_y": 0.0,
         "half_width_px": 40.87342184245591, "detected": True},
        {"index": 1, "origin_x": 138.2102560342563, "origin_y": 0.0,
         "half_width_px": 40.87342184245591, "detected": True},
        {"index": 2, "origin_x": 219.95709971916813, "origin_y": 0.0,
         "half_width_px": 40.87342184245591, "detected": False},
    ],
    "notes": ["The channels run 2.5 degrees left of vertical."],
}


def test_a_v1_manifest_is_drawn_as_lanes():
    geometry = ChannelGeometry.from_legacy_manifest(V1_CONFINEMENT)
    assert geometry.source == GEOMETRY_FROM_V1
    assert not geometry.applied, "a v1 run's lines must never gate anything"
    assert geometry.n_lanes == 3
    assert [lane.detected for lane in geometry.lanes] == [True, True, False]
    for lane in geometry.lanes:
        assert lane.tilt_from_vertical_deg == pytest.approx(-2.4994, abs=1e-3)
    assert geometry.lanes[1].origin == pytest.approx((138.2102560342563, 0.0))
    assert any("v1" in n for n in geometry.notes)


def test_an_empty_v1_manifest_reads_as_no_lanes():
    geometry = ChannelGeometry.from_legacy_manifest({})
    assert geometry.lanes == [] and not geometry.applied


@requires_samples
def test_the_baseline_manifests_read_as_lanes():
    from conftest import BASELINE_DIR as baseline

    if not baseline.is_dir():
        pytest.skip("the frozen v1.3.0 baseline is not present (set CORRIDOR_BASELINE_DIR)")
    for run in sorted(baseline.iterdir()):
        manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
        geometry = ChannelGeometry.from_legacy_manifest(manifest["confinement"])
        assert geometry.n_lanes == manifest["confinement"]["n_channels"]


# --------------------------------------------------------------------------
# The deprecated v1 reader still works for its legacy callers
# --------------------------------------------------------------------------


def test_legacy_resolve_axis_still_recovers_a_tilt():
    from corridor.core.confinement import AXIS_FROM_RIDGES, resolve_axis

    axis = resolve_axis(synthetic_device(tilt_deg=-2.5), ConfinementConfig(), pixel_size_um=PIXEL_SIZE_UM)
    assert axis.source == AXIS_FROM_RIDGES
    assert axis.tilt_from_vertical_deg == pytest.approx(-2.5, abs=0.7)


def test_legacy_configured_axis_still_overrides_and_says_so():
    from corridor.core.confinement import AXIS_FROM_CONFIG, resolve_axis

    axis = resolve_axis(
        synthetic_device(tilt_deg=-2.5), ConfinementConfig(mode=AXIS_VERTICAL),
        pixel_size_um=PIXEL_SIZE_UM,
    )
    assert axis.source == AXIS_FROM_CONFIG
    assert (axis.ux, axis.uy) == (0.0, 1.0)
    assert any("tilted" in note for note in axis.notes)


def test_legacy_channel_assignment_still_works():
    from corridor.core.confinement import Channel, ConfinementAxis, assign_channels

    axis = ConfinementAxis(
        ux=0.0, uy=1.0, source="configured", confidence=1.0,
        channels=[Channel(0, (45.0, 0.0), 40.0), Channel(1, (127.0, 0.0), 40.0)],
    )
    assert float(axis.unit @ axis.normal) == pytest.approx(0.0, abs=1e-12)
    detections = [make_detection(0, 44.0, 100.0), make_detection(0, 130.0, 100.0)]
    assign_channels(detections, axis)
    assert [d.channel for d in detections] == [0, 1]


def test_the_legacy_axis_becomes_lanes_for_the_transition():
    from corridor.core.confinement import Channel, ConfinementAxis

    axis = ConfinementAxis(
        ux=0.0, uy=1.0, source="channel_ridges", confidence=1.0,
        channels=[Channel(0, (45.0, 0.0), 41.0), Channel(1, (127.0, 0.0), 41.0)],
    )
    geometry = ChannelGeometry.from_legacy_axis(axis)
    assert geometry.applied and geometry.n_lanes == 2
    assert geometry.lane_of(130.0, 50.0) == 1


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
@pytest.mark.parametrize("name,n_lanes", SUPPLIED)
def test_supplied_device_geometry(name, n_lanes):
    geometry = _detect(tifffile.imread(SAMPLE_DIR / name))
    assert geometry.source == GEOMETRY_FROM_RIDGES
    assert geometry.applied
    assert geometry.n_lanes == n_lanes
    # One rigid device, but each lane is fitted on its own: measured, every
    # lane of every field tilts 1.3-4.3 degrees left of vertical.
    for lane in geometry.lanes:
        assert -5.0 < lane.tilt_from_vertical_deg < -1.0, lane.tilt_from_vertical_deg


@requires_samples
def test_supplied_wide_fields_share_one_pitch():
    pitches = [_detect(tifffile.imread(SAMPLE_DIR / n)).pitch_px for n in ("052924_1.tif", "052924_2.tif")]
    assert all(78 < p < 90 for p in pitches), pitches
    assert abs(pitches[0] - pitches[1]) < 8
