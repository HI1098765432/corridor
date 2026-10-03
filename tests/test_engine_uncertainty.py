"""Bot 6c -- the precision floor (``docs/ENGINE_4D.md`` acceptance).

The engine recovers sub-pixel motion, so it must also refuse to call noise
motion.  These tests pin that a displacement below its floor is flagged
below-floor, that the floor grows the way the physics says it should (worse
with a smaller object, more noise or a larger registration error), and that the
centroid-uncertainty formula is exactly the contract's.
"""

from __future__ import annotations

import math

import pytest

from corridor.engine.uncertainty import (
    DEFAULT_BOUNDARY_SIGMA_PX,
    PrecisionFloor,
    centroid_sigma_px,
    displacement_floor,
    volume_rate_floor,
)


# --------------------------------------------------------------------------
# A sub-floor move is flagged; a supra-floor move is resolved.
# --------------------------------------------------------------------------


def test_sub_floor_move_flagged_and_supra_floor_resolved():
    """An 800-px cell: a 0.05 px twitch is below the floor, a 1 px step is not."""
    floor = displacement_floor(
        area_px=800.0, noise_to_signal=0.2, registration_sigma_px=0.1, pixel_size_um=0.5
    )
    # the floor itself is sub-pixel: a real step clears it, a jitter does not
    assert floor.floor_px < 0.3

    twitch = floor.assess(0.05)
    assert twitch.below_floor is True
    assert twitch.ratio < 1.0
    assert twitch.to_dict()["verdict"] == "below_floor"

    step = floor.assess(1.0)
    assert step.below_floor is False
    assert step.ratio > 1.0
    assert step.to_dict()["verdict"] == "resolved"
    assert step.magnitude_um == pytest.approx(0.5)  # 1 px * 0.5 um/px


def test_significance_factor_raises_the_bar():
    """A stricter k turns a marginal move from resolved into below-floor."""
    floor = displacement_floor(area_px=400.0, registration_sigma_px=0.1, pixel_size_um=0.5)
    magnitude = floor.floor_px * 1.5
    assert floor.assess(magnitude, significance_k=1.0).below_floor is False
    assert floor.assess(magnitude, significance_k=2.0).below_floor is True


# --------------------------------------------------------------------------
# The centroid-uncertainty formula is the contract's: 0.5 / sqrt(area).
# --------------------------------------------------------------------------


def test_centroid_sigma_matches_contract_formula():
    for area in (100.0, 400.0, 800.0, 2000.0):
        assert centroid_sigma_px(area) == pytest.approx(0.5 / math.sqrt(area))
    # a bigger cell has a better-defined centroid
    assert centroid_sigma_px(2000.0) < centroid_sigma_px(100.0)


def test_zero_area_is_infinite_floor():
    """A vanished object has no measurable centroid -- its floor is infinite."""
    floor = displacement_floor(area_px=0.0)
    assert math.isinf(floor.centroid_sigma_px)
    assert math.isinf(floor.floor_px)
    assert floor.assess(1e9).below_floor is True  # nothing clears an infinite floor


# --------------------------------------------------------------------------
# The floor grows with noise, registration error, and a smaller object.
# --------------------------------------------------------------------------


def test_floor_grows_with_noise():
    quiet = displacement_floor(area_px=800.0, noise_to_signal=0.0)
    noisy = displacement_floor(area_px=800.0, noise_to_signal=1.0)
    assert noisy.floor_px > quiet.floor_px
    # noise enters as (1 + NSR) on the edge jitter
    assert noisy.edge_sigma_px == pytest.approx(2.0 * DEFAULT_BOUNDARY_SIGMA_PX)


def test_floor_grows_with_registration_error():
    clean = displacement_floor(area_px=800.0, registration_sigma_px=0.0)
    drifty = displacement_floor(area_px=800.0, registration_sigma_px=0.5)
    assert drifty.floor_px > clean.floor_px


def test_floor_grows_as_object_shrinks():
    big = displacement_floor(area_px=2000.0)
    small = displacement_floor(area_px=100.0)
    assert small.floor_px > big.floor_px


def test_displacement_is_sqrt2_over_a_single_position():
    """A displacement is a difference of two positions, so its floor is sqrt(2) larger."""
    floor = displacement_floor(area_px=800.0, registration_sigma_px=0.2)
    assert floor.floor_px == pytest.approx(math.sqrt(2.0) * floor.position_sigma_px)


# --------------------------------------------------------------------------
# Velocity and volume-rate floors.
# --------------------------------------------------------------------------


def test_velocity_floor_divides_by_elapsed_time():
    floor = displacement_floor(area_px=800.0, registration_sigma_px=0.1, pixel_size_um=0.5)
    assert floor.velocity_floor_um_per_min(10.0) == pytest.approx(floor.floor_um / 10.0)
    assert floor.velocity_floor_um_per_min(0.0) is None


def test_velocity_floor_none_without_calibration():
    floor = displacement_floor(area_px=800.0)  # no pixel size
    assert floor.floor_um is None
    assert floor.velocity_floor_um_per_min(10.0) is None


def test_volume_rate_floor_scales_with_surface_and_time():
    small_surface = volume_rate_floor(surface_px=200.0, dt_frames=1.0)
    big_surface = volume_rate_floor(surface_px=800.0, dt_frames=1.0)
    assert big_surface["floor_vox_per_frame"] > small_surface["floor_vox_per_frame"]
    # a longer baseline lowers the rate floor
    slow = volume_rate_floor(surface_px=800.0, dt_frames=4.0)
    assert slow["floor_vox_per_frame"] < big_surface["floor_vox_per_frame"]
    # calibrated um^3/min floor is filled only with a voxel size and interval
    assert small_surface["floor_um3_per_min"] is None
    calibrated = volume_rate_floor(
        surface_px=800.0, dt_frames=1.0, voxel_um3=0.125, frame_interval_min=10.0
    )
    assert calibrated["floor_um3_per_min"] is not None and calibrated["floor_um3_per_min"] > 0


# --------------------------------------------------------------------------
# Negative inputs are rejected; evidence round-trips.
# --------------------------------------------------------------------------


def test_negative_inputs_rejected():
    with pytest.raises(ValueError):
        displacement_floor(area_px=800.0, noise_to_signal=-0.1)
    with pytest.raises(ValueError):
        displacement_floor(area_px=800.0, registration_sigma_px=-0.1)


def test_evidence_is_inspectable():
    floor = displacement_floor(
        area_px=800.0, noise_to_signal=0.3, registration_sigma_px=0.2, pixel_size_um=0.5
    )
    d = floor.to_dict()
    assert set(d) >= {
        "area_px", "edge_sigma_px", "centroid_sigma_px", "position_sigma_px",
        "floor_px", "floor_um",
    }
    assert isinstance(floor, PrecisionFloor)
    verdict = floor.assess(0.3).to_dict()
    assert set(verdict) >= {"magnitude_px", "floor_px", "ratio", "below_floor", "verdict"}
