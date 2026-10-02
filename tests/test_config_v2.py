"""Configuration v2 (contract §3): new blocks, legacy upgrade, no inheritance.

The rule for the transition is additive: every v1 field and every saved v1
project must still load, while the v2 fields exist beside them.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from corridor.core.config import (
    CHANNEL_CONSTRAINT_AUTO,
    CHANNEL_CONSTRAINT_OFF,
    CalibrationConfig,
    GeometryConfig,
    ImportConfig,
    MeasurementConfig,
    RunConfig,
    Scale,
    TrackingConfig,
)
from corridor.core.recovery import RecoveryConfig

from conftest import FRAME_INTERVAL_MIN, PIXEL_SIZE_UM


def json_round_trip(config: RunConfig) -> RunConfig:
    return RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))


# --------------------------------------------------------------------------
# Scale
# --------------------------------------------------------------------------


def test_z_is_never_assumed_equal_to_xy():
    flat = Scale.from_values(PIXEL_SIZE_UM, FRAME_INTERVAL_MIN)
    assert flat.z_step_um is None
    assert flat.calibrated_z is False
    assert flat.anisotropy is None
    assert flat.spacing_zyx_um is None


def test_anisotropy_is_z_step_over_pixel_size():
    s = Scale.from_values(0.5, 10.0, 2.0)
    assert s.calibrated_z
    assert s.anisotropy == pytest.approx(4.0)
    assert s.spacing_zyx_um == (2.0, 0.5, 0.5)


def test_anisotropy_needs_a_calibrated_pixel_size_too():
    """A Z step in um over the 1.0 placeholder pixel size is not a ratio."""
    s = Scale.from_values(None, 10.0, 2.0)
    assert s.calibrated_z and not s.calibrated_space
    assert s.anisotropy is None
    assert s.spacing_zyx_um is None


@pytest.mark.parametrize("bad", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_an_invalid_z_step_is_unknown_not_one(bad):
    s = Scale.from_values(0.5, 10.0, bad)
    assert s.z_step_um is None
    assert s.calibrated_z is False


def test_frames_to_hr_derives_from_minutes():
    s = Scale.from_values(PIXEL_SIZE_UM, FRAME_INTERVAL_MIN)
    assert s.frames_to_hr(3) == pytest.approx(3 * FRAME_INTERVAL_MIN / 60.0)
    assert s.frames_to_hr(3) * 60.0 == pytest.approx(s.frames_to_min(3))


def test_scale_keeps_its_v1_positional_constructor():
    s = Scale(0.5, 10.0, True, True)
    assert s.z_step_um is None and s.anisotropy is None


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------


def test_gate_is_derived_from_the_unmatched_cost():
    cfg = TrackingConfig(unmatched_chi2=25.0)
    assert cfg.effective_gate_chi2 == 50.0
    cfg.unmatched_chi2 = 15.0
    assert cfg.effective_gate_chi2 == 30.0  # moves down too: no ratchet


def test_legacy_gate_field_still_loads_and_is_not_the_v2_gate():
    cfg = RunConfig.from_dict({"tracking": {"unmatched_chi2": 15.0, "gate_chi2": 50.0}}).tracking
    assert cfg.gate_chi2 == 50.0
    assert cfg.effective_gate_chi2 == 30.0


def test_initial_speed_sigma_defaults_to_a_third_of_the_speed_gate():
    cfg = TrackingConfig()
    assert cfg.initial_speed_sigma_um_per_min is None
    assert cfg.effective_initial_speed_sigma_um_per_min == pytest.approx(5.0 / 3.0)
    cfg.initial_speed_sigma_um_per_min = 0.9
    assert cfg.effective_initial_speed_sigma_um_per_min == 0.9


def test_every_v2_tracking_field_exists():
    names = {
        "max_speed_um_per_min", "area_ratio_min", "area_ratio_max",
        "position_sigma_um", "shape_position_fraction", "width_position_fraction",
        "velocity_sigma_um_per_min", "initial_speed_sigma_um_per_min", "sigma_ln_area",
        "w_shape", "w_orientation", "orientation_min_eccentricity", "w_reversal",
        "direction_noise_floor_um", "w_overlap", "gap_penalty_chi2", "unmatched_chi2",
        "max_gap", "min_observations", "global_gap_closing", "channel_constraint",
    }
    assert names <= set(TrackingConfig().__dataclass_fields__)
    assert TrackingConfig().channel_constraint == CHANNEL_CONSTRAINT_AUTO


def test_a_saved_channel_identity_opt_out_becomes_constraint_off():
    cfg = RunConfig.from_dict({"tracking": {"enforce_channel_identity": False}}).tracking
    assert cfg.channel_constraint == CHANNEL_CONSTRAINT_OFF
    explicit = RunConfig.from_dict(
        {"tracking": {"enforce_channel_identity": False, "channel_constraint": "auto"}}
    ).tracking
    assert explicit.channel_constraint == CHANNEL_CONSTRAINT_AUTO


# --------------------------------------------------------------------------
# Whole-run (de)serialisation
# --------------------------------------------------------------------------


def test_new_blocks_round_trip_through_json():
    config = RunConfig()
    config.geometry.detect_walls = False
    config.measurement.reference_point_px = (12.5, 40.0)
    config.measurement.msd_min_pairs = 5
    config.import_.axes = "TZYX"
    config.import_.channel_index = 1
    config.import_.labels_path = "labels.tif"
    config.calibration.z_step_um = 2.0

    data = config.to_dict()
    assert "import" in data and "import_" not in data
    restored = json_round_trip(config)

    assert restored.geometry.detect_walls is False
    assert restored.measurement.reference_point_px == (12.5, 40.0)
    assert isinstance(restored.measurement.reference_point_px, tuple)
    assert restored.measurement.msd_min_pairs == 5
    assert restored.import_ == ImportConfig("TZYX", 1, "labels.tif")
    assert restored.calibration.z_step_um == 2.0
    assert restored == config


def test_nested_types_are_restored_not_left_as_json_shapes():
    """The v1 _from_dict never reached its dataclass branch (string
    annotations), so tuples came back as lists and only a hand-kept list of
    sections was rebuilt. Every nested type now comes from the annotation."""
    config = RunConfig()
    config.segmentation.normalize_percentiles = (3.0, 97.0)
    config.segmentation.ensemble_model_paths = ("a", "b")
    restored = json_round_trip(config)
    assert isinstance(restored.recovery, RecoveryConfig)
    assert isinstance(restored.geometry, GeometryConfig)
    assert isinstance(restored.measurement, MeasurementConfig)
    assert restored.segmentation.normalize_percentiles == (3.0, 97.0)
    assert isinstance(restored.segmentation.normalize_percentiles, tuple)
    assert restored.segmentation.ensemble_model_paths == ("a", "b")
    assert isinstance(restored.segmentation.channels, tuple)
    assert restored == config


def test_import_key_is_accepted_under_either_name():
    assert RunConfig.from_dict({"import": {"axes": "ZYX"}}).import_.axes == "ZYX"
    assert RunConfig.from_dict({"import_": {"axes": "TYX"}}).import_.axes == "TYX"


def test_legacy_confinement_block_maps_into_geometry():
    legacy = {
        "confinement": {
            "mode": "angle",
            "angle_deg": 87.0,
            "detect_walls": False,
            "multichannel_warn_ratio": 2.0,
            "min_channel_pitch_um": 20.0,
            "min_channel_pitch_px": 41.0,
        }
    }
    config = RunConfig.from_dict(legacy)
    assert config.geometry == GeometryConfig(
        detect_walls=False, min_channel_pitch_um=20.0, min_channel_pitch_px=41.0
    )
    assert not hasattr(config.geometry, "mode")
    # The legacy block itself still loads for the readers that use it.
    assert config.confinement.mode == "angle"
    assert config.confinement.angle_deg == 87.0


def test_a_geometry_only_dict_keeps_the_legacy_reader_in_step():
    config = RunConfig.from_dict({"geometry": {"detect_walls": False}})
    assert config.geometry.detect_walls is False
    assert config.confinement.detect_walls is False


def test_geometry_wins_when_both_blocks_are_present():
    config = RunConfig.from_dict(
        {"confinement": {"detect_walls": False}, "geometry": {"detect_walls": True}}
    )
    assert config.geometry.detect_walls is True
    assert config.confinement.detect_walls is False


def test_from_dict_never_mutates_its_input():
    data = {
        "confinement": {"detect_walls": False},
        "tracking": {"enforce_channel_identity": False},
        "import": {"axes": "TYX"},
    }
    before = copy.deepcopy(data)
    RunConfig.from_dict(data)
    assert data == before


def test_a_v1_saved_project_still_loads():
    """A config_json as 1.3.0 stored it: no geometry/measurement/import, and
    keys this version no longer has."""
    v1 = RunConfig().to_dict()
    for key in ("geometry", "measurement", "import"):
        v1.pop(key)
    v1["tracking"]["a_retired_setting"] = 3
    v1["retired_section"] = {"x": 1}
    config = RunConfig.from_dict(json.loads(json.dumps(v1)))
    assert config.geometry == GeometryConfig()
    assert config.measurement == MeasurementConfig()
    assert config.import_ == ImportConfig()


# --------------------------------------------------------------------------
# New projects (critique C6)
# --------------------------------------------------------------------------


def saved_default_config() -> dict:
    """What 1.3.0 stores as default_config after a KK1 run with overrides."""
    config = RunConfig()
    config.input_path = "D:/kk1/movie.tif"
    config.output_dir = "C:/Users/x/Corridor/projects/7"
    config.calibration = CalibrationConfig(
        pixel_size_um=0.639, frame_interval_min=15.0, z_step_um=2.0
    )
    config.segmentation.model_path = "D:/somewhere/other_model"
    config.segmentation.builtin_model = "cpsam"
    config.segmentation.use_custom_model = False
    config.segmentation.ensemble_model_paths = ("D:/kk1_model",)
    config.segmentation.cellprob_threshold = -1.0
    config.tracking.max_gap = 5
    config.geometry.detect_walls = False
    config.import_ = ImportConfig(axes="TZYX", channel_index=2, labels_path="D:/labels.tif")
    config.measurement.reference_point_px = (10.0, 20.0)
    config.measurement.msd_min_pairs = 6
    return json.loads(json.dumps(config.to_dict()))


def test_a_new_project_inherits_no_calibration_paths_or_model():
    config = RunConfig.for_new_project(saved_default_config())
    assert config.input_path == "" and config.output_dir == ""
    assert config.calibration == CalibrationConfig()
    seg = config.segmentation
    assert seg.model_path is None
    assert seg.builtin_model == "cyto3"  # the dataclass default, never "cpsam"
    assert seg.use_custom_model is True
    assert seg.ensemble_model_paths == ()
    assert config.import_ == ImportConfig()
    assert config.measurement.reference_point_px is None


def test_a_new_project_keeps_the_tuning():
    config = RunConfig.for_new_project(saved_default_config())
    assert config.segmentation.cellprob_threshold == -1.0
    assert config.tracking.max_gap == 5
    assert config.geometry.detect_walls is False
    assert config.measurement.msd_min_pairs == 6


def test_a_new_project_from_nothing_is_the_default():
    assert RunConfig.for_new_project(None) == RunConfig()
    assert RunConfig.for_new_project({}) == RunConfig()


def test_for_new_project_does_not_touch_the_saved_dict():
    saved = saved_default_config()
    before = copy.deepcopy(saved)
    RunConfig.for_new_project(saved)
    assert saved == before


# --------------------------------------------------------------------------
# The seams (interfaces.py) import without any stage and type-check
# structurally.
# --------------------------------------------------------------------------


def test_interfaces_are_structural_and_intervals_are_checked():
    from corridor.core.interfaces import Prediction, Predictor, Segmenter, Tracker

    class Constant:
        target = "net_speed_um_per_hr"
        feature_names = ("area_um2",)

        def predict(self, features: np.ndarray):
            return [Prediction(1.0, 0.5, 1.5, 0.9) for _ in range(len(features))]

    assert isinstance(Constant(), Predictor)
    assert not isinstance(Constant(), Segmenter)
    assert not isinstance(Constant(), Tracker)
    assert len(Constant().predict(np.zeros((3, 1)))) == 3
    with pytest.raises(ValueError):
        Prediction(1.0, 2.0, 0.5, 0.9)
    with pytest.raises(ValueError):
        Prediction(1.0, 0.5, 1.5, 1.0)
