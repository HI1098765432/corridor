"""Configuration v2 (contract §3): new blocks, legacy upgrade, no inheritance.

The rule for the transition is additive: every v1 field and every saved v1
project must still load, while the v2 fields exist beside them.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from corridor.core.config import (
    CHANNEL_CONSTRAINT_AUTO,
    CHANNEL_CONSTRAINT_OFF,
    TRACKING_V2_ONLY_FIELDS,
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


def test_an_opt_out_saved_beside_a_default_constraint_is_kept():
    """What a build that wrote both fields stored after the 1.x panel's
    unchecked box: the legacy False beside an "auto" nobody chose."""
    data = {"tracking": {"enforce_channel_identity": False, "channel_constraint": "auto"}}
    cfg = RunConfig.from_dict(data).tracking
    assert cfg.channel_constraint == CHANNEL_CONSTRAINT_OFF
    assert cfg.enforce_channel_identity is False
    v2_opt_out = RunConfig.from_dict({"tracking": {"channel_constraint": "off"}}).tracking
    assert v2_opt_out.enforce_channel_identity is False


def test_the_v1_panel_opt_out_survives_a_save():
    """The real UI path: advanced.apply_to writes only the legacy field."""
    config = RunConfig()
    config.tracking.enforce_channel_identity = False
    assert config.tracking.channel_constraint == CHANNEL_CONSTRAINT_OFF
    restored = json_round_trip(config).tracking
    assert restored.enforce_channel_identity is False
    assert restored.channel_constraint == CHANNEL_CONSTRAINT_OFF


def test_a_re_enable_after_an_opt_out_survives_a_save():
    """Whichever field the writer knows, turning the gate back on sticks."""
    for field_name, value in (("enforce_channel_identity", True), ("channel_constraint", "auto")):
        config = RunConfig.from_dict({"tracking": {"enforce_channel_identity": False}})
        setattr(config.tracking, field_name, value)
        restored = json_round_trip(config).tracking
        assert restored.enforce_channel_identity is True, field_name
        assert restored.channel_constraint == CHANNEL_CONSTRAINT_AUTO, field_name


def test_the_channel_sync_stores_nothing_beside_the_fields():
    """run.json dumps tracking.__dict__; a helper attribute would land in it."""
    cfg = TrackingConfig(enforce_channel_identity=False)
    cfg.channel_constraint = "auto"
    assert set(cfg.__dict__) == {f.name for f in fields(TrackingConfig)}


def test_v2_only_fields_are_fields_and_no_v1_code_reads_them():
    """If this fails, a stage now reads the named field: take it out of
    TRACKING_V2_ONLY_FIELDS so run.json can record it as applied."""
    assert TRACKING_V2_ONLY_FIELDS <= {f.name for f in fields(TrackingConfig)}
    core = Path(__file__).resolve().parents[1] / "src" / "corridor" / "core"
    for path in core.glob("*.py"):
        if path.name == "config.py":
            continue
        text = path.read_text(encoding="utf-8")
        for name in TRACKING_V2_ONLY_FIELDS:
            assert not re.search(rf"\.{name}\b", text), f"{path.name} reads {name}"


# --------------------------------------------------------------------------
# Process noise in physical time (contract §5)
# --------------------------------------------------------------------------


def physical(q: tuple[float, float, float], scale: Scale) -> tuple[float, float, float]:
    """(um^2, um^2/min, (um/min)^2) from process_noise_px's image units."""
    p, t = scale.pixel_size_um, scale.frame_interval_min
    return (q[0] * p * p, q[1] * p * p / t, q[2] * (p / t) ** 2)


def test_process_noise_means_the_same_motion_at_any_frame_interval():
    """40 minutes is one 40 min frame, two 20 min frames or four 10 min
    frames; the prediction's covariance after it must not care which."""
    cfg = TrackingConfig()
    results = [
        physical(cfg.process_noise_px(Scale.from_values(0.467, interval), 40.0 / interval),
                 Scale.from_values(0.467, interval))
        for interval in (10.0, 20.0, 40.0)
    ]
    for r in results[1:]:
        assert r == pytest.approx(results[0], rel=1e-12)
    sigma2 = cfg.velocity_sigma_um_per_min**2
    assert results[0] == pytest.approx((sigma2 * 40**3 / 3, sigma2 * 40**2 / 2, sigma2 * 40))


def test_process_noise_does_not_depend_on_the_pixel_size_physically():
    cfg = TrackingConfig()
    kk1 = Scale.from_values(0.639, 15.0)
    kk2 = Scale.from_values(0.467, 15.0)
    assert physical(cfg.process_noise_px(kk1, 2), kk1) == pytest.approx(
        physical(cfg.process_noise_px(kk2, 2), kk2), rel=1e-12
    )
    # ...while in image units it does: the same motion is more KK2 pixels.
    assert cfg.process_noise_px(kk2, 2)[2] > cfg.process_noise_px(kk1, 2)[2]


def test_uncalibrated_process_noise_is_read_in_pixels_and_frames():
    cfg = TrackingConfig(velocity_sigma_um_per_min=0.3)
    q = cfg.process_noise_px(Scale.from_values(None, None), 3)
    assert q == pytest.approx((0.09 * 27 / 3, 0.09 * 9 / 2, 0.09 * 3))


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


def test_an_edited_walls_value_wins_from_either_block():
    """to_dict writes both blocks, so a default beside an edited value is
    not a choice; the edited one is, whichever block holds it."""
    for block in ("confinement", "geometry"):
        data = {"confinement": {}, "geometry": {}}
        data[block]["detect_walls"] = False
        config = RunConfig.from_dict(data)
        assert config.geometry.detect_walls is False, block
        assert config.confinement.detect_walls is False, block


def test_geometry_breaks_a_tie_between_two_edited_values():
    config = RunConfig.from_dict(
        {"confinement": {"min_channel_pitch_um": 20.0}, "geometry": {"min_channel_pitch_um": 25.0}}
    )
    assert config.geometry.min_channel_pitch_um == 25.0
    assert config.confinement.min_channel_pitch_um == 25.0


def test_the_two_walls_blocks_mirror_each_other_in_memory():
    """A v1 reader of confinement sees what a v2 writer put in geometry, and
    back; and a re-enable after a saved opt-out sticks."""
    config = RunConfig()
    config.geometry.detect_walls = False
    assert config.confinement.detect_walls is False
    config.confinement.min_channel_pitch_um = 22.0
    assert config.geometry.min_channel_pitch_um == 22.0
    config.confinement.mode = "vertical"  # an axis key has no v2 twin
    assert not hasattr(config.geometry, "mode")

    restored = json_round_trip(config)
    restored.geometry.detect_walls = True
    again = json_round_trip(restored)
    assert again.confinement.detect_walls is True and again.geometry.detect_walls is True


def test_a_replaced_walls_block_wins_and_the_old_one_is_unlinked():
    config = RunConfig()
    old = config.geometry
    config.geometry = GeometryConfig(detect_walls=False)
    assert config.confinement.detect_walls is False
    old.detect_walls = True
    assert config.confinement.detect_walls is False
    config.confinement = type(config.confinement)(detect_walls=True)
    assert config.geometry.detect_walls is True


def test_the_walls_link_is_invisible_and_does_not_leak_into_copies():
    config = RunConfig()
    assert "_walls_partner" not in repr(config)
    assert "_walls_partner" not in json.dumps(config.to_dict())
    twin = copy.deepcopy(config)
    assert twin == config
    twin.geometry.detect_walls = False
    assert twin.confinement.detect_walls is False
    assert config.geometry.detect_walls is True and config.confinement.detect_walls is True


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
    config.segmentation.diameter = 45.0
    config.segmentation.min_extent_px = 30
    config.segmentation.min_area_px = 50
    config.segmentation.ensemble_min_fragment_px = 35
    config.segmentation.apply_normalisation_preset("local_sharpen")
    config.geometry.min_channel_pitch_px = 52.0
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
    assert config.segmentation.normalisation_mode == "local_sharpen"
    assert config.tracking.max_gap == 5
    assert config.geometry.detect_walls is False
    assert config.measurement.msd_min_pairs == 6


def test_a_new_project_inherits_no_size_in_pixels():
    """KK1's pixel is 0.639 um and KK2's 0.467 um: a size tuned in pixels on
    one is a different physical size on the other."""
    config = RunConfig.for_new_project(saved_default_config())
    seg, defaults = config.segmentation, RunConfig().segmentation
    for name in ("diameter", "min_extent_px", "min_area_px", "ensemble_min_fragment_px"):
        assert getattr(seg, name) == getattr(defaults, name), name
    assert config.geometry.min_channel_pitch_px == GeometryConfig().min_channel_pitch_px
    assert config.confinement.min_channel_pitch_px == GeometryConfig().min_channel_pitch_px


def test_a_new_project_from_nothing_is_the_default():
    assert RunConfig.for_new_project(None) == RunConfig()
    assert RunConfig.for_new_project({}) == RunConfig()


@pytest.mark.parametrize("bad", [None, [1, 2], "x", 3])
def test_a_block_that_is_not_an_object_loads_as_the_default(bad):
    data = {name: bad for name in ("segmentation", "tracking", "geometry", "recovery", "import")}
    assert RunConfig.from_dict(data) == RunConfig()
    assert RunConfig.for_new_project({"segmentation": bad}) == RunConfig()


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


@pytest.mark.parametrize(
    "value, lower, upper",
    [
        pytest.param(1.0, float("nan"), float("nan"), id="nan-bounds"),
        pytest.param(1.0, float("-inf"), float("inf"), id="infinite-bounds"),
        pytest.param(float("nan"), 0.0, 1.0, id="nan-value"),
        pytest.param(5.0, 0.0, 1.0, id="value-above"),
        pytest.param(-0.1, 0.0, 1.0, id="value-below"),
    ],
)
def test_a_point_prediction_cannot_pass_as_an_interval(value, lower, upper):
    from corridor.core.interfaces import Prediction

    with pytest.raises(ValueError):
        Prediction(value, lower, upper, 0.9)


def test_the_interval_bounds_are_inclusive():
    """Whether a zero-width interval is calibrated is for the coverage test
    to say; the container only refuses what cannot be an interval at all."""
    from corridor.core.interfaces import Prediction

    assert Prediction(2.0, 2.0, 2.0, 0.9).upper == 2.0
    assert Prediction(0.0, 0.0, 1.0, 0.9).lower == 0.0
