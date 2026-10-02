"""The Advanced panel and Settings dialog without an axis or a model choice.

Pinned here:

*   No control can describe a migration axis or pick a model (contract §2, §5).
*   The tracking controls write the v2 fields, and the gate is shown as the
    derived ``2 x unmatched``. The legacy ``gate_chi2`` (still read by the
    1.x tracker) is written equal to it -- assigned, never ratcheted, which
    is how 1.x let the two drift apart.
*   A control does not silently rewrite a stored value just because the
    panel was opened: not a per-hour speed stored per minute, not a position
    uncertainty or a Z step with more decimals than the spin box shows.
*   Settings reports the validated model and its verification, and shows the
    ModelUnavailable text verbatim when the file is missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6", reason="the interface is not installed")

from PySide6.QtWidgets import (  # noqa: E402
    QAbstractSpinBox,
    QApplication,
    QLineEdit,
    QPushButton,
)

from corridor.core import model_registry  # noqa: E402
from corridor.core.config import RunConfig  # noqa: E402
from corridor.ui.widgets.advanced import AdvancedPanel  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def test_no_axis_and_no_model_controls(qt_app):
    panel = AdvancedPanel()
    for name in ("axis_mode", "axis_angle", "enforce_channels", "model_path",
                 "sigma_along", "sigma_across"):
        assert not hasattr(panel, name), name
    buttons = {button.text() for button in panel.findChildren(QPushButton)}
    assert "Change" not in buttons, "no 'Change model' button remains"
    standalone = [
        edit for edit in panel.findChildren(QLineEdit)
        if not isinstance(edit.parent(), QAbstractSpinBox)
    ]
    assert not standalone, "no free-text path field remains"


def test_detection_effort_offers_only_same_model_rungs(qt_app):
    panel = AdvancedPanel()
    modes = [panel.ensemble.itemData(i) for i in range(panel.ensemble.count())]
    assert modes == ["off", "thresholds", "wide"]
    assert "2× slower" in panel.ensemble.itemText(1)
    assert "4× slower" in panel.ensemble.itemText(2)


def test_a_legacy_companion_model_rung_loads_as_single_pass_with_a_note(qt_app):
    config = RunConfig()
    config.segmentation.ensemble = "max_recall"
    panel = AdvancedPanel()
    panel.set_config(config)
    assert panel.ensemble.currentData() == "off"
    assert not panel.ensemble_note.isHidden()
    assert "ran other models" in panel.ensemble_note.text()
    panel.apply_to(config)
    assert config.segmentation.ensemble == "off"


def test_tracking_controls_write_the_v2_fields(qt_app):
    config = RunConfig()
    config.tracking.gate_chi2 = 30.0
    panel = AdvancedPanel()
    panel.set_config(config)

    assert panel.respect_walls.isChecked()
    panel.respect_walls.setChecked(False)
    panel.position_sigma.setValue(0.8)
    panel.max_gap.setValue(5)
    panel.unmatched.setValue(20.0)
    panel.apply_to(config)

    trk = config.tracking
    assert trk.channel_constraint == "off"
    assert trk.position_sigma_um == pytest.approx(0.8)
    assert trk.max_gap == 5
    assert trk.unmatched_chi2 == pytest.approx(20.0)
    assert trk.effective_gate_chi2 == pytest.approx(40.0)
    # The 1.x tracker on this branch still rejects links above gate_chi2, so
    # the note's "more than 2U are never made" is only true if it equals 2U.
    assert trk.gate_chi2 == trk.effective_gate_chi2 == pytest.approx(40.0)
    assert "40.0" in panel.gate_note.text()


@pytest.mark.parametrize("unmatched", [40.0, 5.0])
def test_the_legacy_gate_follows_the_unmatched_cost_both_ways(qt_app, unmatched):
    # Raising U past the old stored gate (30) and lowering it below half of
    # it: assigned, not max()'d, so the gate never lags behind 2U.
    config = RunConfig()
    config.tracking.gate_chi2 = 30.0
    panel = AdvancedPanel()
    panel.set_config(config)
    panel.unmatched.setValue(unmatched)
    panel.apply_to(config)
    assert config.tracking.gate_chi2 == pytest.approx(2.0 * unmatched)
    assert config.tracking.gate_chi2 == config.tracking.effective_gate_chi2
    assert f"{2.0 * unmatched:.1f}" in panel.gate_note.text()


def test_untouched_fine_values_are_not_rounded_by_the_spin_boxes(qt_app):
    class Meta:
        axes = "TZYX"
        pixel_size_um = None
        frame_interval_min = None

    config = RunConfig()
    config.tracking.position_sigma_um = 0.375  # the spin box shows 0.38
    config.calibration.z_step_um = 0.123456  # the spin box shows 0.1235
    panel = AdvancedPanel()
    panel.set_config(config, Meta())
    panel.apply_to(config)
    assert config.tracking.position_sigma_um == 0.375
    assert config.calibration.z_step_um == 0.123456

    # A value the user did change is taken as shown.
    panel.position_sigma.setValue(0.5)
    panel.z_step.setValue(1.5)
    panel.apply_to(config)
    assert config.tracking.position_sigma_um == pytest.approx(0.5)
    assert config.calibration.z_step_um == pytest.approx(1.5)


def test_an_untouched_speed_keeps_its_exact_stored_value(qt_app):
    config = RunConfig()
    config.tracking.max_speed_um_per_min = 4.6712
    panel = AdvancedPanel()
    panel.set_config(config)
    assert panel.max_speed.suffix().strip() == "µm/h"
    panel.apply_to(config)
    assert config.tracking.max_speed_um_per_min == 4.6712

    panel.max_speed.setValue(120.0)
    panel.apply_to(config)
    assert config.tracking.max_speed_um_per_min == pytest.approx(2.0)


def test_model_fields_are_never_written(qt_app):
    config = RunConfig()
    config.segmentation.model_path = None
    config.segmentation.use_custom_model = True
    panel = AdvancedPanel()
    panel.set_config(config)
    panel.apply_to(config)
    assert config.segmentation.model_path is None
    assert config.segmentation.use_custom_model is True


def test_z_step_is_offered_only_for_a_stack_with_z(qt_app):
    class Meta:
        axes = "TZYX"
        pixel_size_um = None
        frame_interval_min = None

    panel = AdvancedPanel()
    panel.set_config(RunConfig())
    assert panel.z_step.isHidden()
    panel.set_config(RunConfig(), Meta())
    assert not panel.z_step.isHidden()
    panel.z_step.setValue(2.0)
    config = panel.apply_to(RunConfig())
    assert config.calibration.z_step_um == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@pytest.fixture
def settings(qt_app, tmp_path, monkeypatch):
    from corridor.store import db
    from corridor.ui import dialogs

    monkeypatch.setattr(dialogs, "_gpu_available", lambda: False)
    store = db.Store(tmp_path / "projects.db")
    yield store, dialogs
    store.close()


def test_settings_reports_a_missing_model_verbatim(settings, monkeypatch):
    store, dialogs = settings

    def missing(dimensionality="2D"):
        raise model_registry.ModelUnavailable(
            model_registry.MODEL_UNAVAILABLE_MESSAGE, [(Path("C:/nowhere/model"), "missing")]
        )

    monkeypatch.setattr(model_registry, "resolve_model", missing)
    dialog = dialogs.SettingsDialog(store)
    block = dialog.model_block
    assert not block.status.verified
    assert block.fields["status"]._value.full_text() == "missing or does not match"
    assert block.message.text().startswith(model_registry.MODEL_UNAVAILABLE_MESSAGE)
    assert "nowhere" in block.message.text()
    # The registry's id still shows, so the reader knows what is expected.
    assert block.fields["model_id"]._value.full_text() == "jhu_confined_cp3_combi"
    assert not hasattr(dialog, "model_path")


def test_settings_reports_a_verified_model(settings, monkeypatch, tmp_path):
    store, dialogs = settings
    spec = model_registry.production_spec("2D")
    resolved = model_registry.ResolvedModel(spec=spec, path=tmp_path / "m", sha256=spec.sha256)
    monkeypatch.setattr(model_registry, "resolve_model", lambda dimensionality="2D": resolved)
    dialog = dialogs.SettingsDialog(store)
    block = dialog.model_block
    assert block.fields["status"]._value.full_text() == "verified"
    assert block.fields["sha"]._value.full_text() == f"{spec.sha256[:12]}…"
    assert block.fields["version"]._value.full_text() == spec.model_version
    assert block.message.isHidden()
    dialog._save_and_close()
    assert store.get_setting("model_path", None) is None, "no model path is saved"
