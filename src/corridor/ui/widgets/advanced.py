"""Advanced parameters, disclosed only when asked for.

Nothing here is hidden from the user, and nothing here is in their way. The
defaults are the ones the supplied data was labelled with, and each control
says what it means in the units the researcher thinks in.

What is deliberately *not* here in 2.0:

*   **No migration axis.** The tracker is axis-free (contract §5); the only
    device-related choice is whether detected channel walls may stop a link
    (``TrackingConfig.channel_constraint``).
*   **No model picker.** The model is not a setting (contract §2). Production
    resolves one SHA-256-verified file; the dataset screen and Settings show
    which, and nothing here can change it.
*   **No hard-gate control.** The gate is derived, ``2 x unmatched cost``
    (``TrackingConfig.effective_gate_chi2``), so it is shown, never edited --
    the 1.x panel ratcheted a stored gate upwards, which is how the two
    drifted apart. The legacy ``gate_chi2`` is written equal to it on every
    apply, for the 1.x tracker that still reads it.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.config import (
    CHANNEL_CONSTRAINT_AUTO,
    CHANNEL_CONSTRAINT_OFF,
    ENSEMBLE_LABELS,
    ENSEMBLE_OFF,
    ENSEMBLE_THRESHOLDS,
    ENSEMBLE_WIDE,
    NORMALISATION_LABELS,
    NORMALISATION_MODES,
    RunConfig,
    SegmentationConfig,
)
from ..theme import SPACE
from .common import divider, label

#: The detection-effort rungs offered. Each re-runs the *same* validated model
#: at other thresholds; the 1.x rungs that ran other models (``models``,
#: ``max_recall``) are gone with the model choice itself (contract §2).
ENSEMBLE_CHOICES = (ENSEMBLE_OFF, ENSEMBLE_THRESHOLDS, ENSEMBLE_WIDE)

#: Speeds are shown per hour, the unit results are reported in (contract §6),
#: and stored per minute, the unit the configuration has always used.
MIN_PER_HR = 60.0


def _spin(
    minimum: float, maximum: float, step: float, decimals: int, suffix: str = ""
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    box.setAlignment(Qt.AlignRight)
    box.setMinimumWidth(104)
    return box


def _int_spin(minimum: int, maximum: int, suffix: str = "") -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    box.setAlignment(Qt.AlignRight)
    box.setMinimumWidth(104)
    return box


def _section(text: str) -> QLabel:
    heading = label(text.upper(), "tertiary")
    font = heading.font()
    font.setBold(True)
    font.setLetterSpacing(QFont.PercentageSpacing, 112)
    heading.setFont(font)
    return heading


def _is_3d(metadata: Any) -> bool:
    """Whether the opened file has a Z axis, read from whatever it reports."""
    if metadata is None:
        return False
    axes = str(getattr(metadata, "axes", "") or "")
    if "Z" in axes.upper():
        return True
    return str(getattr(metadata, "dimensionality", "") or "").upper() == "3D"


class AdvancedPanel(QWidget):
    """Edits a RunConfig in place and reports changes."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = RunConfig()
        self._metadata: Any = None
        self._loading = False
        #: What a unit-converting control displayed when the config was loaded;
        #: see :meth:`_converted`.
        self._shown_at_load: dict[str, float] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["lg"])

        layout.addWidget(_section("Calibration"))
        layout.addLayout(self._calibration_form())
        layout.addWidget(divider())

        layout.addWidget(_section("Segmentation"))
        layout.addLayout(self._segmentation_form())
        layout.addWidget(divider())

        layout.addWidget(_section("Tracking"))
        layout.addLayout(self._tracking_form())
        layout.addStretch(1)

    # ------------------------------------------------------------------ forms
    def _form(self) -> QFormLayout:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        # When the panel is narrow, a label sits above its control rather than
        # being squeezed against it until the value is cut off.
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setHorizontalSpacing(SPACE["lg"])
        form.setVerticalSpacing(SPACE["md"])
        form.setContentsMargins(0, 0, 0, 0)
        return form

    def _calibration_form(self) -> QFormLayout:
        form = self._form()
        self.pixel_size = _spin(0.0, 100.0, 0.001, 6, " µm/px")
        self.pixel_size.setSpecialValueText("from the file")
        self.pixel_size.valueChanged.connect(self._emit)
        form.addRow("Pixel size", self.pixel_size)

        self.frame_interval = _spin(0.0, 10000.0, 0.1, 4, " min")
        self.frame_interval.setSpecialValueText("from the file")
        self.frame_interval.valueChanged.connect(self._emit)
        form.addRow("Frame interval", self.frame_interval)

        self.z_step = _spin(0.0, 1000.0, 0.1, 4, " µm")
        self.z_step.setSpecialValueText("from the file")
        self.z_step.valueChanged.connect(self._emit)
        self.z_step.setToolTip(
            "Distance between Z planes. Never assumed equal to the pixel size: "
            "without it, 3-D results are reported in voxels and every µm³ and "
            "µm² column stays empty."
        )
        self.z_step_label = QLabel("Z step")
        form.addRow(self.z_step_label, self.z_step)

        self.calibration_note = label("", "tertiary")
        self.calibration_note.setWordWrap(True)
        form.addRow("", self.calibration_note)
        return form

    def _segmentation_form(self) -> QFormLayout:
        form = self._form()
        self.cellprob = _spin(-12.0, 12.0, 0.5, 2)
        self.cellprob.valueChanged.connect(self._emit)
        self.cellprob.setToolTip(
            "Lower finds dimmer objects. On the supplied data, values below -3 "
            "produce fewer detections, not more, because the extra candidates "
            "fail the flow check."
        )
        form.addRow("Cell probability", self.cellprob)

        self.flow = _spin(0.0, 3.0, 0.1, 2)
        self.flow.valueChanged.connect(self._emit)
        self.flow.setToolTip(
            "Higher accepts less consistent shapes. Above 0.4 this model starts "
            "finding objects in empty channels."
        )
        form.addRow("Flow threshold", self.flow)

        self.diameter = _spin(0.0, 500.0, 1.0, 1, " px")
        self.diameter.setSpecialValueText("from the model")
        self.diameter.valueChanged.connect(self._emit)
        form.addRow("Cell diameter", self.diameter)

        self.min_extent = _int_spin(0, 500, " px")
        self.min_extent.valueChanged.connect(self._emit)
        self.min_extent.setToolTip(
            "Objects whose longer side is smaller than this are discarded. "
            "Everything removed is counted in the run's diagnostics."
        )
        form.addRow("Smallest object", self.min_extent)

        self.normalisation = QComboBox()
        for mode in NORMALISATION_MODES:
            self.normalisation.addItem(NORMALISATION_LABELS[mode], mode)
        # A configuration edited outside this panel may match no preset. Rather
        # than silently snapping it to the nearest one, the panel grows an entry
        # that says so.
        self.normalisation.addItem("Custom", "custom")
        self.normalisation.currentIndexChanged.connect(self._emit)
        self.normalisation.setToolTip(
            "How the image is rescaled before the model sees it.\n\n"
            "This matters more than it looks. The two halves of the supplied "
            "training data hold cells of the same size (12.1 and 12.2 px wide) "
            "at very different contrast (0.22 and 0.38 of the image range above "
            "background), and that difference — not shape, not scale — is most "
            "of what a model trained on one meets in the other.\n\n"
            "Local contrast rescales each region to its own background instead "
            "of to a global range one bright structure can dominate.\n\n"
            "See docs/ACCURACY.md for what each setting was measured to do."
        )
        form.addRow("Image normalisation", self.normalisation)

        self.ensemble = QComboBox()
        for mode in ENSEMBLE_CHOICES:
            # The cost is what this rung really does: every pass runs the one
            # validated model, so there is no companion model whose absence
            # could make the advertised cost untrue (1.x had to check for that).
            cost = SegmentationConfig(ensemble=mode).ensemble_cost_factor()
            text = ENSEMBLE_LABELS.get(mode, mode)
            if cost > 1:
                text += f"  ·  {cost}× slower"
            self.ensemble.addItem(text, mode)
        self.ensemble.currentIndexChanged.connect(self._emit)
        self.ensemble.setToolTip(
            "How hard to look for cells the default settings miss.\n\n"
            "Extra passes re-run the same validated model at other thresholds. "
            "They can only add detections, never remove them, and anything only "
            "they found is marked in the results. On the supplied labelled "
            "images (docs/recall_experiment.json), two thresholds raises recall "
            "from 0.846 to 0.854 for twice the time, and four thresholds to "
            "0.874 for four times the time while precision falls from 0.832 to "
            "0.793."
        )
        form.addRow("Detection effort", self.ensemble)
        self.ensemble_note = label("", "tertiary")
        self.ensemble_note.setWordWrap(True)
        self.ensemble_note.hide()
        form.addRow("", self.ensemble_note)

        self.use_gpu = QCheckBox("Use the GPU if available")
        self.use_gpu.toggled.connect(self._emit)
        form.addRow("", self.use_gpu)
        return form

    def _tracking_form(self) -> QFormLayout:
        form = self._form()
        self.respect_walls = QCheckBox("Respect detected channel walls")
        self.respect_walls.toggled.connect(self._emit)
        self.respect_walls.setToolTip(
            "Never link a cell across a channel wall.\n\n"
            "Applied only when the walls are actually detected in the image; "
            "on a field with no visible walls nothing is constrained. The run "
            "records whether it was applied."
        )
        form.addRow("", self.respect_walls)

        self.position_sigma = _spin(0.01, 50.0, 0.05, 2, " µm")
        self.position_sigma.valueChanged.connect(self._emit)
        self.position_sigma.setToolTip(
            "How precisely a cell's centre is known in one frame, whatever its "
            "shape. Each cell's own length and width add to this along and "
            "across its body, so a long thin cell is trusted less along its "
            "length than across it."
        )
        form.addRow("Position uncertainty", self.position_sigma)

        self.max_speed = _spin(0.1, 30000.0, 6.0, 1, " µm/h")
        self.max_speed.valueChanged.connect(self._emit)
        self.max_speed.setToolTip(
            "No link implying a faster movement is ever made, across a gap "
            "included. A hard physical limit, not a typical speed."
        )
        form.addRow("Fastest plausible cell", self.max_speed)

        self.max_gap = _int_spin(0, 20, " frames")
        self.max_gap.valueChanged.connect(self._emit)
        self.max_gap.setToolTip(
            "How many frames in a row a cell may be missing and still be "
            "recognised as the same cell afterwards."
        )
        form.addRow("Allowed disappearance", self.max_gap)

        self.unmatched = _spin(1.0, 200.0, 1.0, 1)
        self.unmatched.valueChanged.connect(self._unmatched_changed)
        self.unmatched.setToolTip(
            "How poor a match has to be before the tracker prefers to leave the "
            "cell unmatched. In units of squared standard deviations."
        )
        form.addRow("Unmatched cost", self.unmatched)
        self.gate_note = label("", "tertiary")
        self.gate_note.setWordWrap(True)
        form.addRow("", self.gate_note)

        self.min_observations = _int_spin(1, 100)
        self.min_observations.valueChanged.connect(self._emit)
        form.addRow("Minimum observations", self.min_observations)
        return form

    # ------------------------------------------------------------------ state
    def set_config(self, config: RunConfig, metadata: Any = None) -> None:
        self._loading = True
        self._config = config
        self._metadata = metadata

        cal = config.calibration
        self.pixel_size.setValue(cal.pixel_size_um or 0.0)
        self.frame_interval.setValue(cal.frame_interval_min or 0.0)
        self.z_step.setValue(getattr(cal, "z_step_um", None) or 0.0)
        self._shown_at_load["z_step"] = self.z_step.value()
        three_d = _is_3d(metadata)
        self.z_step.setVisible(three_d)
        self.z_step_label.setVisible(three_d)
        self._describe_calibration()

        seg = config.segmentation
        self.cellprob.setValue(seg.cellprob_threshold)
        self.flow.setValue(seg.flow_threshold)
        self.diameter.setValue(seg.diameter or 0.0)
        self.min_extent.setValue(seg.min_extent_px)
        index = self.normalisation.findData(seg.normalisation_mode)
        self.normalisation.setCurrentIndex(max(0, index))
        index = self.ensemble.findData(seg.ensemble)
        if index < 0:
            # A saved 1.x configuration naming a rung that ran other models.
            # It runs as a single pass now (contract §2); say so rather than
            # silently showing a different setting from the one saved.
            previous = ENSEMBLE_LABELS.get(seg.ensemble, seg.ensemble)
            self.ensemble_note.setText(
                f"The saved setting “{previous}” ran other models, which "
                "Corridor 2.0 no longer does. It runs as a single pass."
            )
            self.ensemble_note.show()
            index = self.ensemble.findData(ENSEMBLE_OFF)
        else:
            self.ensemble_note.hide()
        self.ensemble.setCurrentIndex(max(0, index))
        self.use_gpu.setChecked(seg.use_gpu)

        trk = config.tracking
        self.respect_walls.setChecked(
            getattr(trk, "channel_constraint", CHANNEL_CONSTRAINT_AUTO) != CHANNEL_CONSTRAINT_OFF
        )
        self.position_sigma.setValue(trk.position_sigma_um)
        self._shown_at_load["position_sigma"] = self.position_sigma.value()
        self.max_speed.setValue(trk.max_speed_um_per_min * MIN_PER_HR)
        self._shown_at_load["max_speed"] = self.max_speed.value()
        self.max_gap.setValue(trk.max_gap)
        self.unmatched.setValue(trk.unmatched_chi2)
        self.min_observations.setValue(trk.min_observations)
        self._describe_gate()
        self._loading = False

    def _describe_calibration(self) -> None:
        if self._metadata is None:
            self.calibration_note.setText("")
            return
        parts = []
        pixel = getattr(self._metadata, "pixel_size_um", None)
        interval = getattr(self._metadata, "frame_interval_min", None)
        if pixel is not None and getattr(pixel, "known", False):
            parts.append(f"file says {pixel.describe(' µm/px')}")
        if interval is not None and getattr(interval, "known", False):
            parts.append(f"{interval.describe(' min')}")
        z_step = getattr(self._metadata, "z_step_um", None)
        if z_step is not None:
            value = getattr(z_step, "value", z_step)
            if getattr(z_step, "known", None) is not False and value:
                parts.append(f"Z step {float(value):.6g} µm")
        self.calibration_note.setText("  ·  ".join(parts))

    def apply_to(self, config: RunConfig) -> RunConfig:
        """Copy the current control values into ``config``.

        Model selection is never written: the legacy ``model_path`` /
        ``use_custom_model`` fields are left exactly as they were, and the
        segmentation service ignores them (contract §2).
        """
        config.calibration.pixel_size_um = (
            self.pixel_size.value() if self.pixel_size.value() > 0 else None
        )
        config.calibration.frame_interval_min = (
            self.frame_interval.value() if self.frame_interval.value() > 0 else None
        )
        if hasattr(config.calibration, "z_step_um"):
            shown = self.z_step.value()
            config.calibration.z_step_um = (
                self._converted("z_step", shown, config.calibration.z_step_um, 1.0)
                if shown > 0
                else None
            )

        seg = config.segmentation
        seg.cellprob_threshold = self.cellprob.value()
        seg.flow_threshold = self.flow.value()
        seg.diameter = self.diameter.value() if self.diameter.value() > 0 else None
        seg.min_extent_px = self.min_extent.value()
        # "Custom" means the fields were set elsewhere and this panel has no
        # opinion about them; applying a preset for it would discard them.
        chosen = self.normalisation.currentData()
        if chosen != "custom":
            seg.apply_normalisation_preset(chosen)
        seg.ensemble = self.ensemble.currentData()
        seg.use_gpu = self.use_gpu.isChecked()

        trk = config.tracking
        trk.channel_constraint = (
            CHANNEL_CONSTRAINT_AUTO if self.respect_walls.isChecked() else CHANNEL_CONSTRAINT_OFF
        )
        # Kept in step for the 1.x tracker until the integration removes it.
        trk.enforce_channel_identity = self.respect_walls.isChecked()
        trk.position_sigma_um = self._converted(
            "position_sigma", self.position_sigma.value(), trk.position_sigma_um, 1.0
        )
        trk.max_speed_um_per_min = self._converted(
            "max_speed", self.max_speed.value(), trk.max_speed_um_per_min, MIN_PER_HR
        )
        trk.max_gap = self.max_gap.value()
        trk.unmatched_chi2 = self.unmatched.value()
        # v2 derives the gate (effective_gate_chi2 = 2U) and never reads
        # gate_chi2, but the 1.x tracker still running on this branch rejects
        # every link costing more than gate_chi2. Leaving it as loaded would
        # make raising the unmatched cost silently do nothing above the old
        # gate, while the note under the control says 2U. So the legacy field
        # is set to exactly the derived value -- assigned, not max()'d: the
        # 1.x ratchet is how the two drifted apart in the first place.
        trk.gate_chi2 = trk.effective_gate_chi2
        trk.min_observations = self.min_observations.value()
        return config

    def _converted(self, key: str, shown: float, stored: Any, factor: float) -> Any:
        """The stored value, unless the user actually changed the shown one.

        A spin box rounds what it displays; writing the rounded number back
        on every apply would silently change a saved parameter just because
        the panel was opened (4.6712 µm/min -> 280.3 µm/h -> 4.67167 µm/min;
        a position uncertainty of 0.375 µm -> 0.38; a Z step of 0.123456 µm
        -> 0.1235). So an untouched control keeps the exact stored value, and
        ``factor`` is 1 for a control shown in its stored unit.
        """
        if stored is not None and abs(shown - self._shown_at_load.get(key, float("nan"))) < 1e-9:
            return stored
        return shown / factor

    # ----------------------------------------------------------------- events
    def _unmatched_changed(self) -> None:
        self._describe_gate()
        self._emit()

    def _describe_gate(self) -> None:
        gate = 2.0 * self.unmatched.value()
        self.gate_note.setText(
            f"Links costing more than {gate:.1f} (twice the unmatched cost) are "
            "never made. Derived, so the two cannot drift apart."
        )

    def _emit(self) -> None:
        if not self._loading:
            self.changed.emit()
