"""Advanced parameters, disclosed only when asked for.

Nothing here is hidden from the user, and nothing here is in their way. The
defaults are the ones the supplied data was labelled with, and each control
says what it means in the units the researcher thinks in.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.config import (
    AXIS_ANGLE,
    AXIS_AUTO,
    AXIS_HORIZONTAL,
    AXIS_VERTICAL,
    RunConfig,
)
from ...core.imaging import StackMetadata
from ..theme import PALETTE, SPACE
from .common import divider, ghost_button, label

AXIS_CHOICES = [
    ("Detect from the channel walls", AXIS_AUTO),
    ("Vertical", AXIS_VERTICAL),
    ("Horizontal", AXIS_HORIZONTAL),
    ("A specific angle", AXIS_ANGLE),
]


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


class AdvancedPanel(QWidget):
    """Edits a RunConfig in place and reports changes."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = RunConfig()
        self._metadata: StackMetadata | None = None
        self._loading = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["lg"])

        layout.addWidget(_section("Calibration"))
        layout.addLayout(self._calibration_form())
        layout.addWidget(divider())

        layout.addWidget(_section("Confinement"))
        layout.addLayout(self._confinement_form())
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

        self.calibration_note = label("", "tertiary")
        self.calibration_note.setWordWrap(True)
        form.addRow("", self.calibration_note)
        return form

    def _confinement_form(self) -> QFormLayout:
        form = self._form()
        self.axis_mode = QComboBox()
        for text, value in AXIS_CHOICES:
            self.axis_mode.addItem(text, value)
        self.axis_mode.currentIndexChanged.connect(self._axis_changed)
        form.addRow("Migration axis", self.axis_mode)

        self.axis_angle = _spin(-180.0, 180.0, 1.0, 1, "°")
        self.axis_angle.valueChanged.connect(self._emit)
        self.axis_angle_row = self.axis_angle
        form.addRow("Angle", self.axis_angle)

        self.enforce_channels = QCheckBox("Keep cells in their own channel")
        self.enforce_channels.toggled.connect(self._emit)
        form.addRow("", self.enforce_channels)
        return form

    def _segmentation_form(self) -> QFormLayout:
        form = self._form()
        row = QHBoxLayout()
        row.setSpacing(SPACE["sm"])
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Bundled model")
        self.model_path.setReadOnly(True)
        browse = ghost_button("Change", "folder", self._choose_model)
        row.addWidget(self.model_path, 1)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("Cellpose model", holder)

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

        self.use_gpu = QCheckBox("Use the GPU if available")
        self.use_gpu.toggled.connect(self._emit)
        form.addRow("", self.use_gpu)
        return form

    def _tracking_form(self) -> QFormLayout:
        form = self._form()
        self.max_gap = _int_spin(0, 20, " frames")
        self.max_gap.valueChanged.connect(self._emit)
        self.max_gap.setToolTip(
            "How many frames in a row a cell may be missing and still be "
            "recognised as the same cell afterwards."
        )
        form.addRow("Allowed disappearance", self.max_gap)

        self.max_speed = _spin(0.01, 500.0, 0.5, 2, " µm/min")
        self.max_speed.valueChanged.connect(self._emit)
        form.addRow("Fastest plausible cell", self.max_speed)

        self.sigma_along = _spin(0.1, 100.0, 0.5, 2, " µm")
        self.sigma_along.valueChanged.connect(self._emit)
        self.sigma_along.setToolTip("Expected error of the motion prediction along the channel.")
        form.addRow("Along-channel spread", self.sigma_along)

        self.sigma_across = _spin(0.01, 100.0, 0.1, 2, " µm")
        self.sigma_across.valueChanged.connect(self._emit)
        self.sigma_across.setToolTip(
            "Expected sideways wander. Smaller values enforce the confinement "
            "prior more strongly."
        )
        form.addRow("Across-channel spread", self.sigma_across)

        self.unmatched = _spin(1.0, 200.0, 1.0, 1)
        self.unmatched.valueChanged.connect(self._emit)
        self.unmatched.setToolTip(
            "How poor a match has to be before the tracker prefers to leave the "
            "cell unmatched. In units of squared standard deviations."
        )
        form.addRow("Unmatched cost", self.unmatched)

        self.min_observations = _int_spin(1, 100)
        self.min_observations.valueChanged.connect(self._emit)
        form.addRow("Minimum observations", self.min_observations)
        return form

    # ------------------------------------------------------------------ state
    def set_config(self, config: RunConfig, metadata: StackMetadata | None = None) -> None:
        self._loading = True
        self._config = config
        self._metadata = metadata

        cal = config.calibration
        self.pixel_size.setValue(cal.pixel_size_um or 0.0)
        self.frame_interval.setValue(cal.frame_interval_min or 0.0)
        self._describe_calibration()

        index = self.axis_mode.findData(config.confinement.mode)
        self.axis_mode.setCurrentIndex(max(0, index))
        self.axis_angle.setValue(config.confinement.angle_deg)
        self.axis_angle.setEnabled(config.confinement.mode == AXIS_ANGLE)
        self.enforce_channels.setChecked(config.tracking.enforce_channel_identity)

        seg = config.segmentation
        self.model_path.setText(seg.model_path or "")
        self.model_path.setToolTip(seg.model_path or "No model selected")
        self.cellprob.setValue(seg.cellprob_threshold)
        self.flow.setValue(seg.flow_threshold)
        self.diameter.setValue(seg.diameter or 0.0)
        self.min_extent.setValue(seg.min_extent_px)
        self.use_gpu.setChecked(seg.use_gpu)

        trk = config.tracking
        self.max_gap.setValue(trk.max_gap)
        self.max_speed.setValue(trk.max_speed_um_per_min)
        self.sigma_along.setValue(trk.sigma_along_um)
        self.sigma_across.setValue(trk.sigma_perp_um)
        self.unmatched.setValue(trk.unmatched_chi2)
        self.min_observations.setValue(trk.min_observations)
        self._loading = False

    def _describe_calibration(self) -> None:
        if self._metadata is None:
            self.calibration_note.setText("")
            return
        pixel = self._metadata.pixel_size_um
        interval = self._metadata.frame_interval_min
        parts = []
        if pixel.known:
            parts.append(f"file says {pixel.describe(' µm/px')}")
        if interval.known:
            parts.append(f"{interval.describe(' min')}")
        self.calibration_note.setText("  ·  ".join(parts))

    def apply_to(self, config: RunConfig) -> RunConfig:
        """Copy the current control values into ``config``."""
        config.calibration.pixel_size_um = (
            self.pixel_size.value() if self.pixel_size.value() > 0 else None
        )
        config.calibration.frame_interval_min = (
            self.frame_interval.value() if self.frame_interval.value() > 0 else None
        )
        config.confinement.mode = self.axis_mode.currentData()
        config.confinement.angle_deg = self.axis_angle.value()

        seg = config.segmentation
        text = self.model_path.text().strip()
        seg.model_path = text or None
        seg.use_custom_model = bool(text)
        seg.cellprob_threshold = self.cellprob.value()
        seg.flow_threshold = self.flow.value()
        seg.diameter = self.diameter.value() if self.diameter.value() > 0 else None
        seg.min_extent_px = self.min_extent.value()
        seg.use_gpu = self.use_gpu.isChecked()

        trk = config.tracking
        trk.max_gap = self.max_gap.value()
        trk.max_speed_um_per_min = self.max_speed.value()
        trk.sigma_along_um = self.sigma_along.value()
        trk.sigma_perp_um = self.sigma_across.value()
        trk.unmatched_chi2 = self.unmatched.value()
        # The hard ceiling is two unmatched decisions: keep them consistent so
        # the optimiser can never accept a pairing the gate would refuse.
        trk.gate_chi2 = max(trk.gate_chi2, 2.0 * trk.unmatched_chi2)
        trk.min_observations = self.min_observations.value()
        trk.enforce_channel_identity = self.enforce_channels.isChecked()
        return config

    # ----------------------------------------------------------------- events
    def _axis_changed(self) -> None:
        self.axis_angle.setEnabled(self.axis_mode.currentData() == AXIS_ANGLE)
        self._emit()

    def _choose_model(self) -> None:
        start = self.model_path.text() or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a Cellpose model", start, "All files (*)"
        )
        if path:
            self.model_path.setText(path)
            self.model_path.setToolTip(path)
            self._emit()

    def _emit(self) -> None:
        if not self._loading:
            self.changed.emit()
