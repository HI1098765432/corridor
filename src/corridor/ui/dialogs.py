"""Dialogs: settings, about, axis order, and an error box that respects the reader.

The error dialog exists because a traceback is not an error message. It says
what happened and what to do, and keeps the technical detail one click away
for the case where someone does want it.

Settings no longer holds a model path (contract §2). 1.x wrote a
``model_path`` setting that nothing ever read, so the control was dead; in
2.0 the model is not a setting at all, and the dialog *reports* which
validated model production uses and whether this machine's copy matches its
checksum.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import app_meta
from ..core import updates
from ..store import db
from .model_status import ModelStatus, verified_model
from .theme import PALETTE, SPACE
from .widgets.common import Field, divider, ghost_button, label, primary_button


def _gpu_available() -> bool:
    """Whether a CUDA GPU is usable. Imports torch, so only called on demand."""
    try:
        from ..core.segmentation import gpu_available  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - a missing backend means no GPU
        return False
    try:
        return bool(gpu_available())
    except Exception:  # noqa: BLE001
        return False


def _cellpose_version() -> str:
    try:
        from ..core.segmentation import cellpose_version  # noqa: PLC0415

        return str(cellpose_version())
    except Exception:  # noqa: BLE001
        return "not installed"


class ErrorDialog(QDialog):
    def __init__(self, message: str, detail: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(app_meta.APP_NAME)
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["lg"]
        )
        layout.setSpacing(SPACE["md"])

        headline, _, rest = message.partition("\n\n")
        self.headline = label(headline.strip() or "Something went wrong", "subtitle")
        self.headline.setWordWrap(True)
        layout.addWidget(self.headline)

        if rest.strip():
            body = label(rest.strip(), "secondary")
            body.setWordWrap(True)
            layout.addWidget(body)

        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setPlainText(detail or "No further detail was recorded.")
        self._detail.setFixedHeight(190)
        self._detail.setStyleSheet(
            f"background: {PALETTE.surface_sunken}; border: 1px solid {PALETTE.border};"
            f"border-radius: 8px; font-family: Consolas, monospace; font-size: 11px;"
            f"color: {PALETTE.text_secondary};"
        )
        self._detail.hide()
        layout.addWidget(self._detail)

        row = QHBoxLayout()
        self._toggle = ghost_button("Technical details", "chevron-right", self._toggle_detail)
        row.addWidget(self._toggle)
        row.addStretch(1)
        close = primary_button("Close", self.accept)
        row.addWidget(close)
        layout.addLayout(row)

    def _toggle_detail(self) -> None:
        shown = not self._detail.isVisible()
        self._detail.setVisible(shown)
        self._toggle.setText("Hide details" if shown else "Technical details")
        self.adjustSize()


class ModelBlock(QWidget):
    """Read-only rows describing the production model and its verification."""

    def __init__(self, status: ModelStatus, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.status = status
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["xs"])
        self.fields: dict[str, Field] = {}
        for key, name in (
            ("model_id", "Model"),
            ("version", "Version"),
            ("sha", "SHA-256"),
            ("status", "Status"),
        ):
            field = Field(name)
            self.fields[key] = field
            layout.addWidget(field)
        self.message = label("", "tertiary")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.show_status(status)

    def show_status(self, status: ModelStatus) -> None:
        self.status = status
        self.fields["model_id"].set_value(status.model_id or "—")
        self.fields["version"].set_value(status.model_version or "—")
        self.fields["sha"].set_value(status.sha_prefix or "—", status.sha256 or "")
        self.fields["status"].set_value(
            status.status_text, str(status.path) if status.path else ""
        )
        # The full ModelUnavailable text, verbatim, with the paths tried.
        self.message.setText(status.message)
        self.message.setVisible(bool(status.message))


class SettingsDialog(QDialog):
    def __init__(self, store: db.Store, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["lg"])
        layout.setSpacing(SPACE["lg"])

        layout.addWidget(label("Settings", "title"))

        form = QFormLayout()
        form.setHorizontalSpacing(SPACE["xl"])
        form.setVerticalSpacing(SPACE["md"])
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        # Where exports go by default.
        export_row = QHBoxLayout()
        self.export_dir = QLineEdit(
            str(store.get_setting("export_dir", str(Path.home() / "Documents")))
        )
        self.export_dir.setCursorPosition(0)
        browse = ghost_button("Change", "folder", self._choose_export)
        export_row.addWidget(self.export_dir, 1)
        export_row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(export_row)
        form.addRow("Export folder", holder)

        self.use_gpu = QCheckBox("Use the GPU when one is available")
        self.use_gpu.setChecked(bool(store.get_setting("use_gpu", False)))
        if not _gpu_available():
            self.use_gpu.setEnabled(False)
            self.use_gpu.setText("Use the GPU  ·  no compatible GPU found")
        form.addRow("Processing", self.use_gpu)

        # Whether to contact the internet at all. Off unless the user turned it
        # on: this application otherwise makes no network calls, which is worth
        # keeping true by default on a workstation holding unpublished data.
        self.check_updates = QCheckBox("Check for a newer version at startup")
        self.check_updates.setChecked(updates.is_enabled(store))
        self.check_updates.setToolTip(
            "Asks GitHub once per session whether a newer version exists. "
            "Nothing about you, your images or your results is sent, and "
            "Corridor never downloads or installs anything on its own — "
            "it shows a link and you decide."
        )
        form.addRow("Updates", self.check_updates)
        layout.addLayout(form)

        layout.addWidget(divider())
        layout.addWidget(label("Segmentation model", "subtitle"))
        explanation = label(
            "Corridor uses one validated model, checked against its recorded "
            "SHA-256 before every analysis. It is not a setting.",
            "tertiary",
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.model_block = ModelBlock(verified_model("2D"))
        layout.addWidget(self.model_block)

        layout.addWidget(divider())
        location = label(f"Projects are stored in  {db.app_data_dir()}", "tertiary")
        location.setWordWrap(True)
        layout.addWidget(location)

        row = QHBoxLayout()
        about = ghost_button("About", "", lambda: AboutDialog(self).exec())
        row.addWidget(about)
        row.addStretch(1)
        row.addWidget(primary_button("Done", self._save_and_close))
        layout.addLayout(row)

    def _choose_export(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Export folder", self.export_dir.text())
        if path:
            self.export_dir.setText(path)

    def _save_and_close(self) -> None:
        self.store.set_setting("export_dir", self.export_dir.text())
        self.store.set_setting("use_gpu", self.use_gpu.isChecked())
        # Recorded through the same path the first-run question uses, so the
        # "has been asked" flag is set either way and the prompt does not
        # reappear after somebody has chosen here.
        updates.record_choice(self.store, self.check_updates.isChecked())
        self.accept()


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"About {app_meta.APP_NAME}")
        self.setFixedWidth(460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["lg"])
        layout.setSpacing(SPACE["sm"])

        layout.addWidget(label(app_meta.APP_NAME, "title"))
        layout.addWidget(label(app_meta.APP_TAGLINE, "secondary"))
        layout.addSpacing(SPACE["md"])

        status = verified_model("2D")
        model_text = (
            f"{status.model_id} {status.model_version}  ·  {status.status_text}"
            if status.model_id
            else "no validated model registered"
        )
        lines = [
            f"Version {app_meta.APP_VERSION}",
            f"Cellpose {_cellpose_version()}",
            f"Model: {model_text}",
            f"GPU: {'available' if _gpu_available() else 'not available, using the CPU'}",
        ]
        for line in lines:
            layout.addWidget(label(line, "tertiary"))

        layout.addSpacing(SPACE["md"])
        layout.addWidget(divider())
        layout.addSpacing(SPACE["sm"])

        help_text = label(
            "Open a TIFF time-lapse, press Analyse, then review the tracks over "
            "the image. Results are written before the viewer opens, and stay in "
            "your projects folder so you can reopen them later.\n\n"
            "Shortcuts:  Ctrl+O open  ·  Ctrl+E export  ·  Space play  ·  "
            "← → step  ·  0 fit  ·  Esc back",
            "secondary",
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        layout.addSpacing(SPACE["md"])
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(primary_button("Close", self.accept))
        layout.addLayout(row)


def describe_axes(axes: str) -> str:
    """A reader's description of a canonical axis order."""
    text = str(axes).upper()
    has_t, has_z = "T" in text, "Z" in text
    if has_t and has_z:
        return "a time-lapse of Z-stacks (time and depth)"
    if has_t:
        return "a time-lapse: each plane is a time point"
    if has_z:
        return "a Z-stack: each plane is a depth, one time point"
    return "a single image"


class AxisOrderDialog(QDialog):
    """Asks which axis order a file has when its metadata cannot say.

    A Z axis read as time is a wrong answer, not a degraded one (contract §4),
    so the importer refuses to guess and the user chooses. Nothing is
    preselected: the choice must be made, not accepted by default.
    """

    def __init__(
        self, choices: Sequence[str], message: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Which axis is which?")
        self.setMinimumWidth(480)
        self._choice: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["lg"])
        layout.setSpacing(SPACE["md"])
        title = label("Corridor cannot tell time from depth in this file", "subtitle")
        title.setWordWrap(True)
        layout.addWidget(title)
        body = label(message or "", "secondary")
        body.setWordWrap(True)
        body.setVisible(bool(message))
        layout.addWidget(body)

        self.group = QButtonGroup(self)
        self.buttons: list[QRadioButton] = []
        for value in choices:
            button = QRadioButton(f"{value}  —  {describe_axes(value)}")
            button.setProperty("axes", str(value))
            self.group.addButton(button)
            self.buttons.append(button)
            layout.addWidget(button)
        self.group.buttonToggled.connect(self._toggled)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(ghost_button("Cancel", "", self.reject))
        self.ok_button = primary_button("Open", self.accept)
        self.ok_button.setEnabled(False)
        row.addWidget(self.ok_button)
        layout.addLayout(row)

    def _toggled(self, button: QRadioButton, checked: bool) -> None:
        if checked:
            self._choice = str(button.property("axes"))
            self.ok_button.setEnabled(True)

    def choose(self, axes: str) -> None:
        for button in self.buttons:
            if button.property("axes") == axes:
                button.setChecked(True)

    @property
    def choice(self) -> str | None:
        return self._choice
