"""Dialogs: settings, about, and an error box that respects the reader.

The error dialog exists because a traceback is not an error message. It says
what happened and what to do, and keeps the technical detail one click away
for the case where someone does want it.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import app_meta, resources
from ..core.segmentation import cellpose_version, gpu_available
from ..store import db
from .theme import PALETTE, SPACE, stylesheet
from .widgets.common import divider, ghost_button, label, primary_button


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
        title = label(headline.strip() or "Something went wrong", "subtitle")
        title.setWordWrap(True)
        layout.addWidget(title)

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
        browse = ghost_button("Change", "folder", self._choose_export)
        export_row.addWidget(self.export_dir, 1)
        export_row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(export_row)
        form.addRow("Export folder", holder)

        # Default model.
        model_row = QHBoxLayout()
        default_model = store.get_setting("model_path", "") or (
            str(resources.bundled_model_path() or "")
        )
        self.model_path = QLineEdit(default_model)
        model_browse = ghost_button("Change", "folder", self._choose_model)
        model_row.addWidget(self.model_path, 1)
        model_row.addWidget(model_browse)
        model_holder = QWidget()
        model_holder.setLayout(model_row)
        form.addRow("Cellpose model", model_holder)

        self.use_gpu = QCheckBox("Use the GPU when one is available")
        self.use_gpu.setChecked(bool(store.get_setting("use_gpu", False)))
        if not gpu_available():
            self.use_gpu.setEnabled(False)
            self.use_gpu.setText("Use the GPU  ·  no compatible GPU found")
        form.addRow("Processing", self.use_gpu)
        layout.addLayout(form)

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

    def _choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a Cellpose model", self.model_path.text(), "All files (*)"
        )
        if path:
            self.model_path.setText(path)

    def _save_and_close(self) -> None:
        self.store.set_setting("export_dir", self.export_dir.text())
        self.store.set_setting("model_path", self.model_path.text())
        self.store.set_setting("use_gpu", self.use_gpu.isChecked())
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

        model = resources.bundled_model_path()
        lines = [
            f"Version {app_meta.APP_VERSION}",
            f"Cellpose {cellpose_version()}",
            f"Model: {Path(model).name if model else 'not bundled'}",
            f"GPU: {'available' if gpu_available() else 'not available, using the CPU'}",
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
