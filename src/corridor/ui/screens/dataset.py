"""The dataset screen: what Corridor understood, and one button.

Everything shown here was read from the file. The researcher is asked for
nothing the microscope already recorded, and told clearly about anything it
did not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...core.config import RunConfig
from ...core.imaging import StackMetadata
from ..theme import PALETTE, SPACE
from ..widgets.advanced import AdvancedPanel
from ..widgets.common import (
    Badge,
    Field,
    Metric,
    StackedField,
    Spinner,
    divider,
    ghost_button,
    label,
    primary_button,
)
from ..widgets.image_canvas import ImageCanvas


class DatasetScreen(QWidget):
    """Preview, interpreted metadata, and the analyse action."""

    back_requested = Signal()
    analyse_requested = Signal()
    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.metadata: StackMetadata | None = None
        self.config = RunConfig()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        root.addWidget(divider())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.canvas = ImageCanvas()
        self.canvas.layers.masks = False
        self.canvas.layers.centroids = False
        self.canvas.layers.trails = False
        self.canvas.layers.labels = False
        body.addWidget(self.canvas, 1)

        body.addWidget(divider(vertical=True))
        body.addWidget(self._build_side_panel())
        root.addLayout(body, 1)

    # ----------------------------------------------------------------- header
    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(62)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(SPACE["md"], 0, SPACE["lg"], 0)
        layout.setSpacing(SPACE["md"])

        back = ghost_button("", "back", self.back_requested.emit)
        back.setFixedWidth(38)
        back.setToolTip("Back")

        self.title = label("", "subtitle")
        self.subtitle = label("", "tertiary")
        text = QVBoxLayout()
        text.setSpacing(0)
        text.addWidget(self.title)
        text.addWidget(self.subtitle)

        self.advanced_button = ghost_button("Advanced", "settings")
        self.advanced_button.setCheckable(True)
        self.advanced_button.toggled.connect(self._toggle_advanced)

        layout.addWidget(back)
        layout.addLayout(text)
        layout.addStretch(1)
        layout.addWidget(self.advanced_button)
        return header

    # ------------------------------------------------------------- side panel
    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("SidePanel")
        panel.setFixedWidth(384)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        self._panel_layout = QVBoxLayout(content)
        self._panel_layout.setContentsMargins(
            SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["xl"]
        )
        self._panel_layout.setSpacing(SPACE["lg"])

        # Headline numbers.
        metrics = QHBoxLayout()
        metrics.setSpacing(SPACE["xl"])
        self.metric_frames = Metric("frames")
        self.metric_size = Metric("pixels")
        self.metric_duration = Metric("duration")
        metrics.addWidget(self.metric_frames)
        metrics.addWidget(self.metric_size)
        metrics.addWidget(self.metric_duration)
        metrics.addStretch(1)
        self._panel_layout.addLayout(metrics)
        self._panel_layout.addWidget(divider())

        self.field_interval = StackedField("Frame interval")
        self.field_pixel = StackedField("Pixel size")
        self.field_axes = Field("Layout")
        self.field_source = Field("Original frames")
        self.field_model = Field("Model")
        for field in (
            self.field_interval, self.field_pixel, self.field_axes,
            self.field_source, self.field_model,
        ):
            self._panel_layout.addWidget(field)

        self.notes = label("", "tertiary")
        self.notes.setWordWrap(True)
        self._panel_layout.addWidget(self.notes)

        self.warning_badge = Badge("", "warning")
        self.warning_badge.hide()
        self._panel_layout.addWidget(self.warning_badge)

        # Advanced parameters, hidden until asked for.
        self.advanced = AdvancedPanel()
        self.advanced.hide()
        self._advanced_divider = divider()
        self._advanced_divider.hide()
        self._panel_layout.addWidget(self._advanced_divider)
        self._panel_layout.addWidget(self.advanced)

        self._panel_layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # The action sits at the bottom of the panel, always visible.
        outer.addWidget(divider())
        action = QWidget()
        action_layout = QVBoxLayout(action)
        action_layout.setContentsMargins(
            SPACE["xl"], SPACE["lg"], SPACE["xl"], SPACE["xl"]
        )
        action_layout.setSpacing(SPACE["sm"])

        self.analyse_button = primary_button("Analyse", self.analyse_requested.emit)
        self.analyse_button.setMinimumHeight(44)
        action_layout.addWidget(self.analyse_button)

        self.status_row = QWidget()
        status_layout = QHBoxLayout(self.status_row)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(SPACE["sm"])
        self.spinner = Spinner()
        self.spinner.hide()
        self.status_label = label("", "secondary")
        self.cancel_button = ghost_button("Stop", "", self.cancel_requested.emit)
        self.cancel_button.hide()
        status_layout.addWidget(self.spinner)
        status_layout.addWidget(self.status_label, 1)
        status_layout.addWidget(self.cancel_button)
        self.status_row.hide()
        action_layout.addWidget(self.status_row)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.hide()
        action_layout.addWidget(self.progress)

        outer.addWidget(action)
        return panel

    # ------------------------------------------------------------------ state
    def set_dataset(
        self, metadata: StackMetadata, stack: np.ndarray, config: RunConfig
    ) -> None:
        self.metadata = metadata
        self.config = config
        self.title.setText(metadata.path.name)
        self.subtitle.setText(str(metadata.path.parent))
        self.subtitle.setToolTip(str(metadata.path))

        self.canvas.set_stack(stack)
        self.canvas.fit_to_view()

        self.metric_frames.set_value(str(metadata.n_frames))
        self.metric_size.set_value(f"{metadata.width}×{metadata.height}")
        duration = metadata.duration_min
        self.metric_duration.set_value(
            f"{duration / 60:.1f} h" if duration and duration >= 90
            else (f"{duration:.0f} min" if duration else "—")
        )

        from ...core.imaging import source_label

        interval = metadata.frame_interval_min
        self.field_interval.set_value(
            f"{interval.value:.6g} min" if interval.known else "unknown",
            source_label(interval.source),
        )
        pixel = metadata.pixel_size_um
        self.field_pixel.set_value(
            f"{pixel.value:.6g} µm/px" if pixel.known else "unknown",
            source_label(pixel.source),
        )
        self.field_axes.set_value(f"{metadata.axes_raw} · {metadata.axes_interpretation}")
        if metadata.source_frames:
            self.field_source.set_value(
                f"{metadata.source_frames[0]}–{metadata.source_frames[-1]}"
                f" of {metadata.source_frame_total}"
            )
            self.field_source.show()
        else:
            self.field_source.hide()

        model = config.segmentation.model_path
        self.field_model.set_value(
            Path(model).name if model else "built-in " + config.segmentation.builtin_model,
            model or "",
        )

        self.notes.setText("\n".join(f"· {note}" for note in metadata.notes))
        self.notes.setVisible(bool(metadata.notes))

        missing = []
        if not pixel.known:
            missing.append("pixel size")
        if not interval.known:
            missing.append("frame interval")
        if missing:
            self.warning_badge.setText(
                f"Set the {' and '.join(missing)} under Advanced to get physical units"
            )
            self.warning_badge.show()
        else:
            self.warning_badge.hide()

        self.advanced.set_config(config, metadata)

    def _toggle_advanced(self, shown: bool) -> None:
        self.advanced.setVisible(shown)
        self._advanced_divider.setVisible(shown)

    # ----------------------------------------------------------------- status
    def set_busy(self, busy: bool, message: str = "") -> None:
        self.analyse_button.setEnabled(not busy)
        self.advanced.setEnabled(not busy)
        self.status_row.setVisible(busy)
        self.cancel_button.setVisible(busy)
        self.progress.setVisible(busy)
        if busy:
            self.spinner.start()
            self.status_label.setText(message)
        else:
            self.spinner.stop()
            self.progress.setValue(0)

    def set_stage(self, name: str, detail: str = "") -> None:
        text = name if not detail else f"{name} — {detail}"
        self.status_label.setText(text)
        self.progress.setRange(0, 0)  # indeterminate until steps arrive

    def set_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self.progress.setRange(0, 0)
            return
        self.progress.setRange(0, total)
        self.progress.setValue(done)
