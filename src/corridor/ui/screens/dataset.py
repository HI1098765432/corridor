"""The dataset screen: what Corridor understood, and one button.

Everything shown here was read from the file. The researcher is asked for
nothing the microscope already recorded, and told clearly about anything it
did not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.config import RunConfig
from ...core.imaging import StackMetadata
from ..model_status import registered_model
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


def preview_stack(stack: np.ndarray) -> np.ndarray:
    """What the 2-D preview canvas shows: the stack, or its Z projection.

    A ``TZYX`` stack is previewed as its maximum projection over Z, frame by
    frame, which shows every cell at once; the results screen then reviews it
    plane by plane in the orthogonal viewer.
    """
    if stack is not None and stack.ndim == 4:
        return stack.max(axis=1)
    return stack


def _depth(metadata, stack: np.ndarray | None) -> int | None:
    """The number of Z slices, or None for a 2-D time-lapse."""
    axes = str(getattr(metadata, "axes", "") or "")
    if stack is not None and stack.ndim == 4:
        return int(stack.shape[1])
    if "Z" in axes.upper():
        shape = getattr(metadata, "shape", None) or ()
        index = axes.upper().index("Z")
        if index < len(shape):
            return int(shape[index])
    return None


class DatasetScreen(QWidget):
    """Preview, interpreted metadata, and the analyse action."""

    back_requested = Signal()
    analyse_requested = Signal()
    cancel_requested = Signal()
    #: Measure and track a label image instead of segmenting (``True``), or
    #: go back to the validated model (``False``).
    labels_requested = Signal(bool)

    #: The panel grows when the advanced parameters are shown, so the controls
    #: have room instead of being clipped.
    PANEL_WIDTH = 384
    PANEL_WIDTH_ADVANCED = 492

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

        # During analysis the preview is replaced by a 4-D block of the movie
        # (x, y, time-as-depth) that fills in frame by frame as the engine works
        # through them -- the "meshing" building up as the hard code runs.
        self.block_view = QLabel()
        self.block_view.setAlignment(Qt.AlignCenter)
        self.block_view.setMinimumSize(240, 240)
        self.block_view.setStyleSheet(f"background: {PALETTE.canvas};")
        self._view_stack = QStackedWidget()
        self._view_stack.addWidget(self.canvas)       # index 0: preview
        self._view_stack.addWidget(self.block_view)   # index 1: 4-D block
        body.addWidget(self._view_stack, 1)
        #: Downscaled grayscale slices, prepared once per run for a cheap redraw.
        self._block_slices: list[np.ndarray] | None = None

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
        self._panel = panel
        panel.setFixedWidth(self.PANEL_WIDTH)
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
        # The only route for 3-D data (no 3-D model is validated), and the way
        # to analyse a segmentation made elsewhere. ImportConfig.labels_path.
        self.labels_button = ghost_button("Use a label image…", "", self._labels_clicked)
        self.labels_button.setToolTip(
            "Measure and track an existing label image (one integer label per "
            "cell, same frames as this file) instead of segmenting with the model."
        )
        self._labels_set = False
        self._panel_layout.addWidget(self.labels_button, 0, Qt.AlignLeft)

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

        self.canvas.set_stack(preview_stack(stack))
        self.canvas.fit_to_view()
        self._prepare_block(preview_stack(stack))

        self.metric_frames.set_value(str(metadata.n_frames))
        depth = _depth(metadata, stack)
        self.metric_size.set_value(
            f"{metadata.width}×{metadata.height}" + (f"×{depth}" if depth else "")
        )
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
        axes_parts = [
            str(part)
            for part in (
                getattr(metadata, "axes", None),
                getattr(metadata, "axes_raw", None),
                getattr(metadata, "axes_interpretation", None),
            )
            if part
        ]
        # De-duplicated: a canonical order equal to the file's own reads once.
        self.field_axes.set_value("  ·  ".join(dict.fromkeys(axes_parts)) or "—")
        if metadata.source_frames:
            self.field_source.set_value(
                f"{metadata.source_frames[0]}–{metadata.source_frames[-1]}"
                f" of {metadata.source_frame_total}"
            )
            self.field_source.show()
        else:
            self.field_source.hide()

        self.show_model(config, "3D" if depth else "2D")

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

    def show_model(self, config: RunConfig, dimensionality: str) -> None:
        """Which segmentation will run: the registered model, or imported labels.

        Read from the registry, not from the configuration: the configuration
        no longer chooses a model (contract §2), and a stale ``model_path`` in
        a saved project must never be displayed as if it were used. Nothing
        is hashed here; the check happens when Analyse is pressed.
        """
        labels_path = getattr(getattr(config, "import_", None), "labels_path", None)
        self._labels_set = bool(labels_path)
        self.labels_button.setText(
            "Segment with the model instead" if labels_path else "Use a label image…"
        )
        if labels_path:
            self.field_model.set_value(
                f"imported labels: {Path(labels_path).name}", str(labels_path)
            )
            return
        status = registered_model(dimensionality)
        if status.model_id:
            self.field_model.set_value(
                f"{status.model_id} {status.model_version}",
                f"SHA-256 {status.sha256}\nVerified against this checksum before the analysis starts.",
            )
        else:
            self.field_model.set_value("none validated for " + dimensionality, status.message)

    def _labels_clicked(self) -> None:
        self.labels_requested.emit(not self._labels_set)

    def _toggle_advanced(self, shown: bool) -> None:
        self.advanced.setVisible(shown)
        self._advanced_divider.setVisible(shown)
        self._panel.setFixedWidth(
            self.PANEL_WIDTH_ADVANCED if shown else self.PANEL_WIDTH
        )

    # ----------------------------------------------------------------- status
    def set_busy(self, busy: bool, message: str = "") -> None:
        self.analyse_button.setEnabled(not busy)
        self.labels_button.setEnabled(not busy)
        self.advanced.setEnabled(not busy)
        self.status_row.setVisible(busy)
        self.cancel_button.setVisible(busy)
        self.progress.setVisible(busy)
        if busy:
            self.spinner.start()
            self.status_label.setText(message)
            self._draw_block(0.0)
            self._view_stack.setCurrentWidget(self.block_view)
        else:
            self.spinner.stop()
            self.progress.setValue(0)
            self._view_stack.setCurrentWidget(self.canvas)

    def set_stage(self, name: str, detail: str = "") -> None:
        text = name if not detail else f"{name} — {detail}"
        self.status_label.setText(text)
        self.progress.setRange(0, 0)  # indeterminate until steps arrive
        # Stages after segmentation have no per-frame progress; show the block
        # fully meshed once the frames have been worked through.
        if name and name.lower() not in ("reading", "segmenting", "starting"):
            self._draw_block(1.0)

    def set_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self.progress.setRange(0, 0)
            return
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self._draw_block(done / total)

    # ------------------------------------------------------------------- 4-D block
    def _prepare_block(self, flat: np.ndarray, max_px: int = 150, max_frames: int = 40) -> None:
        """Cache small grayscale slices of the movie for a cheap block redraw.

        ``flat`` is the T×Y×X preview (Z already projected). Frames are capped
        and downscaled so compositing the isometric block stays instant on the
        UI thread even for a long stack."""
        try:
            if flat is None or flat.ndim != 3 or flat.shape[0] == 0:
                self._block_slices = None
                return
            n = flat.shape[0]
            idx = np.linspace(0, n - 1, min(n, max_frames)).round().astype(int)
            lo, hi = np.percentile(flat, [1, 99])
            scale = max_px / max(flat.shape[1], flat.shape[2])
            sh = max(1, int(flat.shape[1] * scale)); sw = max(1, int(flat.shape[2] * scale))
            ys = np.linspace(0, flat.shape[1] - 1, sh).astype(int)
            xs = np.linspace(0, flat.shape[2] - 1, sw).astype(int)
            slices = []
            for t in idx:
                g = np.clip((flat[t][np.ix_(ys, xs)] - lo) / (hi - lo + 1e-6), 0, 1)
                slices.append((g * 255).astype(np.uint8))
            self._block_slices = slices
        except Exception:  # noqa: BLE001 - a missing preview must never break a run
            self._block_slices = None

    def _draw_block(self, frac: float) -> None:
        """Composite the cached slices into an isometric x-y-time block with the
        first ``frac`` of frames drawn solid (meshed) and the rest dim."""
        slices = self._block_slices
        if not slices:
            return
        try:
            n = len(slices); sh, sw = slices[0].shape
            dx, dy = max(4, sw // 12), max(3, sh // 14)
            H = sh + dy * (n - 1); W = sw + dx * (n - 1)
            canvas = np.zeros((H, W, 3), np.float32)
            n_done = max(1, int(round(n * max(0.0, min(1.0, frac)))))
            for t in range(n - 1, -1, -1):
                ox, oy = dx * t, dy * (n - 1 - t)
                g = slices[t].astype(np.float32)
                tile = np.stack([g, g, g], -1)
                if t < n_done:
                    tile[..., 1] += g * 0.25          # processed frames tint green
                else:
                    tile *= 0.3                        # pending frames dim
                canvas[oy:oy + sh, ox:ox + sw] = np.maximum(canvas[oy:oy + sh, ox:ox + sw], tile)
            arr = np.clip(canvas, 0, 255).astype(np.uint8)
            img = QImage(arr.data, W, H, 3 * W, QImage.Format_RGB888).copy()
            pm = QPixmap.fromImage(img)
            target = self.block_view.size()
            if target.width() > 10 and target.height() > 10:
                pm = pm.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.block_view.setPixmap(pm)
        except Exception:  # noqa: BLE001 - never let the preview break the analysis
            pass
