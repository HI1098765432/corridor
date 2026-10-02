"""The results workspace.

Priorities, in order: the microscopy is the object; overlays must be
switchable without hunting; a questionable track must be one click from the
frame where it went wrong.

2.0 changes what is reported, not how it is reviewed:

*   **Speeds are per hour** and derive from the canonical µm/min value
    (contract §6). The detail panel adds MTrackJ's Len (cumulative path) and
    D2S (net distance from the start), the MSD exponent, and a log-log MSD
    plot of the selected track.
*   **Exports are named for what they hold**: one track as CSV or XLSX, all
    tracks, all summaries, the MSD curves, or the whole bundle. A reference
    point set on the image is stored with the analysis and fills D2R.
*   **No axis.** Lanes are an optional overlay; nothing defaults to vertical.
*   **3-D results** are shown in an orthogonal viewer (XY / XZ / YZ) in place
    of the 2-D canvas, driven by the same timeline.
*   **The provenance panel never prints "None"** (critique C5): every row is
    composed only from the parts that exist, and a row with none shows an em
    dash.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.config import ENSEMBLE_LABELS, NORMALISATION_LABELS
from ..analysis_view import (
    DASH,
    dimensionality_of,
    finite,
    first_present,
    fmt_number,
    has_msd,
    load_reference_point,
    msd_rows_for,
    msd_series,
    path_metrics,
    save_reference_point,
    schema_version_of,
    speed_um_per_hr,
)
from ..icons import icon, pixmap
from ..lanes import lanes_for_manifest
from ..theme import PALETTE, SPACE, track_color
from ..widgets.common import (
    Badge,
    Field,
    Metric,
    divider,
    ghost_button,
    label,
    layer_toggle,
    primary_button,
)
from ..widgets.image_canvas import ImageCanvas
from ..widgets.msd_plot import EMPTY_TEXT as MSD_EMPTY_TEXT
from ..widgets.msd_plot import MsdPlot
from ..widgets.ortho_viewer import OrthoViewer
from ..workers import (
    EXPORT_BUNDLE,
    EXPORT_LABELS,
    EXPORT_MSD_CSV,
    EXPORT_SUMMARIES_CSV,
    EXPORT_TRACK_CSV,
    EXPORT_TRACK_XLSX,
    EXPORT_TRACKS_CSV,
    TRACK_EXPORTS,
)

SEVERITY_TONE = {"critical": "danger", "warning": "warning", "info": "neutral"}

#: The order of the Export menu: the selected track first, because that is
#: what a reviewer is looking at, then the whole analysis.
EXPORT_MENU = (
    EXPORT_TRACK_CSV,
    EXPORT_TRACK_XLSX,
    None,
    EXPORT_TRACKS_CSV,
    EXPORT_SUMMARIES_CSV,
    EXPORT_MSD_CSV,
    None,
    EXPORT_BUNDLE,
)


class Sparkline(QWidget):
    """Speed against time for one track, small enough to sit in a panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(58)
        self._values: list[tuple[float, float]] = []
        self._colour = PALETTE.accent
        self._marker: float | None = None

    def set_series(
        self, values: list[tuple[float, float]], colour: str, marker: float | None = None
    ) -> None:
        self._values = [(x, y) for x, y in values if y is not None and math.isfinite(y)]
        self._colour = colour
        self._marker = marker
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(1, 4, -1, -4)
        painter.fillRect(QRectF(self.rect()), QColor(PALETTE.surface_sunken))
        if len(self._values) < 2:
            painter.setPen(QColor(PALETTE.text_tertiary))
            painter.drawText(rect, Qt.AlignCenter, "not enough observations")
            painter.end()
            return

        xs = [v[0] for v in self._values]
        ys = [v[1] for v in self._values]
        x0, x1 = min(xs), max(xs)
        y1 = max(ys)
        y0 = 0.0
        if x1 <= x0:
            x1 = x0 + 1
        if y1 <= y0:
            y1 = y0 + 1

        def to_point(x: float, y: float) -> tuple[float, float]:
            return (
                rect.left() + (x - x0) / (x1 - x0) * rect.width(),
                rect.bottom() - (y - y0) / (y1 - y0) * rect.height(),
            )

        pen = QPen(QColor(self._colour), 1.8)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        previous = None
        for x, y in self._values:
            point = to_point(x, y)
            if previous is not None:
                painter.drawLine(previous[0], previous[1], point[0], point[1])
            previous = point

        if self._marker is not None and x0 <= self._marker <= x1:
            marker_x = to_point(self._marker, 0)[0]
            painter.setPen(QPen(QColor(PALETTE.text_tertiary), 1, Qt.DashLine))
            painter.drawLine(marker_x, rect.top(), marker_x, rect.bottom())
        painter.end()


class TimelineBar(QWidget):
    """Frame control: scrub, step, play."""

    frame_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setProperty("role", "statusbar")
        self._playing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACE["lg"], SPACE["sm"], SPACE["lg"], SPACE["sm"])
        layout.setSpacing(SPACE["md"])

        self.play_button = ghost_button("", "play", self.toggle_play)
        self.play_button.setFixedWidth(38)
        self.play_button.setToolTip("Play (Space)")
        back = ghost_button("", "step-back", lambda: self.step(-1))
        back.setFixedWidth(34)
        forward = ghost_button("", "step-forward", lambda: self.step(1))
        forward.setFixedWidth(34)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.frame_changed.emit)

        self.readout = label("", "secondary")
        self.readout.setMinimumWidth(190)
        self.readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(back)
        layout.addWidget(self.play_button)
        layout.addWidget(forward)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.readout)

    def configure(self, n_frames: int) -> None:
        self.slider.setRange(0, max(0, n_frames - 1))
        self.slider.setEnabled(n_frames > 1)
        self.play_button.setEnabled(n_frames > 1)

    def set_frame(self, index: int) -> None:
        if self.slider.value() != index:
            self.slider.blockSignals(True)
            self.slider.setValue(index)
            self.slider.blockSignals(False)

    def set_readout(self, text: str) -> None:
        self.readout.setText(text)

    def step(self, delta: int) -> None:
        self.slider.setValue(
            max(self.slider.minimum(), min(self.slider.maximum(), self.slider.value() + delta))
        )

    def toggle_play(self) -> None:
        self._playing = not self._playing
        if self._playing:
            self._timer.start(280)
            self.play_button.setIcon(icon("pause", PALETTE.text_secondary, 18))
        else:
            self._timer.stop()
            self.play_button.setIcon(icon("play", PALETTE.text_secondary, 18))

    def stop(self) -> None:
        if self._playing:
            self.toggle_play()

    def _advance(self) -> None:
        if self.slider.value() >= self.slider.maximum():
            self.slider.setValue(self.slider.minimum())
        else:
            self.step(1)


def _join(parts: list[str | None], separator: str = "  ·  ") -> str | None:
    """Join the parts that exist; None when none do (shown as an em dash)."""
    present = [p for p in parts if p]
    return separator.join(present) if present else None


def _duration_text(minutes: Any) -> str:
    value = finite(minutes)
    if value is None:
        return DASH
    return f"{value / 60:.1f} h" if value >= 90 else f"{value:.0f} min"


class ResultsScreen(QWidget):
    """Image, overlays, timeline, inspector."""

    back_requested = Signal()
    #: One of the ``workers.EXPORT_*`` keys.
    export_requested = Signal(str)
    napari_requested = Signal()
    open_folder_requested = Signal()
    #: The reference point changed: a tuple (x, y[, z]) in px, or None.
    reference_point_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.analysis = None
        self._selected_track: int | None = None
        self._three_d = False
        self.reference_point_px: tuple[float, ...] | None = None
        self.reference_error: str = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # The views exist before the toolbar, which wires its toggles to them.
        self.canvas = ImageCanvas()
        self.canvas.track_clicked.connect(self._on_canvas_click)
        self.canvas.frame_changed.connect(self._on_frame_changed)
        self.canvas.point_picked.connect(self._reference_picked)
        self.ortho = OrthoViewer()
        self.ortho.track_clicked.connect(self._on_canvas_click)
        self.ortho.frame_changed.connect(self._on_frame_changed)
        self.ortho.point_picked.connect(self._reference_picked)

        root.addWidget(self._build_header())
        root.addWidget(divider())
        root.addWidget(self._build_toolbar())
        root.addWidget(divider())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)

        canvas_holder = QWidget()
        canvas_layout = QVBoxLayout(canvas_holder)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.canvas)
        self.view_stack.addWidget(self.ortho)
        canvas_layout.addWidget(self.view_stack, 1)
        self.timeline = TimelineBar()
        self.timeline.frame_changed.connect(self._timeline_moved)
        canvas_layout.addWidget(divider())
        canvas_layout.addWidget(self.timeline)

        splitter.addWidget(canvas_holder)
        splitter.addWidget(self._build_inspector())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 380])
        root.addWidget(splitter, 1)
        self._update_export_actions()

    # ----------------------------------------------------------------- header
    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(62)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(SPACE["md"], 0, SPACE["lg"], 0)
        layout.setSpacing(SPACE["sm"])

        back = ghost_button("", "back", self.back_requested.emit)
        back.setFixedWidth(38)
        back.setToolTip("Back")

        self.title = label("", "subtitle")
        self.subtitle = label("", "tertiary")
        text = QVBoxLayout()
        text.setSpacing(0)
        text.addWidget(self.title)
        text.addWidget(self.subtitle)

        self.warning_badge = Badge("", "warning")
        self.warning_badge.hide()

        self.napari_button = ghost_button("Napari", "layers", self.napari_requested.emit)
        self.napari_button.setToolTip("Open this result in Napari for deep inspection")
        self.folder_button = ghost_button("", "folder", self.open_folder_requested.emit)
        self.folder_button.setFixedWidth(38)
        self.folder_button.setToolTip("Open the results folder")

        self.export_button = primary_button("Export")
        self.export_button.setIcon(icon("export", "#FFFFFF", 16))
        self.export_menu = QMenu(self.export_button)
        self.export_actions: dict[str, QAction] = {}
        for kind in EXPORT_MENU:
            if kind is None:
                self.export_menu.addSeparator()
                continue
            action = QAction(EXPORT_LABELS[kind], self.export_menu)
            action.triggered.connect(lambda _checked=False, k=kind: self.export_requested.emit(k))
            self.export_menu.addAction(action)
            self.export_actions[kind] = action
        self.export_button.setMenu(self.export_menu)

        layout.addWidget(back)
        layout.addLayout(text)
        layout.addStretch(1)
        layout.addWidget(self.warning_badge)
        layout.addSpacing(SPACE["sm"])
        layout.addWidget(self.napari_button)
        layout.addWidget(self.folder_button)
        layout.addWidget(self.export_button)
        return header

    # ---------------------------------------------------------------- toolbar
    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(48)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(SPACE["lg"], SPACE["xs"], SPACE["lg"], SPACE["xs"])
        layout.setSpacing(SPACE["xs"])

        self.toggle_image = layer_toggle("Image", "image", True)
        self.toggle_masks = layer_toggle("Outlines", "layers", True)
        self.toggle_points = layer_toggle("Centroids", "target", True)
        self.toggle_trails = layer_toggle("Tracks", "route", True)
        self.toggle_labels = layer_toggle("IDs", "hash", True)
        self.toggle_lanes = layer_toggle("Lanes", "grid", False)
        self.toggle_lanes.setToolTip("Channel lanes recorded by this run")

        for button in (
            self.toggle_image, self.toggle_masks, self.toggle_points,
            self.toggle_trails, self.toggle_labels, self.toggle_lanes,
        ):
            button.toggled.connect(self._sync_layers)
            layout.addWidget(button)

        layout.addSpacing(SPACE["md"])
        self.reference_button = layer_toggle("Set reference point", "plus", False)
        self.reference_button.setToolTip(
            "Click the image to set the point that distances to reference (D2R) "
            "are measured from in exports. Stored with this analysis."
        )
        self.reference_button.toggled.connect(self._reference_mode_toggled)
        layout.addWidget(self.reference_button)
        self.clear_reference_button = ghost_button("", "close", self.clear_reference_point)
        self.clear_reference_button.setFixedWidth(34)
        self.clear_reference_button.setToolTip("Remove the reference point")
        self.clear_reference_button.hide()
        layout.addWidget(self.clear_reference_button)

        layout.addStretch(1)
        reset = ghost_button("", "zoom-reset", self.canvas_reset)
        reset.setFixedWidth(38)
        reset.setToolTip("Fit to window (0)")
        layout.addWidget(reset)
        return bar

    def canvas_reset(self) -> None:
        self.canvas.fit_to_view()

    def _sync_layers(self) -> None:
        for layers in (self.canvas.layers, self.ortho.layers):
            layers.image = self.toggle_image.isChecked()
            layers.masks = self.toggle_masks.isChecked()
            layers.centroids = self.toggle_points.isChecked()
            layers.trails = self.toggle_trails.isChecked()
            layers.labels = self.toggle_labels.isChecked()
            layers.lanes = self.toggle_lanes.isChecked()
        self.canvas.update()
        if self._three_d:
            self.ortho.fit_to_view()

    # -------------------------------------------------------------- inspector
    def _build_inspector(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("Inspector")
        panel.setMinimumWidth(340)
        panel.setMaximumWidth(500)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_track_tab(), "Tracks")
        self.tabs.addTab(self._build_checks_tab(), "Checks")
        self.tabs.addTab(self._build_run_tab(), "Run")
        outer.addWidget(self.tabs, 1)
        return panel

    def _build_track_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["sm"])
        layout.setSpacing(SPACE["md"])

        summary = QHBoxLayout()
        summary.setSpacing(SPACE["xl"])
        self.metric_tracks = Metric("tracks")
        self.metric_detections = Metric("detections")
        self.metric_speed = Metric("mean µm/h")
        summary.addWidget(self.metric_tracks)
        summary.addWidget(self.metric_detections)
        summary.addWidget(self.metric_speed)
        summary.addStretch(1)
        layout.addLayout(summary)
        layout.addWidget(divider())

        split = QSplitter(Qt.Vertical)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(1)
        self.track_list = QListWidget()
        self.track_list.setMinimumHeight(90)
        self.track_list.currentItemChanged.connect(self._on_track_selected)
        split.addWidget(self.track_list)

        self.detail_box = QWidget()
        detail_layout = QVBoxLayout(self.detail_box)
        detail_layout.setContentsMargins(0, SPACE["sm"], 0, 0)
        detail_layout.setSpacing(SPACE["xs"])
        title_row = QHBoxLayout()
        title_row.setSpacing(SPACE["sm"])
        self.detail_title = label("", "subtitle")
        title_row.addWidget(self.detail_title)
        title_row.addStretch(1)
        self.export_track_button = ghost_button(
            "Export track", "export", lambda: self.export_requested.emit(EXPORT_TRACK_CSV)
        )
        self.export_track_button.setToolTip(
            "Every observation of this track as CSV, with Len, D2S, D2P and, "
            "when a reference point is set, D2R"
        )
        title_row.addWidget(self.export_track_button)
        detail_layout.addLayout(title_row)
        self.sparkline = Sparkline()
        self.sparkline.setToolTip("Speed (µm/h) against frame")
        detail_layout.addWidget(self.sparkline)
        self.detail_fields: dict[str, Field] = {}
        for key, name, hint in (
            ("frames", "Frames", ""),
            ("observations", "Observations", ""),
            ("gaps", "Missing frames", ""),
            ("duration", "Duration", ""),
            ("len", "Len (path length)", "Cumulative path length from the first point"),
            ("d2s", "D2S (net from start)", "Straight-line distance from the first point to the last"),
            ("mean", "Mean speed", "Mean of the per-step speeds"),
            ("max", "Peak speed", "Largest per-step speed"),
            ("straight", "Straightness", "D2S divided by Len"),
            ("alpha", "MSD α", "Exponent of a power-law fit, MSD ~ t^α"),
        ):
            field = Field(name)
            if hint:
                field._name.setToolTip(hint)
            self.detail_fields[key] = field
            detail_layout.addWidget(field)
        msd_caption = label("MSD against lag, log-log  ·  point size = pairs averaged", "tertiary")
        msd_caption.setWordWrap(True)
        detail_layout.addWidget(msd_caption)
        self.msd_plot = MsdPlot()
        detail_layout.addWidget(self.msd_plot)
        self.detail_flags = label("", "tertiary")
        self.detail_flags.setWordWrap(True)
        detail_layout.addWidget(self.detail_flags)
        detail_layout.addStretch(1)
        self.detail_box.hide()

        detail_scroll = QScrollArea()
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        detail_scroll.setWidget(self.detail_box)
        split.addWidget(detail_scroll)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)
        layout.addWidget(split, 1)
        return page

    def _build_checks_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["lg"])
        layout.setSpacing(SPACE["md"])
        self.checks_summary = label("", "secondary")
        self.checks_summary.setWordWrap(True)
        layout.addWidget(self.checks_summary)
        self.checks_list = QListWidget()
        self.checks_list.setWordWrap(True)
        self.checks_list.itemActivated.connect(self._on_check_activated)
        self.checks_list.itemClicked.connect(self._on_check_activated)
        layout.addWidget(self.checks_list, 1)

        # An empty bordered box is not an empty state. When there is nothing to
        # review, say so once and leave the space quiet.
        self.checks_empty = QWidget()
        empty_layout = QVBoxLayout(self.checks_empty)
        empty_layout.setContentsMargins(0, SPACE["3xl"], 0, 0)
        empty_layout.setSpacing(SPACE["sm"])
        empty_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        glyph = QLabel()
        glyph.setPixmap(pixmap("check", PALETTE.text_tertiary, 30))
        glyph.setAlignment(Qt.AlignCenter)
        message = label("Nothing needs attention", "secondary")
        message.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(glyph)
        empty_layout.addWidget(message)
        layout.addWidget(self.checks_empty, 1)
        return page

    #: Run-tab rows, in display order.
    RUN_ROWS = (
        ("file", "File"),
        ("shape", "Stack"),
        ("axes", "Layout"),
        ("dimensionality", "Dimensionality"),
        ("source_frames", "Original frames"),
        ("pixel", "Pixel size"),
        ("interval", "Frame interval"),
        ("z_step", "Z step"),
        ("schema", "Output schema"),
        ("model", "Model"),
        ("model_version", "Model version"),
        ("model_hash", "Model checksum"),
        ("cellpose", "Cellpose"),
        ("thresholds", "Thresholds"),
        ("min_extent", "Smallest object"),
        ("normalisation", "Normalisation"),
        ("detection_effort", "Detection effort"),
        ("raw_kept", "Instances raw → kept"),
        ("geometry", "Channel walls applied"),
        ("max_gap", "Allowed disappearance"),
        ("max_speed", "Speed limit"),
        ("elapsed", "Took"),
    )

    def _build_run_tab(self) -> QWidget:
        page = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["lg"])
        layout.setSpacing(SPACE["sm"])
        self.run_fields: dict[str, Field] = {}
        for key, name in self.RUN_ROWS:
            field = Field(name)
            self.run_fields[key] = field
            layout.addWidget(field)
        layout.addStretch(1)
        scroll.setWidget(inner)
        holder = QVBoxLayout(page)
        holder.setContentsMargins(0, 0, 0, 0)
        holder.addWidget(scroll)
        return page

    # --------------------------------------------------------------- loading
    @property
    def selected_track(self) -> int | None:
        return self._selected_track

    @property
    def three_d(self) -> bool:
        return self._three_d

    def active_view(self):
        """The view showing this analysis: the 2-D canvas or the ortho viewer."""
        return self.ortho if self._three_d else self.canvas

    def load(self, analysis, stack: np.ndarray) -> None:
        self.analysis = analysis
        self._reset_selection()

        source = analysis.source_path
        self.title.setText(source.name if source else analysis.directory.name)
        self.subtitle.setText(str(source.parent) if source else "")

        dimensionality = dimensionality_of(analysis, stack)
        self._three_d = dimensionality == "3D" and stack is not None and stack.ndim == 4
        size = (float(stack.shape[-1]), float(stack.shape[-2])) if stack is not None else (0.0, 0.0)
        if self._three_d:
            self.canvas.clear()
            self.ortho.rows_for_frame = analysis.rows_for_frame
            self.ortho.selected_track = None
            self.ortho.set_volume(stack, analysis.masks, anisotropy=self._anisotropy(analysis))
            self.view_stack.setCurrentWidget(self.ortho)
        else:
            self.ortho.set_volume(None)
            self.canvas.rows_for_frame = analysis.rows_for_frame
            self.canvas.rows_for_track = analysis.rows_for_track
            self.canvas.set_stack(stack)
            self.canvas.set_masks(analysis.masks)
            self.canvas.selected_track = None
            self.view_stack.setCurrentWidget(self.canvas)

        lanes = lanes_for_manifest(analysis.manifest, size)
        self.canvas.lanes = lanes
        self.toggle_lanes.setEnabled(bool(lanes) and not self._three_d)
        self.toggle_lanes.setToolTip(
            f"{len(lanes)} channel lane(s) recorded by this run"
            if lanes
            else "This run recorded no channel lanes"
        )
        if not lanes:
            self.toggle_lanes.setChecked(False)

        self._set_reference(load_reference_point(analysis.directory), persist=False)
        if not self._three_d:
            self.canvas.fit_to_view()

        self.timeline.configure(stack.shape[0] if stack is not None else 0)
        self.timeline.set_frame(0)
        self._update_readout(0)

        self._populate_tracks()
        self._populate_checks()
        self._populate_run()
        self._sync_layers()

    @staticmethod
    def _anisotropy(analysis) -> float | None:
        """Z step over pixel size, when both are calibrated; else None."""
        cal = (analysis.manifest or {}).get("calibration") or {}
        z_step = finite(cal.get("z_step_um"))
        pixel = finite(cal.get("pixel_size_um"))
        if z_step and pixel and z_step > 0 and pixel > 0:
            return z_step / pixel
        return None

    def _reset_selection(self) -> None:
        """Clear everything that belonged to the previous dataset.

        Without this, the inspector keeps showing the last dataset's track
        statistics next to the new dataset's image, which is exactly the kind
        of stale result that makes a measurement untrustworthy.
        """
        self._selected_track = None
        self.canvas.selected_track = None
        self.ortho.selected_track = None
        self.track_list.clearSelection()
        self.detail_box.hide()
        self.detail_title.setText("")
        self.detail_flags.setText("")
        for field in self.detail_fields.values():
            field.set_value(DASH)
        self.sparkline.set_series([], PALETTE.accent, None)
        self.msd_plot.clear()
        self.warning_badge.hide()
        self.timeline.stop()
        self.reference_button.setChecked(False)
        self._update_export_actions()

    def _populate_tracks(self) -> None:
        analysis = self.analysis
        if analysis is None:
            return
        results = analysis.manifest.get("results", {}) or {}
        n_tracks = results.get("n_tracks")
        n_detections = results.get("n_detections")
        self.metric_tracks.set_value(str(n_tracks if n_tracks is not None else len(analysis.summaries)))
        self.metric_detections.set_value(
            str(n_detections if n_detections is not None else len(analysis.detections))
        )
        mean = speed_um_per_hr(results, "mean_speed")
        self.metric_speed.set_value(fmt_number(mean, 1))
        self.metric_speed.setToolTip("Mean of each track's mean speed, in µm/h")

        # Signals are blocked while the list is rebuilt: clearing a list that
        # had a current row makes Qt move "current" onto whichever item comes
        # next, which re-selected a track of the *new* analysis that nobody
        # clicked (measured: loading a second analysis after selecting track
        # 1 left track 2 selected).
        self.track_list.blockSignals(True)
        self.track_list.clear()
        self._fill_track_list(analysis)
        self.track_list.setCurrentItem(None)
        self.track_list.blockSignals(False)

    def _fill_track_list(self, analysis) -> None:
        for summary in sorted(
            analysis.summaries, key=lambda s: -(s.get("n_observations") or 0)
        ):
            track_id = int(summary.get("track_id", 0))
            observations = summary.get("n_observations") or 0
            speed = speed_um_per_hr(summary, "mean_speed")
            first, last = summary.get("first_frame"), summary.get("last_frame")
            pieces = []
            if first is not None and last is not None:
                pieces.append(f"frames {first}–{last}")
            pieces.append(f"{observations} obs")
            if speed is not None:
                pieces.append(f"{speed:.1f} µm/h")
            item = QListWidgetItem(f"Track {track_id}\n{'  ·  '.join(pieces)}")
            item.setData(Qt.UserRole, track_id)
            item.setForeground(QColor(PALETTE.text))
            chip = QPixmap(10, 10)
            chip.fill(QColor(track_color(track_id)))
            item.setIcon(chip)
            if summary.get("flags"):
                item.setToolTip(str(summary["flags"]))
            self.track_list.addItem(item)

    def _populate_checks(self) -> None:
        analysis = self.analysis
        if analysis is None:
            return
        self.checks_list.clear()
        issues = analysis.issues
        severe = [i for i in issues if i.get("severity") in ("critical", "warning")]
        self.checks_list.setVisible(bool(issues))
        self.checks_empty.setVisible(not issues)
        self.checks_summary.setVisible(bool(issues))
        if issues:
            self.checks_summary.setText(
                f"{len(severe)} to look at, {len(issues) - len(severe)} for information."
            )
        for issue in issues:
            title = issue.get("title") or issue.get("code")
            detail = issue.get("detail") or ""
            item = QListWidgetItem(f"{title}\n{detail}")
            item.setData(Qt.UserRole, issue)
            severity = issue.get("severity", "info")
            if severity == "critical":
                item.setForeground(QColor(PALETTE.danger))
            elif severity == "warning":
                item.setForeground(QColor(PALETTE.warning))
            else:
                item.setForeground(QColor(PALETTE.text_secondary))
            self.checks_list.addItem(item)

        if severe:
            self.warning_badge.setText(f"{len(severe)} to check")
            self.warning_badge.set_tone(
                "danger" if any(i.get("severity") == "critical" for i in severe) else "warning"
            )
            self.warning_badge.show()
        else:
            self.warning_badge.hide()

    def _populate_run(self) -> None:
        """Fill every provenance row, unconditionally.

        Every row is written on every call (a row left over from the previous
        analysis would describe the wrong run), and every row is composed only
        of the parts its manifest really has, so a missing key shows an em
        dash -- never "None px", "prob None" or "None frames" (critique C5).
        """
        analysis = self.analysis
        if analysis is None:
            return
        m = getattr(analysis, "manifest", None) or {}
        inp = m.get("input") or {}
        cal = m.get("calibration") or {}
        seg = m.get("segmentation") or {}
        trk = m.get("tracking") or {}
        env = m.get("environment") or {}
        res = m.get("results") or {}
        model = m.get("model") or {}
        geometry = m.get("channel_geometry")

        def put(key: str, value: Any, hint: str | None = "") -> None:
            field = self.run_fields.get(key)
            if field:
                text = DASH if value is None or value == "" else str(value)
                field.set_value(text, hint or "")

        def number(value: Any, spec: str = "g") -> str | None:
            v = finite(value)
            return None if v is None else format(v, spec)

        put("file", inp.get("name"), inp.get("path"))

        shape = inp.get("shape") or inp.get("shape_tyx") or []
        axes_order = inp.get("axes") or ("TYX" if inp.get("shape_tyx") else None)
        if shape:
            text = " × ".join(str(int(v)) for v in shape)
            put("shape", f"{text}  ({axes_order})" if axes_order else text)
        else:
            put("shape", None)

        put(
            "axes",
            _join(
                [
                    inp.get("axes"),
                    inp.get("axes_reported"),
                    inp.get("axes_interpretation"),
                ]
            ),
            f"axis order source: {inp['axes_source']}" if inp.get("axes_source") else "",
        )
        dimensionality = dimensionality_of(analysis) if m else None
        put("dimensionality", dimensionality)

        frames = inp.get("source_frames")
        total = inp.get("source_frame_total")
        put(
            "source_frames",
            (f"{frames[0]}–{frames[-1]}" + (f" of {total}" if total is not None else ""))
            if frames
            else None,
        )
        pixel = number(cal.get("pixel_size_um"), ".6g")
        source = cal.get("pixel_size_um_source")
        put("pixel", f"{pixel} µm/px" if pixel else ("unknown" if m else None),
            f"source: {source}" if source else "")
        interval = number(cal.get("frame_interval_min"), ".6g")
        source = cal.get("frame_interval_min_source")
        put("interval", f"{interval} min" if interval else ("unknown" if m else None),
            f"source: {source}" if source else "")
        z_step = number(cal.get("z_step_um"), ".6g")
        source = cal.get("z_step_um_source")
        if dimensionality == "3D":
            put("z_step", f"{z_step} µm" if z_step else "unknown (results in voxels)",
                f"source: {source}" if source else "")
        else:
            put("z_step", None)
        self.run_fields["z_step"].setVisible(dimensionality == "3D")

        schema = schema_version_of(analysis)
        put(
            "schema",
            schema,
            "1 = written by Corridor 1.x and upgraded on loading; 2 = Corridor 2.0"
            if schema is not None
            else "",
        )

        # The model: run.json["model"] in v2, the 1.x path and checksum otherwise.
        model_path = seg.get("model_path")
        model_id = model.get("model_id") or (Path(model_path).name if model_path else None)
        if model_id and model.get("developer_override"):
            model_id = f"{model_id}  ·  developer override, not validated"
        put("model", model_id, _join([model.get("architecture"), model_path], "  ·  ") or "")
        put("model_version", model.get("model_version"),
            f"training data: {model['training_dataset_version']}"
            if model.get("training_dataset_version") else "")
        sha = model.get("sha256") or seg.get("model_sha256")
        put("model_hash", f"{sha[:12]}…" if sha else None, sha or "")
        put("cellpose", env.get("cellpose"),
            f"required: {model['cellpose_version']}" if model.get("cellpose_version") else "")

        cellprob = number(seg.get("cellprob_threshold"))
        flow = number(seg.get("flow_threshold"))
        put(
            "thresholds",
            _join([f"prob {cellprob}" if cellprob else None, f"flow {flow}" if flow else None]),
        )
        extent = seg.get("min_extent_px")
        put("min_extent", f"{number(extent)} px" if number(extent) else None)

        mode = seg.get("normalisation_mode")
        # The tooltip carries the actual numbers, so a reader never has to trust
        # that a friendly name means what they assume. An older analysis has
        # none of these keys, and an empty tooltip is the honest answer there.
        hint = ""
        if mode:
            hint = _join(
                [
                    f"percentiles {seg['normalize_percentiles']}"
                    if seg.get("normalize_percentiles") is not None else None,
                    f"tile {seg['normalize_tile_px']} px"
                    if seg.get("normalize_tile_px") is not None else None,
                    f"sharpen {seg['normalize_sharpen_px']} px"
                    if seg.get("normalize_sharpen_px") is not None else None,
                ],
                ", ",
            ) or ""
        put("normalisation", NORMALISATION_LABELS.get(mode, mode) if mode else None, hint)

        # Written unconditionally, like every other row here. An analysis saved
        # before this field existed must show "—", not whichever value the
        # previously inspected analysis left behind.
        rung = seg.get("ensemble")
        text = None
        if rung:
            text = ENSEMBLE_LABELS.get(rung, rung)
            passes = seg.get("ensemble_passes") or 1
            if passes > 1:
                text += f"  ·  {passes} passes"
            borrowed = seg.get("detections_from_fallback")
            if borrowed:
                text += f"  ·  {borrowed} extra detection(s)"
        put("detection_effort", text)

        raw = seg.get("raw_instances_per_frame")
        kept = seg.get("kept_instances_per_frame")
        if raw or kept:
            raw_sum, kept_sum = sum(raw or []), sum(kept or [])
            put(
                "raw_kept",
                f"{raw_sum} → {kept_sum}"
                + (f" ({raw_sum - kept_sum} filtered)" if raw and kept and raw_sum != kept_sum else ""),
            )
        else:
            put("raw_kept", None)

        put("geometry", *self._geometry_text(geometry, m))

        max_gap = trk.get("max_gap")
        max_delta = trk.get("max_delta_frames")
        put(
            "max_gap",
            _join(
                [
                    f"{max_gap} frames" if max_gap is not None else None,
                    f"(gap ≤ {max_delta})" if max_delta is not None else None,
                ],
                " ",
            ),
        )
        speed = finite(trk.get("max_speed_um_per_min"))
        put("max_speed", f"{speed:g} µm/min ({speed * 60:g} µm/h)" if speed is not None else None)
        elapsed = finite(res.get("elapsed_seconds"))
        put("elapsed", f"{elapsed:.1f} s" if elapsed is not None else None)

    @staticmethod
    def _geometry_text(geometry: Any, manifest: dict[str, Any]) -> tuple[str | None, str]:
        """'yes' / 'no' with the lanes behind it, or an em dash with why."""
        if isinstance(geometry, dict):
            applied = geometry.get("applied")
            lanes = geometry.get("lanes") or []
            source = geometry.get("source")
            parts = [
                None if applied is None else ("yes" if applied else "no"),
                f"{len(lanes)} lane(s)" if isinstance(lanes, list) else None,
                f"from {source}" if source else None,
            ]
            notes = geometry.get("notes")
            hint = "; ".join(notes) if isinstance(notes, list) else (str(notes) if notes else "")
            return _join(parts), hint
        if isinstance(manifest.get("confinement"), dict) and manifest.get("confinement"):
            # A 1.x run had channel lines and an axis, but no lane gate to report.
            return None, "Not recorded: this run was written by Corridor 1.x."
        return None, ""

    # ---------------------------------------------------------------- events
    def _timeline_moved(self, index: int) -> None:
        self.active_view().set_frame(index)

    def _on_frame_changed(self, index: int) -> None:
        self.timeline.set_frame(index)
        self._update_readout(index)
        if self._selected_track is not None:
            self._show_track_detail(self._selected_track)

    def _update_readout(self, index: int) -> None:
        analysis = self.analysis
        if analysis is None:
            return
        total = analysis.n_frames or self.active_view().n_frames
        parts = [f"Frame {index + 1} of {total}"]
        source = analysis.source_frames
        if source and index < len(source):
            parts.append(f"original {source[index]}")
        interval = analysis.frame_interval_min
        if interval:
            minutes = index * interval
            parts.append(
                f"{minutes / 60:.2f} h" if minutes >= 90 else f"{minutes:.0f} min"
            )
        diagnostic = analysis.diagnostic_for(index)
        if diagnostic and diagnostic.get("raw_instances") is not None:
            raw = diagnostic.get("raw_instances") or 0
            kept = diagnostic.get("kept_instances") or 0
            parts.append(f"{kept} cell{'s' if kept != 1 else ''}" + (f" of {raw}" if raw != kept else ""))
        self.timeline.set_readout("   ·   ".join(parts))

    def _on_canvas_click(self, track_id: int) -> None:
        if track_id < 0:
            self.select_track(None)
            return
        self.select_track(track_id)

    def _on_track_selected(self, current: QListWidgetItem | None, _previous=None) -> None:
        if current is None:
            return
        track_id = current.data(Qt.UserRole)
        if track_id is not None:
            self.select_track(int(track_id), from_list=True)

    def select_track(self, track_id: int | None, from_list: bool = False) -> None:
        self._selected_track = track_id
        self.canvas.selected_track = track_id
        self.canvas.update()
        if self._three_d:
            self.ortho.select_track(track_id)
        self._update_export_actions()
        if track_id is None:
            self.detail_box.hide()
            self.track_list.clearSelection()
            return
        if not from_list:
            for row in range(self.track_list.count()):
                item = self.track_list.item(row)
                if item.data(Qt.UserRole) == track_id:
                    self.track_list.blockSignals(True)
                    self.track_list.setCurrentItem(item)
                    self.track_list.blockSignals(False)
                    break
        self.tabs.setCurrentIndex(0)
        self._show_track_detail(track_id)

    def _update_export_actions(self) -> None:
        has_track = self._selected_track is not None and self.analysis is not None
        for kind in TRACK_EXPORTS:
            action = self.export_actions.get(kind)
            if action is None:
                continue
            action.setEnabled(has_track)
            base = EXPORT_LABELS[kind]
            action.setText(
                base.replace("Selected track", f"Track {self._selected_track}")
                if has_track
                else base
            )
        for kind, action in self.export_actions.items():
            if kind not in TRACK_EXPORTS:
                action.setEnabled(self.analysis is not None)
        if has_track:
            self.export_track_button.setText(f"Export track {self._selected_track}")

    def _show_track_detail(self, track_id: int) -> None:
        analysis = self.analysis
        if analysis is None:
            return
        summary = analysis.summary_for(track_id) or {}
        rows = analysis.rows_for_track(track_id)
        self.detail_title.setText(f"Track {track_id}")
        self.export_track_button.setText(f"Export track {track_id}")

        first, last = summary.get("first_frame"), summary.get("last_frame")
        self.detail_fields["frames"].set_value(
            f"{first}–{last}" if first is not None and last is not None else DASH
        )
        observations = summary.get("n_observations")
        self.detail_fields["observations"].set_value(
            str(observations) if observations is not None else DASH
        )
        missing, gaps = summary.get("total_missing_frames"), summary.get("n_gaps")
        self.detail_fields["gaps"].set_value(
            f"{missing or 0} in {gaps or 0} gap(s)"
            if missing is not None or gaps is not None
            else DASH
        )
        self.detail_fields["duration"].set_value(_duration_text(summary.get("duration_min")))

        metrics = path_metrics(rows, summary, analysis.pixel_size_um)
        self.detail_fields["len"].set_value(fmt_number(metrics.length, 1, f" {metrics.unit}"))
        self.detail_fields["d2s"].set_value(fmt_number(metrics.from_start, 1, f" {metrics.unit}"))
        self.detail_fields["mean"].set_value(
            fmt_number(speed_um_per_hr(summary, "mean_speed"), 1, " µm/h")
        )
        self.detail_fields["max"].set_value(
            fmt_number(speed_um_per_hr(summary, "max_speed"), 1, " µm/h")
        )
        self.detail_fields["straight"].set_value(fmt_number(summary.get("straightness"), 2))

        alpha = finite(summary.get("msd_alpha"))
        r2 = finite(first_present(summary, "msd_alpha_r2", "msd_r2", "msd_fit_r2"))
        msd_rows = msd_rows_for(analysis, track_id)
        if alpha is not None:
            text = f"{alpha:.2f}" + (f"  (r² {r2:.2f})" if r2 is not None else "")
            self.detail_fields["alpha"].set_value(text)
        elif msd_rows or has_msd(analysis) or "msd_alpha" in summary:
            self.detail_fields["alpha"].set_value(
                MSD_EMPTY_TEXT,
                "Fitted only over lags with enough observation pairs, up to half "
                "the track's span, and only when enough such lags exist.",
            )
        else:
            self.detail_fields["alpha"].set_value(DASH, "This analysis has no MSD curves.")

        flags = summary.get("flags")
        self.detail_flags.setText(f"Flags: {flags}" if flags else "")
        self.detail_flags.setVisible(bool(flags))

        colour = track_color(track_id)
        series = [
            (float(r["frame"]), speed_um_per_hr(r))
            for r in rows
            if r.get("frame") is not None and speed_um_per_hr(r) is not None
        ]
        self.sparkline.set_series(series, colour, float(self.active_view().frame))
        self.msd_plot.set_series(msd_series(msd_rows, analysis.frame_interval_min), colour)
        self.detail_box.show()

    def _on_check_activated(self, item: QListWidgetItem) -> None:
        issue = item.data(Qt.UserRole) or {}
        frame = issue.get("frame")
        track_id = issue.get("track_id")
        if frame is not None:
            self.active_view().set_frame(int(frame))
        if track_id is not None:
            self.select_track(int(track_id))
            self.tabs.setCurrentIndex(1)

    # -------------------------------------------------------- reference point
    def _reference_mode_toggled(self, enabled: bool) -> None:
        self.canvas.set_picking(enabled)
        self.ortho.set_picking(enabled)

    def set_reference_mode(self, enabled: bool) -> None:
        self.reference_button.setChecked(enabled)

    def _reference_picked(self, x: float, y: float, z: float | None = None) -> None:
        point = (float(x), float(y)) if z is None else (float(x), float(y), float(z))
        self._set_reference(point, persist=True)
        self.reference_button.setChecked(False)

    def clear_reference_point(self) -> None:
        self._set_reference(None, persist=True)

    def _set_reference(self, point: tuple[float, ...] | None, *, persist: bool) -> None:
        self.reference_point_px = tuple(point) if point is not None else None
        self.canvas.reference_point = (
            (point[0], point[1]) if point is not None else None  # type: ignore[assignment]
        )
        self.ortho.reference_point = self.reference_point_px
        self.canvas.update()
        self.clear_reference_button.setVisible(point is not None)
        self.reference_button.setToolTip(
            (
                f"Reference point at x {point[0]:.0f}, y {point[1]:.0f}"
                + (f", z {point[2]:.0f}" if len(point) > 2 else "")
                + " px. Click to move it."
            )
            if point is not None
            else "Click the image to set the point that distances to reference "
            "(D2R) are measured from in exports. Stored with this analysis."
        )
        self.reference_error = ""
        if persist and self.analysis is not None:
            directory = getattr(self.analysis, "directory", None)
            if directory is not None:
                try:
                    save_reference_point(directory, self.reference_point_px)
                except OSError as exc:
                    # Kept for this session; the export still uses it. Saying so
                    # beats pretending it was stored.
                    self.reference_error = f"The reference point could not be saved: {exc}"
            self.reference_point_changed.emit(self.reference_point_px)
