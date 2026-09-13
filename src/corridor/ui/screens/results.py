"""The results workspace.

Priorities, in order: the microscopy is the object; overlays must be
switchable without hunting; a questionable track must be one click from the
frame where it went wrong.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...store.project import SavedAnalysis
from ..icons import icon
from ..theme import PALETTE, RADIUS, SPACE, TYPE, track_color
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

SEVERITY_TONE = {"critical": "danger", "warning": "warning", "info": "neutral"}


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


class ResultsScreen(QWidget):
    """Image, overlays, timeline, inspector."""

    back_requested = Signal()
    export_requested = Signal()
    napari_requested = Signal()
    reanalyse_requested = Signal()
    open_folder_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.analysis: SavedAnalysis | None = None
        self._selected_track: int | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

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
        self.canvas = ImageCanvas()
        self.canvas.track_clicked.connect(self._on_canvas_click)
        self.canvas.frame_changed.connect(self._on_frame_changed)
        canvas_layout.addWidget(self.canvas, 1)
        self.timeline = TimelineBar()
        self.timeline.frame_changed.connect(self.canvas.set_frame)
        canvas_layout.addWidget(divider())
        canvas_layout.addWidget(self.timeline)

        splitter.addWidget(canvas_holder)
        splitter.addWidget(self._build_inspector())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 360])
        root.addWidget(splitter, 1)

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
        self.export_button = primary_button("Export", self.export_requested.emit)

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
        self.toggle_labels = layer_toggle("IDs", "plus", True)
        self.toggle_axis = layer_toggle("Channels", "grid", False)

        for button in (
            self.toggle_image, self.toggle_masks, self.toggle_points,
            self.toggle_trails, self.toggle_labels, self.toggle_axis,
        ):
            button.toggled.connect(self._sync_layers)
            layout.addWidget(button)

        layout.addStretch(1)
        reset = ghost_button("", "zoom-reset", self.canvas_reset)
        reset.setFixedWidth(38)
        reset.setToolTip("Fit to window (0)")
        layout.addWidget(reset)
        return bar

    def canvas_reset(self) -> None:
        self.canvas.fit_to_view()

    def _sync_layers(self) -> None:
        layers = self.canvas.layers
        layers.image = self.toggle_image.isChecked()
        layers.masks = self.toggle_masks.isChecked()
        layers.centroids = self.toggle_points.isChecked()
        layers.trails = self.toggle_trails.isChecked()
        layers.labels = self.toggle_labels.isChecked()
        layers.axis = self.toggle_axis.isChecked()
        layers.channels = self.toggle_axis.isChecked()
        self.canvas.update()

    # -------------------------------------------------------------- inspector
    def _build_inspector(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("Inspector")
        panel.setMinimumWidth(340)
        panel.setMaximumWidth(480)
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
        layout.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["lg"])
        layout.setSpacing(SPACE["md"])

        summary = QHBoxLayout()
        summary.setSpacing(SPACE["xl"])
        self.metric_tracks = Metric("tracks")
        self.metric_detections = Metric("detections")
        self.metric_speed = Metric("mean speed")
        summary.addWidget(self.metric_tracks)
        summary.addWidget(self.metric_detections)
        summary.addWidget(self.metric_speed)
        summary.addStretch(1)
        layout.addLayout(summary)
        layout.addWidget(divider())

        self.track_list = QListWidget()
        self.track_list.currentItemChanged.connect(self._on_track_selected)
        layout.addWidget(self.track_list, 1)

        self.detail_box = QWidget()
        detail_layout = QVBoxLayout(self.detail_box)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(SPACE["xs"])
        self.detail_title = label("", "subtitle")
        detail_layout.addWidget(self.detail_title)
        self.sparkline = Sparkline()
        detail_layout.addWidget(self.sparkline)
        self.detail_fields: dict[str, Field] = {}
        for key, name in (
            ("frames", "Frames"),
            ("observations", "Observations"),
            ("gaps", "Missing frames"),
            ("duration", "Duration"),
            ("mean", "Mean speed"),
            ("max", "Peak speed"),
            ("net", "Net displacement"),
            ("straight", "Straightness"),
        ):
            field = Field(name)
            self.detail_fields[key] = field
            detail_layout.addWidget(field)
        self.detail_flags = label("", "tertiary")
        self.detail_flags.setWordWrap(True)
        detail_layout.addWidget(self.detail_flags)
        self.detail_box.hide()
        layout.addWidget(self.detail_box)
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
        return page

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
        for key, name in (
            ("file", "File"),
            ("shape", "Stack"),
            ("axes", "Layout"),
            ("source_frames", "Original frames"),
            ("pixel", "Pixel size"),
            ("interval", "Frame interval"),
            ("axis", "Migration axis"),
            ("channels", "Channels"),
            ("model", "Model"),
            ("model_hash", "Model checksum"),
            ("cellpose", "Cellpose"),
            ("thresholds", "Thresholds"),
            ("min_extent", "Smallest object"),
            ("raw_kept", "Instances raw → kept"),
            ("max_gap", "Allowed disappearance"),
            ("max_speed", "Speed limit"),
            ("elapsed", "Took"),
        ):
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
    def load(self, analysis: SavedAnalysis, stack: np.ndarray) -> None:
        self.analysis = analysis
        self._selected_track = None

        source = analysis.source_path
        self.title.setText(source.name if source else analysis.directory.name)
        self.subtitle.setText(str(source.parent) if source else "")

        self.canvas.rows_for_frame = analysis.rows_for_frame
        self.canvas.rows_for_track = analysis.rows_for_track
        self.canvas.set_stack(stack)
        self.canvas.set_masks(analysis.masks)
        self.canvas.selected_track = None

        confinement = analysis.manifest.get("confinement", {})
        self.canvas.axis_vector = (
            float(confinement.get("ux", 0.0)), float(confinement.get("uy", 1.0))
        )
        self.canvas.channel_lines = [
            (float(c.get("origin_x", 0.0)), float(c.get("origin_y", 0.0)))
            for c in confinement.get("channels", [])
        ]
        self.canvas.fit_to_view()

        self.timeline.configure(stack.shape[0])
        self.timeline.set_frame(0)
        self._update_readout(0)

        self._populate_tracks()
        self._populate_checks()
        self._populate_run()
        self._sync_layers()

    def _populate_tracks(self) -> None:
        analysis = self.analysis
        if analysis is None:
            return
        results = analysis.manifest.get("results", {})
        self.metric_tracks.set_value(str(results.get("n_tracks", len(analysis.summaries))))
        self.metric_detections.set_value(str(results.get("n_detections", len(analysis.detections))))
        mean = results.get("mean_speed_um_per_min")
        self.metric_speed.set_value(f"{mean:.2f}" if mean else "—")
        self.metric_speed.setToolTip("Mean of each track's mean speed, in µm/min")

        self.track_list.clear()
        for summary in sorted(
            analysis.summaries, key=lambda s: -(s.get("n_observations") or 0)
        ):
            track_id = int(summary.get("track_id", 0))
            observations = summary.get("n_observations") or 0
            speed = summary.get("mean_speed_um_per_min")
            first, last = summary.get("first_frame"), summary.get("last_frame")
            pieces = [f"frames {first}–{last}", f"{observations} obs"]
            if speed:
                pieces.append(f"{speed:.2f} µm/min")
            item = QListWidgetItem(f"Track {track_id}\n{'  ·  '.join(pieces)}")
            item.setData(Qt.UserRole, track_id)
            item.setForeground(QColor(PALETTE.text))
            swatch = QColor(track_color(track_id))
            from PySide6.QtGui import QPixmap

            chip = QPixmap(10, 10)
            chip.fill(swatch)
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
        if not issues:
            self.checks_summary.setText("Nothing needs attention.")
        else:
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
        analysis = self.analysis
        if analysis is None:
            return
        m = analysis.manifest
        inp = m.get("input", {})
        cal = m.get("calibration", {})
        seg = m.get("segmentation", {})
        trk = m.get("tracking", {})
        conf = m.get("confinement", {})
        env = m.get("environment", {})
        res = m.get("results", {})

        def put(key: str, value: Any, hint: str = "") -> None:
            field = self.run_fields.get(key)
            if field:
                field.set_value("—" if value in (None, "") else str(value), hint)

        put("file", inp.get("name"), inp.get("path", ""))
        shape = inp.get("shape_tyx") or []
        put("shape", f"{shape[0]} × {shape[1]} × {shape[2]}" if len(shape) == 3 else None)
        put("axes", f"{inp.get('axes_reported')} · {inp.get('axes_interpretation')}")
        frames = inp.get("source_frames")
        put(
            "source_frames",
            f"{frames[0]}–{frames[-1]} of {inp.get('source_frame_total')}" if frames else None,
        )
        pixel = cal.get("pixel_size_um")
        put(
            "pixel",
            f"{pixel:.6g} µm/px" if pixel else "unknown",
            f"source: {cal.get('pixel_size_um_source')}",
        )
        interval = cal.get("frame_interval_min")
        put(
            "interval",
            f"{interval:.6g} min" if interval else "unknown",
            f"source: {cal.get('frame_interval_min_source')}",
        )
        tilt = conf.get("tilt_from_vertical_deg")
        put(
            "axis",
            f"{tilt:+.2f}° from vertical" if tilt is not None else None,
            f"source: {conf.get('source')}",
        )
        put("channels", conf.get("n_channels"))
        model = seg.get("model_path")
        put("model", Path(model).name if model else None, model or "")
        sha = seg.get("model_sha256")
        put("model_hash", f"{sha[:12]}…" if sha else None, sha or "")
        put("cellpose", env.get("cellpose"))
        put(
            "thresholds",
            f"prob {seg.get('cellprob_threshold')}  ·  flow {seg.get('flow_threshold')}",
        )
        put("min_extent", f"{seg.get('min_extent_px')} px")
        raw = seg.get("raw_instances_per_frame") or []
        kept = seg.get("kept_instances_per_frame") or []
        put(
            "raw_kept",
            f"{sum(raw)} → {sum(kept)}"
            + (f" ({sum(raw) - sum(kept)} filtered)" if sum(raw) != sum(kept) else ""),
        )
        put(
            "max_gap",
            f"{trk.get('max_gap')} frames (gap ≤ {trk.get('max_delta_frames')})",
        )
        put("max_speed", f"{trk.get('max_speed_um_per_min')} µm/min")
        elapsed = res.get("elapsed_seconds")
        put("elapsed", f"{elapsed:.1f} s" if elapsed else None)

    # ---------------------------------------------------------------- events
    def _on_frame_changed(self, index: int) -> None:
        self.timeline.set_frame(index)
        self._update_readout(index)
        if self._selected_track is not None:
            self._show_track_detail(self._selected_track)

    def _update_readout(self, index: int) -> None:
        analysis = self.analysis
        if analysis is None:
            return
        total = analysis.n_frames
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

    def _show_track_detail(self, track_id: int) -> None:
        analysis = self.analysis
        if analysis is None:
            return
        summary = analysis.summary_for(track_id) or {}
        rows = analysis.rows_for_track(track_id)
        self.detail_title.setText(f"Track {track_id}")

        def fmt(value: Any, suffix: str = "", digits: int = 2) -> str:
            if value is None:
                return "—"
            if isinstance(value, float):
                return f"{value:.{digits}f}{suffix}"
            return f"{value}{suffix}"

        self.detail_fields["frames"].set_value(
            f"{summary.get('first_frame')}–{summary.get('last_frame')}"
        )
        self.detail_fields["observations"].set_value(fmt(summary.get("n_observations")))
        self.detail_fields["gaps"].set_value(
            f"{summary.get('total_missing_frames') or 0} in {summary.get('n_gaps') or 0} gap(s)"
        )
        self.detail_fields["duration"].set_value(fmt(summary.get("duration_min"), " min", 0))
        self.detail_fields["mean"].set_value(
            fmt(summary.get("mean_speed_um_per_min"), " µm/min")
        )
        self.detail_fields["max"].set_value(
            fmt(summary.get("max_speed_um_per_min"), " µm/min")
        )
        self.detail_fields["net"].set_value(fmt(summary.get("net_displacement_um"), " µm", 1))
        self.detail_fields["straight"].set_value(fmt(summary.get("straightness"), "", 2))
        flags = summary.get("flags")
        self.detail_flags.setText(f"Flags: {flags}" if flags else "")
        self.detail_flags.setVisible(bool(flags))

        series = [
            (float(r["frame"]), r.get("speed_um_per_min"))
            for r in rows
            if r.get("speed_um_per_min") is not None
        ]
        self.sparkline.set_series(
            series, track_color(track_id), float(self.canvas.frame)
        )
        self.detail_box.show()

    def _on_check_activated(self, item: QListWidgetItem) -> None:
        issue = item.data(Qt.UserRole) or {}
        frame = issue.get("frame")
        track_id = issue.get("track_id")
        if frame is not None:
            self.canvas.set_frame(int(frame))
        if track_id is not None:
            self.select_track(int(track_id))
            self.tabs.setCurrentIndex(1)
