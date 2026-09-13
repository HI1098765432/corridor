"""The microscopy canvas: image, masks, centroids and trajectories.

Everything is drawn with one painter onto one widget, in image coordinates
mapped through a single scale-and-offset transform.  That keeps overlays
exactly registered to the pixels at any zoom, which is the whole point: a
reviewer must be able to see that a track really sits on the cell.

Rendering cost is kept off the interaction path by caching, per frame, the
8-bit display image and the mask outline layer.  Both are derived data, so
they are recomputed silently whenever the source changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from ..theme import PALETTE, TYPE, track_color

MIN_SCALE = 0.05
MAX_SCALE = 60.0


@dataclass
class Layers:
    image: bool = True
    masks: bool = True
    centroids: bool = True
    trails: bool = True
    labels: bool = True
    axis: bool = False
    channels: bool = False


def _stretch_to_uint8(frame: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    f = np.asarray(frame, dtype=np.float32)
    if f.size == 0:
        return np.zeros(f.shape, dtype=np.uint8)
    lo, hi = np.percentile(f, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return np.zeros(f.shape, dtype=np.uint8)
    return np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def _outline_mask(labels: np.ndarray) -> np.ndarray:
    """Boolean image of instance boundaries, computed without SciPy."""
    m = labels > 0
    if not m.any():
        return m
    eroded = m.copy()
    eroded[1:, :] &= m[:-1, :]
    eroded[:-1, :] &= m[1:, :]
    eroded[:, 1:] &= m[:, :-1]
    eroded[:, :-1] &= m[:, 1:]
    # A boundary between two touching instances must also be drawn, otherwise
    # a merged pair looks like one object.
    seam = np.zeros_like(m)
    seam[1:, :] |= (labels[1:, :] != labels[:-1, :]) & m[1:, :] & m[:-1, :]
    seam[:, 1:] |= (labels[:, 1:] != labels[:, :-1]) & m[:, 1:] & m[:, :-1]
    return (m & ~eroded) | seam


class ImageCanvas(QWidget):
    """Pan/zoom microscopy view with registered analysis overlays."""

    track_clicked = Signal(int)  # track_id, or -1 for "nothing here"
    frame_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

        self._stack: np.ndarray | None = None
        self._masks: np.ndarray | None = None
        self._frame = 0
        self._scale = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._panning = False
        self._pan_anchor = QPoint()
        self._fit_pending = True

        self._image_cache: dict[int, QPixmap] = {}
        self._outline_cache: dict[int, QPixmap] = {}

        self.layers = Layers()
        self.rows_for_frame: Callable[[int], list[dict[str, Any]]] = lambda _f: []
        self.rows_for_track: Callable[[int], list[dict[str, Any]]] = lambda _t: []
        self.selected_track: int | None = None
        self.axis_vector: tuple[float, float] | None = None
        self.channel_lines: list[tuple[float, float]] = []  # (origin_x, origin_y)
        self.trail_length = 0  # 0 == the whole track so far

    # ------------------------------------------------------------------ data
    def set_stack(self, stack: np.ndarray | None) -> None:
        self._stack = stack
        self._image_cache.clear()
        self._frame = 0
        self._fit_pending = True
        # Fit after the event loop has laid this widget out. Fitting now would
        # measure whatever size the widget has before its page is shown, which
        # for a stacked page that is not current is not its real size.
        QTimer.singleShot(0, self._fit_if_pending)
        self.update()

    def _fit_if_pending(self) -> None:
        if self._fit_pending and self._stack is not None:
            self.fit_to_view()

    def set_masks(self, masks: np.ndarray | None) -> None:
        self._masks = masks
        self._outline_cache.clear()
        self.update()

    def clear(self) -> None:
        self._stack = None
        self._masks = None
        self._image_cache.clear()
        self._outline_cache.clear()
        self.selected_track = None
        self.channel_lines = []
        self.axis_vector = None
        self.update()

    @property
    def n_frames(self) -> int:
        return 0 if self._stack is None else int(self._stack.shape[0])

    @property
    def frame(self) -> int:
        return self._frame

    def set_frame(self, index: int) -> None:
        if self._stack is None:
            return
        index = max(0, min(int(index), self.n_frames - 1))
        if index != self._frame:
            self._frame = index
            self.frame_changed.emit(index)
            self.update()

    # ------------------------------------------------------------- transform
    def _image_size(self) -> tuple[int, int]:
        if self._stack is None:
            return (0, 0)
        return (int(self._stack.shape[2]), int(self._stack.shape[1]))

    def fit_to_view(self) -> None:
        w, h = self._image_size()
        if not w or not h:
            return
        margin = 28
        available_w = max(1, self.width() - margin * 2)
        available_h = max(1, self.height() - margin * 2)
        self._scale = max(MIN_SCALE, min(available_w / w, available_h / h))
        self._centre()
        # Only stop re-fitting once the widget has a real size. Fitting to a
        # widget that has not been laid out yet leaves the image tiny and
        # stuck in a corner.
        if self.width() > 200 and self.height() > 200:
            self._fit_pending = False
        self.update()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._fit_pending:
            self.fit_to_view()

    def _centre(self) -> None:
        w, h = self._image_size()
        self._offset = QPointF(
            (self.width() - w * self._scale) / 2.0,
            (self.height() - h * self._scale) / 2.0,
        )

    def zoom_by(self, factor: float, anchor: QPointF | None = None) -> None:
        w, h = self._image_size()
        if not w:
            return
        new_scale = max(MIN_SCALE, min(MAX_SCALE, self._scale * factor))
        if new_scale == self._scale:
            return
        anchor = anchor or QPointF(self.width() / 2.0, self.height() / 2.0)
        before = self._to_image(anchor)
        self._scale = new_scale
        after = self._to_image(anchor)
        self._offset += QPointF(
            (after.x() - before.x()) * self._scale,
            (after.y() - before.y()) * self._scale,
        )
        self.update()

    def reset_view(self) -> None:
        self.fit_to_view()

    def _to_widget(self, x: float, y: float) -> QPointF:
        return QPointF(x * self._scale + self._offset.x(), y * self._scale + self._offset.y())

    def _to_image(self, point: QPointF) -> QPointF:
        return QPointF(
            (point.x() - self._offset.x()) / self._scale,
            (point.y() - self._offset.y()) / self._scale,
        )

    # --------------------------------------------------------------- caching
    def _frame_pixmap(self, index: int) -> QPixmap | None:
        if self._stack is None:
            return None
        cached = self._image_cache.get(index)
        if cached is not None:
            return cached
        grey = np.ascontiguousarray(_stretch_to_uint8(self._stack[index]))
        h, w = grey.shape
        image = QImage(grey.data, w, h, w, QImage.Format_Grayscale8).copy()
        pixmap = QPixmap.fromImage(image)
        if len(self._image_cache) > 48:
            self._image_cache.clear()
        self._image_cache[index] = pixmap
        return pixmap

    def _outline_pixmap(self, index: int) -> QPixmap | None:
        if self._masks is None or index >= len(self._masks):
            return None
        cached = self._outline_cache.get(index)
        if cached is not None:
            return cached
        labels = np.asarray(self._masks[index])
        border = _outline_mask(labels)
        h, w = border.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        colour = QColor(PALETTE.mask_outline)
        rgba[border] = (colour.red(), colour.green(), colour.blue(), 235)
        buffer = np.ascontiguousarray(rgba)
        image = QImage(buffer.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        pixmap = QPixmap.fromImage(image)
        if len(self._outline_cache) > 48:
            self._outline_cache.clear()
        self._outline_cache[index] = pixmap
        return pixmap

    # --------------------------------------------------------------- events
    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit_if_pending()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._stack is None:
            return
        steps = event.angleDelta().y() / 120.0
        self.zoom_by(1.15**steps, QPointF(event.position()))
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._stack is None:
            return
        if event.button() in (Qt.MiddleButton, Qt.RightButton) or (
            event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier
        ):
            self._panning = True
            self._pan_anchor = event.position().toPoint()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
        elif event.button() == Qt.LeftButton:
            self._select_at(QPointF(event.position()))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._panning:
            delta = event.position().toPoint() - self._pan_anchor
            self._pan_anchor = event.position().toPoint()
            self._offset += QPointF(delta.x(), delta.y())
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._panning:
            self._panning = False
            self.setCursor(QCursor(Qt.ArrowCursor))

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.fit_to_view()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key in (Qt.Key_Left, Qt.Key_Comma):
            self.set_frame(self._frame - 1)
        elif key in (Qt.Key_Right, Qt.Key_Period):
            self.set_frame(self._frame + 1)
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_by(1.25)
        elif key == Qt.Key_Minus:
            self.zoom_by(1 / 1.25)
        elif key == Qt.Key_0:
            self.fit_to_view()
        else:
            super().keyPressEvent(event)

    def _select_at(self, position: QPointF) -> None:
        point = self._to_image(position)
        best_id, best_distance = -1, 1e12
        for row in self.rows_for_frame(self._frame):
            x, y = row.get("x_px"), row.get("y_px")
            if x is None or y is None:
                continue
            distance = (x - point.x()) ** 2 + (y - point.y()) ** 2
            if distance < best_distance:
                best_distance, best_id = distance, int(row.get("track_id", -1))
        # A click within ~24 screen pixels of a centroid selects it.
        threshold = (24.0 / max(self._scale, 1e-6)) ** 2
        self.track_clicked.emit(best_id if best_distance <= threshold else -1)

    # -------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(PALETTE.surface_sunken))
        if self._stack is None:
            painter.end()
            return
        painter.setRenderHint(QPainter.Antialiasing, True)

        w, h = self._image_size()
        target = QRectF(self._to_widget(0, 0), self._to_widget(w, h))

        if self.layers.image:
            pixmap = self._frame_pixmap(self._frame)
            if pixmap is not None:
                painter.setRenderHint(
                    QPainter.SmoothPixmapTransform, self._scale < 2.0
                )
                painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))
                painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        else:
            painter.fillRect(target, QColor("#101214"))

        if self.layers.masks:
            outline = self._outline_pixmap(self._frame)
            if outline is not None:
                painter.drawPixmap(target, outline, QRectF(outline.rect()))

        if self.layers.channels and self.channel_lines:
            self._paint_channels(painter, h)
        if self.layers.axis and self.axis_vector:
            self._paint_axis(painter, w, h)
        if self.layers.trails:
            self._paint_trails(painter)
        if self.layers.centroids or self.layers.labels:
            self._paint_markers(painter)

        # A thin frame keeps the image distinct from the canvas behind it.
        painter.setPen(QPen(QColor(PALETTE.border_strong), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(target)
        painter.end()

    def _paint_channels(self, painter: QPainter, height: int) -> None:
        ux, uy = self.axis_vector or (0.0, 1.0)
        pen = QPen(QColor(PALETTE.axis_guide), 1.0, Qt.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        span = height * 2
        for origin_x, origin_y in self.channel_lines:
            start = self._to_widget(origin_x - ux * span, origin_y - uy * span)
            end = self._to_widget(origin_x + ux * span, origin_y + uy * span)
            painter.drawLine(start, end)

    def _paint_axis(self, painter: QPainter, width: int, height: int) -> None:
        ux, uy = self.axis_vector or (0.0, 1.0)
        cx, cy = width / 2.0, height / 2.0
        length = min(width, height) * 0.35
        pen = QPen(QColor(PALETTE.axis_guide), 2.0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(
            self._to_widget(cx - ux * length, cy - uy * length),
            self._to_widget(cx + ux * length, cy + uy * length),
        )
        head = self._to_widget(cx + ux * length, cy + uy * length)
        painter.setBrush(QColor(PALETTE.axis_guide))
        painter.drawEllipse(head, 4, 4)
        painter.setBrush(Qt.NoBrush)

    def _paint_trails(self, painter: QPainter) -> None:
        ids = {
            int(r["track_id"])
            for r in self.rows_for_frame(self._frame)
            if r.get("track_id") is not None
        }
        if self.selected_track is not None:
            ids.add(int(self.selected_track))
        for track_id in sorted(ids):
            rows = [
                r for r in self.rows_for_track(track_id)
                if r.get("frame") is not None and r["frame"] <= self._frame
            ]
            if self.trail_length:
                rows = rows[-self.trail_length:]
            if len(rows) < 2:
                continue
            selected = self.selected_track == track_id
            colour = QColor(track_color(track_id))
            if self.selected_track is not None and not selected:
                colour.setAlpha(90)
            pen = QPen(colour, 3.0 if selected else 2.0)
            pen.setCosmetic(True)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)

            path = QPainterPath()
            previous_frame = None
            for row in rows:
                point = self._to_widget(row["x_px"], row["y_px"])
                gap = row.get("gap_frames") or 1
                if previous_frame is None:
                    path.moveTo(point)
                elif gap and gap > 1:
                    # A bridged gap is drawn dashed: the position between those
                    # frames was predicted, never measured.
                    painter.drawPath(path)
                    dashed = QPen(colour, 2.0, Qt.DotLine)
                    dashed.setCosmetic(True)
                    painter.setPen(dashed)
                    painter.drawLine(path.currentPosition(), point)
                    painter.setPen(pen)
                    path = QPainterPath()
                    path.moveTo(point)
                else:
                    path.lineTo(point)
                previous_frame = row["frame"]
            painter.drawPath(path)

    def _paint_markers(self, painter: QPainter) -> None:
        font = QFont()
        font.setPointSize(TYPE["caption"])
        font.setBold(True)
        painter.setFont(font)

        for row in self.rows_for_frame(self._frame):
            x, y = row.get("x_px"), row.get("y_px")
            if x is None or y is None:
                continue
            track_id = int(row.get("track_id", 0))
            selected = self.selected_track == track_id
            colour = QColor(track_color(track_id))
            if self.selected_track is not None and not selected:
                colour.setAlpha(110)
            centre = self._to_widget(x, y)

            if self.layers.centroids:
                radius = 7.0 if selected else 5.0
                painter.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
                painter.setBrush(colour)
                painter.drawEllipse(centre, radius, radius)
                painter.setBrush(Qt.NoBrush)
                if selected:
                    ring = QPen(colour, 2.0)
                    ring.setCosmetic(True)
                    painter.setPen(ring)
                    painter.drawEllipse(centre, radius + 6, radius + 6)

            if self.layers.labels:
                label = str(track_id)
                offset = QPointF(centre.x() + 11, centre.y() - 9)
                rect = QRectF(offset.x() - 2, offset.y() - 12, 13 + 7 * len(label), 17)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, 0, 0, 150))
                painter.drawRoundedRect(rect, 4, 4)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QColor(255, 255, 255, 235))
                painter.drawText(rect, Qt.AlignCenter, label)
