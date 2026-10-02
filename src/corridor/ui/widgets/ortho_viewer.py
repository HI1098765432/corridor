"""Orthogonal views of a 3-D time-lapse: XY, XZ and YZ around one cursor.

A ``TZYX`` result cannot be reviewed on the 2-D canvas: a slice hides every
cell above and below it, and a projection merges cells that only overlap in
Z. Three orthogonal planes through one cursor are the standard answer (and
what Fiji's *Orthogonal Views* shows), and they need no 3-D renderer -- Qt3D
is excluded from the build, and napari stays an optional extra.

Built the same way as :class:`~corridor.ui.widgets.image_canvas.ImageCanvas`:
each plane is a QImage made from a contrast-stretched array, overlays are
painted in that plane's own coordinates through one transform, and nothing
is cached that is not derived data.

Two facts drive the details:

*   **Z is drawn to scale only when the Z step is known.** Optical stacks are
    routinely sampled more coarsely in Z than in XY, so the Z axis of the
    XZ/YZ panes is stretched by the run's anisotropy (``z_step_um /
    pixel_size_um``). Without a calibrated Z step it is drawn one slice per
    pixel and the Z readout says so, rather than guessing a spacing.
*   **One contrast per time point, not per plane.** Stretching each plane on
    its own would make the same voxel a different grey in each pane; the
    percentiles are taken over the whole volume at that time point.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGridLayout, QSlider, QVBoxLayout, QWidget

from ..analysis_view import row_xy, row_z
from ..theme import PALETTE, SPACE, TYPE, track_color
from .common import label
from .image_canvas import Layers, _outline_mask

#: A track row is drawn on an XZ/YZ pane when its centroid lies within this
#: many pixels of the plane. Cells in the supplied data are 9-15 px wide, so
#: 6 px catches a cell the plane passes through without catching neighbours.
PLANE_TOLERANCE_PX = 6.0
#: ...and on the XY pane when within this many slices of the current Z.
SLICE_TOLERANCE = 0.5


def _grey_qimage(plane: np.ndarray, lo: float, hi: float) -> QImage:
    f = np.asarray(plane, dtype=np.float32)
    if hi > lo:
        grey = np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    else:
        grey = np.zeros(f.shape, dtype=np.uint8)
    grey = np.ascontiguousarray(grey)
    h, w = grey.shape
    return QImage(grey.data, w, h, w, QImage.Format_Grayscale8).copy()


def _outline_qimage(labels: np.ndarray) -> QImage:
    border = _outline_mask(np.asarray(labels))
    h, w = border.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    colour = QColor(PALETTE.mask_outline)
    rgba[border] = (colour.red(), colour.green(), colour.blue(), 235)
    buffer = np.ascontiguousarray(rgba)
    return QImage(buffer.data, w, h, w * 4, QImage.Format_RGBA8888).copy()


class SlicePane(QWidget):
    """One plane, fitted to the pane at its physical aspect ratio."""

    clicked = Signal(float, float)  # (u, v) in plane units: px or slices

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.setMinimumSize(120, 90)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._image: QPixmap | None = None
        self._overlay: QPixmap | None = None
        self._size = (0, 0)  # (n_u, n_v) samples
        self._aspect = (1.0, 1.0)  # physical size of one sample along u, v
        self._cursor: tuple[float, float] | None = None
        self._cursor_colour = PALETTE.text_tertiary
        self._markers: list[tuple[float, float, str, bool, str]] = []
        #: The D2R reference point in this plane, as (u, v, solid), or None.
        self._reference: tuple[float, float, bool] | None = None
        self.show_image = True
        self.show_labels = True

    def set_content(
        self,
        image: QImage | None,
        overlay: QImage | None,
        size: tuple[int, int],
        aspect: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self._image = QPixmap.fromImage(image) if image is not None else None
        self._overlay = QPixmap.fromImage(overlay) if overlay is not None else None
        self._size = (int(size[0]), int(size[1]))
        self._aspect = (float(aspect[0]) or 1.0, float(aspect[1]) or 1.0)
        self.update()

    def set_cursor(self, u: float, v: float, colour: str) -> None:
        self._cursor = (float(u), float(v))
        self._cursor_colour = colour
        self.update()

    def set_markers(self, markers: list[tuple[float, float, str, bool, str]]) -> None:
        """``(u, v, colour, solid, label)`` per marker."""
        self._markers = markers
        self.update()

    @property
    def markers(self) -> list[tuple[float, float, str, bool, str]]:
        return list(self._markers)

    def set_reference(self, uv: tuple[float, float] | None, solid: bool = True) -> None:
        """Where the reference point lies in this plane, or None to hide it."""
        self._reference = (float(uv[0]), float(uv[1]), bool(solid)) if uv is not None else None
        self.update()

    @property
    def reference(self) -> tuple[float, float, bool] | None:
        return self._reference

    # ------------------------------------------------------------ transform
    def _target(self) -> QRectF:
        n_u, n_v = self._size
        if not n_u or not n_v:
            return QRectF()
        width_phys, height_phys = n_u * self._aspect[0], n_v * self._aspect[1]
        margin = 6.0
        top = 18.0  # room for the pane title
        scale = min(
            max(1.0, self.width() - 2 * margin) / width_phys,
            max(1.0, self.height() - margin - top) / height_phys,
        )
        w, h = width_phys * scale, height_phys * scale
        return QRectF(
            (self.width() - w) / 2.0, top + (self.height() - top - margin - h) / 2.0, w, h
        )

    def to_widget(self, u: float, v: float) -> QPointF:
        rect = self._target()
        n_u, n_v = self._size
        return QPointF(
            rect.left() + u / max(n_u, 1) * rect.width(),
            rect.top() + v / max(n_v, 1) * rect.height(),
        )

    def to_plane(self, point: QPointF) -> tuple[float, float]:
        rect = self._target()
        n_u, n_v = self._size
        if rect.width() <= 0 or rect.height() <= 0:
            return (0.0, 0.0)
        return (
            (point.x() - rect.left()) / rect.width() * n_u,
            (point.y() - rect.top()) / rect.height() * n_v,
        )

    # -------------------------------------------------------------- events
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton or not self._size[0]:
            return
        u, v = self.to_plane(QPointF(event.position()))
        n_u, n_v = self._size
        if 0 <= u <= n_u and 0 <= v <= n_v:
            # Reported as pixel/slice indices: the centre of sample i is i.
            # Markers and the crosshair are drawn at i + 0.5 for the same
            # reason (see ImageCanvas._to_overlay).
            self.clicked.emit(float(u) - 0.5, float(v) - 0.5)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(PALETTE.surface_sunken))
        font = QFont()
        font.setPointSize(max(7, TYPE["caption"] - 2))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(PALETTE.text_tertiary))
        painter.drawText(QRectF(6, 2, 120, 14), Qt.AlignLeft | Qt.AlignVCenter, self.title)
        target = self._target()
        if target.isEmpty():
            painter.end()
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        if self.show_image and self._image is not None:
            painter.drawPixmap(target, self._image, QRectF(self._image.rect()))
        else:
            painter.fillRect(target, QColor("#101214"))
        if self._overlay is not None:
            painter.drawPixmap(target, self._overlay, QRectF(self._overlay.rect()))

        for u, v, colour, solid, text in self._markers:
            centre = self.to_widget(u + 0.5, v + 0.5)
            fill = QColor(colour)
            if not solid:
                fill.setAlpha(90)
            painter.setPen(QPen(QColor(255, 255, 255, 200 if solid else 90), 1.2))
            painter.setBrush(fill)
            painter.drawEllipse(centre, 4.0, 4.0)
            painter.setBrush(Qt.NoBrush)
            if self.show_labels and text and solid:
                painter.setPen(QColor(255, 255, 255, 230))
                painter.drawText(QPointF(centre.x() + 6, centre.y() - 5), text)

        if self._reference is not None:
            self._paint_reference(painter)

        if self._cursor is not None:
            pen = QPen(QColor(self._cursor_colour), 1.0, Qt.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            point = self.to_widget(self._cursor[0] + 0.5, self._cursor[1] + 0.5)
            painter.drawLine(QPointF(target.left(), point.y()), QPointF(target.right(), point.y()))
            painter.drawLine(QPointF(point.x(), target.top()), QPointF(point.x(), target.bottom()))

        painter.setPen(QPen(QColor(PALETTE.border_strong), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(target)
        painter.end()

    def _paint_reference(self, painter: QPainter) -> None:
        """The same ringed crosshair the 2-D canvas draws, faint off-slice."""
        u, v, solid = self._reference  # type: ignore[misc]
        centre = self.to_widget(u + 0.5, v + 0.5)
        alpha = 255 if solid else 110
        painter.setBrush(Qt.NoBrush)
        marker = QColor(PALETTE.reference_marker)
        marker.setAlpha(alpha)
        for colour, width in ((QColor(0, 0, 0, int(170 * alpha / 255)), 4.0), (marker, 2.0)):
            pen = QPen(colour, width)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawEllipse(centre, 7, 7)
            cx, cy = centre.x(), centre.y()
            for (x0, y0, x1, y1) in (
                (cx - 13, cy, cx - 4, cy), (cx + 4, cy, cx + 13, cy),
                (cx, cy - 13, cx, cy - 4), (cx, cy + 4, cx, cy + 13),
            ):
                painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))


class OrthoViewer(QWidget):
    """XY / XZ / YZ panes with a Z slider, for one time point of a TZYX stack."""

    track_clicked = Signal(int)  # track_id, or -1
    frame_changed = Signal(int)
    #: A click in picking mode, as (x, y, z) in pixels and slices.
    point_picked = Signal(float, float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stack: np.ndarray | None = None
        self._masks: np.ndarray | None = None
        self._frame = 0
        self._cursor = [0.0, 0.0, 0]  # x, y (px), z (slice index)
        self._contrast: dict[int, tuple[float, float]] = {}
        self._picking = False
        self.anisotropy: float | None = None
        self.layers = Layers()
        self.rows_for_frame: Callable[[int], list[dict[str, Any]]] = lambda _f: []
        self.selected_track: int | None = None
        #: The D2R reference point (x, y[, z]) in pixels and slices, or None.
        #: Set through :meth:`set_reference_point` so the panes redraw it.
        self.reference_point: tuple[float, ...] | None = None

        self.xy = SlicePane("XY")
        self.xz = SlicePane("XZ")
        self.yz = SlicePane("YZ")
        self.xy.clicked.connect(self._xy_clicked)
        self.xz.clicked.connect(self._xz_clicked)
        self.yz.clicked.connect(self._yz_clicked)

        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setRange(0, 0)
        self.z_slider.valueChanged.connect(self.set_z)
        self.z_readout = label("", "secondary")
        self.z_readout.setWordWrap(True)
        self.cursor_readout = label("", "tertiary")
        self.cursor_readout.setWordWrap(True)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        controls_layout.setSpacing(SPACE["sm"])
        controls_layout.addWidget(label("Z slice", "tertiary"))
        controls_layout.addWidget(self.z_slider)
        controls_layout.addWidget(self.z_readout)
        controls_layout.addWidget(self.cursor_readout)
        controls_layout.addStretch(1)

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        grid.addWidget(self.xy, 0, 0)
        grid.addWidget(self.yz, 0, 1)
        grid.addWidget(self.xz, 1, 0)
        grid.addWidget(controls, 1, 1)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 1)

    # ---------------------------------------------------------------- data
    def set_volume(
        self,
        stack: np.ndarray | None,
        masks: np.ndarray | None = None,
        anisotropy: float | None = None,
    ) -> None:
        """``stack`` and ``masks`` are ``TZYX``; a ``ZYX`` stack is one time point."""
        if stack is not None and stack.ndim == 3:
            stack = stack[np.newaxis]
        if masks is not None and masks.ndim == 3:
            masks = masks[np.newaxis]
        if stack is not None and stack.ndim != 4:
            raise ValueError(f"the orthogonal viewer needs a TZYX stack, got shape {stack.shape}")
        self._stack = stack
        self._masks = masks
        self.anisotropy = float(anisotropy) if anisotropy and anisotropy > 0 else None
        self._contrast.clear()
        self._frame = 0
        if stack is None:
            self.z_slider.setRange(0, 0)
            self._refresh()
            return
        _, n_z, n_y, n_x = stack.shape
        self._cursor = [n_x / 2.0, n_y / 2.0, n_z // 2]
        self.z_slider.blockSignals(True)
        self.z_slider.setRange(0, max(0, n_z - 1))
        self.z_slider.setValue(n_z // 2)
        self.z_slider.blockSignals(False)
        self._refresh()

    @property
    def n_frames(self) -> int:
        return 0 if self._stack is None else int(self._stack.shape[0])

    @property
    def n_slices(self) -> int:
        return 0 if self._stack is None else int(self._stack.shape[1])

    @property
    def frame(self) -> int:
        return self._frame

    @property
    def cursor(self) -> tuple[float, float, int]:
        return (float(self._cursor[0]), float(self._cursor[1]), int(self._cursor[2]))

    def set_frame(self, index: int) -> None:
        if self._stack is None:
            return
        index = max(0, min(int(index), self.n_frames - 1))
        if index != self._frame:
            self._frame = index
            self._follow_selection()
            self.frame_changed.emit(index)
            self._refresh()

    def set_z(self, z: int) -> None:
        if self._stack is None:
            return
        z = max(0, min(int(z), self.n_slices - 1))
        self._cursor[2] = z
        if self.z_slider.value() != z:
            self.z_slider.blockSignals(True)
            self.z_slider.setValue(z)
            self.z_slider.blockSignals(False)
        self._refresh()

    def set_cursor(self, x: float, y: float, z: float | None = None) -> None:
        if self._stack is None:
            return
        _, n_z, n_y, n_x = self._stack.shape
        self._cursor[0] = float(min(max(x, 0.0), n_x - 1))
        self._cursor[1] = float(min(max(y, 0.0), n_y - 1))
        if z is not None:
            self.set_z(int(round(z)))
        else:
            self._refresh()

    def select_track(self, track_id: int | None) -> None:
        """Select a track and centre the crosshair on it in this frame."""
        self.selected_track = track_id
        self._follow_selection()
        self._refresh()

    def set_reference_point(self, point: tuple[float, ...] | None) -> None:
        """Show the D2R reference point in every pane it lies in."""
        self.reference_point = tuple(float(v) for v in point) if point is not None else None
        self._update_reference()

    def set_picking(self, enabled: bool) -> None:
        self._picking = bool(enabled)
        cursor = QCursor(Qt.CrossCursor if self._picking else Qt.ArrowCursor)
        for pane in (self.xy, self.xz, self.yz):
            pane.setCursor(cursor)

    @property
    def picking(self) -> bool:
        return self._picking

    def fit_to_view(self) -> None:
        """Panes always fit; kept so the results screen can treat both views alike."""
        self._refresh()

    # --------------------------------------------------------------- planes
    def plane_arrays(self) -> dict[str, np.ndarray]:
        """The three raw planes through the cursor at the current time point.

        ``xy`` is (Y, X) at the cursor's Z; ``xz`` is (Z, X) at its Y; ``yz`` is
        (Y, Z) at its X, transposed so it shares the XY pane's vertical axis.
        """
        if self._stack is None:
            return {}
        volume = self._stack[self._frame]
        x, y, z = int(self._cursor[0]), int(self._cursor[1]), int(self._cursor[2])
        return {"xy": volume[z], "xz": volume[:, y, :], "yz": volume[:, :, x].T}

    def mask_planes(self) -> dict[str, np.ndarray]:
        if self._masks is None or self._frame >= len(self._masks):
            return {}
        labels = np.asarray(self._masks[self._frame])
        x, y, z = int(self._cursor[0]), int(self._cursor[1]), int(self._cursor[2])
        return {"xy": labels[z], "xz": labels[:, y, :], "yz": labels[:, :, x].T}

    def _contrast_for(self, frame: int) -> tuple[float, float]:
        cached = self._contrast.get(frame)
        if cached is not None:
            return cached
        volume = np.asarray(self._stack[frame], dtype=np.float32)  # type: ignore[index]
        # Every other voxel in each axis is ample for two percentiles and keeps
        # a large volume off the interaction path.
        sample = volume[::2, ::2, ::2] if volume.size > 2_000_000 else volume
        lo, hi = (float(v) for v in np.percentile(sample, [0.5, 99.5]))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(volume.min()), float(volume.max())
        self._contrast[frame] = (lo, hi)
        return lo, hi

    def _refresh(self) -> None:
        if self._stack is None:
            for pane in (self.xy, self.xz, self.yz):
                pane.set_content(None, None, (0, 0))
            self.z_readout.setText("")
            self.cursor_readout.setText("")
            return
        planes = self.plane_arrays()
        masks = self.mask_planes() if self.layers.masks else {}
        lo, hi = self._contrast_for(self._frame)
        z_scale = self.anisotropy or 1.0
        _, n_z, n_y, n_x = self._stack.shape
        geometry = {
            "xy": ((n_x, n_y), (1.0, 1.0)),
            "xz": ((n_x, n_z), (1.0, z_scale)),
            "yz": ((n_z, n_y), (z_scale, 1.0)),
        }
        for key, pane in (("xy", self.xy), ("xz", self.xz), ("yz", self.yz)):
            pane.show_image = self.layers.image
            pane.show_labels = self.layers.labels
            overlay = _outline_qimage(masks[key]) if key in masks else None
            size, aspect = geometry[key]
            pane.set_content(_grey_qimage(planes[key], lo, hi), overlay, size, aspect)

        x, y, z = self._cursor
        colour = PALETTE.text_tertiary
        if self.selected_track is not None and self._selected_position() is not None:
            colour = track_color(int(self.selected_track))
        self.xy.set_cursor(x, y, colour)
        self.xz.set_cursor(x, z, colour)
        self.yz.set_cursor(z, y, colour)
        self._update_markers()
        self._update_reference()

        step = (
            f"drawn to scale ({z_scale:.2f} px per slice)"
            if self.anisotropy
            else "Z spacing unknown: drawn one pixel per slice"
        )
        self.z_readout.setText(f"Slice {int(z) + 1} of {n_z}  ·  {step}")
        self.cursor_readout.setText(f"Cursor x {x:.0f}, y {y:.0f}, z {int(z)}")

    def _selected_position(self) -> tuple[float, float, float | None] | None:
        if self.selected_track is None:
            return None
        for row in self.rows_for_frame(self._frame):
            if row.get("track_id") is not None and int(row["track_id"]) == int(self.selected_track):
                xy = row_xy(row)
                if xy is not None:
                    return (xy[0], xy[1], row_z(row))
        return None

    def _follow_selection(self) -> None:
        position = self._selected_position()
        if position is None or self._stack is None:
            return
        x, y, z = position
        _, n_z, n_y, n_x = self._stack.shape
        self._cursor[0] = float(min(max(x, 0.0), n_x - 1))
        self._cursor[1] = float(min(max(y, 0.0), n_y - 1))
        if z is not None:
            z_index = int(min(max(round(z), 0), n_z - 1))
            self._cursor[2] = z_index
            self.z_slider.blockSignals(True)
            self.z_slider.setValue(z_index)
            self.z_slider.blockSignals(False)

    def _update_markers(self) -> None:
        if not (self.layers.centroids or self.layers.labels):
            for pane in (self.xy, self.xz, self.yz):
                pane.set_markers([])
            return
        cx, cy, cz = self._cursor
        xy_markers, xz_markers, yz_markers = [], [], []
        for row in self.rows_for_frame(self._frame):
            position = row_xy(row)
            if position is None or row.get("track_id") is None:
                continue
            track_id = int(row["track_id"])
            colour = track_color(track_id)
            text = str(track_id)
            x, y = position
            z = row_z(row)
            if z is None:
                xy_markers.append((x, y, colour, True, text))
                continue
            near_slice = abs(z - cz) <= SLICE_TOLERANCE
            xy_markers.append((x, y, colour, near_slice, text))
            if abs(y - cy) <= PLANE_TOLERANCE_PX:
                xz_markers.append((x, z, colour, True, text))
            if abs(x - cx) <= PLANE_TOLERANCE_PX:
                yz_markers.append((z, y, colour, True, text))
        self.xy.set_markers(xy_markers)
        self.xz.set_markers(xz_markers)
        self.yz.set_markers(yz_markers)

    def _update_reference(self) -> None:
        """Place the reference point on the panes, by the markers' tolerances.

        XY always shows it (solid on its own slice, faint elsewhere, so the
        user can find it from any Z); XZ and YZ show it only when their plane
        passes near it, as for a cell. A point set on the 2-D canvas has no Z
        and is drawn on XY only.
        """
        point = self.reference_point
        if point is None or self._stack is None or not getattr(self.layers, "reference", True):
            for pane in (self.xy, self.xz, self.yz):
                pane.set_reference(None)
            return
        x, y = point[0], point[1]
        z = point[2] if len(point) > 2 else None
        cx, cy, cz = self._cursor
        self.xy.set_reference((x, y), solid=z is None or abs(z - cz) <= SLICE_TOLERANCE)
        self.xz.set_reference(
            (x, z) if z is not None and abs(y - cy) <= PLANE_TOLERANCE_PX else None
        )
        self.yz.set_reference(
            (z, y) if z is not None and abs(x - cx) <= PLANE_TOLERANCE_PX else None
        )

    # --------------------------------------------------------------- clicks
    def _pick_or_move(self, x: float, y: float, z: float, *, select: bool) -> None:
        if self._picking:
            self.point_picked.emit(float(x), float(y), float(z))
            return
        if select:
            best_id, best = -1, 1e12
            for row in self.rows_for_frame(self._frame):
                position = row_xy(row)
                if position is None or row.get("track_id") is None:
                    continue
                rz = row_z(row)
                if rz is not None and abs(rz - z) > SLICE_TOLERANCE + 1:
                    continue
                distance = (position[0] - x) ** 2 + (position[1] - y) ** 2
                if distance < best:
                    best, best_id = distance, int(row["track_id"])
            if best <= PLANE_TOLERANCE_PX**2 * 4:
                self.track_clicked.emit(best_id)
                return
        self.set_cursor(x, y, z)

    def _xy_clicked(self, u: float, v: float) -> None:
        self._pick_or_move(u, v, self._cursor[2], select=True)

    def _xz_clicked(self, u: float, v: float) -> None:
        self._pick_or_move(u, self._cursor[1], v, select=False)

    def _yz_clicked(self, u: float, v: float) -> None:
        self._pick_or_move(self._cursor[0], v, u, select=False)
