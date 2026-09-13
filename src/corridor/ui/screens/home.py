"""The first screen.

One thing to do, in the middle, with room around it.  Previous work sits
underneath, quiet enough to ignore and close enough to click.  There is no
tutorial: if the empty state needs explaining, it is the wrong empty state.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ... import app_meta
from ...store.db import ProjectRecord, status_label
from ..icons import icon, pixmap
from ..theme import PALETTE, RADIUS, SPACE
from ..widgets.common import (
    Badge,
    ClickableFrame,
    ghost_button,
    label,
    primary_button,
)

TIFF_SUFFIXES = {".tif", ".tiff", ".ome.tif"}


def is_tiff(path: str | Path) -> bool:
    name = str(path).lower()
    return name.endswith(".tif") or name.endswith(".tiff")


def first_tiff(mime: QMimeData) -> str | None:
    if not mime.hasUrls():
        return None
    for url in mime.urls():
        local = url.toLocalFile()
        if local and is_tiff(local):
            return local
    return None


class DropZone(QWidget):
    """The single obvious action."""

    file_chosen = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(260)
        self.setCursor(Qt.PointingHandCursor)
        self._hover = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE["2xl"], SPACE["2xl"], SPACE["2xl"], SPACE["2xl"])
        layout.setSpacing(SPACE["md"])
        layout.setAlignment(Qt.AlignCenter)

        glyph = label("")
        glyph.setPixmap(pixmap("image", PALETTE.text_tertiary, 44))
        glyph.setAlignment(Qt.AlignCenter)

        headline = label("Drop a time-lapse here", "title")
        headline.setAlignment(Qt.AlignCenter)

        hint = label("TIFF stack  ·  time first", "tertiary")
        hint.setAlignment(Qt.AlignCenter)

        browse = primary_button("Choose a file", self._browse)
        browse_row = QHBoxLayout()
        browse_row.addStretch(1)
        browse_row.addWidget(browse)
        browse_row.addStretch(1)

        layout.addWidget(glyph)
        layout.addSpacing(SPACE["xs"])
        layout.addWidget(headline)
        layout.addWidget(hint)
        layout.addSpacing(SPACE["lg"])
        layout.addLayout(browse_row)

    # -- interaction -------------------------------------------------------
    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a time-lapse", "", "TIFF stacks (*.tif *.tiff);;All files (*)"
        )
        if path:
            self.file_chosen.emit(path)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._browse()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if first_tiff(event.mimeData()):
            self._hover = True
            self.update()
            event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._hover = False
        self.update()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        path = first_tiff(event.mimeData())
        self._hover = False
        self.update()
        if path:
            event.acceptProposedAction()
            self.file_chosen.emit(path)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(rect, RADIUS["lg"] + 4, RADIUS["lg"] + 4)
        painter.fillPath(
            path, QColor(PALETTE.accent_wash if self._hover else PALETTE.surface)
        )
        pen = QPen(
            QColor(PALETTE.accent if self._hover else PALETTE.border_strong),
            1.6 if self._hover else 1.2,
        )
        pen.setStyle(Qt.DashLine)
        pen.setDashPattern([7, 6])
        painter.setPen(pen)
        painter.drawPath(path)
        painter.end()


class RecentRow(ClickableFrame):
    """One previous analysis."""

    open_requested = Signal(str)
    remove_requested = Signal(str)

    def __init__(self, record: ProjectRecord, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.record = record
        self.setProperty("role", "card")
        self.setMinimumHeight(70)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACE["lg"], SPACE["md"], SPACE["md"], SPACE["md"])
        layout.setSpacing(SPACE["lg"])

        thumb = label("")
        thumb.setFixedSize(44, 44)
        thumb.setAlignment(Qt.AlignCenter)
        preview = record.preview_path
        if preview.exists():
            from PySide6.QtGui import QPixmap

            image = QPixmap(str(preview)).scaled(
                44, 44, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            thumb.setPixmap(image)
            thumb.setStyleSheet(f"border-radius: {RADIUS['sm']}px;")
        else:
            thumb.setPixmap(pixmap("image", PALETTE.text_tertiary, 22))
            thumb.setStyleSheet(
                f"background: {PALETTE.surface_sunken}; border-radius: {RADIUS['sm']}px;"
            )

        text = QVBoxLayout()
        text.setSpacing(2)
        title = label(record.name, "subtitle")
        details: list[str] = []
        if record.shape_text():
            details.append(record.shape_text())
        if record.n_tracks is not None:
            details.append(f"{record.n_tracks} tracks")
        details.append(record.updated_at[:10])
        text.addWidget(title)
        text.addWidget(label("  ·  ".join(details), "tertiary"))

        tone = {
            "complete": "accent",
            "failed": "danger",
            "running": "warning",
            "cancelled": "warning",
        }.get(record.status, "neutral")
        badge = Badge(status_label(record.status), tone)

        remove = ghost_button("", "trash")
        remove.setToolTip("Remove from this list")
        remove.setFixedWidth(38)
        remove.clicked.connect(lambda: self.remove_requested.emit(self.record.id))

        layout.addWidget(thumb)
        layout.addLayout(text, 1)
        if not record.source_exists:
            layout.addWidget(Badge("File moved", "warning"))
        layout.addWidget(badge)
        layout.addWidget(remove)

        self.clicked.connect(lambda: self.open_requested.emit(self.record.id))


class HomeScreen(QWidget):
    """Empty state plus recent work."""

    file_chosen = Signal(str)
    project_opened = Signal(str)
    project_removed = Signal(str)
    settings_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # -- a very light top bar: identity left, settings right ------------
        bar = QWidget()
        bar.setFixedHeight(58)
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(SPACE["xl"], 0, SPACE["lg"], 0)
        wordmark = label(app_meta.APP_NAME, "subtitle")
        settings = ghost_button("", "settings")
        settings.setToolTip("Settings")
        settings.setFixedWidth(38)
        settings.clicked.connect(self.settings_requested.emit)
        bar_layout.addWidget(wordmark)
        bar_layout.addStretch(1)
        bar_layout.addWidget(settings)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        self._content = QVBoxLayout(content)
        self._content.setContentsMargins(0, SPACE["xl"], 0, SPACE["3xl"])
        self._content.setSpacing(SPACE["xl"])
        # With nothing to show underneath, the one action belongs in the middle
        # of the window rather than pinned to the top of a mostly empty page.
        self._content.addStretch(3)

        centred = QWidget()
        centred.setMaximumWidth(720)
        centred.setMinimumWidth(460)
        self._column = QVBoxLayout(centred)
        self._column.setContentsMargins(SPACE["xl"], 0, SPACE["xl"], 0)
        self._column.setSpacing(SPACE["xl"])

        self.drop_zone = DropZone()
        self.drop_zone.file_chosen.connect(self.file_chosen.emit)
        self._column.addWidget(self.drop_zone)
        self._column.setStretch(0, 0)

        self._recent_header = label("Recent", "secondary")
        self._recent_header.hide()
        self._column.addWidget(self._recent_header)

        self._recent_box = QVBoxLayout()
        self._recent_box.setSpacing(SPACE["sm"])
        self._column.addLayout(self._recent_box)
        self._column.addStretch(1)

        holder = QHBoxLayout()
        holder.addStretch(1)
        holder.addWidget(centred)
        holder.addStretch(1)
        self._content.addLayout(holder)
        self._content.addStretch(4)

        scroll.setWidget(content)
        outer.addWidget(bar)
        outer.addWidget(scroll, 1)

    # -- recent list -------------------------------------------------------
    def set_recent(self, records: list[ProjectRecord]) -> None:
        while self._recent_box.count():
            item = self._recent_box.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._recent_header.setVisible(bool(records))
        # Recent work pushes the drop zone up towards the top of the page.
        self._content.setStretch(0, 0 if records else 3)
        self._content.setStretch(self._content.count() - 1, 1 if records else 4)
        for record in records:
            row = RecentRow(record)
            row.open_requested.connect(self.project_opened.emit)
            row.remove_requested.connect(self.project_removed.emit)
            self._recent_box.addWidget(row)

    # -- window-wide drag and drop ----------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if first_tiff(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        path = first_tiff(event.mimeData())
        if path:
            event.acceptProposedAction()
            self.file_chosen.emit(path)
