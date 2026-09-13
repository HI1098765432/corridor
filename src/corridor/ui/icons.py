"""Vector glyphs drawn at runtime.

Icons are painted rather than loaded so that they stay sharp at any display
scaling, recolour with the palette, and add no binary assets to the installer.
The set is deliberately tiny: an interface that needs thirty icons is usually
one that needs fewer features on screen at once.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

from .theme import PALETTE


def _pen(painter: QPainter, colour: str, width: float = 1.6) -> None:
    pen = QPen(QColor(colour), width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)


def _draw(name: str, painter: QPainter, colour: str) -> None:
    """Glyphs are drawn inside a 24x24 box."""
    _pen(painter, colour)

    if name == "folder":
        path = QPainterPath()
        path.moveTo(3.5, 7.5)
        path.lineTo(9.5, 7.5)
        path.lineTo(11.5, 10.0)
        path.lineTo(20.5, 10.0)
        path.lineTo(20.5, 18.5)
        path.lineTo(3.5, 18.5)
        path.closeSubpath()
        painter.drawPath(path)

    elif name == "image":
        painter.drawRoundedRect(QRectF(3.5, 5.5, 17, 13), 2.5, 2.5)
        painter.drawEllipse(QPointF(9.0, 10.0), 1.7, 1.7)
        path = QPainterPath()
        path.moveTo(5.0, 17.0)
        path.lineTo(10.5, 12.0)
        path.lineTo(14.0, 15.0)
        path.lineTo(16.5, 12.8)
        path.lineTo(19.5, 16.0)
        painter.drawPath(path)

    elif name == "play":
        path = QPainterPath()
        path.moveTo(8.0, 5.5)
        path.lineTo(18.5, 12.0)
        path.lineTo(8.0, 18.5)
        path.closeSubpath()
        painter.setBrush(QColor(colour))
        painter.drawPath(path)

    elif name == "pause":
        painter.setBrush(QColor(colour))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(QRectF(8.0, 5.5, 3.4, 13), 1.2, 1.2)
        painter.drawRoundedRect(QRectF(13.6, 5.5, 3.4, 13), 1.2, 1.2)

    elif name == "step-back":
        painter.setBrush(QColor(colour))
        painter.setPen(Qt.NoPen)
        path = QPainterPath()
        path.moveTo(16.5, 6.0)
        path.lineTo(16.5, 18.0)
        path.lineTo(8.5, 12.0)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawRoundedRect(QRectF(6.0, 6.0, 2.0, 12), 1.0, 1.0)

    elif name == "step-forward":
        painter.setBrush(QColor(colour))
        painter.setPen(Qt.NoPen)
        path = QPainterPath()
        path.moveTo(7.5, 6.0)
        path.lineTo(7.5, 18.0)
        path.lineTo(15.5, 12.0)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawRoundedRect(QRectF(16.0, 6.0, 2.0, 12), 1.0, 1.0)

    elif name == "export":
        path = QPainterPath()
        path.moveTo(12.0, 15.5)
        path.lineTo(12.0, 4.5)
        painter.drawPath(path)
        arrow = QPainterPath()
        arrow.moveTo(8.0, 8.5)
        arrow.lineTo(12.0, 4.5)
        arrow.lineTo(16.0, 8.5)
        painter.drawPath(arrow)
        tray = QPainterPath()
        tray.moveTo(4.5, 14.0)
        tray.lineTo(4.5, 19.5)
        tray.lineTo(19.5, 19.5)
        tray.lineTo(19.5, 14.0)
        painter.drawPath(tray)

    elif name == "settings":
        painter.drawEllipse(QPointF(12.0, 12.0), 3.0, 3.0)
        painter.drawEllipse(QPointF(12.0, 12.0), 7.6, 7.6)
        for dx, dy in ((0, -9.2), (0, 9.2), (-9.2, 0), (9.2, 0)):
            painter.drawLine(
                QPointF(12 + dx * 0.72, 12 + dy * 0.72),
                QPointF(12 + dx * 0.95, 12 + dy * 0.95),
            )

    elif name == "layers":
        for offset in (0.0, 3.6, 7.2):
            path = QPainterPath()
            path.moveTo(12.0, 4.5 + offset)
            path.lineTo(19.5, 8.0 + offset)
            path.lineTo(12.0, 11.5 + offset)
            path.lineTo(4.5, 8.0 + offset)
            path.closeSubpath()
            painter.drawPath(path)

    elif name == "target":
        painter.drawEllipse(QPointF(12.0, 12.0), 7.0, 7.0)
        painter.drawEllipse(QPointF(12.0, 12.0), 2.0, 2.0)
        painter.drawLine(QPointF(12, 2.5), QPointF(12, 5.5))
        painter.drawLine(QPointF(12, 18.5), QPointF(12, 21.5))
        painter.drawLine(QPointF(2.5, 12), QPointF(5.5, 12))
        painter.drawLine(QPointF(18.5, 12), QPointF(21.5, 12))

    elif name == "route":
        painter.drawEllipse(QPointF(6.5, 17.5), 2.4, 2.4)
        painter.drawEllipse(QPointF(17.5, 6.5), 2.4, 2.4)
        path = QPainterPath()
        path.moveTo(8.6, 16.0)
        path.cubicTo(13.0, 15.0, 11.0, 9.0, 15.6, 7.8)
        painter.drawPath(path)

    elif name == "grid":
        for x in (7.0, 12.0, 17.0):
            painter.drawLine(QPointF(x, 4.5), QPointF(x, 19.5))

    elif name == "warning":
        path = QPainterPath()
        path.moveTo(12.0, 4.5)
        path.lineTo(21.0, 19.5)
        path.lineTo(3.0, 19.5)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawLine(QPointF(12, 10.0), QPointF(12, 14.5))
        painter.drawPoint(QPointF(12, 17.0))

    elif name == "check":
        path = QPainterPath()
        path.moveTo(5.5, 12.5)
        path.lineTo(10.0, 17.0)
        path.lineTo(18.5, 7.5)
        painter.drawPath(path)

    elif name == "close":
        painter.drawLine(QPointF(6.5, 6.5), QPointF(17.5, 17.5))
        painter.drawLine(QPointF(17.5, 6.5), QPointF(6.5, 17.5))

    elif name == "chevron-down":
        path = QPainterPath()
        path.moveTo(6.5, 9.5)
        path.lineTo(12.0, 15.0)
        path.lineTo(17.5, 9.5)
        painter.drawPath(path)

    elif name == "chevron-right":
        path = QPainterPath()
        path.moveTo(9.5, 6.5)
        path.lineTo(15.0, 12.0)
        path.lineTo(9.5, 17.5)
        painter.drawPath(path)

    elif name == "back":
        path = QPainterPath()
        path.moveTo(11.0, 6.5)
        path.lineTo(5.5, 12.0)
        path.lineTo(11.0, 17.5)
        painter.drawPath(path)
        painter.drawLine(QPointF(5.5, 12.0), QPointF(19.0, 12.0))

    elif name == "hash":
        painter.drawLine(QPointF(9.5, 5.0), QPointF(7.8, 19.0))
        painter.drawLine(QPointF(16.0, 5.0), QPointF(14.3, 19.0))
        painter.drawLine(QPointF(5.2, 9.5), QPointF(18.8, 9.5))
        painter.drawLine(QPointF(4.7, 14.5), QPointF(18.3, 14.5))

    elif name == "plus":
        painter.drawLine(QPointF(12, 6.0), QPointF(12, 18.0))
        painter.drawLine(QPointF(6.0, 12), QPointF(18.0, 12))

    elif name == "trash":
        painter.drawLine(QPointF(4.5, 7.0), QPointF(19.5, 7.0))
        path = QPainterPath()
        path.moveTo(6.5, 7.0)
        path.lineTo(7.5, 19.5)
        path.lineTo(16.5, 19.5)
        path.lineTo(17.5, 7.0)
        painter.drawPath(path)
        painter.drawLine(QPointF(9.5, 7.0), QPointF(10.0, 4.5))
        painter.drawLine(QPointF(14.5, 7.0), QPointF(14.0, 4.5))
        painter.drawLine(QPointF(10.0, 4.5), QPointF(14.0, 4.5))

    elif name == "zoom-reset":
        painter.drawEllipse(QPointF(10.5, 10.5), 6.0, 6.0)
        painter.drawLine(QPointF(15.0, 15.0), QPointF(20.0, 20.0))
        painter.drawLine(QPointF(7.5, 10.5), QPointF(13.5, 10.5))


def icon(name: str, colour: str | None = None, size: int = 24) -> QIcon:
    """Return a crisp icon for ``name`` in ``colour``."""
    colour = colour or PALETTE.text_secondary
    scale = size / 24.0
    pixmap = QPixmap(size, size)
    pixmap.setDevicePixelRatio(1.0)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.scale(scale, scale)
    _draw(name, painter, colour)
    painter.end()
    return QIcon(pixmap)


def pixmap(name: str, colour: str | None = None, size: int = 24) -> QPixmap:
    colour = colour or PALETTE.text_secondary
    scale = size / 24.0
    result = QPixmap(size, size)
    result.fill(Qt.transparent)
    painter = QPainter(result)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.scale(scale, scale)
    _draw(name, painter, colour)
    painter.end()
    return result
