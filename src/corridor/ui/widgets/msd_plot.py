"""A small log-log MSD plot for one track, painted with QPainter.

QtCharts and matplotlib are excluded from the build (packaging/corridor.spec),
so this follows the Sparkline approach: one painter, no dependency.

What it shows, and why:

*   **Lag time in hours on x, MSD in µm² on y, both logarithmic.** A power
    law ``MSD ~ t^alpha`` is a straight line there, and alpha is its slope,
    which is the number the detail panel reports beside it.
*   **Points weighted by n_pairs.** The longest lags are averaged over the
    fewest observation pairs and are the least reliable (that is why the fit
    stops at ``msd_max_lag_fraction``), so a point's area grows with the pairs
    behind it and a well-supported lag reads heavier than a two-pair one.
*   Reference slopes of 1 (diffusive) and 2 (ballistic) through the first
    point, faint, so a reader can see where the curve sits without trusting
    a number.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..analysis_view import MsdSeries
from ..theme import PALETTE, TYPE

EMPTY_TEXT = "not enough lags"


def _decade_ticks(lo: float, hi: float) -> list[float]:
    start, stop = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    return [10.0**k for k in range(start, stop + 1) if lo <= 10.0**k <= hi]


def _tick_text(value: float) -> str:
    if value >= 1000 or value < 0.01:
        return f"1e{int(round(math.log10(value)))}"
    return f"{value:g}"


class MsdPlot(QWidget):
    """MSD against lag for the selected track."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(150)
        self._series = MsdSeries((), "h", "µm²")
        self._colour = PALETTE.accent
        self._message = EMPTY_TEXT

    # ----------------------------------------------------------------- state
    def set_series(self, series: MsdSeries, colour: str, empty_message: str = EMPTY_TEXT) -> None:
        self._series = series
        self._colour = colour
        self._message = empty_message
        self.update()

    def clear(self) -> None:
        self.set_series(MsdSeries((), "h", "µm²"), PALETTE.accent)

    @property
    def n_points(self) -> int:
        return len(self._series.points)

    @property
    def is_empty(self) -> bool:
        return not self._series.points

    def point_radii(self) -> list[float]:
        """Marker radius of each point, in widget pixels (area ~ n_pairs)."""
        points = self._series.points
        if not points:
            return []
        most = max(max(p.n_pairs for p in points), 1)
        return [2.0 + 4.0 * math.sqrt(max(p.n_pairs, 0) / most) for p in points]

    # -------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(QRectF(self.rect()), QColor(PALETTE.surface_sunken))
        font = QFont()
        font.setPointSize(max(7, TYPE["caption"] - 3))
        painter.setFont(font)

        if self.is_empty:
            painter.setPen(QColor(PALETTE.text_tertiary))
            painter.drawText(QRectF(self.rect()), Qt.AlignCenter, self._message)
            painter.end()
            return

        points = self._series.points
        plot = QRectF(self.rect()).adjusted(40, 8, -10, -24)
        lags = [p.lag for p in points]
        msds = [p.msd for p in points]
        x_lo, x_hi = min(lags), max(lags)
        y_lo, y_hi = min(msds), max(msds)
        # Pad by a fraction of a decade so the end points are not on the frame.
        x_lo, x_hi = x_lo / 1.25, x_hi * 1.25
        y_lo, y_hi = y_lo / 1.6, y_hi * 1.6
        lx0, lx1 = math.log10(x_lo), math.log10(x_hi)
        ly0, ly1 = math.log10(y_lo), math.log10(y_hi)

        def to_point(lag: float, msd: float) -> QPointF:
            return QPointF(
                plot.left() + (math.log10(lag) - lx0) / (lx1 - lx0) * plot.width(),
                plot.bottom() - (math.log10(msd) - ly0) / (ly1 - ly0) * plot.height(),
            )

        # Frame and decade ticks.
        axis_pen = QPen(QColor(PALETTE.border_strong), 1)
        painter.setPen(axis_pen)
        painter.drawRect(plot)
        painter.setPen(QColor(PALETTE.text_tertiary))
        for tick in _decade_ticks(x_lo, x_hi):
            x = to_point(tick, y_lo).x()
            painter.drawLine(QPointF(x, plot.bottom()), QPointF(x, plot.bottom() + 3))
            painter.drawText(QRectF(x - 20, plot.bottom() + 3, 40, 12), Qt.AlignCenter, _tick_text(tick))
        for tick in _decade_ticks(y_lo, y_hi):
            y = to_point(x_lo, tick).y()
            painter.drawLine(QPointF(plot.left() - 3, y), QPointF(plot.left(), y))
            painter.drawText(QRectF(0, y - 6, plot.left() - 5, 12), Qt.AlignRight | Qt.AlignVCenter, _tick_text(tick))

        lag_unit = "lag (h)" if self._series.lag_unit == "h" else "lag (frames)"
        painter.drawText(
            QRectF(plot.left(), self.height() - 12, plot.width(), 12), Qt.AlignRight, lag_unit
        )
        painter.drawText(QRectF(2, 0, 60, 10), Qt.AlignLeft, f"MSD ({self._series.msd_unit})")

        # Reference slopes through the first point.
        painter.save()
        painter.setClipRect(plot)
        first = points[0]
        for slope in (1.0, 2.0):
            guide = QPen(QColor(PALETTE.text_tertiary), 1, Qt.DotLine)
            guide.setCosmetic(True)
            painter.setPen(guide)
            end_msd = first.msd * (x_hi / first.lag) ** slope
            painter.drawLine(to_point(first.lag, first.msd), to_point(x_hi, end_msd))
        painter.restore()

        colour = QColor(self._colour)
        line = QColor(colour)
        line.setAlpha(120)
        pen = QPen(line, 1.2)
        painter.setPen(pen)
        for a, b in zip(points, points[1:]):
            painter.drawLine(to_point(a.lag, a.msd), to_point(b.lag, b.msd))

        painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
        painter.setBrush(colour)
        for point, radius in zip(points, self.point_radii()):
            painter.drawEllipse(to_point(point.lag, point.msd), radius, radius)
        painter.end()
