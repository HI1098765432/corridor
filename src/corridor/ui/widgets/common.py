"""Small shared building blocks."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..icons import icon
from ..theme import PALETTE, RADIUS, SPACE, TYPE


def label(text: str, role: str = "", parent: QWidget | None = None) -> QLabel:
    widget = QLabel(text, parent)
    if role:
        widget.setProperty("role", role)
    widget.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return widget


def divider(vertical: bool = False) -> QFrame:
    line = QFrame()
    line.setProperty("role", "vdivider" if vertical else "divider")
    if vertical:
        line.setFixedWidth(1)
        line.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
    else:
        line.setFixedHeight(1)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return line


def spacer(width: int = 0, height: int = 0) -> QWidget:
    widget = QWidget()
    widget.setFixedSize(QSize(width, height) if (width or height) else QSize(0, 0))
    if not width and not height:
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return widget


def primary_button(text: str, on_click: Callable[[], None] | None = None) -> QPushButton:
    button = QPushButton(text)
    button.setProperty("variant", "primary")
    button.setCursor(Qt.PointingHandCursor)
    if on_click:
        button.clicked.connect(on_click)
    return button


def ghost_button(
    text: str, glyph: str = "", on_click: Callable[[], None] | None = None
) -> QPushButton:
    button = QPushButton(text)
    button.setProperty("variant", "ghost")
    button.setCursor(Qt.PointingHandCursor)
    if glyph:
        button.setIcon(icon(glyph, PALETTE.text_secondary, 18))
    if on_click:
        button.clicked.connect(on_click)
    return button


def quiet_button(
    text: str, glyph: str = "", on_click: Callable[[], None] | None = None
) -> QPushButton:
    button = QPushButton(text)
    button.setCursor(Qt.PointingHandCursor)
    if glyph:
        button.setIcon(icon(glyph, PALETTE.text_secondary, 18))
    if on_click:
        button.clicked.connect(on_click)
    return button


def layer_toggle(text: str, glyph: str, checked: bool = True) -> QToolButton:
    button = QToolButton()
    button.setText(f"  {text}")
    button.setIcon(icon(glyph, PALETTE.text_secondary, 17))
    button.setCheckable(True)
    button.setChecked(checked)
    button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    button.setCursor(Qt.PointingHandCursor)
    return button


class Card(QFrame):
    """A bordered surface with generous internal padding."""

    def __init__(self, parent: QWidget | None = None, padding: int = SPACE["xl"]) -> None:
        super().__init__(parent)
        self.setProperty("role", "card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(padding, padding, padding, padding)
        self.body.setSpacing(SPACE["md"])


class Metric(QWidget):
    """A number with a quiet caption beneath it."""

    def __init__(
        self, caption: str, value: str = "—", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        self._value = label(value, "metric")
        self._caption = label(caption.upper(), "tertiary")
        font = self._caption.font()
        font.setLetterSpacing(QFont.PercentageSpacing, 108)
        self._caption.setFont(font)
        layout.addWidget(self._value)
        layout.addWidget(self._caption)

    def set_value(self, value: str) -> None:
        self._value.setText(value)


class Field(QWidget):
    """A labelled value shown as one quiet row."""

    def __init__(
        self, name: str, value: str = "—", hint: str = "", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["md"])
        self._name = label(name, "secondary")
        self._value = label(value)
        self._value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self._name)
        layout.addStretch(1)
        layout.addWidget(self._value)
        if hint:
            self.setToolTip(hint)

    def set_value(self, value: str, hint: str = "") -> None:
        self._value.setText(value)
        if hint:
            self.setToolTip(hint)


class StackedField(QWidget):
    """A name, a value, and where the value came from, on separate lines.

    Used where the provenance matters as much as the number: a calibration
    read from the file and one typed in by hand must not look the same.
    """

    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(SPACE["md"])
        self._name = label(name, "secondary")
        self._value = label("—")
        self._value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(self._name)
        top.addStretch(1)
        top.addWidget(self._value)
        layout.addLayout(top)
        self._source = label("", "tertiary")
        self._source.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self._source)

    def set_value(self, value: str, source: str = "") -> None:
        self._value.setText(value)
        self._source.setText(source)
        self._source.setVisible(bool(source))


class Badge(QLabel):
    """A small status pill."""

    TONES = {
        "neutral": (PALETTE.surface_sunken, PALETTE.text_secondary),
        "accent": (PALETTE.accent_wash, PALETTE.accent),
        "warning": (PALETTE.warning_wash, PALETTE.warning),
        "danger": (PALETTE.danger_wash, PALETTE.danger),
    }

    def __init__(self, text: str, tone: str = "neutral", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.set_tone(tone)

    def set_tone(self, tone: str) -> None:
        background, foreground = self.TONES.get(tone, self.TONES["neutral"])
        self.setStyleSheet(
            f"background: {background}; color: {foreground};"
            f"border-radius: {RADIUS['pill']}px; padding: 3px 10px;"
            f"font-size: {TYPE['caption']}px; font-weight: 600;"
        )


class Spinner(QWidget):
    """An indeterminate activity ring, used only while something really runs."""

    def __init__(self, parent: QWidget | None = None, size: int = 18) -> None:
        super().__init__(parent)
        self._angle = 0
        self.setFixedSize(size, size)
        from PySide6.QtCore import QTimer

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start(40)
        self.show()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def _tick(self) -> None:
        self._angle = (self._angle + 12) % 360
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect().adjusted(2, 2, -2, -2)
        painter.setPen(QPen(QColor(PALETTE.border), 2))
        painter.drawEllipse(rect)
        pen = QPen(QColor(PALETTE.accent), 2)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, -self._angle * 16, 100 * 16)
        painter.end()


class ClickableFrame(QFrame):
    """A card that behaves like a button."""

    clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)
