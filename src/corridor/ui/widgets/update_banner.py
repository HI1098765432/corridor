"""Asking once whether to check for updates, and saying so quietly when there is one.

Two widgets, and the split between them is deliberate.

:class:`UpdateConsent` is asked **once**, on first run, before any network call
has been made. Until now this application contacted nothing, and that is worth
preserving by default: a microscope workstation holding unpublished data is
often offline on purpose, and software that quietly phones home is not welcome
on one. So the question is asked in plain language, "no" is remembered
permanently, and no answer at all is treated as no.

:class:`UpdateBanner` reports a newer version. It offers a link and it does not
download anything. A program that fetches an executable and runs it is the exact
shape of the thing every security guide warns about, and "but it is our own
executable" is precisely what an attacker who has compromised a release channel
is relying on. The user downloads it themselves, Windows checks the signature,
and a person decides.

Both are dismissible and neither blocks work. An update notice appearing over
somebody's analysis is an interruption, not a service.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ... import app_meta
from ...core.updates import Release
from ..theme import PALETTE, SPACE
from .common import ghost_button, label, primary_button


class _Strip(QFrame):
    """A single quiet row across the top of the window."""

    def __init__(self, tone: str = "accent", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("UpdateStrip")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        # The accent wash, not the accent: this is information, not an alarm,
        # and it sits above work the user came here to do.
        self.setStyleSheet(
            f"#UpdateStrip {{ background: {PALETTE.accent_wash}; "
            f"border-bottom: 1px solid {PALETTE.accent_border}; }}"
        )


class UpdateConsent(_Strip):
    """The one-time question. Nothing has touched the network when this appears."""

    answered = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(SPACE["lg"], SPACE["md"], SPACE["lg"], SPACE["md"])
        row.setSpacing(SPACE["md"])

        text = QVBoxLayout()
        text.setSpacing(2)
        headline = label("Check for updates?", "primary")
        font = headline.font()
        font.setBold(True)
        headline.setFont(font)
        detail = label(
            "Corridor has never contacted the internet. Checking asks GitHub once "
            "per session whether a newer version exists — it sends nothing about "
            "you, your images or your results, and it never downloads or installs "
            "anything on its own.",
            "secondary",
        )
        detail.setWordWrap(True)
        text.addWidget(headline)
        text.addWidget(detail)
        row.addLayout(text, 1)

        no = ghost_button("Stay offline", "", lambda: self._answer(False))
        yes = primary_button("Check for updates", lambda: self._answer(True))
        row.addWidget(no, 0, Qt.AlignTop)
        row.addWidget(yes, 0, Qt.AlignTop)

    def _answer(self, enabled: bool) -> None:
        self.answered.emit(enabled)
        self.hide()


class UpdateBanner(_Strip):
    """A newer version exists. Here is the link; the decision is yours."""

    dismissed = Signal(str)
    open_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._version = ""
        row = QHBoxLayout(self)
        row.setContentsMargins(SPACE["lg"], SPACE["md"], SPACE["lg"], SPACE["md"])
        row.setSpacing(SPACE["md"])

        text = QVBoxLayout()
        text.setSpacing(2)
        self.headline = label("", "primary")
        font = self.headline.font()
        font.setBold(True)
        self.headline.setFont(font)
        self.detail = label("", "secondary")
        self.detail.setWordWrap(True)
        text.addWidget(self.headline)
        text.addWidget(self.detail)
        row.addLayout(text, 1)

        self.later = ghost_button("Not now", "", self._dismiss)
        self.open = primary_button(
            "Open the download page", lambda: self.open_requested.emit(self._url)
        )
        row.addWidget(self.later, 0, Qt.AlignTop)
        row.addWidget(self.open, 0, Qt.AlignTop)
        self._url = ""
        self.hide()

    def show_release(self, release: Release) -> None:
        self._version = release.version
        self._url = release.url
        self.headline.setText(
            f"{app_meta.APP_NAME} {release.version} is available "
            f"(this is {app_meta.APP_VERSION})"
        )
        summary = release.summary(lines=3)
        self.detail.setText(
            summary or "Open the download page to see what changed."
        )
        self.open.setEnabled(bool(self._url))
        self.show()

    def _dismiss(self) -> None:
        # Remembered, so the same version is not announced at every launch.
        self.dismissed.emit(self._version)
        self.hide()
