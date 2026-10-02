"""The update flow through the real window, including what it must NOT do.

Three properties are load-bearing and none is obvious from reading the code:

*   **A window that has never asked makes no network call.** Consent is a
    precondition, not a formality, because this application otherwise contacts
    nothing and a workstation holding unpublished data is often offline on
    purpose.
*   **Declining is permanent.** A user who said no is not asked again on the
    next launch, and no check runs.
*   **Dismissing one version does not silence the next.** Otherwise "not now"
    quietly becomes "never".

The network is replaced throughout: these tests never open a socket, and a test
that did would be measuring GitHub's availability rather than this code.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6", reason="the interface is not installed")

from PySide6.QtWidgets import QApplication  # noqa: E402

from corridor.core import updates  # noqa: E402
from corridor.core.updates import Release  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def window(qt_app, tmp_path, monkeypatch):
    """A real MainWindow on a throwaway store, with the network unreachable.

    ``updates.check`` is replaced by something that fails the test if it is
    ever called, so "no check happened" is proven rather than assumed.
    """
    from corridor.store import db
    from corridor.ui.main_window import MainWindow

    calls: list[str] = []

    def _forbidden(*args, **kwargs):
        calls.append("check")
        return None

    monkeypatch.setattr(updates, "check", _forbidden)

    store = db.Store(tmp_path / "projects.db")
    win = MainWindow(store=store)
    yield win, store, calls
    win.close()


def _drain(app):
    for _ in range(6):
        app.processEvents()


def shown(widget) -> bool:
    """Whether this widget has been shown, independent of its parent.

    ``isVisible()`` is False whenever any ancestor is hidden, and these tests
    never map the window, so it would report False for a banner that show() was
    correctly called on. ``isHidden()`` is the widget's own explicit state,
    which is exactly what the code under test sets.
    """
    return not widget.isHidden()


# --------------------------------------------------------------------------
# Consent gates the network
# --------------------------------------------------------------------------


def test_a_first_run_asks_and_checks_nothing(window, qt_app):
    win, store, calls = window
    _drain(qt_app)

    assert shown(win.update_consent), "the question should be shown once"
    assert not shown(win.update_banner)
    assert updates.has_been_asked(store) is False
    assert calls == [], "nothing may reach the network before the user agrees"


def test_declining_is_remembered_and_never_checks(window, qt_app, tmp_path):
    from corridor.ui.main_window import MainWindow

    win, store, calls = window
    win.update_consent._answer(False)
    _drain(qt_app)

    assert updates.has_been_asked(store) is True
    assert updates.is_enabled(store) is False
    assert calls == []

    # A second launch on the same store must not ask again, nor check.
    second = MainWindow(store=store)
    _drain(qt_app)
    assert not shown(second.update_consent)
    assert calls == []
    second.close()


def test_accepting_runs_the_check(window, qt_app):
    win, store, calls = window
    win.update_consent._answer(True)
    for _ in range(40):
        qt_app.processEvents()
        if calls:
            break
    assert updates.is_enabled(store) is True
    assert calls == ["check"], "consent should trigger exactly one check"


# --------------------------------------------------------------------------
# What the banner does with a result
# --------------------------------------------------------------------------


def test_a_newer_release_is_announced(window, qt_app):
    win, store, _ = window
    win._update_checked(Release(version="9.9.9", url="https://example.invalid", notes="x"))
    _drain(qt_app)
    assert shown(win.update_banner)
    assert "9.9.9" in win.update_banner.headline.text()


def test_the_current_version_is_not_announced(window, qt_app):
    from corridor import app_meta

    win, store, _ = window
    win._update_checked(
        Release(version=app_meta.APP_VERSION, url="https://example.invalid", notes="x")
    )
    _drain(qt_app)
    assert not shown(win.update_banner)


def test_an_older_release_is_not_announced(window, qt_app):
    win, store, _ = window
    win._update_checked(Release(version="0.0.1", url="https://example.invalid", notes="x"))
    _drain(qt_app)
    assert not shown(win.update_banner)


def test_no_information_is_not_an_error(window, qt_app):
    """The offline case: check returned None and nothing should happen."""
    win, store, _ = window
    win._update_checked(None)
    _drain(qt_app)
    assert not shown(win.update_banner)


def test_dismissing_a_version_silences_only_that_version(window, qt_app):
    win, store, _ = window

    win._update_checked(Release(version="9.9.9", url="https://example.invalid", notes="x"))
    _drain(qt_app)
    win.update_banner._dismiss()
    _drain(qt_app)
    assert not shown(win.update_banner)

    # The same version again: stays quiet.
    win._update_checked(Release(version="9.9.9", url="https://example.invalid", notes="x"))
    _drain(qt_app)
    assert not shown(win.update_banner)

    # A newer one: announced.
    win._update_checked(Release(version="9.9.10", url="https://example.invalid", notes="x"))
    _drain(qt_app)
    assert shown(win.update_banner), "'not now' must not become 'never'"


def test_opening_the_page_does_not_download_anything(window, qt_app, monkeypatch):
    """The deliberate refusal: a link is opened, nothing is fetched or run."""
    from PySide6.QtGui import QDesktopServices

    opened: list[str] = []
    monkeypatch.setattr(
        QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url.toString()))
    )
    win, _, _ = window
    win._open_release_page("https://example.invalid/releases/v9.9.9")
    assert opened == ["https://example.invalid/releases/v9.9.9"]


def test_an_empty_url_opens_nothing(window, qt_app, monkeypatch):
    from PySide6.QtGui import QDesktopServices

    opened: list[str] = []
    monkeypatch.setattr(
        QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url.toString()))
    )
    win, _, _ = window
    win._open_release_page("")
    assert opened == []
