"""Application bootstrap."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from .. import app_meta, resources
from ..store import db
from .theme import PALETTE, stylesheet


def _configure_windows_identity() -> None:
    """Give the app its own taskbar identity rather than Python's."""
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            f"{app_meta.APP_PUBLISHER}.{app_meta.APP_ID}.{app_meta.APP_VERSION}"
        )
    except Exception:  # noqa: BLE001 - cosmetic only
        pass


def create_application(argv: list[str] | None = None) -> QApplication:
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(app_meta.APP_NAME)
    app.setApplicationDisplayName(app_meta.APP_NAME)
    app.setOrganizationName(app_meta.APP_PUBLISHER)
    app.setApplicationVersion(app_meta.APP_VERSION)

    icon_file = resources.icon_path()
    if icon_file:
        app.setWindowIcon(QIcon(str(icon_file)))

    font = QFont("Segoe UI", 10)
    font.setHintingPreference(QFont.PreferFullHinting)
    app.setFont(font)
    app.setStyleSheet(stylesheet())
    return app


def run_app(files: list[str] | None = None) -> int:
    _configure_windows_identity()
    app = create_application(sys.argv)

    from .main_window import MainWindow

    window = MainWindow(db.Store())
    window.show()

    for candidate in files or []:
        path = Path(candidate)
        if path.exists() and path.suffix.lower() in (".tif", ".tiff"):
            window.open_file(str(path))
            break

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_app(sys.argv[1:]))
