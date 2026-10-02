"""Application identity. The version itself lives in ``_version.py``.

``APP_VERSION`` stays importable from here because the code that reads it
binds it early: ``core.updates.is_newer`` takes it as a default argument at
import time, and the AppUserModelID in ``ui.app`` embeds it. Re-exporting the
one source keeps every one of those readers on the same value; a second literal
here is exactly the drift ``tests/test_version.py`` exists to catch.
"""

from __future__ import annotations

from ._version import __version__ as APP_VERSION

APP_NAME = "Corridor"
APP_TAGLINE = "Confined cell migration analysis"
APP_PUBLISHER = "Corridor"
APP_ID = "Corridor"

# Windows registry / install identity. Stable across versions on purpose:
# changing it would orphan the previous installation's uninstaller.
APP_GUID = "{6F4B1D0E-2C7A-4F63-9E11-0A5C8B3D7A21}"

# Where user data lives: %LOCALAPPDATA%/Corridor
LOCAL_DIR_NAME = "Corridor"

__all__ = [
    "APP_NAME",
    "APP_TAGLINE",
    "APP_VERSION",
    "APP_PUBLISHER",
    "APP_ID",
    "APP_GUID",
    "LOCAL_DIR_NAME",
]
