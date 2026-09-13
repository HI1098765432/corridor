"""Single source of truth for application identity and version."""

from __future__ import annotations

APP_NAME = "Corridor"
APP_TAGLINE = "Confined cell migration analysis"
APP_VERSION = "1.1.0"
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
