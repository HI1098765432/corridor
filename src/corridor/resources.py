"""Locating files that ship with the application.

The same code must work three ways: from a source checkout, from a PyInstaller
one-folder bundle, and from an installed copy under Program Files.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Name of the Cellpose model bundled with the application.
BUNDLED_MODEL_NAME = "cyto2_phase_microfluidic_KK1KK2_combi"

#: SHA-256 of the model this application was built against. Used only to tell
#: the user whether the model they are running is the expected one.
BUNDLED_MODEL_SHA256 = (
    "b33bdbdab395a27051b1bf10897b66888abcc24da3b3ddd41814fea970177cd6"
)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """Directory that contains bundled, read-only assets."""
    if is_frozen():
        base = getattr(sys, "_MEIPASS", None)
        if base:
            return Path(base)
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def asset(*parts: str) -> Path:
    return resource_root().joinpath("assets", *parts)


def bundled_model_path() -> Path | None:
    """The Cellpose model shipped with the app, if it is present."""
    candidates = [
        asset("models", BUNDLED_MODEL_NAME),
        resource_root() / "models" / BUNDLED_MODEL_NAME,
    ]
    if not is_frozen():
        # A source checkout can also use the model from the supplied data tree.
        repo = Path(__file__).resolve().parents[2]
        candidates.append(
            repo
            / "data"
            / "confinedmig_cellTrack"
            / "CellPose_TrainData"
            / "KK1KK2_combiModel"
            / "models"
            / BUNDLED_MODEL_NAME
        )
    env = os.environ.get("CORRIDOR_MODEL")
    if env:
        candidates.insert(0, Path(env))
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def icon_path() -> Path | None:
    for name in ("corridor.ico", "corridor.png"):
        p = asset(name)
        if p.exists():
            return p
    return None
