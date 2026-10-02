"""Locating files that ship with the application.

The same code must work three ways: from a source checkout, from a PyInstaller
one-folder bundle, and from an installed copy under Program Files.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Name of the Cellpose model bundled with the application. The registry
#: (``assets/model_registry.json``) is the authority; this is its file name,
#: kept for v1 callers.
BUNDLED_MODEL_NAME = "cyto2_phase_microfluidic_KK1KK2_combi"

#: SHA-256 of the model this application was built against, pinned since
#: 1.0.0. ``tests/test_model_registry.py`` checks the registry agrees.
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
    """The verified production 2-D model file, or None.

    Delegates to :mod:`corridor.core.model_registry`: every candidate is
    hashed and only a file matching the registry's SHA-256 is returned.  1.x
    returned the first file that merely *existed*, with a ``CORRIDOR_MODEL``
    variable checked before everything else, so any file could be
    substituted without a trace; that variable is gone.  The developer
    override (``CORRIDOR_DEVELOPER`` + ``CORRIDOR_DEVELOPER_MODEL``) is not
    the production model either, so it is not returned here even when set.
    """
    # Imported here: model_registry imports this module.
    from corridor.core import model_registry as registry

    try:
        spec = registry.production_spec("2D")
    except registry.ModelUnavailable:
        return None
    hasher = getattr(registry, "_sha256_cached", registry.sha256_file)
    for path in registry.candidate_paths(spec):
        try:
            if path.is_file() and hasher(path) == spec.sha256:
                return path.resolve()
        except OSError:
            continue
    return None


def icon_path() -> Path | None:
    for name in ("corridor.ico", "corridor.png"):
        p = asset(name)
        if p.exists():
            return p
    return None
