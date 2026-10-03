"""Corridor 2.1 -- the deterministic 4D accuracy engine (``docs/ENGINE_4D.md``).

A foundation segmenter proposes, deterministic mathematics verifies, and the
complete T x Z x Y x X experiment resolves ambiguity.  Every stage here is a
narrow, deterministic worker ("bot") with explicit inputs and outputs, its
numerical evidence exposed, and a ground-truth test.  **No model touches a
pixel in this package**: the only dependencies are numpy, scipy and
scikit-image.  torch and cellpose are never imported from ``corridor.engine``.

This module currently exposes two bots:

* :mod:`corridor.engine.registration4d` -- Bot 1, whole-volume registration.
* :mod:`corridor.engine.static_atlas`    -- Bot 2, the static device atlas.
"""

from __future__ import annotations

from .registration4d import (
    RegistrationConfig,
    RegistrationResult,
    RegistrationRow,
    choose_reference_timepoint,
    register_stack,
)
from .static_atlas import (
    ARTIFACT,
    CHANNEL_WALL,
    CLASS_CODES,
    CLASS_NAMES,
    OUTSIDE_DEVICE,
    STATIC_BACKGROUND,
    UNKNOWN,
    VALID_CELL_REGION,
    AtlasConfig,
    AtlasResult,
    build_atlas,
)

__all__ = [
    # Bot 1 -- registration
    "RegistrationConfig",
    "RegistrationResult",
    "RegistrationRow",
    "register_stack",
    "choose_reference_timepoint",
    # Bot 2 -- static atlas
    "AtlasConfig",
    "AtlasResult",
    "build_atlas",
    "CLASS_NAMES",
    "CLASS_CODES",
    "STATIC_BACKGROUND",
    "CHANNEL_WALL",
    "OUTSIDE_DEVICE",
    "ARTIFACT",
    "VALID_CELL_REGION",
    "UNKNOWN",
]
