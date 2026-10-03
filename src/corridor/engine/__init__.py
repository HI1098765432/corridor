"""Corridor 2.1 — the deterministic 4D accuracy engine.

Design contract: ``docs/ENGINE_4D.md``.

The thesis: the trained segmentation model is near its peak at F1 ~0.73 on raw
per-frame IoU-0.5, and per-frame deterministic post-processing was *measured*
(``docs/deterministic_ceiling_KK2.json``) to carry it only to ~0.747 — because
the residual error lives in the **time and Z dimensions**, not in a single
frame.  This package recovers it there.  **AI proposes, deterministic
mathematics verifies, and the full T×Z×Y×X experiment resolves ambiguity.**

Every bot is one module that takes plain numpy arrays plus a frozen settings
dataclass and returns a frozen result whose numbers are inspectable
(``to_dict``) and whose behaviour has a ground-truth test.  **No bot imports
torch or cellpose, and none touches a pixel the proposer did not label** — the
only dependencies are numpy, scipy and scikit-image.

Import a bot by its submodule so that importing one never drags in another's
work::

    from corridor.engine import z_consensus, temporal_delta, consensus

The orchestrator (:mod:`corridor.engine.pipeline4d`) imports the bots lazily,
so the package builds and runs before every bot exists.
"""

from __future__ import annotations

__all__ = [
    "proposer",
    "consensus",
    "z_consensus",
    "temporal_delta",
    "object4d",
    "surface_delta",
    "uncertainty",
    "pipeline4d",
]
