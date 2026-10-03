"""Corridor 2.1 -- the deterministic 4D accuracy engine.

The design contract is ``docs/ENGINE_4D.md``.  A foundation segmenter proposes
masks; the bots in this package verify, link and disambiguate them with
deterministic mathematics only.  **No bot in this package imports torch or
cellpose, and none touches a pixel the proposer did not already label.**  Each
bot takes plain numpy arrays plus a frozen settings dataclass, returns a frozen
result dataclass with ``to_dict()`` for its evidence file, and carries a
ground-truth test with a known answer.

The two bots defined here are the T x Z disambiguators of section 2:

* :mod:`corridor.engine.z_consensus` (Bot 4) -- links 2-D slice components into
  3-D objects across Z, repairing a dropped slice, deciding only Z membership.
* :mod:`corridor.engine.temporal_delta` (Bot 5) -- measures the forward/backward
  consistency ``E_FB`` of a proposed correspondence across time.
"""

from __future__ import annotations
"""Corridor 2.1 deterministic 4D engine (design contract: ``docs/ENGINE_4D.md``).

Every bot here is a narrow, deterministic worker on plain numpy arrays: a
foundation segmenter proposes, this mathematics verifies, and the full T x Z x
Y x X experiment resolves ambiguity.  No language model and no trained network
touches a pixel in this package -- the only dependencies are numpy, scipy and
scikit-image.  Each bot exposes its numerical evidence (``to_dict``) and is
validated against synthetic ground truth with a known answer.

This file deliberately does not import the bot modules, so importing one bot
never drags in another's dependencies.


__all__ = ["object4d", "surface_delta", "uncertainty"]
