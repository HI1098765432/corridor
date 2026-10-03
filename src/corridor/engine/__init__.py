"""Corridor 2.1 deterministic 4D engine (design contract: ``docs/ENGINE_4D.md``).

Every bot here is a narrow, deterministic worker on plain numpy arrays: a
foundation segmenter proposes, this mathematics verifies, and the full T x Z x
Y x X experiment resolves ambiguity.  No language model and no trained network
touches a pixel in this package -- the only dependencies are numpy, scipy and
scikit-image.  Each bot exposes its numerical evidence (``to_dict``) and is
validated against synthetic ground truth with a known answer.

This file deliberately does not import the bot modules, so importing one bot
never drags in another's dependencies.
"""

from __future__ import annotations

__all__ = ["object4d", "surface_delta", "uncertainty"]
