"""Corridor 2.1 -- the deterministic 4D accuracy engine.

The thesis (``docs/ENGINE_4D.md``): the trained segmentation model is near its
peak at F1 ~0.73 on raw per-frame IoU-0.5; the accuracy still missing is
sitting unused *in the data* -- at the border, in the image gradient, in the
label truth, and across time and Z -- and deterministic mathematics recovers
it.  **AI proposes, math verifies, time and Z disambiguate.**  No model, and no
learned weight, touches a pixel in this package: every bot here imports numpy,
scipy and scikit-image only.

Each bot is one module, takes plain numpy arrays plus a frozen settings
dataclass, and returns a frozen result whose numbers are inspectable and whose
behaviour has a ground-truth test.  The two bots that land first are the
proposer *interface* (Bot 3, ``proposer``) and the referee (Bot 8,
``consensus``), joined by the orchestrator skeleton (``pipeline4d``).  The
remaining bots -- registration, static atlas, Z consensus, temporal delta,
measurement -- are imported lazily by the pipeline, so the package builds and
runs before they all exist (``docs/ENGINE_4D.md`` SS3).

Only the lightweight, model-free public surface is re-exported here; the
pipeline is reached through its own module so that importing the package never
forces the lazy bots to resolve.
"""

from __future__ import annotations

from .consensus import (
    ARTIFACT,
    CHANNEL_WALL,
    OUTSIDE_DEVICE,
    STATIC_BACKGROUND,
    UNKNOWN,
    VALID_CELL_REGION,
    ConsensusConfig,
    ConsensusState,
    Decision,
    Evidence,
    referee,
)
from .proposer import (
    GIVEN_MASKS,
    GivenMasksProposer,
    Normalization,
    Proposal,
    Proposer,
    ProposerBackend,
    ProposerUnavailable,
)

__all__ = [
    # proposer (Bot 3)
    "Proposal",
    "Proposer",
    "ProposerBackend",
    "ProposerUnavailable",
    "Normalization",
    "GivenMasksProposer",
    "GIVEN_MASKS",
    # consensus / referee (Bot 8)
    "ConsensusState",
    "ConsensusConfig",
    "Evidence",
    "Decision",
    "referee",
    "STATIC_BACKGROUND",
    "CHANNEL_WALL",
    "OUTSIDE_DEVICE",
    "ARTIFACT",
    "VALID_CELL_REGION",
    "UNKNOWN",
]
