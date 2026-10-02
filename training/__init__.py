"""Research: everything that trains, evaluates or curates data for Corridor.

Nothing in ``src/corridor`` imports this package and the installer never
contains it (``docs/NEXT_GENERATION.md`` section 9). It may import
``corridor`` -- the scorer is ``corridor.core.metrics``, so a research figure
and an application figure mean the same thing -- but never the other way round.

Run modules from the repository root, e.g. ``python -m training.datasets``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: The supplied stills. Only the canonical KK1/KK2 folders are ever walked:
#: KK1_KK2_combi/ holds byte-identical copies and would double-count.
TRAIN_ROOT = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
SAMPLE_ROOT = ROOT / "data" / "confinedmig_cellTrack" / "sample_data"
REGISTRY_DIR = ROOT / "build" / "registry"

if str(ROOT / "src") not in sys.path:
    # Scripts and tests run from a checkout without installing corridor.
    sys.path.insert(0, str(ROOT / "src"))
