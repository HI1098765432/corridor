"""Does a cell's shape now predict how far it migrates next? (research only)

Design contract: ``docs/NEXT_GENERATION.md`` section 9, last bullet. Results and
the honest reading of them: ``docs/PREDICTION.md``.

Nothing in ``src/corridor`` imports this package and the installer never
contains it. It reads Corridor result folders (``masks.npz``, ``tracks.csv``,
``run.json``) and never runs segmentation or tracking itself, so it measures
the published pipeline's outputs rather than a private re-analysis of them.

Modules, in the order the evidence flows:

- ``features``  -- three feature sets kept apart on purpose: *strict
  morphology* (one binary mask, nothing else), *phenotype* (masked intensity
  texture, explicitly not morphology) and *motion history* (previous speeds,
  a comparator, never a morphology result).
- ``dataset``   -- per-observation samples with targets at +1/+3/+6 frames and
  experiment/movie/track ids; splits never cut a track.
- ``models``    -- numpy estimators (no scikit-learn in the app venv), the
  mandatory baselines, a tiny torch autoencoder on standardised mask crops,
  and split-conformal intervals.
- ``evaluate``  -- leave-one-group-out cross-validation, metrics, the
  track-block permutation test and the history Delta (C - B).
- ``synthetic`` -- trajectories with a planted shape->speed signal and a null,
  so the pipeline is shown to find a real signal and to refuse a fake one
  before it is pointed at real cells.
- ``experiment`` -- the real-data run that writes
  ``docs/prediction_experiment.json``.
"""

from __future__ import annotations

__all__ = ["features", "dataset", "models", "evaluate", "synthetic", "experiment"]
