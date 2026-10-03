# Handoff

Full handoff document, with the charts and the reasoning:
https://claude.ai/code/artifact/f81d4efe-ef67-445f-8b2d-3f2b94f60147

This file is the operational core only, so it cannot drift from the commands.

## Commit first

The last commit is 1.1.0, dated 13 September. The working tree is 1.3.0, with 23
modified and 73 untracked files. v1.2.0 and v1.3.0 were both built and published
from a tree git has no record of, so neither shipped build can currently be
reproduced from source control.

Everything in this handoff except the published binaries - the corrected labels,
every experiment script, the trained checkpoints, the tests - exists in exactly one
place on one disk. Commit before running anything else, and before a training run
touches `build/`. The grouping of those commits was left to you rather than guessed
at.

## State

- v1.3.0 is published and the update check works. Installer and source are on the
  GitHub releases page and both were re-checked and return 200.
- Best held-out detection score: **F1 0.7273** — the round 1 contrast-augmented
  checkpoint (`build/models/models/corridor_contrast_invariant`), read with
  percentile window (3, 97) at diameter 36, trained on KK1 and scored on KK2.
- Ceiling imposed by the labels: **withdrawn and re-derived.** The old 0.942
  (KK2) / 0.968 (KK1) assumed filename order was time order; it is not (see
  `docs/RESEARCH_V2.md`). On the true ImageJ time index only consecutive-in-
  time pairs can be checked, and in them just **2 undrawn cells per group**
  turn up (not 7 and 9), giving a label ceiling of ~0.967 (KK2) / 0.969 (KK1)
  over the checkable subset (`docs/label_completeness_v2_*.json`). The labels
  are therefore cleaner than the old figure implied, and the gap from the best
  held-out F1 (0.7273) to that ceiling is the model's, not the labeller's.
- `corridor_contrast_invariant_w3_97` is round 2 and is **not** used: training with
  the window applied scored 0.6452 against its own 0.6533 start.

## Next, in order

1. Restart the killed training run (about two hours):

       OMP_NUM_THREADS=4 ./.venv/Scripts/python.exe scripts/train_contrast_invariant.py --epochs 60 --copies 2 --scale-aug --corrected-labels build/corrected_reference/KK1 --out docs/train_round4.json

   Then score it the way the shipping configuration reads images, which the
   training script does not do for you:

       ./.venv/Scripts/python.exe scripts/experiment_diameter.py --model trained --test-group KK2 --window 3,97

   Beat 0.7273 or the checkpoint does not ship.

2. Matched-budget control (about two hours). `--no-augment` sets copies to zero, so
   it does **not** match the budget on its own; raise the epochs instead. 40 images
   over 168 epochs is the same 6,720 presentations as the augmented run's 112 x 60:

       OMP_NUM_THREADS=4 ./.venv/Scripts/python.exe scripts/train_contrast_invariant.py --no-augment --epochs 168 --out docs/train_control_matched.json

   Until this runs, the +0.042 attributed to contrast training is unattributed: its
   comparison received no further training at all.

3. Stop scoring border fragments. 18 of the 64 errors are cells cut off by the edge
   of the field, a median 3.5 px from the border at half a normal cell's area. Add a
   border-margin exclusion to `scripts/evaluate_against_both.py` and report both
   figures. This changes the measurement, not the model — say so wherever the new
   number appears.

4. Refine the 17 near-miss outlines. Median IoU 0.43 against a 0.5 threshold, 8 of 9
   within 0.1 of it, zero splits and one merge. The instances are right and only the
   boundaries are off, so a single refinement pass over the predicted mask is the
   cheapest remaining F1 on the board.

5. Lead with the corrected reference (now rebuilt on true time order,
   `build/corrected_reference_v2/`). The re-derived label ceiling is ~0.967
   (KK2) / 0.969 (KK1), not 0.942/0.968. Keep reporting both references.

Not on this list: more inference strategies. Eight were measured and all stalled
together, which is what sent the work to the training side.

## Do not re-derive

Five claims were withdrawn after they were measured and failed. The long version is
in `docs/TOWARDS_99.md`; read it before reviving any of them.

- Boundary sharpness as a quality signal — failed its own control (1.196 against a
  human 1.241). The metric rewarded what the method optimised.
- Reconstruction as a teacher — 0.492 where the model itself scored 0.725.
- "Recovery fills 1 of 13 gaps" — measured through a defect that skipped every
  interior gap in silence. True figure: 10 of 13, no false positives.
- Training at the normalisation you ship — measured worse. See round 2 above.
- Pre-clipping to a percentile window — destroys cells (8 to 21 % of cell pixels sit
  outside the 1st-99th band). Pass a window to Cellpose's own `normalize` argument.
  Related: `normalize={"lowhigh": (lo, hi)}` is broken in cellpose 3.1.1.3 and
  silently returns zero masks.

## Constraints that still hold

- `cellpose>=3,<4`. Do not migrate the custom model to Cellpose 4.
- Do not publish the research datasets. `scripts/make_source_zip.py` enforces this
  with an allow-list and re-checks the finished archive.
- Check licences before bundling third-party assets: `scripts/audit_licences.py`
  reads the built application's own metadata and fails the release on an unaccounted
  copyleft dependency.
