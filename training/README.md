# training/

Research code: datasets, splits, augmentation, training and evaluation for the
segmentation model. Nothing in `src/corridor` imports it and the installer never
contains it. Run everything from the repository root. The design contract is
`docs/NEXT_GENERATION.md` section 9; the measurements behind each choice are in
`docs/RESEARCH_V2.md`.

| Module | What it does | Writes |
|---|---|---|
| `datasets.py` | A registry of every still, read from its ImageJ metadata: experiment `<yyyymmdd>-s<series>`, true time index, interval, pixel size, bit depth, crop, instances, hashes, duplicates, split | `build/registry/dataset_v1.json` |
| `splits.py` | Locked experiment-level splits. Test is all of KK2, val is 20230615-s04, train is the rest of KK1. The sample movies (20240529-s01) are never used for training | `build/registry/splits_v1.json` |
| `scoring.py` | `core.metrics` plus border-margin exclusion, AP over IoU 0.50:0.95, splits and merges, and breakdowns | none |
| `refine.py` | Per-instance boundary refinement (`image`, `contour`). Moves each boundary at most `max_px` and never creates, merges or deletes an instance | none |
| `evaluate_segmentation.py` | Scores one Cellpose 3 checkpoint on one split, against corrected v2 first and the original labels beside it | `build/eval/*.json` |
| `augmentations.py` | A recorded, orientation-free policy. Photometric operations never touch the labels | none |
| `train_cellpose3.py` | Cellpose 3 fine-tuning from an explicit start checkpoint, with a policy and a split | `build/models/models/<name>`, `build/models/reports/<name>.json` |
| `train_cellpose_sam.py` | The same for Cellpose-SAM. Runs in `.venv-cp4` only; `--smoke` is the feasibility probe | same |
| `hard_examples.py` | Ranks unlabelled movie frames from Corridor result folders for annotation | JSON ranking |
| `registry.py` | The research model registry, keyed by SHA-256 | `build/registry/models.json` |

`training/predict/` (morphology predicts migration) belongs to a separate work
package and has its own docstring.

## Order of use

```bash
PY=./.venv/Scripts/python.exe
# 1. Reference and ceilings, on true time order (numpy only)
$PY scripts/build_corrected_reference.py --group KK1      # and --group KK2
$PY scripts/measure_label_completeness.py --group KK1     # and --group KK2
# 2. Registry and locked splits
$PY -m training.datasets
$PY -m training.splits
# 3. Train (check first with --dry-run, which imports no torch)
$PY -m training.train_cellpose3 --start <checkpoint> --name <new> --epochs 60 --window 3,97 --dry-run
# 4. Evaluate on val while choosing; on test once
$PY -m training.evaluate_segmentation --checkpoint build/models/models/<new> --split val \
    --window 3,97 --diameter none --border-margin 3 --refine image --refine contour
# Cellpose-SAM, in its own environment
./.venv-cp4/Scripts/python.exe -m training.train_cellpose_sam --start build/cp4_models/cpsam_v2 --name <new> --smoke
```

## Rules the code enforces

- **No default model.** Every checkpoint is an explicit path, hashed into the
  report. Cellpose 3 and Cellpose 4 both silently substitute a built-in model
  for a bad path or name.
- **No overwrites.** Checkpoints, training reports and evaluation reports are
  refused if they already exist. Neither the v2 nor the `--legacy` mode of the
  reference scripts writes over the published version-1 files. A versioned
  registry file (`dataset_v1.json`, `splits_v1.json`) with different content
  is refused; `--supersede` replaces it only while no registered model used
  it, and keeps the old file under `build/registry/superseded/`.
- **No leakage.** Training refuses validation, test and never-train images.
  Pixel-identical stills are deduplicated. `hard_examples.py` refuses results
  from held-out experiments, and results whose experiment it cannot
  establish, unless told otherwise, and then records it.
- **Held out is proven, not assumed.** `evaluate_segmentation.py` reports
  `held_out: false` unless the checkpoint and every ancestor are in
  `build/registry/models.json` with the experiments they were trained on, none
  of them in the scored split. Every pre-v2 checkpoint (round 4 included) was
  trained on all of KK1, the validation experiment 20230615-s04 included, so
  it cannot be validated on val. To score a model from outside these tools on
  test, register what it saw first, e.g.
  `$PY -m training.registry add --id <id> --path <ckpt> --trained-on-experiments 20230615-s04,20230704-s02,20240418-s01,20240418-s42,20241223-s01`
  (or `none` for a public model that never saw these stills).
- **Corrected beside original.** Scores against the corrected reference are
  always reported next to the original-label scores, with the number of cells
  the correction adds.
- **Privacy.** Source names in the ImageJ labels carry experimental
  conditions. They are kept only in the gitignored
  `build/registry/dataset_v1.json` (`source_name_private`). Committed files name
  experiments by id only.
