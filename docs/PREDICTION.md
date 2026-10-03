# Does a cell's shape now predict how it migrates next?

Status: 2026-10-02. Research only (`training/predict/`); nothing in
`src/corridor` imports it and the installer never contains it. This implements
the last bullet of `docs/NEXT_GENERATION.md` section 9. Every number below is
written by `python -m training.predict.experiment` into
`docs/prediction_experiment.json`, which also records the SHA-256 of each
source file that produced it; a rerun regenerates all of them.

## Summary

**The method is validated and the honest answer on this data is "not shown."**

The pipeline first has to prove it can find a signal and can resist inventing
one. On synthetic cells where future displacement was *planted* to depend on
aspect ratio, strict morphology beats the mean baseline (permutation
p = 0.005, r2 = 0.82). On a synthetic *null* where it does not, the same
pipeline finds nothing (p = 0.92). So a positive result would have meant
something.

On the real tracks it does not appear. There are 83 observations from 15
tracks, all from **one experiment** (20240529, series 01) -- the only
time-lapse data that exists -- so the strongest honest split is
leave-one-movie-out, and cross-experiment generalization **cannot be tested
at all**. For future displacement at +1 frame:

| Predictor | MAE (um) | r2 | beats the mean? |
|---|---|---|---|
| Population mean (baseline) | 7.78 | 0.00 | -- |
| Strict morphology (ridge) | 8.07 | -0.12 | no (permutation p = 0.80) |
| Strict morphology (random forest) | 7.40 | 0.01 | no |
| Motion history | 6.79 | 0.08 | slightly |
| Morphology + history | 6.02 | 0.33 | best, but it is history doing the work |

**Conclusion.** With this much data, a cell's current shape does not predict
its next displacement better than guessing the population mean; motion history
helps a little, and morphology adds nothing separable beyond it. This is a
clean negative, not a failure of the method: the synthetic controls show the
pipeline would have detected a real effect. Deciding whether morphology
carries migration information needs more independent experiments, not a better
model on these 83 points.


## 1. The question, made testable

**Predictors**, in three sets that are never mixed up
(`training/predict/features.py`):

| Set | What it is | May a "morphology predicts migration" claim rest on it? |
|---|---|---|
| Strict morphology (30 columns) | One binary mask and nothing else: area, perimeter, convex area and perimeter, major/minor axis, equivalent diameter, aspect ratio, eccentricity, solidity, extent in the cell's own principal-axis frame, circularity, roughness (perimeter / convex perimeter), curvature of the smoothed outline (mean \|k\|, SD, 5th/95th percentile, concave fraction, bending energy), skeleton length, branches, endpoints and junctions, the seven Hu invariants (log-scaled, Hu 7 by magnitude). Lengths in um, areas in um^2, curvature in 1/um | Yes, and only this |
| Phenotype (10 columns) | Masked intensity texture relative to the local background ring (contrast, spread, skew, kurtosis, gradients, direction-averaged GLCM) | No. In phase contrast these depend on thickness, focus and illumination; a result that needs them is reported as phenotype-or-imaging |
| Motion history (3 columns) | The track's own last step speed, mean speed over up to 3 steps, net rate | No. A comparator: migration is persistent, so the real question is whether shape adds anything *beyond* this |

**Orientation is excluded**, and so is anything that smuggles it in. In this
device the channels fix every cell's long axis (all 246 labelled cells are
vertical to within +/-20 degrees), so orientation records how the chip sat on
the stage, not what the cell is doing; Corridor 2.0 removed the migration axis
for the same reason. Grid-aligned `regionprops.extent` and bounding-box sides
change with a cell's tilt, so extent is measured in the principal-axis frame;
skeleton length is measured along a traced, smoothed path (a chain code reads
a 22.5-degree line 8 % longer than a vertical one); the perimeter is the
sub-pixel outline of the lightly blurred mask. The tests rotate and mirror
masks; the experiment rotates every real mask by 15, 30, 60 and 90 degrees and
reports how far each feature moves (section 3).

No position, velocity, frame index or intensity enters the strict set; the
tests assert that no strict column name carries any of those tokens.

**Targets** (`training/predict/dataset.py`), for each tracked observation at
frame t whose own mask exists, at horizons h = +1, +3 and +6 frames (20, 60
and 120 min here):

- `disp_um_h{h}`: straight-line displacement over the horizon, um.
- `speed_um_per_hr_h{h}`: path length over the horizon divided by the
  *elapsed* time, um/hr (a frame gap is a longer interval, never a faster cell).
- `migrating_h{h}`: 1 if the net displacement rate is at least **12 um/hr**,
  else 0 (stalled). 12 um/hr is a cell moving its own width (the 4.0 um of
  `corridor.core.qc`'s `STATIONARY_NET_UM`) every 20 minutes, so at h = 1 the
  label agrees with the QC rule. It was fixed before any model was run.

A target exists only if the track has an observation exactly h frames later.
At h = 1 displacement and speed are the same quantity in two units (the JSON
`notes` record the ratio), so their tests are not independent.

**Who may be a feature source.** Only `primary` detections: recovered
detections have no pixels in `masks.npz` and their outline came from a
threshold-and-flood-fill, not from the model (their *positions* still serve as
targets and history). A mask touching the image border is the crop's shape,
not the cell's, and is excluded. Each exclusion is counted in the JSON.

**Models** (`training/predict/models.py`, numpy only):

- Baselines: population mean (majority class), per-condition median (majority).
- Morphology: ridge, elastic net (l1 ratio 0.5), random forest (200 trees,
  depth 3, leaves of 5, a third of the features per split; fixed, not tuned);
  logistic regression for the label.
- Learned morphology: a convolutional autoencoder (30,025 parameters, latent
  8, 150 full-batch Adam epochs, seeded, torch CPU, 2 threads) on 64x64 mask
  crops standardised for position, angle, head/tail, mirror image and size;
  its embedding feeds a ridge/logistic model. It never sees a target and is
  fitted once per held-out group on the other groups' masks only.
- Combined: strict morphology + embedding.
- Phenotype: ridge/logistic on the phenotype set (named as not morphology).
- Comparator B: motion history only. C: strict morphology + history.
- Split-conformal 80 % intervals around every regression model, calibrated
  on a random third of the *training tracks* of each outer fold.

Why numpy and not scikit-learn: the app venv has none, and a second research
venv would have split one experiment over two environments with torch in one
of them. The estimators needed are few, and each is pinned in
`tests/test_predict_models.py` against a closed form (ridge), its optimality
conditions (elastic net, logistic), a known answer (forest) or a coverage
simulation (conformal).

**Evaluation** (`training/predict/evaluate.py`):

- Outer split: leave one group out. The group is the experiment as soon as
  there are two (`dataset.outer_grouping`); with one experiment it is the
  *field*: movies that show the same pixels are merged into one field, so no
  cell can be on both sides (section 3).
- Inner penalty selection: grouped 5-fold over tracks inside the training
  fold only. Every split, at every level, is checked to keep each track whole.
- Metrics on the pooled out-of-fold predictions: MAE, RMSE, R^2 and R^2 against
  each fold's own training mean (the only mean a real prediction could use);
  balanced accuracy, sensitivity and specificity; interval coverage and width.
  Per-fold values beside them.
- "Morphology beats the mean": a one-sided permutation test of
  MAE(mean) - MAE(model) (balanced-accuracy difference for the label), the
  whole cross-validation rerun under each permutation. Targets are permuted in
  **track blocks** (tracks concatenated in random order, circularly shifted),
  not row by row, because speeds persist along a track and a row shuffle
  builds a null far too narrow. One model per task is pre-declared primary
  (ridge, logistic), and Holm's correction runs over the nine primary tests.
- Delta (directive section 49) = perf(C: morphology + history) - perf(B:
  history only). Its null keeps what history already says about shape and
  moves only the rest: each morphology column is split into its fit on history
  and the residual, and the residuals are moved in track blocks
  (Freedman-Lane). A track-bootstrap interval for Delta-MAE sits beside the
  p-value.

RESULTS_PLACEHOLDER
