# Does a cell's shape now predict how it migrates next?

Status: 2026-10-02. Research only (`training/predict/`); nothing in
`src/corridor` imports it and the installer never contains it. This implements
the last bullet of `docs/NEXT_GENERATION.md` section 9. Every number below is
written by `python -m training.predict.experiment` into
`docs/prediction_experiment.json`, which also records the SHA-256 of each
source file that produced it (they match the files as committed); a rerun
regenerates all of them.

## Summary

**The method finds a planted signal and refuses a fake one. On the real tracks,
cell shape does not predict future migration better than the mean, at any
horizon. With one experiment, nothing here can speak to a new experiment.**

- **Validation first.** On synthetic movies where a cell's future speed was
  planted to follow its aspect ratio, strict morphology beat the mean on all
  nine targets. Every permutation p-value was at its floor (0.005 for
  regression, 0.01 for the label). On a null with the same speed distribution
  and persistence but no link to shape, no target came out significant: the
  smallest p was 0.69. On 30 independent nulls the test fired at the 0.05 level
  2 times. On 10 independent planted datasets it fired 10 times.
- **The real data.** There are 27 tracks and 155 observations in 5 movies. All
  of them are crops of **one acquisition, 20240529-s01**. One movie
  (`052924_t1`) turned out to be a pixel-identical sub-crop of another, with
  its one cell segmented twice, which leaves 26 distinct tracks. Only 15 tracks
  (83 observations, 3 non-overlapping fields) have a mask, a past step and a
  target at +1 frame. At +6 frames there are 7 tracks (28 observations) in 2
  fields.
- **The result.** The pre-declared morphology model (ridge, or logistic for the
  migrating label) does not beat the mean or majority baseline on any of the 9
  target x horizon tests:
  - The raw p-values range from 0.08 to 0.998, and none survives Holm's
    correction (smallest adjusted p 0.72).
  - At +1 frame the out-of-fold MAE is 8.07 um for morphology against 7.78 um
    for the mean, and R^2 against the training mean is -0.10.
  - The only predictor that beats the mean is the cell's own **recent speed**,
    and only at +1 frame: MAE 6.79 um, p = 0.005.
  - Adding shape to that history lowered the +1-frame MAE to 6.02 um (Delta-MAE
    +0.78 um). The p-value was 0.055, the track bootstrap interval was -0.07 to
    1.77, and the label's balanced accuracy rose 0.65 -> 0.71 (p = 0.02).
  - None of these Delta tests survives Holm (smallest adjusted p 0.18), and at
    +3 frames the same addition made the errors much worse.
- **How easily this goes wrong.** Under leave-one-*movie*-out with the
  duplicated cell kept, the same morphology model "beats" the mean on 4 of 6
  regression targets (p = 0.031 to 0.046). One cell counted twice turns a null
  into apparent findings. The leak-free split reports p = 0.80 to 0.998 for the
  same targets.
- **The honest conclusion.** This is a negative result on 15 cells, not
  evidence that shape carries no information. Leave-one-field-out within one
  acquisition is the strongest split these data allow, and cross-experiment
  generalisation cannot be tested at all. Section 6 says what data would
  change that.

## 1. The question, made testable

**Predictors**, in three sets that are never mixed up
(`training/predict/features.py`):

| Set | What it is | May a "morphology predicts migration" claim rest on it? |
|---|---|---|
| Strict morphology (30 columns) | One binary mask and nothing else: area, perimeter, convex area and perimeter, major/minor axis, equivalent diameter, aspect ratio, eccentricity, solidity, extent in the cell's own principal-axis frame, circularity, roughness (perimeter / convex perimeter), curvature of the smoothed outline (mean \|k\|, SD, 5th/95th percentile, concave fraction, bending energy), skeleton length, branches, endpoints and junctions, the seven Hu invariants (log-scaled, Hu 7 by magnitude). Lengths in um, areas in um^2, curvature in 1/um | Yes, and only this |
| Phenotype (10 columns) | Masked intensity texture relative to the local background ring (contrast, spread, skew, kurtosis, gradients, direction-averaged GLCM) | No. In phase contrast these depend on thickness, focus and illumination; a result that needs them is reported as phenotype-or-imaging |
| Motion history (3 columns) | The track's own last step speed, mean speed over up to 3 steps, net rate (um/hr, by elapsed time) | No. A comparator: migration is persistent, so the real question is whether shape adds anything *beyond* this |

**Orientation is excluded**, and so is anything that smuggles it in. In this
device the channels fix every cell's long axis (all 246 labelled cells are
vertical to within +/-20 degrees, `build/maps/research-data.md`). Orientation
therefore records how the chip sat on the stage, not what the cell is doing,
and Corridor 2.0 removed the migration axis for the same reason. Several
standard measures also change with a cell's tilt, so each is replaced by an
orientation-free version:

- Grid-aligned `regionprops.extent` and bounding-box sides become extent in
  the principal-axis frame.
- Skeleton length is measured along a traced, smoothed path. A chain code
  reads a 22.5-degree line 8 % longer than a vertical one.
- The perimeter is the sub-pixel outline of the lightly blurred mask.

No position, velocity, frame index or intensity enters the strict set, and the
tests assert that no strict column name carries any of those tokens.

**Targets** (`training/predict/dataset.py`) are computed for each tracked
observation at frame t whose own mask exists, at horizons of +1, +3 and +6
frames (20, 60 and 120 min here):

- `disp_um_h{h}`: straight-line displacement over the horizon, um.
- `speed_um_per_hr_h{h}`: path length over the horizon divided by the
  *elapsed* time, um/hr (a frame gap is a longer interval, never a faster cell).
- `migrating_h{h}`: 1 if the net displacement rate is at least **12 um/hr**,
  else 0 (stalled). 12 um/hr is a cell moving its own width every 20 minutes;
  the width is the 4.0 um of `STATIONARY_NET_UM` in `corridor.core.qc`. At
  h = 1 the label therefore agrees with the QC rule. The threshold was fixed
  before any model was run.

A target exists only if the track has an observation exactly h frames later.
At h = 1, displacement and speed are one quantity in two units (the measured
speed/displacement ratio is 2.999 = 60 / 20.007 min for every row), so those two
rows of every table repeat each other and their tests are not independent.

**Who may be a feature source.** Only `primary` detections qualify. Recovered
detections have no pixels in `masks.npz`, and their outline came from a
threshold-and-flood-fill, not from the model; their *positions* still serve as
targets and history. A mask touching the image border is the crop's shape,
not the cell's. Every exclusion is counted (section 3).

**Models** (`training/predict/models.py`, numpy only):

- Baselines: the population mean (majority class for the label) and the
  per-condition median (per-condition majority). With one condition, the
  latter is the population median.
- Morphology: ridge, elastic net (l1 ratio 0.5), a random forest (200 trees,
  depth 3, leaves of 5, a third of the features per split; fixed, not tuned),
  and logistic regression for the label.
- Learned morphology: a convolutional autoencoder (30,025 parameters, latent
  dimension 8, 150 full-batch Adam epochs, seeded, torch CPU, 2 threads).
  - It works on 64x64 mask crops standardised for position, angle, head/tail,
    mirror image and size, so it can only learn shape.
  - Its embedding feeds a ridge or logistic model.
  - It never sees a target, and it is fitted once per held-out group on the
    other groups' masks only.
- Combined: strict morphology + embedding.
- Phenotype: ridge or logistic on the phenotype set (named as not morphology).
- Comparators: B is motion history only; C is strict morphology + history.
- Split-conformal 80 % intervals around every regression model, calibrated on
  a random third of the *training tracks* of each outer fold.

**Why numpy and not scikit-learn.** The app venv has no scikit-learn, and a
separate research venv would have split one experiment over two environments,
one of them carrying torch. The estimators needed are few. Each is pinned in
`tests/test_predict_models.py` against one of:

- a closed form (ridge);
- its optimality conditions (elastic net, logistic);
- a known answer (forest);
- a coverage simulation (conformal).

**Evaluation** (`training/predict/evaluate.py`):

- **Outer split: leave one group out.** The group is the experiment as soon as
  there are two (`dataset.outer_grouping`). With one experiment it is the
  *field*: movies that show the same pixels are merged into one field, so no
  cell can be on both sides (section 3).
- **Inner penalty selection:** grouped 5-fold over tracks, inside the training
  fold only. Every split at every level (outer, inner, conformal calibration)
  is checked to keep each track whole.
- **Metrics** are computed on the pooled out-of-fold predictions:
  - MAE, RMSE, R^2, and R^2 against each fold's own training mean (the only
    mean a real prediction could have used);
  - balanced accuracy, sensitivity and specificity for the label;
  - interval coverage and width.

  Per-fold values sit beside them.
- **"Morphology beats the mean"** is a one-sided permutation test of
  MAE(mean) - MAE(model), or the balanced-accuracy difference for the label.
  - The whole cross-validation is rerun under each permutation.
  - Targets move in **track blocks**: tracks are concatenated in random order
    and circularly shifted, so persistence along a track survives and no track
    keeps its own targets.
  - One model per task is pre-declared primary (ridge, or logistic for the
    label). Holm's correction runs over the nine primary tests.
- **Delta** (directive section 49) = perf(C: morphology + history) -
  perf(B: history only).
  - Its null keeps what history already says about shape and moves only the
    rest. Each morphology column is split into its least-squares fit on
    history (no target involved) and the residual, and the residuals are moved
    in track blocks (Freedman-Lane).
  - A track-bootstrap interval for Delta-MAE sits beside the p-value, and Holm
    runs over the nine Delta tests.

## 2. Validation first: a planted signal and a null

`training/predict/synthetic.py` writes Corridor-shaped result folders
(`masks.npz`, `tracks.csv`, `run.json`), so validation goes through exactly the
code path the real data does. The setup:

- 4 movies x 5 cells x 11 frames, 1 um/px, 10 min/frame, seed 11.
- One cell per vertical lane, shaped like the supplied cells.
- Speeds persist along a track, differ between cells, and carry a per-movie
  offset.

The two cases differ in one thing only:

- **Planted:** each step's speed is 24 + 8 x (aspect ratio now - 4.75) um/hr,
  plus noise.
- **Null:** the same formula, driven by an independent copy of the
  aspect-ratio process, so the speeds have the identical distribution and
  persistence and no link to the drawn shape.

| Case | Regression targets (6) | Migrating label (3) |
|---|---|---|
| Planted | ridge beats the mean on all 6; p = 0.005 each (the floor at 199 permutations); R^2 vs training mean 0.82-0.87; e.g. speed +1: MAE 5.19 vs 12.83 um/hr; 80 % intervals cover 0.76-0.79 | balanced accuracy 0.94 / 0.90 / 0.86; p = 0.01 each (the floor at 99) |
| Null | p = 0.805-0.95; R^2 vs training mean -0.16 to -0.41 | p = 0.69-0.93 |

Calibration on independent datasets (`synthetic_validation.calibration`):

| Test | Datasets | Rejections at 0.05 | Rate (exact 95 % CI) |
|---|---|---|---|
| Ridge, track-block permutation, null | 30 | 2 | 0.067 (0.008-0.221) |
| Ridge, row shuffle, null | 30 | 4 | 0.133 (0.038-0.307) |
| Ridge, track-block, planted (power) | 10 | 10 | 1.0 (0.69-1.0) |
| Logistic, track-block, null, +1 / +3 frames | 20 / 20 | 1 / 0 | 0.05 / 0.0 |

The track-block test holds its level within what 30 datasets can resolve.
The row shuffle rejected twice as often, but with 30 nulls the two intervals
overlap, so this run does not *measure* how much worse it is; it is not used.

The generator's first outline redrew a lopsided shape every frame, so the mask
centroid (the tracked position) jittered with cell length, even with no shape
-> motion link. On that one null dataset the classifier reached p = 0.04 (+1)
and 0.01 (+3). Over 20 independent lopsided nulls it rejected 2 of 20 (+1) and
0 of 20 (+3), which is compatible with chance. The strict null now draws a
point-symmetric outline, and both outlines are reported.

## 3. The real data

Five Corridor v1.3.0 result folders, `build/baseline_v1.3.0/<movie>/`
(0.467 um/px, 20.007 min/frame):

| | Tracks | Observations |
|---|---|---|
| As written by Corridor | 27 | 155 |
| After dropping the duplicated cell (`052924_t1` track 1) | 26 | 142 |
| With a strict-feature mask | -- | 125 |

The 30 observations that are not feature sources break down as follows:

| Reason | Observations |
|---|---|
| Recovered, so no pixels in `masks.npz` | 16 |
| Mask touches the image border | 1 |
| Duplicate of an overlapping movie | 13 |

No label/area mismatches and no multi-piece masks occurred.

**Experiment.** Every movie's ImageJ label names the same acquisition,
**20240529-s01**, so there is exactly one experiment and the outer split is
by field.

**Fields.** Movies are crops, and crops can overlap.
`dataset.find_region_overlaps` compares every pair at their shared *source*
frames.

- `052924_t1` is `052924_1` at offset (-6, 27): tile NCC 1.000 at source
  frames 1, 10 and 18. Its only track is `052924_1`'s cell seen again, so the
  track is dropped and the two movies form one field.
- `052924_1` and `052924_2` are different regions: best NCC 0.91, 0.84 and
  0.79, at offsets that disagree.
- `052924_t2_empty` (source frames 23-27) and `052924_t3_dual` (42-52) share no
  instant with any other movie. Their best tile NCC against `052924_1` is 0.950
  and 0.972, while different regions of this device photographed at the same
  instant reach 0.95 because the channel walls repeat. So whether they show
  `052924_1`'s lanes, and possibly its cells, hours later cannot be decided
  from pixels. They are kept as their own fields and bounded by a sensitivity
  analysis (section 5).

**What enters each task.** A task needs a mask, at least one past step, and a
target:

| Horizon | Observations | Tracks | Fields (observations each) | Migrating / stalled |
|---|---|---|---|---|
| +1 frame (20 min) | 83 | 15 | 3 (50 / 30 / 3) | 52 / 31 |
| +3 frames (60 min) | 58 | 13 | 3 (33 / 24 / 1) | 35 / 23 |
| +6 frames (120 min) | 28 | 7 | 2 (15 / 13) | 15 / 13 |

When the `052924_1` field is held out at +1 frame, the model learns from 6
tracks.

**The strict features on real masks.** The audit rotates each of the 125 masks
by 15, 30, 60 and 90 degrees and compares the median change of each feature to
how much the cells differ from one another (`feature_invariance_real_masks`):

| Feature | Rotation noise / between-cell SD |
|---|---|
| `concave_fraction` | 1.72 |
| `roughness` | 1.05 |
| `curvature_p05_per_um` | 0.49 |
| `solidity` | 0.35 |
| `curvature_mean_abs_per_um` | 0.21 |
| `extent_principal` | 0.14 |
| All the others | 0.09 or less |
| Area, axes, aspect ratio, perimeters, Hu 1-7 | 0.022 or less |
| Skeleton counts | 0 |

On these nearly convex cells, the first two measure the pixel grid more than
the cell. Real cells sit within +/-20 degrees, so 90-degree rotations overstate
the noise. The features stay in the pre-declared set, because removing them
after seeing results would be a post-hoc choice, and the penalised models and
the permutation null both account for noise columns. Read any future result
that leans on them with this table in hand.

The autoencoder reconstructs the masks with IoU 0.88-0.90 on its training
masks and 0.80-0.90 on the held-out field. Its three fits took 112 s in total;
81.5 s of that was the first fit, which includes importing torch on a busy
machine.

## 4. Results

Out-of-fold MAE (um for displacement, um/hr for speed) with the permutation
p-value against the mean baseline. The primary rows have 999 permutations,
elastic net and forest have 49 (descriptive only), and the others have 199.
"R^2 tr" is R^2 against each fold's own training mean.

| Target | Mean | Median | **Ridge morph. (primary)** | Elastic net | Forest | Autoencoder | Phenotype | **History (B)** | **Morph. + history (C)** |
|---|---|---|---|---|---|---|---|---|---|
| disp +1 | 7.78 | 7.17 | **8.07** (p 0.80, R^2 tr -0.10) | 8.04 (0.82) | 7.40 (0.04) | 7.82 (0.62) | 7.79 (0.38) | **6.79** (0.005, R^2 tr 0.09) | **6.02** (0.005, R^2 tr 0.34) |
| disp +3 | 16.87 | 16.86 | **29.40** (p 0.998, R^2 tr -2.0) | 16.97 (0.62) | 16.73 (0.10) | 16.85 (0.175) | 18.68 (0.81) | 17.11 (0.575) | 25.03 (0.985) |
| disp +6 | 28.69 | 31.54 | **28.91** (p 0.716) | 28.69 (0.62) | 28.91 (0.56) | 27.01 (0.055) | 28.72 (0.395) | 29.72 (0.55) | 28.88 (0.59) |
| speed +1 | 23.33 | 21.50 | **24.19** (p 0.80) | 24.54 (0.88) | 22.18 (0.06) | 23.44 (0.62) | 23.35 (0.38) | **20.37** (0.005) | **18.04** (0.005) |
| speed +3 | 16.99 | 17.91 | **18.91** (p 0.867) | 17.50 (0.74) | 17.48 (0.26) | 16.97 (0.19) | 18.41 (0.805) | 17.29 (0.62) | 24.33 (0.985) |
| speed +6 | 15.20 | 16.60 | **15.28** (p 0.69) | 15.20 (0.68) | 15.21 (0.40) | 14.58 (0.03) | 14.69 (0.11) | 15.94 (0.575) | 15.27 (0.64) |

Balanced accuracy for the migrating label (p against the majority class):

| Horizon | Majority | **Logistic morph. (primary, 499 perm.)** | Forest | Autoencoder | Phenotype | History (B) | Morph. + history (C) |
|---|---|---|---|---|---|---|---|
| +1 | 0.50 | **0.63** (p 0.08) | 0.50 (0.50) | 0.38 (0.95) | 0.44 (0.83) | 0.65 (0.07) | 0.71 (0.04) |
| +3 | 0.50 | **0.53** (p 0.552) | 0.45 (0.72) | 0.54 (0.47) | 0.42 (0.98) | 0.50 (0.58) | 0.53 (0.49) |
| +6 | 0.28 | **0.51** (p 0.176) | 0.32 (0.60) | 0.58 (0.12) | 0.31 (0.67) | 0.51 (0.35) | 0.44 (0.34) |

At +6 frames the majority baseline's pooled balanced accuracy is 0.28, not
0.5. Each of the two folds' training majority is the minority class of the
other fold, and the permutation statistic is measured against that baseline.
Against 0.5, no model is above 0.58 at +6.

Reading the tables:

- **Primary tests: 0 of 9 significant.** The Holm-adjusted p-values run from
  0.72 (migrating +1) to 1.0. At +1 frame the track bootstrap of
  MAE(mean) - MAE(ridge) is -0.29 um, with 95 % CI -1.45 to 1.08 over 15
  tracks.
- **The mean is not even the strongest constant.** The targets are
  right-skewed (displacement +1: mean 10.0 um, median 5.8 um), so the median
  has a lower MAE at +1 frame (7.17 vs 7.78 um) and every morphology model
  loses to it there.
- **Ridge on strict morphology fails badly at +3 frames:** MAE 29.4 against
  16.9. The error is concentrated in the held-out `052924_2` field (fold MAE
  38.6 against 13.8 for the mean). With 9 training tracks and 30 correlated columns, a linear model
  trained on the other crop regions extrapolates wildly. The penalised and
  tree models stay at the baseline there.
- **The two nominal p < 0.05 values for morphology are not primary tests.**
  The forest at displacement +1 has p = 0.04 at 49 permutations, and the
  autoencoder at speed +6 has p = 0.03 at 199. Neither repeats at the other
  horizons or in the matching row (disp +6: p 0.055; speed +1: 0.06). There are
  33 secondary morphology tests (4 models x 6 regression targets + 3 x 3 label
  targets). Two nominal hits among them is what chance produces at the 0.05
  level: 1.65 would be expected if the tests were independent, and they are not.
- **History beats the mean at +1 frame only**, and C beats it as well. History
  explains a median of 6 % (+1), 11 % (+3) and 24 % (+6) of the variance of each
  strict column. Most of the shape is therefore not a copy of the recent
  speed, which is what the Freedman-Lane null keeps in place.

Delta = perf(C) - perf(B) (`delta_history`; positive = adding shape helped):

| Target | B | C | Delta | Freedman-Lane p | Track bootstrap 95 % CI of Delta-MAE |
|---|---|---|---|---|---|
| disp +1 | MAE 6.79 | 6.02 | **+0.78 um** (R^2 +0.25) | 0.055 | -0.07 to 1.77 |
| disp +3 | 17.11 | 25.03 | -7.92 um | 0.99 | -15.41 to -0.70 |
| disp +6 | 29.72 | 28.88 | +0.85 um | 0.345 | -0.13 to 2.38 |
| speed +1 | 20.37 | 18.04 | +2.33 um/hr | 0.055 | -0.21 to 5.32 |
| speed +3 | 17.29 | 24.33 | -7.04 um/hr | 0.985 | -13.65 to 0.27 |
| speed +6 | 15.94 | 15.27 | +0.67 um/hr | 0.36 | -0.21 to 1.63 |
| migrating +1 | BA 0.650 | 0.711 | **+0.061** | **0.02** | -- |
| migrating +3 | 0.504 | 0.533 | +0.030 | 0.47 | -- |
| migrating +6 | 0.508 | 0.436 | -0.072 | 0.37 | -- |

Holm over the nine Delta tests leaves none significant: the smallest adjusted
p is 0.18 (migrating +1), and 0.44 for the two +1 regression rows, which are
one test in two units. The +1-frame hint is consistent across the regression
and the label. It is the one thing worth pre-registering for new data
(section 6), but it is not a finding.

**Intervals.** Nominal coverage is 80 %. The empirical coverage is
`interval_coverage` per model:

| Horizon | Coverage | Mean width |
|---|---|---|
| +1 (mean and ridge) | 0.81 | about 21 um |
| +3 | 0.52-0.85; the speed +3 mean baseline covers only 12 of the 33 observations of the `052924_1` field (0.36) | -- |
| +6 | 0.96-1.00 | 140-340 um for displacement, against a mean displacement of 35 um |

History models undercover at +1 frame (0.63). The split-conformal guarantee
assumes calibration and test cells are exchangeable, and a held-out field with
faster cells is not. The coverage is reported, not claimed.

## 5. Sensitivity analyses

| Analysis | Primary morphology model vs mean | Read |
|---|---|---|
| Leave-one-*movie*-out, duplicated cell kept (`sensitivity_leave_one_movie_out_with_duplicate`) | disp +1: 7.41 vs 7.82, **p 0.031**; disp +3: **p 0.039**; speed +1: **p 0.031**; speed +3: **p 0.046**; +6: p 0.94 and 0.58 | A leak, not a result: the same cell sits in training and test |
| Only targets whose path never touches a recovered detection (`sensitivity_primary_only_targets`) | worse than the mean on all 6 regression targets, p 0.735-1.0 | Recovery does not hide a morphology signal |
| Only the two fields the pixels prove distinct, i.e. without `052924_t3_dual` (`sensitivity_pixel_established_fields_only`) | disp +1: 7.99 vs 8.02, p 0.16; label +1: BA 0.59, p 0.14; all others p 0.18-1.0 | The undecidable crop changes nothing |

The first row is why the split was changed. The task brief named
leave-one-movie-out as the strongest available split, and the first run of
this experiment used it. The overlap check then found `052924_t1` inside
`052924_1`, and the split became leave-one-field-out. That change was made for
leakage, against the direction of the result: it removed four nominal
positives.

## 6. What this can and cannot support, and what data would change it

**Supported:**

- On 15 cells from one acquisition, neither the handcrafted shape features,
  nor a learned mask embedding, nor the masked texture predicts the next 20,
  60 or 120 minutes of migration better than a constant, when tested on a crop
  region the model never saw.
- The cell's own speed over the last hour predicts the next 20 minutes
  modestly (R^2 0.09 against the training mean).
- The pipeline that says so finds a planted shape -> speed effect of the size
  simulated in section 2, and holds its false-positive rate on nulls with the
  same persistence structure.

**Not supported:**

- That shape carries *no* migration information. The power shown on synthetic
  data is for a large effect (R^2 about 0.8). How small an effect 15 cells
  could detect was not measured, and nothing here bounds it.
- Anything about a new experiment, dish, day, device, condition or camera.
  There is one experiment. Leave-one-field-out holds out a crop region of the
  same dish at the same time, under the same medium, focus and segmentation
  error pattern.
- Anything at +6 frames. There are 7 tracks in 2 fields, and the intervals
  there are wider than the movements they bound.

**Data that would make the question testable:**

1. **More experiments, from both cameras.** The labelled stills in
   `CellPose_TrainData` were cut from eight acquisitions (five KK1 at
   0.639 um/px, three KK2 at 0.467 um/px; `docs/RESEARCH_V2.md` section 1).
   20240529-s01 is the only one of them with a movie here, and stills are not
   tracks. Running Corridor on the full time-lapse movies of
   those acquisitions changes the outer split by itself:
   `dataset.outer_grouping` switches to leave-one-experiment-out as soon as two
   experiments are present, the autoencoder pool follows it, and `build_dataset`
   refuses to pool calibrated and uncalibrated folders.
2. **Longer tracks.** A +6-frame target with a one-step history needs 8
   consecutive observations, and only 7 tracks have them.
3. **Masks for recovered detections.** 16 of 155 observations have a position
   and no outline.
4. **Tracking ground truth for at least one movie** (`NEXT_GENERATION.md`
   section 10). Every target here is the tracker's own linking, and a linking
   error is a target error that nothing in this experiment can see.
5. **The experimental condition of each acquisition.** The per-condition
   baseline is mandatory and is the population median until two conditions
   exist.
6. **A pre-registered first test.** On new data, test Delta at +1 frame
   (`disp_um_h1` and `migrating_h1`) with every setting in this JSON unchanged.
   It is the only hint here, and a pre-declared single test is the only way to
   tell whether it was chance.

## 7. Reproduce

```bash
PY=./.venv/Scripts/python.exe
OMP_NUM_THREADS=2 $PY -m training.predict.experiment            # ~75 min here, writes docs/prediction_experiment.json
OMP_NUM_THREADS=2 $PY -m training.predict.experiment --smoke --out <tmp>.json   # every code path, ~4 min, numbers meaningless
OMP_NUM_THREADS=2 $PY -m pytest tests/test_predict_features.py tests/test_predict_models.py tests/test_predict_synthetic.py -q
```

The full run took 4468 s on this machine. It ran at below-normal priority
beside a CPU training run. That run time is dominated by the permutation
counts (`experiment.N_PERM`, set by cost and never by outcome) and the
classifier calibration. The three test files ran 43 tests in 45 s.

These things changed after real numbers had first been seen. None of it moved
towards a result:

- the elastic-net penalty grid and the permutation counts, for run time;
- the outer split, from movie to field, for leakage;
- an added sensitivity analysis, on pixel-established fields only.

All three are listed in the `experiment.py` docstring. The automatic switch to
experiment-level splits (`dataset.outer_grouping`) was added in the same
revision; it changes nothing on one experiment.
