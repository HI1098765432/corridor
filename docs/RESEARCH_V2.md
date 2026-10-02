# Research v2: true time order, re-derived ceilings, a traceable toolkit

Status: 2026-10-02. This implements `docs/NEXT_GENERATION.md` section 9 and
handoff items 3-5. The reference, ceiling and calibration figures are written
by the committed tools into the JSON outputs listed at the end, and a rerun
regenerates them. Figures marked *(dev)* were measured while the tools were
built, by code that is not in the repository; they are recorded here and in the
docstrings but nothing regenerates them.

## 1. What changed, and why the old figures are withdrawn

The labelled stills are not contiguous frames. Each still's ImageJ label
(`t:N/M - <name>.nd2 (series S)`), `finterval` and `XResolution` say which movie
and time point it was cut from. `src/corridor/learn/sequences.py` now groups the
stills by experiment (`<yyyymmdd>-s<series>`) and size, splits each group into
crops by image registration, orders each crop by its true time index, and records
the offset between frames and the elapsed minutes between them. The old
behaviour (group by shape, order by filename number) is kept only behind
`order="filename"` / `--legacy`.

| Experiment | Group | Split | Stills (labelled) | Instances | Times | Interval (min) | um/px | Crops |
|---|---|---|---|---|---|---|---|---|
| 20230615-s04 | KK1 | val | 5 (5) | 17 | t1-t67 | 10.19 | 0.6394 | 1 |
| 20230704-s02 | KK1 | train | 1 (1) | 0 | t34 | 20.27 | 0.6394 | 1 |
| 20240418-s01 | KK1 | train | 12 (12) | 25 | t1-t80 | 15.26 | 0.6394 | 1 |
| 20240418-s42 | KK1 | train | 10 (10) | 40 | t23-t74 | 15.26 | 0.6394 | 1 |
| 20241223-s01 | KK1 | train | 12 (12) | 56 | t15-t145 | 10.15 | 0.6394 | 1 |
| 20240409-s01 | KK2 | test | 1 (0) | 0 | t24 | 10.00 | 0.4671 | 1 |
| 20240529-s01 | KK2 | test | 15 (15) | 51 | t1-t54 | 20.01 | 0.4671 | 2 |
| 20240529-s02 | KK2 | test | 16 (16) | 57 | t1-t54 | 20.01 | 0.4671 | 1 |

KK1 is 12-bit and KK2 16-bit. The date of 20241223 comes from the file name
and the date of 20230704 from the Nikon start time in UTC, because neither
label carries a date. The interval is Nikon's measured mean period, for
example 915.4 s for a nominal 900 s.

Facts the old code did not know:

- **Two crops share KK2 20240529-s01.** `052924_6` and `052924_14` are both
  t1. Wall-suppressed registration splits the 306x621 group into crops of 8 and
  7 frames. No pair across the two crops scores above 0.153 *(dev)*; the
  weakest link any crop needs is 0.237 (`link_ncc` in
  `docs/corrected_reference_v2_KK1.json`). The old shape grouping saw a frame
  correlation of 0.10 here and skipped the group as "not a movie".
- **Frames of a crop are offset.** Frames sit up to 38 px from their crop's
  first frame on 20240529-s02, and up to 19 px on 20240418-s01. A 45 px
  same-cell rule on unregistered pixels was measuring the stage as well as the
  cell.
- **Duplicates.** `041824_2` and `041824_7` are pixel-identical (t67) and were
  labelled twice. Their three cells match at best IoU 0.677, 0.714 and 0.564.
  Annotator agreement therefore bounds AP at high IoU thresholds. Training
  keeps only one of the pair.

**Withdrawn:** the label-completeness ceilings **0.968 (KK1)** and **0.942
(KK2)** and the version 1 corrected reference (`build/corrected_reference/`, +9
KK1 and +7 KK2 cells). `--legacy` reproduces them exactly: both reports are
identical, and 39/39 KK1 and 31/31 KK2 mask files are byte-identical. That
proves they came from filename adjacency. On true time order, 2 of the 9 KK1
additions and 0 of the 7 KK2 additions survive:

| v1 addition | Verdict on true time order |
|---|---|
| 041824_2, 041824_7 (t67) | bracket t48-t69 spans 320 min |
| 041824_4 x2 (t1) | first labelled time of its crop: nothing earlier to bracket from |
| 041824_9 (t71) | **survives** (t69-t73, 61 min) |
| 041824_18 (t56) | **survives** (t54-t59, 76 min) |
| 122324_9 (t69), 122324_10 (t74) | brackets of 91 and 162 min |
| 041824_17 (t54) | bracket of 122 min |
| 052924_21, _22 x2 | brackets of 120 min |
| 052924_26 x2 (t29) | bracket t27-t48 spans 420 min |
| 052924_24, 052924_28 | checkable, but a drawn cell touches the interpolated outline (in `052924_24` the "missing" cell is the end of a drawn one, whose centroid was beyond 45 px) |

## 2. The time-aware bracket rule (`src/corridor/learn/brackets.py`)

A cell is "missed at t" only if it was drawn at the nearest earlier labelled
time t0 and the nearest later labelled time t1 of the same crop, and the rule
can say it is the same cell. Each choice is a measurement:

- **Speed bound: 3 um/min.** The fastest centroid step anywhere is 2.77 um/min,
  from KK1 stills one frame (10 min) apart. In the 114 one-frame steps of the
  v1.3.0 baseline tracks the fastest is 1.98 um/min, and the tracker's 5 um/min
  gate never bound. The median is 0.44 um/min (stills) and 0.35 um/min
  (tracks). *(dev)*
- **Identity by overlap, not distance.** The shortest bracket in the data is
  51 min, so a distance radius of 5 um + 3 um/min x span is at least 158 um. The
  nearest other cell in a frame is a median 58 um away on KK2 (5th percentile
  37 um, about one channel pitch) and 75 um on KK1 *(dev)*. With a distance
  radius, **0 of 198** bracketed cells (109 KK1, 89 KK2) have a unique partner
  (`distance_only_identity` in the label-completeness JSONs). Instead, t0 and
  t1 masks are one cell when, after registration, they overlap at IoU >= 0.15
  and the pairing is unique both ways. The speed bound remains as a
  plausibility gate. Overlap needs no channel direction.
- **Longest bracket: 90 min.** This is a leave-one-out on cells that were
  drawn: for each overlap-linked t0/t1 pair, is a drawn cell where the
  time-interpolated outline puts it (IoU >= 0.15)? Pooled from
  `calibration_leave_one_out` of both label-completeness JSONs:

  | Bracket span | Linked pairs | Drawn where predicted | Left out: pairs into frames with nothing drawn |
  |---|---|---|---|
  | 51-80 min | 26 | 25 (96 %) | 3 (`041824_9`, `052924_7`) |
  | 91-122 min | 12 | 6 (50 %) | 0 |
  | 153-540 min | 17 | 10 (59 %) | 4 (`052924`, `052924_26`) |

  A pair into a frame with nothing drawn always "fails". That is a missing
  label, the thing being measured, not an interpolation error, so those pairs
  are not part of the calibration. Past 80 min the interpolated position stops
  predicting where a drawn cell is, so an empty spot there is no longer
  evidence of a missed label. The spans in the data fall into these clusters,
  so any limit from 80.0 up to (not including) 91.3 min gives identical
  results; the case rests on the 12 pairs of the middle cluster, so the cut is
  measured but not precise. 122324_9 misses the limit by 1.3 min.
- **Position by elapsed-time fraction**, not the midpoint. The outline is the
  human mask from the nearer labelled time, translated.
- **"Drawn here"**: a drawn mask touching that outline grown by 5 um vetoes the
  claim. 5 um is twice the worst centroid difference between the two
  annotations of 041824_2/_7 (2.48 um *(dev)*). This is conservative: a neighbour can
  veto a missed cell, but cannot invent one.
- Evidence is always the original labels, so additions never chain. An outline
  with fewer than 80 px inside the frame is not claimed.

`scripts/build_corrected_reference.py` and `scripts/measure_label_completeness.py`
call the same function, so the cells added and the cells counted as missing are
the same cells.

## 3. Corrected reference v2 and the re-derived ceilings

New cells (`build/corrected_reference_v2/{KK1,KK2}/`, one file for every
labelled still, `34-34` included):

- KK1: `041824_9` t71 (t69-t73, 61 min, f=0.50) and `041824_18` t56
  (t54-t59, 76 min, f=0.40). 138 + 2 = 140.
- KK2: two cells in `052924_7` t3 (t1-t5, 80 min). That crop of 20240529-s01
  was never examined in v1. 108 + 2 = 110.

The ceiling is tp = labelled and fp = undrawn, with recall 1,
P = tp / (tp + fp) and **F1 = 2P / (P + 1)**, on the images that could be
checked. The interval is the exact binomial (Clopper-Pearson) interval on the
missing rate m = fp / (tp + fp), mapped through F1 = 2(1 - m) / (2 - m)
(`ceiling_interval_95` in the JSONs, computed with `scipy.stats.binomtest`):

| | Checked images | Labelled there | Undrawn | Ceiling on checked images | 95 % interval | Whole-group bound |
|---|---|---|---|---|---|---|
| KK1 | 7 of 40 | 31 | 2 | P 0.9394, **F1 0.9688** | missing 0.74-20.2 %, F1 0.8875-0.9963 | 138 / 2: F1 0.9928 |
| KK2 | 8 of 31 | 29 | 2 | P 0.9355, **F1 0.9667** | missing 0.79-21.4 %, F1 0.8800-0.9960 | 108 / 2: F1 0.9908 |

An image is checked only where the rule could have found an undrawn cell: a
bracket of 90 min or less, something drawn on both sides of it, and at least
one cell linked across it. The labelled cells of any other image could never
have been offset by an undrawn one, so counting them as tp would only raise the
ceiling. Within the checked images the rule tested 12 (KK1) and 17 (KK2)
linked cells; a missed cell that links to nothing, or is new at t, cannot be
found at all, so these ceilings remain upper bounds.

- KK1 checked: 041824_5, _9, _15, _18 and 122324_5, _7, _8.
- KK1 unchecked:
  - 4 first and 4 last times of their crops
  - 20 with longer brackets
  - 2 beside the unannotated 041824_9 (041824_8, _10)
  - 2 whose bracket links no cell (122324_4, _6)
  - 34-34, which is in no sequence
- KK2 checked: 052924_7, _17, _18, _19, _23, _24, _28, _29.
- KK2 unchecked: 3 first, 3 last, 14 with longer brackets, 2 beside an
  unannotated frame (052924_8 beside 052924_7, 052924_25 beside 052924_26),
  1 whose bracket links no cell (052924_20).

The whole-group bound counts only the cells found in checked images, so it is
an upper bound on F1, and a weak one.

**What this means.** A ceiling estimated from 2 events cannot be pinned down.
The data are consistent with any reference-imposed cap from about 0.88 to
1.00. Both withdrawn figures, 0.968 and 0.942, lie inside those intervals: they
are not disproved, only no longer supported. Three of the four undrawn cells
are in frames where nothing at all is drawn (`041824_9`, `052924_7`), so what
the rule mostly finds is unannotated frames, not cells skipped inside annotated
ones. Lead with the corrected figure, keep the original beside it, and do not
quote either ceiling without its interval.

## 4. Splits (`training/splits.py`, locked)

| Split | Experiments | Labelled images | Instances |
|---|---|---|---|
| train | 20230704-s02, 20240418-s01, 20240418-s42, 20241223-s01 | 35 | 121 |
| val | 20230615-s04 | 5 | 17 |
| test | 20240409-s01, 20240529-s01, 20240529-s02 | 31 | 108 |

- **Test is all of KK2.** This is the published held-out direction, and it is
  also a different instrument.
- **Val is 20230615-s04.** It is the only labelled KK1 experiment that is its
  own acquisition (20240418-s01 and -s42 are two stage positions of one .nd2
  file). It is also small enough to leave 121 of 138 KK1 instances for
  training.
- No experiment or acquisition straddles splits; the code refuses one that
  would.
- **The five sample movies are 20240529-s01** (t1-t52 of the same 54-frame
  movie as half the test stills, read from their own labels). They must never
  be trained on or mined for hard examples while KK2 is the test set.
  `hard_examples.py` refuses them by default, and refuses any result whose
  experiment it cannot establish.
- **No pre-v2 checkpoint can be validated on val.** `corridor_contrast_invariant`
  and the round-4 run of `scripts/train_contrast_invariant.py` were trained on
  every labelled KK1 still, 20230615-s04 included, and so is anything
  fine-tuned from them. `evaluate_segmentation` and `train_cellpose3` report
  `held_out: false` for any checkpoint whose registered lineage does not prove
  otherwise (`training/registry.py`).

## 5. Other findings from building the tools

- **Label shift in the old scale augmentation.** `scale_variant` in
  `scripts/train_contrast_invariant.py` resizes labels with `cv2.INTER_NEAREST`,
  which shifts every label the same way against the `INTER_LINEAR` image. Over
  cells 9-12 px wide at every sub-pixel phase the shift is +0.38 px on average
  at 0.75x and +0.54 px at 1.35x, up to 0.75 and 0.91 px;
  `INTER_NEAREST_EXACT` leaves +0.13 and -0.06 px on average, scattered both
  ways (`test_nearest_exact_removes_most_of_the_label_shift`). Every
  `--scale-aug` run trained on systematically shifted labels for its scaled
  copies. `training/augmentations.py` uses the exact mode.
- **Label leak in the old contrast augmentation.** It rescaled the image only
  near labelled cells, which paints a halo exactly where the labels are. The new
  contrast operation rescales the whole field about an isotropic local
  background (Gaussian, sigma 20 px). That also replaces `median_filter((1, 41))`,
  which ran across the channels.
- **Refinement is not neutral on human outlines.** Applied to the labels
  themselves (train and val only; the test split was not used for this),
  `refine_to_image` with 3 px leaves 59.5 % (train) and 70.6 % (val) of cells
  at IoU >= 0.5 with their own outline. Their AP over 0.50:0.95 is 0.09 and
  0.10. The geodesic contour leaves 99.2 % and 100 %, with AP 0.34 and 0.44
  *(dev; reproducible with `evaluate_segmentation --predictions` on an .npz of
  the human labels)*. Whether either helps predictions has to be measured on
  val (`evaluate_segmentation --refine`). It must not be assumed from handoff
  item 4.
- **OpenCV's thread pool can kill a run.** Under memory pressure `cv2.resize`
  intermittently crashed inside OpenCV's worker pool (Windows fatal exception
  0xc000070a) during the augmentation tests, after which every cv2 call raised.
  `training/augmentations.py` and both trainers switch OpenCV's pool off
  before its first call.
- **Four scripts are not ported.** `verify_suspect_labels`, `diag_label_gaps`,
  `recover_missing_labels` and `experiment_reconstruction` compare raw pixel
  positions against fixed px thresholds and by default write over published
  `docs/*.json`. They name no order, so `find_sequences` still gives them the
  withdrawn filename ordering, with a `FutureWarning`: a default rerun of
  `diag_label_gaps` and `verify_suspect_labels` reproduces `docs/label_gaps.json`
  and `docs/suspect_labels.json` byte for byte. Their figures rest on filename
  adjacency like the withdrawn ceilings, and they must be ported to
  `src/corridor/learn/brackets.py` before they ask for `order="time"`.

## 6. How to run

```bash
PY=./.venv/Scripts/python.exe
$PY scripts/build_corrected_reference.py --group KK1            # v2; --legacy reruns v1
$PY scripts/measure_label_completeness.py --group KK1           # v2; --legacy reruns v1
$PY -m training.datasets && $PY -m training.splits             # build/registry/; changed content
                                                               # is refused (--supersede, see README)
$PY -m training.registry add --id <id> --path <ckpt> --trained-on-experiments <ids|none>
$PY -m training.evaluate_segmentation --checkpoint <path> --split val --window 3,97 \
    --diameter none --border-margin 3 --refine image --refine contour   # --dry-run first
$PY scripts/evaluate_against_both.py --group KK2 --border-margin 3     # v2 reference by default
$PY -m training.train_cellpose3 --start <path> --name <new> --dry-run
./.venv-cp4/Scripts/python.exe -m training.train_cellpose_sam --start build/cp4_models/cpsam_v2 --name <new> --smoke
$PY -m training.hard_examples <result_dir> ... --out ranking.json
$PY -m training.registry verify
```

Border-margin figures change the measurement, not the model, and are reported
beside the unmodified figure. Outputs:

- `docs/corrected_reference_v2_{KK1,KK2}.json` and
  `docs/label_completeness_v2_{KK1,KK2}.json` (committed)
- `build/corrected_reference_v2/`, `build/corrected_reference_legacy/` and
  `build/registry/` (gitignored)
