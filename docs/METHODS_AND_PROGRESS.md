# Corridor: how it works, how it was built, how the accuracy got here

Plain and reproducible. Every number is measured; the script that produces it is
named. Raw numbers: `docs/ENGINE_ACCURACY.md`, `docs/ACCURACY.md`. Benchmarks:
`training/benchmark/`.

## The two parts

- **Model (Cellpose):** in one frame, draw a mask on each cell. Trained, because
  "what a cell looks like" can't be written as a rule.
- **Hard code (deterministic):** detect-recover across frames, keep each cell's
  identity through time, measure migration. Rule-based, because the physics of a
  confined cell is fixed: it stays in its lane and moves continuously.

Detection is step 1 — nothing downstream is right if a cell is never found. Both
parts detect: the model per frame, the hard code across frames (it recovers cells
the model missed, using time and the lane). The accuracy work in 2.0/2.1 was in
the hard code; the model was not retrained.

## 1. The model

- File `cyto2_phase_microfluidic_KK1KK2_combi`, Cellpose v3, started from `cyto2`
  and fine-tuned on **71 hand-labelled stills** (40 KK1 at 0.639 µm/px, 31 KK2 at
  0.467 µm/px; 246 labelled cells). Both instruments trained together.
- Locked by SHA-256 (`b33bdbda…`) in `src/corridor/assets/model_registry.json`;
  verified at load, no fallback.
- Training: Cellpose fine-tuning from `cyto2` (lab notebook
  `automatedTraining.ipynb`; reproducible pipeline in `training/`).
- Detection accuracy (per frame, IoU ≥ 0.5 vs human masks): **0.84**
  in-distribution, **0.73** held-out, **0.82–0.86** on faint wide-field movies.
  Per cell across a movie it finds every cell (see §5).

## 2. The hard code (`core/pipeline.py`)

read → segment (model) → find lanes → track → recover → measure → write.

- **Lanes (`core/geometry.py`):** locate the device walls, derive each lane's
  centre and width. No migration direction is inferred. A cell cannot leave its
  lane — the key fact for tracking and recovery.
- **Track.** Two backends (`tracking.reconstructor`):
  - `kalman` (default, `core/tracking.py`): a Kalman filter whose position
    uncertainty is shaped by each cell's body (uncertain along its length,
    precise across it), with global gap-closing.
  - `overlap` / lane-primary (`engine/reconstruct.py`): links by mask overlap
    with containment, and when lanes are known by lane + position-along-lane
    continuity (`_lane_primary_link`), which survives a cell jumping several
    body-lengths per frame.
- **Recover (`core/recovery.py`):** where a track predicts a cell in a missed
  frame, search the image there and add a flagged detection that must pass the
  same cost model. A background-subtraction detector
  (`training/benchmark/bg_detector.py`) also finds faint moving cells the model
  missed (§5).
- **Measure (`core/measurements.py`):** speed (µm/min, µm/hr), path length,
  distance-to-reference, gap-aware MSD (µm²). Recovered/interpolated points are
  flagged.

## 3. Changes, weak version → now, with the reason each worked

**v1.3.0 (weak):** fixed migration axis, centroid-distance linking. Elongating
cells' centroids lurch 30–49 px frame to frame, so one cell split into many
tracks. Real movie `052924_t1`: recall **0.31**
(`training/benchmark/overlap_baseline.py`).

**v2.0.0 — axis-free, shape-aware Kalman (`core/tracking.py`).** The centre is
expected to be uncertain along the body, so the lurch is modelled. Clean real
movies: **F1 1.00**, correct identity (`tests/test_regression.py`). Also:
SHA-locked model, schema v2 output, MTrackJ exports, 3-D import.

**Overlap-with-containment linking (`engine/reconstruct.py`).** Overlap is stable
under elongation; centroid is not. Single change, naive **0.31 → 1.00** on real
elongating cells (`headtohead_kalman_vs_overlap.py`).

**271-trial hardening (`architecture_gapfill_study.py`).** Support gate (drop
one-frame blobs) + interior gap-fill + lane-exclusivity prior. Mean trajectory F1
**0.82 → 0.94** (0.821 → 0.863 → 0.915 → 0.937). `lane_exclusive` is off by
default: it can drop a real brief second cell sharing a lane.

**v2.1.0 — lane-primary association (`_lane_primary_link`).** On wider data cells
move 1.6–3.4 body-lengths/frame; overlap then breaks (≈0.84). A cell's lane is
stable to 1–3 px, so identity keys on the lane and links along it. Five real
labelled mini-movies: **0.89 → 0.99** (`real_hard_movie_lane_primary.py`). An
earlier 0.25–0.50 reading was a measurement bug (truth built by overlap-linking
masks, which also breaks under large motion); against the lane truth it is 0.99.

**v2.1.0 — 4-D loading view.** During analysis the preview becomes an x–y–time
block that fills in frame by frame as the engine works (`ui/screens/dataset.py`).

## 4. Accuracy

| what | before | now | reason |
|---|---|---|---|
| tracking, elongating cells (real) | 0.31 | 1.00 | overlap/shape vs centroid |
| tracking, large-motion cells (real) | 0.89 | 0.99 | lane-primary vs overlap |
| tracking, 271 synthetic trials | 0.82 | 0.94 | gate + gap-fill + lane prior |
| detection, per frame (model) | 0.73–0.86 | 0.73–0.86 | unchanged |
| detection, per cell/trajectory | — | 1.00 | every cell found in enough frames |
| noise robustness | — | perfect to 0.25×, precision 1.0 | noise causes misses, not phantoms |

Robustness (`synthetic_stress_real.py`): on a real movie with injected noise,
detection is unchanged up to 0.25× the image noise and degrades gradually above
it; precision stays 1.0 (misses, never phantom cells). Injected static
cell-shaped artifacts add no phantom cells.

## 5. Detection of faint cells: what is and isn't possible

Step 1 is done at the level that matters: on a normal acquisition every cell is
found, and on the hardest faint movie every cell is still found in enough frames
to be tracked (**per-cell recall 1.0**). What is missed is isolated faint
**frames** of cells that are found in their other frames.

Recovering those specific frames was attacked with ~15 methods: the model, a
lower threshold, multi-threshold ensemble, Cellpose-SAM, interpolation, image
recovery, self-templating, per-lane kymograph ridge-following, track-guided
search, background subtraction, and shape / motion / length / temporal-coherence
filters. Result:

- Background subtraction **finds** the faint cells — recall **0.98**.
- But they sit at the **sensor noise floor**: raising recall to 0.98 floods false
  positives (precision **0.22**). No feature separates a faint cell from noise.
  Spatially they are the same brightness and shape. Temporally they are also
  inseparable — real migrating cells jitter *more* (median 49 px vs 18 px) and are
  *less* straight (0.3 vs 0.6) than noise, because they stall, surge and reverse,
  so coherence filters remove real cells, not noise
  (`training/benchmark/bg_coherence.py`).

This is an imaging limit (signal-to-noise), not a code limit. The lever is more
labelled faint cells for the model or higher-SNR acquisition — not another
tracking rule. Per-cell capture is already complete, so migration speed, path
length and MSD (computed per trajectory) are not affected by a missing faint
frame.

## 6. Reproduce

- Build app: `python scripts/build_release.py` (needs PyInstaller; installer step
  needs Inno Setup). CI publishes the signed installer on a `v*` tag
  (`.github/workflows/release.yml`) once the model is available to it
  (secret `CORRIDOR_MODEL_URL`).
- Re-derive published numbers: `python scripts/audit_reported_numbers.py`.
- Tracking: `overlap_baseline.py`, `headtohead_kalman_vs_overlap.py`,
  `architecture_gapfill_study.py`, `real_hard_movie_lane_primary.py`.
- Detection / robustness: `detection_recall_real.py`, `bg_detector.py`,
  `bg_coherence.py`, `synthetic_stress_real.py`.
- Push faint-frame recall: label more faint/entering cells into
  `data/.../CellPose_TrainData`, retrain via `training/`.
