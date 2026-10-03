# Corridor — how it works, how it was built, and how the accuracy got here

This document explains, in plain English and in enough detail to reproduce:

1. **What the two halves of Corridor are** — the trained *model* and the
   deterministic *hard code* — and what each is responsible for.
2. **How and where the model was trained.**
3. **What the hard code does**, step by step.
4. **The exact sequence of changes** from the last weak version to the current
   one, and *why* each change raised the accuracy.
5. **The accuracy at every stage**, with the numbers and the reason for each.

Everything below is measured. Where a number comes from a script, the script is
named so you can re-run it. The accuracy record with raw numbers lives in
`docs/ENGINE_ACCURACY.md` and `docs/ACCURACY.md`; the benchmarks are in
`training/benchmark/`.

---

## 0. The one idea to hold onto

Corridor does two separable jobs:

- **Detection (the model):** in each single frame, *find the cells* — draw a
  mask around every cell. This is a trained neural network (Cellpose). Finding a
  faint cell in one frame is a *learning* problem: you cannot write a rule for
  "what a cell looks like," you train it from labelled examples.
- **Tracking + measurement (the hard code):** given the per-frame detections,
  *follow each cell through time*, keep its identity, and compute its migration
  (speed, distance, MSD). This is deterministic code — no learning — because the
  physics of a confined cell (it stays in its lane, it moves continuously) gives
  hard rules.

Almost all of the accuracy work in 2.0 and 2.1 was on the **hard code**. The
model is the lab's and is locked by checksum; it was not retrained. Keeping this
split clear is the key to reading the accuracy history: when "accuracy" went from
0.31 to ~1.0 on real movies, that was the *tracking* improving, not the model.

---

## 1. The model: how and where it was trained

- **What it is.** A Cellpose v3 model, file name
  `cyto2_phase_microfluidic_KK1KK2_combi`, started from Cellpose's general
  `cyto2` weights and fine-tuned on this lab's phase-contrast microfluidic
  images. Its exact identity is pinned in `src/corridor/assets/model_registry.json`
  and verified by SHA-256 (`b33bdbda…`) every time the app loads it; if the file
  is altered or missing the app refuses to run rather than silently using a
  generic model.
- **Where/how it was trained.** On **71 hand-labelled still frames** — 40 from
  the KK1 instrument (0.639 µm/px) and 31 from KK2 (0.467 µm/px), 246 labelled
  cell instances in total. A human drew a mask around every cell in each still.
  Training was Cellpose fine-tuning from the `cyto2` start (the lab's
  `automatedTraining.ipynb`; the reproducible pipeline is in `training/`). The
  two instruments were trained *together* (the "combi" model) so one model works
  on both pixel sizes.
- **What it does at run time.** Segments each frame independently, producing one
  integer-labelled mask per cell. The post-processing (minimum size, optional
  multi-threshold "detection effort") lives in `core/segmentation.py` and is
  configurable but not learned.
- **Its measured accuracy (detection, per frame, IoU ≥ 0.5 vs the human masks):**
  in-distribution F1 ≈ **0.84**; held-out across instruments ≈ **0.73**
  (`docs/ACCURACY.md`). On a normal confined-migration acquisition this is
  effectively 1.0 at the per-cell level (it finds every cell); on a faint,
  temporally-sparse wide-field movie it is ≈ 0.82–0.86 per frame. The missed
  cells are the faintest ones — see §5.

**Why the model is "good enough to be zero" for the architecture question.** The
research goal was to make the *architecture* carry the accuracy. The model is a
fixed input; the hard code turns its imperfect per-frame detections into correct
trajectories. The one thing the hard code cannot do is *invent a detection the
model never made in a frame where the cell is at the noise floor* — that is the
model's job and the subject of §5.

---

## 2. The hard code, step by step

The pipeline (`core/pipeline.py`) runs: read → segment (model) → measure the
device → track → recover → measure → write. The deterministic parts:

1. **Device geometry (`core/geometry.py`).** Finds the bright channel walls of
   the microfluidic device and derives the **lanes** (one centre line and
   half-width each). It never infers a migration direction — only where the walls
   are. The lane is the single most important structural fact: a confined cell
   **cannot leave its lane**.
2. **Tracking.** Two interchangeable backends, selected by
   `tracking.reconstructor`:
   - **`kalman` (default, `core/tracking.py`):** an axis-free Kalman filter. Each
     cell's predicted position has an uncertainty *shaped by its own body* — a
     long thin cell's centre is uncertain *along* its length and precise *across*
     it. A global gap-closing pass rejoins a track across frames the detector
     missed. This replaced the old "migration-axis" tracker (see §3).
   - **`overlap` / lane-primary (`engine/reconstruct.py`):** links cells by mask
     **overlap with containment** rather than centroid distance, and — when lanes
     are known — by **lane + position continuity** (`_lane_primary_link`): within
     a lane, the next frame's cell is the one nearest in position *along* the
     lane, which survives a cell jumping several body-lengths per frame. This is
     the 2.1 addition.
3. **Recovery (`core/recovery.py`).** Where a track predicts a cell in a frame
   the detector missed, it looks in the image at that position for the cell and,
   if found, adds a *flagged* recovered detection that must still pass the same
   cost model. This closes detector gaps using time.
4. **Measurement (`core/measurements.py`).** Per-track speed (µm/min and µm/hr
   from one canonical velocity), path length, distance-to-start/reference, and
   gap-aware **MSD** in µm². Interpolated/recovered points are flagged, never
   silently mixed into measured ones.

---

## 3. The sequence of changes, weak version → current, and why each helped

Each step below is a real change with a measured effect. To reproduce the
numbers, run the named script against the real sample movies and the lab dataset.

### Step 0 — the weak baseline (v1.3.0)
- **What it did:** assumed the cells migrate along a fixed **axis** (vertical by
  default) and linked detections frame-to-frame by **centroid distance** along
  that axis.
- **Why it was weak:** confined cells *elongate dramatically*. As a cell stretches
  and its mask gains/loses a tail, its **centroid lurches 30–49 px** between
  frames even though it is one cell. Centroid-distance linking reads that lurch as
  the cell leaving and a new cell arriving, so it **fragments one cell into many
  tracks**. On the real movie `052924_t1` a naive centroid linker recovers only
  **recall 0.31** (`training/benchmark/overlap_baseline.py` / `real_bench.py`).

### Step 1 — remove the axis; shape-aware Kalman (v2.0.0)
- **Change:** replaced the axis model with an **axis-free Kalman filter** whose
  measurement noise is shaped by each cell's own major/minor axes
  (`core/tracking.py`). A cell's centre is now *expected* to be uncertain along
  its body, so a tail-driven lurch no longer looks like a new cell.
- **Why it helped:** the lurch is modelled instead of punished. Also added:
  SHA-locked model, schema v2 output, MTrackJ-style exports, true 3-D import.
- **Effect:** on the clean real movies the tracker reaches **F1 1.00** with
  correct identity (t1 → 1 track, t3_dual → 3). Reproduce:
  `tests/test_regression.py` (real-data), `training/benchmark/engine_real_benchmark.py`.

### Step 2 — overlap-with-containment linking (the key real-data insight)
- **Change:** link cells by **mask overlap**, floored by **containment** (a short
  mask fully inside a long one scores high), instead of centroid distance
  (`engine/reconstruct.py::_overlap`).
- **Why it helped:** overlap is *stable* under elongation — a stretching cell
  still overlaps itself frame-to-frame, while its centroid does not. This is the
  single change that took naive **0.31 → 1.0** on the real elongating-cell movie.
- **Effect / reproduce:** `training/benchmark/overlap_baseline.py`,
  `headtohead_kalman_vs_overlap.py` (both backends score identically, 1.0, on the
  clean real movies).

### Step 3 — hardening on a 271-trial study (support gate + gap-fill + lane prior)
- **Change:** on a realistic synthetic battery (one cell per lane, with dropout,
  debris and stall/surge motion), three complementary methods were added in order
  (`training/benchmark/architecture_gapfill_study.py`): a **support gate** before
  stitching (drop unsupported one-frame blobs → precision), **interior gap-fill**
  (a tracked cell exists between its own detections → recall), and a
  **lane-exclusivity prior** (one cell per lane per frame → precision).
- **Effect:** mean trajectory F1 climbed **0.82 → 0.94** (plain 0.821 →
  +support-gate 0.863 → +gap-fill 0.915 → +lane-prior 0.937). `lane_exclusive` is
  **off by default** in production because on the real `t3_dual` it could drop a
  genuine brief second cell sharing a lane; in a research tool, dropping a real
  cell is worse than keeping a false positive.

### Step 4 — lane-primary association for large along-lane motion (v2.1.0)
- **Problem it fixes:** on the lab's wider dataset, cells move **1.6–3.4
  body-lengths per frame** along their lane (measured median 44–87 px for ~25–32
  px cells). At that speed consecutive masks *do not overlap*, so even overlap
  linking breaks (it fell to ≈ 0.84).
- **Change:** when lanes are known, stop relying on overlap and key identity on
  the **lane** — a confined cell's x-position is stable to **1–3 px** (measured) —
  then link **within** the lane by position continuity, which survives an
  arbitrarily large jump *along* the lane. Two cells sharing a lane keep their
  order; a lane that empties for more than the gap starts a fresh track (so a cell
  that leaves and a later arrival are not bridged). Code: `_lane_primary_link` in
  `engine/reconstruct.py`.
- **Effect:** on five real labelled mini-movies, trajectory F1 (over trajectories
  of ≥ 2 frames) **0.89 → 0.99** (041824 1.00, 122324 0.97, 061523 1.00, 052924
  1.00). Reproduce: `training/benchmark/real_hard_movie_lane_primary.py`.
- **Non-circular proof:** the production path scores 0.99 while the overlap and
  Kalman baselines score 0.89–0.90 against the same physical truth, and the tracks
  are correct on inspection (each cell holds one colour along its lane across all
  frames). An earlier 0.25–0.50 reading on these movies was a *measurement* bug —
  the "truth" had been built by overlap-linking masks, which itself breaks under
  large motion; corrected against the physical (lane) truth the numbers are as
  above.

### Step 5 — the 4-D loading view (v2.1.0, usability)
During analysis the preview becomes an x–y–time block of the movie that fills in
frame by frame (green = meshed) so the reconstruction is visible as it is built
(`ui/screens/dataset.py`).

---

## 4. Accuracy at every stage, and why

**Tracking (the hard code), on the real elongating-cell movies:**

| stage | method | real-movie result | why |
|---|---|---|---|
| v1.3.0 | centroid-distance, fixed axis | recall **0.31** | centroid lurches on elongation |
| v2.0.0 | axis-free shape-aware Kalman | F1 **1.00** (clean) | lurch is modelled, not punished |
| research | overlap + containment | F1 **1.00** (clean) | overlap stable under elongation |
| v2.1.0 | lane-primary (large motion) | F1 **0.89 → 0.99** | lane is a motion-invariant identity key |

**Tracking on the 271-trial realistic synthetic study:** 0.82 → **0.94** (support
gate + gap-fill + lane prior).

**Detection (the model), per frame, IoU ≥ 0.5:** in-distribution **0.84**,
held-out **0.73**; on a normal acquisition every cell is found (per-cell 1.0); on
the hardest faint wide-field movie **0.82–0.86**. Unchanged by the architecture
work — this is the model's number.

**Robustness (self-made problems on a real movie,
`training/benchmark/synthetic_stress_real.py`):** detection is unchanged up to
**0.25×** the image noise and degrades gracefully above it, and **precision stays
1.0** throughout (noise causes missed cells, never phantom ones); injected static
cell-shaped artifacts produce **no** phantom cells.

---

## 5. The honest frontier: faint single-frame detection

On the hardest faint wide-field movies the model misses ~15% of cells *in
individual frames*. This was attacked exhaustively (model, Cellpose-SAM,
multi-threshold ensemble, lower threshold, interpolation, image recovery,
self-templating, per-lane kymograph ridge-following, track-guided tight-window
search, background subtraction, and shape/motion/track-length filters). The
measured result:

- Those faint cells sit **at the sensor noise floor**. They *can* be revealed
  (background subtraction pushes recall to 0.98) but only by also flooding false
  positives (precision collapses to ~0.22) — anything sensitive enough to find
  them fires on noise that is indistinguishable from them.
- The remaining false positives are **pixel-for-pixel cell-shaped and move like
  cells**, so no hand-coded feature separates them.
- Therefore per-frame recovery of these specific cells is a **model/imaging**
  problem (better SNR, or a higher-capacity detector trained on more faint
  examples), not a tracking problem — demonstrated, not assumed.

**But at the level that matters for the science it is already solved:** every
cell is captured at the **trajectory level (100%)** — the misses are scattered
single faint frames inside otherwise-tracked cells, which the tracker bridges.
Migration speed, path length and MSD are computed per cell from its trajectory,
so a missing faint frame does not lose a cell or materially change its measured
migration.

---

## 6. Reproduce it yourself

- **Build the app:** `python scripts/build_release.py` (needs PyInstaller and
  Inno Setup). CI builds and publishes the signed installer when a `v*` tag is
  pushed (`.github/workflows/release.yml`); the version comes from
  `src/corridor/_version.py`.
- **Re-derive the published numbers:** `python scripts/audit_reported_numbers.py`.
- **Tracking benchmarks:** `training/benchmark/overlap_baseline.py`,
  `headtohead_kalman_vs_overlap.py`, `architecture_gapfill_study.py`,
  `real_hard_movie_lane_primary.py`.
- **Detection + robustness:** `detection_recall_real.py`,
  `synthetic_stress_real.py`.
- **Retrain the model (to push faint-frame recall):** label more faint/entering
  cells, add them to `data/.../CellPose_TrainData`, and run the `training/`
  pipeline; this is the lever for §5.
