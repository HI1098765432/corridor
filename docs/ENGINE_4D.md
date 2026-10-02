# Corridor 2.1 — the deterministic 4D engine (design contract)

Status: in progress, 2026-10-02. Direction from the PI review the same day:
**a foundation segmenter proposes, deterministic mathematics verifies, and the
complete T×Z×Y×X experiment resolves ambiguity.** No language model touches a
pixel. Every stage is a narrow, deterministic worker ("bot") with explicit
inputs and outputs, its numerical evidence exposed, and a ground-truth test.

## 0. The measured starting point

Cellpose-SAM `cpsam_v2` (Cellpose 4.2.1.1, hash-checked weights) was run
zero-shot on the labelled KK2 stills before this design was written, as the
PI asks. It scored F1 0.00 (19 predictions, 8 cells, 0 matches at IoU 0.5).
It outlines each microfluidic channel (~320 × 93 px) as the object; the
11-px-wide cells inside reach IoU ≤ 0.06 (`build/logs/cpsam_probe.json`,
diagnosis in `build/logs/cpsam_diag.py`). So the backend is not switched on
faith. The static atlas below removes exactly the structure `cpsam_v2`
latches onto, and **Experiment B′ (cpsam_v2 on atlas-subtracted images)**
decides whether the deterministic system rescues zero-shot Cellpose 4 before
anyone fine-tunes it. The lab's validated CP3 model stays the locked
production proposer until a benchmark says otherwise.

Cellpose 3 and 4 cannot share one Python environment (different major
versions; a CP3 checkpoint does not load in CP4). The Cellpose 4 proposer
therefore runs as a **worker process** in its own environment
(`.venv-cp4`), exchanging arrays through files. The engine never imports
Cellpose itself.

## 1. Data object

One array `I[t, z, y, x]` with calibration `(dx = dy, dz, dt)`. A 2D movie
has Z = 1 and every bot handles that case without special code paths.
Coordinates everywhere are `(x, y, z)` in pixels / slices plus their
physical versions; nothing is assumed isotropic.

## 2. The bots

Each bot lives in its own module under `src/corridor/engine/`, takes plain
numpy arrays plus a frozen settings dataclass, and returns a frozen result
dataclass with `to_dict()` for the evidence file it writes.

**Bot 1 — registration (`registration4d.py`).** Estimates per-timepoint
translation `(Δx, Δy, Δz)` of the whole volume against one reference
timepoint, chosen as the sharpest (highest variance of the Laplacian) near
the temporal centre. It does not chain frame to frame, because chained
errors accumulate. It uses subpixel phase correlation (upsampled DFT). The
uncertainty is reported per axis from the correlation peak. It applies the
inverse shift with linear interpolation and records
`registration.csv` (t, dx, dy, dz, error, reference_t). Rotation is
estimated only if a setting enables it. It never segments anything.

**Bot 2 — static atlas (`static_atlas.py`).** On the registered data it
computes:
- `B = median_t`, `MAD = median_t |V' − B|`, and the change score
  `D_t = |V' − B| / (1.4826 MAD + ε)`.
- Per-voxel persistence (the fraction of timepoints with `D_t < k`), edge
  stability (the median over t of the gradient-orientation agreement with `B`)
  and Z continuity.

It writes a class map: `STATIC_BACKGROUND`, `CHANNEL_WALL` (combining the
lane geometry from `core/geometry.py`), `OUTSIDE_DEVICE`, `ARTIFACT`,
`VALID_CELL_REGION`, `UNKNOWN`. **Low change alone never marks a region
static**, because a paused cell does not move. Static needs structural
persistence: low change in at least `persist_fraction` of timepoints, a
stable edge geometry, a size or extent beyond plausible cell morphology, and
no track passing through it. It also exports `B` so proposers can be given
the residual `V' − B` (Experiment B′). It never tracks.

**Bot 3 — proposer (`proposer.py`, `cellpose4_worker.py`).** One interface
with two backends: the locked CP3 lab model (in-process) and Cellpose 4
`cpsam_v2` (worker process, explicit hash-checked weights path, never a bare
name). It retains the mask, the cell-probability field and the flows, with
the normalisation recorded. It proposes; it decides nothing.

**Bot 4 — Z consensus (`z_consensus.py`).** It links 2D slice components into
3D objects with a LAP between adjacent slices. The cost is
`w1(1−IoU) + w2·d_centroid/σ + w3|ln(A_b/A_a)| + w4·d_shape`. A slice
missing between two strongly matched neighbours is repaired by
interpolating the neighbouring masks (signed-distance interpolation). The
cost of every link is reported. It decides only Z membership.

**Bot 5 — temporal delta (`temporal_delta.py`).** For object `A_t` it measures
the displacement to `t+1` by ROI phase correlation on the registered images.
It projects `A_t` forward and `A_{t+1}` backward and computes
`E_FB = d(A_t, back(A_{t+1})) + d(A_{t+1}, fwd(A_t))`, with `d` as
1 − IoU plus centroid distance in pixels. Windows of ±1 and ±2 frames are
used. It never estimates volume.

**Bot 6 — measurement (`object4d.py`, `surface_delta.py`, `uncertainty.py`).**
- **Objects.** It builds 4D cell tubes, `C_k(x, y, z, t)`, from tracked
  objects.
- **Per time:** the centroid (sub-pixel), volume and surface area (from the
  existing `detections.extract_detections_3d`), dV/dt, velocity and
  acceleration in 3D.
- **Motion split into translation and deformation:**
  - Whole-cell translation by ROI phase correlation.
  - Local surface displacement `u(s, t)` from the signed-distance fields
    `φ_t` and `φ_{t+1}`. The translation is removed first; then it reports
    extension and retraction area or volume and their locations relative
    to the motion direction (front, rear, flank).
- **Precision floor.** Every displacement carries an uncertainty built from
  pixel size, Z spacing, the noise-to-signal ratio inside the ROI, the
  boundary uncertainty (±0.5 px by default, propagated to the centroid) and
  the registration error. A motion below its floor is reported as such,
  never as biology.

**Bot 7 — identity.** This is the axis-free tracker that already exists
(`core/tracking.py`): every cost term is printed, with a link margin and
forward/backward gap closing. The engine feeds it the consensus-confirmed
objects. A rejected correspondence can be explained term by term, with the
predicted and measured displacement, overlap, volume ratio and E_FB.

**Bot 8 — referee (`consensus.py`).** It combines the evidence (proposer,
atlas class, Z support, past and future temporal support, motion
plausibility, shape plausibility) into explicit states: `CONFIRMED`,
`LIKELY`, `AMBIGUOUS`, `REJECTED`, plus `TEMPORAL_RECOVERY_CANDIDATE`. That
last state covers an object missing at t but present at t−1 and t+1, with
matching pixel structure and plausible motion. Recovery either derives the
mask from adjacent evidence or only flags it, according to a strictness
setting. The rules are explicit thresholds, every decision lists the
evidence that made it, and a proposal inside a confident `CHANNEL_WALL` is
rejected.

## 3. Orchestration (`pipeline4d.py`)

```
load T[Z]YX → calibrate → register → atlas → propose (on V' or V' − B)
→ reject static/impossible → Z consensus → temporal consensus → referee
→ identity (tracker) → 4D tubes → measurements ± uncertainty → QC → export
```

Outputs are added to the v2 result folder:
- `registration.csv`
- `atlas.npz` (background and class map)
- `consensus.csv` (one row per object and time: state and every evidence
  value)
- `tubes.csv` (per track and time: volume, surface area, dV/dt,
  translation, extension, retraction, uncertainty)
- `surface_delta.csv`

The existing `tracks.csv`, `track_summary.csv` and `track_msd.csv` are
unchanged in schema. CLI: `--engine 4d` (the default stays the 2.0 pipeline
until the benchmark below says otherwise).

## 4. Acceptance

Synthetic ground truth for every bot:
- **Registration:** known subpixel drift recovered within 0.05 px.
- **Atlas:** walls are marked static, and a paused cell is not.
- **Z consensus:** a dropped slice is repaired, and two touching objects stay
  separate.
- **Temporal delta:** E_FB is low for a true match and high for a swap.
- **Measurement:** known volume growth, known translation without
  deformation, and known one-sided extension are recovered within the
  stated floor.
- **Referee:** a planted false positive in a wall is rejected, and a planted
  missing frame becomes a recovery candidate.

Real-data benchmark, on the same held-out labelled stills and the same
matcher as everything else:
- the lab model, raw;
- the lab model with the atlas and referee;
- `cpsam_v2`, raw;
- `cpsam_v2` on `V' − B` (B′);
- `cpsam_v2` on `V' − B` with the referee.

Fine-tuning Cellpose 4 is justified only if the best zero-shot arm still
fails reproducibly after the deterministic stages.
