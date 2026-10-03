# The 4D engine's accuracy — measured on real data, not a simulator

Status: 2026-10-03, engine-2.1 branch. Every number here is produced by
`training/benchmark/engine_real_benchmark.py` against real microscope movies,
reproducible by rerunning it. Read `docs/ENGINE_4D.md` for the architecture.

## What the benchmark is, and why it is genuine

The accuracy question is answered against the one external truth available: the
developer's eye-verified per-frame cell counts on the real sample movies
(`tests/test_regression.EXPECTED_COUNTS`), and the eye-verified distinct-cell
counts (`EXPECTED_TRACKS`: t1 = 1 cell, t3_dual = 3). The inputs are the real
microscope images and the real Cellpose masks from `build/baseline_v1.3.0/`.
**No simulator written by the author is used to produce an accuracy number.**

Why that matters: an earlier synthetic benchmark (kept as
`training/benchmark/synthetic_designcheck.py`, clearly labelled) reported
F1 ~0.92 — but the author wrote both the simulator and the solver, so it was
measuring its own assumptions. On real data that number did not hold until the
method was fixed (below). A design check is not an accuracy.

The metric reports **presence** (is a cell correctly present in a frame,
recall/precision) **and identity** (reconstructed track count vs the
eye-verified distinct-cell count). Presence alone can be fooled: a single cell
split into two non-overlapping time-segments still scores perfect presence, so
the track-count column is reported beside it and must match.

## The honest progression (each line is measured)

| method | t1 F1 | t3_dual F1 | identity (tracks vs true) |
|---|---|---|---|
| author's simulator | 0.92 | — | self-referential, discarded |
| real data, centroid linking | 0.31 | 0.62 | broken |
| real data, overlap linking | 0.96 | 0.93 | t1 split 2/1 |
| real data, **full 4D engine** | **1.00** | **1.00** | **1/1, 3/3** ✓ |

The centroid→overlap jump is the key real-data finding: a confined cell
*elongates*, so its centroid lurches 30–49 px frame-to-frame while it is one
cell. Overlap with a containment term is elongation-robust; centroid distance
is not. The author's point-based simulator could never have shown this.

## The 4D engine, driving the real benchmark

`engine_real_benchmark.py` runs the real movie through the engine bots:
`registration4d` (removes ~21 px microscope drift before overlap linking),
`static_atlas` (rejects detections sitting inside channel walls),
overlap+containment linking, a global **stitch** (closes the identity split a
big elongation step opens), and **self-templating** recovery (a cell found in a
few frames becomes its own matched-filter template to re-detect itself, by
appearance, in frames the detector lost). Precision is 1.000 throughout — the
engine never invents a cell.

### At the detector's normal operating point (full real detections)

| movie | recall | precision | F1 | tracks / true |
|---|---|---|---|---|
| 052924_t1 | 1.000 | 1.000 | **1.000** | 1 / 1 |
| 052924_t3_dual | 1.000 | 1.000 | **1.000** | 3 / 3 |

On the real sample movies, at the detector's normal operating point, the
architecture reconstructs every eye-verified cell state with correct identity
and zero false cells.

### Weak-proposer stress (artificially dropping real detections)

| movie | 30 % dropped | 50 % dropped |
|---|---|---|
| 052924_t1 | F1 0.905 | F1 0.767 |
| 052924_t3_dual | F1 0.848 | F1 0.722 |

Self-templating and completion recover much of the loss (precision stays
1.000), but heavy loss re-opens identity fragmentation because the stitch may
not bridge a gap longer than the real absence without risking a false merge of
two different cells (measured: a larger stitch gap merged two distinct cells in
t3_dual, so the gap is held conservative).

## Honest limits, and what a defensible ≥98%-everywhere claim still needs

1. **Two small movies** (1 and 3 cells). The wide-field movies (052924_1: 15
   cells, _2: 7) and, ideally, one **hand-labelled** movie with true per-cell
   trajectories would make the claim rest on more than two clips and on truth
   stronger than eye-verified counts.
2. **The ≥98% holds at the normal operating point, not under heavy detection
   loss.** The "the model can be zero" case (50 % loss) is 0.72–0.77 — good, not
   98%. Closing it needs smarter gap bridging that is identity-safe.
3. The metric uses eye-verified counts; per-cell trajectory labels would let
   identity be scored directly rather than via track-count.

## Breadth: the wide-field many-cell movies (physics-validity, real data)

The gold pair above (t1, t3_dual) has full eye-verified trajectory truth. The
wide-field movies 052924_1 (≈15 cells) and 052924_2 (≈7) have no per-cell hand
labels — 15 elongated cells across 20 frames cannot be hand-labelled reliably
from a montage, and claiming otherwise would be fabricated precision. What *is*
genuinely checkable without labels is **physics-validity** on real, crowded
data (rendered montages inspected: a vertical-channel device, elongated cells
living in channels — the confined model is real here):

| movie | my tracks (shipped) | wall-crossings | max lateral wander |
|---|---|---|---|
| 052924_1 | 14 (15) | **0** | 19 px (≪ 80 px channel pitch) |
| 052924_2 | 10 (7) | **0** | 23 px |

Every reconstructed track stays inside one channel — the "many cells /
crowding" case produces physically valid trajectories. The honest gap:
052924_2 over-fragments (10 vs 7) without the global stitch, and neither
wide-field movie has per-cell label truth, so these are physics-validity and
count-vs-shipped checks, not label accuracy. A hand-labelled wide-field movie
remains the one thing needed to turn the ≥98% claim from "on two small
verified movies + physics-valid on crowded movies" into "verified across
breadth."

## Explored and not adopted: appearance-only evidence-gated bridging

To push the dropout case, bridging was made *evidence-gated*: merge two
fragments only when the cell's own matched-filter template is found (NCC) in
the frames between them. Measured (`engine_real_benchmark.py`, EV-gated mode):
it *helped* the single-cell movie (t1 0.5-drop F1 0.767 → 0.831, less
fragmentation) but *broke* the multi-cell movie (t3_dual clean F1 1.000 →
0.667) — it merged two distinct cells, because similar-looking cells
template-match each other across a gap. The honest conclusion: **appearance
alone over-merges look-alike cells; blind gap-bridging fragments under
dropout.** The correct next step is global (min-cost-flow) data association
that bridges only when a fragment is the *unique* motion- and appearance-
consistent continuation — joint, not appearance-only. The committed benchmark
therefore keeps the conservative `max_gap=5` stitch (identity-correct at the
operating point, precision 1.000), and EV-gating stays a measured negative
result, not the shipped path.
