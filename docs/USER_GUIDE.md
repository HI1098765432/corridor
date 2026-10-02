# Corridor — how it works and how to use it

Confined cell migration analysis for Windows. You give it a phase-contrast
time-lapse of cells in microfluidic channels; it gives you each cell's
trajectory and migration speed, together with everything you need to check
whether to believe them.

---

## Contents

1. [Install](#1-install)
2. [The five-minute version](#2-the-five-minute-version)
3. [What it does to your images, stage by stage](#3-what-it-does-to-your-images-stage-by-stage)
4. [Reading the results screen](#4-reading-the-results-screen)
5. [Every output file](#5-every-output-file)
6. [Which speed number to publish](#6-which-speed-number-to-publish)
7. [Settings that matter, and when to change them](#7-settings-that-matter-and-when-to-change-them)
8. [Quality-control warnings and what each one means](#8-quality-control-warnings-and-what-each-one-means)
9. [How accurate is it](#9-how-accurate-is-it)
10. [Running without the interface](#10-running-without-the-interface)
11. [When something goes wrong](#11-when-something-goes-wrong)

---

## 1. Install

Download **`Corridor-1.2.0-Setup.exe`** and run it. It installs for your user
account only, so no administrator password is needed.

Everything is inside: Python, PyTorch, Cellpose v3, Napari, and the trained
segmentation model. Nothing else to install, no internet needed after download.
About 1 GB on disk.

Windows will show a **SmartScreen warning** on first run ("Windows protected
your PC"). Click *More info* → *Run anyway*. The installer **is** signed, but
with a self-signed certificate; removing that warning requires a commercial
certificate tied to a verified legal identity. You can confirm you have the real
file by checking its fingerprint:

```powershell
Get-FileHash Corridor-1.2.0-Setup.exe -Algorithm SHA256
```

It should read `99bb4a4bdf552c2cde4932fdd8692e4113a74ce99c1a8d57d784e8757e29720f`.

**After installing, check it works:**

```powershell
& "$env:LOCALAPPDATA\Corridor\Corridor.exe" --self-test
```

Nine checks run: the scientific stack loads, the interface toolkit loads, the
bundled model matches its published checksum, a full PyTorch/Cellpose pass
executes, a known trajectory tracks to the expected velocity, every output
format round-trips, the project store opens, the processing device is reported,
and Napari builds its layers. All nine should say `[ok]`.

---

## 2. The five-minute version

1. **Open Corridor** and drag your `.tif` time-lapse onto the window.
2. Check the two numbers it shows you: **pixel size** and **frame interval**.
   It reads these from the file. If it says *"not found"*, type them in — every
   speed depends on them and a wrong value scales every result.
3. Click **Analyse**. A 20-frame stack takes roughly 1–3 minutes on a laptop
   CPU.
4. Read the **Checks** tab first, not the numbers. It lists anything that
   deserves a second look.
5. Open **`track_summary.csv`** for one row per cell.

That is the whole workflow. The rest of this document is about understanding
and trusting what comes out.

---

## 3. What it does to your images, stage by stage

### Stage 1 — Reading the file

Reads the TIFF's axis order rather than assuming it. A stack can be stored as
`TYX`, `ZYX`, `QYX` and several other orders, and a file with a colour axis is
*never* treated as if that axis were time.

It then looks for **calibration** in this order, and tells you which it used:

| Source | What it reads |
|---|---|
| Embedded ND2 metadata | `dCalibration` (µm/px) and `dAvgPeriodDiff` (ms between frames) |
| ImageJ TIFF tags | `finterval`, `spacing` |
| Plain TIFF resolution tags | X/Y resolution |
| You | Whatever you type in |

Every result carries the source, so "0.467 µm/px (embedded ND2 metadata)" is
a different claim from "0.46 µm/px (you typed it)" and the interface shows
which one applies.

> **Why this matters.** The research code this replaced assumed 10 minutes per
> frame where the files actually record 20.0069, and 0.46 µm/px where they
> record 0.467060343. The timing error alone overstated every velocity by
> **97 %**. Calibration is the single largest source of wrong numbers in this
> kind of analysis, and it is silent.

### Stage 2 — Finding the cells

Each frame goes through **Cellpose v3** with the bundled model
(`cyto2_phase_microfluidic_KK1KK2_combi`), trained by the researchers on 71
hand-labelled phase-contrast images of this device.

Two counts are kept separately for every frame: how many objects Cellpose
produced, and how many survived the size filter. Without that separation, "the
cell disappeared" is unattributable — you cannot tell whether the model failed
or a filter removed it. Both are in `segmentation_diagnostics.csv`.

### Stage 3 — Measuring the device

Cells in channels move along the channel, so the software needs to know which
way that is. It does **not** guess from the cells — a round cell has no
direction, and a cell's own orientation is redefined every frame.

Instead it measures the **device**: it builds a temporal median of the stack
(cells move, walls do not, so the walls survive and the cells wash out), traces
the wall ridges with iterative re-centring, and fits the migration axis to
them. On the supplied data this finds a **−2.2° to −2.8° tilt** and a channel
pitch of 81.5–85 px, and it locates six channels in a wide field.

Each channel boundary is marked as **measured** or **inferred**. An inferred
boundary is one that was too faint to see and was filled in from the spacing of
its neighbours — an assumption about the device, and flagged as such.

### Stage 4 — Linking cells between frames

This is the part that decides identity, and it is where most tracking software
quietly guesses. Corridor does not. Every possible pairing is scored in
**chi-square units** — each term is a squared error divided by the spread it is
allowed to have, so a value of 1.0 means "one standard deviation off":

```
cost = (distance along the channel  / expected spread)²
     + (distance across the channel / expected spread)²
     + (log of the area ratio       / expected spread)²
     + a penalty for reversing direction
     + a penalty for an implausible orientation
```

The pairings are solved as a single assignment problem that includes a real
**"no match" option** for every cell. Linking two detections has to beat the
cost of leaving both unmatched, so a bad link is rejected rather than forced.

Four things are structural, not tunable preferences:

- **A cell never crosses a channel wall.** Different channels are different
  cells, full stop.
- **Gaps are handled by elapsed time.** A cell missing for three frames is
  predicted three frames forward, with a correspondingly wider tolerance — not
  treated as if one frame had passed.
- **A cell that stalls is still the same cell.** The expected spread scales
  with the predicted travel, so a cell decelerating from 44 px/frame to 0 does
  not get split into two tracks.
- **Refusals are recorded.** Every track that starts mid-stack and was *not*
  joined to an earlier one appears in `unlinked_starts.csv` with the distance,
  the gap, the implied speed, what the match would have cost, and which rule
  refused it. You can disagree with the judgement on the numbers.

### Stage 4b — Looking again where a track says a cell should be

Where a track has a **gap** — observed before *and* after, missing in between —
the cell was there and the detector missed it. Corridor re-examines those frames
using the position interpolated between the two observations, then tries three
tiers in order: re-segmenting a crop, re-segmenting it at a permissive
threshold, and a plain intensity test with no network involved.

Measured against real holes: **10 of 13 filled, with no false positives**, at a
median error of 3.5 px (about a quarter of a cell's width).

Frames *after* a track's last observation are **not** examined by default. A
track that ends may be a cell that left, died, or drifted out of focus, and
there is no evidence on the far side. Measured, those frames produced one true
recovery and two false ones — so they are off, and turning them on is a
deliberate act.

Anything recovered is marked with its tier and a confidence in
`detections.csv`, and counted in `run.json`. A recovered position is a weaker
claim than a detection and never looks identical to one.

### Stage 5 — Velocities

Instantaneous velocity between consecutive observations, always divided by the
**elapsed time**, never by "one frame". Per-track summaries follow in
[section 6](#6-which-speed-number-to-publish).

### Stage 6 — Saving

Results are written to disk **before** any viewer opens, and each stage saves as
it finishes. If tracking fails, the segmentation that took ten minutes is
already saved and still valid.

---

## 4. Reading the results screen

**Overview** — the trajectory overlay on your images. Use the timeline to step
through frames; click a track to select it.

**Tracks** — one row per cell. Click a row to highlight it on the image.

**Checks** — *read this first.* Every quality-control finding, most serious
first, each one clickable to jump to the frame it concerns.

**Provenance** — what actually produced these numbers: the model file and its
checksum, the Cellpose version, every threshold, where the calibration came
from, the normalisation and detection-effort settings, and how many instances
were produced versus kept.

---

## 5. Every output file

| File | One row per | What it is for |
|---|---|---|
| `track_summary.csv` | track | **The main result.** Duration, path length, straightness, six speed estimates |
| `tracks.csv` | observation | Position, area, gap, match cost, instantaneous velocity |
| `detections.csv` | detected object | Everything found, independent of tracking, with its provenance |
| `segmentation_diagnostics.csv` | frame | Raw vs kept counts, and Cellpose's own message |
| `tracking_events.csv` | frame | Matches, new tracks, dormancies, suspected merges |
| `qc_issues.csv` | finding | Everything worth a second look |
| `unlinked_starts.csv` | refusal | Why a mid-stack track was not joined to an earlier one |
| `recovery_attempts.csv` | attempt | Every gap examined, whether or not a cell was found |
| `run.json` | run | Full provenance: versions, model checksum, every parameter |
| `masks.npz` | run | The label image stack |

Column names always carry their units. `speed_um_per_min` and
`speed_px_per_frame` are two different numbers and both are exported, so a
column's meaning never depends on whether a calibration happened to be present.

---

## 6. Which speed number to publish

`track_summary.csv` gives six. They are not interchangeable, and the difference
between them is measured, not theoretical:

| Column | What it is | Use it when |
|---|---|---|
| **`net_speed_um_per_min`** | Straight-line start→end distance ÷ elapsed time | **The default choice.** Most robust to missed detections |
| **`along_speed_um_per_min`** | Net progress *along the channel* ÷ elapsed time | Confined migration assays — usually what you mean |
| `path_speed_um_per_min` | Total path length ÷ elapsed time | You want distance travelled including wandering |
| `mean_speed_um_per_min` | Mean of the per-interval speeds | Comparing with older analyses that used it |
| `median_speed_um_per_min` | Median of the per-interval speeds | You want resistance to a single outlier interval |
| `max_speed_um_per_min` | Fastest single interval | Looking for bursts |

**Why net speed is the default.** Deleting detections at random from a stack
whose true answer is known:

| detections lost | net speed error (median / 90th pct) | mean speed error |
|---:|---:|---:|
| 10 % | **0.0 % / 2.9 %** | 5.2 % / 12.4 % |
| 15 % | **0.0 % / 2.7 %** | 9.1 % / 23.7 % |
| 20 % | **0.0 % / 4.6 %** | 7.2 % / 19.3 % |
| 30 % | **0.0 % / 8.7 %** | 11.6 % / 27.6 % |
| 40 % | 3.6 % / 62.0 % | 14.2 % / 48.1 % |

Net speed reads only the first and last observation, so a missed frame in
between costs it nothing. The mean of instantaneous speeds loses a sample *and*
merges two short intervals into one long one, which under-reads a wandering
cell.

**The cliff is at 40 %, and it is a cliff, not a slope.** Up to 30 % loss the
median error is zero and at most 3 % of tracks truncate. At 40 % the endpoints
themselves start disappearing, 13 % of tracks truncate, and the 90th-percentile
error jumps from 8.7 % to 62 %. So the rule is a threshold, not a gradient:
**check `segmentation_diagnostics.csv` for frame coverage before quoting
anything**, and if more than about a third of frames found nothing, treat the
speeds as indicative only.

`straightness` (net ÷ path, from 0 to 1) tells you how much the cell wandered.
Near 1 means it went straight; well below means net and path speeds will differ
a lot, and you should say which you are quoting.

---

## 7. Settings that matter, and when to change them

Open **Advanced** in the analysis screen. The defaults are the ones the supplied
data was labelled with; every control states its meaning in the units you think
in.

### Calibration — *check this every time*

**Pixel size** and **frame interval**. Read from the file when present. If the
interface says the source is anything other than your file's own metadata,
verify them against your acquisition software. Everything downstream scales
linearly with these.

### Image normalisation — *the setting most worth knowing about*

How the image is rescaled before the model sees it. This matters far more than
it sounds, and the reason is measured.

**The model is not contrast-invariant.** Rescaling only a cell's brightness —
same position, same shape, same noise, on images the model was *trained on* —
gives recall 0.83 at its native contrast, 0.10 at a quarter of it, and 0.51 at
two and a half times it. It peaks at the contrast it was trained on and falls
away in **both** directions.

So if your cells are dimmer or brighter than the training data, the model may
simply not respond, and no threshold can fix that — the probability field goes
flat rather than low.

| Setting | What it does | Use it when |
|---|---|---|
| **Whole frame (default)** | Cellpose's own percentiles (1, 99) | Your images resemble the supplied data |
| **Stretch contrast** | Narrower window (3, 97) | **Cells are faint, or detection is poor on new data** |
| Local contrast | Rescales each tile to its own background | Uneven illumination across the field |
| Sharpen edges | Unsharp mask first | Slightly out-of-focus acquisitions |

On a genuinely different acquisition day, **Stretch contrast raised F1 from
0.482 to 0.653** — no retraining, just the right window. It is the first thing
to try when a new dataset detects poorly. It costs a little accuracy on data
that was already well-matched, which is why it is not the default.

### Detection effort

How hard to look for cells the default settings miss. Extra passes can only
*add* detections, and anything only they found is marked in `detections.csv`.

**Leave this on "Single pass" unless you have a reason.** Measured end to end on
data with known answers, **no higher setting improved the result**: two settings
split a correct trajectory in half, and three put cells in frames that contain
none. The reason is that a false detection from channel-wall texture appears in
the *same place* every frame, and a stationary object is the most
self-consistent thing a tracker can be shown — so it links happily.

The rungs that run several models show "no other model found" on a standard
install, because only the combined model ships. That label is honest: choosing
them there would change nothing.

### Segmentation thresholds

- **Cell probability** (default 0.0) — lower finds dimmer objects. Below −3 it
  produces *fewer* detections, not more, because the extra candidates fail the
  flow check.
- **Flow threshold** (default 0.4) — higher accepts less consistent shapes.
  Above 0.4 this model starts finding objects in empty channels.
- **Smallest object** (default 20 px) — objects whose longer side is below this
  are discarded. Everything removed is counted in the diagnostics.

### Tracking

- **Allowed disappearance** (`max_gap`) — how many consecutive frames a cell may
  be missing and still be the same cell.
- **Speed limit** (default 5.0 µm/min) — a hard physical gate.
- **Along/across-channel spread** — how far a cell may deviate from its
  predicted position. Across-channel tolerance is tied to the cell's own width,
  so a wide cell is allowed more lateral room than a narrow one.
- **Unmatched cost** (default 15) — how poor a match must be before the tracker
  prefers to leave a cell unmatched. Raising it forces more links; lowering it
  makes the tracker more sceptical.
- **Keep cells in their own channel** — leave on unless your device has no
  walls.

---

## 8. Quality-control warnings and what each one means

| Warning | What it means | What to do |
|---|---|---|
| `no_pixel_size` / `no_frame_interval` | **Critical.** No calibration, so results are in pixels and frames | Enter the values |
| `segmentation_gap` | A frame between detections found nothing | Check whether the cell is visible; consider Stretch contrast |
| `stationary_track` | A track ends less than a cell's width from where it began | Either a stuck cell or a fixed feature of the device tracked as one — the images distinguish them |
| `fallback_dependent_track` | Half or more of a track's positions came only from a permissive pass | Look at the overlay before using it |
| `merge_suspected` | A track's predicted position falls inside a larger object matched to another track | Two cells may have touched; no position was invented |
| `unlinked_start` | A track began mid-stack and was not joined to an earlier one | The cost and the refusing rule are in `unlinked_starts.csv` |
| `fast_track` | Peak speed near the limit | Confirm it is one cell and not two |
| `lateral_drift` | More movement across the channel than along it | Check the migration axis was found correctly |
| `inferred_channel` | A wall was too faint to see and was placed from its neighbours' spacing | If a cell looks like it changed channel, check here first |
| `gap_bridged` | Informational: a track was reacquired after missing frames | Nothing, unless the gap is long |

---

## 9. How accurate is it

Three different questions, three different answers. Conflating them is the
easiest way to over- or under-trust this software. Full detail in
`docs/ACCURACY.md`.

| What is being counted | Accuracy |
|---|---|
| **Finding each cell, in each frame**, on data like the training set | **84 %** (F1 0.839) |
| **Finding each cell**, on a genuinely different acquisition day | **48 %** default, **73 %** with contrast-trained weights and Stretch contrast |
| **Keeping the right cell on the right trajectory** | **No error** on any available ground truth |
| **The migration speed you publish** | **Exact at the median, within ~3 %** at realistic detection loss |

**Why speed can be ~100 % while detection is 84 %.** Speed is distance ÷ time.
Missing the cell in a few middle frames changes neither where it started, nor
where it finished, nor how long it took. The failure mode of a missed detection
is a *shorter* trajectory, not a wrong one.

**The catch.** That only holds while detection is good. On unfamiliar data with
48 % detection, more than half the frames are missing, which is past the 40 %
cliff — speeds there are not reliable until detection improves. Check frame
coverage first.

**A known limit of the yardstick.** The 71 reference images are themselves
incomplete: at least 11 % of the cells in one half carry no label at all,
verified without the model by finding cells the annotator drew at *t−1* and
*t+1* but not at *t*. A perfect detector scored against those labels could not
exceed F1 0.942. Any accuracy figure quoted against them — including all of the
above — is therefore a measure of *agreement with an imperfect reference*, not
of ground truth.

---

## 10. Running without the interface

```powershell
Corridor.exe "C:\data\movie.tif" -o "C:\results\movie" --headless
```

`Corridor.exe --help` lists every parameter. The same analysis runs, writing the
same files. Useful for batches:

```powershell
Get-ChildItem C:\data\*.tif | ForEach-Object {
  & Corridor.exe $_.FullName -o "C:\results\$($_.BaseName)" --headless
}
```

Without `-o` or `--headless`, the interface opens — so double-clicking a `.tif`
associated with Corridor shows you the file rather than silently analysing it.

---

## 11. When something goes wrong

**"This analysis needs Cellpose version 3"** — the bundled runtime was replaced.
Reinstall. The model is a Cellpose 3 model and will not behave the same under
version 4, so Corridor refuses rather than quietly changing the science.

**No cells found anywhere** — check the Checks tab for `no_pixel_size`, then try
**Stretch contrast**. If the probability field is genuinely flat, the cells are
outside the model's response band and no threshold will help.

**Cells found in empty channels** — the flow threshold is probably above 0.4,
or Detection effort is above "Single pass".

**One cell became several tracks** — look at `unlinked_starts.csv` for the cost
and the rule that refused the join. If the cost is well below the gate and only
the gap rule refused it, raise **Allowed disappearance**.

**Two cells became one track** — check for `merge_suspected`, and confirm the
channel boundaries were measured rather than inferred.

**It is slow** — this runs on the CPU. Roughly 3–10 seconds per frame depending
on field size. A CUDA GPU is used automatically if present. Detection effort
above "Single pass" multiplies the time by the stated factor.

**Results look different after an update** — `run.json` records every parameter
and the model checksum for each analysis. Compare the two files.

---

## Reproducing any number in this guide

Every figure here comes from a script in the source archive:

```bash
python scripts/evaluate_models.py                 # held-out model performance
python scripts/experiment_recall.py               # eight detection strategies
python scripts/experiment_contrast_response.py    # the contrast response curve
python scripts/experiment_velocity_robustness.py  # speed under detection loss
python scripts/experiment_recovery.py             # gap recovery, scored
python scripts/experiment_fallback.py             # the detection ladder, end to end
python scripts/measure_label_completeness.py      # how incomplete the labels are
```

Results are written to `docs/*.json` beside the documents that quote them.
