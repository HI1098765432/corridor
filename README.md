# Corridor

**Confined cell migration analysis.** Open a phase-contrast time-lapse, get
trustworthy cell trajectories and migration velocities.

Corridor segments cells with a custom Cellpose v3 model, links them between
frames with a confinement-aware assignment model, and reports velocities in
explicit physical units — with the provenance of every number attached.

---

## Documentation

**[`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)** is the manual: install, what each
stage does to your images, every output column, which speed number to publish,
what each quality-control warning means, and what to do when something looks
wrong. Read that first.

[`docs/ACCURACY.md`](docs/ACCURACY.md) is the measured accuracy, at three levels.
[`docs/TOWARDS_99.md`](docs/TOWARDS_99.md) is the ongoing research log, failures
included.

## Download

**[Download the Windows installer](https://github.com/HI1098765432/corridor/releases/latest)**

No Python, no Cellpose install, no command line. The installer bundles the
runtime and the trained model.

---

## What it does

```
time-lapse TIFF
  → metadata interpretation      pixel size and frame interval read from the file
  → Cellpose v3 segmentation     custom microfluidic model
  → segmentation diagnostics     raw vs kept instance counts, per frame
  → detections                   centroid, area, shape
  → confinement-aware tracking   linear assignment with real unmatched costs
  → trajectories and velocities  µm/min, along and across the channel
  → quality-control review       every questionable frame is one click away
  → CSV export                   stable schemas, units in the column names
```

## Why the numbers can be trusted

- **Calibration is read, never assumed.** Pixel size and frame interval come
  from the file's embedded acquisition metadata, and the interface says where
  each value came from. A dataset with no calibration reports pixel units
  rather than inventing micrometres.
- **The frame count is the frames present.** A crop exported from a longer
  acquisition carries the parent's `SizeT`; Corridor uses the actual array.
- **Original frame numbers survive.** `t:42/54` in the source becomes a
  `source_frame` column in the output.
- **The migration axis is measured from the device**, not from cell shape. The
  channel walls are traced from a temporal median projection, giving the axis,
  its tilt, and the channel boundaries that associations may not cross.
- **Every cost term is in chi-square units**, so each threshold reads as a
  number of standard deviations rather than an arbitrary weight.
- **Gaps are gap-aware.** Prediction, velocity update and gating all scale with
  the number of elapsed frames.
- **Ambiguity is preserved.** When one mask covers two cells, no centroid is
  invented: the second identity goes dormant and the frame is flagged.
- **Judgements show their evidence.** When a cell disappears and something
  appears later, the refusal to join them is recorded with the distance, the
  gap, the implied speed and what the match would have cost — so a reviewer can
  disagree with it on the numbers rather than taking it on faith.
- **Results are saved before any viewer opens.**

## Output files

| File | Contents |
| --- | --- |
| `tracks.csv` | one row per observation: position, area, gap, match cost, velocity |
| `track_summary.csv` | one row per track: duration, path length, straightness, and six speed estimators |
| `detections.csv` | every segmented object, independent of tracking |
| `segmentation_diagnostics.csv` | raw vs kept instance counts per frame |
| `tracking_events.csv` | matches, new tracks, dormancies, suspected merges per frame |
| `qc_issues.csv` | everything worth a second look |
| `unlinked_starts.csv` | for each track beginning mid-stack, why it was not joined to an earlier one |
| `run.json` | full provenance: versions, model checksum, every parameter |
| `masks.npz` | the label stack |

Velocity columns are named for their units: `speed_um_per_min`,
`v_along_um_per_min`, `v_across_um_per_min`, plus `speed_px_per_frame`.

`track_summary.csv` reports the speed several ways rather than choosing for
you, because they do not degrade alike when detections are missed:
`net_speed_um_per_min` and `along_speed_um_per_min` read only the endpoints and
are the robust ones; `mean_speed_um_per_min` and `median_speed_um_per_min`
average the intervals and drift two to three times faster;
`path_speed_um_per_min` sits between. The measurement is in
[`docs/ACCURACY.md`](docs/ACCURACY.md).

## Running without the interface

```bash
corridor path/to/stack.tif -o results/
corridor stack.tif --max-gap 4 --flow 0.5 --axis vertical
corridor stack.tif --pixel-size 0.4671 --frame-interval 20.0069
```

`corridor --help` lists every parameter. The same analysis runs headless.

## How accurate is it?

**[`docs/ACCURACY.md`](docs/ACCURACY.md) is the full answer**, with every number
reproducible from a script in `scripts/`. It exists because the question has
three different answers and quoting the wrong one is misleading in both
directions:

| What is counted | Measured |
| --- | --- |
| Each cell outline, per frame | F1 **0.839** in-distribution, **0.30–0.48** held out |
| Cell identity along a trajectory | **no error** on any available ground truth |
| Net migration speed — the published quantity | **exact at the median**, within **2.7–2.9 %** at p90 |

The detector is the weak link and it cannot be argued up: measuring eight
detection strategies, recall can be bought to 0.902 only by paying precision
down to 0.703, and **F1 never exceeds 0.843 at any price**. Raising that needs
more labelled data and retraining, not parameters.

What saves the result is that a missed detection costs far less than it looks.
Displacement is divided by elapsed time rather than by one frame, so the net
speed of a track survives losing a fifth of its detections almost unchanged.
The failure mode is a *shorter* trajectory, not a wrong one.

The held-out collapse was also diagnosed rather than just reported: the two
halves of this dataset differ in **contrast**, not in cell size (12.1 vs 12.2 px
wide, but 0.219 vs 0.378 of the image range above background), which is why the
model trained on the high-contrast half loses the most. See
[`docs/MODEL_EVALUATION.md`](docs/MODEL_EVALUATION.md) for the per-model split.

## Requirements

Cellpose **v3** (`cellpose>=3,<4`). The supplied custom model is a Cellpose 3
model; Cellpose 4 refuses to load it, and Corridor refuses to run under it
rather than silently changing the science.

## Development

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest          # unit and regression tests
.venv/Scripts/python scripts/gui_e2e.py # end-to-end through the real UI
```

Build the Windows installer:

```bash
python scripts/build_release.py
```

## Layout

```
src/corridor/
  core/          imaging, segmentation, detections, confinement,
                 tracking, measurements, qc, pipeline, export, config
  store/         SQLite project index and on-disk analyses
  ui/            Qt application: screens, widgets, workers, theme
  viz/           optional Napari inspection
tests/           unit, regression and real-data tests
packaging/       PyInstaller spec and Inno Setup script
```

## Licence

MIT for the application. The bundled Cellpose model belongs to the researchers
who trained it; see `LICENSE`.

The installer carries a whole Python runtime, so what it bundles and on what
terms is written down in
**[`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md)** rather than left implicit. Five
components are copyleft — PySide6/shiboken6, fastremap and fill_voids under the
LGPL-3, certifi and tqdm under the MPL-2.0 — and the one-folder, uncompressed
build in `packaging/corridor.spec` is what satisfies the LGPL's requirement that
those parts remain replaceable. That list is generated, not hand-kept:

```bash
python scripts/audit_licences.py
```

It reads the built application's own package metadata and exits non-zero if any
copyleft dependency appears that the document does not account for. Writing it
found three that a hand-written list had missed.

Neither the trained model nor the research images appear in the source archive.
Publishing those is the researchers' decision, not this software's.
