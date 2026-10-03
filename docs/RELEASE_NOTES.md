Confined cell migration analysis for Windows. Drop in a phase-contrast
time-lapse, get cell trajectories and migration velocities you can check.

## Install

Download **Corridor-2.0.0-Setup.exe** below and run it. It installs for the
current user, so no administrator is needed. Python, PyTorch, Cellpose and the
validated segmentation model are all included — there is nothing else to
install. (Napari is now an optional developer extra, not bundled by default.)

## New in 2.0.0

2.0 is the first release built from a committed, reproducible source tree, and
it is a deep rewrite. The output format changes (schema v2), which is why the
major version moves; analyses saved by 1.x still open.

- **One validated model, locked by its checksum.** The app resolves exactly one
  segmentation model — the lab's — and verifies its SHA-256 before loading. If
  the file is missing or altered it stops with a clear message; it never
  silently falls back to a generic model. The model picker and the
  "change model" controls are gone. Segmentation is byte-for-byte identical to
  1.3.0 on every sample movie, so no measured number regressed.
- **Tracking no longer assumes a migration direction.** The old along/across
  model is replaced by a per-cell Kalman filter whose uncertainty is shaped by
  each cell's own body, a global gap-closing pass that rejoins tracks across
  missed frames, and a printed "link margin" for every link. A device's
  channel walls are still respected, as lanes, without a global axis.
- **Per-track measurements you can export.** Selected-track and all-tracks
  export to CSV or Excel, with MTrackJ-equivalent columns (cumulative path,
  distance from start, distance from previous, distance from a reference
  point), speeds in µm/hr, and mean-squared-displacement curves in µm² that
  handle gaps by true elapsed time. MSD is a curve, reported with the number of
  pairs behind each lag; an α is fitted only when there are enough lags.
- **True 3D+t import.** A Z stack is never mistaken for a time-lapse again; the
  importer reads T and Z from metadata and asks when they are ambiguous. 3D
  measurement, morphometry (volume, surface area) and tracking work on imported
  label images. 3D *segmentation* is refused until a 3D-validated model exists,
  rather than guessed.
- **Quality control rewritten** around the new tracker: ambiguous links,
  morphology and size jumps, border entry/exit, likely-missed-detection frames,
  and a critical flag if a developer override or an unvalidated segmentation was
  ever used.
- **Honest accuracy.** The held-out detection F1 is 0.7273 (unchanged — same
  model). The research behind that was corrected: the published 0.942/0.968
  "label ceilings" were withdrawn (they assumed filename order was time order;
  it is not), the contrast-augmentation gain was confirmed against a matched
  control, and Cellpose-SAM (`cpsam_v2`) was measured zero-shot and fails on
  this data — it outlines the microfluidic channels, not the cells. See
  `docs/RESEARCH_V2.md`.

After installing, `Corridor.exe --self-test` checks that the installation is
complete: it loads the model, verifies its checksum, runs the full
torch/Cellpose path, tracks a known trajectory and checks the velocity
arithmetic, round-trips every output format, and exercises Napari.

## New in 1.3.0

- **Corridor can now tell you when a newer version exists — if you let it.**
  Until this release the application made **no network calls at all**, which
  meant an improvement could ship and nobody running it would ever find out.
  That is now fixed, and deliberately not by the usual route.

  It **asks once**, on first run, before anything has touched the network. The
  default is offline: a microscope workstation holding unpublished data is
  frequently offline on purpose, and no answer is treated as no. Say yes and it
  asks GitHub, once per session, whether a newer release exists — sending
  nothing about you, your images or your results.

  It **checks; it does not install**. A program that downloads an executable and
  runs it is precisely the shape of the thing every security guide warns about,
  and "but it is our own executable" is what somebody who had compromised the
  release channel would be relying on. Corridor shows you what exists and hands
  over a link. You download it, Windows checks the signature, and a person
  decides. There is a test that fails if `subprocess`, `ShellExecute` or
  `urlretrieve` ever appear in the updater.

  Everything else about it is quiet: the check runs on a worker thread so a
  captive-portal wifi cannot freeze the window; every failure — no network, a
  proxy, a rate limit, a malformed reply — means "no information" rather than an
  error in front of somebody mid-experiment; and dismissing one version does not
  silence the next, so "not now" never becomes "never". The choice can be
  changed at any time under *Settings → Updates*.

- **One correct instance matcher instead of four.** Scores were computed by
  solving the assignment on IoU and *then* applying the 0.5 threshold, which
  maximises total overlap rather than the number of objects found — so a
  high-overlap pair could capture a prediction a second cell needed. Measured on
  this data the two rules agree on every image, so **no published figure
  changes**; it is fixed anyway, in one place with the counterexample pinned as
  a test, because a metric that is right by luck on today's data is not right.

## New in 1.2.0

- **A measured answer to "how accurate is it?"** — `docs/ACCURACY.md` separates
  the three things that question can mean and gives the number for each: cell
  outlines per frame (F1 0.839, and 0.30-0.48 held out), trajectory identity
  (no error on any available ground truth), and the migration figures that
  actually get published (net speed exact at the median, within 2.7-2.9% at the
  90th percentile, at the detection loss this model really has).
- **Robust speed estimators.** `track_summary.csv` now carries
  `net_speed_um_per_min`, `along_speed_um_per_min` and `path_speed_um_per_min`
  beside the mean and median of instantaneous speeds. The net figures were
  measured to be two to three times less sensitive to a missed detection,
  because they read the endpoints rather than averaging every interval.
- **A detection fallback ladder, off by default.** Five settings under
  *Detection effort*, from a single pass to every model at two thresholds.
  Extra passes can only add detections, and anything only they found is marked
  as such in `detections.csv`. **It is off by default because it was measured
  end to end and no rung improved the result on this data** - two of them split
  a correct trajectory in half, and three put cells in frames that contain
  none. The measurement is in `docs/ACCURACY.md`; the mechanism is that a false
  detection from channel-wall texture sits in the *same place* every frame, and
  a stationary object is the most self-consistent thing a tracker can be shown.
- **Two new quality-control findings** that catch exactly that failure:
  `stationary_track` when a trajectory ends less than a cell's width from where
  it began, after enough elapsed time for that to mean something; and
  `fallback_dependent_track` when half or more of its positions came only from
  a permissive pass. The first measures **net** displacement rather than path
  length, because a fixed object's centroid wanders by a fraction of a pixel
  each frame and a path-length test would let it accumulate its way out of the
  very check meant to catch it. The time bar is in minutes rather than frames,
  because four frames is an hour of the supplied data and two minutes of a fast
  acquisition, and a cell that has not moved 4 µm in two minutes is just a cell.
- **Track-guided recovery now runs, for interior gaps only** — and the story of
  why is the most important correction in this release. Version 1.1 reported
  that recovery "filled 1 of 13 holes" and shipped it disabled on that basis.
  That measurement was taken while a defect skipped **every** interior gap in
  silence: recovery asked the tracker to predict a position, and the tracker's
  prediction refuses to look backwards, which is correct during tracking and
  wrong afterwards. So it examined nothing but the frames past the end of each
  track, and no output file said so.

  Interior gaps now interpolate between the observations on either side. Scored
  against the same ground truth, recovery fills **10 of 13** holes with a median
  error of 3.5 px — about a quarter of a cell's width — and **no false
  positives**. Frames *past* a track's last observation are a different claim
  (the cell may simply have left) and produced every false position in the
  measurement, so they stay off. Everything recovered carries its tier and
  confidence in `detections.csv`, is counted in `run.json`, and trips
  `fallback_dependent_track` if a trajectory comes to lean on it.
- **An *Image normalisation* control, and the reason it exists.** The two halves
  of the supplied training data hold cells of the same size — 12.1 against
  12.2 px wide — at 1.7× different contrast: a cell stands 0.219 of the image
  range above its background in one half and 0.378 in the other. That, and not
  shape or scale, is most of what a model trained on one half meets in the
  other, and it explains why the model trained on the *high*-contrast half is
  the one that loses the most. Contrast can be changed before the image reaches
  the network, so the setting is offered and each option's measured effect is
  recorded in `docs/ACCURACY.md`. The default is unchanged, so existing results
  stay comparable.
- **`docs/THIRD_PARTY.md`, generated rather than hand-kept.**
  `scripts/audit_licences.py` reads the built application's own package
  metadata and fails the release if a copyleft dependency appears that the
  document does not account for. Writing it found three that a hand-written
  list had missed. Neither the trained model nor the research images are in the
  source archive — `scripts/make_source_zip.py` enforces that with an
  allow-list and re-checks the finished archive before it can be published.

## New in 1.1.0

- **Napari is bundled.** No longer an optional extra to install separately.
- **`corridor --self-test`** verifies an installation end to end.
- **`unlinked_starts.csv`.** When a cell disappears and something appears
  later, the tracker's refusal to join them is recorded with the distance, the
  gap, the implied speed, what the match would have cost and which rule refused
  it — so the judgement can be disagreed with on the numbers.
- **Inferred channel boundaries are marked as inferred.** Where a channel wall
  was too faint to see and its position was filled in from the spacing of the
  others, that is now flagged rather than presented as an observation.
- **A measured answer to "how good is the segmentation?"** — see
  `docs/MODEL_EVALUATION.md` in the repository.

## How well does the segmentation work

Measured with a properly constructed held-out split, because the combined model
was trained on all 71 labelled images and has no held-out data of its own.
**`docs/ACCURACY.md` is the full treatment**; this is the summary.

| Model | Evaluated on | Kind | F1 | Recall |
|---|---|---|---:|---:|
| combined | its own training images | fit | 0.80–0.87 | 0.80–0.88 |
| KK1-only | KK2 | **held out** | **0.48** | 0.44 |
| KK2-only | KK1 | **held out** | **0.30** | 0.20 |

Generalisation is roughly half of fit, and the failure mode is **missing
cells, not inventing them** (precision holds at 0.54-0.60). Trajectories
fragment rather than go wrong — the safer failure, but it still biases anything
computed over track lengths. Plan for it.

What that actually costs the answer was then measured rather than assumed.
Deleting detections at random from a stack whose complete answer is known, the
net migration speed stays **exactly right at the median, and within 2.7 % at
the 90th percentile at 15 % loss (4.6 % at 20 %)**, with no track truncated. The mean
of instantaneous speeds drifts two to three times faster, because every missed
frame merges two short intervals into one long one. Above about 30 % loss the
endpoints themselves begin to disappear and the error jumps sharply — that is
the point at which to check frame coverage before quoting a figure.

## What it does

Cellpose v3 segmentation → confinement-aware tracking → velocities in µm/min,
with quality-control review over the image and CSV export. Analyses are kept
locally and reopen without recomputing.

## Corrections to the original analysis

This software fixes defects that changed the numbers:

- **Frame interval.** The research code assumed 10 minutes per frame. The
  sample data records **20.0069 min/frame**, so reported velocities were about
  **97 % too high**. Timing is now read from the file's embedded acquisition
  metadata, and the interface shows where the value came from.
- **Pixel size.** 0.46 µm/px was assumed; the files record 0.467060343 µm/px.
- **Axis order.** The loader claimed to infer the TIFF axis order but returned
  the array unchanged. Axes are now read, and a colour axis is never taken for
  time.
- **Frame count.** A crop exported from a longer acquisition carries the
  parent's `SizeT`; the number of frames actually present is used instead.
- **Migration direction.** The confinement axis was taken from each cell's own
  shape, which is redefined every frame and meaningless for a round cell. It is
  now measured once from the device's channel walls, including their ≈2.4°
  tilt, and cells are never tracked across a wall.
- **Assignment.** The unmatched cost was declared but never used; links were
  forced and then discarded. The assignment problem now contains real dummy
  blocks, so "no match" is a decision the optimiser weighs.
- **Gaps.** Prediction, velocity and gating now all scale with elapsed frames.
- **Save order.** Results were written after the viewer opened, which blocks;
  they are now saved first.

## Units

Velocity columns state their units: `speed_um_per_min`, `v_along_um_per_min`,
`v_across_um_per_min`, alongside `speed_px_per_frame`.

## Requires

Windows 10 or 11, 64-bit. About 1 GB of disk. Runs on the CPU; a CUDA GPU is
used if one is present.

## Verifying the download

The SHA-256 is published beside the installer in
`Corridor-2.0.0-Setup.exe.sha256`:

```powershell
Get-FileHash Corridor-2.0.0-Setup.exe -Algorithm SHA256
```

The installer is also digitally signed, so any modification after build breaks
the signature. The certificate is self-signed, which means it proves the file
has not been altered since it was built but does **not** stop Windows
SmartScreen warning you on first run — that needs a commercially issued
certificate tied to a verified legal identity. Check the SHA-256 above; do not
rely on the absence of a warning.
