Confined cell migration analysis for Windows. Drop in a phase-contrast
time-lapse, get cell trajectories and migration velocities you can check.

## Install

Download **Corridor-1.1.0-Setup.exe** below and run it. It installs for the
current user, so no administrator is needed. Python, PyTorch, Cellpose, Napari
and the trained segmentation model are all included — there is nothing else to
install.

After installing, `Corridor.exe --self-test` checks that the installation is
complete: it loads the model, verifies its checksum, runs the full
torch/Cellpose path, tracks a known trajectory and checks the velocity
arithmetic, round-trips every output format, and exercises Napari.

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
was trained on all 71 labelled images and has no held-out data of its own:

| Model | Evaluated on | Kind | F1 | Recall |
|---|---|---|---:|---:|
| combined | its own training images | fit | 0.80–0.87 | 0.80–0.88 |
| KK1-only | KK2 | **held out** | **0.48** | 0.44 |
| KK2-only | KK1 | **held out** | **0.30** | 0.20 |

Generalisation is roughly half of fit, and the failure mode is **missing
cells, not inventing them** (precision holds at 0.54–0.60). Trajectories
fragment rather than go wrong — the safer failure, but it still biases anything
computed over track lengths. Plan for it.

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
`Corridor-1.1.0-Setup.exe.sha256`:

```powershell
Get-FileHash Corridor-1.1.0-Setup.exe -Algorithm SHA256
```

The installer is also digitally signed, so any modification after build breaks
the signature. The certificate is self-signed, which means it proves the file
has not been altered since it was built but does **not** stop Windows
SmartScreen warning you on first run — that needs a commercially issued
certificate tied to a verified legal identity. Check the SHA-256 above; do not
rely on the absence of a warning.
