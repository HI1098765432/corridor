Confined cell migration analysis for Windows. Drop in a phase-contrast
time-lapse, get cell trajectories and migration velocities you can check.

## Install

Download **Corridor-1.0.0-Setup.exe** below and run it. It installs for the
current user, so no administrator is needed. Python, PyTorch, Cellpose and the
trained segmentation model are all included — there is nothing else to install.

## What it does

Cellpose v3 segmentation → confinement-aware tracking → velocities in µm/min,
with quality-control review over the image and CSV export. Analyses are kept
locally and reopen without recomputing.

## Corrections to the original analysis

This release fixes defects that changed the numbers:

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
- **Gaps.** Prediction, velocity and gating now all scale with elapsed frames,
  so a cell reacquired after an absence no longer acquires several times its
  real speed.
- **Save order.** Results were written after the viewer opened, which blocks;
  they are now saved first.

## Units

Velocity columns state their units: `speed_um_per_min`, `v_along_um_per_min`,
`v_across_um_per_min`, alongside `speed_px_per_frame`.

## Requires

Windows 10 or 11, 64-bit. About 900 MB of disk. Runs on the CPU; a CUDA GPU is
used if one is present.

## Verify the download

The SHA-256 of the installer is published beside it in
`Corridor-1.0.0-Setup.exe.sha256`.

```powershell
Get-FileHash Corridor-1.0.0-Setup.exe -Algorithm SHA256
```
