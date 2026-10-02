# Corridor 1.2.0 — delivery note

**Confined cell migration analysis for Windows**
14 September 2026

---

## Summary

Corridor takes a phase-contrast time-lapse of cells in microfluidic channels and
returns each cell's trajectory and migration speed, with the provenance of every
number attached. It replaces a research notebook whose defects changed the
results — most seriously a frame interval of 10 minutes where the files record
20.0069, which overstated every velocity by **97 %**.

The application is installable, signed, and self-contained: Python, PyTorch,
Cellpose v3, Napari and the trained model are all inside. Nothing else to
install and no internet connection needed after download.

---

## Downloads

| Item | Link |
|---|---|
| **Windows installer** (265 MB) | https://github.com/HI1098765432/corridor/releases/download/v1.2.0/Corridor-1.2.0-Setup.exe |
| **Source code** (142 files, 391 KB) | https://github.com/HI1098765432/corridor/releases/download/v1.2.0/Corridor-1.2.0-source.zip |
| **Release page** | https://github.com/HI1098765432/corridor/releases/tag/v1.2.0 |
| **Repository** | https://github.com/HI1098765432/corridor |

### Verifying the installer

```powershell
Get-FileHash Corridor-1.2.0-Setup.exe -Algorithm SHA256
```

| Property | Value |
|---|---|
| SHA-256 | `99bb4a4bdf552c2cde4932fdd8692e4113a74ce99c1a8d57d784e8757e29720f` |
| Size | 264,663,456 bytes |
| Signed by | `CN=Corridor, O=Corridor, C=GB` |
| Certificate thumbprint | `D0F3C88E3FA14C6D4DA7A0C6F508DBC931674826` |
| Source archive SHA-256 | `a4d1b3f58e8536aaed1ece2c3a947c0c4ec80c316172532cc3f40cc1feb1e63c` |

Both files were downloaded back over plain HTTPS after publication and compared
byte for byte with what was built.

The certificate is **self-signed**, so Windows SmartScreen will warn on first run
(*More info* → *Run anyway*). Removing that warning requires a commercial
certificate tied to a verified legal identity; nothing in the build can
substitute for one, and claiming otherwise would be worse than the warning.

### Installing

The installer is per-user and needs no administrator password. After installing,
verify the installation end to end:

```powershell
& "$env:LOCALAPPDATA\Corridor\Corridor.exe" --self-test
```

Nine checks must report `[ok]`: scientific stack, interface toolkit, bundled
model checksum, full PyTorch/Cellpose pass, tracking and velocity arithmetic,
output round-trips, project store, processing device, Napari.

---

## Documentation

| Document | What it covers |
|---|---|
| **`docs/USER_GUIDE.md`** | **The manual.** Install, every processing stage, every output column, which speed to publish, all ten QC warnings, troubleshooting |
| `docs/ACCURACY.md` | Measured accuracy at three levels, and what each is worth |
| `docs/TOWARDS_99.md` | Research log for the ongoing accuracy work, failures included |
| `docs/MODEL_EVALUATION.md` | Held-out performance of the three supplied models |
| `docs/THIRD_PARTY.md` | Every bundled component and its licence |
| `docs/RELEASE_NOTES.md` | What changed in this version |

All are inside the source archive and in the repository.

---

## Accuracy, stated precisely

The question has three answers because three different things are being
counted. Quoting the wrong one misleads in both directions.

| What is counted | Measured |
|---|---|
| **Each cell, in each frame** — data resembling the training set | **84 %** (F1 0.839) |
| **Each cell, in each frame** — a different acquisition day | **48 %** as shipped; **73 %** with contrast-trained weights and *Stretch contrast* |
| **Cell identity along a trajectory** | **No error** on any available ground truth (16 tracks, 113 positions, 6 channels, zero cross-channel links) |
| **Migration speed** — the published quantity | **Exact at the median, within ~3 %** at realistic detection loss |

**Why speed accuracy exceeds detection accuracy.** Speed is distance ÷ time.
Missing a cell in intermediate frames changes neither where it started, nor
where it finished, nor how long it took. The failure mode of a missed detection
is a *shorter* trajectory, not a wrong one. Measured: deleting a fifth of the
detections at random leaves net speed exactly right in half of trials and within
4.6 % at the 90th percentile.

**The condition on that.** It holds while detection is good. Below roughly 60 %
frame coverage the endpoints themselves start disappearing and the error jumps
sharply. `segmentation_diagnostics.csv` reports coverage per frame and should be
checked before quoting a speed.

**A limit of the reference, not of the software.** The 71 hand-labelled
reference images are themselves incomplete. Verified without using the model —
by finding cells the annotator drew at *t−1* and *t+1* but not at *t*, in images
that are frames of a movie — **at least 10.9 % of the cells in one half carry no
label**, and 6.1 % in the other. Because a correct detection of an unlabelled
cell is scored as a false positive, **a perfect detector could not exceed F1
0.942** against these labels. Every accuracy figure above is therefore agreement
with an imperfect reference, not ground truth.

---

## Defects corrected from the original analysis

| Defect | Effect |
|---|---|
| Frame interval assumed to be 10 min | Velocities overstated by **97 %** |
| Pixel size assumed to be 0.46 µm | Wrong by 1.5 % |
| TIFF axis order claimed to be inferred, but was not | A colour axis could be read as time |
| Frame count taken from the parent acquisition's `SizeT` | Wrong number of frames for any crop |
| Migration direction taken from each cell's own shape | Meaningless for a round cell, redefined every frame |
| Unmatched cost declared but never used | Links forced, then discarded |
| Prediction, velocity and gating not gap-aware | Wrong after any missed frame |
| Results written after the viewer opened | Blocked until the window was closed |

---

## Ongoing work

Detection on unfamiliar data is the weak link and the active research. The
measured root cause is that the model is **not contrast-invariant**: rescaling
only a cell's brightness — same position, shape and noise, on images the model
was trained on — gives recall 0.83 at its native contrast, 0.10 at a quarter of
it and 0.51 at two and a half times it. It peaks at the contrast it was trained
on and falls away in both directions.

Two interventions follow from that and both are in the product:

- **Input side** — *Image normalisation → Stretch contrast* moves an image onto
  the band the model can respond in. Held-out F1 0.482 → 0.653, no retraining.
- **Weight side** — contrast-randomised fine-tuning widens the band itself.
  Combined with the above, **F1 0.726**, recall 0.435 → 0.696.

Target is the **0.942 reference ceiling**. The remaining 64 errors have been
classified individually: 21 contrast-related, 17 outline or split/merge, 13
label-quality, 13 other. Work continues against that table, and the installer
will be rebuilt when a model beats the current one.

---

## Licensing and data handling

MIT for the application. The bundled Cellpose model belongs to the researchers
who trained it.

Five bundled components are copyleft — PySide6/shiboken6, fastremap and
fill_voids under LGPL-3, certifi and tqdm under MPL-2.0. The one-folder,
uncompressed build in `packaging/corridor.spec` is what satisfies the LGPL's
requirement that those parts remain replaceable. The list is generated, not
hand-maintained: `scripts/audit_licences.py` reads the built application's own
package metadata and fails the release if an undocumented copyleft dependency
appears.

**No research data or trained model is in the source archive.**
`scripts/make_source_zip.py` enforces that with an allow-list and re-verifies
the finished archive before it can be published. Publishing the images or the
model is the researchers' decision, not the software's.

---

## Requirements

Windows 10 or 11, 64-bit. About 1 GB of disk. Runs on the CPU; a CUDA GPU is
used automatically if present. Roughly 3–10 seconds per frame depending on field
size.

Cellpose **v3** is pinned (`cellpose>=3,<4`). The supplied model is a Cellpose 3
model; Corridor refuses to start under Cellpose 4 rather than loading it under a
runtime it was not trained against.
