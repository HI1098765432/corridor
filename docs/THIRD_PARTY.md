# Bundled components and what their licences require

The installer is a self-contained application: it carries a Python runtime,
PyTorch, Cellpose, Qt and Napari, so that a researcher does not have to
assemble an environment. Bundling other people's software is a distribution
decision, not just a packaging one, so this is the record of what is included
and on what terms.

## The five that impose conditions

Everything here is permissively licensed **except** five components, under the
**LGPL-3** and the **MPL-2.0**. This list is not hand-maintained — it is what
`scripts/audit_licences.py` reports from the built application, and writing that
script immediately found three entries a hand-written list had missed.

| Component | Version | Licence | Why it is present |
|---|---|---|---|
| PySide6 / shiboken6 | 6.8.1 | **LGPL-3** | The Qt binding the interface is written against |
| fastremap | 1.20.0 | **LGPL-3** | A Cellpose dependency; relabels mask arrays |
| fill_voids | 2.1.2 | **LGPL-3-or-later** | A Cellpose dependency; fills holes in masks |
| certifi | 2026.7.22 | **MPL-2.0** | The CA bundle, reached through Napari's HTTP stack |
| tqdm | 4.70.1 | **MPL-2.0 AND MIT** | Cellpose's progress bars |

The two MPL components are the easier case: MPL-2.0 is *file-level* copyleft,
so it binds only modified copies of its own files. Neither is modified; both
ship exactly as published.

The LGPL is the one that constrains packaging. It allows this application to be
distributed as a combined work provided the recipient can **replace the LGPL
parts with their own build**. Two deliberate choices in
`packaging/corridor.spec` are what make that true, and neither is incidental:

* **`COLLECT`, not a one-file executable.** The application is built as a
  folder, so `Corridor/_internal/` contains Qt's DLLs and the `fastremap`
  extension as ordinary, discrete files. Anyone may substitute their own.
* **`upx=False`.** Compressing those binaries would leave them replaceable only
  in principle. They are shipped exactly as their projects built them.

A one-file build would be smaller and would make this argument harder to
sustain. The size is the price of the licence being satisfied plainly rather
than arguably.

None of these libraries is modified. The corresponding sources are available
from `pypi.org/project/<name>/` at the versions pinned in
`packaging/requirements-locked.txt`.

### What is deliberately *not* shipped

PyInstaller itself is **GPLv2**, which would be a serious problem if any of it
ended up inside the application. It does not: it is a build tool, and only its
bootloader is linked into the executable, under the explicit exception its
authors grant for exactly this. The audit script does not take that on trust --
given a built application it reads the `*.dist-info` directories in
`Corridor/_internal/` and reports what is actually there, rather than guessing
from the build environment.

## The permissive remainder

| Component | Version | Licence |
|---|---|---|
| PyTorch | 2.6.0+cpu | BSD-3-Clause |
| Cellpose | 3.1.1.3 | BSD-3-Clause |
| NumPy | 1.26.4 | BSD-3-Clause |
| SciPy | 1.17.1 | BSD-3-Clause |
| scikit-image | 0.25.2 | BSD-3-Clause |
| tifffile | 2026.3.3 | BSD-3-Clause |
| imagecodecs | 2026.1.14 | BSD-3-Clause |
| roifile | 2026.2.10 | BSD-3-Clause |
| Napari | 0.9.1 | BSD-3-Clause |
| VisPy | 0.16.2 | BSD-3-Clause |
| superqt | 0.8.2 | BSD-3-Clause |
| Numba / llvmlite | 0.67.0 / 0.49.0 | BSD-2-Clause |
| OpenCV (headless) | 4.11.0.86 | Apache-2.0 |
| Pillow | 12.3.0 | MIT-CMU |
| pydantic | 2.13.5 | MIT |
| qtpy | 2.4.3 | MIT |
| natsort | 8.4.0 | MIT |

These require attribution and the retention of their notices, which the
installed `_internal` tree carries in each package's `*.dist-info` directory.

## The trained model

`cyto2_phase_microfluidic_KK1KK2_combi`
(SHA-256 `b33bdbda…970177cd6`) is **not** third-party software. It was trained
by the researchers who supplied this project, from their own images, and it is
bundled because the application is useless without it.

Two consequences follow, and both are implemented rather than merely stated:

* **It is not in the source archive.** `Corridor-<version>-source.zip` contains
  code only. Redistributing a trained model is a decision for the people who
  trained it, so the archive does not make it for them.
* **Neither are the images.** The 71 hand-labelled training images and the
  sample time-lapses stay out of every published artefact. They are research
  data, and publishing them is not this software's call.

The installer does carry the model, because that is the whole point of an
installer, and its checksum is verified at every start by
`Corridor.exe --self-test`.

## Cellpose version

The model is a **Cellpose 3** model. Corridor pins `cellpose>=3,<4` and refuses
to start under Cellpose 4 rather than loading the model under a runtime it was
not trained against. A version check that can be silently satisfied by an
upgrade is not a check; see `require_cellpose_v3()` in
`src/corridor/core/segmentation.py`.

## Reproducing this list

```bash
python scripts/audit_licences.py
```

It reads the installed metadata rather than this file, so a dependency that
changes its licence, or a new one that arrives as somebody's transitive
requirement, shows up as a difference instead of going unnoticed.
