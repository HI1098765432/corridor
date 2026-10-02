# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for Corridor.

*What* is bundled is decided in ``bundle_plan.py`` beside this file, where the
tests and the release workflow read the same lists without running
PyInstaller; this file applies the plan and refuses to build when its evidence
is missing or stale. Notes that matter for this particular bundle:

*   Cellpose's hidden imports are the modules its inference path was measured
    to load (``cellpose_runtime_modules.txt``, from
    ``scripts/measure_runtime_imports.py``), not ``collect_submodules``, which
    shipped the GUI, training and Dask trees. The build stops if the installed
    Cellpose is not the one that was measured.
*   Torch is collected by ``hooks/hook-torch.py``, scoped to CPU inference.
    Never call ``collect_submodules("torch")`` here: it *imports* every torch
    submodule, which doubles an already slow analysis and can stall it.
*   The production model is the one ``src/corridor/assets/model_registry.json``
    names, and its SHA-256 is checked here, so running PyInstaller directly no
    longer skips the hash gate. It ships inside the bundle so the application
    works the moment it is installed. Nothing under ``data/``, ``training/`` or
    ``build/`` is ever collected.
*   Napari is bundled only with ``CORRIDOR_BUNDLE_NAPARI=1``. Before 2.0 it was
    bundled whenever the build venv happened to have it, so CI and local builds
    of one version differed.
*   The Windows version resource must match ``src/corridor/_version.py``
    (``scripts/sync_version.py`` writes it).
*   ``console=False`` so launching the app never flashes a terminal.
"""

import importlib.metadata
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_entry_point,
    collect_submodules,
    copy_metadata,
)

SPEC_DIR = Path(SPECPATH).resolve()
ROOT = SPEC_DIR.parent
for helper_dir in (SPEC_DIR, ROOT / "scripts"):
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))

import bundle_plan as plan  # noqa: E402
import sync_version  # noqa: E402

SRC = plan.SRC
ASSETS = plan.ASSETS

# --------------------------------------------------------------------------
# Evidence the plan rests on
# --------------------------------------------------------------------------

try:
    stale = sync_version.out_of_sync(ROOT)
    if stale:
        raise plan.BundleError(
            "out of date with src/corridor/_version.py: "
            + ", ".join(p.as_posix() for p in stale)
            + "; run python scripts/sync_version.py"
        )
    BUNDLE_NAPARI = plan.napari_requested()
    measurement = plan.load_measurement()
    plan.check_measurement_current(measurement, importlib.metadata.version("cellpose"))
    model = plan.production_model()
    # Icons and the model registry, then the hash-checked weights.
    datas = plan.asset_datas() + plan.model_datas(model)
except (plan.BundleError, sync_version.VersionError) as exc:
    raise SystemExit(f"corridor.spec: {exc}")

# scikit-image resolves its public API through lazy-loader ``.pyi`` stubs, which
# are data files. Cellpose's own data files (GUI help pages and a logo) are not
# collected: only cellpose.gui reads them, and it is excluded.
datas += collect_data_files("skimage")

# --------------------------------------------------------------------------
# Napari (optional deep-inspection viewer), opt-in
# --------------------------------------------------------------------------
# Napari finds its own components through entry points and npe2 manifests
# rather than through imports, so PyInstaller cannot see them by static
# analysis. Its package metadata has to be copied for the discovery to work at
# all, and its YAML manifests are data files.
napari_hiddenimports = []
if BUNDLE_NAPARI:
    try:
        import napari  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"corridor.spec: {plan.NAPARI_ENV}=1 but napari cannot be imported "
            f"in this build environment ({exc})."
        )
    for package in (
        "napari", "napari_svg", "npe2", "vispy", "magicgui", "superqt",
        "app_model", "psygnal", "in_n_out", "pint",
    ):
        try:
            datas += collect_data_files(package, include_py_files=True)
        except Exception:  # noqa: BLE001
            pass
    for distribution in (
        "napari", "napari-svg", "npe2", "vispy", "magicgui", "superqt",
        "app-model", "psygnal", "in-n-out", "pydantic", "pint",
    ):
        try:
            datas += copy_metadata(distribution)
        except Exception:  # noqa: BLE001
            pass
    for package in ("napari", "npe2", "vispy", "magicgui", "superqt", "app_model"):
        try:
            napari_hiddenimports += collect_submodules(package)
        except Exception:  # noqa: BLE001
            pass
    try:
        # Napari's own plugin manifests are registered as entry points. The
        # console plugin is excluded from this build, so shipping its manifest
        # would leave Napari announcing a plugin it cannot import.
        ep_datas, ep_hidden = collect_entry_point("napari.manifest")[:2]
        datas += [d for d in ep_datas if "napari_console" not in str(d[0]).lower()]
        napari_hiddenimports += [
            h for h in ep_hidden if "napari_console" not in h.lower()
        ]
    except Exception:  # noqa: BLE001
        pass

# --------------------------------------------------------------------------
# Hidden imports and exclusions, from the plan
# --------------------------------------------------------------------------

hiddenimports = plan.hidden_imports(measurement) + napari_hiddenimports
try:
    excludes = plan.excludes(measurement, napari=BUNDLE_NAPARI)
except plan.BundleError as exc:
    raise SystemExit(f"corridor.spec: {exc}")

forbidden = plan.forbidden_sources(datas)
forbidden += [
    h for h in hiddenimports if h.split(".")[0] in plan.NEVER_BUNDLED_IMPORTS
]
if forbidden:
    raise SystemExit(
        "corridor.spec: refusing to bundle research code or data:\n  "
        + "\n  ".join(forbidden)
    )

print(
    f"corridor.spec: model {model.model_id} ({model.sha256[:12]}), "
    f"{len(measurement.cellpose_modules)} measured cellpose modules, "
    f"napari {'bundled' if BUNDLE_NAPARI else 'not bundled'}"
)

block_cipher = None

a = Analysis(
    [str(SPEC_DIR / "entry.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(SPEC_DIR / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Corridor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # a scientific app must not flash a console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ASSETS / "corridor.ico"),
    version=str(SPEC_DIR / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Corridor",
)
