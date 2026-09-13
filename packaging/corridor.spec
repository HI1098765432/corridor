# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for Corridor.

Notes that matter for this particular bundle:

*   Cellpose reads several data files at runtime and imports parts of itself
    lazily, so it needs both ``collect_data_files`` and ``collect_submodules``.
*   Torch must not be pruned: PyInstaller cannot see through its dynamic
    imports, and a missing operator module only fails when a user presses
    Analyse.
*   The custom Cellpose model ships inside the bundle so the application works
    the moment it is installed.
*   ``console=False`` so launching the app never flashes a terminal.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

SPEC_DIR = Path(SPECPATH).resolve()
ROOT = SPEC_DIR.parent
SRC = ROOT / "src"
ASSETS = SRC / "corridor" / "assets"

# --------------------------------------------------------------------------
# Data files
# --------------------------------------------------------------------------

datas = []

# Application assets (icon).
for asset in ASSETS.glob("*"):
    if asset.is_file():
        datas.append((str(asset), "assets"))

# The trained Cellpose model. Bundling it is what makes the installed app
# usable without any further downloads or configuration.
MODEL_NAME = "cyto2_phase_microfluidic_KK1KK2_combi"
model_candidates = [
    ASSETS / "models" / MODEL_NAME,
    ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
    / "KK1KK2_combiModel" / "models" / MODEL_NAME,
]
for candidate in model_candidates:
    if candidate.exists():
        datas.append((str(candidate), "assets/models"))
        break
else:
    raise SystemExit(
        f"The Cellpose model {MODEL_NAME} was not found. Looked in:\n  "
        + "\n  ".join(str(c) for c in model_candidates)
    )

# Third-party data files.
for package in ("cellpose", "skimage", "scipy", "numba", "llvmlite", "torch"):
    try:
        datas += collect_data_files(package)
    except Exception:  # noqa: BLE001 - absent optional package
        pass

# --------------------------------------------------------------------------
# Hidden imports
# --------------------------------------------------------------------------

hiddenimports = [
    "corridor",
    "corridor.cli",
    "corridor.ui.app",
    "corridor.ui.main_window",
    "scipy.optimize",
    "scipy.special",
    "scipy._lib.array_api_compat.numpy.fft",
    "skimage.measure",
    "skimage.morphology",
    "skimage.filters",
    "imagecodecs",
    "PIL.Image",
]
for package in ("cellpose", "torch"):
    try:
        hiddenimports += collect_submodules(package)
    except Exception:  # noqa: BLE001
        pass

# --------------------------------------------------------------------------
# Trim what is genuinely not used
# --------------------------------------------------------------------------

excludes = [
    "tkinter", "matplotlib", "pytest", "IPython", "jupyter", "notebook",
    "PyQt5", "PyQt6", "PySide2", "wx",
    # Qt modules a scientific desktop app has no use for. Each is tens of MB.
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtWebView", "PySide6.QtQuick3D", "PySide6.Qt3DCore", "PySide6.Qt3DRender",
    "PySide6.Qt3DAnimation", "PySide6.Qt3DExtras", "PySide6.Qt3DInput", "PySide6.Qt3DLogic",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtBluetooth", "PySide6.QtNfc",
    "PySide6.QtPositioning", "PySide6.QtLocation", "PySide6.QtSerialPort",
    "PySide6.QtSensors", "PySide6.QtTextToSpeech", "PySide6.QtSpatialAudio",
    "PySide6.QtRemoteObjects", "PySide6.QtScxml", "PySide6.QtHelp",
    "PySide6.QtDesigner", "PySide6.QtUiTools", "PySide6.QtPdf", "PySide6.QtPdfWidgets",
    "PySide6.QtQuick", "PySide6.QtQuickWidgets", "PySide6.QtQml",
]

block_cipher = None

a = Analysis(
    [str(SRC / "corridor" / "__main__.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
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
