"""What goes into the frozen bundle, decided from measurement.

``corridor.spec`` builds from this module, and so do ``tests/test_packaging.py``
and the release workflow, so the list a test checks is the list a build uses.
It imports only the standard library: it has to load inside PyInstaller's spec
namespace, in CI before the package is installed, and in a test run that must
not pay for torch.

Four decisions live here, each with the evidence it rests on:

*   **Cellpose hidden imports are the measured list**
    (``cellpose_runtime_modules.txt``, written by
    ``scripts/measure_runtime_imports.py``), not ``collect_submodules``, which
    shipped Cellpose's GUI, training loop and Dask ``contrib`` tree and, through
    their optional imports, whatever those pulled in.
*   **Nothing is excluded that the measurement shows loaded.** Every exclusion
    below is a candidate, applied only while the measurement and a static scan
    of ``src/corridor`` both say nothing needs it.
*   **Napari is opt-in** (``CORRIDOR_BUNDLE_NAPARI=1``). It used to ride in on
    whether the build venv happened to have it, so CI (no Napari) and local
    builds (Napari 0.9.1) produced two different bundles under one version.
*   **The model is the registry's production model, hash-checked here.**
    Running PyInstaller directly used to skip the only hash gate, which lived
    in ``build_release.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

PACKAGING = Path(__file__).resolve().parent
ROOT = PACKAGING.parent
SRC = ROOT / "src"
APP_SOURCE = SRC / "corridor"
ASSETS = APP_SOURCE / "assets"
REGISTRY_PATH = ASSETS / "model_registry.json"
RUNTIME_MODULES_PATH = PACKAGING / "cellpose_runtime_modules.txt"
RUNTIME_IMPORTS_PATH = PACKAGING / "runtime_imports.json"

NAPARI_ENV = "CORRIDOR_BUNDLE_NAPARI"
_ON = {"1", "true", "yes", "on"}
_OFF = {"", "0", "false", "no", "off"}

#: Top-level directories of the repository that never reach the bundle: the
#: researchers' microscopy and weights, the research code (design §9: "the
#: installer never contains it"), and build output.
NEVER_BUNDLED_DIRS = ("data", "training", "build")
#: Import roots that never reach the bundle. ``training`` is excluded outright
#: so that an accidental import from ``src/corridor`` fails the frozen
#: self-test instead of quietly shipping research code.
NEVER_BUNDLED_IMPORTS = ("training",)

#: The application's own entry points plus the scientific modules PyInstaller's
#: static analysis is known to miss (lazy loaders and compiled submodules).
APP_HIDDENIMPORTS = (
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
)

#: Cellpose subtrees inference has no use for: the Qt GUI, training, the
#: denoising and export tools, the CLI, Dask-distributed segmentation, the
#: transformer backbone (the production model is a ResNet) and an MKL probe.
#: Each is excluded only while no measured module is it or lies under it, so a
#: Cellpose upgrade that starts loading one keeps it rather than breaking
#: Analyse.
CELLPOSE_EXCLUSION_CANDIDATES = (
    "cellpose.__main__",
    "cellpose.cli",
    "cellpose.contrib",
    "cellpose.denoise",
    "cellpose.export",
    "cellpose.gui",
    "cellpose.segformer",
    "cellpose.test_mkl",
    "cellpose.train",
)

#: Third-party packages that reach the bundle only through optional imports in
#: other libraries (tqdm's ``pandas()`` helper, scikit-image's Dask path,
#: torch's optional vision hooks). Excluded only when the runtime measurement
#: did not load them, ``src/corridor`` never imports them, and Napari -- which
#: genuinely requires pandas and Dask -- is not being bundled.
OPTIONAL_THIRD_PARTY_CANDIDATES = ("pandas", "dask", "torchvision")

#: Napari and the packages only it brings. Excluded when Napari is not
#: requested: ``corridor.viz.napari_qc`` and ``corridor.selftest`` import
#: napari inside functions, which PyInstaller's static analysis follows, so
#: merely having Napari in the build venv would otherwise bundle it.
NAPARI_STACK = (
    "napari",
    "napari_builtins",
    "napari_console",
    "napari_plugin_engine",
    "napari_svg",
    "npe2",
    "vispy",
    "magicgui",
    "app_model",
    "in_n_out",
    "pint",
)

#: Never needed by a scientific desktop app, whatever was measured.
BASE_EXCLUDES = (
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
    # Napari's optional embedded IPython console pulls in Jupyter and adds
    # well over a hundred megabytes. The viewer works without it; only the
    # terminal button inside Napari is unavailable.
    "napari_console", "qtconsole", "ipykernel", "jupyter_client",
    "jupyter_core", "debugpy", "pydevd",
)


class BundleError(RuntimeError):
    """The bundle cannot be built as planned; the message says why."""


class RegistryError(BundleError):
    """``model_registry.json`` is missing, malformed or ambiguous."""


# --------------------------------------------------------------------------
# Napari opt-in
# --------------------------------------------------------------------------


def napari_requested(environ: Mapping[str, str] | None = None) -> bool:
    """Whether ``CORRIDOR_BUNDLE_NAPARI`` asks for Napari. Off when unset.

    A value that is neither on nor off is an error rather than off: a typo in
    a release build must not quietly ship without the viewer it asked for.
    """
    value = (os.environ if environ is None else environ).get(NAPARI_ENV, "")
    value = value.strip().lower()
    if value in _ON:
        return True
    if value in _OFF:
        return False
    raise BundleError(f"{NAPARI_ENV}={value!r}: use 1 to bundle Napari, 0 or unset not to")


# --------------------------------------------------------------------------
# The production model, from the registry
# --------------------------------------------------------------------------


_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ProductionModel:
    model_id: str
    filename: str
    sha256: str


def production_model(
    registry: Path = REGISTRY_PATH, dimensionality: str = "2D"
) -> ProductionModel:
    """The registry's production model for ``dimensionality``, validated.

    Parsed defensively and independently of ``corridor.core.model_registry``:
    the spec and the release workflow run this before the package is
    importable, and a malformed registry must stop a build with a reason, not
    a KeyError -- or worse, ship whichever entry happened to come first.
    """
    try:
        data = json.loads(registry.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise RegistryError(f"The model registry is missing: {registry}") from exc
    except (OSError, ValueError) as exc:
        raise RegistryError(f"The model registry cannot be read: {registry}: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistryError("The model registry is not a JSON object")

    production = data.get("production")
    if not isinstance(production, dict):
        raise RegistryError("The model registry has no 'production' map")
    model_id = production.get(dimensionality)
    if isinstance(model_id, dict):
        model_id = model_id.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RegistryError(f"The model registry names no {dimensionality} production model")

    models = data.get("models")
    if not isinstance(models, list):
        raise RegistryError("The model registry has no 'models' list")
    entries = [m for m in models if isinstance(m, dict) and m.get("model_id") == model_id]
    if len(entries) != 1:
        raise RegistryError(
            f"The production model {model_id!r} appears {len(entries)} times in the "
            "registry; it must appear exactly once"
        )
    entry = entries[0]

    filename = entry.get("filename")
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or any(sep in filename for sep in ("/", "\\", ":"))
    ):
        # The name is joined onto assets/models here and onto a download path
        # in CI; a separator or drive would write somewhere else.
        raise RegistryError(f"{model_id!r} has no plain file name: {filename!r}")
    sha256 = entry.get("sha256")
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256.strip().lower()):
        raise RegistryError(f"{model_id!r} has no valid SHA-256: {sha256!r}")
    return ProductionModel(model_id, filename, sha256.strip().lower())


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def staged_model_path(model: ProductionModel, assets: Path = ASSETS) -> Path:
    return assets / "models" / model.filename


def verify_staged_model(model: ProductionModel, assets: Path = ASSETS) -> Path:
    """The staged weights, after checking they are the validated ones."""
    path = staged_model_path(model, assets)
    if not path.is_file():
        raise BundleError(
            f"The production model {model.filename} is not staged at {path}. "
            "Run scripts/build_release.py, which stages it from data/ and checks it."
        )
    digest = sha256_file(path)
    if digest != model.sha256:
        raise BundleError(
            f"{path} has SHA-256 {digest}; the registry's {model.model_id!r} is "
            f"{model.sha256}. Refusing to ship a model that is not the validated one."
        )
    return path


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Measurement:
    """What ``scripts/measure_runtime_imports.py`` recorded."""

    cellpose_modules: tuple[str, ...]
    #: Third-party top-level packages loaded by the runtime path; None when the
    #: provenance file is absent, which makes every guarded exclusion stand
    #: down rather than guess.
    third_party_top_level: frozenset[str] | None = None
    cellpose_version: str | None = None
    app_source_imports: dict[str, list[str]] = field(default_factory=dict)


def read_runtime_modules(path: Path = RUNTIME_MODULES_PATH) -> tuple[str, ...]:
    if not path.is_file():
        raise BundleError(
            f"{path.name} is missing; run scripts/measure_runtime_imports.py"
        )
    names = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    if not names:
        raise BundleError(f"{path.name} lists no modules")
    return tuple(names)


def load_measurement(
    modules_path: Path = RUNTIME_MODULES_PATH,
    imports_path: Path = RUNTIME_IMPORTS_PATH,
) -> Measurement:
    modules = read_runtime_modules(modules_path)
    try:
        record = json.loads(imports_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return Measurement(modules)
    top = record.get("third_party_top_level")
    return Measurement(
        cellpose_modules=modules,
        third_party_top_level=frozenset(top) if isinstance(top, list) else None,
        cellpose_version=record.get("cellpose_version"),
        app_source_imports=record.get("app_source_imports") or {},
    )


def check_measurement_current(measurement: Measurement, installed_cellpose: str) -> None:
    """Refuse to build against a Cellpose other than the one measured.

    The module list is evidence about one Cellpose release; after an upgrade it
    is stale, and a stale list fails only when a user presses Analyse.
    """
    if measurement.cellpose_version is None:
        raise BundleError(
            f"{RUNTIME_IMPORTS_PATH.name} does not say which Cellpose was measured; "
            "run scripts/measure_runtime_imports.py"
        )
    if measurement.cellpose_version != installed_cellpose:
        raise BundleError(
            f"The runtime imports were measured with cellpose "
            f"{measurement.cellpose_version}, but this environment has "
            f"{installed_cellpose}. Re-run scripts/measure_runtime_imports.py."
        )


def app_imports(package: str, source: Path = APP_SOURCE) -> list[str]:
    """``file:line`` of every import of ``package`` in the application source.

    Static on purpose: it sees imports inside functions too, which is what
    PyInstaller's analysis follows.
    """
    pattern = re.compile(rf"^\s*(?:import|from)\s+{re.escape(package)}\b", re.MULTILINE)
    sites = []
    for path in sorted(source.rglob("*.py")):
        text = path.read_text(encoding="utf-8-sig")
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            sites.append(f"{path.relative_to(source.parent).as_posix()}:{line}")
    return sites


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


def _covers(prefix: str, name: str) -> bool:
    """Whether excluding ``prefix`` would also remove module ``name``."""
    return name == prefix or name.startswith(prefix + ".")


def hidden_imports(measurement: Measurement) -> list[str]:
    """The application's entry points plus every measured Cellpose module.

    Napari's hidden imports are added by the spec, which needs PyInstaller's
    ``collect_submodules`` to enumerate them.
    """
    names = list(APP_HIDDENIMPORTS)
    names += [m for m in measurement.cellpose_modules if m not in names]
    return names


def excludes(
    measurement: Measurement, *, napari: bool, source: Path = APP_SOURCE
) -> list[str]:
    """Modules PyInstaller must leave out, each one justified above."""
    measured = measurement.cellpose_modules
    loaded = measurement.third_party_top_level
    names = list(dict.fromkeys(BASE_EXCLUDES + NEVER_BUNDLED_IMPORTS))

    for candidate in CELLPOSE_EXCLUSION_CANDIDATES:
        if not any(_covers(candidate, m) for m in measured):
            names.append(candidate)

    if not napari:
        names += [
            n for n in NAPARI_STACK
            if n not in names and (loaded is None or n not in loaded)
        ]
        for candidate in OPTIONAL_THIRD_PARTY_CANDIDATES:
            if loaded is None or candidate in loaded:
                continue  # unmeasured or measured as needed: keep it
            if app_imports(candidate, source):
                continue
            names.append(candidate)
    elif loaded is not None and "torchvision" not in loaded and not app_imports(
        "torchvision", source
    ):
        # Napari needs pandas and Dask, not torchvision.
        names.append("torchvision")

    # The guarantee the guards above exist for, checked once more as a whole.
    # BASE_EXCLUDES are not held to the third-party half: matplotlib or qtconsole
    # "loaded" by the measurement only means an optional try-import found them
    # in that venv, and the bundle is built to work with those imports failing.
    conflicts = [m for m in measured for n in names if _covers(n, m)]
    if conflicts:
        raise BundleError(f"Planned exclusions would remove measured modules: {conflicts}")
    return names


def asset_datas(assets: Path = ASSETS, registry: Path = REGISTRY_PATH) -> list[tuple[str, str]]:
    """The application's own data files: icons and the model registry.

    The registry is required, because the frozen application resolves and
    hash-checks its model through it; the weights go in separately.
    """
    if not registry.is_file():
        raise RegistryError(f"The model registry is missing: {registry}")
    datas = [(str(p), "assets") for p in sorted(assets.glob("*")) if p.is_file()]
    if str(registry) not in {d[0] for d in datas}:
        datas.append((str(registry), "assets"))
    return datas


def model_datas(model: ProductionModel, assets: Path = ASSETS) -> list[tuple[str, str]]:
    return [(str(verify_staged_model(model, assets)), "assets/models")]


def forbidden_sources(
    datas: list[tuple[str, str]], root: Path = ROOT
) -> list[str]:
    """Data sources that sit in a never-bundled directory of the repository."""
    banned = [(root / d).resolve() for d in NEVER_BUNDLED_DIRS]
    found = []
    for source, _dest in datas:
        path = Path(source).resolve()
        if any(path == b or b in path.parents for b in banned):
            found.append(source)
    return found


def _main(argv: list[str] | None = None) -> int:
    """``python packaging/bundle_plan.py model`` prints the production model
    as JSON (the release workflow reads it); ``plan`` prints the hidden
    imports and exclusions for both Napari settings."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Corridor bundle plan")
    parser.add_argument("what", choices=("model", "plan"))
    args = parser.parse_args(argv)
    try:
        if args.what == "model":
            model = production_model()
            print(json.dumps(
                {"model_id": model.model_id, "filename": model.filename,
                 "sha256": model.sha256}
            ))
        else:
            measurement = load_measurement()
            print(json.dumps({
                "hiddenimports": hidden_imports(measurement),
                "excludes_without_napari": excludes(measurement, napari=False),
                "excludes_with_napari": excludes(measurement, napari=True),
            }, indent=2))
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
