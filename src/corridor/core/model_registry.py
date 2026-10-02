"""The validated segmentation model, located and locked by SHA-256.

Corridor 1.x let the model be a setting: a file picker, ``--model``, a
built-in Cellpose name, companion models found beside the primary one, and a
``CORRIDOR_MODEL`` variable that outranked all of them.  A result could
therefore come from a model nobody had validated, and nothing in it said so:
an analysis computed the SHA-256 after the run and never compared it.  Only
the self-test checked the bundled file against its pinned hash.

In 2.0 the model is not a choice.  ``assets/model_registry.json`` lists every
validated model with its checksum, and production resolves exactly one per
dimensionality:

*   The file is hashed **before** it is loaded, and a file that does not match
    is never returned -- from any location.  There is no fallback to
    ``cyto3``, ``cpsam`` or any other model, and Cellpose is only ever given
    the verified absolute path, never a bare model name.
*   The only way around the registry is explicit and loud:
    ``CORRIDOR_DEVELOPER=1`` *and* ``CORRIDOR_DEVELOPER_MODEL=<path>``, or
    :func:`research_model` from a script.  Both return
    ``developer_override=True``, which run.json records and quality control
    raises as a critical issue, so such a result can never pass for a
    validated one.
*   No model is validated on 3-D data, so a 3-D request is refused here --
    with or without the developer override, which swaps the file behind a
    dimensionality the registry validates and cannot open one it does not.
    A 3-D model experiment goes through :func:`research_model`.  3-D
    measurement and tracking still work on an imported label image.

The check runs at resolution time.  The caller must hand the returned path to
Cellpose straight away; a file swapped between the two would not be noticed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from corridor import app_meta, resources

REGISTRY_FILENAME = "model_registry.json"

#: Shown to the user, verbatim, whenever production cannot resolve its model.
#: The paths tried follow it.
MODEL_UNAVAILABLE_MESSAGE = (
    "The validated Corridor segmentation model is missing or does not match the "
    "expected checksum."
)

#: Both must be set (the first to exactly "1") for the override to apply.
ENV_DEVELOPER = "CORRIDOR_DEVELOPER"
ENV_DEVELOPER_MODEL = "CORRIDOR_DEVELOPER_MODEL"
DEVELOPER_OVERRIDE_ID = "developer-override"

DIMENSIONALITIES = ("2D", "3D")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED = (
    "model_id",
    "model_version",
    "architecture",
    "cellpose_version",
    "filename",
    "sha256",
    "training_dataset_version",
    "dimensions",
    "pixel_size_range_um",
    "validated_app_version",
)


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    model_id: str  # "jhu_confined_cp3_combi"
    model_version: str
    architecture: str  # "cellpose3-cyto2-resnet"
    cellpose_version: str  # requirement, e.g. ">=3,<4"
    filename: str  # file name on disk
    sha256: str
    training_dataset_version: str
    dimensions: tuple[str, ...]  # ("2D",)
    pixel_size_range_um: tuple[float, float] | None
    validated_app_version: str
    notes: str = ""

    def accepts_cellpose(self, version: str) -> bool:
        """Whether an installed Cellpose ``version`` meets ``cellpose_version``.

        The model was trained under Cellpose 3; under 4 the same weights do
        not behave the same way, so the requirement is checked rather than
        assumed.  Clauses are comma-separated comparisons on the numeric
        release (``>=3,<4``); a version with no leading number fails.
        """
        installed = _release(version)
        if installed is None:
            return False
        for clause in filter(None, (c.strip() for c in self.cellpose_version.split(","))):
            match = re.match(r"^(>=|<=|==|!=|>|<)\s*([0-9][0-9.]*)$", clause)
            if not match:
                return False
            op, wanted = match.group(1), _release(match.group(2))
            if wanted is None or not _compare(installed, op, wanted):
                return False
        return True


class ModelUnavailable(RuntimeError):
    """The validated model cannot be used. Never a cue to try another model."""

    def __init__(self, message: str, tried: Iterable[tuple[Path, str]] = ()) -> None:
        self.reason = message
        #: ``(path, why it was refused)`` for every location looked at.
        self.tried: tuple[tuple[Path, str], ...] = tuple(tried)
        text = message
        if self.tried:
            text += "\n\nPaths tried:\n" + "\n".join(
                f"  {path}  ({why})" for path, why in self.tried
            )
        super().__init__(text)


@dataclass(frozen=True)
class ResolvedModel:
    spec: ModelSpec
    path: Path
    sha256: str
    developer_override: bool = False

    def to_manifest(self) -> dict[str, Any]:
        """The ``run.json["model"]`` block (contract §2)."""
        return {
            "model_id": self.spec.model_id,
            "model_version": self.spec.model_version,
            "architecture": self.spec.architecture,
            "sha256": self.sha256,
            "cellpose_version": self.spec.cellpose_version,
            "training_dataset_version": self.spec.training_dataset_version,
            "developer_override": self.developer_override,
        }


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def registry_path() -> Path:
    """Where the registry ships: ``assets/`` in a checkout and in the bundle."""
    return resources.asset(REGISTRY_FILENAME)


def load_registry(path: Path | None = None) -> list[ModelSpec]:
    """Read and validate the registry. Malformed entries raise ValueError.

    Strict on purpose: a registry entry is the definition of "validated", so
    a missing checksum or an unknown dimensionality is a packaging error to
    stop on, not a gap to fill with a default.  Unknown keys are ignored so a
    newer registry still loads.
    """
    path = Path(path) if path is not None else registry_path()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = data.get("models") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected an object with a 'models' list")
    specs = [_spec_from_entry(entry, path) for entry in entries]
    ids = [s.model_id for s in specs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate model_id in {ids}")
    return specs


def _spec_from_entry(entry: Any, path: Path) -> ModelSpec:
    if not isinstance(entry, dict):
        raise ValueError(f"{path}: every model entry must be an object")
    missing = [k for k in _REQUIRED if k not in entry]
    if missing:
        raise ValueError(f"{path}: model entry {entry.get('model_id')!r} lacks {missing}")
    sha = str(entry["sha256"]).strip().lower()
    if not _SHA256.match(sha):
        raise ValueError(f"{path}: {entry['model_id']!r} has no valid SHA-256")
    dims = tuple(_normalise_dimensionality(d) for d in entry["dimensions"])
    if not dims:
        raise ValueError(f"{path}: {entry['model_id']!r} lists no dimensions")
    rng = entry["pixel_size_range_um"]
    if rng is not None:
        if len(rng) != 2 or not float(rng[0]) <= float(rng[1]):
            raise ValueError(f"{path}: {entry['model_id']!r} has a malformed pixel_size_range_um")
        rng = (float(rng[0]), float(rng[1]))
    return ModelSpec(
        model_id=str(entry["model_id"]),
        model_version=str(entry["model_version"]),
        architecture=str(entry["architecture"]),
        cellpose_version=str(entry["cellpose_version"]),
        filename=str(entry["filename"]),
        sha256=sha,
        training_dataset_version=str(entry["training_dataset_version"]),
        dimensions=dims,
        pixel_size_range_um=rng,
        validated_app_version=str(entry["validated_app_version"]),
        notes=str(entry.get("notes", "")),
    )


def production_spec(dimensionality: str = "2D") -> ModelSpec:
    """The one registered model for ``dimensionality``, or ModelUnavailable."""
    dim = _normalise_dimensionality(dimensionality)
    try:
        specs = load_registry()
    except (OSError, ValueError) as exc:
        raise ModelUnavailable(
            f"{MODEL_UNAVAILABLE_MESSAGE}\n\nThe model registry could not be read: {exc}"
        ) from exc
    matching = [s for s in specs if dim in s.dimensions]
    if not matching:
        if dim == "3D":
            raise ModelUnavailable(
                "No 3D-validated segmentation model is registered, so a 3-D stack "
                "cannot be segmented. The only validated model was trained and "
                "checked on 2-D phase-contrast frames, and there is no 3-D ground "
                "truth to validate one against. 3-D measurement and tracking still "
                "work on an imported label image."
            )
        raise ModelUnavailable(f"No {dim}-validated segmentation model is registered.")
    if len(matching) > 1:
        raise ModelUnavailable(
            f"The model registry lists {len(matching)} models for {dim} "
            f"({', '.join(s.model_id for s in matching)}); production needs exactly one."
        )
    return matching[0]


# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------


def _user_model_dir() -> Path:
    """``%LOCALAPPDATA%/Corridor/models``: where a model pack would install.

    Mirrors ``store.db.app_data_dir`` (including its ``CORRIDOR_DATA_DIR``
    override, which tests use to keep away from the real user's data)
    without importing the store into core.
    """
    override = os.environ.get("CORRIDOR_DATA_DIR")
    if override:
        base = Path(override)
    elif os.name == "nt":
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        base = Path(local) / app_meta.LOCAL_DIR_NAME
    else:
        share = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        base = Path(share) / app_meta.LOCAL_DIR_NAME
    return base / "models"


def _source_root() -> Path:
    """Repository root of a source checkout (``src/corridor/resources.py``)."""
    return Path(resources.__file__).resolve().parents[2]


def candidate_paths(spec: ModelSpec) -> list[Path]:
    """Every place ``spec.filename`` may be, in search order.

    1.  The bundle's ``assets/models/`` (``sys._MEIPASS`` when frozen,
        ``src/corridor/assets/models`` in a checkout, where the release build
        stages it).
    2.  ``%LOCALAPPDATA%/Corridor/models/``, the future model-pack location.
    3.  In a source checkout only, the supplied data tree's two copies.

    Order decides which *matching* file is used, never whether a mismatching
    one is: every candidate is hashed.
    """
    paths = [resources.asset("models", spec.filename), _user_model_dir() / spec.filename]
    if not resources.is_frozen():
        data = _source_root() / "data" / "confinedmig_cellTrack"
        paths += [
            data / "cp_custom_model" / spec.filename,
            data / "CellPose_TrainData" / "KK1KK2_combiModel" / "models" / spec.filename,
        ]
    unique: list[Path] = []
    for p in paths:
        if p not in unique:
            unique.append(p)
    return unique


# --------------------------------------------------------------------------
# Hashing and resolution
# --------------------------------------------------------------------------

#: (resolved path, size, mtime_ns) -> sha256. The model is 26.6 MB and the
#: GUI resolves it for every run; a file that changes changes its size or
#: mtime.
_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes, uncached."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_cached(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _HASH_CACHE.get(key)
    if cached is None:
        cached = sha256_file(path)
        _HASH_CACHE[key] = cached
    return cached


def resolve_model(dimensionality: str = "2D") -> ResolvedModel:
    """The verified model file for production, or ModelUnavailable.

    The registry is consulted first, so a dimensionality with no validated
    model (3-D) is refused even under the developer override: the contract
    refuses a 3-D stack for segmentation outright.  Then a developer
    override (both environment variables) replaces the registered file, and
    says so in its result.  Otherwise each candidate is hashed in order and
    the first exact match is returned; mismatches and absences are collected
    so the error names every path and why it was refused.
    """
    spec = production_spec(dimensionality)
    override = _developer_override(dimensionality)
    if override is not None:
        return override
    tried: list[tuple[Path, str]] = []
    for path in candidate_paths(spec):
        if not path.exists():
            tried.append((path, "missing"))
            continue
        if not path.is_file():
            tried.append((path, "not a file"))
            continue
        try:
            digest = _sha256_cached(path)
        except OSError as exc:
            tried.append((path, f"unreadable: {exc}"))
            continue
        if digest != spec.sha256:
            tried.append((path, f"checksum mismatch: sha256 {digest}"))
            continue
        return ResolvedModel(spec=spec, path=path.resolve(), sha256=digest)
    raise ModelUnavailable(MODEL_UNAVAILABLE_MESSAGE, tried)


def research_model(path: Path, *, label: str) -> ResolvedModel:
    """An unregistered model for a script, explicitly marked as such.

    The result carries ``developer_override=True`` whatever the file is, so
    anything produced with it is never mistaken for a validated result.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise ModelUnavailable(
            f"The research model {label!r} does not exist.", [(path, "missing")]
        )
    return _unregistered(path, model_id=f"research:{label}", dims=(), note=f"research model {label!r}")


def _developer_override(dimensionality: str) -> ResolvedModel | None:
    flag = os.environ.get(ENV_DEVELOPER, "").strip()
    target = os.environ.get(ENV_DEVELOPER_MODEL, "").strip()
    if flag != "1" or not target:
        return None
    path = Path(target).expanduser()
    if not path.is_file():
        # The developer asked for a specific file; quietly running the
        # production model instead would answer a different question.
        raise ModelUnavailable(
            f"{ENV_DEVELOPER}=1 names a developer model that does not exist.",
            [(path, "missing")],
        )
    return _unregistered(
        path,
        model_id=DEVELOPER_OVERRIDE_ID,
        dims=(_normalise_dimensionality(dimensionality),),
        note=f"set by {ENV_DEVELOPER_MODEL}",
    )


def _unregistered(path: Path, *, model_id: str, dims: tuple[str, ...], note: str) -> ResolvedModel:
    digest = _sha256_cached(path)
    spec = ModelSpec(
        model_id=model_id,
        model_version="unregistered",
        architecture="unknown",
        cellpose_version="",
        filename=path.name,
        sha256=digest,
        training_dataset_version="unknown",
        dimensions=dims,
        pixel_size_range_um=None,
        validated_app_version="",
        notes=f"Not validated ({note}).",
    )
    return ResolvedModel(spec=spec, path=path.resolve(), sha256=digest, developer_override=True)


# --------------------------------------------------------------------------


def _normalise_dimensionality(value: str) -> str:
    text = str(value).strip().upper()
    if text not in DIMENSIONALITIES:
        raise ValueError(f"dimensionality must be one of {DIMENSIONALITIES}, got {value!r}")
    return text


def _release(version: str) -> tuple[int, ...] | None:
    match = re.match(r"^\s*v?(\d+(?:\.\d+)*)", str(version))
    if not match:
        return None
    return tuple(int(p) for p in match.group(1).split("."))


def _compare(a: tuple[int, ...], op: str, b: tuple[int, ...]) -> bool:
    width = max(len(a), len(b))
    a = a + (0,) * (width - len(a))
    b = b + (0,) * (width - len(b))
    return {
        ">=": a >= b,
        "<=": a <= b,
        ">": a > b,
        "<": a < b,
        "==": a == b,
        "!=": a != b,
    }[op]
