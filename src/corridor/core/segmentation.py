"""Cellpose v3 segmentation, the measurable post-filter, and imported labels.

This module deliberately keeps two numbers apart for every frame: how many
instances Cellpose produced, and how many survived post-processing.  Without
that separation "the cell disappeared" is unattributable, and the temptation
is to fix the tracker for a problem that lives in segmentation.

The model is not a setting (contract §2).  :class:`SegmentationService`
resolves exactly one file through :mod:`corridor.core.model_registry`, hashes
it again immediately before Cellpose reads it, and hands Cellpose only that
verified absolute path.  The 1.x routes to any other model are gone:

*   ``CellposeModel(model_type=...)`` and bare names.  Cellpose 3.1 also
    falls back to ``cyto3`` *silently* when ``pretrained_model`` names a path
    that does not exist (``models.get_model_params``: a warning, then the
    default), so the file is checked to exist right before the call and the
    path Cellpose reports having loaded is compared with it afterwards.
*   ``SegmentationConfig.model_path`` / ``builtin_model`` /
    ``use_custom_model`` are never read to choose a model; a legacy value
    that names something else is recorded as ignored, never obeyed.
*   Companion-model discovery and the two ladder rungs that ran other models
    (``models``, ``max_recall``).  A legacy config naming one runs as ``off``
    and says so.  ``thresholds`` and ``wide`` re-run the same validated
    model at other thresholds and stay.

Imported label images (:func:`load_label_stack`) are the other source of a
segmentation, and the only route for 3-D data while no 3-D model is
validated.  Their provenance and checksum travel with the result so an
imported segmentation can never pass for one the validated model produced.
"""

from __future__ import annotations

import hashlib
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

import numpy as np

from . import model_registry
from .config import (
    ENSEMBLE_MAX_RECALL,
    ENSEMBLE_MODELS,
    ENSEMBLE_OFF,
    ENSEMBLE_THRESHOLDS,
    ENSEMBLE_WIDE,
    SegmentationConfig,
)
from .detections import (
    SOURCE_ENSEMBLE,
    SOURCE_PRIMARY,
    Detection,
    FrameDiagnostics,
    extract_detections,
    extract_detections_3d,
)
from .imaging import CANONICAL_AXES, UnsupportedStackError, read_label_array
from .model_registry import MODEL_UNAVAILABLE_MESSAGE, ModelUnavailable, ResolvedModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import Scale

__all__ = [
    "SOURCE_ENSEMBLE",
    "SOURCE_PRIMARY",
    "PROVENANCE_MODEL",
    "PROVENANCE_IMPORTED",
    "CellposeUnavailableError",
    "FilterResult",
    "SegmentationOutput",
    "SegmentationService",
    "cellpose_version",
    "check_cellpose_for",
    "discover_companion_models",
    "file_sha256",
    "filter_instances",
    "gpu_available",
    "load_label_stack",
    "merge_labelled",
    "threshold_passes",
]

_LOG = logging.getLogger(__name__)

#: Cellpose emits this when the flow field contains no basins at all.
_NO_MASKS_RE = re.compile(r"no (seeds|masks) found", re.IGNORECASE)

#: Where a segmentation came from (``SegmentationOutput.provenance``).
PROVENANCE_MODEL = "model"
PROVENANCE_IMPORTED = "imported_labels"

#: The ladder rungs that re-run the validated model at other thresholds.
_THRESHOLD_RUNGS = (ENSEMBLE_OFF, ENSEMBLE_THRESHOLDS, ENSEMBLE_WIDE)
#: The rungs that ran other models; removed in 2.0, loaded as ``off``.
_REMOVED_RUNGS = (ENSEMBLE_MODELS, ENSEMBLE_MAX_RECALL)


class CellposeUnavailableError(RuntimeError):
    """Raised when the Cellpose runtime cannot run the resolved model."""


def cellpose_version() -> str:
    """The installed Cellpose version, without importing Cellpose.

    ``import cellpose`` imports torch (``cellpose/version.py`` does), which
    costs seconds and hundreds of megabytes; the version is in the package
    metadata, which is where ``cellpose.version`` itself reads it from.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return str(version("cellpose"))
        except PackageNotFoundError:
            pass
    except Exception:  # pragma: no cover - importlib.metadata is stdlib
        pass
    try:  # pragma: no cover - a frozen build without the dist-info
        import cellpose

        return str(getattr(cellpose, "version", "unknown"))
    except Exception:  # pragma: no cover - only on a broken install
        return "unavailable"


def check_cellpose_for(resolved: ResolvedModel) -> str:
    """Confirm the installed Cellpose meets the model's requirement, before loading.

    Cellpose 4 refuses a Cellpose 3 checkpoint only through an ``assert``
    (on the absent ``W2`` key), which ``python -O`` and an optimised freeze
    remove, and CP4 releases before 4.0.8 load it and segment garbage.  The
    requirement is therefore checked explicitly against the registry entry
    (``>=3,<4`` for the lab model) rather than left to Cellpose.
    """
    installed = cellpose_version()
    if not resolved.spec.accepts_cellpose(installed):
        raise CellposeUnavailableError(
            f"The segmentation model {resolved.spec.model_id} needs Cellpose "
            f"{resolved.spec.cellpose_version}, but version {installed} is installed. "
            "The validated model was trained with Cellpose 3 and does not behave the same "
            "way under version 4."
        )
    return installed


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@contextmanager
def _capture_cellpose_messages(sink: list[str]) -> Iterator[None]:
    """Route Cellpose's own log lines into a list for per-frame diagnostics."""

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                sink.append(record.getMessage())
            except Exception:
                pass

    handler = _Handler(level=logging.INFO)
    targets = [logging.getLogger("cellpose"), logging.getLogger("cellpose.models"),
               logging.getLogger("cellpose.dynamics"), logging.getLogger("cellpose.core")]
    for logger in targets:
        logger.addHandler(handler)
    try:
        yield
    finally:
        for logger in targets:
            logger.removeHandler(handler)


# --------------------------------------------------------------------------
# Post-filter
# --------------------------------------------------------------------------


@dataclass
class FilterResult:
    mask: np.ndarray
    removed_extents: list[int]
    removed_areas: list[float]
    raw_count: int
    kept_count: int
    #: New label (1..K in the filtered mask) -> the label it had before
    #: filtering.  Without this the relabelling silently destroys any
    #: per-instance bookkeeping done upstream, which is exactly what the
    #: ensemble's provenance tags are.
    label_map: dict[int, int] = field(default_factory=dict)


def filter_instances(mask: np.ndarray, cfg: SegmentationConfig) -> FilterResult:
    """Drop instances that are too small to be a cell, and relabel 1..K.

    The research notebook's rule -- keep an instance when the larger dimension
    of its bounding box is at least ``min_extent`` pixels -- is preserved
    exactly, so results stay comparable.  What changes is that everything it
    removes is now counted.

    A ``(Z, Y, X)`` volume is filtered on the same XY terms: the extent is
    the larger XY bounding-box side, and ``min_area_px`` and the reported
    removed areas are the object's XY footprint (``Detection.area_px`` in
    3-D), so a pixel threshold is never compared with a voxel count.  A
    border-touching object in 3-D also includes one cut by the first or last
    slice.
    """
    mask = np.asarray(mask)
    if mask.size == 0:
        return FilterResult(mask.astype(np.int32), [], [], 0, 0, {})
    if mask.ndim not in (2, 3):
        raise ValueError(f"filter_instances needs a (Y, X) or (Z, Y, X) mask, got {mask.shape}")
    mask = mask.astype(np.int32, copy=False)
    labels = np.unique(mask)
    labels = labels[labels != 0]
    if labels.size == 0:
        return FilterResult(np.zeros_like(mask, dtype=np.int32), [], [], 0, 0, {})

    from scipy import ndimage

    h, w = mask.shape[-2:]
    boxes = ndimage.find_objects(mask)
    keep: list[int] = []
    removed_extents: list[int] = []
    removed_areas: list[float] = []

    for lab in labels:
        box = boxes[int(lab) - 1]
        if box is None:
            continue
        region = mask[box] == lab
        footprint = region.any(axis=0) if mask.ndim == 3 else region
        rows, cols = box[-2], box[-1]
        extent = max(int(rows.stop - rows.start), int(cols.stop - cols.start))
        area = float(footprint.sum())
        drop = extent < cfg.min_extent_px or (cfg.min_area_px and area < cfg.min_area_px)
        if not drop and cfg.drop_border_touching:
            touches = rows.start == 0 or cols.start == 0 or rows.stop >= h or cols.stop >= w
            if mask.ndim == 3:
                touches = touches or box[0].start == 0 or box[0].stop >= mask.shape[0]
            drop = bool(touches)
        if drop:
            removed_extents.append(extent)
            removed_areas.append(area)
        else:
            keep.append(int(lab))

    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    label_map: dict[int, int] = {}
    for new_id, lab in enumerate(keep, start=1):
        lut[lab] = new_id
        label_map[new_id] = lab
    return FilterResult(
        lut[mask], removed_extents, removed_areas, int(labels.size), len(keep), label_map
    )


def discover_companion_models(primary: str | Path | None) -> tuple[str, ...]:
    """Deprecated: companion models are never discovered or run in 2.0.

    1.x looked for sibling checkpoints beside the primary model and ran them
    in the ``models``/``max_recall`` rungs, so a result could depend on which
    unvalidated files happened to sit in a folder.  Always returns ``()``;
    it exists only so legacy importers (``pipeline.py``, the advanced panel)
    still load until they are rewritten.
    """
    return ()


def _largest_component(region: np.ndarray) -> np.ndarray:
    """The biggest connected piece of a boolean mask, or the mask if it is one."""
    from scipy import ndimage

    labelled, count = ndimage.label(region)
    if count <= 1:
        return region
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0  # background
    return labelled == int(sizes.argmax())


def merge_labelled(
    passes: list[tuple[np.ndarray, str]],
    *,
    overlap: float = 0.3,
    min_fragment_px: int = 20,
) -> tuple[np.ndarray, dict[int, str]]:
    """Union several label images, keeping the first claim on every pixel.

    Instances are considered in the order given, so the primary pass is
    authoritative and every later pass can only *add*.  A candidate that is
    mostly inside an already-accepted instance is the same cell seen twice and
    is dropped; a candidate that merely clips one is trimmed to the free pixels
    and kept only if enough of it survives.

    Returns the merged label image and a map from label to which pass found it.
    That second return value matters: "the model found this" and "only the
    permissive pass found this" are different claims, and collapsing them would
    hide exactly the instances a reader should look at first.
    """
    if not passes:
        return np.zeros((0, 0), dtype=np.int32), {}

    out = np.zeros_like(passes[0][0], dtype=np.int32)
    sources: dict[int, str] = {}
    next_label = 1
    for mask, tag in passes:
        mask = np.asarray(mask)
        if mask.shape != out.shape:
            continue
        for value in np.unique(mask):
            if value == 0:
                continue
            region = mask == value
            claimed = out[region]
            if claimed.any():
                if float((claimed > 0).sum()) / float(region.sum()) > overlap:
                    continue
                region = region & (out == 0)
                # Removing the claimed pixels can cut the candidate in two --
                # an elongated cell clipped across its middle leaves a piece at
                # each end. Labelling both as one instance would put its
                # centroid in the gap between them, which is a position no cell
                # occupies. Keep the largest piece and discard the rest.
                region = _largest_component(region)
                if int(region.sum()) < min_fragment_px:
                    continue
            out[region] = next_label
            sources[next_label] = tag
            next_label += 1
    return out, sources


def threshold_passes(cfg: SegmentationConfig) -> tuple[tuple[tuple[float, float], ...], list[str]]:
    """The extra ``(cellprob, flow)`` passes of the validated model, and notes.

    Only the rungs that re-run the same model survive.  A legacy ``models``
    or ``max_recall`` value runs as ``off`` -- not as its threshold part,
    because what was measured for those rungs was the combination with other
    models, and half of it is a configuration nobody measured.
    """
    mode = cfg.ensemble
    if mode in _REMOVED_RUNGS:
        return (), [
            f"The fallback rung '{mode}' ran other, unvalidated models and was removed in "
            "Corridor 2.0; this run used a single pass ('off')."
        ]
    if mode not in _THRESHOLD_RUNGS:
        return (), [f"Unknown fallback rung {mode!r}; a single pass ('off') was used."]
    passes = tuple(
        (float(cellprob), float(flow))
        for path, cellprob, flow in cfg.ensemble_passes()
        if path is None
    )
    return passes, []


def _legacy_model_notes(cfg: SegmentationConfig, resolved: ResolvedModel) -> list[str]:
    """Record, never obey, a v1 configuration's own idea of the model."""
    notes: list[str] = []
    if cfg.model_path:
        try:
            same = Path(cfg.model_path).resolve() == Path(resolved.path).resolve()
        except OSError:
            same = False
        if not same:
            notes.append(
                f"The configuration names the model file {cfg.model_path}; it was ignored. "
                "Corridor runs only the model resolved and verified by checksum."
            )
    if not cfg.use_custom_model:
        notes.append(
            f"The configuration asks for the built-in Cellpose model "
            f"'{cfg.builtin_model}'; it was ignored. Corridor never falls back to a "
            "built-in model."
        )
    if cfg.ensemble_model_paths:
        notes.append(
            f"{len(cfg.ensemble_model_paths)} companion model path(s) in the configuration "
            "were ignored; only the validated model runs."
        )
    return notes


def _require_registered(resolved: ResolvedModel, path: Path, digest: str) -> None:
    """Refuse a model that claims to be validated without being the registered file.

    ``ResolvedModel`` is a plain public dataclass, so its ``sha256`` and its
    ``spec`` are the caller's claims.  Comparing the file only with
    ``resolved.sha256`` let a hand-built ``ResolvedModel(spec=<production
    spec>, path=<any file>, sha256=<that file's hash>)`` load an unvalidated
    file and be recorded as the production model with ``developer_override``
    False (adversarial review, reproduced).  So a model that is not marked as
    an override must carry a spec that is, field for field, an entry of the
    registry on disk, and the file must hash to *that entry's* checksum.  A
    developer override or research model is exempt by design: it is loaded
    for what it is and recorded as such.
    """
    if resolved.developer_override:
        return
    try:
        registered = model_registry.load_registry()
    except (OSError, ValueError) as exc:
        raise ModelUnavailable(
            f"{MODEL_UNAVAILABLE_MESSAGE}\n\nThe model registry could not be read: {exc}",
            [(path, "registry unreadable")],
        ) from exc
    if resolved.spec not in registered:
        raise ModelUnavailable(
            f"{MODEL_UNAVAILABLE_MESSAGE}\n\nThe model claims to be "
            f"{resolved.spec.model_id} {resolved.spec.model_version}, which is not an entry "
            "of the model registry. An unregistered model runs only as a developer override.",
            [(path, "not a registered model")],
        )
    if digest != resolved.spec.sha256:
        raise ModelUnavailable(
            MODEL_UNAVAILABLE_MESSAGE,
            [
                (
                    path,
                    f"checksum mismatch: sha256 {digest}, but {resolved.spec.model_id} "
                    f"is registered as {resolved.spec.sha256}",
                )
            ],
        )


#: Appended to every 3-D refusal. The likeliest way to meet it by mistake is
#: a 2-D time-lapse whose planes the file labels as Z slices (ImageJ calls
#: every plain stack "slices"), and the way out of that is the axis order.
_AXES_HINT_3D = (
    "If this file is really a 2-D time-lapse whose planes are labelled as Z slices, "
    "set the axis order to TYX in the import settings (or with --axes TYX)."
)


def _not_validated_for(resolved: ResolvedModel, dimensionality: str) -> str:
    dims = ", ".join(resolved.spec.dimensions) or "no dimensionality"
    if dimensionality == "3D":
        return (
            f"The segmentation model {resolved.spec.model_id} is validated for {dims} only, "
            "so a 3-D stack cannot be segmented with it. No 3D-validated segmentation "
            "model is registered, and there is no 3-D ground truth to validate one "
            "against. 3-D measurement and tracking still work on an imported label image. "
            + _AXES_HINT_3D
        )
    return (
        f"The segmentation model {resolved.spec.model_id} is validated for {dims} only, "
        f"not {dimensionality}."
    )


# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------


@dataclass
class SegmentationOutput:
    masks: np.ndarray  # (T, [Z,] Y, X) int32, post-filter
    raw_masks: np.ndarray  # (T, [Z,] Y, X) int32, straight from Cellpose or the file
    detections: list[Detection]
    diagnostics: list[FrameDiagnostics]
    #: The verified absolute path Cellpose loaded ("" for imported labels).
    model_path: str
    model_sha256: str | None
    cellpose_version: str
    used_gpu: bool
    #: Segmentation passes that actually ran per frame: 1 plus the threshold
    #: rungs; 0 for imported labels, where nothing ran.
    passes_per_frame: int = 1
    #: Always empty since 2.0 (no companion models); kept for v1 readers.
    unavailable_models: list[str] = field(default_factory=list)
    #: The model that produced the masks, with its registry entry; None for
    #: imported labels. ``model.to_manifest()`` is run.json's "model" block.
    model: ResolvedModel | None = None
    #: PROVENANCE_MODEL or PROVENANCE_IMPORTED.
    provenance: str = PROVENANCE_MODEL
    #: SHA-256 and path of an imported label file.
    labels_sha256: str | None = None
    labels_path: str | None = None
    dimensionality: str = "2D"
    #: What the run did differently from what was asked (ignored legacy
    #: settings, removed rungs, label-file reading), for run.json and QC.
    notes: list[str] = field(default_factory=list)

    @property
    def developer_override(self) -> bool:
        return bool(self.model is not None and self.model.developer_override)

    def model_manifest(self) -> dict[str, Any] | None:
        """``run.json["model"]`` (contract §2), or None when no model ran."""
        return self.model.to_manifest() if self.model is not None else None


class SegmentationService:
    """Resolves the validated model once, loads it once, segments on demand.

    ``model`` is for tests and research scripts that pass an explicit
    :class:`~corridor.core.model_registry.ResolvedModel` (for example from
    ``model_registry.research_model``, which is always marked as a developer
    override); production passes nothing and the registry decides.
    ``scale`` supplies the Z anisotropy and voxel spacing for 3-D stacks.
    """

    def __init__(
        self,
        cfg: SegmentationConfig,
        *,
        model: ResolvedModel | None = None,
        scale: "Scale | None" = None,
    ) -> None:
        self.cfg = cfg
        self.scale = scale
        self._requested = model
        self._resolved: ResolvedModel | None = None
        self._model = None
        self._used_gpu = False
        self._passes, notes = threshold_passes(cfg)
        self.notes: list[str] = list(notes)
        #: Always empty: no optional model exists to be unavailable.
        self.unavailable_models: list[str] = []

    # -- model resolution and loading ------------------------------------------

    @property
    def resolved_model(self) -> ResolvedModel | None:
        """The verified model, once a segmentation call has resolved it."""
        return self._resolved

    @property
    def model(self):
        """The loaded Cellpose network for 2-D frames (recovery, self-test)."""
        return self._network("2D")

    def _resolve(self, dimensionality: str) -> ResolvedModel:
        if self._resolved is None:
            if self._requested is not None:
                resolved = self._requested
            else:
                try:
                    resolved = model_registry.resolve_model(dimensionality)
                except ModelUnavailable as exc:
                    if dimensionality != "3D" or "3D-validated" not in exc.reason:
                        raise
                    # The registry's "no 3D-validated model" refusal (it has
                    # no paths to list); add the way out for a mislabelled file.
                    raise ModelUnavailable(f"{exc.reason} {_AXES_HINT_3D}") from exc
            if dimensionality not in resolved.spec.dimensions and not resolved.developer_override:
                raise ModelUnavailable(_not_validated_for(resolved, dimensionality))
            self._resolved = resolved
            self.notes.extend(_legacy_model_notes(self.cfg, resolved))
        elif (
            dimensionality not in self._resolved.spec.dimensions
            and not self._resolved.developer_override
        ):
            raise ModelUnavailable(_not_validated_for(self._resolved, dimensionality))
        return self._resolved

    def _network(self, dimensionality: str):
        resolved = self._resolve(dimensionality)
        if self._model is None:
            self._model = self._load(resolved)
        return self._model

    def _load(self, resolved: ResolvedModel):
        """Hand Cellpose the verified absolute path, and nothing else.

        The file is re-hashed here, immediately before Cellpose reads it:
        the registry verified it at resolution time, but an explicit
        ``model`` may have been resolved long before, and a file replaced in
        between would otherwise be loaded unchecked.
        """
        check_cellpose_for(resolved)
        path = Path(resolved.path)
        if not path.is_absolute():
            raise ModelUnavailable(
                f"{MODEL_UNAVAILABLE_MESSAGE}\n\nThe model path is not absolute.", [(path, "relative")]
            )
        if not path.is_file():
            raise ModelUnavailable(MODEL_UNAVAILABLE_MESSAGE, [(path, "missing")])
        digest = model_registry.sha256_file(path)
        if digest != resolved.sha256:
            raise ModelUnavailable(
                MODEL_UNAVAILABLE_MESSAGE, [(path, f"checksum mismatch: sha256 {digest}")]
            )
        _require_registered(resolved, path, digest)

        want_gpu = bool(self.cfg.use_gpu and gpu_available())
        from cellpose import models

        try:
            network = models.CellposeModel(pretrained_model=str(path), gpu=want_gpu)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            raise CellposeUnavailableError(
                f"The Cellpose model could not be loaded.\n\n{exc}"
            ) from exc

        loaded = getattr(network, "pretrained_model", None)
        if isinstance(loaded, (list, tuple)):
            loaded = loaded[0] if loaded else None
        if loaded and Path(str(loaded)).resolve() != path.resolve():
            raise ModelUnavailable(
                f"{MODEL_UNAVAILABLE_MESSAGE}\n\nCellpose loaded {loaded} instead of the "
                "verified file; the result would not come from the validated model.",
                [(path, "not the file Cellpose loaded")],
            )
        self._used_gpu = want_gpu
        return network

    # -- 2-D -------------------------------------------------------------------

    def _eval(
        self,
        model,
        image: np.ndarray,
        cellprob: float,
        flow: float,
        messages: list[str],
    ) -> np.ndarray:
        with _capture_cellpose_messages(messages):
            result = model.eval(
                image,
                channels=list(self.cfg.channels),
                diameter=self.cfg.diameter,
                cellprob_threshold=cellprob,
                flow_threshold=flow,
                normalize=self.cfg.normalize_argument(),
            )
        return np.asarray(result[0]).astype(np.int32)

    def segment_frame(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, str, dict[int, str]]:
        """Run Cellpose on one 2-D frame.

        Returns ``(raw label image, message, label -> provenance)``.  With the
        fallback ladder off this is a single Cellpose call and every label is
        ``primary``.  With a threshold rung on, the extra passes of the same
        model are merged in afterwards and anything only they found is
        tagged, so a reader can always separate an ordinary detection from
        one that needed a more permissive setting to appear at all.
        """
        if np.ndim(image) != 2:
            raise ValueError(f"segment_frame takes one 2-D frame, got shape {np.shape(image)}")
        network = self._network("2D")
        messages: list[str] = []
        primary = self._eval(
            network,
            image,
            self.cfg.cellprob_threshold,
            self.cfg.flow_threshold,
            messages,
        )
        note = next((m for m in messages if _NO_MASKS_RE.search(m)), "")

        if not self._passes:
            return primary, note, {
                int(v): SOURCE_PRIMARY for v in np.unique(primary) if v
            }

        stack: list[tuple[np.ndarray, str]] = [(primary, SOURCE_PRIMARY)]
        for cellprob, flow in self._passes:
            try:
                stack.append(
                    (self._eval(network, image, cellprob, flow, messages), SOURCE_ENSEMBLE)
                )
            except Exception as exc:  # noqa: BLE001 - one failed pass is not fatal
                _LOG.warning("fallback pass failed, continuing: %s", exc)

        merged, sources = merge_labelled(
            stack,
            overlap=self.cfg.ensemble_merge_overlap,
            min_fragment_px=self.cfg.ensemble_min_fragment_px,
        )
        return merged, note, sources

    def segment_crop(
        self, crop: np.ndarray, override: tuple[float, float] | None = None
    ) -> np.ndarray | None:
        """Segment a small region, optionally at different thresholds.

        Cellpose normalises per image, so running on a crop is not merely the
        same computation on fewer pixels: a cell that sits well below the
        brightest structure of the whole field can sit at the top of the range
        of a crop, and become visible to the network without changing any
        threshold. That is why this is a separate entry point rather than a
        parameter on the frame-level call.
        """
        if crop is None or crop.size < 64:
            return None
        cellprob = self.cfg.cellprob_threshold
        flow = self.cfg.flow_threshold
        if override is not None:
            cellprob, flow = override
        network = self._network("2D")
        messages: list[str] = []
        with _capture_cellpose_messages(messages):
            result = network.eval(
                np.ascontiguousarray(crop),
                channels=list(self.cfg.channels),
                diameter=self.cfg.diameter,
                cellprob_threshold=cellprob,
                flow_threshold=flow,
                normalize=self.cfg.normalize_argument(),
            )
        return np.asarray(result[0]).astype(np.int32)

    # -- 3-D -------------------------------------------------------------------

    def _anisotropy(self) -> float:
        anisotropy = self.scale.anisotropy if self.scale is not None else None
        if anisotropy is None:
            # Cellpose 3 reads anisotropy=None as 1.0, i.e. as Z sampled like
            # XY, which the contract forbids assuming.
            raise UnsupportedStackError(
                "3-D segmentation needs the Z step and the pixel size, because the model "
                "must know how far apart the slices are; neither may be assumed equal to "
                "the other. Enter the Z step (and pixel size) in the calibration settings."
            )
        return float(anisotropy)

    def segment_volume(self, volume: np.ndarray) -> tuple[np.ndarray, str]:
        """Run Cellpose in 3-D on one ``(Z, Y, X)`` volume.

        Only a model validated for 3-D, or a developer override, gets here;
        the production registry has none, so production raises
        ModelUnavailable before anything loads.  Arguments are those of
        Cellpose 3.1.1.3's ``CellposeModel.eval``: ``do_3D=True`` with
        ``z_axis=0`` and the anisotropy from the calibrated scale.  The
        channel was chosen at import, so the volume is one grayscale channel
        and ``channels=[0, 0]``.
        """
        volume = np.asarray(volume)
        if volume.ndim != 3:
            raise ValueError(f"segment_volume takes one (Z, Y, X) volume, got {volume.shape}")
        self._resolve("3D")
        anisotropy = self._anisotropy()
        network = self._network("3D")
        messages: list[str] = []
        with _capture_cellpose_messages(messages):
            result = network.eval(
                volume,
                channels=[0, 0],
                z_axis=0,
                do_3D=True,
                anisotropy=anisotropy,
                diameter=self.cfg.diameter,
                cellprob_threshold=self.cfg.cellprob_threshold,
                flow_threshold=self.cfg.flow_threshold,
                normalize=self.cfg.normalize_argument(),
            )
        masks = np.asarray(result[0]).astype(np.int32)
        # Cellpose squeezes its output, which drops a length-1 Z or Y.
        if masks.shape != volume.shape and masks.size == volume.size:
            masks = masks.reshape(volume.shape)
        if masks.shape != volume.shape:
            raise CellposeUnavailableError(
                f"Cellpose returned 3-D masks of shape {masks.shape} for a volume of {volume.shape}."
            )
        note = next((m for m in messages if _NO_MASKS_RE.search(m)), "")
        return masks, note

    # -- whole stacks ------------------------------------------------------------

    def segment(
        self,
        stack: np.ndarray,
        *,
        progress: Callable[[int, int], bool] | None = None,
    ) -> SegmentationOutput:
        """The ``interfaces.Segmenter`` entry point; see :meth:`run_stack`."""
        return self.run_stack(stack, progress=progress)

    def run_stack(
        self,
        stack: np.ndarray,
        *,
        progress: Callable[[int, int], bool] | None = None,
    ) -> SegmentationOutput:
        """Segment a ``(T, Y, X)`` or ``(T, Z, Y, X)`` stack. ``progress`` may return False to cancel.

        The dimensionality is the array's: :func:`imaging.load_stack` always
        returns ``T[Z]YX``, so a 4-D array is a 3-D time-lapse, never a 2-D
        one with a channel.  The model is resolved before any pixel is read,
        so a missing or mismatching model (or a 3-D stack and the 2-D-only
        production model) fails at once.
        """
        stack = np.asarray(stack)
        if stack.ndim == 4:
            return self._run_3d(stack, progress)
        if stack.ndim != 3:
            raise ValueError(f"run_stack takes a (T, Y, X) or (T, Z, Y, X) stack, got {stack.shape}")
        resolved = self._resolve("2D")

        n = int(stack.shape[0])
        raw_frames: list[np.ndarray] = []
        kept_frames: list[np.ndarray] = []
        diagnostics: list[FrameDiagnostics] = []
        detections: list[Detection] = []

        for t in range(n):
            raw, note, raw_sources = self.segment_frame(stack[t])
            filtered = filter_instances(raw, self.cfg)
            raw_frames.append(raw)
            kept_frames.append(filtered.mask)
            diagnostics.append(_diagnostics(t, filtered, note))
            frame_detections = extract_detections(filtered.mask, t, intensity=stack[t])
            for det in frame_detections:
                if raw_sources.get(filtered.label_map.get(det.label, -1)) == SOURCE_ENSEMBLE:
                    # Found only by a more permissive pass.  It is a real
                    # candidate, but it has not cleared the bar the primary
                    # settings set, so it is marked and the tracker is left to
                    # decide whether it belongs to a trajectory.
                    det.source = SOURCE_ENSEMBLE
                    det.confidence = 0.75
            detections.extend(frame_detections)
            if progress is not None and progress(t + 1, n) is False:
                raise KeyboardInterrupt("Segmentation cancelled.")

        return self._output(
            resolved, raw_frames, kept_frames, detections, diagnostics,
            spatial=stack.shape[1:], passes=1 + len(self._passes), dimensionality="2D",
        )

    def _run_3d(
        self, stack: np.ndarray, progress: Callable[[int, int], bool] | None
    ) -> SegmentationOutput:
        resolved = self._resolve("3D")
        self._anisotropy()
        spacing = self.scale.spacing_zyx_um if self.scale is not None else None
        single = (
            "The fallback rungs were measured on 2-D frames only; the 3-D stack was "
            "segmented in a single pass."
        )
        if self._passes and single not in self.notes:
            self.notes.append(single)
        n = int(stack.shape[0])
        raw_frames: list[np.ndarray] = []
        kept_frames: list[np.ndarray] = []
        diagnostics: list[FrameDiagnostics] = []
        detections: list[Detection] = []
        for t in range(n):
            raw, note = self.segment_volume(stack[t])
            filtered = filter_instances(raw, self.cfg)
            raw_frames.append(raw)
            kept_frames.append(filtered.mask)
            diagnostics.append(_diagnostics(t, filtered, note))
            detections.extend(
                extract_detections_3d(filtered.mask, t, intensity=stack[t], spacing_zyx_um=spacing)
            )
            if progress is not None and progress(t + 1, n) is False:
                raise KeyboardInterrupt("Segmentation cancelled.")
        return self._output(
            resolved, raw_frames, kept_frames, detections, diagnostics,
            spatial=stack.shape[1:], passes=1, dimensionality="3D",
        )

    def _output(
        self,
        resolved: ResolvedModel,
        raw_frames: list[np.ndarray],
        kept_frames: list[np.ndarray],
        detections: list[Detection],
        diagnostics: list[FrameDiagnostics],
        *,
        spatial: tuple[int, ...],
        passes: int,
        dimensionality: str,
    ) -> SegmentationOutput:
        empty = np.zeros((0, *spatial), np.int32)
        return SegmentationOutput(
            masks=np.stack(kept_frames).astype(np.int32) if kept_frames else empty,
            raw_masks=np.stack(raw_frames).astype(np.int32) if raw_frames else empty.copy(),
            detections=detections,
            diagnostics=diagnostics,
            model_path=str(resolved.path),
            model_sha256=resolved.sha256,
            cellpose_version=cellpose_version(),
            used_gpu=self._used_gpu,
            passes_per_frame=passes,
            unavailable_models=[],
            model=resolved,
            provenance=PROVENANCE_MODEL,
            dimensionality=dimensionality,
            notes=list(self.notes),
        )


def _diagnostics(frame: int, filtered: FilterResult, note: str) -> FrameDiagnostics:
    return FrameDiagnostics(
        frame=frame,
        raw_count=filtered.raw_count,
        kept_count=filtered.kept_count,
        removed_count=filtered.raw_count - filtered.kept_count,
        removed_extents=filtered.removed_extents,
        removed_areas=filtered.removed_areas,
        cellpose_message=note,
    )


# --------------------------------------------------------------------------
# Imported label images
# --------------------------------------------------------------------------

_INT32_MAX = int(np.iinfo(np.int32).max)


def _as_labels(arr: np.ndarray, name: str) -> np.ndarray:
    """An int32 instance-label array, or a refusal that says what is wrong."""
    if arr.dtype == bool:
        raise UnsupportedStackError(
            f"{name} is a binary mask, not a label image: touching cells would be one "
            "object. Export instance labels (one integer per cell) instead."
        )
    if np.issubdtype(arr.dtype, np.integer):
        values = arr
    elif np.issubdtype(arr.dtype, np.floating):
        # Fiji saves labels as 32-bit float; accept them only if every value
        # is a whole number, because a fractional "label" is not one.
        if not np.all(np.isfinite(arr)) or not np.array_equal(arr, np.round(arr)):
            raise UnsupportedStackError(
                f"{name} holds non-integer values, so it is not a label image."
            )
        values = arr
    else:
        raise UnsupportedStackError(f"{name} has pixel type {arr.dtype}; labels must be integers.")
    if values.size and float(values.min()) < 0:
        raise UnsupportedStackError(f"{name} holds negative values, so it is not a label image.")
    if values.size and float(values.max()) > _INT32_MAX:
        raise UnsupportedStackError(f"{name} holds labels above {_INT32_MAX}.")
    return np.ascontiguousarray(values, dtype=np.int32)


def load_label_stack(
    path: str | Path,
    axes: str,
    *,
    image: np.ndarray | None = None,
    expected_shape: tuple[int, ...] | None = None,
    scale: "Scale | None" = None,
    label_axes: str | None = None,
    progress: Callable[[int, int], bool] | None = None,
) -> SegmentationOutput:
    """Use an integer label TIFF as the segmentation, instead of running a model.

    ``axes`` is the canonical order of the image the labels belong to
    (``metadata.axes``: ``YX``, ``TYX``, ``ZYX`` or ``TZYX``).  The labels
    must match the image's ``T[Z]YX`` shape exactly -- pass ``image`` (the
    loaded stack, which also supplies intensities) or ``expected_shape`` --
    and a mismatch is refused rather than cropped or padded.  The bare
    two-argument call ``load_label_stack(path, axes)`` works but can check
    only the dimensionality, and says so in ``notes``; production should
    always pass ``image``.  A label file
    whose own metadata establishes its axes is read by them; one that does
    not (the usual case) is read in ``label_axes``, or else in ``axes``.

    Labels are measured as they are: no post-filter, because these are the
    user's objects, not a model's candidates.  3-D detections get µm values
    only when ``scale`` has a calibrated Z step and pixel size.
    """
    path = Path(path)
    if image is not None:
        expected: tuple[int, ...] | None = tuple(int(n) for n in np.shape(image))
    elif expected_shape is not None:
        expected = tuple(int(n) for n in expected_shape)
    else:
        expected = None
    if str(axes).upper() not in CANONICAL_AXES:
        raise ValueError(f"axes must be one of {CANONICAL_AXES}, got {axes!r}")

    arr, label_canonical, notes = read_label_array(path, axes, label_axes=label_axes)
    if expected is None:
        # The two-argument form of the interface: nothing to compare sizes
        # with, so at least the dimensionality must be the image's -- a 2-D
        # label movie for a Z stack is the T/Z confusion in another form.
        rank = 4 if "Z" in str(axes).upper() else 3
        if arr.ndim != rank:
            raise UnsupportedStackError(
                f"The label image {path.name} reads as '{label_canonical}' {arr.shape}, but "
                f"the image is '{str(axes).upper()}'. Labels must have the image's dimensions."
            )
        notes.append(
            "The label image was not compared with the image's size (no image was given); "
            "only its dimensionality was checked."
        )
    elif arr.shape != expected:
        raise UnsupportedStackError(
            f"The label image {path.name} is {arr.shape} (read as '{label_canonical}') but "
            f"the image is {expected} ('{str(axes).upper()}'). Labels must match the image "
            "frame for frame and plane for plane."
        )
    labels = _as_labels(arr, path.name)
    dimensionality = "3D" if labels.ndim == 4 else "2D"
    spacing = scale.spacing_zyx_um if (scale is not None and dimensionality == "3D") else None
    if dimensionality == "3D" and spacing is None:
        notes.append(
            "The Z step or the pixel size is not calibrated, so imported 3-D objects are "
            "measured in voxels only."
        )
    notes.append(f"The segmentation was imported from {path.name}; no model ran.")

    n = int(labels.shape[0])
    detections: list[Detection] = []
    diagnostics: list[FrameDiagnostics] = []
    for t in range(n):
        frame = labels[t]
        intensity = None if image is None else np.asarray(image[t])
        if dimensionality == "3D":
            found = extract_detections_3d(frame, t, intensity=intensity, spacing_zyx_um=spacing)
        else:
            found = extract_detections(frame, t, intensity=intensity)
        count = int(np.count_nonzero(np.unique(frame)))
        diagnostics.append(FrameDiagnostics(frame=t, raw_count=count, kept_count=count))
        detections.extend(found)
        if progress is not None and progress(t + 1, n) is False:
            raise KeyboardInterrupt("Label import cancelled.")

    return SegmentationOutput(
        masks=labels,
        raw_masks=labels,
        detections=detections,
        diagnostics=diagnostics,
        model_path="",
        model_sha256=None,
        cellpose_version="",
        used_gpu=False,
        passes_per_frame=0,
        unavailable_models=[],
        model=None,
        provenance=PROVENANCE_IMPORTED,
        labels_sha256=model_registry.sha256_file(path),
        labels_path=str(path.resolve()),
        dimensionality=dimensionality,
        notes=notes,
    )
