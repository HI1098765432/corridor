"""Cellpose v3 segmentation and the measurable post-filter.

This module deliberately keeps two numbers apart for every frame: how many
instances Cellpose produced, and how many survived post-processing.  Without
that separation "the cell disappeared" is unattributable, and the temptation
is to fix the tracker for a problem that lives in segmentation.
"""

from __future__ import annotations

import hashlib
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from .config import SegmentationConfig
from .detections import (
    SOURCE_ENSEMBLE,
    SOURCE_PRIMARY,
    Detection,
    FrameDiagnostics,
    extract_detections,
)

_LOG = logging.getLogger(__name__)

#: Cellpose emits this when the flow field contains no basins at all.
_NO_MASKS_RE = re.compile(r"no (seeds|masks) found", re.IGNORECASE)


class CellposeUnavailableError(RuntimeError):
    """Raised when the Cellpose runtime or the model file cannot be used."""


def cellpose_version() -> str:
    try:
        import cellpose

        return str(getattr(cellpose, "version", "unknown"))
    except Exception:  # pragma: no cover - only on a broken install
        return "unavailable"


def require_cellpose_v3() -> str:
    """Confirm a Cellpose 3.x runtime, which the custom model requires."""
    version = cellpose_version()
    major = version.split(".")[0]
    if not major.isdigit() or int(major) != 3:
        raise CellposeUnavailableError(
            f"This analysis needs Cellpose version 3, but version {version} is installed. "
            "The supplied custom model was trained with Cellpose 3 and will not behave "
            "the same way under version 4."
        )
    return version


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
    """
    mask = np.asarray(mask)
    if mask.size == 0:
        return FilterResult(mask.astype(np.int32), [], [], 0, 0, {})
    mask = mask.astype(np.int32, copy=False)
    labels = np.unique(mask)
    labels = labels[labels != 0]
    if labels.size == 0:
        return FilterResult(np.zeros_like(mask, dtype=np.int32), [], [], 0, 0, {})

    h, w = mask.shape[:2]
    keep: list[int] = []
    removed_extents: list[int] = []
    removed_areas: list[float] = []

    for lab in labels:
        ys, xs = np.where(mask == lab)
        if ys.size == 0:
            continue
        bh = int(ys.max() - ys.min() + 1)
        bw = int(xs.max() - xs.min() + 1)
        extent = max(bh, bw)
        area = float(ys.size)
        drop = extent < cfg.min_extent_px or (cfg.min_area_px and area < cfg.min_area_px)
        if not drop and cfg.drop_border_touching:
            if ys.min() == 0 or xs.min() == 0 or ys.max() >= h - 1 or xs.max() >= w - 1:
                drop = True
        if drop:
            removed_extents.append(extent)
            removed_areas.append(area)
        else:
            keep.append(int(lab))

    out = np.zeros_like(mask, dtype=np.int32)
    label_map: dict[int, int] = {}
    for new_id, lab in enumerate(keep, start=1):
        out[mask == lab] = new_id
        label_map[new_id] = lab
    return FilterResult(
        out, removed_extents, removed_areas, int(labels.size), len(keep), label_map
    )


#: Cellpose checkpoints for this architecture are tens of megabytes; anything
#: far smaller sharing the folder is not one.
MIN_MODEL_BYTES = 1 << 20
#: A sanity bound on how much of a directory tree discovery will walk.
MAX_COMPANION_DIRS = 64


def discover_companion_models(primary: str | Path | None) -> tuple[str, ...]:
    """Find sibling Cellpose models beside the one being used.

    The supplied training folder holds three models: one trained on both halves
    of the data and one on each half alone.  A rung that runs "every available
    model" should mean the ones actually present on this machine, not a list
    baked in at build time -- the installed application ships only the combined
    model, so that list would be wrong for most users and right for none.

    Returns paths only; whether any of them loads is decided later, because a
    file that exists and a model that works are different questions.
    """
    if not primary:
        return ()
    path = Path(primary)
    if not path.is_file():
        return ()

    # .../<SomeModel>/models/<file> -> look in the sibling <*>/models/ folders.
    models_dir = path.parent
    if models_dir.name != "models":
        return ()
    root = models_dir.parent.parent
    # Refuse to search a filesystem root. Without this guard a model path that
    # happens to be two levels down from a drive letter turns this into a scan
    # of the entire disk, which is slow, and which returns files that are not
    # models but merely live in a directory called "models".
    if root == root.parent or not root.is_dir():
        return ()

    found: list[str] = []
    for sibling in sorted(root.iterdir())[:MAX_COMPANION_DIRS]:
        candidate_dir = sibling / "models"
        if not candidate_dir.is_dir():
            continue
        for candidate in sorted(candidate_dir.iterdir()):
            if not candidate.is_file() or candidate.suffix:
                # Cellpose writes its checkpoints with no extension; anything
                # with one beside them is a training log or a label file.
                continue
            try:
                if candidate.stat().st_size < MIN_MODEL_BYTES:
                    continue
                if candidate.samefile(path):
                    continue
            except OSError:
                continue
            found.append(str(candidate))
    return tuple(found)



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


# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------


@dataclass
class SegmentationOutput:
    masks: np.ndarray  # (T, Y, X) int32, post-filter
    raw_masks: np.ndarray  # (T, Y, X) int32, straight from Cellpose
    detections: list[Detection]
    diagnostics: list[FrameDiagnostics]
    model_path: str
    model_sha256: str | None
    cellpose_version: str
    used_gpu: bool
    #: Segmentation passes that actually ran per frame. This is not always the
    #: configured cost: a companion model that cannot be loaded is skipped, so
    #: a rung asking for three passes may have run one. Reporting the request
    #: as though it were the work is how a run gets described as something it
    #: was not.
    passes_per_frame: int = 1
    #: Companion models the configuration asked for and that could not be used.
    unavailable_models: list[str] = field(default_factory=list)


class SegmentationService:
    """Loads the model once and segments frames on demand."""

    def __init__(self, cfg: SegmentationConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._model_path: str | None = None
        self._used_gpu = False
        #: Companion models for the fallback rungs, loaded lazily and only if a
        #: rung actually asks for one, keyed by path so a model shared by
        #: several passes is loaded once.
        self._extra_models: dict[str, Any] = {}
        #: Paths a rung asked for and could not load.  Recorded rather than
        #: raised: an optional extra model that is absent should cost some
        #: recall, not the whole analysis.
        self.unavailable_models: list[str] = []

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    def _load(self) -> None:
        require_cellpose_v3()
        from cellpose import models

        target = self.cfg.resolved_model()
        want_gpu = bool(self.cfg.use_gpu and gpu_available())
        try:
            if self.cfg.use_custom_model and self.cfg.model_path:
                path = Path(self.cfg.model_path)
                if not path.exists():
                    raise CellposeUnavailableError(
                        f"The Cellpose model file was not found:\n{path}"
                    )
                self._model = models.CellposeModel(
                    pretrained_model=str(path), gpu=want_gpu
                )
            else:
                self._model = models.CellposeModel(model_type=target, gpu=want_gpu)
        except CellposeUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            raise CellposeUnavailableError(
                f"The Cellpose model could not be loaded.\n\n{exc}"
            ) from exc
        self._model_path = target
        self._used_gpu = want_gpu

    def _extra_model(self, path: str):
        """Load a companion model, or return None if it cannot be used."""
        if path in self._extra_models:
            return self._extra_models[path]
        if path in self.unavailable_models:
            return None
        try:
            from cellpose import models

            if not Path(path).exists():
                raise FileNotFoundError(path)
            model = models.CellposeModel(
                pretrained_model=str(path),
                gpu=bool(self.cfg.use_gpu and gpu_available()),
            )
        except Exception as exc:  # noqa: BLE001 - degrade, do not abort
            _LOG.warning(
                "fallback model unavailable, continuing without it: %s (%s)", path, exc
            )
            self.unavailable_models.append(path)
            return None
        self._extra_models[path] = model
        return model

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
        ``primary``.  With it on, the extra passes are merged in afterwards and
        anything only they found is tagged, so a reader can always separate an
        ordinary detection from one that needed a more permissive setting to
        appear at all.
        """
        messages: list[str] = []
        primary = self._eval(
            self.model,
            image,
            self.cfg.cellprob_threshold,
            self.cfg.flow_threshold,
            messages,
        )
        note = next((m for m in messages if _NO_MASKS_RE.search(m)), "")

        extra_passes = self.cfg.ensemble_passes()
        if not extra_passes:
            return primary, note, {
                int(v): SOURCE_PRIMARY for v in np.unique(primary) if v
            }

        stack: list[tuple[np.ndarray, str]] = [(primary, SOURCE_PRIMARY)]
        for path, cellprob, flow in extra_passes:
            model = self.model if path is None else self._extra_model(path)
            if model is None:
                continue
            try:
                stack.append(
                    (self._eval(model, image, cellprob, flow, messages), SOURCE_ENSEMBLE)
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
        messages: list[str] = []
        with _capture_cellpose_messages(messages):
            result = self.model.eval(
                np.ascontiguousarray(crop),
                channels=list(self.cfg.channels),
                diameter=self.cfg.diameter,
                cellprob_threshold=cellprob,
                flow_threshold=flow,
                normalize=self.cfg.normalize_argument(),
            )
        return np.asarray(result[0]).astype(np.int32)

    def run_stack(
        self,
        stack: np.ndarray,
        *,
        progress: Callable[[int, int], bool] | None = None,
    ) -> SegmentationOutput:
        """Segment every frame. ``progress`` may return False to cancel."""
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
            diagnostics.append(
                FrameDiagnostics(
                    frame=t,
                    raw_count=filtered.raw_count,
                    kept_count=filtered.kept_count,
                    removed_count=filtered.raw_count - filtered.kept_count,
                    removed_extents=filtered.removed_extents,
                    removed_areas=filtered.removed_areas,
                    cellpose_message=note,
                )
            )
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

        model_path = self._model_path or self.cfg.resolved_model()
        sha = None
        if self.cfg.use_custom_model and self.cfg.model_path:
            try:
                sha = file_sha256(self.cfg.model_path)
            except OSError:
                sha = None

        # A companion pass that never loaded did not run, however it was
        # configured. Counting the successful ones is the only honest answer.
        ran = 1 + sum(
            1
            for path, _, _ in self.cfg.ensemble_passes()
            if path is None or path not in self.unavailable_models
        )

        return SegmentationOutput(
            masks=np.stack(kept_frames).astype(np.int32) if kept_frames else np.zeros((0, 0, 0), np.int32),
            raw_masks=np.stack(raw_frames).astype(np.int32) if raw_frames else np.zeros((0, 0, 0), np.int32),
            detections=detections,
            diagnostics=diagnostics,
            model_path=model_path,
            model_sha256=sha,
            cellpose_version=cellpose_version(),
            used_gpu=self._used_gpu,
            passes_per_frame=ran,
            unavailable_models=list(self.unavailable_models),
        )
