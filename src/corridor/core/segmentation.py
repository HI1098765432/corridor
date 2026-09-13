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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from .config import SegmentationConfig
from .detections import Detection, FrameDiagnostics, extract_detections

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


def filter_instances(mask: np.ndarray, cfg: SegmentationConfig) -> FilterResult:
    """Drop instances that are too small to be a cell, and relabel 1..K.

    The research notebook's rule -- keep an instance when the larger dimension
    of its bounding box is at least ``min_extent`` pixels -- is preserved
    exactly, so results stay comparable.  What changes is that everything it
    removes is now counted.
    """
    mask = np.asarray(mask)
    if mask.size == 0:
        return FilterResult(mask.astype(np.int32), [], [], 0, 0)
    mask = mask.astype(np.int32, copy=False)
    labels = np.unique(mask)
    labels = labels[labels != 0]
    if labels.size == 0:
        return FilterResult(np.zeros_like(mask, dtype=np.int32), [], [], 0, 0)

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
    for new_id, lab in enumerate(keep, start=1):
        out[mask == lab] = new_id
    return FilterResult(out, removed_extents, removed_areas, int(labels.size), len(keep))


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


class SegmentationService:
    """Loads the model once and segments frames on demand."""

    def __init__(self, cfg: SegmentationConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._model_path: str | None = None
        self._used_gpu = False

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

    def segment_frame(self, image: np.ndarray) -> tuple[np.ndarray, str]:
        """Run Cellpose on one 2-D frame; returns (raw label image, message)."""
        messages: list[str] = []
        with _capture_cellpose_messages(messages):
            result = self.model.eval(
                image,
                channels=list(self.cfg.channels),
                diameter=self.cfg.diameter,
                cellprob_threshold=self.cfg.cellprob_threshold,
                flow_threshold=self.cfg.flow_threshold,
                normalize=self.cfg.normalize,
            )
        mask = np.asarray(result[0]).astype(np.int32)
        note = next((m for m in messages if _NO_MASKS_RE.search(m)), "")
        return mask, note

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
            raw, note = self.segment_frame(stack[t])
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
            detections.extend(extract_detections(filtered.mask, t, intensity=stack[t]))
            if progress is not None and progress(t + 1, n) is False:
                raise KeyboardInterrupt("Segmentation cancelled.")

        model_path = self._model_path or self.cfg.resolved_model()
        sha = None
        if self.cfg.use_custom_model and self.cfg.model_path:
            try:
                sha = file_sha256(self.cfg.model_path)
            except OSError:
                sha = None

        return SegmentationOutput(
            masks=np.stack(kept_frames).astype(np.int32) if kept_frames else np.zeros((0, 0, 0), np.int32),
            raw_masks=np.stack(raw_frames).astype(np.int32) if raw_frames else np.zeros((0, 0, 0), np.int32),
            detections=detections,
            diagnostics=diagnostics,
            model_path=model_path,
            model_sha256=sha,
            cellpose_version=cellpose_version(),
            used_gpu=self._used_gpu,
        )
