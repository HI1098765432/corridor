"""``run_analysis`` end to end through the REAL ``recover`` and ``collect_issues``.

The review of E2 found that every analysis raised TypeError: the pipeline
still called ``recover`` and ``collect_issues`` with the v1 axis argument after
both had dropped it, and the only test that ran ``run_analysis`` needed the
real data and was skipped, so the suite stayed green.  This one needs nothing
but numpy: a stand-in segmentation service returns known labels for the
frames and segments recovery's crops by thresholding.  Recovery and QC are the
real code, so any drift between the pipeline's calls and either signature
fails here, whichever pipeline (v1 transition or 2.0) is on the branch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy import ndimage

from corridor.core import model_registry, pipeline
from corridor.core.config import RunConfig
from corridor.core.detections import FrameDiagnostics, extract_detections
from corridor.core.model_registry import ResolvedModel
from corridor.core.qc import Issue
from corridor.core.recovery import SOURCE_WINDOW
from corridor.core.segmentation import PROVENANCE_MODEL, SegmentationOutput

PIXEL_UM = 0.5
FRAME_S = 600.0  # 10 min
N_FRAMES = 7
HEIGHT, WIDTH = 320, 200
BACKGROUND, RIDGE, CELL = 1000.0, 1600.0, 3000.0
#: Bright vertical ridges, one lane each (as in the supplied fields, where
#: the lumen is the bright line); dimmer than a cell, so a threshold between
#: the two finds the cell alone, as the network does on real data.
RIDGES_X = (60, 140)
#: The cell: centre x (inside lane 0), first y, px per frame. It is missing
#: from the labels at MISSING but present in the movie, so recovery has a
#: cell to find.
CELL_X, CELL_Y0, CELL_STEP = 61.0, 40.0, 25.0
MISSING = 3


def _ellipse(cx: float, cy: float, a: float = 30.0, b: float = 5.5) -> np.ndarray:
    rows, cols = np.ogrid[:HEIGHT, :WIDTH]
    return ((cols - cx) / b) ** 2 + ((rows - cy) / a) ** 2 <= 1.0


def synthetic_movie(tmp_path: Path) -> tuple[Path, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = np.zeros((N_FRAMES, HEIGHT, WIDTH), np.int32)
    movie = rng.normal(BACKGROUND, 20.0, size=(N_FRAMES, HEIGHT, WIDTH)).astype(np.float32)
    for x in RIDGES_X:
        movie[:, :, x - 2 : x + 3] = RIDGE
    for t in range(N_FRAMES):
        mask = _ellipse(CELL_X, CELL_Y0 + CELL_STEP * t)
        movie[t][mask] = CELL
        if t != MISSING:
            labels[t][mask] = 1
    path = tmp_path / "movie.tif"
    tifffile.imwrite(
        path, movie.astype(np.uint16), imagej=True,
        metadata={"axes": "TYX", "finterval": FRAME_S, "unit": "micron"},
        resolution=(1.0 / PIXEL_UM, 1.0 / PIXEL_UM),
    )
    return path, labels


class StandInService:
    """Returns the given labels for whole frames; thresholds recovery's crops.

    The crop threshold sits between the ridges and the cell, so a crop
    holding both returns only the cell.
    """

    def __init__(self, labels: np.ndarray) -> None:
        self.labels = labels
        self.crops = 0

    def run_stack(self, stack, progress=None):
        detections, diagnostics = [], []
        for t in range(stack.shape[0]):
            found = extract_detections(self.labels[t], t, intensity=stack[t])
            detections.extend(found)
            diagnostics.append(FrameDiagnostics(t, len(found), len(found)))
        spec = model_registry.production_spec("2D")
        resolved = ResolvedModel(spec=spec, path=Path("models") / spec.filename, sha256=spec.sha256)
        return SegmentationOutput(
            masks=self.labels.copy(), raw_masks=self.labels.copy(), detections=detections,
            diagnostics=diagnostics, model_path=str(resolved.path),
            model_sha256=resolved.sha256, cellpose_version="stand-in", used_gpu=False,
            model=resolved, provenance=PROVENANCE_MODEL,
        )

    def segment_crop(self, crop, override=None):
        self.crops += 1
        labels, _ = ndimage.label(crop > 0.5 * (RIDGE + CELL))
        return labels.astype(np.int32)


def test_an_analysis_with_recovery_on_completes_through_the_real_recover_and_qc(
    tmp_path, monkeypatch
):
    movie, labels = synthetic_movie(tmp_path)
    service = StandInService(labels)
    monkeypatch.setattr(pipeline, "SegmentationService", lambda *a, **k: service)
    # The manifest asks for the Cellpose and torch versions; importing torch
    # costs a gigabyte of RAM and says nothing about recovery or QC.
    monkeypatch.setattr(pipeline, "cellpose_version", lambda: "stand-in", raising=False)
    monkeypatch.setattr(pipeline, "gpu_available", lambda: False, raising=False)

    config = RunConfig(input_path=str(movie), output_dir=str(tmp_path / "out"))
    assert config.recovery.enabled, "the shipped default runs recovery"
    result = pipeline.run_analysis(config)
    out = Path(result.output_dir)

    # recover() ran for real and found the cell the labels left out.
    assert service.crops >= 1
    assert result.recovery is not None
    found = [a for a in result.recovery.attempts if a.found]
    assert [(a.frame, a.source) for a in found] == [(MISSING, SOURCE_WINDOW)]
    assert found[0].bracket is not None and found[0].bracket[0] == MISSING - 1

    # Re-tracked: one track through the recovered frame, and every attempt
    # names a track that exists in this result.
    assert result.n_tracks == 1
    assert [o.frame for o in result.tracks[0].observations] == list(range(N_FRAMES))
    ids = {t.id for t in result.tracks}
    assert all(a.track_id in ids for a in result.recovery.attempts)

    # collect_issues() ran for real; the bridged gap no longer exists.
    assert all(isinstance(i, Issue) for i in result.issues)
    assert "likely_missed_detection" not in {i.code for i in result.issues}
    for name in (pipeline.F_QC, pipeline.F_RECOVERY, pipeline.F_TRACKS, pipeline.F_MANIFEST):
        assert (out / name).exists(), name


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
