"""Reading a saved analysis back from disk.

The point of this module: reopening yesterday's work must not rerun Cellpose.
Everything the results workspace needs is reconstructed from the files the run
wrote, so opening a project is an I/O operation, not a computation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core import pipeline
from ..core.export import load_masks

_NUMERIC = {
    "frame", "track_id", "source_frame", "det_label", "channel", "gap_frames",
    "observation_index", "n_observations", "label", "extent_px",
    "bbox_min_x", "bbox_min_y", "bbox_max_x", "bbox_max_y",
    "raw_instances", "kept_instances", "removed_instances",
    "first_frame", "last_frame", "first_source_frame", "last_source_frame",
    "n_gaps", "total_missing_frames", "span_frames",
}


def _coerce(key: str, value: str) -> Any:
    if value == "" or value is None:
        return None
    if key in _NUMERIC:
        try:
            return int(float(value))
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return value


def read_table(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return [
            {k: _coerce(k, v) for k, v in row.items()}
            for row in csv.DictReader(fh)
        ]


@dataclass
class SavedAnalysis:
    """A completed run, loaded from its directory."""

    directory: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    tracks: list[dict[str, Any]] = field(default_factory=list)
    detections: list[dict[str, Any]] = field(default_factory=list)
    summaries: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    unlinked: list[dict[str, Any]] = field(default_factory=list)
    _masks: np.ndarray | None = None

    # -- lazy pixel data ---------------------------------------------------
    @property
    def masks(self) -> np.ndarray | None:
        """Label stack, loaded on first use and then cached."""
        if self._masks is None:
            path = self.directory / pipeline.F_MASKS
            if path.exists():
                self._masks = load_masks(path)
        return self._masks

    def release_masks(self) -> None:
        self._masks = None

    # -- convenience -------------------------------------------------------
    @property
    def n_frames(self) -> int:
        shape = self.manifest.get("input", {}).get("shape_tyx")
        return int(shape[0]) if shape else 0

    @property
    def source_path(self) -> Path | None:
        raw = self.manifest.get("input", {}).get("path")
        return Path(raw) if raw else None

    @property
    def pixel_size_um(self) -> float | None:
        return self.manifest.get("calibration", {}).get("pixel_size_um")

    @property
    def frame_interval_min(self) -> float | None:
        return self.manifest.get("calibration", {}).get("frame_interval_min")

    @property
    def source_frames(self) -> list[int] | None:
        return self.manifest.get("input", {}).get("source_frames")

    def rows_for_frame(self, frame: int) -> list[dict[str, Any]]:
        return [r for r in self.tracks if r.get("frame") == frame]

    def rows_for_track(self, track_id: int) -> list[dict[str, Any]]:
        return [r for r in self.tracks if r.get("track_id") == track_id]

    def track_ids(self) -> list[int]:
        return sorted({int(r["track_id"]) for r in self.tracks if r.get("track_id")})

    def summary_for(self, track_id: int) -> dict[str, Any] | None:
        for s in self.summaries:
            if s.get("track_id") == track_id:
                return s
        return None

    def diagnostic_for(self, frame: int) -> dict[str, Any] | None:
        for d in self.diagnostics:
            if d.get("frame") == frame:
                return d
        return None


def load_analysis(directory: str | Path) -> SavedAnalysis:
    directory = Path(directory)
    manifest_path = directory / pipeline.F_MANIFEST
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            manifest = {}
    return SavedAnalysis(
        directory=directory,
        manifest=manifest,
        tracks=read_table(directory / pipeline.F_TRACKS),
        detections=read_table(directory / pipeline.F_DETECTIONS),
        summaries=read_table(directory / pipeline.F_SUMMARY),
        diagnostics=read_table(directory / pipeline.F_DIAGNOSTICS),
        issues=read_table(directory / pipeline.F_QC),
        unlinked=read_table(directory / pipeline.F_UNLINKED),
    )


def analysis_is_complete(directory: str | Path) -> bool:
    directory = Path(directory)
    return (directory / pipeline.F_MANIFEST).exists() and (
        directory / pipeline.F_TRACKS
    ).exists()


def export_bundle(analysis: SavedAnalysis, destination: str | Path) -> list[Path]:
    """Copy the result files a user would want to keep, into one folder."""
    import shutil

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    wanted = [
        pipeline.F_TRACKS,
        pipeline.F_SUMMARY,
        pipeline.F_DETECTIONS,
        pipeline.F_DIAGNOSTICS,
        pipeline.F_EVENTS,
        pipeline.F_QC,
        pipeline.F_UNLINKED,
        pipeline.F_MANIFEST,
        pipeline.F_MASKS,
        "preview.png",
    ]
    written: list[Path] = []
    for name in wanted:
        source = analysis.directory / name
        if source.exists():
            target = destination / name
            shutil.copy2(source, target)
            written.append(target)
    return written
