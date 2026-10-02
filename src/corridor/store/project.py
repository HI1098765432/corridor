"""Reading a saved analysis back from disk, and exporting from it.

The point of this module: reopening yesterday's work must not rerun Cellpose.
Everything the results workspace needs is reconstructed from the files the run
wrote, so opening a project is an I/O operation, not a computation.

**Two output schemas.**  Schema 2 (``run.json["schema_version"] == 2``) is
read as written.  Schema 1 is every 1.x release (1.0.0 through 1.3.0), which
wrote no version key; it is upgraded *in memory* when loaded:

*   ``tracks.csv`` is re-measured from its own positions (``frame``,
    ``x_px``, ``y_px``) with the v2 definitions -- per-hour units, MTrackJ
    Len/D2S/D2P, acceleration, turning angle -- and the morphology a v1
    ``detections.csv`` recorded is joined back in by ``(frame, det_label)``;
*   ``track_summary.csv`` and the MSD are recomputed from the same positions;
*   ``unlinked_starts.csv``'s along/across jump is turned back into ``dx/dy``
    with the axis the run recorded (exact: it was a rotation);
*   nothing is ever re-segmented, and nothing on disk is rewritten.

Positions, frames, calibration and flags are the v1 run's own, so every
quantity both schemas define (speed in µm/min, net displacement, path length)
comes out the same as the v1 file said, to its 9-decimal rounding.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core import export, pipeline
from ..core.config import MeasurementConfig, Scale
from ..core.export import load_masks
from ..core.measurements import (
    add_reference_distance,
    frame_rows,
    msd_rows,
    summarise,
)

#: ``track_msd.csv``.  Defined here until pipeline.py (which owns the other
#: file names) writes it; the name is fixed by the 2.0 contract (§6).
F_MSD = getattr(pipeline, "F_MSD", "track_msd.csv")

#: Columns whose cells are whole numbers.  Read with ``int()`` when the text is
#: an integer and kept as a float when it is not -- never truncated.
_INTEGER = {
    "frame", "track_id", "source_frame", "det_label", "channel", "gap_frames",
    "observation_index", "n_observations", "label", "extent_px",
    "bbox_min_x", "bbox_min_y", "bbox_min_z", "bbox_max_x", "bbox_max_y", "bbox_max_z",
    "raw_instances", "kept_instances", "removed_instances",
    "first_frame", "last_frame", "first_source_frame", "last_source_frame",
    "n_gaps", "total_missing_frames", "span_frames",
    "lag_frames", "n_pairs", "msd_fit_lags", "persistence_fit_lags",
    "detections", "candidate_tracks", "matched", "new_tracks", "dormant", "terminated",
    "starts_at_frame", "nearest_earlier_track", "that_track_ended_at_frame",
}

#: Columns that are text whatever they happen to contain.  Without this, a
#: Cellpose message reading "nan" or "inf", a note "1", or the id list "3"
#: came back as a float -- ``float()`` parses all of those.
_TEXT = {
    "cellpose_message", "track_flags", "flags", "detection_source", "source",
    "found_by", "detail", "notes", "merge_suspected_tracks", "split_suspected",
    "gap_closed", "severity", "code", "title", "refused_because", "explanation",
    "dimensionality",
}

#: Written as ``true``/``false`` by ``export._clean``.
_BOOLEAN = {"touches_border", "recovered"}


def _finite_float(value: str) -> float | None:
    """A float, or None for text that is not one of the numbers we write.

    ``export._clean`` never writes NaN or infinity (they become empty cells),
    so a cell that parses to one was text, not a number.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _coerce(key: str, value: str | None) -> Any:
    if value is None or value == "":
        return None
    if key in _TEXT:
        return value
    if key in _BOOLEAN:
        lowered = value.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        return value
    if key in _INTEGER:
        try:
            return int(value)
        except ValueError:
            pass
        number = _finite_float(value)
        if number is None:
            return value  # keep what the file says rather than inventing None
        return int(number) if number.is_integer() else number
    number = _finite_float(value)
    return value if number is None else number


def read_table(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return [
            {k: _coerce(k, v) for k, v in row.items() if k is not None}
            for row in csv.DictReader(fh)
        ]


# --------------------------------------------------------------------------
# Tracks rebuilt from rows
# --------------------------------------------------------------------------


@dataclass
class _SavedObservation:
    """What ``measurements`` reads from an observation, rebuilt from a row."""

    frame: int
    x: float
    y: float
    z: float | None = None
    area_px: float | None = None
    det_label: int | None = None
    channel: int = -1
    gap_frames: int = 0
    cost: float | None = None
    link_margin: float | None = None
    source: str | None = None
    confidence: float | None = None
    detection: Any = None


@dataclass
class _SavedTrack:
    id: int
    observations: list[_SavedObservation] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    channel: int = -1


#: Detection fields ``measurements`` reads for the morphology columns.
_DETECTION_FIELDS = (
    "z", "area_px", "label", "source", "confidence",
    "perimeter_px", "major_axis_px", "minor_axis_px", "aspect_ratio",
    "eccentricity", "orientation_rad", "circularity", "solidity",
    "extent_fraction", "mean_intensity", "volume_vox", "volume_um3",
    "surface_area_um2", "sphericity", "elongation", "flatness",
)


def _detection_view(row: dict[str, Any] | None) -> Any:
    """A read-only stand-in for a Detection, from a ``detections.csv`` row.

    Only the morphology fields are carried; anything the file did not record
    is None, so it is reported as unmeasured rather than as a default.
    """
    if row is None:
        return None
    return types.SimpleNamespace(**{k: row.get(k) for k in _DETECTION_FIELDS})


def tracks_from_rows(
    rows: Sequence[dict[str, Any]],
    detections: Sequence[dict[str, Any]] = (),
    *,
    summaries: Sequence[dict[str, Any]] = (),
) -> list[_SavedTrack]:
    """Rebuild measurable tracks from saved ``tracks.csv`` rows.

    ``fragment`` is dropped from the flags because it is not a tracker
    judgement: measurement adds it from ``min_observations``, and keeping it
    would make a re-measurement with a different minimum disagree with itself.
    """
    by_key = {
        (d.get("frame"), d.get("label")): d
        for d in detections
        if d.get("frame") is not None and d.get("label") is not None
    }
    summary_channel = {
        s.get("track_id"): s.get("channel") for s in summaries if s.get("track_id") is not None
    }
    tracks: dict[int, _SavedTrack] = {}
    for row in rows:
        tid, frame = row.get("track_id"), row.get("frame")
        x, y = row.get("x_px"), row.get("y_px")
        if tid is None or frame is None or x is None or y is None:
            continue
        tid = int(tid)
        track = tracks.get(tid)
        if track is None:
            flags = {
                f for f in str(row.get("track_flags") or "").split(";") if f and f != "fragment"
            }
            channel = summary_channel.get(tid, row.get("channel"))
            track = tracks[tid] = _SavedTrack(
                id=tid, flags=flags, channel=int(channel) if channel is not None else -1
            )
        label = row.get("det_label")
        confidence = row.get("segmentation_confidence", row.get("detection_confidence"))
        track.observations.append(
            _SavedObservation(
                frame=int(frame),
                x=float(x),
                y=float(y),
                z=row.get("z_slice"),
                area_px=row.get("area_px"),
                det_label=label,
                channel=int(row["channel"]) if row.get("channel") is not None else track.channel,
                cost=row.get("match_cost_chi2"),
                link_margin=row.get("link_margin_chi2"),
                # 1.0.0 recorded no provenance; every detection then was primary.
                source=row.get("detection_source") or "primary",
                confidence=1.0 if confidence is None else confidence,
                detection=_detection_view(by_key.get((int(frame), label))),
            )
        )
    for track in tracks.values():
        track.observations.sort(key=lambda o: o.frame)
        previous = None
        for obs in track.observations:
            obs.gap_frames = 0 if previous is None else obs.frame - previous
            previous = obs.frame
    return [tracks[k] for k in sorted(tracks)]


# --------------------------------------------------------------------------
# A saved analysis
# --------------------------------------------------------------------------


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
    #: ``tracking_events.csv`` and ``recovery_attempts.csv`` (not loaded in 1.x).
    events: list[dict[str, Any]] = field(default_factory=list)
    recovery: list[dict[str, Any]] = field(default_factory=list)
    #: ``track_msd.csv``; computed from the positions for a v1 run.
    msd: list[dict[str, Any]] = field(default_factory=list)
    #: The schema the files on disk were *written* in.  1 = any 1.x release.
    schema_version: int = export.SCHEMA_VERSION
    #: "2D" or "3D".
    dimensionality: str = "2D"
    #: Application version that wrote a v1 run that was upgraded on load.
    upgraded_from: str | None = None
    #: What the in-memory upgrade did, for display beside the results.
    upgrade_notes: list[str] = field(default_factory=list)
    _masks: np.ndarray | None = None

    # -- lazy pixel data ---------------------------------------------------
    @property
    def masks(self) -> np.ndarray | None:
        """Label stack ``T[Z]YX``, loaded on first use and then cached."""
        if self._masks is None:
            path = self.directory / pipeline.F_MASKS
            if path.exists():
                self._masks = load_masks(path)
        return self._masks

    def release_masks(self) -> None:
        self._masks = None

    # -- convenience -------------------------------------------------------
    @property
    def axes(self) -> str:
        """Canonical axis order of the analysed stack (v1 was always TYX)."""
        return str(self.manifest.get("input", {}).get("axes") or "TYX")

    @property
    def n_frames(self) -> int:
        """Time points, from ``input.shape`` + ``input.axes`` or v1's ``shape_tyx``."""
        info = self.manifest.get("input", {})
        shape = info.get("shape")
        axes = info.get("axes")
        if shape and axes and len(shape) == len(axes):
            return int(shape[axes.index("T")]) if "T" in axes else 1
        legacy = info.get("shape_tyx")
        if legacy:
            return int(legacy[0])
        frames = [r.get("frame") for r in self.diagnostics + self.tracks]
        frames = [f for f in frames if isinstance(f, int)]
        return max(frames) + 1 if frames else 0

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
    def z_step_um(self) -> float | None:
        return self.manifest.get("calibration", {}).get("z_step_um")

    @property
    def scale(self) -> Scale:
        return Scale.from_values(self.pixel_size_um, self.frame_interval_min, self.z_step_um)

    @property
    def source_frames(self) -> list[int] | None:
        return self.manifest.get("input", {}).get("source_frames")

    @property
    def reference_point_px(self) -> tuple[float, ...] | None:
        ref = (self.manifest.get("measurement") or {}).get("reference_point_px")
        return tuple(float(v) for v in ref) if ref else None

    def rows_for_frame(self, frame: int) -> list[dict[str, Any]]:
        return [r for r in self.tracks if r.get("frame") == frame]

    def rows_for_track(self, track_id: int) -> list[dict[str, Any]]:
        return [r for r in self.tracks if r.get("track_id") == track_id]

    def msd_for_track(self, track_id: int) -> list[dict[str, Any]]:
        return [r for r in self.msd if r.get("track_id") == track_id]

    def track_ids(self) -> list[int]:
        return sorted({int(r["track_id"]) for r in self.tracks if r.get("track_id") is not None})

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


def _measurement_config(manifest: dict[str, Any]) -> MeasurementConfig:
    saved = manifest.get("measurement") or {}
    defaults = MeasurementConfig()
    return MeasurementConfig(
        reference_point_px=None,
        msd_min_pairs=int(saved.get("msd_min_pairs", defaults.msd_min_pairs)),
        msd_min_lags_for_fit=int(
            saved.get("msd_min_lags_for_fit", defaults.msd_min_lags_for_fit)
        ),
        msd_max_lag_fraction=float(
            saved.get("msd_max_lag_fraction", defaults.msd_max_lag_fraction)
        ),
    )


def _min_observations(manifest: dict[str, Any]) -> int:
    value = (manifest.get("tracking") or {}).get("min_observations")
    return int(value) if isinstance(value, (int, float)) and value >= 1 else 2


def _upgrade_unlinked(rows: list[dict[str, Any]], confinement: dict[str, Any]) -> list[dict[str, Any]]:
    """v1 ``along/across_channel_px`` back to ``dx/dy_px``.

    v1 computed ``along = step @ u`` and ``across = step @ n`` with
    ``u = (ux, uy)`` and ``n = (-uy, ux)`` (confinement.py), a rotation, so
    ``step = along * u + across * n`` recovers it exactly.  ``mahalanobis``
    did not exist in v1 and stays empty.
    """
    ux, uy = confinement.get("ux"), confinement.get("uy")
    out = []
    for row in rows:
        new = {k: v for k, v in row.items() if k not in ("along_channel_px", "across_channel_px")}
        along, across = row.get("along_channel_px"), row.get("across_channel_px")
        if None not in (ux, uy, along, across):
            new["dx_px"] = along * ux - across * uy
            new["dy_px"] = along * uy + across * ux
        speed = row.get("implied_speed_um_per_min")
        new["implied_speed_um_per_hr"] = speed * 60.0 if isinstance(speed, float) else None
        out.append(new)
    return out


def _upgrade_v1(analysis: SavedAnalysis) -> None:
    manifest = analysis.manifest
    scale = analysis.scale
    tracks = tracks_from_rows(analysis.tracks, analysis.detections, summaries=analysis.summaries)
    measurement = _measurement_config(manifest)
    analysis.tracks = frame_rows(
        tracks, scale,
        source_frames=analysis.source_frames,
        min_observations=_min_observations(manifest),
    )
    analysis.msd = msd_rows(tracks, scale)
    analysis.summaries = [
        s.to_row()
        for s in summarise(
            tracks, scale,
            source_frames=analysis.source_frames,
            min_observations=_min_observations(manifest),
            msd_rows=analysis.msd,
            measurement=measurement,
        )
    ]
    analysis.unlinked = _upgrade_unlinked(analysis.unlinked, manifest.get("confinement") or {})
    version = (manifest.get("application") or {}).get("version")
    analysis.upgraded_from = str(version) if version else "1.x"
    analysis.upgrade_notes = [
        f"Written by Corridor {analysis.upgraded_from} (output schema 1) and upgraded "
        "on loading: per-hour speeds, MTrackJ distances, MSD and summaries were "
        "computed from the saved positions. Nothing was re-segmented.",
        "The v1 along/across-channel columns are not part of schema 2.",
    ]


def load_analysis(directory: str | Path) -> SavedAnalysis:
    """Load a run written by any Corridor release, without recomputing pixels."""
    directory = Path(directory)
    manifest_path = directory / pipeline.F_MANIFEST
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            manifest = {}
    if not isinstance(manifest, dict):
        manifest = {}

    raw_version = manifest.get("schema_version")
    schema = int(raw_version) if isinstance(raw_version, (int, float)) else 1
    axes = str((manifest.get("input") or {}).get("axes") or "")
    dimensionality = manifest.get("dimensionality") or ("3D" if "Z" in axes else "2D")

    analysis = SavedAnalysis(
        directory=directory,
        manifest=manifest,
        tracks=read_table(directory / pipeline.F_TRACKS),
        detections=read_table(directory / pipeline.F_DETECTIONS),
        summaries=read_table(directory / pipeline.F_SUMMARY),
        diagnostics=read_table(directory / pipeline.F_DIAGNOSTICS),
        issues=read_table(directory / pipeline.F_QC),
        unlinked=read_table(directory / pipeline.F_UNLINKED),
        events=read_table(directory / pipeline.F_EVENTS),
        recovery=read_table(directory / pipeline.F_RECOVERY),
        msd=read_table(directory / F_MSD),
        schema_version=schema,
        dimensionality=str(dimensionality),
    )
    if schema < 2:
        _upgrade_v1(analysis)
    elif not analysis.msd and analysis.tracks:
        # A v2 run always writes track_msd.csv; if it is missing (copied by
        # hand, an interrupted bundle) the curves are still in the positions.
        analysis.msd = msd_rows(tracks_from_rows(analysis.tracks), analysis.scale)
    return analysis


def analysis_is_complete(directory: str | Path) -> bool:
    directory = Path(directory)
    return (directory / pipeline.F_MANIFEST).exists() and (
        directory / pipeline.F_TRACKS
    ).exists()


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------


def _with_reference(
    saved: SavedAnalysis, rows: list[dict[str, Any]], reference_point_px: Sequence[float] | None
) -> list[dict[str, Any]]:
    if reference_point_px is None:
        return rows
    scale = saved.scale
    return add_reference_distance(
        rows,
        reference_point_px,
        scale.pixel_size_um if scale.calibrated_space else None,
        z_step_um=scale.z_step_um if scale.calibrated_z else None,
    )


def _run_sheet(saved: SavedAnalysis, reference_point_px: Sequence[float] | None) -> list[dict]:
    """Key/value provenance for a per-track workbook, so it carries its units."""
    app = saved.manifest.get("application") or {}
    info = saved.manifest.get("input") or {}
    ref = reference_point_px if reference_point_px is not None else saved.reference_point_px
    items = [
        ("application", f"{app.get('name', 'Corridor')} {app.get('version', '')}".strip()),
        ("schema_version_on_disk", saved.schema_version),
        ("schema_version_exported", export.SCHEMA_VERSION),
        ("upgraded_from", saved.upgraded_from),
        ("source", info.get("name") or info.get("path")),
        ("dimensionality", saved.dimensionality),
        ("pixel_size_um", saved.pixel_size_um),
        ("frame_interval_min", saved.frame_interval_min),
        ("z_step_um", saved.z_step_um),
        ("reference_point_px", None if ref is None else ", ".join(f"{v:g}" for v in ref)),
    ]
    return [{"key": k, "value": v} for k, v in items]


def export_track(
    saved: SavedAnalysis,
    track_id: int,
    path: str | Path,
    *,
    fmt: str = "csv",
    reference_point_px: Sequence[float] | None = None,
) -> Path:
    """Write one track's observations (``tracks.csv`` v2 columns).

    ``fmt="xlsx"`` writes a workbook with the track, its summary, its MSD
    curve and the run's calibration, so the file is interpretable on its
    own.  ``reference_point_px`` (re)computes D2R for this export without
    changing the saved analysis.
    """
    fmt = fmt.lower().lstrip(".")
    rows = _with_reference(saved, saved.rows_for_track(track_id), reference_point_px)
    if fmt == "csv":
        return export.write_csv(path, export.TRACK_COLUMNS, rows)
    if fmt != "xlsx":
        raise ValueError(f"unknown export format {fmt!r}; expected 'csv' or 'xlsx'")
    summary = saved.summary_for(track_id)
    return export.write_xlsx(
        path,
        {
            f"Track {track_id}": (export.TRACK_COLUMNS, rows),
            "Summary": (export.SUMMARY_COLUMNS, [summary] if summary else []),
            "MSD": (export.MSD_COLUMNS, saved.msd_for_track(track_id)),
            "Run": (["key", "value"], _run_sheet(saved, reference_point_px)),
        },
    )


def export_tracks_csv(saved: SavedAnalysis, path: str | Path) -> Path:
    """Every observation of every track, in schema-2 columns (v1 runs upgraded)."""
    return export.write_csv(path, export.TRACK_COLUMNS, saved.tracks)


def export_summaries_csv(saved: SavedAnalysis, path: str | Path) -> Path:
    return export.write_csv(path, export.SUMMARY_COLUMNS, saved.summaries)


def export_msd_csv(saved: SavedAnalysis, path: str | Path) -> Path:
    return export.write_csv(path, export.MSD_COLUMNS, saved.msd)


def export_bundle(analysis: SavedAnalysis | str | Path, destination: str | Path) -> list[Path]:
    """Copy the result files a user would want to keep, into one folder.

    Files are copied byte for byte: a bundle of a v1 run is still that v1
    run's own record.  The one file a v1 run never wrote, ``track_msd.csv``,
    is written from the MSD computed on loading, so every bundle has it.
    """
    saved = analysis if isinstance(analysis, SavedAnalysis) else load_analysis(analysis)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    wanted = [
        pipeline.F_TRACKS,
        pipeline.F_SUMMARY,
        F_MSD,
        pipeline.F_DETECTIONS,
        pipeline.F_DIAGNOSTICS,
        pipeline.F_EVENTS,
        pipeline.F_QC,
        pipeline.F_RECOVERY,
        pipeline.F_UNLINKED,
        pipeline.F_MANIFEST,
        pipeline.F_MASKS,
        "preview.png",
    ]
    written: list[Path] = []
    for name in wanted:
        source = saved.directory / name
        target = destination / name
        if source.exists():
            shutil.copy2(source, target)
            written.append(target)
        elif name == F_MSD and saved.tracks:
            written.append(export.write_csv(target, export.MSD_COLUMNS, saved.msd))
    return written
