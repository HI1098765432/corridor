"""Writing results to disk.

Two properties matter more than convenience here:

*   **Stable schemas.**  Column order and names are declared explicitly, so a
    script that reads ``tracks.csv`` keeps working and a missing value is an
    empty cell rather than a shifted column.
*   **Atomic writes.**  Every file is written to a temporary name and then
    moved into place, so an interrupted run cannot leave a half-written CSV
    that looks like a complete result.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

DETECTION_COLUMNS = [
    "frame", "source_frame", "elapsed_min", "label", "channel",
    "x", "y", "x_um", "y_um",
    "area_px", "area_um2", "extent_px",
    "bbox_min_x", "bbox_min_y", "bbox_max_x", "bbox_max_y",
    "eccentricity", "orientation_rad", "major_axis_px", "minor_axis_px",
    "solidity", "touches_border",
]

TRACK_COLUMNS = [
    "track_id", "frame", "source_frame", "elapsed_min",
    "x_px", "y_px", "x_um", "y_um",
    "area_px", "area_um2", "det_label", "channel",
    "gap_frames", "match_cost_chi2",
    "vx_px_per_frame", "vy_px_per_frame", "speed_px_per_frame",
    "vx_um_per_min", "vy_um_per_min", "speed_um_per_min",
    "v_along_um_per_min", "v_across_um_per_min",
    "step_px", "step_um",
    "observation_index", "n_observations", "track_flags",
]

SUMMARY_COLUMNS = [
    "track_id", "channel",
    "first_frame", "last_frame", "first_source_frame", "last_source_frame",
    "n_observations", "n_gaps", "total_missing_frames", "span_frames",
    "duration_min",
    "net_displacement_um", "net_along_um", "net_across_um",
    "path_length_um", "straightness",
    "mean_speed_um_per_min", "median_speed_um_per_min", "max_speed_um_per_min",
    "mean_area_px", "flags",
]

DIAGNOSTIC_COLUMNS = [
    "frame", "raw_instances", "kept_instances", "removed_instances",
    "removed_max_extent_px", "removed_max_area_px", "cellpose_message",
]

EVENT_COLUMNS = [
    "frame", "detections", "candidate_tracks", "matched", "new_tracks",
    "dormant", "terminated", "merge_suspected_tracks", "notes",
]

QC_COLUMNS = ["severity", "code", "title", "detail", "frame", "track_id"]

#: Why each mid-stack track was not joined to an earlier one. This is the
#: evidence behind a judgement the tracker made, not a result in itself.
UNLINKED_COLUMNS = [
    "track_id", "starts_at_frame", "nearest_earlier_track",
    "that_track_ended_at_frame", "gap_frames", "distance_px",
    "along_channel_px", "across_channel_px", "implied_speed_um_per_min",
    "would_have_cost_chi2", "refused_because", "explanation",
]


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def _clean(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return repr(round(value, 9)) if abs(value) < 1e15 else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _clean(float(value))
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    return value


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def write_csv(path: str | Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> Path:
    """Write a CSV with a fixed header, even when there are no rows."""
    buf: list[str] = []
    import io

    sio = io.StringIO()
    writer = csv.DictWriter(sio, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: _clean(row.get(c)) for c in columns})
    buf.append(sio.getvalue())
    return atomic_write_text(path, "".join(buf))


def _sanitize(value: Any) -> Any:
    """Replace values that would make the JSON unreadable by other tools.

    ``json.dumps`` writes bare ``NaN`` and ``Infinity`` tokens, which are not
    valid JSON: a manifest containing one cannot be parsed by JavaScript,
    jsonlite, or anything else strict. Because ``numpy.float64`` subclasses
    ``float``, the ``default`` hook never sees those values, so they have to be
    replaced before encoding rather than during it.
    """
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return _sanitize(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if (math.isnan(number) or math.isinf(number)) else number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(
            _sanitize(payload),
            indent=2,
            sort_keys=False,
            default=_json_default,
            allow_nan=False,
        ),
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def save_masks(path: str | Path, masks: np.ndarray) -> Path:
    """Store a label stack compressed; label images compress extremely well."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(tmp, masks=np.asarray(masks, dtype=np.int32))
    produced = tmp if tmp.exists() else Path(str(tmp) + ".npz")
    os.replace(produced, path)
    return path


def load_masks(path: str | Path) -> np.ndarray:
    with np.load(str(path)) as data:
        return np.asarray(data["masks"], dtype=np.int32)
