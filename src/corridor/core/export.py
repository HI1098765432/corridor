"""Writing results to disk.

Two properties matter more than convenience here:

*   **Stable schemas.**  Column order and names are declared explicitly, so a
    script that reads ``tracks.csv`` keeps working and a missing value is an
    empty cell rather than a shifted column.
*   **Atomic writes.**  Every file is written to a temporary name and then
    moved into place, so an interrupted run cannot leave a half-written CSV
    that looks like a complete result.

CSV is the canonical format.  :func:`write_xlsx` exists for the per-track
export a user opens in Excel, and is written with the standard library only
(``zipfile`` + XML): a spreadsheet library would be a new dependency in the
installer for one button.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.sax.saxutils import escape

import numpy as np

#: Version of the *output files* (run.json and the CSVs), not of the
#: application and not of the project database.  1 is every 1.x release (which
#: wrote no version at all); 2 removed the migration-axis columns, added
#: MTrackJ distances, per-hour units, MSD, morphology and Z.
SCHEMA_VERSION = 2

# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

#: Every key ``detections.detections_to_rows`` can produce, 2-D and 3-D.
#: ``x``/``y``/``z`` are pixels and slices; a µm column is empty without the
#: calibration it needs (``z_um``, ``volume_um3`` need the Z step).
DETECTION_COLUMNS = [
    "frame", "source_frame", "elapsed_min", "label", "channel",
    "x", "y", "z", "x_um", "y_um", "z_um",
    "area_px", "area_um2", "extent_px",
    "bbox_min_x", "bbox_min_y", "bbox_min_z", "bbox_max_x", "bbox_max_y", "bbox_max_z",
    "eccentricity", "orientation_rad",
    "major_axis_px", "minor_axis_px", "major_axis_um", "minor_axis_um",
    "aspect_ratio", "solidity", "extent_fraction",
    "perimeter_px", "perimeter_um", "circularity",
    "convex_area_px", "convex_area_um2",
    "equivalent_diameter_px", "equivalent_diameter_um",
    "mean_intensity", "median_intensity",
    "volume_vox", "volume_um3", "surface_area_um2",
    "principal_axis_1_um", "principal_axis_2_um", "principal_axis_3_um",
    "elongation", "flatness", "sphericity",
    "touches_border", "source", "confidence",
]

#: ``tracks.csv`` v2.  See ``measurements.frame_rows`` for every definition.
#: MTrackJ names: Len = ``cumulative_path_um``, D2S = ``distance_from_start_um``,
#: D2P = ``distance_from_previous_um``, D2R = ``distance_from_reference_um``.
#: ``z_slice`` is the Z position in slices (a slice is not a pixel).
TRACK_COLUMNS = [
    "track_id", "frame", "source_frame", "elapsed_min", "elapsed_hr",
    "x_px", "y_px", "z_slice", "x_um", "y_um", "z_um",
    "det_label", "channel", "observation_index", "n_observations",
    "gap_frames", "match_cost_chi2", "link_margin_chi2",
    "distance_from_previous_px", "distance_from_previous_um",
    "cumulative_path_px", "cumulative_path_um",
    "distance_from_start_px", "distance_from_start_um",
    "distance_from_reference_px", "distance_from_reference_um",
    "vx_px_per_frame", "vy_px_per_frame", "speed_px_per_frame",
    "vx_um_per_min", "vy_um_per_min", "vz_um_per_min", "speed_um_per_min",
    "vx_um_per_hr", "vy_um_per_hr", "vz_um_per_hr", "speed_um_per_hr",
    "acceleration_um_per_hr2", "turning_angle_deg",
    "area_px", "area_um2", "perimeter_px", "perimeter_um",
    "major_axis_px", "minor_axis_px", "major_axis_um", "minor_axis_um",
    "aspect_ratio", "eccentricity", "orientation_rad", "circularity",
    "solidity", "extent_fraction", "mean_intensity",
    "volume_vox", "volume_um3", "surface_area_um2",
    "sphericity", "elongation", "flatness",
    "track_flags", "detection_source", "segmentation_confidence",
]

#: ``track_summary.csv`` v2: exactly the fields of ``measurements.TrackSummary``
#: (tests/test_manifest.py holds the two together).
SUMMARY_COLUMNS = [
    "track_id", "channel", "dimensionality",
    "first_frame", "last_frame", "first_source_frame", "last_source_frame",
    "n_observations", "n_gaps", "total_missing_frames", "span_frames",
    "duration_min", "duration_hr",
    "net_displacement_px", "net_displacement_um",
    "path_length_px", "path_length_um",
    "max_distance_from_start_um", "straightness",
    "mean_speed_um_per_min", "median_speed_um_per_min", "max_speed_um_per_min",
    "mean_speed_um_per_hr", "median_speed_um_per_hr", "max_speed_um_per_hr",
    "net_speed_um_per_min", "net_speed_um_per_hr",
    "path_speed_um_per_min", "path_speed_um_per_hr",
    "mean_speed_px_per_frame", "net_speed_px_per_frame",
    "mean_turning_angle_deg", "directional_autocorrelation",
    "persistence_time_min", "persistence_time_hr",
    "persistence_fit_r2", "persistence_fit_lags",
    "msd_alpha", "msd_alpha_r2", "msd_fit_lags",
    "mean_area_px", "mean_area_um2", "mean_volume_um3",
    "min_link_margin_chi2", "flags",
]

#: ``track_msd.csv``: one row per (track, actual frame lag).  MSD is in µm²
#: (``msd_px2`` for uncalibrated data), never µm.
MSD_COLUMNS = [
    "track_id", "lag_frames", "lag_time_min", "lag_time_hr", "n_pairs",
    "msd_um2", "msd_px2",
]

DIAGNOSTIC_COLUMNS = [
    "frame", "raw_instances", "kept_instances", "removed_instances",
    "removed_max_extent_px", "removed_max_area_px", "cellpose_message",
]

#: ``split_suspected`` and ``gap_closed`` hold ";"-joined track ids, like
#: ``merge_suspected_tracks``.
EVENT_COLUMNS = [
    "frame", "detections", "candidate_tracks", "matched", "new_tracks",
    "dormant", "terminated", "merge_suspected_tracks",
    "split_suspected", "gap_closed", "notes",
]

QC_COLUMNS = ["severity", "code", "title", "detail", "frame", "track_id"]

#: Every attempt to find a cell a track predicted but segmentation missed,
#: successful or not. A failed attempt is as informative as a successful one.
RECOVERY_COLUMNS = [
    "track_id", "frame", "predicted_x", "predicted_y", "recovered",
    "found_by", "confidence", "offset_from_prediction_px", "detail",
]

#: Why each mid-stack track was not joined to an earlier one. This is the
#: evidence behind a judgement the tracker made, not a result in itself.
#: ``dx/dy/dz_px`` is the jump in isotropic pixels (dz empty in 2-D) and
#: ``mahalanobis`` its distance under the predicted state's covariance; they
#: replace v1's ``along_channel_px``/``across_channel_px``.
UNLINKED_COLUMNS = [
    "track_id", "starts_at_frame", "nearest_earlier_track",
    "that_track_ended_at_frame", "gap_frames", "distance_px",
    "dx_px", "dy_px", "dz_px", "mahalanobis",
    "implied_speed_um_per_min", "implied_speed_um_per_hr",
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


def _atomic_write(path: str | Path, write) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        write(fd)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    def write(fd: int) -> None:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as fh:
            fh.write(text)

    return _atomic_write(path, write)


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    def write(fd: int) -> None:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)

    return _atomic_write(path, write)


def write_csv(path: str | Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> Path:
    """Write a CSV with a fixed header, even when there are no rows."""
    sio = io.StringIO()
    writer = csv.DictWriter(sio, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: _clean(row.get(c)) for c in columns})
    return atomic_write_text(path, sio.getvalue())


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
    """Store a label stack (``T[Z]YX``) compressed; label images compress well."""
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


# --------------------------------------------------------------------------
# XLSX (Office Open XML, SpreadsheetML), standard library only
# --------------------------------------------------------------------------

_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
_CT_SHEET = "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"
_CT_WORKBOOK = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"
_CT_STYLES = "application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"

#: Excel's limits.  Exceeding them produces a file Excel "repairs" by
#: deleting data, so they are refused instead.
XLSX_MAX_ROWS = 1_048_576
XLSX_MAX_COLUMNS = 16_384
#: Longest text a cell holds; longer text is cut to this, never dropped.
XLSX_MAX_CELL_CHARS = 32_767
_SHEET_NAME_MAX = 31
_SHEET_NAME_FORBIDDEN = re.compile(r"[\[\]:*?/\\]")
#: Characters XML 1.0 cannot carry at all, even escaped.
_XML_ILLEGAL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


def sanitise_sheet_names(names: Sequence[str]) -> list[str]:
    """Make names valid, unique Excel sheet names, keeping order.

    Excel's rules: 1-31 characters, none of ``[ ] : * ? / \\``, not starting
    or ending with an apostrophe, unique ignoring case, and not ``History``
    (reserved).  Forbidden characters become ``_``; a duplicate gets `` (2)``,
    `` (3)``... within the 31 characters.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in names:
        name = _SHEET_NAME_FORBIDDEN.sub("_", _XML_ILLEGAL.sub("", str(raw)))
        name = name.strip().strip("'").strip() or "Sheet"
        if name.lower() == "history":
            name = "History_"
        name = name[:_SHEET_NAME_MAX]
        candidate, n = name, 1
        while candidate.lower() in seen:
            n += 1
            suffix = f" ({n})"
            candidate = name[: _SHEET_NAME_MAX - len(suffix)] + suffix
        seen.add(candidate.lower())
        out.append(candidate)
    return out


def _column_letter(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA."""
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _xlsx_cell(ref: str, value: Any, style: int = 0) -> str:
    """One ``<c>`` element, or "" for an empty cell.

    Numbers are written as numbers (so Excel can sort and plot them), with the
    same 9-decimal rounding as the CSV so both formats carry the same value.
    None, NaN and infinities are empty cells, as in the CSV.  Booleans are
    Excel booleans.  Everything else is inline text.
    """
    s = f' s="{style}"' if style else ""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return f'<c r="{ref}"{s} t="b"><v>{int(bool(value))}</v></c>'
    if isinstance(value, (int, np.integer)):
        return f'<c r="{ref}"{s}><v>{int(value)}</v></c>'
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return ""
        text = repr(round(number, 9)) if abs(number) < 1e15 else repr(number)
        return f'<c r="{ref}"{s}><v>{text}</v></c>'
    text = _XML_ILLEGAL.sub("", str(value))[:XLSX_MAX_CELL_CHARS]
    if text == "":
        return ""
    return (
        f'<c r="{ref}"{s} t="inlineStr"><is><t xml:space="preserve">'
        f"{escape(text)}</t></is></c>"
    )


def _sheet_xml(columns: Sequence[str], rows: Sequence[Mapping[str, Any]], bold_header: bool) -> str:
    if len(columns) > XLSX_MAX_COLUMNS:
        raise ValueError(f"{len(columns)} columns exceed Excel's {XLSX_MAX_COLUMNS}")
    if len(rows) + 1 > XLSX_MAX_ROWS:
        raise ValueError(f"{len(rows)} rows exceed Excel's {XLSX_MAX_ROWS - 1} data rows")
    letters = [_column_letter(i) for i in range(len(columns))]
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        f'<worksheet xmlns="{_NS_MAIN}">',
    ]
    if columns:
        # Freeze the header row so it stays visible while scrolling.
        parts.append(
            '<sheetViews><sheetView workbookViewId="0">'
            '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
            "</sheetView></sheetViews>"
        )
    parts.append("<sheetData>")
    header_style = 1 if bold_header else 0
    parts.append(
        '<row r="1">'
        + "".join(
            _xlsx_cell(f"{letters[i]}1", str(c), header_style) for i, c in enumerate(columns)
        )
        + "</row>"
    )
    for r, row in enumerate(rows, start=2):
        cells = "".join(
            _xlsx_cell(f"{letters[i]}{r}", row.get(c)) for i, c in enumerate(columns)
        )
        parts.append(f'<row r="{r}">{cells}</row>')
    parts.append("</sheetData></worksheet>")
    return "".join(parts)


_STYLES_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<styleSheet xmlns="{_NS_MAIN}">'
    '<fonts count="2">'
    '<font><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
    '<font><b/><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
    "</fonts>"
    '<fills count="2"><fill><patternFill patternType="none"/></fill>'
    '<fill><patternFill patternType="gray125"/></fill></fills>'
    '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
    '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
    '<cellXfs count="2">'
    '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
    '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
    "</cellXfs>"
    '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
    "</styleSheet>"
)


def write_xlsx(
    path: str | Path,
    sheets: Mapping[str, tuple[Sequence[str], Iterable[Mapping[str, Any]]]],
    *,
    bold_header: bool = True,
) -> Path:
    """Write a workbook: one sheet per ``name -> (columns, rows)``, in order.

    The minimal valid package: content types, relationships, workbook,
    styles (a bold header font) and one worksheet each, with inline strings
    so no shared-string table has to be kept consistent.  Sheet names are
    made valid by :func:`sanitise_sheet_names`.  Written atomically, like
    every other result file.
    """
    if not sheets:
        raise ValueError("a workbook needs at least one sheet")
    names = sanitise_sheet_names(list(sheets))
    bodies = [
        _sheet_xml(list(columns), list(rows), bold_header)
        for columns, rows in sheets.values()
    ]
    n = len(names)
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'<Override PartName="/xl/workbook.xml" ContentType="{_CT_WORKBOOK}"/>'
        f'<Override PartName="/xl/styles.xml" ContentType="{_CT_STYLES}"/>'
        + "".join(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="{_CT_SHEET}"/>'
            for i in range(1, n + 1)
        )
        + "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{_NS_PKG_REL}">'
        f'<Relationship Id="rId1" Type="{_NS_REL}/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<workbook xmlns="{_NS_MAIN}" xmlns:r="{_NS_REL}"><sheets>'
        + "".join(
            f'<sheet name="{escape(name, {chr(34): "&quot;"})}" sheetId="{i}" r:id="rId{i}"/>'
            for i, name in enumerate(names, start=1)
        )
        + "</sheets></workbook>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{_NS_PKG_REL}">'
        + "".join(
            f'<Relationship Id="rId{i}" Type="{_NS_REL}/worksheet" '
            f'Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, n + 1)
        )
        + f'<Relationship Id="rId{n + 1}" Type="{_NS_REL}/styles" Target="styles.xml"/>'
        "</Relationships>"
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # [Content_Types].xml first: some readers sniff the package by it.
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/styles.xml", _STYLES_XML)
        for i, body in enumerate(bodies, start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", body)
    return atomic_write_bytes(path, buffer.getvalue())
