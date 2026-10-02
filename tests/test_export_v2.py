"""Output schema v2 and the dependency-free XLSX writer (contract §7)."""

from __future__ import annotations

import zipfile
from dataclasses import fields
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from corridor.core import export
from corridor.core.detections import (
    detections_to_rows,
    extract_detections,
    extract_detections_3d,
)
from corridor.core.measurements import TrackSummary

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

#: Columns that existed only because of the v1 migration axis.
AXIS_COLUMNS = {
    "v_along_um_per_min", "v_across_um_per_min", "net_along_um", "net_across_um",
    "along_speed_um_per_min", "along_channel_px", "across_channel_px",
}


def read_xlsx(path):
    """Minimal reader for what write_xlsx produces: {sheet: [[cell, ...], ...]}."""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        for name in names:
            if name.endswith(".xml") or name.endswith(".rels"):
                ET.fromstring(zf.read(name))  # every part is well-formed XML
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        sheets = [s.get("name") for s in workbook.find("m:sheets", NS)]
        out = {}
        for i, name in enumerate(sheets, start=1):
            root = ET.fromstring(zf.read(f"xl/worksheets/sheet{i}.xml"))
            table = []
            for row in root.find("m:sheetData", NS):
                cells = {}
                for c in row:
                    ref = c.get("r")
                    col = "".join(ch for ch in ref if ch.isalpha())
                    kind = c.get("t")
                    if kind == "inlineStr":
                        value = c.find("m:is/m:t", NS).text
                    elif kind == "b":
                        value = c.find("m:v", NS).text == "1"
                    else:
                        value = float(c.find("m:v", NS).text)
                    cells[col] = (value, c.get("s"))
                table.append(cells)
            out[name] = table
        return out


def test_schema_version_is_two():
    assert export.SCHEMA_VERSION == 2


def test_no_axis_column_survives_in_any_schema():
    for columns in (
        export.TRACK_COLUMNS, export.SUMMARY_COLUMNS, export.MSD_COLUMNS,
        export.DETECTION_COLUMNS, export.UNLINKED_COLUMNS, export.EVENT_COLUMNS,
    ):
        assert not AXIS_COLUMNS & set(columns)
        assert len(columns) == len(set(columns)), "duplicate column"


def test_summary_columns_are_the_summary_fields_in_order():
    assert export.SUMMARY_COLUMNS == [f.name for f in fields(TrackSummary)]


def test_new_columns_are_declared():
    assert {"cumulative_path_um", "distance_from_start_um", "distance_from_previous_um",
            "distance_from_reference_um", "speed_um_per_hr", "acceleration_um_per_hr2",
            "turning_angle_deg", "link_margin_chi2", "segmentation_confidence",
            "detection_source", "z_um", "volume_um3"} <= set(export.TRACK_COLUMNS)
    assert {"split_suspected", "gap_closed"} <= set(export.EVENT_COLUMNS)
    assert {"dx_px", "dy_px", "dz_px", "mahalanobis"} <= set(export.UNLINKED_COLUMNS)
    assert export.MSD_COLUMNS == [
        "track_id", "lag_frames", "lag_time_min", "lag_time_hr", "n_pairs",
        "msd_um2", "msd_px2",
    ]


def test_every_detection_field_has_a_column():
    """A key detections_to_rows produces but the schema lacks is silently dropped."""
    frame = np.zeros((20, 20), np.int32)
    frame[5:10, 5:12] = 1
    volume = np.zeros((6, 20, 20), np.int32)
    volume[1:4, 5:10, 5:12] = 1
    detections = extract_detections(frame, 0, intensity=frame.astype(float))
    detections += extract_detections_3d(volume, 0, spacing_zyx_um=(2.0, 0.5, 0.5))
    rows = detections_to_rows(
        detections, pixel_size_um=0.5, frame_interval_min=10.0,
        source_frames=[3], z_step_um=2.0,
    )
    keys = set().union(*rows)
    assert keys <= set(export.DETECTION_COLUMNS), keys - set(export.DETECTION_COLUMNS)


# --------------------------------------------------------------------------
# XLSX
# --------------------------------------------------------------------------


def test_xlsx_round_trips_types(tmp_path):
    path = export.write_xlsx(
        tmp_path / "out.xlsx",
        {
            "Data": (
                ["n", "f", "s", "b", "empty", "nan", "np"],
                [
                    {"n": 3, "f": 0.1234567891234, "s": "a <b> & 'c'", "b": True,
                     "empty": None, "nan": float("nan"), "np": np.float32(2.5)},
                    {"n": -1, "f": 1e-7, "s": "  padded  ", "b": False},
                ],
            ),
            "Other": (["x"], []),
        },
    )
    book = read_xlsx(path)
    assert list(book) == ["Data", "Other"]
    header, first, second = book["Data"]
    # Header is text, in the bold style.
    assert [header[c][0] for c in "ABCDEFG"] == ["n", "f", "s", "b", "empty", "nan", "np"]
    assert all(header[c][1] == "1" for c in header)
    assert first["A"][0] == 3.0
    assert first["B"][0] == pytest.approx(0.123456789)  # the CSV's 9-decimal rounding
    assert first["C"][0] == "a <b> & 'c'"
    assert first["D"][0] is True
    assert "E" not in first and "F" not in first, "None and NaN are empty cells"
    assert first["G"][0] == 2.5
    assert second["B"][0] == pytest.approx(1e-7)
    assert second["C"][0] == "  padded  "
    assert second["D"][0] is False
    assert book["Other"] == [{"A": ("x", "1")}]


def test_xlsx_package_parts(tmp_path):
    path = export.write_xlsx(tmp_path / "p.xlsx", {"A": (["x"], [{"x": 1}])})
    with zipfile.ZipFile(path) as zf:
        assert zf.namelist()[0] == "[Content_Types].xml"
        assert {"_rels/.rels", "xl/workbook.xml", "xl/_rels/workbook.xml.rels",
                "xl/styles.xml", "xl/worksheets/sheet1.xml"} <= set(zf.namelist())
        assert zf.testzip() is None
    assert [p.name for p in tmp_path.iterdir()] == ["p.xlsx"], "no temporary file left"


def test_xlsx_strips_characters_xml_cannot_carry(tmp_path):
    path = export.write_xlsx(tmp_path / "c.xlsx", {"S": (["t"], [{"t": "a\x00b\x07c\td"}])})
    assert read_xlsx(path)["S"][1]["A"][0] == "abc\td"


def test_sheet_names_follow_excel_rules():
    names = export.sanitise_sheet_names(
        ["Track 1/2", "a" * 40, "Track 1_2", "track 1_2", "'quoted'", "", "History", "x:y?"]
    )
    assert names[0] == "Track 1_2"
    assert names[1] == "a" * 31
    assert names[2] == "Track 1_2 (2)"
    assert names[3] == "track 1_2 (3)"
    assert names[4] == "quoted"
    assert names[5] == "Sheet"
    assert names[6] == "History_"
    assert names[7] == "x_y_"
    assert len({n.lower() for n in names}) == len(names)
    assert all(1 <= len(n) <= 31 for n in names)


def test_long_duplicate_names_stay_within_31_characters():
    names = export.sanitise_sheet_names(["b" * 40, "b" * 35])
    assert names == ["b" * 31, "b" * 27 + " (2)"]


def test_xlsx_refuses_an_empty_workbook(tmp_path):
    with pytest.raises(ValueError):
        export.write_xlsx(tmp_path / "e.xlsx", {})


def test_xlsx_opens_in_openpyxl_when_available(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = export.write_xlsx(
        tmp_path / "o.xlsx", {"T": (["a", "b"], [{"a": 1.5, "b": "x"}])}
    )
    sheet = openpyxl.load_workbook(path)["T"]
    assert [c.value for c in sheet[2]] == [1.5, "x"]


def test_csv_keeps_its_clean_semantics(tmp_path):
    path = export.write_csv(
        tmp_path / "c.csv", ["a", "b", "c", "d"],
        [{"a": float("nan"), "b": True, "c": 0.1 + 0.2, "d": np.int64(4)}],
    )
    assert path.read_text(encoding="utf-8").splitlines()[1] == ",true,0.3,4"
