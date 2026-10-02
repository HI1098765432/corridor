"""Local project storage: it must survive closing the application."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from corridor.core import export, pipeline
from corridor.core.config import RunConfig
from corridor.store import db
from corridor.store.project import (
    F_MSD,
    analysis_is_complete,
    export_bundle,
    export_msd_csv,
    export_summaries_csv,
    export_track,
    export_tracks_csv,
    load_analysis,
    read_table,
)


@pytest.fixture
def store(tmp_path, monkeypatch) -> db.Store:
    monkeypatch.setenv("CORRIDOR_DATA_DIR", str(tmp_path / "appdata"))
    instance = db.Store()
    yield instance
    instance.close()


def test_data_directory_honours_the_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CORRIDOR_DATA_DIR", str(tmp_path / "elsewhere"))
    assert db.app_data_dir() == tmp_path / "elsewhere"


def test_create_and_read_back_a_project(store, tmp_path):
    source = tmp_path / "movie.tif"
    source.write_bytes(b"not really a tiff")
    record = store.create_project(source)

    assert record.path.is_dir()
    assert record.status == db.STATUS_NEW

    fetched = store.get_project(record.id)
    assert fetched is not None
    assert fetched.source_path == str(source)
    assert fetched.name == "movie"


def test_project_survives_a_new_store_instance(store, tmp_path, monkeypatch):
    source = tmp_path / "movie.tif"
    source.write_bytes(b"x")
    record = store.create_project(source)
    record.status = db.STATUS_COMPLETE
    record.n_tracks = 7
    record.n_frames = 18
    record.config = RunConfig().to_dict()
    store.update_project(record)
    store.close()

    reopened = db.Store()
    try:
        again = reopened.get_project(record.id)
        assert again is not None
        assert again.status == db.STATUS_COMPLETE
        assert again.n_tracks == 7
        assert again.config["tracking"]["max_gap"] == 3
    finally:
        reopened.close()


def test_recent_projects_are_most_recent_first(store, tmp_path):
    ids = []
    for name in ("a", "b", "c"):
        source = tmp_path / f"{name}.tif"
        source.write_bytes(b"x")
        ids.append(store.create_project(source).id)
    # Touch the first one so it becomes the newest.
    first = store.get_project(ids[0])
    store.update_project(first)
    recent = store.recent_projects()
    assert [r.id for r in recent][0] == ids[0]
    assert len(recent) == 3


def test_find_by_source_returns_the_existing_project(store, tmp_path):
    source = tmp_path / "movie.tif"
    source.write_bytes(b"x")
    record = store.create_project(source)
    assert store.find_by_source(source).id == record.id
    assert store.find_by_source(tmp_path / "other.tif") is None


def test_deleting_a_project_never_touches_the_source(store, tmp_path):
    source = tmp_path / "precious.tif"
    source.write_bytes(b"irreplaceable microscopy")
    record = store.create_project(source)
    (record.path / "tracks.csv").write_text("track_id\n", encoding="utf-8")

    store.delete_project(record.id, remove_files=True)

    assert store.get_project(record.id) is None
    assert not record.path.exists(), "the project folder should be gone"
    assert source.exists(), "the original microscopy file must never be deleted"
    assert source.read_bytes() == b"irreplaceable microscopy"


def test_settings_round_trip(store):
    assert store.get_setting("missing", "fallback") == "fallback"
    store.set_setting("use_gpu", True)
    store.set_setting("export_dir", "C:/somewhere")
    assert store.get_setting("use_gpu") is True
    assert store.get_setting("export_dir") == "C:/somewhere"
    store.set_setting("use_gpu", False)
    assert store.get_setting("use_gpu") is False


def test_settings_survive_reopening(store, tmp_path):
    store.set_setting("export_dir", str(tmp_path))
    store.close()
    reopened = db.Store()
    try:
        assert reopened.get_setting("export_dir") == str(tmp_path)
    finally:
        reopened.close()


def test_status_labels_are_human_readable():
    assert db.status_label(db.STATUS_COMPLETE) == "Analysed"
    assert db.status_label(db.STATUS_NEW) == "Not analysed"
    assert db.status_label("something_else") == "something_else"


# --------------------------------------------------------------------------
# Saved analyses
# --------------------------------------------------------------------------


def _write_minimal_analysis(directory: Path, *, schema_version: int | None = 2) -> None:
    """A two-observation v2 run (``schema_version=None`` writes it as 1.x did)."""
    directory.mkdir(parents=True, exist_ok=True)
    export.write_csv(
        directory / pipeline.F_TRACKS,
        export.TRACK_COLUMNS,
        [
            {"track_id": 1, "frame": 0, "x_px": 10.0, "y_px": 20.0, "area_px": 100.0,
             "gap_frames": 0, "n_observations": 2, "cumulative_path_um": 0.0},
            {"track_id": 1, "frame": 1, "x_px": 10.0, "y_px": 40.0, "area_px": 110.0,
             "gap_frames": 1, "speed_um_per_min": 0.47, "speed_um_per_hr": 28.2,
             "n_observations": 2, "cumulative_path_um": 9.342},
        ],
    )
    export.write_csv(
        directory / pipeline.F_SUMMARY,
        export.SUMMARY_COLUMNS,
        [{"track_id": 1, "n_observations": 2, "first_frame": 0, "last_frame": 1,
          "flags": ""}],
    )
    export.write_csv(
        directory / F_MSD, export.MSD_COLUMNS,
        [{"track_id": 1, "lag_frames": 1, "n_pairs": 1, "msd_um2": 87.27, "msd_px2": 400.0}],
    )
    export.write_csv(directory / pipeline.F_DETECTIONS, export.DETECTION_COLUMNS, [])
    export.write_csv(
        directory / pipeline.F_DIAGNOSTICS, export.DIAGNOSTIC_COLUMNS,
        [{"frame": 0, "raw_instances": 1, "kept_instances": 1, "cellpose_message": "nan"},
         {"frame": 1, "raw_instances": 1, "kept_instances": 1, "cellpose_message": "inf"}],
    )
    export.write_csv(directory / pipeline.F_QC, export.QC_COLUMNS, [])
    export.write_csv(
        directory / pipeline.F_EVENTS, export.EVENT_COLUMNS,
        [{"frame": 1, "detections": 1, "merge_suspected_tracks": "3",
          "split_suspected_tracks": "4", "gap_closed_tracks": "1"}],
    )
    export.write_csv(
        directory / pipeline.F_RECOVERY, export.RECOVERY_COLUMNS,
        [{"track_id": 1, "frame": 1, "recovered": False, "found_by": "windowed",
          "detail": "1"}],
    )
    manifest = {
        "input": {"path": "C:/x/movie.tif", "axes": "TYX", "shape": [2, 50, 30],
                  "source_frames": [42, 43]},
        "calibration": {"pixel_size_um": 0.4671, "frame_interval_min": 20.0069},
        "dimensionality": "2D",
        "results": {"n_tracks": 1},
    }
    if schema_version is not None:
        manifest = {"schema_version": schema_version, **manifest}
    else:
        manifest["input"] = {"path": "C:/x/movie.tif", "shape_tyx": [2, 50, 30],
                             "source_frames": [42, 43]}
        manifest["confinement"] = {"ux": 0.0, "uy": 1.0, "channels": []}
    export.write_json(directory / pipeline.F_MANIFEST, manifest)
    export.save_masks(directory / pipeline.F_MASKS, np.zeros((2, 50, 30), np.int32))


def test_saved_analysis_round_trip(tmp_path):
    directory = tmp_path / "run"
    _write_minimal_analysis(directory)
    assert analysis_is_complete(directory)

    analysis = load_analysis(directory)
    assert analysis.schema_version == 2 and analysis.upgraded_from is None
    assert analysis.n_frames == 2
    assert analysis.pixel_size_um == pytest.approx(0.4671)
    assert analysis.frame_interval_min == pytest.approx(20.0069)
    assert analysis.source_frames == [42, 43]
    assert analysis.track_ids() == [1]
    assert len(analysis.rows_for_track(1)) == 2
    assert len(analysis.rows_for_frame(1)) == 1
    assert analysis.summary_for(1)["n_observations"] == 2
    assert analysis.msd_for_track(1)[0]["msd_um2"] == pytest.approx(87.27)
    assert analysis.masks.shape == (2, 50, 30)
    # Loaded in 2.0, never in 1.x.
    # A single id is text, not the float 1.0 (the _coerce trap).
    assert analysis.events[0]["gap_closed_tracks"] == "1"
    assert analysis.events[0]["split_suspected_tracks"] == "4"
    assert analysis.events[0]["merge_suspected_tracks"] == "3"
    assert analysis.recovery[0]["recovered"] is False


def test_a_v2_run_is_read_as_written(tmp_path):
    """No recomputation: the files are the record."""
    directory = tmp_path / "run"
    _write_minimal_analysis(directory)
    row = load_analysis(directory).rows_for_frame(1)[0]
    assert row["speed_um_per_min"] == pytest.approx(0.47)
    assert row["cumulative_path_um"] == pytest.approx(9.342)


def test_a_run_without_a_schema_version_is_upgraded_from_its_positions(tmp_path):
    directory = tmp_path / "run"
    _write_minimal_analysis(directory, schema_version=None)
    analysis = load_analysis(directory)
    assert analysis.schema_version == 1
    row = analysis.rows_for_frame(1)[0]
    expected = 20.0 * 0.4671 / 20.0069
    assert row["speed_um_per_min"] == pytest.approx(expected)
    assert row["speed_um_per_hr"] == pytest.approx(expected * 60.0)
    assert row["cumulative_path_um"] == pytest.approx(20.0 * 0.4671)
    assert analysis.n_frames == 2


def test_numeric_columns_come_back_as_numbers(tmp_path):
    directory = tmp_path / "run"
    _write_minimal_analysis(directory)
    analysis = load_analysis(directory)
    row = analysis.rows_for_frame(1)[0]
    assert isinstance(row["track_id"], int)
    assert isinstance(row["x_px"], float)
    assert row["speed_um_per_min"] == pytest.approx(0.47)
    # An empty cell must be None, never the string "".
    assert analysis.rows_for_frame(0)[0]["speed_um_per_min"] is None


def test_text_cells_are_never_turned_into_numbers(tmp_path):
    """``float()`` parses "nan", "inf" and "1": a message or an id list is text."""
    directory = tmp_path / "run"
    _write_minimal_analysis(directory)
    analysis = load_analysis(directory)
    assert analysis.diagnostic_for(0)["cellpose_message"] == "nan"
    assert analysis.diagnostic_for(1)["cellpose_message"] == "inf"
    assert analysis.events[0]["merge_suspected_tracks"] == "3"
    assert analysis.recovery[0]["detail"] == "1"


def test_integer_columns_are_never_truncated(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("frame,track_id,label,unknown\n2.5,7.0,x9,nan\n", encoding="utf-8")
    (row,) = read_table(path)
    assert row["frame"] == 2.5, "int(float('2.5')) would silently say 2"
    assert row["track_id"] == 7 and isinstance(row["track_id"], int)
    assert row["label"] == "x9", "kept as written rather than replaced by None"
    assert row["unknown"] == "nan", "the writer never emits NaN; this cell is text"


def test_event_id_lists_stay_text_in_either_spelling(tmp_path):
    """The tracker's ``*_tracks`` names and the bare draft names are both text."""
    path = tmp_path / "e.csv"
    path.write_text(
        "frame,split_suspected,gap_closed,split_suspected_tracks,gap_closed_tracks\n"
        "2,3,4,5,6\n",
        encoding="utf-8",
    )
    assert read_table(path) == [{
        "frame": 2, "split_suspected": "3", "gap_closed": "4",
        "split_suspected_tracks": "5", "gap_closed_tracks": "6",
    }]


def test_booleans_come_back_as_booleans(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("touches_border,recovered\ntrue,false\n", encoding="utf-8")
    assert read_table(path) == [{"touches_border": True, "recovered": False}]


def test_three_d_masks_and_frames_from_the_v2_shape(tmp_path):
    directory = tmp_path / "run"
    _write_minimal_analysis(directory)
    manifest = json.loads((directory / pipeline.F_MANIFEST).read_text(encoding="utf-8"))
    manifest["input"].update(axes="TZYX", shape=[3, 4, 50, 30])
    manifest["dimensionality"] = "3D"
    export.write_json(directory / pipeline.F_MANIFEST, manifest)
    export.save_masks(directory / pipeline.F_MASKS, np.zeros((3, 4, 50, 30), np.int32))
    analysis = load_analysis(directory)
    assert analysis.n_frames == 3
    assert analysis.dimensionality == "3D"
    assert analysis.masks.shape == (3, 4, 50, 30)

    manifest["input"].update(axes="ZYX", shape=[4, 50, 30])
    export.write_json(directory / pipeline.F_MANIFEST, manifest)
    assert load_analysis(directory).n_frames == 1, "a single volume is one time point"


def test_incomplete_directory_is_not_complete(tmp_path):
    directory = tmp_path / "half"
    directory.mkdir()
    (directory / "masks.npz").write_bytes(b"")
    assert not analysis_is_complete(directory)


def test_export_bundle_copies_the_result_files(tmp_path):
    directory = tmp_path / "run"
    _write_minimal_analysis(directory)
    analysis = load_analysis(directory)
    destination = tmp_path / "exported"

    written = export_bundle(analysis, destination)

    names = {p.name for p in written}
    assert {"tracks.csv", "track_summary.csv", "run.json", "masks.npz",
            "recovery_attempts.csv", F_MSD, "tracking_events.csv"} <= names
    assert (destination / "tracks.csv").read_text(encoding="utf-8").startswith("track_id,")
    # A directory works as well as a loaded analysis.
    again = export_bundle(directory, tmp_path / "again")
    assert {p.name for p in again} == names


def test_export_one_track_as_csv_and_xlsx(tmp_path):
    directory = tmp_path / "run"
    _write_minimal_analysis(directory)
    analysis = load_analysis(directory)

    csv_path = export_track(analysis, 1, tmp_path / "t1.csv")
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(export.TRACK_COLUMNS) and len(lines) == 3

    xlsx_path = export_track(analysis, 1, tmp_path / "t1.xlsx", fmt="xlsx",
                             reference_point_px=(10.0, 0.0))
    with zipfile.ZipFile(xlsx_path) as zf:
        workbook = zf.read("xl/workbook.xml").decode("utf-8")
        sheet1 = zf.read("xl/worksheets/sheet1.xml").decode("utf-8")
    for name in ("Track 1", "Summary", "MSD", "Run"):
        assert f'name="{name}"' in workbook
    # D2R for this export only: 20 px and 40 px from (10, 0).
    assert f"<v>{round(20 * 0.4671, 9)!r}</v>" in sheet1
    assert analysis.rows_for_track(1)[0].get("distance_from_reference_um") is None

    with pytest.raises(ValueError):
        export_track(analysis, 1, tmp_path / "t1.json", fmt="json")


def test_whole_analysis_exports(tmp_path):
    directory = tmp_path / "run"
    _write_minimal_analysis(directory)
    analysis = load_analysis(directory)
    for writer, columns in (
        (export_tracks_csv, export.TRACK_COLUMNS),
        (export_summaries_csv, export.SUMMARY_COLUMNS),
        (export_msd_csv, export.MSD_COLUMNS),
    ):
        path = writer(analysis, tmp_path / f"{writer.__name__}.csv")
        rows = read_table(path)
        assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(columns)
        assert rows and rows[0]["track_id"] == 1


# --------------------------------------------------------------------------
# Atomic writing
# --------------------------------------------------------------------------


def test_csv_has_a_header_even_with_no_rows(tmp_path):
    path = export.write_csv(tmp_path / "empty.csv", export.TRACK_COLUMNS, [])
    text = path.read_text(encoding="utf-8")
    assert text.strip() == ",".join(export.TRACK_COLUMNS)
    assert read_table(path) == []


def test_writing_leaves_no_temporary_files(tmp_path):
    export.write_csv(tmp_path / "a.csv", ["x"], [{"x": 1}])
    export.write_json(tmp_path / "b.json", {"x": 1})
    export.save_masks(tmp_path / "c.npz", np.zeros((1, 2, 2), np.int32))
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_nan_and_infinity_are_written_as_empty(tmp_path):
    path = export.write_csv(
        tmp_path / "n.csv", ["a", "b", "c"],
        [{"a": float("nan"), "b": float("inf"), "c": 1.5}],
    )
    line = path.read_text(encoding="utf-8").splitlines()[1]
    assert line == ",,1.5"


def test_manifest_json_survives_numpy_types(tmp_path):
    path = export.write_json(
        tmp_path / "m.json",
        {"i": np.int64(3), "f": np.float32(1.5), "a": np.arange(3), "nan": np.float64("nan")},
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["i"] == 3
    assert data["f"] == pytest.approx(1.5)
    assert data["a"] == [0, 1, 2]
    assert data["nan"] is None


def test_masks_round_trip(tmp_path):
    masks = np.random.default_rng(0).integers(0, 5, size=(3, 8, 9)).astype(np.int32)
    path = export.save_masks(tmp_path / "m.npz", masks)
    assert np.array_equal(export.load_masks(path), masks)


# --------------------------------------------------------------------------
# Configuration persistence
# --------------------------------------------------------------------------


def test_config_round_trips_through_json():
    config = RunConfig()
    config.tracking.max_gap = 5
    config.segmentation.cellprob_threshold = -1.5
    config.calibration.pixel_size_um = 0.25
    config.confinement.mode = "angle"
    config.confinement.angle_deg = 87.0

    restored = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    assert restored.tracking.max_gap == 5
    assert restored.segmentation.cellprob_threshold == -1.5
    assert restored.calibration.pixel_size_um == 0.25
    assert restored.confinement.mode == "angle"
    assert restored.confinement.angle_deg == 87.0


def test_config_tolerates_unknown_and_missing_keys():
    """A project saved by an older or newer version must still open."""
    data = {
        "tracking": {"max_gap": 2, "a_setting_from_the_future": 9},
        "segmentation": {},
        "unknown_section": {"x": 1},
    }
    config = RunConfig.from_dict(data)
    assert config.tracking.max_gap == 2
    assert config.tracking.max_speed_um_per_min == 5.0  # default preserved
    assert config.segmentation.flow_threshold == 0.4


# --------------------------------------------------------------------------
# Command line behaviour
# --------------------------------------------------------------------------


def test_a_bare_file_argument_opens_the_application():
    """What a double-click and the file association do must show the app."""
    from corridor.cli import build_parser, wants_interface

    parser = build_parser()
    assert wants_interface(parser.parse_args([])) is True
    assert wants_interface(parser.parse_args(["movie.tif"])) is True
    assert wants_interface(parser.parse_args(["--gui", "movie.tif"])) is True


def test_naming_an_output_directory_runs_headless():
    from corridor.cli import build_parser, wants_interface

    parser = build_parser()
    assert wants_interface(parser.parse_args(["movie.tif", "-o", "out"])) is False
    assert wants_interface(parser.parse_args(["movie.tif", "--headless"])) is False
    # An explicit --gui still wins, even with an output directory.
    assert wants_interface(parser.parse_args(["--gui", "movie.tif", "-o", "out"])) is True


def test_the_transition_pipeline_keeps_the_unlinked_jump(tmp_path, vertical_axis):
    """pipeline.save_result must write dx/dy, and loading must keep them.

    Schema 2 has no along/across columns and ``write_csv`` drops unknown keys,
    so writing the 1.x tracker's along/across under their v1 names lost the
    jump for good.  The run this writes has no ``schema_version`` yet, so it
    is loaded through the v1 upgrade, which must not blank what is there.
    """
    import types
    from dataclasses import fields as dc_fields

    from corridor.core.config import Scale
    from corridor.core.tracking import UnlinkedStart

    names = {f.name for f in dc_fields(UnlinkedStart)}
    jump = (
        {"along_px": 11.0, "across_px": 20.0}  # 1.x tracker: axis frame
        if "along_px" in names
        else {"dx_px": -20.0, "dy_px": 11.0}  # 2.0 tracker: image frame
    )
    unlinked = UnlinkedStart(
        track_id=3, frame=5, candidate_track_id=2, candidate_last_frame=3, gap_frames=2,
        distance_px=22.8, speed_um_per_min=0.5, cost_chi2=40.0, **jump,
    )
    result = types.SimpleNamespace(
        scale=Scale.from_values(0.5, 10.0),
        metadata=types.SimpleNamespace(source_frames=list(range(6))),
        all_detections=[], rows=[], summaries=[], events=[], issues=[], recovery=None,
        segmentation=types.SimpleNamespace(diagnostics=[], masks=np.zeros((6, 8, 8), np.int32)),
        unlinked=[unlinked], axis=vertical_axis,
        manifest={"input": {"shape_tyx": [6, 8, 8]}, "confinement": vertical_axis.to_dict()},
    )
    pipeline.save_result(result, tmp_path / "run")

    (written,) = read_table(tmp_path / "run" / pipeline.F_UNLINKED)
    assert written["dx_px"] == pytest.approx(-20.0) and written["dy_px"] == pytest.approx(11.0)
    assert written["implied_speed_um_per_hr"] == pytest.approx(30.0)
    (loaded,) = load_analysis(tmp_path / "run").unlinked
    assert loaded["dx_px"] == pytest.approx(-20.0) and loaded["dy_px"] == pytest.approx(11.0)
    assert set(loaded) == set(export.UNLINKED_COLUMNS)
