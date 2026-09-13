"""Local project storage: it must survive closing the application."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from corridor.core import export, pipeline
from corridor.core.config import RunConfig
from corridor.store import db
from corridor.store.project import (
    analysis_is_complete,
    export_bundle,
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


def _write_minimal_analysis(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    export.write_csv(
        directory / pipeline.F_TRACKS,
        export.TRACK_COLUMNS,
        [
            {"track_id": 1, "frame": 0, "x_px": 10.0, "y_px": 20.0, "area_px": 100.0,
             "gap_frames": 0, "n_observations": 2},
            {"track_id": 1, "frame": 1, "x_px": 10.0, "y_px": 40.0, "area_px": 110.0,
             "gap_frames": 1, "speed_um_per_min": 0.47, "n_observations": 2},
        ],
    )
    export.write_csv(
        directory / pipeline.F_SUMMARY,
        export.SUMMARY_COLUMNS,
        [{"track_id": 1, "n_observations": 2, "first_frame": 0, "last_frame": 1}],
    )
    export.write_csv(directory / pipeline.F_DETECTIONS, export.DETECTION_COLUMNS, [])
    export.write_csv(directory / pipeline.F_DIAGNOSTICS, export.DIAGNOSTIC_COLUMNS, [])
    export.write_csv(directory / pipeline.F_QC, export.QC_COLUMNS, [])
    export.write_json(
        directory / pipeline.F_MANIFEST,
        {
            "input": {"path": "C:/x/movie.tif", "shape_tyx": [2, 50, 30],
                      "source_frames": [42, 43]},
            "calibration": {"pixel_size_um": 0.4671, "frame_interval_min": 20.0069},
            "confinement": {"ux": 0.0, "uy": 1.0, "channels": []},
            "results": {"n_tracks": 1},
        },
    )
    export.save_masks(directory / pipeline.F_MASKS, np.zeros((2, 50, 30), np.int32))


def test_saved_analysis_round_trip(tmp_path):
    directory = tmp_path / "run"
    _write_minimal_analysis(directory)
    assert analysis_is_complete(directory)

    analysis = load_analysis(directory)
    assert analysis.n_frames == 2
    assert analysis.pixel_size_um == pytest.approx(0.4671)
    assert analysis.frame_interval_min == pytest.approx(20.0069)
    assert analysis.source_frames == [42, 43]
    assert analysis.track_ids() == [1]
    assert len(analysis.rows_for_track(1)) == 2
    assert len(analysis.rows_for_frame(1)) == 1
    assert analysis.summary_for(1)["n_observations"] == 2
    assert analysis.masks.shape == (2, 50, 30)


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
    assert {"tracks.csv", "track_summary.csv", "run.json", "masks.npz"} <= names
    assert (destination / "tracks.csv").read_text(encoding="utf-8").startswith("track_id,")


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
