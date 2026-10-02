"""Every output a 1.x release wrote must still open, upgraded in memory (§7).

The fixtures in ``tests/fixtures/legacy`` are synthetic numbers in the exact
file layout, column headers and ``run.json`` key structure of 1.0.0, 1.1.0
and 1.3.0 (1.2.0 wrote what 1.3.0 did); ``make_fixtures.py`` there says how
they were built.  The last test opens every real legacy run present on disk.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path

import pytest

from corridor.core import export
from corridor.core.config import RunConfig
from corridor.store import db
from corridor.store.project import (
    F_MSD,
    SavedAnalysis,
    export_bundle,
    export_tracks_csv,
    load_analysis,
    read_table,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "legacy"
VERSIONS = {"v1_0_0": "1.0.0", "v1_1_0": "1.1.0", "v1_3_0": "1.3.0"}


@pytest.fixture(params=sorted(VERSIONS))
def legacy(request) -> tuple[str, SavedAnalysis]:
    return VERSIONS[request.param], load_analysis(FIXTURES / request.param)


def rows_of(analysis, tid):
    return analysis.rows_for_track(tid)


def test_a_v1_run_is_recognised_and_upgraded(legacy):
    version, analysis = legacy
    assert analysis.schema_version == 1
    assert analysis.upgraded_from == version
    assert analysis.upgrade_notes and "re-segmented" in analysis.upgrade_notes[0]
    assert analysis.dimensionality == "2D"
    assert analysis.n_frames == 6
    assert analysis.masks.shape == (6, 60, 40)
    assert analysis.track_ids() == [1, 2, 3]


def test_upgraded_rows_have_the_v2_columns_and_no_axis(legacy):
    _, analysis = legacy
    for row in analysis.tracks:
        assert set(row) == set(export.TRACK_COLUMNS)
    for row in analysis.summaries:
        assert set(row) == set(export.SUMMARY_COLUMNS)


def test_per_hour_units_and_mtrackj_distances_are_derived(legacy):
    """Track 1: 6 px/frame at 0.5 µm/px and 10 min/frame = 18 µm/hr, missed at frame 3."""
    _, analysis = legacy
    rows = rows_of(analysis, 1)
    assert [r["frame"] for r in rows] == [0, 1, 2, 4, 5]
    assert [r["distance_from_previous_um"] for r in rows] == [None, 3.0, 3.0, 6.0, 3.0]
    assert [r["cumulative_path_um"] for r in rows] == [0.0, 3.0, 6.0, 12.0, 15.0]
    assert [r["distance_from_start_um"] for r in rows] == [0.0, 3.0, 6.0, 12.0, 15.0]
    assert rows[0]["speed_um_per_hr"] is None
    assert all(r["speed_um_per_hr"] == pytest.approx(18.0) for r in rows[1:])
    assert all(r["distance_from_reference_um"] is None for r in rows)

    (summary,) = [s for s in analysis.summaries if s["track_id"] == 1]
    assert summary["net_speed_um_per_hr"] == pytest.approx(18.0)
    assert summary["mean_speed_um_per_hr"] == pytest.approx(18.0)
    assert summary["duration_hr"] == pytest.approx(50.0 / 60.0)
    assert summary["n_gaps"] == 1 and summary["total_missing_frames"] == 1


def test_the_upgrade_reproduces_what_the_v1_file_said(legacy):
    """Same positions, same arithmetic: the shared quantities must agree."""
    version, analysis = legacy
    directory = FIXTURES / {v: k for k, v in VERSIONS.items()}[version]
    original = {(r["track_id"], r["frame"]): r for r in read_table(directory / "tracks.csv")}
    for row in analysis.tracks:
        v1 = original[(row["track_id"], row["frame"])]
        for key in ("speed_um_per_min", "vx_um_per_min", "vy_um_per_min", "x_um",
                    "elapsed_min", "speed_px_per_frame", "area_um2", "gap_frames"):
            if v1[key] is None:
                assert row[key] is None
            else:
                assert row[key] == pytest.approx(v1[key], abs=1e-8)
        assert row["distance_from_previous_um"] == (
            pytest.approx(v1["step_um"], abs=1e-8) if v1["step_um"] is not None else None
        )
    old_summary = {s["track_id"]: s for s in read_table(directory / "track_summary.csv")}
    for s in analysis.summaries:
        v1 = old_summary[s["track_id"]]
        for key in ("net_displacement_um", "path_length_um", "mean_speed_um_per_min",
                    "max_speed_um_per_min", "straightness", "duration_min", "mean_area_px"):
            if v1[key] is None:
                assert s[key] is None
            else:
                assert s[key] == pytest.approx(v1[key], abs=1e-8)
        assert s["flags"] == (v1["flags"] or "")


def test_msd_is_computed_from_the_positions(legacy):
    _, analysis = legacy
    curve = {r["lag_frames"]: r for r in analysis.msd_for_track(1)}
    # Frames 0, 1, 2, 4, 5: actual lags, with the pair counts that follow.
    assert {k: r["n_pairs"] for k, r in curve.items()} == {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}
    for lag, r in curve.items():
        assert r["msd_um2"] == pytest.approx((3.0 * lag) ** 2)
    # Only lag 1 has 3 pairs: too few lags for an exponent.
    assert analysis.summary_for(1)["msd_alpha"] is None
    assert analysis.msd_for_track(3) == []


def test_morphology_is_joined_back_from_detections(legacy):
    _, analysis = legacy
    first = rows_of(analysis, 1)[0]
    assert first["eccentricity"] == pytest.approx(0.98)
    assert first["major_axis_um"] == pytest.approx(15.0)
    assert first["perimeter_px"] is None, "v1 never measured a perimeter"


def test_fragments_and_flags_survive(legacy):
    _, analysis = legacy
    assert analysis.summary_for(3)["flags"] == "fragment"
    assert {r["track_flags"] for r in rows_of(analysis, 3)} == {"fragment"}


def test_provenance_and_recovery_from_1_1_0(legacy):
    version, analysis = legacy
    track2 = rows_of(analysis, 2)
    if version == "1.0.0":
        # 1.0.0 had no recovery and recorded no provenance: every position was primary.
        assert [r["frame"] for r in track2] == [1, 2]
        assert {r["detection_source"] for r in track2} == {"primary"}
        assert analysis.recovery == [] and analysis.unlinked == []
        return
    assert track2[-1]["detection_source"] == "intensity"
    assert track2[-1]["segmentation_confidence"] == pytest.approx(0.4)
    (attempt,) = analysis.recovery
    assert attempt["recovered"] is True and attempt["found_by"] == "intensity"
    assert len(analysis.events) == 6


def test_unlinked_along_across_becomes_dx_dy(legacy):
    version, analysis = legacy
    if version == "1.0.0":
        pytest.skip("1.0.0 wrote no unlinked_starts.csv")
    row = next(r for r in analysis.unlinked if r["track_id"] == 3)
    assert row["dx_px"] == pytest.approx(-20.0)
    assert row["dy_px"] == pytest.approx(11.0)
    assert "along_channel_px" not in row and "across_channel_px" not in row
    assert row["mahalanobis"] is None if "mahalanobis" in row else True
    assert row["implied_speed_um_per_hr"] == pytest.approx(row["implied_speed_um_per_min"] * 60)


def test_text_that_looks_like_a_number_stays_text(legacy):
    _, analysis = legacy
    assert analysis.diagnostic_for(3)["cellpose_message"] == "nan"
    assert analysis.diagnostic_for(0)["cellpose_message"] is None


def test_a_v1_bundle_keeps_its_record_and_gains_the_msd(legacy, tmp_path):
    version, analysis = legacy
    written = export_bundle(analysis, tmp_path / "bundle")
    names = {p.name for p in written}
    assert {"tracks.csv", "track_summary.csv", "run.json", "masks.npz", F_MSD} <= names
    if version != "1.0.0":
        assert "recovery_attempts.csv" in names
    source = analysis.directory / "tracks.csv"
    assert (tmp_path / "bundle" / "tracks.csv").read_bytes() == source.read_bytes()
    msd = read_table(tmp_path / "bundle" / F_MSD)
    assert msd and set(msd[0]) == set(export.MSD_COLUMNS)


def test_a_v1_run_exports_in_schema_2(legacy, tmp_path):
    _, analysis = legacy
    path = export_tracks_csv(analysis, tmp_path / "all.csv")
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(export.TRACK_COLUMNS)


def test_loading_never_writes_into_the_run(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(FIXTURES / "v1_3_0", run)
    before = {p.name: p.stat().st_mtime_ns for p in run.iterdir()}
    load_analysis(run)
    assert {p.name: p.stat().st_mtime_ns for p in run.iterdir()} == before


# --------------------------------------------------------------------------
# The database's copies (config_json, manifest_json)
# --------------------------------------------------------------------------


def test_the_database_copies_of_a_v1_project_still_load(tmp_path, monkeypatch):
    monkeypatch.setenv("CORRIDOR_DATA_DIR", str(tmp_path / "appdata"))
    config_json = json.loads((FIXTURES / "db_config_json_v1_3_0.json").read_text(encoding="utf-8"))
    manifest_json = json.loads((FIXTURES / "v1_3_0" / "run.json").read_text(encoding="utf-8"))

    store = db.Store()
    try:
        source = tmp_path / "movie.nd2"
        source.write_bytes(b"x")
        record = store.create_project(source)
        record.config, record.manifest = config_json, manifest_json
        store.update_project(record)
    finally:
        store.close()
    reopened = db.Store()
    try:
        again = reopened.get_project(record.id)
    finally:
        reopened.close()

    config = RunConfig.from_dict(again.config)
    assert config.tracking.max_gap == 3
    # The v1 confinement block reaches the v2 geometry; its axis does not.
    assert config.geometry.detect_walls is False
    assert config.geometry.min_channel_pitch_um == 20.0
    assert config.tracking.channel_constraint == "off"
    # A new project never inherits the old model choice or calibration.
    fresh = RunConfig.for_new_project(again.config)
    assert fresh.segmentation.ensemble_model_paths in ((), [])
    assert fresh.calibration.pixel_size_um is None

    from_db = SavedAnalysis(directory=tmp_path, manifest=again.manifest)
    assert from_db.n_frames == 6
    assert from_db.pixel_size_um == 0.5 and from_db.frame_interval_min == 10.0
    assert from_db.source_frames == [11, 12, 13, 14, 15, 16]


# --------------------------------------------------------------------------
# Every real legacy run on disk
# --------------------------------------------------------------------------


def _real_runs() -> list[Path]:
    """Runs under ``data/_runs`` and ``build/baseline_v1.3.0``.

    Looked for in this checkout and, when set, under ``CORRIDOR_LEGACY_ROOT``
    (a worktree has no data/ or build/ of its own).  Read only.
    """
    roots = [Path(__file__).resolve().parents[1]]
    if os.environ.get("CORRIDOR_LEGACY_ROOT"):
        roots.append(Path(os.environ["CORRIDOR_LEGACY_ROOT"]))
    found: dict[str, Path] = {}
    for root in roots:
        for pattern in ("data/_runs/*/run.json", "build/baseline_v1.3.0/*/run.json"):
            for manifest in glob.glob(str(root / pattern)):
                found.setdefault(str(Path(manifest).parent.resolve()), Path(manifest).parent)
    return sorted(found.values())


REAL_RUNS = _real_runs()


@pytest.mark.skipif(not REAL_RUNS, reason="no legacy runs on disk")
@pytest.mark.parametrize(
    "run", REAL_RUNS, ids=[f"{p.parent.name}/{p.name}" for p in REAL_RUNS]
)
def test_every_real_legacy_run_loads(run):
    analysis = load_analysis(run)
    assert analysis.schema_version == 1 and analysis.upgraded_from
    assert analysis.n_frames > 0
    assert analysis.masks is not None and analysis.masks.shape[0] == analysis.n_frames
    original = {(r["track_id"], r["frame"]): r for r in read_table(run / "tracks.csv")}
    assert len(analysis.tracks) == len(original)
    for row in analysis.tracks:
        assert set(row) == set(export.TRACK_COLUMNS)
        v1 = original[(row["track_id"], row["frame"])]
        if v1["speed_um_per_min"] is not None:
            assert row["speed_um_per_min"] == pytest.approx(v1["speed_um_per_min"], abs=1e-8)
            assert row["speed_um_per_hr"] == pytest.approx(v1["speed_um_per_min"] * 60, abs=1e-6)
    assert [s["track_id"] for s in analysis.summaries] == analysis.track_ids()
    multi = {s["track_id"] for s in analysis.summaries if s["n_observations"] >= 2}
    assert {r["track_id"] for r in analysis.msd} == multi
