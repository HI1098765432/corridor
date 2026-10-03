"""run.json (schema 2), and the joins between detections.csv, tracks.csv and recovery_attempts.csv.

Both of these were broken in 1.x by changes that every other test happily
accepted, because nothing fast built a manifest or checked that the CSVs refer
to the same objects. A defect that makes every real run fail should not need a
real run to find.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from corridor.core import export, model_registry, pipeline
from corridor.core.config import RunConfig
from corridor.core.detections import detections_to_rows
from corridor.core.geometry import GEOMETRY_FROM_RIDGES, ChannelGeometry, Lane
from corridor.core.imaging import SOURCE_USER, Calibrated, StackMetadata
from corridor.core.measurements import TrackSummary, frame_rows, msd_rows, summarise
from corridor.core.model_registry import ResolvedModel
from corridor.core.recovery import SOURCE_WINDOW, RecoveryAttempt, RecoveryResult
from corridor.core.segmentation import SegmentationOutput
from corridor.core.tracking import Observation, Track, track_detections
from corridor.store.project import read_table

from conftest import FRAME_INTERVAL_MIN, PIXEL_SIZE_UM, make_detection, straight_track

#: Contract §2: exactly these keys, so a reader can rely on every one.
MODEL_KEYS = {
    "model_id", "model_version", "architecture", "sha256", "cellpose_version",
    "training_dataset_version", "developer_override",
}


@pytest.fixture
def metadata(tmp_path) -> StackMetadata:
    return StackMetadata(
        path=tmp_path / "synthetic.tif",
        n_frames=8,
        height=324,
        width=90,
        dtype="uint16",
        axes_raw="TYX",
        axes_interpretation="TYX",
        pixel_size_um=Calibrated(PIXEL_SIZE_UM, SOURCE_USER),
        frame_interval_min=Calibrated(FRAME_INTERVAL_MIN, SOURCE_USER),
        axes="TYX",
        axes_used="TYX",
    )


@pytest.fixture
def geometry() -> ChannelGeometry:
    """One lane measured from walls, as in the narrow 052924_t* crops."""
    lane = Lane(index=0, origin=(45.0, 0.0), direction=(0.0, 1.0), half_width_px=45.0)
    return ChannelGeometry(
        lanes=[lane], source=GEOMETRY_FROM_RIDGES, confidence=0.95, applied=True,
        image_shape=(324, 90),
    )


def production_model() -> ResolvedModel:
    """The registry's production entry, without needing the weights on disk."""
    spec = model_registry.production_spec("2D")
    return ResolvedModel(spec=spec, path=Path("models") / spec.filename, sha256=spec.sha256)


def segmentation_output(detections, model: ResolvedModel | None = None) -> SegmentationOutput:
    return SegmentationOutput(
        masks=np.zeros((8, 324, 90), np.int32),
        raw_masks=np.zeros((8, 324, 90), np.int32),
        detections=list(detections),
        diagnostics=[],
        model_path=str(model.path) if model else "",
        model_sha256=model.sha256 if model else None,
        cellpose_version="3.1.1.3",
        used_gpu=False,
        passes_per_frame=1,
        model=model,
    )


def build(metadata, geometry, scale, tracking_config, primary, recovery=None):
    """The pipeline's own sequence after segmentation, on hand-made detections."""
    config = RunConfig()
    config.tracking = tracking_config
    detections = list(primary) + (list(recovery.detections) if recovery else [])
    tracks, events = track_detections(
        detections, metadata.n_frames, scale, tracking_config, geometry=geometry
    )
    msd = msd_rows(tracks, scale)
    summaries = summarise(tracks, scale, msd_rows=msd)
    segmentation = segmentation_output(primary, production_model())
    manifest = pipeline.build_manifest(
        config, metadata, scale, geometry, segmentation, tracks, summaries,
        elapsed_s=1.0, output_dir=None, recovery=recovery,
    )
    return pipeline.AnalysisResult(
        config=config, metadata=metadata, scale=scale, geometry=geometry,
        segmentation=segmentation, tracks=tracks, events=events,
        rows=frame_rows(tracks, scale), summaries=summaries, issues=[],
        recovery=recovery, msd=msd, manifest=manifest, model=segmentation.model,
    )


def recovered(primary, *where: tuple[int, float]) -> list:
    """Windowed-tier detections whose crop labels collide on purpose, relabelled
    exactly as the pipeline relabels one recovery pass."""
    found = []
    for frame, y in where:
        det = make_detection(frame, 45.0, y, label=1)
        det.source = SOURCE_WINDOW
        det.confidence = 0.8
        found.append(det)
    pipeline._relabel_recovered(primary, found)
    return found


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------


def test_a_manifest_can_be_built_at_all(metadata, geometry, scale, tracking_config):
    """The regression: a name error here failed every run and no fast test saw it."""
    result = build(metadata, geometry, scale, tracking_config, straight_track(6))
    m = result.manifest
    assert m["schema_version"] == 2 == export.SCHEMA_VERSION
    assert m["application"]["name"]
    assert m["segmentation"]["ensemble"] == "off"
    assert m["segmentation"]["ensemble_passes"] == 1


def test_the_manifest_is_json_with_no_bare_nan(metadata, geometry, scale, tracking_config):
    """``NaN`` is not JSON; numpy scalars subclass float and slip past a ``default`` hook."""
    result = build(metadata, geometry, scale, tracking_config, straight_track(6))
    text = json.dumps(export._sanitize(result.manifest), default=str, allow_nan=False)
    assert "NaN" not in text


def test_the_model_block_is_the_contract_s(metadata, geometry, scale, tracking_config):
    result = build(metadata, geometry, scale, tracking_config, straight_track(6))
    block = result.manifest["model"]
    assert set(block) == MODEL_KEYS
    spec = model_registry.production_spec("2D")
    assert block["model_id"] == spec.model_id
    assert block["sha256"] == spec.sha256
    assert block["developer_override"] is False


def test_an_imported_segmentation_has_no_model_block(metadata, geometry, scale, tracking_config):
    tracks, _ = track_detections(straight_track(6), 8, scale, tracking_config, geometry=geometry)
    manifest = pipeline.build_manifest(
        RunConfig(), metadata, scale, geometry, segmentation_output(straight_track(6)),
        tracks, summarise(tracks, scale), elapsed_s=0.0, output_dir=None,
    )
    assert manifest["model"] is None


def test_channel_geometry_replaces_confinement(metadata, geometry, scale, tracking_config):
    """No axis anywhere: lanes, their source, and whether the gate was applied."""
    m = build(metadata, geometry, scale, tracking_config, straight_track(6)).manifest
    assert "confinement" not in m
    geo = m["channel_geometry"]
    assert geo["source"] == GEOMETRY_FROM_RIDGES
    assert geo["applied"] is True and geo["lane_gate_applied"] is True
    assert geo["n_lanes"] == 1 and len(geo["lanes"]) == 1
    assert geo["channel_constraint"] == "auto"

    def keys(value):
        if isinstance(value, dict):
            for k, v in value.items():
                yield k
                yield from keys(v)
        elif isinstance(value, list):
            for v in value:
                yield from keys(v)

    found = set(keys(m))
    for gone in ("ux", "uy", "angle_deg", "net_along_um", "along_channel_px", "confinement"):
        assert gone not in found, f"the v1 axis key {gone!r} is still in run.json"


def test_input_and_calibration_carry_axes_shape_and_z(metadata, geometry, scale, tracking_config):
    m = build(metadata, geometry, scale, tracking_config, straight_track(6)).manifest
    assert m["dimensionality"] == "2D"
    assert m["input"]["axes"] == "TYX"
    assert m["input"]["shape"] == [8, 324, 90]
    assert m["input"]["dimensionality"] == "2D"
    cal = m["calibration"]
    assert cal["pixel_size_um"] == PIXEL_SIZE_UM and cal["pixel_size_um_source"] == SOURCE_USER
    # A 2-D movie has no Z step, and nothing assumes one.
    assert cal["z_step_um"] is None and cal["anisotropy"] is None
    assert "z_step_um_source" in cal


def test_the_tracking_block_records_what_ran(metadata, geometry, scale, tracking_config):
    trk = build(metadata, geometry, scale, tracking_config, straight_track(6)).manifest["tracking"]
    # The gate is derived (2U), never the stored legacy number.
    assert trk["gate_chi2"] == 2 * trk["unmatched_chi2"]
    assert trk["max_delta_frames"] == trk["max_gap"] + 1
    # v1-only fields are recorded apart, so nothing implies they were applied.
    assert "sigma_along_um" not in trk and "sigma_along_um" in trk["v1_fields_not_applied"]


def test_results_report_primary_and_total_detections(metadata, geometry, scale, tracking_config):
    """``n_detections`` is the row count of detections.csv, not just the first pass."""
    primary = straight_track(6, step=20.0)
    result = build(
        metadata, geometry, scale, tracking_config, primary,
        recovery=RecoveryResult(detections=recovered(primary, (6, 140.0))),
    )
    res = result.manifest["results"]
    assert res["n_detections"] == len(result.all_detections) == len(primary) + 1
    assert res["n_detections_primary"] == len(primary) == result.n_primary_detections
    assert res["n_detections_recovered"] == 1
    assert result.n_detections == res["n_detections"]


def test_mean_speed_is_in_both_units_and_the_estimator_is_named(
    metadata, geometry, scale, tracking_config
):
    res = build(metadata, geometry, scale, tracking_config, straight_track(6)).manifest["results"]
    assert res["mean_speed_um_per_hr"] == pytest.approx(res["mean_speed_um_per_min"] * 60)
    assert res["mean_net_speed_um_per_hr"] == pytest.approx(res["mean_net_speed_um_per_min"] * 60)
    expected = 20.0 * PIXEL_SIZE_UM / FRAME_INTERVAL_MIN  # a straight run: all three agree
    assert res["mean_speed_um_per_min"] == pytest.approx(expected)
    assert res["mean_net_speed_um_per_min"] == pytest.approx(expected)
    for name in ("median_speed", "median_net_speed"):
        assert res[f"{name}_um_per_min"] == pytest.approx(expected)
        assert res[f"{name}_um_per_hr"] == pytest.approx(expected * 60)
    estimators = res["speed_estimators"]
    assert estimators["robust_estimator"] == "median_net_speed"
    assert "NOT robust" in estimators["mean_net_speed"]
    assert res["n_tracks_with_speed"] == 1


def test_one_outlier_fragment_moves_the_mean_and_not_the_median():
    """The failure that actually happens: a spurious 2-observation track.

    Shaped on 052924_1 (2.0 defaults), where one 3.80 um/min fragment lifted
    the mean net speed of 12 tracks by 48 %.  Here: four cells at 0.5 um/min
    and one fragment at 4.0.
    """
    from types import SimpleNamespace

    def track(v):
        return SimpleNamespace(mean_speed_um_per_min=v, net_speed_um_per_min=v)

    cells = [track(0.5) for _ in range(4)]
    clean = pipeline.speed_results(cells)
    noisy = pipeline.speed_results(cells + [track(4.0)])
    assert clean["mean_net_speed_um_per_min"] == pytest.approx(0.5)
    assert noisy["mean_net_speed_um_per_min"] == pytest.approx(1.2)  # +140 %
    robust = noisy["speed_estimators"]["robust_estimator"]
    assert noisy[f"{robust}_um_per_min"] == pytest.approx(0.5) == clean[f"{robust}_um_per_min"]
    # A one-observation track has no speed and is not counted.
    single = pipeline.speed_results(cells + [SimpleNamespace(mean_speed_um_per_min=None,
                                                             net_speed_um_per_min=None)])
    assert single["n_tracks_with_speed"] == 4
    assert pipeline.speed_results([])["median_net_speed_um_per_min"] is None


def test_a_pixel_size_override_leaves_a_calibration_block_that_agrees_with_itself(
    metadata, geometry, scale, tracking_config
):
    """Review finding: the override replaced X while Y stayed the file's value.

    The file reports square 0.467 um pixels; the user enters 0.639.
    """
    from corridor.core.config import CalibrationConfig

    file_px = Calibrated(0.46706, "nd2_info")
    metadata.pixel_size_um = file_px
    metadata.pixel_size_y_um = file_px
    reported = pipeline.ReportedCalibration.of(metadata)
    pixel, interval, used_scale = pipeline.effective_calibration(
        metadata, CalibrationConfig(pixel_size_um=0.639)
    )
    metadata.pixel_size_um, metadata.frame_interval_min = pixel, interval
    tracks, _ = track_detections(straight_track(6), 8, used_scale, tracking_config, geometry=geometry)
    manifest = pipeline.build_manifest(
        RunConfig(), metadata, used_scale, geometry, segmentation_output(straight_track(6)),
        tracks, summarise(tracks, used_scale), elapsed_s=0.0, output_dir=None, reported=reported,
    )
    cal = manifest["calibration"]
    assert cal["pixel_size_um"] == cal["pixel_size_y_um"] == 0.639
    assert cal["pixel_size_um_source"] == cal["pixel_size_y_um_source"] == SOURCE_USER
    assert cal["anisotropic_pixels"] is False
    file = cal["reported_by_file"]
    assert file["pixel_size_x_um"] == file["pixel_size_y_um"] == 0.46706
    assert file["pixel_size_x_um_source"] == file["pixel_size_y_um_source"] == "nd2_info"


def test_an_anisotropic_file_stays_flagged_whatever_is_entered(
    metadata, geometry, scale, tracking_config
):
    """One entered number cannot make non-square pixels square (imaging.read_metadata)."""
    metadata.pixel_size_y_um = Calibrated(PIXEL_SIZE_UM * 1.2, "tiff_tag")
    metadata.anisotropic_pixels = True
    cal = build(metadata, geometry, scale, tracking_config, straight_track(6)).manifest["calibration"]
    assert cal["anisotropic_pixels"] is True
    # Measured with one size; the file's Y size is kept where it is labelled as the file's.
    assert cal["pixel_size_y_um"] == cal["pixel_size_um"] == PIXEL_SIZE_UM
    assert cal["reported_by_file"]["pixel_size_y_um"] == pytest.approx(PIXEL_SIZE_UM * 1.2)


def test_building_a_manifest_does_not_import_torch():
    """Versions come from package metadata; ``import torch`` costs seconds and hundreds of MB."""
    code = (
        "import sys\n"
        "import numpy as np\n"
        "from corridor.core import pipeline\n"
        "from corridor.core.segmentation import SegmentationOutput\n"
        "seg = SegmentationOutput(np.zeros((1, 2, 2), np.int32), np.zeros((1, 2, 2), np.int32),"
        " [], [], '', None, '', False)\n"
        "env = pipeline._environment(seg)\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "assert env['gpu_available'] is None\n"
        "print(env['cellpose'])\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
        env={**__import__("os").environ, "PYTHONPATH": str(Path(pipeline.__file__).parents[2])},
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().startswith("3.")


# --------------------------------------------------------------------------
# detections.csv and tracks.csv describe the same objects
# --------------------------------------------------------------------------


def test_every_track_row_finds_its_detection(metadata, geometry, scale, tracking_config):
    """The join a user will obviously try, on a result containing recovery.

    A recovered position carries a label from the crop it was found in, which
    routinely collides with a primary label in the same frame. If the two files
    disagree, the join silently returns another cell's measurements.
    """
    primary = straight_track(6, step=20.0)
    result = build(
        metadata, geometry, scale, tracking_config, primary,
        recovery=RecoveryResult(detections=recovered(primary, (6, 140.0))),
    )
    assert len(result.all_detections) == len(primary) + 1
    rows = detections_to_rows(result.all_detections)
    index = {(r["frame"], r["label"]) for r in rows}
    missing = [r for r in result.rows if (r["frame"], r["det_label"]) not in index]
    assert missing == [], "a track row points at a detection that is not exported"
    # And the recovered position really is in a track, by its new label.
    assert any(r["detection_source"] == SOURCE_WINDOW for r in result.rows)


def test_no_frame_exports_two_detections_with_the_same_label(
    metadata, geometry, scale, tracking_config
):
    primary = straight_track(6, step=20.0)
    primary.append(make_detection(3, 45.0, 250.0, label=2))
    found = recovered(primary, (3, 80.0), (3, 300.0))
    result = build(
        metadata, geometry, scale, tracking_config, primary,
        recovery=RecoveryResult(detections=found),
    )
    pairs = [(d.frame, d.label) for d in result.all_detections]
    assert len(pairs) == len(set(pairs))
    assert sorted(d.label for d in found) == [3, 4]


def test_a_result_without_recovery_exports_exactly_the_primary_detections(
    metadata, geometry, scale, tracking_config
):
    primary = straight_track(6, step=20.0)
    result = build(metadata, geometry, scale, tracking_config, primary)
    assert len(result.all_detections) == len(primary)


def test_the_summary_columns_match_the_dataclass(metadata, geometry, scale, tracking_config):
    """A column list that drifts from the fields writes one column's value
    under another column's name, which no reader would ever suspect."""
    assert export.SUMMARY_COLUMNS == [f.name for f in fields(TrackSummary)]
    result = build(metadata, geometry, scale, tracking_config, straight_track(6))
    assert list(result.summaries[0].to_row()) == export.SUMMARY_COLUMNS


# --------------------------------------------------------------------------
# recovery_attempts.csv joins tracks.csv on the FINAL track id (critique C4)
# --------------------------------------------------------------------------


def _track(track_id: int, *obs: tuple, ) -> Track:
    """A track from ``(frame, det_label[, Detection])`` triples."""
    return Track(
        id=track_id,
        observations=[
            Observation(frame=o[0], x=45.0, y=20.0 * o[0], area_px=800.0, det_label=o[1],
                        cost=None, gap_frames=1, detection=o[2] if len(o) > 2 else None)
            for o in obs
        ],
    )


def test_recovery_rows_carry_the_final_track_id():
    """First-pass ids and final ids differ; the attempt is re-keyed to the final one.

    The first pass saw this cell as track 7 (frames 0, 1, 3), with frame 2
    missing; recovery found it at frame 2 and relabelled it 5; the second
    pass renumbered the cell as track 2.  A second attempt found nothing at
    frame 5 of track 8, whose bracketing observations ended up in final
    track 1.  The tracker's ``id_map`` cannot do this: it maps the second
    pass's own stage-1 ids, and 7 and 8 are first-pass ids.
    """
    found = make_detection(2, 45.0, 40.0, label=5)
    final = [_track(1, (4, 2), (6, 2)), _track(2, (0, 1), (1, 1), (2, 5, found), (3, 1))]
    result = RecoveryResult(
        detections=[found],
        attempts=[
            RecoveryAttempt(7, 2, 45.0, 40.0, found=True, source=SOURCE_WINDOW,
                            bracket=(1, 1, 3, 1), detection=found),
            RecoveryAttempt(8, 5, 45.0, 100.0, found=False, detail="nothing",
                            bracket=(4, 2, 6, 2), duplicate_of_label=3),
        ],
    )
    rows = pipeline.recovery_attempt_rows(result, final)
    assert [r["track_id"] for r in rows] == [2, 1]
    assert [r["first_pass_track_id"] for r in rows] == [7, 8]
    assert rows[0]["det_label"] == 5 and rows[1]["det_label"] is None
    assert (rows[0]["bracket_frame_before"], rows[0]["bracket_label_before"]) == (1, 1)
    assert (rows[0]["bracket_frame_after"], rows[0]["bracket_label_after"]) == (3, 1)
    assert (rows[1]["bracket_frame_before"], rows[1]["bracket_frame_after"]) == (4, 6)
    assert rows[1]["duplicate_of_label"] == 3
    # The attempts themselves now hold the final ids, so the in-memory result
    # and the file agree.
    assert [a.track_id for a in result.attempts] == [2, 1]


def test_a_found_attempt_follows_its_detection_not_its_bracket():
    """The recovered cell can end up in another track than the one it was sought for."""
    found = make_detection(2, 45.0, 40.0, label=5)
    final = [_track(1, (1, 1), (3, 1)), _track(2, (2, 5, found))]
    attempt = RecoveryAttempt(7, 2, 45.0, 40.0, found=True, bracket=(1, 1, 3, 1), detection=found)
    (row,) = pipeline.recovery_attempt_rows(RecoveryResult([found], [attempt]), final)
    assert row["track_id"] == 2 and row["det_label"] == 5


def test_an_attempt_no_final_track_holds_gets_no_id_rather_than_a_wrong_one():
    attempt = RecoveryAttempt(7, 2, 45.0, 40.0, found=False, bracket=(1, 1, 3, 1))
    (row,) = pipeline.recovery_attempt_rows(
        RecoveryResult(attempts=[attempt]), [_track(1, (1, 9), (3, 9))]
    )
    assert row["track_id"] is None and row["first_pass_track_id"] == 7
    assert "no final track" in row["detail"]


def test_save_result_never_writes_first_pass_ids(tmp_path, metadata, geometry, scale, tracking_config):
    """A result assembled without the pipeline's re-keying is re-keyed on writing."""
    primary = straight_track(6, step=20.0)
    result = build(metadata, geometry, scale, tracking_config, primary)
    (track,) = result.tracks
    obs = track.observations
    attempt = RecoveryAttempt(
        99, 3, 45.0, 60.0, found=False,
        bracket=(obs[2].frame, obs[2].det_label, obs[3].frame, obs[3].det_label),
    )
    result.recovery = RecoveryResult(attempts=[attempt])
    pipeline.save_result(result, tmp_path / "run")
    (row,) = read_table(tmp_path / "run" / pipeline.F_RECOVERY)
    assert row["track_id"] == track.id and row["first_pass_track_id"] == 99


def test_recovery_attempts_csv_has_the_bracket_columns(tmp_path):
    columns = pipeline.recovery_columns()
    for name in ("track_id", *pipeline.RECOVERY_V2_COLUMNS):
        assert columns.count(name) == 1
    path = export.write_csv(tmp_path / "r.csv", columns, [])
    assert path.read_text(encoding="utf-8").startswith("track_id,frame,")
    # Every key an attempt reports is a column: write_csv drops the others.
    row = RecoveryAttempt(1, 2, 0.0, 0.0, found=False, bracket=(1, 1, 3, 1)).to_row()
    assert set(row) <= set(columns)
    # A key an attempt reports beyond the known columns still reaches the file.
    assert pipeline.recovery_columns([{"track_id": 1, "tier_seconds": 0.2}])[-1] == "tier_seconds"


def test_scale_from_effective_calibration_carries_z_only_for_3d(tmp_path, metadata):
    from corridor.core.config import CalibrationConfig

    _, _, scale = pipeline.effective_calibration(metadata, CalibrationConfig(z_step_um=2.0))
    assert scale.z_step_um is None, "a 2-D movie must not acquire a Z step from an override"
    metadata.axes = "TZYX"
    metadata.n_slices = 4
    _, _, scale = pipeline.effective_calibration(metadata, CalibrationConfig(z_step_um=2.0))
    assert scale.z_step_um == 2.0 and scale.anisotropy == pytest.approx(2.0 / PIXEL_SIZE_UM)
