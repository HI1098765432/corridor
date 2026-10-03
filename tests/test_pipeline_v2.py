"""The 2.0 pipeline end to end, without Cellpose.

Segmentation comes from an imported label image (the contract's other route,
and the only one for 3-D) or from a stand-in service that returns the same
labels with the registry's production model attached.  Everything after it --
calibration, lanes, axis-free tracking, recovery provenance, measurement,
QC and every file of schema 2 -- is the real code.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
import tifffile

from corridor.core import export, model_registry, pipeline
from corridor.core.config import RunConfig
from corridor.core.detections import extract_detections
from corridor.core.geometry import ChannelGeometry
from corridor.core.imaging import AmbiguousAxes
from corridor.core.model_registry import MODEL_UNAVAILABLE_MESSAGE, ModelUnavailable, ResolvedModel
from corridor.core.recovery import SOURCE_WINDOW, RecoveryAttempt, RecoveryResult
from corridor.core.segmentation import PROVENANCE_IMPORTED, PROVENANCE_MODEL, SegmentationOutput
from corridor.core.detections import FrameDiagnostics
from corridor.store.project import load_analysis, read_table

PIXEL_UM = 0.5
FRAME_S = 600.0  # 10 min
N_FRAMES = 8
HEIGHT, WIDTH = 320, 240
#: Bright vertical walls; the ridge detector returns one lane per wall group.
WALLS_X = (40, 120, 200)
#: Two cells, each in its own lane, moving down at different speeds (px/frame).
CELLS = {1: (40.0, 30.0, 15.0), 2: (120.0, 40.0, 10.0)}


def _ellipse(shape, cx, cy, a=30.0, b=5.5) -> np.ndarray:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    return ((cols - cx) / b) ** 2 + ((rows - cy) / a) ** 2 <= 1.0


def synthetic_movie(tmp_path: Path, *, drop: dict[int, set[int]] | None = None):
    """(movie path, labels path, labels array). ``drop`` removes cells from frames."""
    rng = np.random.default_rng(0)
    drop = drop or {}
    labels = np.zeros((N_FRAMES, HEIGHT, WIDTH), np.int32)
    movie = rng.normal(1000, 20, size=(N_FRAMES, HEIGHT, WIDTH)).astype(np.float32)
    for x in WALLS_X:
        movie[:, :, x - 2 : x + 3] += 2500.0
    for t in range(N_FRAMES):
        for label, (cx, cy, step) in CELLS.items():
            if t in drop.get(label, set()):
                continue
            mask = _ellipse((HEIGHT, WIDTH), cx + 1.0, cy + step * t)
            labels[t][mask] = label
            movie[t][mask] += 800.0
    movie_path = tmp_path / "movie.tif"
    tifffile.imwrite(
        movie_path, movie.astype(np.uint16), imagej=True,
        metadata={"axes": "TYX", "finterval": FRAME_S, "unit": "micron"},
        resolution=(1.0 / PIXEL_UM, 1.0 / PIXEL_UM),
    )
    labels_path = tmp_path / "labels.tif"
    tifffile.imwrite(labels_path, labels)
    return movie_path, labels_path, labels


def config_for(tmp_path: Path, movie: Path, labels: Path | None = None) -> RunConfig:
    config = RunConfig(input_path=str(movie), output_dir=str(tmp_path / "out"))
    if labels is not None:
        config.import_.labels_path = str(labels)
    return config


V2_FILES = (
    pipeline.F_TRACKS, pipeline.F_SUMMARY, pipeline.F_MSD, pipeline.F_DETECTIONS,
    pipeline.F_EVENTS, pipeline.F_QC, pipeline.F_RECOVERY, pipeline.F_UNLINKED,
    pipeline.F_DIAGNOSTICS, pipeline.F_MASKS, pipeline.F_MANIFEST,
)


# --------------------------------------------------------------------------
# Imported labels: the whole run, every file
# --------------------------------------------------------------------------


def test_an_imported_segmentation_runs_end_to_end(tmp_path):
    movie, labels_path, labels = synthetic_movie(tmp_path)
    result = pipeline.run_analysis(config_for(tmp_path, movie, labels_path))
    out = result.output_dir

    for name in V2_FILES:
        assert (out / name).exists(), f"{name} was not written"
    assert not (out / pipeline.F_RAW_MASKS).exists(), "an imported label file is its own raw record"

    m = json.loads((out / pipeline.F_MANIFEST).read_text(encoding="utf-8"))
    assert m["schema_version"] == 2
    assert m["model"] is None
    assert m["segmentation"]["provenance"] == PROVENANCE_IMPORTED
    assert m["segmentation"]["labels_sha256"] == model_registry.sha256_file(labels_path)
    assert "imported" in m["recovery"]["skipped"] and m["recovery"]["attempted"] == 0
    assert m["input"]["axes"] == "TYX" and m["input"]["shape"] == [N_FRAMES, HEIGHT, WIDTH]
    assert m["calibration"]["pixel_size_um"] == pytest.approx(PIXEL_UM)
    assert m["calibration"]["frame_interval_min"] == pytest.approx(FRAME_S / 60.0)

    # Lanes from the walls, gate applied, and each cell stays in its own lane.
    geo = m["channel_geometry"]
    assert geo["source"] == "channel_ridges" and geo["applied"] is True
    assert geo["n_lanes"] == len(WALLS_X)
    assert result.n_tracks == len(CELLS)
    for tr in result.tracks:
        assert len({o.channel for o in tr.observations}) == 1
        assert len({o.det_label for o in tr.observations}) == 1, "identities were swapped"

    # Speeds in both units from the one canonical velocity.
    for summary in result.summaries:
        (label,) = {o.det_label for o in next(t for t in result.tracks if t.id == summary.track_id).observations}
        expected = CELLS[label][2] * PIXEL_UM / (FRAME_S / 60.0)
        assert summary.net_speed_um_per_min == pytest.approx(expected, rel=0.02)
        assert summary.net_speed_um_per_hr == pytest.approx(summary.net_speed_um_per_min * 60)

    # detections.csv rows are what run.json counts; masks are T[Z]YX.
    assert len(read_table(out / pipeline.F_DETECTIONS)) == m["results"]["n_detections"]
    objects = sum(len(np.unique(frame)) - 1 for frame in labels)
    assert m["results"]["n_detections"] == result.n_detections == objects
    assert np.array_equal(export.load_masks(out / pipeline.F_MASKS), labels)

    # It reloads as schema 2, MSD included.
    saved = load_analysis(out)
    assert saved.schema_version == 2 and saved.msd
    assert {r["track_id"] for r in saved.msd} == {t.id for t in result.tracks}


def test_the_overlap_reconstructor_backend_runs_end_to_end(tmp_path):
    """The ``tracking.reconstructor == "overlap"`` backend (engine.reconstruct)
    is a production path, not just a benchmark: the whole pipeline runs on it
    and returns the same two correct, identity-stable, per-lane tracks the
    default Kalman backend returns on this movie."""
    from corridor.core.config import RECONSTRUCTOR_OVERLAP

    movie, labels_path, _ = synthetic_movie(tmp_path)
    config = config_for(tmp_path, movie, labels_path)
    config.tracking.reconstructor = RECONSTRUCTOR_OVERLAP
    result = pipeline.run_analysis(config)

    assert result.n_tracks == len(CELLS)
    for tr in result.tracks:
        assert len({o.channel for o in tr.observations}) == 1
        assert len({o.det_label for o in tr.observations}) == 1, "identities were swapped"
    # Speeds still come out right through the overlap backend.
    for summary in result.summaries:
        (label,) = {o.det_label for o in next(t for t in result.tracks if t.id == summary.track_id).observations}
        expected = CELLS[label][2] * PIXEL_UM / (FRAME_S / 60.0)
        assert summary.net_speed_um_per_min == pytest.approx(expected, rel=0.02)
    # And the run reloads as a complete schema-2 result.
    saved = load_analysis(result.output_dir)
    assert saved.schema_version == 2 and saved.msd


def test_run_json_is_written_last(tmp_path, monkeypatch):
    movie, labels_path, _ = synthetic_movie(tmp_path)
    order: list[str] = []
    for name in ("write_csv", "write_json", "save_masks"):
        real = getattr(export, name)

        def spy(path, *args, _real=real, **kwargs):
            order.append(Path(path).name)
            return _real(path, *args, **kwargs)

        monkeypatch.setattr(export, name, spy)
    pipeline.run_analysis(config_for(tmp_path, movie, labels_path))
    assert order[-1] == pipeline.F_MANIFEST
    assert order.count(pipeline.F_MANIFEST) == 1


def test_a_stale_run_json_does_not_survive_a_failed_rerun(tmp_path, monkeypatch):
    """run.json marks a complete result; an interrupted rerun must not keep the old one."""
    movie, labels_path, _ = synthetic_movie(tmp_path)
    config = config_for(tmp_path, movie, labels_path)
    pipeline.run_analysis(config)
    out = Path(config.output_dir)
    assert (out / pipeline.F_MANIFEST).exists()

    def broken(*args, **kwargs):
        raise RuntimeError("tracking failed")

    monkeypatch.setattr(pipeline, "track_detections", broken)
    with pytest.raises(RuntimeError, match="tracking failed"):
        pipeline.run_analysis(config)
    assert not (out / pipeline.F_MANIFEST).exists()
    # The stages before the failure are still on disk.
    assert (out / pipeline.F_MASKS).exists() and (out / pipeline.F_DETECTIONS).exists()


def test_cancelling_during_segmentation_raises_cancelled(tmp_path):
    movie, labels_path, _ = synthetic_movie(tmp_path)

    class CancelAtSegmentation(pipeline.NullProgress):
        def __init__(self):
            self.seen = False

        def step(self, done, total):
            self.seen = True

        def cancelled(self):
            return self.seen

    with pytest.raises(pipeline.Cancelled):
        pipeline.run_analysis(config_for(tmp_path, movie, labels_path), CancelAtSegmentation())


# --------------------------------------------------------------------------
# The refusals propagate: no guessing, no fallback
# --------------------------------------------------------------------------


def test_ambiguous_axes_propagate_and_an_explicit_order_resolves_them(tmp_path):
    _, labels_path, labels = synthetic_movie(tmp_path)
    plain = tmp_path / "plain.tif"
    tifffile.imwrite(plain, (labels > 0).astype(np.uint16) * 1000 + 500)
    config = config_for(tmp_path, plain, labels_path)
    with pytest.raises(AmbiguousAxes) as info:
        pipeline.run_analysis(config)
    assert "TYX" in info.value.choices
    # Refused before anything was written: not even an empty directory.
    assert not Path(config.output_dir).exists()

    config.import_.axes = "TYX"
    # A plain TIFF carries no calibration either; without it 5 um/min would
    # be read as 5 px/frame and these cells would be too fast to link.
    config.calibration.pixel_size_um = PIXEL_UM
    config.calibration.frame_interval_min = FRAME_S / 60.0
    result = pipeline.run_analysis(config)
    assert result.manifest["input"]["axes_source"] == "user_override"
    cal = result.manifest["calibration"]
    assert cal["pixel_size_um_source"] == "user_override"
    # The block agrees with itself: Y is measured with the size entered, and
    # what the file said (nothing) is kept apart, labelled as the file's.
    assert cal["pixel_size_y_um"] == cal["pixel_size_um"] == pytest.approx(PIXEL_UM)
    assert cal["pixel_size_y_um_source"] == "user_override"
    assert cal["reported_by_file"]["pixel_size_x_um"] is None
    assert cal["reported_by_file"]["frame_interval_min"] is None
    assert result.n_tracks == len(CELLS)


def test_a_missing_model_raises_model_unavailable_and_nothing_else_runs(tmp_path, monkeypatch):
    movie, _, _ = synthetic_movie(tmp_path)
    monkeypatch.delenv(model_registry.ENV_DEVELOPER, raising=False)
    monkeypatch.delenv(model_registry.ENV_DEVELOPER_MODEL, raising=False)
    absent = tmp_path / "models" / "absent"
    monkeypatch.setattr(model_registry, "candidate_paths", lambda spec: [absent])
    config = config_for(tmp_path, movie)
    with pytest.raises(ModelUnavailable) as info:
        pipeline.run_analysis(config)
    assert MODEL_UNAVAILABLE_MESSAGE in str(info.value)
    assert str(absent) in str(info.value)
    assert not Path(config.output_dir).exists(), "a refused run left a directory behind"


def test_a_refusal_leaves_a_previous_result_untouched(tmp_path, monkeypatch):
    """The directory is opened only once there is something to write in it."""
    movie, labels_path, _ = synthetic_movie(tmp_path)
    config = config_for(tmp_path, movie, labels_path)
    pipeline.run_analysis(config)
    out = Path(config.output_dir)
    before = {p.name: p.read_bytes() for p in out.iterdir()}

    def refuse(*args, **kwargs):
        raise ModelUnavailable(MODEL_UNAVAILABLE_MESSAGE, [(tmp_path / "absent", "missing")])

    monkeypatch.setattr(pipeline, "load_label_stack", refuse)
    with pytest.raises(ModelUnavailable):
        pipeline.run_analysis(config)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


# --------------------------------------------------------------------------
# The model route: E2 interfaces, and recovery keyed to final ids
# --------------------------------------------------------------------------


def production_model() -> ResolvedModel:
    spec = model_registry.production_spec("2D")
    return ResolvedModel(spec=spec, path=Path("models") / spec.filename, sha256=spec.sha256)


class LabelService:
    """Stands in for SegmentationService: returns given labels, records the model."""

    instances: list["LabelService"] = []

    def __init__(self, cfg, *, model=None, scale=None, labels=None):
        self.cfg, self.model_arg, self.scale = cfg, model, scale
        self.labels = labels
        LabelService.instances.append(self)

    def run_stack(self, stack, progress=None):
        detections, diagnostics = [], []
        for t in range(stack.shape[0]):
            found = extract_detections(self.labels[t], t, intensity=stack[t])
            detections.extend(found)
            diagnostics.append(FrameDiagnostics(t, len(found), len(found)))
        resolved = self.model_arg or production_model()
        return SegmentationOutput(
            masks=self.labels.copy(), raw_masks=self.labels.copy(), detections=detections,
            diagnostics=diagnostics, model_path=str(resolved.path),
            model_sha256=resolved.sha256, cellpose_version="3.1.1.3", used_gpu=False,
            model=resolved, provenance=PROVENANCE_MODEL,
        )


def test_the_model_route_uses_the_e2_interfaces_and_final_track_ids(tmp_path, monkeypatch):
    """Cell 1 is missing at frames 2-5, so the first pass splits it in two (ids 1 and 3).

    Recovery (the E2 interface, stood in for here) finds it at frames 3 and 4
    with a colliding crop label; the second pass then links one track through
    them, renumbered 1.  An attempt made for first-pass track 3 must come out
    keyed to final track 1, and every attempt must join tracks.csv.
    """
    movie, _, labels = synthetic_movie(tmp_path, drop={1: {2, 3, 4, 5}})
    monkeypatch.setattr(
        pipeline, "SegmentationService",
        lambda cfg, *, model=None, scale=None: LabelService(cfg, model=model, scale=scale, labels=labels),
    )
    calls: dict[str, dict] = {}

    def recover(stack, tracks, service, scale, tracking_cfg, recovery_cfg, *, geometry=None,
                background=None, progress=None):
        calls["recover"] = dict(
            tracks=[(t.id, [o.frame for o in t.observations]) for t in tracks],
            service=service, geometry=geometry, background=background,
            tracking=tracking_cfg, recovery=recovery_cfg,
        )
        head = next(t for t in tracks if t.observations[-1].frame == 1)
        tail = next(t for t in tracks if t.observations[0].frame == 6)
        bracket = (1, head.observations[-1].det_label, 6, tail.observations[0].det_label)
        cx, cy, step = CELLS[1]
        found, attempts = [], []
        for frame in (3, 4):
            mask = np.zeros((HEIGHT, WIDTH), np.int32)
            mask[_ellipse((HEIGHT, WIDTH), cx + 1.0, cy + step * frame)] = 1  # crop label 1
            (det,) = extract_detections(mask, frame, intensity=stack[frame])
            det.source, det.confidence = SOURCE_WINDOW, 0.8
            found.append(det)
            attempts.append(RecoveryAttempt(
                head.id, frame, det.x, det.y, found=True, source=SOURCE_WINDOW,
                confidence=0.8, offset_px=0.0, bracket=bracket, detection=det,
            ))
        # Made for the first-pass tail (id 3); nothing found.
        attempts.append(RecoveryAttempt(
            tail.id, 5, cx, cy + step * 5, found=False, detail="nothing", bracket=bracket,
        ))
        return RecoveryResult(detections=found, attempts=attempts)

    def collect_issues(metadata, scale, diagnostics, events, tracks, summaries, cfg, *,
                       geometry=None, unlinked=(), model=None, dimensionality="2D",
                       provenance="model", count_series=None):
        calls["qc"] = dict(geometry=geometry, model=model, dimensionality=dimensionality,
                           provenance=provenance, count_series=count_series, n_tracks=len(tracks))
        return []

    monkeypatch.setattr(pipeline.recovery_mod, "recover", recover)
    monkeypatch.setattr(pipeline.qc, "collect_issues", collect_issues)

    result = pipeline.run_analysis(config_for(tmp_path, movie))
    out = result.output_dir

    # recover() got the first pass: cell 1 split into two tracks (ids 1 and 3).
    first = dict(calls["recover"]["tracks"])
    assert first[1] == [0, 1] and first[3] == [6, 7]
    assert isinstance(calls["recover"]["geometry"], ChannelGeometry)
    assert calls["recover"]["geometry"].applied
    assert calls["recover"]["background"].shape == (HEIGHT, WIDTH)
    assert calls["recover"]["service"] is LabelService.instances[-1]

    # Re-tracked: one track through the recovered frames, renumbered.
    cell1 = [t for t in result.tracks if t.observations[0].frame == 0 and t.observations[0].x < 80]
    assert len(cell1) == 1 and [o.frame for o in cell1[0].observations] == [0, 1, 3, 4, 6, 7]
    assert result.n_tracks == 2

    # recovery_attempts.csv: final ids, first-pass ids, bracket, and the join.
    rows = read_table(out / pipeline.F_RECOVERY)
    assert [r["first_pass_track_id"] for r in rows] == [1, 1, 3]
    assert [r["track_id"] for r in rows] == [cell1[0].id] * 3
    assert [a.track_id for a in result.recovery.attempts] == [cell1[0].id] * 3
    tracks_csv = {(r["frame"], r["det_label"]): r["track_id"] for r in read_table(out / pipeline.F_TRACKS)}
    for r in rows[:2]:
        assert tracks_csv[(r["frame"], r["det_label"])] == r["track_id"]
        assert r["det_label"] >= 2, "a recovered label collided with the primary one"
    assert (rows[2]["bracket_frame_before"], rows[2]["bracket_frame_after"]) == (1, 6)
    assert rows[2]["det_label"] is None

    # detections.csv holds both kinds, uniquely labelled per frame.
    dets = read_table(out / pipeline.F_DETECTIONS)
    assert len({(d["frame"], d["label"]) for d in dets}) == len(dets) == result.n_detections
    assert result.manifest["results"]["n_detections_recovered"] == 2

    # QC got the 2.0 arguments.
    q = calls["qc"]
    assert isinstance(q["geometry"], ChannelGeometry)
    assert q["model"] is result.model and q["model"].spec.model_id == model_registry.production_spec().model_id
    assert (q["dimensionality"], q["provenance"]) == ("2D", PROVENANCE_MODEL)
    per_frame = [sum(1 for d in dets if d["frame"] == t) for t in range(N_FRAMES)]
    assert q["count_series"] == per_frame

    m = result.manifest
    assert m["model"]["model_id"] == model_registry.production_spec().model_id
    assert m["recovery"]["skipped"] is None and m["recovery"]["recovered"] == 2
    assert (out / pipeline.F_RAW_MASKS).exists()


def test_the_pipeline_has_no_migration_axis():
    source = inspect.getsource(pipeline)
    for gone in ("resolve_axis", "ConfinementAxis", "from .confinement", "assign_channels"):
        assert gone not in source, gone
    names = {f for f in pipeline.AnalysisResult.__dataclass_fields__}
    assert "axis" not in names and {"geometry", "model", "dimensionality", "msd"} <= names


# --------------------------------------------------------------------------
# 3-D: imported labels with a Z step
# --------------------------------------------------------------------------


def test_a_3d_label_import_is_measured_in_um3_and_skips_recovery(tmp_path):
    n_t, n_z, h, w = 4, 12, 64, 64
    zz, yy, xx = np.ogrid[:n_z, :h, :w]
    labels = np.zeros((n_t, n_z, h, w), np.int32)
    for t in range(n_t):
        blob = ((zz - 6) / 3.0) ** 2 + ((yy - 20 - 4 * t) / 8.0) ** 2 + ((xx - 30) / 5.0) ** 2 <= 1
        labels[t][blob] = 1
    movie = tmp_path / "stack.tif"
    tifffile.imwrite(
        movie, (labels * 800 + 1000).astype(np.uint16), imagej=True,
        metadata={"axes": "TZYX", "spacing": 2.0, "unit": "micron", "finterval": FRAME_S},
        resolution=(1.0 / PIXEL_UM, 1.0 / PIXEL_UM),
    )
    labels_path = tmp_path / "labels3d.tif"
    tifffile.imwrite(labels_path, labels)

    result = pipeline.run_analysis(config_for(tmp_path, movie, labels_path))
    m = result.manifest
    assert result.dimensionality == "3D" == m["dimensionality"]
    assert m["input"]["axes"] == "TZYX"
    assert m["calibration"]["z_step_um"] == pytest.approx(2.0)
    assert m["calibration"]["anisotropy"] == pytest.approx(2.0 / PIXEL_UM)
    assert "3-D" in m["recovery"]["skipped"] or "imported" in m["recovery"]["skipped"]
    assert result.n_tracks == 1 and result.tracks[0].n_obs == n_t
    assert export.load_masks(result.output_dir / pipeline.F_MASKS).shape == labels.shape
    (summary,) = result.summaries
    assert summary.dimensionality == "3D" and summary.mean_volume_um3 is not None
    step_um = 4 * PIXEL_UM
    assert summary.net_speed_um_per_min == pytest.approx(step_um / (FRAME_S / 60.0), rel=0.05)
    assert all(not math.isnan(r["msd_um2"]) for r in result.msd)
