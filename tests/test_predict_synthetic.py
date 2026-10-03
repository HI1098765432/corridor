"""Validation first: the pipeline must find a planted signal and refuse a null.

Both datasets come from :mod:`training.predict.synthetic` and go through the
same path as real data: result folders on disk, mask features, targets,
leave-one-field-out CV (each synthetic movie is its own field), and the
track-block permutation test. They differ
only in whether a cell's speed follows its drawn aspect ratio or an
independent copy of it, so a "detection" on the null would be a defect of the
pipeline, not of the data. Seeds are fixed, so the outcome is deterministic.

The leakage tests pin the one rule a grouped evaluation lives or dies by: no
track is ever on both sides of a split, at any level (outer CV, inner penalty
selection, conformal calibration). The outer-unit tests pin what may count as
an experiment-level group: a resolved acquisition, never a placeholder id and
never a second stage position of the same dish, and that every model, the
embedding pool and the sensitivity analyses use the same unit.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# training/ is research code and deliberately not an installed package (the
# installer must never contain it), so the checkout root has to be importable
# whatever pytest.ini's pythonpath says.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.predict import dataset as D  # noqa: E402
from training.predict import evaluate as E  # noqa: E402
from training.predict import experiment as X  # noqa: E402
from training.predict import models as M  # noqa: E402
from training.predict import synthetic as S  # noqa: E402

CONFIG = S.SyntheticConfig(n_movies=4, tracks_per_movie=5, n_frames=11, seed=11)
TARGET = "speed_um_per_hr_h1"
N_PERM = 199  # smallest attainable p = 1/200 = 0.005, so "p < 0.01" is reachable


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    folders = S.write_dataset(replace(CONFIG, planted=True), tmp_path_factory.mktemp("planted"))
    return D.build_dataset(folders, with_phenotype=False)


@pytest.fixture(scope="module")
def null(tmp_path_factory):
    folders = S.write_dataset(replace(CONFIG, planted=False), tmp_path_factory.mktemp("null"))
    return D.build_dataset(folders, with_phenotype=False)


def _primary(task):
    return (E.spec_by_name(E.PRIMARY_REGRESSION, False), E.spec_by_name("mean", False))


def test_planted_signal_is_found(planted):
    task = planted.task(TARGET)
    assert task.summary()["n_movies"] == CONFIG.n_movies
    model, base = _primary(task)
    m = E.metrics(task, E.cross_validate(task, model))
    b = E.metrics(task, E.cross_validate(task, base))
    assert m["mae"] < b["mae"]
    assert m["r2_vs_training_mean"] > 0.2
    pt = E.permutation_test(task, model, base, n_perm=N_PERM, seed=0)
    assert pt["p_value"] < 0.01


def test_null_is_not_found(null):
    task = null.task(TARGET)
    model, base = _primary(task)
    pt = E.permutation_test(task, model, base, n_perm=N_PERM, seed=0)
    assert pt["p_value"] > 0.05


def test_every_outer_split_keeps_tracks_whole(planted):
    task = planted.task("disp_um_h3")
    splits = D.leave_one_group_out(task.groups("movie"), task.track)
    assert len(splits) == CONFIG.n_movies
    for tr, te in splits:
        assert not set(task.track[tr]) & set(task.track[te])
        assert not set(task.movie[tr]) & set(task.movie[te])


def test_a_split_finer_than_a_track_is_refused(planted):
    task = planted.task(TARGET)
    per_row = np.arange(task.n)  # "leave one observation out"
    with pytest.raises(D.LeakageError):
        D.leave_one_group_out(per_row, task.track)


def test_a_track_spanning_two_groups_is_refused():
    df = pd.DataFrame({"track_uid": ["a/1", "a/1", "b/1"], "movie": ["a", "b", "b"]})
    with pytest.raises(D.LeakageError):
        D.assert_tracks_within_groups(df, "movie")


def test_conformal_calibration_holds_out_whole_training_tracks(planted, monkeypatch):
    """Inside every outer fold: calibration and fitting tracks are disjoint,
    together are exactly that fold's training tracks, and never a test track."""
    calls = []
    real = M.split_conformal

    def spy(make, X_train, y_train, groups_train, X_test, **kw):
        lo, hi, info = real(make, X_train, y_train, groups_train, X_test, **kw)
        calls.append((set(np.asarray(groups_train).tolist()), info))
        return lo, hi, info

    monkeypatch.setattr(M, "split_conformal", spy)
    task = planted.task(TARGET)
    r = E.cross_validate(task, E.spec_by_name("ridge_morphology", False))
    splits = D.leave_one_group_out(task.groups("field"), task.track)
    assert len(calls) == len(splits) == CONFIG.n_movies
    for (train_tracks, info), (tr, te) in zip(calls, splits):
        fit, cal = set(info["fit_tracks"]), set(info["cal_tracks"])
        assert fit and cal and not fit & cal
        assert fit | cal == train_tracks == set(task.track[tr])
        assert not (fit | cal) & set(task.track[te])
    m = E.metrics(task, r)
    assert m["interval_infinite_fraction"] == 0.0


def test_track_block_permutation_moves_whole_runs():
    track = np.repeat(["a", "b", "c", "d"], [3, 5, 2, 4])
    rng = np.random.default_rng(0)
    for _ in range(20):
        perm = E.track_block_permutation(track, rng)
        assert sorted(perm) == list(range(len(track)))
        # Consecutive positions take consecutive source rows except at most
        # one wrap per block boundary (4 blocks + the circular cut).
        breaks = np.sum(np.diff(perm) != 1)
        assert breaks <= 4
    with pytest.raises(ValueError):
        E.track_block_permutation(np.array(["a", "b", "a"]), rng)


def test_freedman_lane_moves_only_what_history_does_not_explain():
    rng = np.random.default_rng(0)
    track = np.repeat(np.arange(8), 5)
    history = rng.normal(size=(40, 2))
    A = np.hstack([np.ones((40, 1)), history])
    # A "morphology" block that is an exact linear copy of history: nothing to move.
    copy = history @ np.array([[1.0, -2.0, 0.3], [0.5, 3.0, -1.0]]) + 4.0
    fitted, resid = E.history_residualisation(copy, history)
    np.testing.assert_allclose(resid, 0.0, atol=1e-9)
    np.testing.assert_allclose(fitted + resid[E.track_block_permutation(track, rng)], copy, atol=1e-9)
    # In general the residual is orthogonal to history and the parts add back up.
    other = rng.normal(size=(40, 3)) + 0.5 * history[:, :1]
    fitted, resid = E.history_residualisation(other, history)
    np.testing.assert_allclose(fitted + resid, other)
    np.testing.assert_allclose(A.T @ resid, 0.0, atol=1e-9)


def test_delta_null_of_a_redundant_morphology_block_is_the_observed_delta(planted):
    """If the shape says nothing history does not, every Freedman-Lane permutation
    rebuilds the same block, so C cannot beat its own null: p is exactly 1."""
    task = planted.task(TARGET)
    hist = task.blocks["history"]
    redundant = np.column_stack([hist[:, 0] * 2.0 + 1.0, hist[:, 1] - hist[:, 2], hist[:, 2]])
    task = replace(task, blocks={**task.blocks, "strict": redundant})
    out = E.history_delta(task, n_perm=9, seed=0, n_boot=50)
    assert out["null_mean"] == pytest.approx(out["delta_mae"], abs=1e-6)
    assert out["p_value"] == 1.0
    assert out["strict_variance_explained_by_history_median"] == pytest.approx(1.0)
    ci = out["delta_mae_track_bootstrap"]
    assert ci["estimate"] == pytest.approx(out["delta_mae"])
    assert ci["resampled_unit"] == "track"


def test_targets_recovered_and_border_observations(tmp_path):
    cfg = replace(CONFIG, n_movies=1, tracks_per_movie=3, n_frames=8, seed=5)
    (folder,) = S.write_dataset(cfg, tmp_path)
    tracks = pd.read_csv(folder / "tracks.csv")
    with np.load(folder / "masks.npz") as z:
        masks = z["masks"].copy()
    # Track 1, frame 3: pretend recovery found it (no pixels under its label).
    i = tracks.index[(tracks.track_id == 1) & (tracks.frame == 3)][0]
    tracks.loc[i, ["detection_source", "det_label"]] = ["intensity", 99]
    # Track 2, frame 5: extend its mask to the top edge of the image.
    j = tracks.index[(tracks.track_id == 2) & (tracks.frame == 5)][0]
    col = int(round(tracks.loc[j, "x_px"]))
    masks[5, 0:int(tracks.loc[j, "y_px"]), col] = 2
    tracks.loc[j, "area_px"] = float((masks[5] == 2).sum())
    tracks.to_csv(folder / "tracks.csv", index=False)
    np.savez_compressed(folder / "masks.npz", masks=masks)

    ds = D.build_dataset([folder], with_phenotype=False)
    assert ds.exclusions["recovered_no_mask"] == 1
    assert ds.exclusions["touches_border"] == 1
    s = ds.samples
    assert not ((s.track_id == 1) & (s.frame == 3)).any()
    assert not ((s.track_id == 2) & (s.frame == 5)).any()
    # The recovered position is still the *target* of the frame before it.
    t1 = tracks[tracks.track_id == 1].set_index("frame")
    row = s[(s.track_id == 1) & (s.frame == 2)].iloc[0]
    expected = np.hypot(t1.loc[3, "x_um"] - t1.loc[2, "x_um"], t1.loc[3, "y_um"] - t1.loc[2, "y_um"])
    assert row["disp_um_h1"] == pytest.approx(expected)
    assert row["target_recovered_h1"] == 1.0
    # Mean speed over 3 frames = path / elapsed hours.
    path = sum(np.hypot(t1.loc[k + 1, "x_um"] - t1.loc[k, "x_um"],
                        t1.loc[k + 1, "y_um"] - t1.loc[k, "y_um"]) for k in (2, 3, 4))
    assert row["speed_um_per_hr_h3"] == pytest.approx(path / (3 * cfg.frame_interval_min / 60))
    rate = expected / (cfg.frame_interval_min / 60)
    assert row["migrating_h1"] == float(rate >= D.MIGRATING_MIN_NET_RATE_UM_PER_HR)
    # First observations have no history, last ones no target.
    assert s[s.observation_index == 0]["hist_speed_last_um_per_hr"].isna().all()
    assert s[s.observation_index == cfg.n_frames - 1]["disp_um_h1"].isna().all()


def test_only_crops_compared_at_the_same_instant_count_as_established_fields():
    """The supplied-data situation: two big crops proven distinct at shared
    frames, one later crop that shares no frame with anything."""
    folders = [{"movie": "a", "field": "a+a_sub", "n_observations": 80},
               {"movie": "a_sub", "field": "a+a_sub", "n_observations": 0},
               {"movie": "b", "field": "b", "n_observations": 52},
               {"movie": "late", "field": "late", "n_observations": 8}]
    evidence = [{"small": "a", "large": "b", "verdict": "different regions"},
                {"small": "a_sub", "large": "a", "verdict": "same pixels"},
                {"small": "a_sub", "large": "b", "verdict": "different regions"},
                {"small": "late", "large": "a", "verdict": "undecidable from pixels (no shared instant)"},
                {"small": "late", "large": "b", "verdict": "different regions"}]
    assert D.pixel_established_fields(folders, evidence) == ["a+a_sub", "b"]
    # One undecidable movie pair is enough to keep two fields from both counting.
    evidence[2]["verdict"] = "not checked: no source frame numbers"
    assert D.pixel_established_fields(folders, evidence) == ["a+a_sub"]
    evidence[2]["verdict"] = "not checked: no image"
    assert D.pixel_established_fields(folders, evidence) == ["a+a_sub"]
    # Two resolved acquisitions are two dishes: distinct without any pixels.
    evidence[2]["verdict"] = "different acquisitions"
    assert D.pixel_established_fields(folders, evidence) == ["a+a_sub", "b"]


def test_stage_positions_of_one_day_are_one_acquisition(tmp_path):
    """The supplied nd2 is T(54) x XY(57): series 01 is one stage position of a dish."""
    import tifffile

    assert D._acquisition_of({}, "20240529-s01", None) == "20240529"
    assert D._acquisition_of({}, "20240529-s02", None) == "20240529"
    assert D.outer_split([D._acquisition_of({}, e, None)
                          for e in ("20240529-s01", "20240529-s02")])[0] == "field"
    assert D._acquisition_of({"acquisition": "dish-B"}, "20240529-s02", None) == "dish-B"
    assert D._acquisition_of({"experiment": "exp-A"}, "exp-A", None) == "exp-A"
    assert D._acquisition_of({}, None, None) is None
    # From the ImageJ labels; an undated source names no experiment (it was
    # once "unknown-s01", one shared id for every undated series 1).
    dated, undated = tmp_path / "dated.tif", tmp_path / "undated.tif"
    frames = np.zeros((2, 8, 8), np.uint16)
    tifffile.imwrite(dated, frames, imagej=True, metadata={
        "Labels": ["t:1/2 - 20240529 secret condition.nd2 (series 03)",
                   "t:2/2 - 20240529 secret condition.nd2 (series 03)"]})
    tifffile.imwrite(undated, frames, imagej=True, metadata={
        "Labels": ["t:1/2 - secret.nd2 (series 01)", "t:2/2 - secret.nd2 (series 01)"]})
    assert D._source_from_tiff(dated) == ("20240529", 3)
    assert D._experiment_from_tiff(dated) == "20240529-s03"
    assert D._acquisition_of({}, D._experiment_from_tiff(dated), dated) == "20240529"
    assert D._experiment_from_tiff(undated) is None
    assert D._acquisition_of({}, None, undated) is None


def test_a_movie_nothing_identifies_is_never_its_own_group(tmp_path):
    """No run.json id and no image: placeholders, an unchecked overlap and no
    phenotype, each recorded; and the real-data run refuses to start."""
    cfg = replace(CONFIG, n_movies=3, tracks_per_movie=2, n_frames=6, seed=4)
    folders = S.write_dataset(cfg, tmp_path)
    for f in folders:
        run = json.loads((f / "run.json").read_text(encoding="utf-8"))
        run.pop("experiment")
        (f / "run.json").write_text(json.dumps(run), encoding="utf-8")
    ds = D.build_dataset(folders, with_phenotype=True)
    assert set(ds.samples["acquisition"]) == {f"{D.UNRESOLVED}{f.name}" for f in folders}
    unit, why = D.outer_split(ds.samples["acquisition"])
    assert unit == "field" and "no resolved acquisition" in why
    assert D.outer_split(["20240529", "unknown:x"])[0] == "field"
    assert D.outer_split(["20240529", "20240612"]) == ("acquisition", "2 resolved acquisitions")
    assert [e["verdict"] for e in ds.overlap_evidence] == ["not checked: no image"] * 3
    assert ds.checks["overlap_check"] == "incomplete: 3 of 3 movie pairs not compared"
    assert ds.checks["phenotype"].startswith("dropped: no image for ")
    assert ds.checks["unresolved_acquisitions"] == sorted(f.name for f in folders)
    assert len(D.pixel_established_fields(ds.folders, ds.overlap_evidence)) == 1
    with pytest.raises(X.MissingImagesError):
        X.preflight(folders)


class _CountingAutoencoder:
    """Stands in for the torch autoencoder: records how many crops each fit saw."""

    fits: list[int] = []

    def __init__(self, latent_dim: int = 8, epochs: int = 0, seed: int = 0, **_):
        self.latent_dim = latent_dim

    def fit(self, crops):
        type(self).fits.append(len(crops))
        self.n_params_, self.loss_history_ = 0, [0.0]
        return self

    def transform(self, crops):
        crops = np.asarray(crops, dtype=float)
        return np.column_stack([crops.mean(axis=(1, 2)), crops[:, :32].mean(axis=(1, 2))])

    def reconstruction_iou(self, crops):
        return 1.0


@pytest.fixture(scope="module")
def two_acquisitions(tmp_path_factory):
    cfg = replace(CONFIG, n_movies=4, tracks_per_movie=3, n_frames=8, n_experiments=2, seed=3)
    return D.build_dataset(S.write_dataset(cfg, tmp_path_factory.mktemp("two_acq")),
                           with_phenotype=False)


@pytest.fixture
def cheap_and_spied(monkeypatch):
    """One permutation per test, 3 trees, the counting autoencoder, and a record
    of the outer unit of every cross-validation anything runs."""
    for k in X.N_PERM:
        monkeypatch.setitem(X.N_PERM, k, 1)
    monkeypatch.setattr(X, "FOREST_TREES", 3)
    monkeypatch.setattr(M, "MaskAutoencoder", _CountingAutoencoder)
    monkeypatch.setattr(_CountingAutoencoder, "fits", [])
    units: list[str] = []
    real = E.cross_validate

    def spy(task, spec, **kw):
        units.append(kw.get("group_by", "field"))
        return real(task, spec, **kw)

    monkeypatch.setattr(E, "cross_validate", spy)
    return units


def test_every_model_and_the_embedding_pool_hold_out_acquisitions(two_acquisitions, cheap_and_spied):
    ds = two_acquisitions
    group_by, why, pool = X.outer_unit(ds)
    acquisitions = {"synthetic-experiment-0", "synthetic-experiment-1"}
    assert (group_by, why) == ("acquisition", "2 resolved acquisitions")
    assert set(pool) == acquisitions and len(pool) == len(ds.all_crops)
    embedder = E.Embedder(ds.all_crops, pool, epochs=1, latent_dim=2)
    out = X.evaluate_task(ds.task(TARGET), embedder, group_by, phenotype_status="not requested")

    assert out["outer_split_unit"] == "acquisition"
    assert cheap_and_spied and set(cheap_and_spied) == {"acquisition"}
    for name, model in out["models"].items():
        assert {f["held_out"] for f in model["folds"]} == acquisitions, name
    assert out["skipped_models"] == {"phenotype_ridge": "phenotype: not requested"}
    # One autoencoder per held-out acquisition, fitted on the other one's masks only.
    outside = sorted(int(np.sum(pool != a)) for a in acquisitions)
    assert sorted(_CountingAutoencoder.fits) == outside
    with pytest.raises(ValueError):
        embedder.for_held_out(ds.samples["field"].iloc[0])  # a field is not a pool group here


def test_the_established_fields_sensitivity_splits_by_the_outer_unit(two_acquisitions, cheap_and_spied):
    est = X.established_fields_sensitivity(two_acquisitions, "acquisition")
    assert est["outer_split_unit"] == "acquisition"
    ran = [t for t in est["targets"].values() if "skipped" not in t]
    assert ran and cheap_and_spied and set(cheap_and_spied) == {"acquisition"}


def test_the_condition_baseline_uses_labels_never_ids(tmp_path):
    """With conditions supplied, and each condition in both acquisitions, the
    per-condition median is a real baseline under leave-one-acquisition-out."""
    cfg = replace(CONFIG, n_movies=4, tracks_per_movie=2, n_frames=6, n_experiments=2, seed=6)
    folders = S.write_dataset(cfg, tmp_path)
    unlabelled = D.build_dataset(folders, with_phenotype=False)
    assert set(unlabelled.samples["condition"]) == {D.UNSPECIFIED_CONDITION}
    # Movies 0 and 2 are acquisition 0, movies 1 and 3 acquisition 1.
    labels = {folders[0].name: "cond-alpha", folders[1].name: "cond-alpha",
              folders[2].name: "cond-beta", folders[3].name: "cond-beta"}
    ds = D.build_dataset(folders, with_phenotype=False, conditions=labels)
    assert all(f["condition_known"] for f in ds.folders)
    assert "cond-" not in json.dumps(ds.folders)  # only whether a label exists is reported
    task = ds.task(TARGET)
    r = E.cross_validate(task, E.spec_by_name("condition_median", False),
                         group_by="acquisition", conformal=False)
    for tr, te in D.leave_one_group_out(task.groups("acquisition"), task.track):
        for c in ("cond-alpha", "cond-beta"):
            rows = te[task.condition[te] == c]
            expected = np.median(task.y[tr][task.condition[tr] == c])
            np.testing.assert_allclose(r.pred[rows], expected)
    assert len(np.unique(r.pred)) > 2


def test_the_rotation_audit_pools_tilts_and_keeps_90_degrees_apart(tmp_path):
    """A 90-degree turn is lossless on the grid; pooled with the tilts it once
    pulled the reported noise down, so it is a per-angle control only."""
    cfg = replace(CONFIG, n_movies=1, tracks_per_movie=2, n_frames=3, seed=8)
    folders = S.write_dataset(cfg, tmp_path)
    audit = X.invariance_audit(D.build_dataset(folders, with_phenotype=False), folders)
    assert audit["n_masks"] == 6
    assert audit["tilt_angles_deg"] == list(X.TILT_ANGLES_DEG) and 90 not in X.TILT_ANGLES_DEG
    by_angle = audit["rotation_noise_to_cell_spread_by_angle"]
    assert set(by_angle) == {str(a) for a in X.TILT_ANGLES_DEG + X.CONTROL_ANGLES_DEG}
    assert by_angle["90"]["area_um2"] == 0.0
    assert audit["rotation_noise_to_cell_spread"]["area_um2"] > 0.0
    worst = audit["rotation_noise_to_cell_spread_worst_tilt"]["area_um2"]
    assert worst == max(by_angle[str(a)]["area_um2"] for a in X.TILT_ANGLES_DEG)


def test_an_unfinished_report_never_replaces_the_result(tmp_path):
    out = tmp_path / "result.json"
    out.write_text(json.dumps({"complete": True, "version": "old"}), encoding="utf-8")
    X._write({"complete": False, "version": "mid-run"}, out)
    assert json.loads(out.read_text(encoding="utf-8"))["version"] == "old"
    assert json.loads(X.partial_path(out).read_text(encoding="utf-8"))["complete"] is False
    X._write({"complete": True, "version": "new"}, out)
    assert json.loads(out.read_text(encoding="utf-8"))["version"] == "new"
    assert not X.partial_path(out).exists()
    with pytest.raises(SystemExit):
        X.main(["--smoke", "--out", str(X.DEFAULT_OUT)])


def test_a_sub_crop_of_another_movie_is_one_field_and_not_counted_twice(tmp_path):
    """The 052924_t1-inside-052924_1 case, built on purpose.

    Movie ``wide`` has three cells; movie ``narrow`` is a pixel-identical crop
    of it holding only the middle lane, over the same source frames, segmented
    "again" (here: copied). The narrow movie's track must be recognised as the
    wide movie's cell, dropped, and the two movies must form one field, so no
    split can put that cell on both sides.
    """
    import json

    import tifffile

    cfg = replace(CONFIG, n_movies=1, tracks_per_movie=3, n_frames=8, seed=21)
    (wide,) = S.write_dataset(cfg, tmp_path / "w")
    with np.load(wide / "masks.npz") as z:
        masks = z["masks"]
    rng = np.random.default_rng(0)
    texture = rng.normal(1000.0, 60.0, masks.shape[1:])  # static background, like channel walls
    image = (texture[None] - 300.0 * (masks > 0) + rng.normal(0, 5, masks.shape)).astype(np.float32)
    tifffile.imwrite(wide / "input.tif", image)
    run = json.loads((wide / "run.json").read_text())
    run["input"].update(path=str(wide / "input.tif"), source_frames=list(range(1, 9)))
    (wide / "run.json").write_text(json.dumps(run))

    r0, c0 = 5, cfg.lane_pitch_px  # the narrow crop: middle lane, 5 rows down
    narrow = tmp_path / "narrow"
    narrow.mkdir()
    sub = masks[:, r0:, c0:c0 + cfg.lane_pitch_px]
    np.savez_compressed(narrow / "masks.npz", masks=sub)
    tifffile.imwrite(narrow / "input.tif", image[:, r0:, c0:c0 + cfg.lane_pitch_px])
    tracks = pd.read_csv(wide / "tracks.csv")
    mid = tracks[tracks.track_id == 2].copy()
    mid["track_id"] = 1
    mid["x_px"] -= c0
    mid["y_px"] -= r0
    mid["x_um"] = mid["x_px"] * cfg.pixel_size_um
    mid["y_um"] = mid["y_px"] * cfg.pixel_size_um
    mid.to_csv(narrow / "tracks.csv", index=False)
    run_n = dict(run, input={"path": str(narrow / "input.tif"), "source_frames": list(range(1, 9))})
    (narrow / "run.json").write_text(json.dumps(run_n))

    ds = D.build_dataset([wide, narrow], with_phenotype=False)
    assert ds.overlaps and ds.overlaps[0]["small"] == "narrow"
    assert ds.overlaps[0]["offset_rc"] == [r0, c0]
    assert ds.exclusions["duplicate_of_overlapping_movie"] == cfg.n_frames
    assert set(ds.samples["movie"]) == {wide.name}
    assert ds.samples["field"].nunique() == 1

    without = D.build_dataset([wide, narrow], with_phenotype=False, detect_overlaps=False)
    assert set(without.samples["movie"]) == {wide.name, "narrow"}  # the leak this prevents
