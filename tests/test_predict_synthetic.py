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
selection, conformal calibration).
"""

from __future__ import annotations

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


def test_conformal_calibration_is_split_by_track(planted):
    task = planted.task(TARGET)
    r = E.cross_validate(task, E.spec_by_name("ridge_morphology", False))
    m = E.metrics(task, r)
    assert 0.0 <= m["interval_coverage"] <= 1.0
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
    # Two known experiments are two dishes: distinct without any pixels.
    evidence[2]["verdict"] = "different experiments"
    assert D.pixel_established_fields(folders, evidence) == ["a+a_sub", "b"]


def test_the_outer_split_becomes_experiments_as_soon_as_there_are_two(tmp_path):
    assert D.outer_grouping(["20240529-s01"] * 5) == "field"
    cfg = replace(CONFIG, n_movies=4, tracks_per_movie=3, n_frames=8, n_experiments=2, seed=3)
    ds = D.build_dataset(S.write_dataset(cfg, tmp_path), with_phenotype=False)
    assert D.outer_grouping(ds.samples["experiment"]) == "experiment"
    task = ds.task(TARGET)
    splits = D.leave_one_group_out(task.groups("experiment"), task.track)
    assert len(splits) == 2
    for tr, te in splits:
        assert not set(task.experiment[tr]) & set(task.experiment[te])
        assert not set(task.movie[tr]) & set(task.movie[te])


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
