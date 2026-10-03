"""Run the morphology -> future-migration experiment and write every number.

    OMP_NUM_THREADS=2 python -m training.predict.experiment

Order, as the design contract requires: the synthetic validation (planted
signal found, null refused, permutation-test calibration) runs first and is
reported first; then the real result folders in ``build/baseline_v1.3.0/`` are
analysed and reported whatever the result. Output:
``docs/prediction_experiment.json``. The prose reading is ``docs/PREDICTION.md``.

Nothing here is tuned on the real outcome. Horizons, the migrating threshold,
features, targets, models, the primary model per task and seeds were fixed
before any real result was seen. Three things changed after real numbers had
been printed, none of them towards a result: the elastic-net penalty grid and
the permutation counts (both for run time, see ``models.ElasticNetRegressor``
and ``N_PERM``), and the outer split, from leave-one-movie-out to
leave-one-field-out, after the overlap check found the same cell in two movies
(``dataset`` module docstring). The leaky split is still reported, as a
sensitivity analysis, so the effect of that change is visible. One further
sensitivity analysis was added after that run and changes nothing in the
primary analysis: the primary models on only the fields the pixels prove
distinct (``dataset.pixel_established_fields``), which bounds what the crops
that share no instant with any other movie could contribute.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from . import dataset as D
from . import evaluate as E
from . import features as F
from . import synthetic as S

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOLDERS = REPO_ROOT / "build" / "baseline_v1.3.0"
DEFAULT_OUT = REPO_ROOT / "docs" / "prediction_experiment.json"

SEED = 0
TARGET_KINDS = ("disp_um", "speed_um_per_hr", "migrating")
#: Permutation counts are set by cost, never by outcome; the resolution each
#: one allows (smallest attainable p = 1 / (n + 1)) is reported with every p.
#: Measured cost of one permutation on the real data on this (busy) machine:
#: ridge ~0.025 s, logistic ~0.3 s, elastic net ~1.4 s, forest 1-3 s.
N_PERM = {
    "primary_regression": 999,
    "primary_classification": 499,
    "secondary_regression": 199,
    "secondary_classification": 99,
    "slow": 49,  # elastic net and forest: resolution-limited, descriptive only
    "delta_regression": 199,
    "delta_classification": 99,
}
FOREST_TREES = 200
AE_EPOCHS = 150
AE_LATENT_DIM = 8
#: Synthetic validation: same generator settings as tests/test_predict_synthetic.py.
SYNTH_CONFIG = S.SyntheticConfig(n_movies=4, tracks_per_movie=5, n_frames=11, seed=11)
#: Calibration study: independent null datasets, each tested under both schemes.
N_CALIBRATION_NULL = 30
N_CALIBRATION_PLANTED = 10
N_PERM_CALIBRATION = 99
#: The classifier costs ~10x the ridge per permutation, so it is calibrated on
#: the first 20 of the same null datasets with 49 permutations.
N_CALIBRATION_NULL_CLASSIFICATION = 20
N_PERM_CALIBRATION_CLASSIFICATION = 49
#: Rotations used for the invariance audit on real masks (degrees).
AUDIT_ANGLES = (15, 30, 60, 90)


def _source_hashes() -> dict:
    import hashlib

    here = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(here.glob("*.py"))}


def _clean(obj):
    """JSON-safe: numpy scalars to Python, non-finite floats to None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Synthetic validation


def _rate_with_ci(hits: int, n: int, level: float = 0.95) -> dict:
    """Proportion with its exact (Clopper-Pearson) interval: 30 nulls is not many."""
    from scipy.stats import beta

    lo = 0.0 if hits == 0 else float(beta.ppf((1 - level) / 2, hits, n - hits + 1))
    hi = 1.0 if hits == n else float(beta.ppf(1 - (1 - level) / 2, hits + 1, n - hits))
    return {"hits": int(hits), "n": int(n), "rate": hits / n if n else None,
            "ci_low": lo, "ci_high": hi, "ci_level": level}


def _primary_on(ds: D.PredictionDataset) -> dict:
    res = {}
    for kind in TARGET_KINDS:
        for h in D.HORIZONS_FRAMES:
            task = ds.task(f"{kind}_h{h}")
            cls = task.classification
            spec = E.spec_by_name(E.PRIMARY_CLASSIFICATION if cls else E.PRIMARY_REGRESSION, cls)
            base = E.spec_by_name("majority" if cls else "mean", cls)
            m = E.metrics(task, E.cross_validate(task, spec, seed=SEED))
            b = E.metrics(task, E.cross_validate(task, base, seed=SEED))
            pt = E.permutation_test(task, spec, base, seed=SEED, n_perm=N_PERM[
                "secondary_classification" if cls else "secondary_regression"])
            key = "balanced_accuracy" if cls else "mae"
            res[f"{kind}_h{h}"] = {"n": task.n, "model": spec.name, key: m[key],
                                   f"baseline_{key}": b[key], "p_value": pt["p_value"],
                                   "n_permutations": pt["n_permutations"],
                                   **({} if cls else {"interval_coverage": m["interval_coverage"],
                                                      "r2_vs_training_mean": m["r2_vs_training_mean"]})}
    return res


SYNTH_CASES = (
    ("planted", {"planted": True}),
    ("null", {"planted": False}),
    # The generator's first outline (synthetic.py): no shape -> motion link, but
    # a centroid that jitters with cell length. Reported beside the strict null.
    ("null_lopsided_outline", {"planted": False, "lopsided_outline": True}),
)


def synthetic_validation(workdir: Path) -> dict:
    out: dict = {"config": asdict(SYNTH_CONFIG), "primary": {}}
    for name, overrides in SYNTH_CASES:
        cfg = replace(SYNTH_CONFIG, **overrides)
        ds = D.build_dataset(S.write_dataset(cfg, workdir / name), with_phenotype=False)
        out["primary"][name] = _primary_on(ds)
        log(f"synthetic {name} done")

    # Calibration of the permutation test on independent strict nulls: the
    # false-positive rate of the track-block scheme and of a naive row shuffle
    # (regression), of the track-block scheme for the classifier, and power on
    # independent planted datasets.
    cal = {"n_null_datasets": N_CALIBRATION_NULL, "n_planted_datasets": N_CALIBRATION_PLANTED,
           "regression": {"target": "speed_um_per_hr_h1", "model": E.PRIMARY_REGRESSION,
                          "n_permutations": N_PERM_CALIBRATION,
                          "null_p": {"track_block": [], "row": []}, "planted_p": []},
           "classification": {"targets": ["migrating_h1", "migrating_h3"],
                              "model": E.PRIMARY_CLASSIFICATION,
                              "n_permutations": N_PERM_CALIBRATION_CLASSIFICATION,
                              "n_null_datasets": N_CALIBRATION_NULL_CLASSIFICATION,
                              "scheme": "track_block",
                              "null_p": {"migrating_h1": [], "migrating_h3": [],
                                         "migrating_h1_lopsided_outline": [],
                                         "migrating_h3_lopsided_outline": []}}}
    reg, clf = cal["regression"], cal["classification"]
    spec = E.spec_by_name(E.PRIMARY_REGRESSION, False)
    base = E.spec_by_name("mean", False)
    cspec = E.spec_by_name(E.PRIMARY_CLASSIFICATION, True)
    cbase = E.spec_by_name("majority", True)
    for i in range(N_CALIBRATION_NULL + N_CALIBRATION_PLANTED):
        planted = i >= N_CALIBRATION_NULL
        cfg = replace(SYNTH_CONFIG, planted=planted, seed=1000 + i)
        ds = D.build_dataset(S.write_dataset(cfg, workdir / f"cal_{i:03d}"), with_phenotype=False)
        task = ds.task("speed_um_per_hr_h1")
        if planted:
            reg["planted_p"].append(E.permutation_test(
                task, spec, base, n_perm=N_PERM_CALIBRATION, seed=SEED)["p_value"])
        else:
            for scheme in ("track_block", "row"):
                reg["null_p"][scheme].append(E.permutation_test(
                    task, spec, base, n_perm=N_PERM_CALIBRATION, seed=SEED, scheme=scheme)["p_value"])
            if i < N_CALIBRATION_NULL_CLASSIFICATION:
                lop = D.build_dataset(S.write_dataset(replace(cfg, lopsided_outline=True),
                                                      workdir / f"cal_{i:03d}_lopsided"),
                                      with_phenotype=False)
                for target in clf["targets"]:
                    for suffix, source in (("", ds), ("_lopsided_outline", lop)):
                        clf["null_p"][target + suffix].append(E.permutation_test(
                            source.task(target), cspec, cbase,
                            n_perm=N_PERM_CALIBRATION_CLASSIFICATION, seed=SEED)["p_value"])
        if i % 5 == 4:
            log(f"calibration dataset {i + 1}/{N_CALIBRATION_NULL + N_CALIBRATION_PLANTED}")
    for block in (reg, clf):
        for scheme, ps in block["null_p"].items():
            ps = np.array(ps)
            block[f"false_positive_at_0.05_{scheme}"] = _rate_with_ci(int(np.sum(ps <= 0.05)), len(ps))
            block[f"median_null_p_{scheme}"] = float(np.median(ps))
    ps = np.array(reg["planted_p"])
    reg["power_at_0.05_track_block"] = _rate_with_ci(int(np.sum(ps <= 0.05)), len(ps))
    out["calibration"] = cal
    return out


# ---------------------------------------------------------------------------
# 2. Checks on the real data that do not involve any target


def invariance_audit(ds: D.PredictionDataset, folders: list[Path]) -> dict:
    """Median relative change of each strict feature when real masks are rotated."""
    from scipy import ndimage

    changes: dict[str, list[float]] = {}
    abs_changes: dict[str, list[float]] = {}
    values: dict[str, list[float]] = {}
    n_masks = 0
    for folder in folders:
        lf = D.load_result_folder(folder)
        rows = ds.samples[ds.samples.movie == lf.movie]
        if rows.empty:
            continue
        masks = lf.masks()
        for _, r in rows.iterrows():
            m = masks[int(r.frame)] == int(r.det_label)
            sl = ndimage.find_objects(m.astype(np.uint8))[0]
            crop = np.pad(m[sl], 4)
            ref = F.strict_morphology(crop, lf.pixel_size_um)
            for k, v in ref.items():
                values.setdefault(k, []).append(v)
            for ang in AUDIT_ANGLES:
                rot = ndimage.rotate(crop.astype(np.uint8), ang, order=0, reshape=True) > 0
                got = F.strict_morphology(rot, lf.pixel_size_um)
                for k, v in ref.items():
                    denom = abs(v) if abs(v) > 1e-9 else 1.0
                    changes.setdefault(k, []).append(abs(got[k] - v) / denom)
                    abs_changes.setdefault(k, []).append(abs(got[k] - v))
            n_masks += 1
    spread = {k: float(np.std(v)) for k, v in values.items()}
    return {
        "n_masks": n_masks, "angles_deg": list(AUDIT_ANGLES),
        "method": "nearest-neighbour rotation of each real mask, features recomputed. "
                  "relative change = |f(rot) - f| / |f| (misleading for features near 0, "
                  "e.g. concave_fraction); rotation_noise_to_cell_spread = median "
                  "|f(rot) - f| / SD of f across the real masks (the number that says "
                  "whether rotation noise could masquerade as a difference between cells)",
        "median_relative_change": {k: float(np.median(v)) for k, v in changes.items()},
        "p90_relative_change": {k: float(np.quantile(v, 0.9)) for k, v in changes.items()},
        "between_cell_sd": spread,
        "rotation_noise_to_cell_spread": {
            k: (float(np.median(abs_changes[k]) / spread[k]) if spread[k] > 0 else None)
            for k in abs_changes},
    }


# ---------------------------------------------------------------------------
# 3. The real experiment


def evaluate_task(task, embedder: E.Embedder, group_by: str = "field") -> dict:
    cls = task.classification
    specs = E.CLASSIFICATION_SPECS if cls else E.REGRESSION_SPECS
    base = E.spec_by_name("majority" if cls else "mean", cls)
    out: dict = {"summary": task.summary(), "outer_split_unit": group_by, "models": {},
                 "permutation": {}}
    results = {}
    for spec in specs:
        if spec.blocks == ("phenotype",) and task.blocks["phenotype"].shape[1] == 0:
            continue
        t0 = time.time()
        r = E.cross_validate(task, spec, group_by=group_by, seed=SEED, embedder=embedder,
                             forest_trees=FOREST_TREES)
        results[spec.name] = r
        out["models"][spec.name] = {"family": spec.family, "blocks": list(spec.blocks),
                                    "estimator": spec.estimator, **E.metrics(task, r),
                                    "folds": r.fold_info, "seconds": round(time.time() - t0, 2)}
    primary = E.PRIMARY_CLASSIFICATION if cls else E.PRIMARY_REGRESSION
    for spec in specs:
        if spec.family == "baseline" or spec.name not in results:
            continue
        kind = "classification" if cls else "regression"
        n_perm = (N_PERM[f"primary_{kind}"] if spec.name == primary
                  else N_PERM["slow"] if spec.estimator in ("forest", "elasticnet")
                  else N_PERM[f"secondary_{kind}"])
        t0 = time.time()
        pt = E.permutation_test(task, spec, base, n_perm=n_perm, seed=SEED, group_by=group_by,
                                embedder=embedder, forest_trees=FOREST_TREES)
        pt["primary"] = spec.name == primary
        pt["seconds"] = round(time.time() - t0, 1)
        out["permutation"][spec.name] = pt
    if not cls:
        out["primary_effect_track_bootstrap"] = E.track_bootstrap_ci(
            task, results[primary], results["mean"], seed=SEED)
    out["delta_history"] = E.history_delta(
        task, n_perm=N_PERM["delta_classification" if cls else "delta_regression"], seed=SEED,
        group_by=group_by)
    return out


def run(folders: list[Path], out_path: Path, *, skip_synthetic: bool = False) -> dict:
    t_start = time.time()
    report: dict = {
        "generated_by": "python -m training.predict.experiment",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "design_contract": "docs/NEXT_GENERATION.md section 9 (morphology -> future migration)",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "omp_num_threads": os.environ.get("OMP_NUM_THREADS")},
        # The code that produced these numbers: training/ is uncommitted research
        # code, so a git hash alone would not identify it.
        "source_sha256": _source_hashes(),
        "settings": {
            "seed": SEED, "horizons_frames": list(D.HORIZONS_FRAMES),
            "migrating_min_net_rate_um_per_hr": D.MIGRATING_MIN_NET_RATE_UM_PER_HR,
            "conformal_nominal_coverage": 1.0 - E.CONFORMAL_ALPHA,
            "outer_split": "leave one group out; the group is the experiment from two experiments "
                           "up, else the field (movies sharing pixels merged into one field); "
                           "the unit used is data.outer_split_unit",
            "inner_selection": "grouped 5-fold over tracks inside the training fold",
            "primary_models": {"regression": E.PRIMARY_REGRESSION,
                               "classification": E.PRIMARY_CLASSIFICATION},
            "n_permutations": dict(N_PERM),
            "permutation_scheme": "track_block",
            "delta_permutation": "Freedman-Lane: strict morphology residualised on history (no target), "
                                 "residuals moved in track blocks",
            "forest": {"n_trees": FOREST_TREES, "max_depth": 3, "min_leaf": 5, "max_features": 0.33},
            "autoencoder": {"latent_dim": AE_LATENT_DIM, "epochs": AE_EPOCHS, "lr": 5e-3,
                            "input": "64x64 standardised mask crop",
                            "fitted": "once per held-out group, on every usable mask of the other groups"},
            "history_steps": F.HISTORY_STEPS,
            "contour_blur_px": F.CONTOUR_BLUR_PX, "contour_smooth_px": F.CONTOUR_SMOOTH_PX,
            "concave_tolerance_per_px": F.CONCAVE_TOLERANCE_PER_PX,
        },
    }
    if not skip_synthetic:
        log("synthetic validation")
        with tempfile.TemporaryDirectory(prefix="corridor_predict_") as tmp:
            report["synthetic_validation"] = synthetic_validation(Path(tmp))
        _write(report, out_path)

    log("building the real dataset")
    ds = D.build_dataset(folders, with_phenotype=True)
    report["data"] = {
        "folders": ds.folders,
        # As Corridor wrote them; ``folders[].n_tracks`` is already net of duplicates.
        "n_tracks_total": int(sum(f["n_tracks_in_folder"] for f in ds.folders)),
        "n_observations_total": int(sum(f["n_observations_in_folder"] for f in ds.folders)),
        "n_experiments": int(len({f["experiment"] for f in ds.folders})),
        "n_fields": int(len({f["field"] for f in ds.folders})),
        "n_tracks_after_dropping_duplicates": int(sum(f["n_tracks"] for f in ds.folders)),
        "n_observations_after_dropping_duplicates": int(sum(f["n_observations"] for f in ds.folders)),
        "exclusions_as_feature_source": ds.exclusions,
        "n_feature_observations": int(len(ds.samples)),
        "n_mask_crops_for_embedding": int(len(ds.all_crops)),
        "strict_features": ds.strict_columns,
        "phenotype_features": ds.phenotype_columns,
        "history_features": ds.history_columns,
        "multi_component_masks": int((ds.samples["mask_components"] > 1).sum()),
    }
    group_by = D.outer_grouping(ds.samples["experiment"])
    report["data"]["outer_split_unit"] = group_by
    log("invariance audit and overlap check")
    report["feature_invariance_real_masks"] = invariance_audit(ds, folders)
    report["leakage_checks"] = {
        "tracks_within_fields": True,  # build_dataset raises otherwise
        "same_pixels_ncc_threshold": D.SAME_PIXELS_NCC,
        "duplicate_max_px": D.DUPLICATE_MAX_PX,
        "overlaps_found": ds.overlaps,
        "pairwise_evidence": ds.overlap_evidence,
    }
    _write(report, out_path)

    # The embedding pool is keyed by the outer unit, so the autoencoder used to
    # predict a held-out group never saw that group's masks.
    pool_groups = ds.all_crop_experiments if group_by == "experiment" else ds.all_crop_fields
    embedder = E.Embedder(ds.all_crops, pool_groups, seed=SEED, epochs=AE_EPOCHS,
                          latent_dim=AE_LATENT_DIM)
    report["results"] = {}
    primary_p = {}
    for kind in TARGET_KINDS:
        for h in D.HORIZONS_FRAMES:
            target = f"{kind}_h{h}"
            task = ds.task(target, require_history=True, require_phenotype=True)
            log(f"{target}: n={task.n}, tracks={len(np.unique(task.track))}, "
                f"{group_by}s={len(np.unique(task.groups(group_by)))}")
            res = evaluate_task(task, embedder, group_by)
            report["results"][target] = res
            primary = E.PRIMARY_CLASSIFICATION if task.classification else E.PRIMARY_REGRESSION
            primary_p[target] = res["permutation"][primary]["p_value"]
            _write(report, out_path)
    report["autoencoder_folds"] = embedder.diagnostics
    adjusted = E.holm(primary_p)
    report["primary_tests_holm"] = {
        "family": "primary morphology model vs mean/majority, every target x horizon",
        "raw_p": primary_p, "holm_adjusted_p": adjusted,
        "n_significant_at_0.05_after_holm": int(sum(p <= 0.05 for p in adjusted.values())),
    }
    delta_p = {t: r["delta_history"]["p_value"] for t, r in report["results"].items()}
    delta_adj = E.holm(delta_p)
    report["delta_tests_holm"] = {
        "family": "Delta = perf(morphology + history) - perf(history), every target x horizon",
        "raw_p": delta_p, "holm_adjusted_p": delta_adj,
        "n_significant_at_0.05_after_holm": int(sum(p <= 0.05 for p in delta_adj.values())),
    }
    # One frame ahead, the path *is* the displacement, so the two regression
    # targets are one quantity in two units and their tests are not independent.
    rows = ds.samples.dropna(subset=["disp_um_h1"])
    ratio = rows["speed_um_per_hr_h1"] / rows["disp_um_h1"].where(rows["disp_um_h1"] > 0)
    report["notes"] = {
        "h1_speed_over_displacement_ratio": {"min": float(ratio.min()), "max": float(ratio.max()),
                                             "expected_60_over_frame_interval_min":
                                                 60.0 / ds.folders[0]["frame_interval_min"]},
    }

    # Sensitivity: targets whose path never touches a recovered detection.
    sens = {}
    for kind in ("disp_um", "speed_um_per_hr"):
        for h in D.HORIZONS_FRAMES:
            target = f"{kind}_h{h}"
            task = ds.task(target, require_history=True, require_phenotype=True)
            keep = task.rows[f"target_recovered_h{h}"].to_numpy() == 0
            if keep.sum() < 10 or len(np.unique(task.groups(group_by)[keep])) < 2:
                sens[target] = {"n": int(keep.sum()), "skipped": "too few rows or groups"}
                continue
            sub = _subset(task, keep)
            spec = E.spec_by_name(E.PRIMARY_REGRESSION, False)
            base = E.spec_by_name("mean", False)
            m = E.metrics(sub, E.cross_validate(sub, spec, group_by=group_by, seed=SEED, conformal=False))
            b = E.metrics(sub, E.cross_validate(sub, base, group_by=group_by, seed=SEED, conformal=False))
            pt = E.permutation_test(sub, spec, base, n_perm=N_PERM["secondary_regression"], seed=SEED,
                                    group_by=group_by)
            sens[target] = {"n": sub.n, "n_dropped": int((~keep).sum()), "mae": m["mae"],
                            "baseline_mae": b["mae"], "p_value": pt["p_value"]}
    report["sensitivity_primary_only_targets"] = sens

    # Sensitivity: the split the task first specified -- leave one *movie* out,
    # with 052924_t1's duplicated cell kept -- for the primary regression model.
    # It shows how much a leaky split moves the answer; it is not the result.
    log("sensitivity: leave-one-movie-out with the duplicate kept")
    leaky = D.build_dataset(folders, with_phenotype=True, detect_overlaps=False)
    lomo = {}
    for kind in ("disp_um", "speed_um_per_hr"):
        for h in D.HORIZONS_FRAMES:
            task = leaky.task(f"{kind}_h{h}", require_history=True, require_phenotype=True)
            spec = E.spec_by_name(E.PRIMARY_REGRESSION, False)
            base = E.spec_by_name("mean", False)
            m = E.metrics(task, E.cross_validate(task, spec, group_by="movie", seed=SEED, conformal=False))
            b = E.metrics(task, E.cross_validate(task, base, group_by="movie", seed=SEED, conformal=False))
            pt = E.permutation_test(task, spec, base, n_perm=N_PERM["primary_regression"],
                                    seed=SEED, group_by="movie")
            lomo[f"{kind}_h{h}"] = {"n": task.n, "n_tracks": int(len(np.unique(task.track))),
                                    "n_movies": int(len(np.unique(task.movie))),
                                    "mae": m["mae"], "baseline_mae": b["mae"],
                                    "r2_vs_training_mean": m["r2_vs_training_mean"],
                                    "p_value": pt["p_value"], "n_permutations": pt["n_permutations"]}
    report["sensitivity_leave_one_movie_out_with_duplicate"] = lomo

    # Sensitivity: only fields that the pixels prove distinct (compared at the
    # same instants). The other crops share no source frame with any movie, so
    # whether they show 052924_1's lanes hours later is undecidable
    # (dataset.pixel_established_fields); dropping them bounds their effect.
    kept = D.pixel_established_fields(ds.folders, ds.overlap_evidence)
    log(f"sensitivity: pixel-established fields only ({', '.join(kept)})")
    est: dict = {"fields_kept": kept,
                 "rule": "largest set of fields pairwise compared at a shared instant and found "
                         "to be different regions (greedy by observations)",
                 "targets": {}}
    for kind in TARGET_KINDS:
        for h in D.HORIZONS_FRAMES:
            target = f"{kind}_h{h}"
            task = ds.task(target, require_history=True, require_phenotype=True)
            keep = np.isin(task.field, kept)
            if len(np.unique(task.field[keep])) < 2:
                est["targets"][target] = {"n": int(keep.sum()), "skipped": "fewer than two fields"}
                continue
            sub = _subset(task, keep)
            cls = sub.classification
            spec = E.spec_by_name(E.PRIMARY_CLASSIFICATION if cls else E.PRIMARY_REGRESSION, cls)
            base = E.spec_by_name("majority" if cls else "mean", cls)
            m = E.metrics(sub, E.cross_validate(sub, spec, seed=SEED, conformal=False))
            b = E.metrics(sub, E.cross_validate(sub, base, seed=SEED, conformal=False))
            pt = E.permutation_test(sub, spec, base, seed=SEED, n_perm=N_PERM[
                "secondary_classification" if cls else "secondary_regression"])
            key = "balanced_accuracy" if cls else "mae"
            est["targets"][target] = {"n": sub.n, "n_dropped": int((~keep).sum()),
                                      "n_tracks": int(len(np.unique(sub.track))),
                                      "model": spec.name, key: m[key], f"baseline_{key}": b[key],
                                      "p_value": pt["p_value"], "n_permutations": pt["n_permutations"]}
    report["sensitivity_pixel_established_fields_only"] = est
    report["runtime_seconds"] = round(time.time() - t_start, 1)
    _write(report, out_path)
    log(f"done in {report['runtime_seconds']} s -> {out_path}")
    return report


def _subset(task, keep: np.ndarray):
    from dataclasses import replace as dc_replace

    idx = np.flatnonzero(keep)
    return dc_replace(
        task, y=task.y[idx], blocks={k: v[idx] for k, v in task.blocks.items()},
        crops=task.crops[idx], track=task.track[idx], movie=task.movie[idx], field=task.field[idx],
        experiment=task.experiment[idx], condition=task.condition[idx], frame=task.frame[idx],
        rows=task.rows.iloc[idx].reset_index(drop=True))


def _write(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(report), indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folders", type=Path, default=DEFAULT_FOLDERS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="exercise every code path with tiny permutation counts (numbers meaningless)")
    args = ap.parse_args(argv)
    if args.smoke:
        global N_CALIBRATION_NULL, N_CALIBRATION_PLANTED, N_PERM_CALIBRATION
        global N_CALIBRATION_NULL_CLASSIFICATION, N_PERM_CALIBRATION_CLASSIFICATION, AE_EPOCHS
        for k in N_PERM:
            N_PERM[k] = 3
        N_CALIBRATION_NULL, N_CALIBRATION_PLANTED, N_PERM_CALIBRATION = 1, 1, 3
        N_CALIBRATION_NULL_CLASSIFICATION, N_PERM_CALIBRATION_CLASSIFICATION = 1, 3
        AE_EPOCHS = 5
    folders = sorted(p for p in args.folders.iterdir() if (p / "run.json").is_file())
    run(folders, args.out, skip_synthetic=args.skip_synthetic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
