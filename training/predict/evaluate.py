"""Grouped cross-validation, metrics, the permutation test and the history Delta.

Everything here is out-of-sample at the level the claim is about:

- **Outer split**: leave one group out, groups = fields of view (movies that
  share pixels merged), or acquisitions once there are two resolved ones
  (:func:`~.dataset.outer_split`). Every split is checked with
  :func:`~.dataset.assert_no_track_straddles` before a model sees it.
- **Inner selection** (penalties): grouped by track, inside the outer training
  fold only.
- **Metrics** are computed on the pooled out-of-fold predictions. Per-fold
  values are reported too, because with a handful of movies the spread between
  folds is the honest error bar.

The permutation test, and why it permutes *track blocks*
---------------------------------------------------------
The statistic is how much the model beats the mean baseline out of fold
(MAE_mean - MAE_model; for classes, BA_model - BA_baseline). Its null
distribution comes from re-running the *whole* cross-validation on targets
that have been decoupled from the features. Shuffling single observations would
be wrong here: speeds persist along a track and differ between cells, so the
real data has far fewer independent units than rows, and a row shuffle builds a
null from data with none of that structure -- too narrow a null, and too many
false discoveries. Instead the tracks' target sequences are concatenated in a
random order and circularly shifted by a random offset
(:func:`track_block_permutation`): every target value is used exactly once, runs
of consecutive values stay together (so persistence survives), and most rows
receive another track's targets. Not all: the shift can hand a track some of
its own rows back, which leaves part of any real signal in the null and can
only make the test conservative. The synthetic null in :mod:`.synthetic` is
built to have exactly the structure that fools a row shuffle, and the
experiment reports the false-positive rate of both schemes on it.

Delta (directive section 49)
----------------------------
``Delta = perf(C: morphology + history) - perf(B: history only)``, with
perf = -MAE (so a positive Delta-MAE means morphology made the errors smaller)
and perf = R^2 or balanced accuracy. The question is conditional -- "does
morphology add anything once the cell's own recent speed is known?" -- so the
null must keep whatever morphology shares with history and destroy only the
rest. Shuffling the morphology rows outright would also break their
correlation with history, and C's null would then be "history plus noise
columns" rather than "history plus a redundant copy of history". Instead each
morphology column is split into the part a linear fit on history explains
(kept in place) and its residual (moved in track blocks): a covariate-residual
permutation, what Winkler et al. (2014) list as the Smith procedure. It is not
Freedman-Lane, which permutes the residuals of the *response* under the
reduced model. The fit uses no target, so it cannot leak one. Its level is
measured, not assumed: the experiment runs it on the synthetic nulls for both
the ridge and the logistic Delta (``synthetic_validation.calibration``). A
track-bootstrap interval for Delta-MAE is reported beside the p-value.
"""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import models as M
from .dataset import Task, assert_no_track_straddles, leave_one_group_out

CONFORMAL_ALPHA = 0.2  # 80 % intervals


def _one_blas_thread(fn):
    """See ``models.single_threaded_blas``: tiny matrices, a busy machine."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with M.single_threaded_blas():
            return fn(*args, **kwargs)
    return wrapper


@dataclass(frozen=True)
class ModelSpec:
    name: str
    #: What kind of evidence the model is: "baseline", "morphology",
    #: "learned morphology", "phenotype (not morphology)", "comparator: motion
    #: history", "morphology + history".
    family: str
    blocks: tuple[str, ...]
    estimator: str


REGRESSION_SPECS = (
    ModelSpec("mean", "baseline", (), "mean"),
    ModelSpec("condition_median", "baseline", (), "condition_median"),
    ModelSpec("ridge_morphology", "morphology", ("strict",), "ridge"),
    ModelSpec("elasticnet_morphology", "morphology", ("strict",), "elasticnet"),
    ModelSpec("forest_morphology", "morphology", ("strict",), "forest"),
    ModelSpec("autoencoder_ridge", "learned morphology", ("embedding",), "ridge"),
    ModelSpec("combined_ridge", "morphology", ("strict", "embedding"), "ridge"),
    ModelSpec("phenotype_ridge", "phenotype (not morphology)", ("phenotype",), "ridge"),
    ModelSpec("history_ridge", "comparator: motion history", ("history",), "ridge"),
    ModelSpec("morphology_history_ridge", "morphology + history", ("strict", "history"), "ridge"),
)
CLASSIFICATION_SPECS = (
    ModelSpec("majority", "baseline", (), "majority"),
    ModelSpec("condition_majority", "baseline", (), "condition_majority"),
    ModelSpec("logistic_morphology", "morphology", ("strict",), "logistic"),
    ModelSpec("forest_morphology", "morphology", ("strict",), "forest"),
    ModelSpec("autoencoder_logistic", "learned morphology", ("embedding",), "logistic"),
    ModelSpec("combined_logistic", "morphology", ("strict", "embedding"), "logistic"),
    ModelSpec("phenotype_logistic", "phenotype (not morphology)", ("phenotype",), "logistic"),
    ModelSpec("history_logistic", "comparator: motion history", ("history",), "logistic"),
    ModelSpec("morphology_history_logistic", "morphology + history", ("strict", "history"), "logistic"),
)
#: The pre-declared primary test of "morphology beats the mean": one model per
#: task, so the p-values are not the best of several tries.
PRIMARY_REGRESSION = "ridge_morphology"
PRIMARY_CLASSIFICATION = "logistic_morphology"
HISTORY_B = {"regression": "history_ridge", "classification": "history_logistic"}
HISTORY_C = {"regression": "morphology_history_ridge", "classification": "morphology_history_logistic"}


def spec_by_name(name: str, classification: bool) -> ModelSpec:
    for s in (CLASSIFICATION_SPECS if classification else REGRESSION_SPECS):
        if s.name == name:
            return s
    raise KeyError(name)


def make_estimator(spec: ModelSpec, classification: bool, seed: int = 0,
                   forest_trees: int = 200):
    e = spec.estimator
    if e == "mean":
        return M.MeanRegressor()
    if e == "condition_median":
        return M.ConditionMedianRegressor()
    if e == "majority":
        return M.MajorityClassifier()
    if e == "condition_majority":
        return M.ConditionMajorityClassifier()
    if e == "ridge":
        return M.RidgeRegressor(seed=seed)
    if e == "elasticnet":
        return M.ElasticNetRegressor(seed=seed)
    if e == "logistic":
        return M.LogisticClassifier(seed=seed)
    if e == "forest":
        cls = M.RandomForestClassifier if classification else M.RandomForestRegressor
        return cls(n_trees=forest_trees, seed=seed)
    raise ValueError(e)


# ---------------------------------------------------------------------------
# Learned embeddings, one autoencoder per held-out group


class Embedder:
    """Fits the mask autoencoder on every crop *outside* the held-out group.

    ``pool_crops``/``pool_groups`` are all usable masks (including frames with
    no future target) and their outer groups (``PredictionDataset.group_ids``
    of the outer unit: field or acquisition), so the unsupervised part learns
    from as much shape as exists without ever seeing the held-out group. Fitted
    models are cached by held-out group, so every task, model and permutation
    that holds out the same group reuses one autoencoder (it never sees
    targets, so permuting targets cannot change it).
    """

    def __init__(self, pool_crops: np.ndarray, pool_groups: np.ndarray, *, seed: int = 0,
                 epochs: int = 150, latent_dim: int = 8):
        self.pool_crops = pool_crops
        self.pool_groups = np.asarray(pool_groups)
        self.seed = seed
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.cache: dict = {}
        self.diagnostics: dict = {}

    def for_held_out(self, group) -> M.MaskAutoencoder:
        key = str(group)
        if not np.any(self.pool_groups == group):
            # A group the pool does not know (e.g. a movie name when the pool is
            # keyed by field) would exclude nothing and train on the held-out cells.
            raise ValueError(f"held-out group {group!r} is not a group of the embedding pool")
        if key not in self.cache:
            train = self.pool_crops[self.pool_groups != group]
            t0 = time.perf_counter()
            ae = M.MaskAutoencoder(latent_dim=self.latent_dim, epochs=self.epochs,
                                   seed=self.seed).fit(train)
            fit_seconds = time.perf_counter() - t0
            held = self.pool_crops[self.pool_groups == group]
            self.diagnostics[key] = {
                "n_train_crops": int(len(train)),
                "n_parameters": ae.n_params_,
                "fit_seconds": round(fit_seconds, 1),
                "final_bce": round(ae.loss_history_[-1], 5),
                "reconstruction_iou_train": round(ae.reconstruction_iou(train), 4),
                "reconstruction_iou_held_out": (round(ae.reconstruction_iou(held), 4)
                                                if len(held) else None),
            }
            self.cache[key] = ae
        return self.cache[key]


# ---------------------------------------------------------------------------
# Cross-validation


@dataclass
class CVResult:
    spec: ModelSpec
    pred: np.ndarray
    fold_mean: np.ndarray          # training-fold mean target, per test row
    fold: np.ndarray               # fold index per row
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    fold_info: list[dict] = field(default_factory=list)


def _design(task: Task, spec: ModelSpec, rows: np.ndarray, ae: M.MaskAutoencoder | None,
            block_rows: dict[str, np.ndarray] | None,
            block_values: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Design matrix of ``rows``. ``block_rows`` reads a block from other rows
    (a permutation); ``block_values`` replaces a block's values outright."""
    parts = []
    for b in spec.blocks:
        src = rows if not block_rows or b not in block_rows else block_rows[b][rows]
        if b == "embedding":
            parts.append(ae.transform(task.crops[src]))
        elif block_values and b in block_values:
            parts.append(block_values[b][src])
        else:
            parts.append(task.blocks[b][src])
    if not parts:
        return np.zeros((len(rows), 1))
    return np.hstack(parts)


@_one_blas_thread
def cross_validate(task: Task, spec: ModelSpec, *, group_by: str = "field", seed: int = 0,
                   y: np.ndarray | None = None, block_rows: dict[str, np.ndarray] | None = None,
                   block_values: dict[str, np.ndarray] | None = None,
                   embedder: Embedder | None = None, conformal: bool = True,
                   alpha: float = CONFORMAL_ALPHA, forest_trees: int = 200) -> CVResult:
    y = task.y if y is None else np.asarray(y, dtype=float)
    groups = task.groups(group_by)
    splits = leave_one_group_out(groups, task.track)
    n = task.n
    pred = np.full(n, np.nan)
    fold_mean = np.full(n, np.nan)
    fold = np.full(n, -1)
    lower = np.full(n, np.nan) if conformal and not task.classification else None
    upper = np.full(n, np.nan) if conformal and not task.classification else None
    info = []
    if "embedding" in spec.blocks and embedder is None:
        raise ValueError(f"{spec.name} needs an Embedder")
    for k, (tr, te) in enumerate(splits):
        assert_no_track_straddles(task.track, tr, te)
        held = groups[te][0]
        ae = embedder.for_held_out(held) if "embedding" in spec.blocks else None
        X_tr = _design(task, spec, tr, ae, block_rows, block_values)
        X_te = _design(task, spec, te, ae, block_rows, block_values)

        def make():
            return make_estimator(spec, task.classification, seed, forest_trees)

        model = make().fit(X_tr, y[tr], groups=task.track[tr], condition=task.condition[tr])
        pred[te] = model.predict(X_te, condition=task.condition[te])
        fold_mean[te] = float(np.mean(y[tr]))
        fold[te] = k
        entry = {"held_out": str(held), "n_train": int(len(tr)), "n_test": int(len(te)),
                 "n_train_tracks": int(len(np.unique(task.track[tr]))),
                 "n_test_tracks": int(len(np.unique(task.track[te])))}
        if lower is not None:
            lo, hi, cinfo = M.split_conformal(
                make, X_tr, y[tr], task.track[tr], X_te, alpha=alpha, seed=seed + k,
                condition_train=task.condition[tr], condition_test=task.condition[te])
            lower[te], upper[te] = lo, hi
            entry["conformal"] = {"q": cinfo["q"] if math.isfinite(cinfo["q"]) else None,
                                  "n_cal": cinfo["n_cal"]}
        info.append(entry)
    return CVResult(spec, pred, fold_mean, fold, lower, upper, info)


# ---------------------------------------------------------------------------
# Metrics


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean recall over the classes present in ``y``."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    recalls = [float(np.mean(pred[y == c] == c)) for c in (0.0, 1.0) if np.any(y == c)]
    return float(np.mean(recalls)) if recalls else math.nan


def regression_metrics(y: np.ndarray, r: CVResult) -> dict:
    ok = r.fold >= 0
    y, p, m = y[ok], r.pred[ok], r.fold_mean[ok]
    sse = float(np.sum((y - p) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    sse_fold_mean = float(np.sum((y - m) ** 2))
    out = {
        "n": int(ok.sum()),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(math.sqrt(sse / len(y))),
        # Conventional R^2 of the pooled out-of-fold predictions...
        "r2": 1.0 - sse / sst if sst > 0 else math.nan,
        # ...and against the mean of each fold's *training* targets, the only
        # mean a real prediction could have used. Positive = beats it.
        "r2_vs_training_mean": 1.0 - sse / sse_fold_mean if sse_fold_mean > 0 else math.nan,
        "per_fold_mae": {str(i["held_out"]): float(np.mean(np.abs(y[r.fold[ok] == k] - p[r.fold[ok] == k])))
                         for k, i in enumerate(r.fold_info)},
    }
    if r.lower is not None:
        lo, hi = r.lower[ok], r.upper[ok]
        inside = (y >= lo) & (y <= hi)
        finite = np.isfinite(hi - lo)
        out["interval_nominal"] = 1.0 - CONFORMAL_ALPHA
        out["interval_coverage"] = float(np.mean(inside))
        out["interval_mean_width"] = float(np.mean((hi - lo)[finite])) if finite.any() else None
        out["interval_infinite_fraction"] = float(np.mean(~finite))
        out["per_fold_coverage"] = {str(i["held_out"]): float(np.mean(inside[r.fold[ok] == k]))
                                    for k, i in enumerate(r.fold_info)}
    return out


def classification_metrics(y: np.ndarray, r: CVResult) -> dict:
    ok = r.fold >= 0
    y, p = y[ok], r.pred[ok]
    return {
        "n": int(ok.sum()),
        "n_migrating": int(np.sum(y == 1)),
        "n_stalled": int(np.sum(y == 0)),
        "balanced_accuracy": balanced_accuracy(y, p),
        "sensitivity_migrating": float(np.mean(p[y == 1] == 1)) if np.any(y == 1) else math.nan,
        "specificity_stalled": float(np.mean(p[y == 0] == 0)) if np.any(y == 0) else math.nan,
        "accuracy": float(np.mean(p == y)),
        "per_fold_balanced_accuracy": {str(i["held_out"]): balanced_accuracy(y[r.fold[ok] == k], p[r.fold[ok] == k])
                                       for k, i in enumerate(r.fold_info)},
    }


def metrics(task: Task, r: CVResult, y: np.ndarray | None = None) -> dict:
    y = task.y if y is None else y
    return classification_metrics(y, r) if task.classification else regression_metrics(y, r)


def _score(task: Task, r: CVResult, y: np.ndarray) -> float:
    """The single number permutation tests compare: higher = better."""
    ok = r.fold >= 0
    if task.classification:
        return balanced_accuracy(y[ok], r.pred[ok])
    return -float(np.mean(np.abs(y[ok] - r.pred[ok])))


# ---------------------------------------------------------------------------
# Permutations


def track_block_permutation(track: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Index permutation: tracks' rows concatenated in random order, circularly shifted.

    Rows must already be grouped by track (``Task`` rows are sorted by track,
    then frame). Apply as ``y[perm]``.
    """
    track = np.asarray(track)
    starts = np.flatnonzero(np.r_[True, track[1:] != track[:-1]])
    ends = np.r_[starts[1:], len(track)]
    if len(np.unique(track)) != len(starts):
        raise ValueError("rows are not grouped by track")
    order = rng.permutation(len(starts))
    seq = np.concatenate([np.arange(starts[i], ends[i]) for i in order])
    return np.roll(seq, int(rng.integers(len(seq))))


def row_permutation(track: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The naive scheme, kept only to measure how wrong it is."""
    return rng.permutation(len(track))


SCHEMES = {"track_block": track_block_permutation, "row": row_permutation}


def _permutation_p(null: np.ndarray, observed: float) -> float:
    """(1 + #{null >= observed}) / (n + 1), counting numerical ties as ties.

    A permutation that rebuilds the observed design up to rounding (e.g. a
    morphology block that history explains exactly, see :func:`history_delta`)
    scores within ~1e-11 of the observed value, on either side. Counting those
    as "below" would shrink p for no reason, so the tie band is relative.
    """
    tol = 1e-9 * max(1.0, abs(float(observed)))
    return (1 + int(np.sum(np.asarray(null) >= observed - tol))) / (len(null) + 1)


@_one_blas_thread
def permutation_test(task: Task, spec: ModelSpec, baseline: ModelSpec, *, n_perm: int = 199,
                     seed: int = 0, group_by: str = "field", scheme: str = "track_block",
                     embedder: Embedder | None = None, forest_trees: int = 200) -> dict:
    """p-value for "``spec`` beats ``baseline`` out of fold" (one-sided)."""
    def improvement(y):
        rm = cross_validate(task, spec, group_by=group_by, seed=seed, y=y, embedder=embedder,
                            conformal=False, forest_trees=forest_trees)
        rb = cross_validate(task, baseline, group_by=group_by, seed=seed, y=y,
                            conformal=False)
        return _score(task, rm, y) - _score(task, rb, y)

    observed = improvement(task.y)
    rng = np.random.default_rng(seed)
    shuffle = SCHEMES[scheme]
    null = np.array([improvement(task.y[shuffle(task.track, rng)]) for _ in range(n_perm)])
    p = _permutation_p(null, observed)
    return {
        "statistic": "BA_model - BA_baseline" if task.classification else "MAE_baseline - MAE_model",
        "model": spec.name, "baseline": baseline.name, "scheme": scheme,
        "observed": float(observed), "n_permutations": int(n_perm), "p_value": float(p),
        "null_mean": float(np.mean(null)), "null_q95": float(np.quantile(null, 0.95)),
    }


def history_residualisation(strict: np.ndarray, history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(fitted, residual) of each morphology column regressed on history (with intercept).

    The covariate split used by :func:`history_delta`: ``fitted`` is what
    history already says about the shape and stays in place, ``residual`` is
    what only the shape says and is the part a permutation may move. Least
    squares on the covariates alone -- no target enters, so the split cannot
    carry the outcome into the null.
    """
    strict = np.asarray(strict, dtype=float)
    A = np.hstack([np.ones((len(strict), 1)), np.asarray(history, dtype=float)])
    coef, *_ = np.linalg.lstsq(A, strict, rcond=None)
    fitted = A @ coef
    return fitted, strict - fitted


@_one_blas_thread
def history_delta(task: Task, *, n_perm: int = 199, seed: int = 0, group_by: str = "field",
                  n_boot: int = 2000) -> dict:
    """Delta = perf(C: morphology + history) - perf(B: history only), with a permutation p.

    The null replaces the strict block by ``fitted + residual[perm]`` (module
    docstring), so a morphology column that is only a noisy copy of history
    cannot make C beat B under the null any more than under the observed data.
    """
    kind = "classification" if task.classification else "regression"
    B = spec_by_name(HISTORY_B[kind], task.classification)
    C = spec_by_name(HISTORY_C[kind], task.classification)
    rB = cross_validate(task, B, group_by=group_by, seed=seed, conformal=False)
    rC = cross_validate(task, C, group_by=group_by, seed=seed, conformal=False)
    mB, mC = metrics(task, rB), metrics(task, rC)
    observed = _score(task, rC, task.y) - _score(task, rB, task.y)
    fitted, resid = history_residualisation(task.blocks["strict"], task.blocks["history"])
    explained = 1.0 - resid.var(axis=0) / np.maximum(task.blocks["strict"].var(axis=0), 1e-300)
    rng = np.random.default_rng(seed + 1)
    null = []
    for _ in range(n_perm):
        perm = track_block_permutation(task.track, rng)
        rP = cross_validate(task, C, group_by=group_by, seed=seed, conformal=False,
                            block_values={"strict": fitted + resid[perm]})
        null.append(_score(task, rP, task.y) - _score(task, rB, task.y))
    null = np.array(null)
    p = _permutation_p(null, observed)
    out = {"B": B.name, "C": C.name,
           "permutation": "covariate-residual (Smith): strict morphology residualised on "
                          "history, residuals moved in track blocks",
           "strict_variance_explained_by_history_median": float(np.median(explained)),
           "n_permutations": int(n_perm), "p_value": float(p),
           "null_mean": float(np.mean(null)), "null_q95": float(np.quantile(null, 0.95))}
    if task.classification:
        out.update({"delta_balanced_accuracy": mC["balanced_accuracy"] - mB["balanced_accuracy"],
                    "B_balanced_accuracy": mB["balanced_accuracy"],
                    "C_balanced_accuracy": mC["balanced_accuracy"]})
    else:
        out.update({"delta_mae": mB["mae"] - mC["mae"],
                    "delta_r2": mC["r2"] - mB["r2"],
                    "B_mae": mB["mae"], "C_mae": mC["mae"], "B_r2": mB["r2"], "C_r2": mC["r2"],
                    "delta_mae_track_bootstrap": track_bootstrap_ci(task, rC, rB, n_boot=n_boot,
                                                                    seed=seed)})
    return out


def track_bootstrap_ci(task: Task, r_model: CVResult, r_base: CVResult, *, n_boot: int = 2000,
                       seed: int = 0, level: float = 0.95) -> dict:
    """CI for MAE_base - MAE_model, resampling whole tracks (predictions held fixed)."""
    ok = (r_model.fold >= 0) & (r_base.fold >= 0)
    d = np.abs(task.y - r_base.pred) - np.abs(task.y - r_model.pred)
    tracks = np.unique(task.track[ok])
    per_track = [d[ok & (task.track == t)] for t in tracks]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(tracks), len(tracks))
        stats.append(float(np.mean(np.concatenate([per_track[i] for i in pick]))))
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"estimate": float(np.mean(d[ok])), "ci_low": float(lo), "ci_high": float(hi),
            "level": level, "resampled_unit": "track", "n_tracks": int(len(tracks))}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (family = the dict)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
