"""Estimators, baselines, the mask autoencoder and split-conformal intervals.

Why numpy and not scikit-learn: the app venv has no scikit-learn, and the
tests of this package must run in the venv the integrator runs pytest in,
next to the torch install the autoencoder needs. A separate research venv
would have meant two environments for one experiment, with torch in neither
or both. The estimators needed are few and small (ridge, elastic net,
L2-logistic, a CART random forest) and each is pinned against a closed form or
a known answer in ``tests/test_predict_models.py``.

Every estimator has the same shape:

``fit(X, y, *, groups=None, condition=None) -> self`` and
``predict(X, *, condition=None) -> ndarray``.

``groups`` are track ids. Hyperparameters are chosen by *grouped* inner
cross-validation over tracks, never by leaving out single observations:
consecutive frames of one cell are near-duplicates, and choosing a penalty by
predicting a frame from its neighbours picks the least regularised model
every time.
"""

from __future__ import annotations

import contextlib
import ctypes
import glob
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .dataset import group_kfold

RIDGE_ALPHAS = np.logspace(-2, 4, 13)
INNER_FOLDS = 5


# ---------------------------------------------------------------------------
# BLAS threads


def _numpy_openblas():
    """numpy's bundled OpenBLAS, if this build has one (ILP64 symbols end in 64_)."""
    libs = glob.glob(os.path.join(os.path.dirname(np.__file__), os.pardir, "numpy.libs", "libopenblas*"))
    for path in libs:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for suffix in ("64_", ""):
            get = getattr(lib, f"openblas_get_num_threads{suffix}", None)
            put = getattr(lib, f"openblas_set_num_threads{suffix}", None)
            if get is not None and put is not None:
                return get, put
    return None


_OPENBLAS = _numpy_openblas()


@contextlib.contextmanager
def single_threaded_blas():
    """Run numpy linear algebra on one thread inside the block.

    Every matrix here is tiny (at most a few hundred rows by ~40 columns). With
    two OpenBLAS threads on a machine already at 100 % CPU, one 31x31
    ``np.linalg.solve`` measured 1056 us against 52 us on one thread: the
    threads spend their time waiting for each other. ``OMP_NUM_THREADS`` cannot
    fix this from inside a test run, because numpy is imported (and OpenBLAS
    sized) before this package is. Silently a no-op on builds without
    OpenBLAS.
    """
    if _OPENBLAS is None:
        yield
        return
    get, put = _OPENBLAS
    before = int(get())
    put(1)
    try:
        yield
    finally:
        put(before)


# ---------------------------------------------------------------------------
# Preprocessing


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        # A constant column carries nothing; scale 1 leaves it at exactly 0.
        self.scale_ = np.where(sd > 1e-12, sd, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_


# ---------------------------------------------------------------------------
# Baselines


class MeanRegressor:
    """Population mean of the training targets."""

    def fit(self, X, y, *, groups=None, condition=None):
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X, *, condition=None):
        return np.full(len(X), self.value_)


class ConditionMedianRegressor:
    """Training median of the sample's condition, else the overall median.

    It differs from the population median only when a held-out sample's
    condition also occurs in the training fold, i.e. when one condition was
    run in at least two of the held-out groups (fields or acquisitions). The
    condition must therefore be a real label supplied with the data
    (``dataset.load_result_folder``), never derived from the acquisition or
    experiment id: under a leave-one-acquisition-out split such a label is by
    construction never in training. No condition labels exist for the supplied
    movies, so there every sample is ``unspecified`` and this *is* the
    population median; it is kept because the directive makes it mandatory.
    """

    def fit(self, X, y, *, groups=None, condition=None):
        y = np.asarray(y, dtype=float)
        self.overall_ = float(np.median(y))
        self.by_ = {}
        if condition is not None:
            condition = np.asarray(condition)
            for c in np.unique(condition):
                self.by_[c] = float(np.median(y[condition == c]))
        return self

    def predict(self, X, *, condition=None):
        if condition is None:
            return np.full(len(X), self.overall_)
        return np.array([self.by_.get(c, self.overall_) for c in condition], dtype=float)


class MajorityClassifier:
    """Predicts the training majority class (balanced accuracy 0.5 by construction)."""

    def fit(self, X, y, *, groups=None, condition=None):
        y = np.asarray(y, dtype=float)
        self.value_ = float(np.mean(y) >= 0.5)
        return self

    def predict(self, X, *, condition=None):
        return np.full(len(X), self.value_)

    def predict_proba(self, X, *, condition=None):
        return self.predict(X)


class ConditionMajorityClassifier(ConditionMedianRegressor):
    """Per-condition majority class (the median of a 0/1 target, ties to 1)."""

    def predict(self, X, *, condition=None):
        return (super().predict(X, condition=condition) >= 0.5).astype(float)

    def predict_proba(self, X, *, condition=None):
        return self.predict(X, condition=condition)


# ---------------------------------------------------------------------------
# Ridge and elastic net


_SVD_CACHE: dict = {}
_SVD_CACHE_MAX = 512


def _svd(Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD of a design matrix, memoised on its exact bytes.

    A permutation test refits the same design matrices hundreds of times with
    only ``y`` changed, and for ridge everything that depends on X alone is the
    SVD. Keying on the bytes (not on object identity) means a cache hit is the
    same numbers by construction.
    """
    key = (Xs.shape, Xs.tobytes())
    hit = _SVD_CACHE.get(key)
    if hit is None:
        if len(_SVD_CACHE) >= _SVD_CACHE_MAX:
            _SVD_CACHE.clear()
        hit = np.linalg.svd(Xs, full_matrices=False)
        _SVD_CACHE[key] = hit
    return hit


def _ridge_path(Xs: np.ndarray, yc: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Coefficients (n_alphas, p) for standardised X and centred y, via one SVD."""
    U, s, Vt = _svd(np.ascontiguousarray(Xs))
    uty = U.T @ yc
    d = s[None, :] / (s[None, :] ** 2 + alphas[:, None])
    return (d * uty[None, :]) @ Vt


class RidgeRegressor:
    """L2-penalised least squares on standardised features, penalty by grouped inner CV.

    Objective: ||y - b0 - X b||^2 + alpha ||b||^2 (sum, not mean, of squares),
    intercept unpenalised.
    """

    def __init__(self, alphas: np.ndarray = RIDGE_ALPHAS, inner_folds: int = INNER_FOLDS,
                 alpha: float | None = None, seed: int = 0):
        self.alphas = np.asarray(alphas, dtype=float)
        self.inner_folds = inner_folds
        self.fixed_alpha = alpha
        self.seed = seed

    def _select(self, X, y, groups) -> float:
        if self.fixed_alpha is not None:
            return float(self.fixed_alpha)
        folds = group_kfold(groups, self.inner_folds, self.seed) if groups is not None else []
        if not folds:
            return float(np.median(self.alphas))
        err = np.zeros(len(self.alphas))
        for tr, va in folds:
            sc = Standardizer().fit(X[tr])
            yc = y[tr] - y[tr].mean()
            B = _ridge_path(sc.transform(X[tr]), yc, self.alphas)
            pred = sc.transform(X[va]) @ B.T + y[tr].mean()
            err += ((pred - y[va, None]) ** 2).sum(axis=0)
        return float(self.alphas[int(np.argmin(err))])

    def fit(self, X, y, *, groups=None, condition=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.alpha_ = self._select(X, y, groups)
        self.scaler_ = Standardizer().fit(X)
        self.intercept_ = float(y.mean())
        self.coef_ = _ridge_path(self.scaler_.transform(X), y - self.intercept_,
                                 np.array([self.alpha_]))[0]
        return self

    def predict(self, X, *, condition=None):
        return self.scaler_.transform(X) @ self.coef_ + self.intercept_


def _soft(x: float, t: float) -> float:
    return math.copysign(max(abs(x) - t, 0.0), x)


def elastic_net_cd(Xs: np.ndarray, yc: np.ndarray, alpha: float, l1_ratio: float,
                   beta0: np.ndarray | None = None, max_iter: int = 5000,
                   tol: float = 1e-6) -> np.ndarray:
    """Coordinate descent for (1/2n)||y - Xb||^2 + alpha*(r||b||_1 + (1-r)/2 ||b||^2).

    ``Xs`` standardised (unit variance columns), ``yc`` centred; the same
    objective and parametrisation as scikit-learn's ``ElasticNet``. Works on the
    Gram matrix (p is ~30, n up to a few hundred) and, as glmnet does, sweeps
    only the non-zero coefficients until they settle, then one full sweep to
    confirm nothing new should enter. Converged when no coefficient moves by
    more than ``tol`` times the largest one.
    """
    n, p = Xs.shape
    G = (Xs.T @ Xs) / n
    c = (Xs.T @ yc) / n
    diag = np.diag(G).copy()
    beta = np.zeros(p) if beta0 is None else np.array(beta0, dtype=float)
    Gb = G @ beta
    l1 = alpha * l1_ratio
    l2 = alpha * (1.0 - l1_ratio)

    def sweep(coords) -> float:
        biggest = 0.0
        for j in coords:
            if diag[j] == 0.0:
                continue
            old = beta[j]
            rho = c[j] - Gb[j] + diag[j] * old
            new = _soft(rho, l1) / (diag[j] + l2)
            if new != old:
                Gb[:] += G[:, j] * (new - old)
                beta[j] = new
                biggest = max(biggest, abs(new - old))
        return biggest

    everything = range(p)
    for _ in range(max_iter):
        if sweep(everything) <= tol * max(1.0, float(np.max(np.abs(beta)))):
            break
        active = np.flatnonzero(beta).tolist()
        for _ in range(max_iter):
            if sweep(active) <= tol * max(1.0, float(np.max(np.abs(beta)))):
                break
    return beta


def elastic_net_path(Xs: np.ndarray, yc: np.ndarray, alphas: np.ndarray, l1_ratio: float,
                     max_iter: int = 20000, tol: float = 1e-6) -> np.ndarray:
    """The same objective for every alpha at once (rows), by accelerated proximal gradient.

    Strict morphology features are nearly collinear (area, convex area,
    equivalent diameter, perimeter...; condition number ~1e15 on the synthetic
    set), which makes coordinate descent crawl at small penalties: 2.2 s per
    path there, against 0.46 s for this FISTA (with adaptive restart) run on all
    alphas as one matrix iteration, reaching the same objective to 1e-8. Used
    for penalty selection; the chosen penalty is then polished with
    :func:`elastic_net_cd`, so the final coefficients satisfy the optimality
    conditions to coordinate-descent precision.
    """
    n, p = Xs.shape
    alphas = np.asarray(alphas, dtype=float)
    G = (Xs.T @ Xs) / n
    c = (Xs.T @ yc) / n
    lip = float(np.linalg.eigvalsh(G).max())
    l1 = alphas * l1_ratio
    l2 = alphas * (1.0 - l1_ratio)
    step = 1.0 / (lip + l2)
    B = np.zeros((len(alphas), p))
    Z = B.copy()
    t = 1.0
    for _ in range(max_iter):
        V = Z - step[:, None] * (Z @ G - c + l2[:, None] * Z)
        B_new = np.sign(V) * np.maximum(np.abs(V) - (step * l1)[:, None], 0.0)
        if np.any(np.sum((Z - B_new) * (B_new - B), axis=1) > 0):
            t = 1.0  # adaptive restart: momentum is pointing uphill
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        Z = B_new + ((t - 1.0) / t_new) * (B_new - B)
        moved = float(np.max(np.abs(B_new - B)))
        B, t = B_new, t_new
        if moved <= tol * max(1.0, float(np.max(np.abs(B)))):
            break
    return B


class ElasticNetRegressor:
    """Elastic net (l1_ratio 0.5) with the penalty chosen by grouped inner CV.

    The penalty grid runs from the smallest penalty that zeroes every
    coefficient down to 1 % of it. Below that the model is close to unpenalised
    least squares on ~30 near-collinear columns -- the regime ridge already
    covers -- and the solver needs most of its time there (23 s per
    cross-validation on the real data with the grid down to 0.1 %).
    """

    def __init__(self, l1_ratio: float = 0.5, n_alphas: int = 10, inner_folds: int = INNER_FOLDS,
                 alpha: float | None = None, seed: int = 0):
        self.l1_ratio = l1_ratio
        self.n_alphas = n_alphas
        self.inner_folds = inner_folds
        self.fixed_alpha = alpha
        self.seed = seed

    def _grid(self, Xs, yc) -> np.ndarray:
        amax = np.max(np.abs(Xs.T @ yc)) / (len(yc) * self.l1_ratio)
        amax = max(amax, 1e-6)
        return amax * np.logspace(0, -2, self.n_alphas)

    def _path(self, Xs, yc, alphas):
        return elastic_net_path(Xs, yc, alphas, self.l1_ratio)

    def fit(self, X, y, *, groups=None, condition=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.scaler_ = Standardizer().fit(X)
        Xs = self.scaler_.transform(X)
        self.intercept_ = float(y.mean())
        yc = y - self.intercept_
        if self.fixed_alpha is not None:
            self.alpha_ = float(self.fixed_alpha)
        else:
            alphas = self._grid(Xs, yc)
            folds = group_kfold(groups, self.inner_folds, self.seed) if groups is not None else []
            if folds:
                err = np.zeros(len(alphas))
                for tr, va in folds:
                    sc = Standardizer().fit(X[tr])
                    B = self._path(sc.transform(X[tr]), y[tr] - y[tr].mean(), alphas)
                    pred = sc.transform(X[va]) @ B.T + y[tr].mean()
                    err += ((pred - y[va, None]) ** 2).sum(axis=0)
                self.alpha_ = float(alphas[int(np.argmin(err))])
            else:
                self.alpha_ = float(alphas[len(alphas) // 2])
        start = elastic_net_path(Xs, yc, np.array([self.alpha_]), self.l1_ratio)[0]
        self.coef_ = elastic_net_cd(Xs, yc, self.alpha_, self.l1_ratio, beta0=start)
        return self

    def predict(self, X, *, condition=None):
        return self.scaler_.transform(X) @ self.coef_ + self.intercept_


# ---------------------------------------------------------------------------
# Logistic regression (classification)


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    """Weights so each class carries half the total, as balanced accuracy does."""
    y = np.asarray(y, dtype=float)
    n1 = y.sum()
    n0 = len(y) - n1
    w = np.where(y == 1, len(y) / (2.0 * max(n1, 1)), len(y) / (2.0 * max(n0, 1)))
    return w


def _logistic_newton(Xs: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float,
                     max_iter: int = 100, theta0: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Weighted L2 logistic regression by Newton's method; intercept unpenalised."""
    n, p = Xs.shape
    A = np.hstack([np.ones((n, 1)), Xs])
    theta = np.zeros(p + 1) if theta0 is None else theta0.copy()
    pen = np.full(p + 1, lam)
    pen[0] = 0.0
    for _ in range(max_iter):
        z = A @ theta
        mu = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = A.T @ (w * (mu - y)) + pen * theta
        H = (A * (w * mu * (1 - mu))[:, None]).T @ A + np.diag(pen + 1e-9)
        step = np.linalg.solve(H, grad)
        theta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(theta[0]), theta[1:]


class LogisticClassifier:
    """Class-balanced L2 logistic regression, penalty by grouped inner CV (weighted log-loss)."""

    def __init__(self, lambdas: np.ndarray = np.logspace(-2, 3, 11),
                 inner_folds: int = INNER_FOLDS, seed: int = 0):
        self.lambdas = np.asarray(lambdas, dtype=float)
        self.inner_folds = inner_folds
        self.seed = seed

    def fit(self, X, y, *, groups=None, condition=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.constant_ = None
        if len(np.unique(y)) < 2:
            self.constant_ = float(y[0]) if len(y) else 0.0
            return self
        folds = group_kfold(groups, self.inner_folds, self.seed) if groups is not None else []
        if folds:
            loss = np.zeros(len(self.lambdas))
            for tr, va in folds:
                if len(np.unique(y[tr])) < 2:
                    continue
                sc = Standardizer().fit(X[tr])
                Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
                w_tr = _balanced_weights(y[tr])
                w_va = _balanced_weights(y[va]) if len(np.unique(y[va])) == 2 else np.ones(len(va))
                theta = None
                # Strongest penalty first, each fit warm-started from the last.
                for k in np.argsort(-self.lambdas):
                    lam = self.lambdas[k]
                    b0, b = _logistic_newton(Xtr, y[tr], w_tr, lam, theta0=theta)
                    theta = np.r_[b0, b]
                    p = 1.0 / (1.0 + np.exp(-np.clip(b0 + Xva @ b, -30, 30)))
                    p = np.clip(p, 1e-6, 1 - 1e-6)
                    loss[k] -= np.sum(w_va * (y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p)))
            self.lambda_ = float(self.lambdas[int(np.argmin(loss))])
        else:
            self.lambda_ = float(np.median(self.lambdas))
        self.scaler_ = Standardizer().fit(X)
        self.intercept_, self.coef_ = _logistic_newton(self.scaler_.transform(X), y,
                                                       _balanced_weights(y), self.lambda_)
        return self

    def predict_proba(self, X, *, condition=None):
        if self.constant_ is not None:
            return np.full(len(X), self.constant_)
        z = self.intercept_ + self.scaler_.transform(X) @ self.coef_
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict(self, X, *, condition=None):
        return (self.predict_proba(X) >= 0.5).astype(float)


# ---------------------------------------------------------------------------
# Random forest (CART, weighted squared error)


@dataclass
class _Tree:
    feature: list[int] = field(default_factory=list)
    threshold: list[float] = field(default_factory=list)
    left: list[int] = field(default_factory=list)
    right: list[int] = field(default_factory=list)
    value: list[float] = field(default_factory=list)

    def predict(self, X: np.ndarray) -> np.ndarray:
        node = np.zeros(len(X), dtype=int)
        feature = np.array(self.feature)
        threshold = np.array(self.threshold)
        left = np.array(self.left)
        right = np.array(self.right)
        value = np.array(self.value)
        active = feature[node] >= 0
        while active.any():
            idx = np.flatnonzero(active)
            f = feature[node[idx]]
            go_left = X[idx, f] <= threshold[node[idx]]
            node[idx] = np.where(go_left, left[node[idx]], right[node[idx]])
            active = feature[node] >= 0
        return value[node]


def _grow(X, y, w, rng, max_depth, min_leaf, n_try) -> _Tree:
    tree = _Tree()

    def new_node(value: float) -> int:
        tree.feature.append(-1)
        tree.threshold.append(0.0)
        tree.left.append(-1)
        tree.right.append(-1)
        tree.value.append(value)
        return len(tree.value) - 1

    p = X.shape[1]

    def build(idx: np.ndarray, depth: int) -> int:
        wi, yi = w[idx], y[idx]
        tot_w, tot_wy = float(wi.sum()), float((wi * yi).sum())
        node = new_node(tot_wy / tot_w)
        m = len(idx)
        if depth >= max_depth or m < 2 * min_leaf:
            return node
        # All candidate features at once: one argsort and one cumsum per node
        # instead of a Python loop over features (a forest CV went from 10.0 s
        # to 3.7 s on the synthetic set).
        feats = rng.choice(p, size=n_try, replace=False)
        Xn = X[np.ix_(idx, feats)]
        order = np.argsort(Xn, axis=0, kind="stable")
        xs = np.take_along_axis(Xn, order, axis=0)
        ws, ys = wi[order], yi[order]
        cw = np.cumsum(ws, axis=0)[:-1]
        cwy = np.cumsum(ws * ys, axis=0)[:-1]
        # Gain in weighted SSE = left^2/wl + right^2/wr - parent^2/w.
        gain = cwy**2 / cw + (tot_wy - cwy) ** 2 / (tot_w - cw) - tot_wy**2 / tot_w
        pos = np.arange(1, m)[:, None]
        valid = (pos >= min_leaf) & (pos <= m - min_leaf) & (xs[1:] > xs[:-1])
        if not valid.any():
            return node
        gain = np.where(valid, gain, -np.inf)
        k, fi = np.unravel_index(int(np.argmax(gain)), gain.shape)
        if gain[k, fi] <= 1e-12:
            return node
        f = int(feats[fi])
        thr = float(0.5 * (xs[k, fi] + xs[k + 1, fi]))
        mask = X[idx, f] <= thr
        tree.feature[node] = f
        tree.threshold[node] = thr
        tree.left[node] = build(idx[mask], depth + 1)
        tree.right[node] = build(idx[~mask], depth + 1)
        return node

    build(np.arange(len(y)), 0)
    return tree


class RandomForestRegressor:
    """Bagged CART trees on weighted squared error. Fixed, pre-declared settings.

    Shallow trees with large leaves on purpose: with a few dozen tracks a deep
    forest memorises cells. The settings are not tuned (tuning on this little
    data would itself overfit) and are recorded in the experiment JSON.
    """

    def __init__(self, n_trees: int = 200, max_depth: int = 3, min_leaf: int = 5,
                 max_features: float = 0.33, seed: int = 0, balanced: bool = False):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.seed = seed
        self.balanced = balanced

    def fit(self, X, y, *, groups=None, condition=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        n_try = max(1, int(round(self.max_features * p)))
        base_w = _balanced_weights(y) if self.balanced else np.ones(n)
        self.trees_ = []
        for _ in range(self.n_trees):
            boot = rng.integers(0, n, n)
            counts = np.bincount(boot, minlength=n).astype(float)
            keep = counts > 0
            idx = np.flatnonzero(keep)
            self.trees_.append(_grow(X[idx], y[idx], (counts * base_w)[idx], rng,
                                     self.max_depth, self.min_leaf, n_try))
        return self

    def predict(self, X, *, condition=None):
        X = np.asarray(X, dtype=float)
        return np.mean([t.predict(X) for t in self.trees_], axis=0)


class RandomForestClassifier(RandomForestRegressor):
    """The forest on a 0/1 target with class-balanced weights; predicts p >= 0.5."""

    def __init__(self, **kw):
        kw.setdefault("balanced", True)
        super().__init__(**kw)

    def predict_proba(self, X, *, condition=None):
        return super().predict(X)

    def predict(self, X, *, condition=None):
        return (self.predict_proba(X) >= 0.5).astype(float)


# ---------------------------------------------------------------------------
# Learned mask embedding (torch, CPU, tiny)


class MaskAutoencoder:
    """A small convolutional autoencoder on 64x64 standardised mask crops.

    About 30 k parameters, full-batch Adam, seeded and single-process. It sees
    masks only -- never a target -- and is fitted once per held-out outer group
    (field or acquisition) on the other groups' masks (``evaluate.Embedder``),
    so a held-out group's shapes never shape the embedding that is used to
    predict it. Torch is imported here and nowhere else in the package, so the
    rest runs without it.
    """

    def __init__(self, latent_dim: int = 8, epochs: int = 150, lr: float = 5e-3,
                 seed: int = 0, threads: int = 2):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.threads = threads

    def _build(self):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        enc = nn.Sequential(
            nn.Conv2d(1, 8, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(8, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 16, 4, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(16 * 8 * 8, self.latent_dim),
        )
        dec = nn.Sequential(
            nn.Linear(self.latent_dim, 16 * 8 * 8), nn.ReLU(),
            nn.Unflatten(1, (16, 8, 8)),
            nn.ConvTranspose2d(16, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, 2, 1),
        )
        return enc, dec

    def fit(self, crops: np.ndarray) -> "MaskAutoencoder":
        import torch

        torch.set_num_threads(self.threads)
        self.enc_, self.dec_ = self._build()
        x = torch.from_numpy(np.asarray(crops, dtype=np.float32)[:, None])
        params = list(self.enc_.parameters()) + list(self.dec_.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_history_ = []
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(self.dec_(self.enc_(x)), x)
            loss.backward()
            opt.step()
            self.loss_history_.append(float(loss.detach()))
        self.n_params_ = int(sum(p.numel() for p in params))
        return self

    def transform(self, crops: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            z = self.enc_(torch.from_numpy(np.asarray(crops, dtype=np.float32)[:, None]))
        return z.numpy().astype(float)

    def reconstruction_iou(self, crops: np.ndarray) -> float:
        import torch

        with torch.no_grad():
            x = torch.from_numpy(np.asarray(crops, dtype=np.float32)[:, None])
            rec = (self.dec_(self.enc_(x)) > 0).numpy()[:, 0]
        truth = np.asarray(crops) > 0.5
        inter = (rec & truth).sum(axis=(1, 2))
        union = (rec | truth).sum(axis=(1, 2))
        return float(np.mean(inter / np.maximum(union, 1)))


# ---------------------------------------------------------------------------
# Split-conformal intervals


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest |residual|; inf if n is too small.

    Infinite rather than clipped: with fewer than (1-alpha)/alpha calibration
    points no finite interval carries the guarantee, and saying so is the
    honest answer.
    """
    r = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    n = len(r)
    k = math.ceil((n + 1) * (1.0 - alpha))
    if n == 0 or k > n:
        return math.inf
    return float(r[k - 1])


def split_conformal(make_model, X_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray,
                    X_test: np.ndarray, *, alpha: float = 0.2, cal_fraction: float = 1 / 3,
                    seed: int = 0, condition_train=None, condition_test=None
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
    """(lower, upper, info) for an (1-alpha) split-conformal interval.

    The calibration set is a random third of the training *tracks* (never of
    observations: a calibration frame whose neighbour was used for fitting has
    an optimistically small residual, and the interval would undercover). The
    guarantee assumes calibration and test samples are exchangeable; a
    held-out field or acquisition need not be, which is exactly what the
    empirical coverage in the report checks.
    """
    rng = np.random.default_rng(seed)
    tracks = np.unique(groups_train)
    n_cal = max(1, int(round(cal_fraction * len(tracks))))
    if len(tracks) - n_cal < 1:
        n = len(X_test)
        return np.full(n, -math.inf), np.full(n, math.inf), {"q": math.inf, "n_cal": 0}
    cal_tracks = set(rng.choice(tracks, size=n_cal, replace=False).tolist())
    is_cal = np.array([g in cal_tracks for g in groups_train])
    fit_idx, cal_idx = np.flatnonzero(~is_cal), np.flatnonzero(is_cal)
    cond_fit = None if condition_train is None else np.asarray(condition_train)[fit_idx]
    cond_cal = None if condition_train is None else np.asarray(condition_train)[cal_idx]
    model = make_model().fit(X_train[fit_idx], y_train[fit_idx],
                             groups=np.asarray(groups_train)[fit_idx], condition=cond_fit)
    resid = y_train[cal_idx] - model.predict(X_train[cal_idx], condition=cond_cal)
    q = conformal_quantile(resid, alpha)
    centre = model.predict(X_test, condition=condition_test)
    return centre - q, centre + q, {
        "q": q, "n_cal": int(len(cal_idx)), "n_cal_tracks": int(n_cal),
        "cal_tracks": sorted(cal_tracks, key=str),
        "fit_tracks": sorted(set(np.asarray(groups_train)[fit_idx].tolist()), key=str),
    }
