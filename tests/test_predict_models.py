"""The numpy estimators against closed forms, optimality conditions and known answers.

scikit-learn is not in the app venv, so these are the evidence that the
hand-written estimators solve the problems they claim to.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# training/ is research code and deliberately not an installed package (the
# installer must never contain it), so the checkout root has to be importable
# whatever pytest.ini's pythonpath says.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.predict import models as M  # noqa: E402
from training.predict.dataset import group_kfold  # noqa: E402


def _data(n=120, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * rng.uniform(0.5, 3.0, p) + rng.normal(size=p)
    beta = np.array([2.0, -1.0, 0.0, 0.0, 0.5, 0.0])[:p]
    y = 3.0 + X @ beta + rng.normal(0, 0.5, n)
    groups = np.repeat(np.arange(n // 6), 6)
    return X, y, groups


def test_ridge_matches_the_closed_form():
    X, y, _ = _data()
    alpha = 7.5
    model = M.RidgeRegressor(alpha=alpha).fit(X, y)
    Xs = model.scaler_.transform(X)
    yc = y - y.mean()
    expected = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(X.shape[1]), Xs.T @ yc)
    np.testing.assert_allclose(model.coef_, expected, rtol=1e-9, atol=1e-10)
    tiny = M.RidgeRegressor(alpha=1e-10).fit(X, y)
    A = np.hstack([np.ones((len(X), 1)), X])
    ols = np.linalg.lstsq(A, y, rcond=None)[0]
    np.testing.assert_allclose(tiny.predict(X), A @ ols, rtol=1e-7)


def test_ridge_penalty_is_chosen_by_whole_tracks(monkeypatch):
    """The inner folds receive the tracks, keep them whole, and that changes the answer.

    Each cell's features and target are constant along its track and unrelated
    across cells, with more features than cells. Predicting a frame from its
    own track's other frames is then trivial and rewards the least penalty;
    predicting an unseen cell is impossible and rewards the most.
    """
    seen = []
    real = M.group_kfold

    def spy(groups, k, seed=0):
        folds = real(groups, k, seed)
        seen.append((np.asarray(groups).copy(), folds))
        return folds

    monkeypatch.setattr(M, "group_kfold", spy)
    rng = np.random.default_rng(0)
    n_tracks, per, p = 24, 5, 30
    track = np.repeat(np.arange(n_tracks), per)
    X = rng.normal(size=(n_tracks, p))[track] + 0.01 * rng.normal(size=(len(track), p))
    y = rng.normal(size=n_tracks)[track] + 0.01 * rng.normal(size=len(track))

    by_track = M.RidgeRegressor(seed=1).fit(X, y, groups=track).alpha_
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0][0], track)
    assert len(seen[0][1]) == M.INNER_FOLDS
    for tr, va in seen[0][1]:
        assert not set(track[tr]) & set(track[va])

    by_row = M.RidgeRegressor(seed=1).fit(X, y, groups=np.arange(len(y))).alpha_
    assert by_track == M.RIDGE_ALPHAS.max()
    assert by_row <= M.RIDGE_ALPHAS[2]


def test_elastic_net_satisfies_its_optimality_conditions():
    X, y, _ = _data(p=6)
    model = M.ElasticNetRegressor(alpha=0.05, l1_ratio=0.5).fit(X, y)
    Xs = model.scaler_.transform(X)
    r = (y - y.mean()) - Xs @ model.coef_
    n = len(y)
    l1, l2 = 0.05 * 0.5, 0.05 * 0.5
    grad = Xs.T @ r / n - l2 * model.coef_
    for j, b in enumerate(model.coef_):
        if b != 0:
            assert grad[j] == pytest.approx(l1 * np.sign(b), abs=1e-5)
        else:
            assert abs(grad[j]) <= l1 + 1e-6
    # Strong penalty zeroes the irrelevant columns first.
    sparse = M.ElasticNetRegressor(alpha=0.4, l1_ratio=0.9).fit(X, y)
    assert np.all(sparse.coef_[[2, 3, 5]] == 0)
    assert sparse.coef_[0] != 0


def test_logistic_gradient_vanishes_and_it_separates():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=200) > 0.4).astype(float)
    model = M.LogisticClassifier(lambdas=np.array([1.0])).fit(X, y)
    Xs = model.scaler_.transform(X)
    p = model.predict_proba(X)
    w = M._balanced_weights(y)
    grad = Xs.T @ (w * (p - y)) + 1.0 * model.coef_
    np.testing.assert_allclose(grad, 0.0, atol=1e-6)
    assert np.mean(model.predict(X) == y) > 0.85


def test_forest_finds_a_step_and_ignores_noise():
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(300, 5))
    y = 3.0 * (X[:, 2] > 0.5) + rng.normal(0, 0.1, 300)
    model = M.RandomForestRegressor(n_trees=60, max_depth=3, min_leaf=5, max_features=1.0, seed=0)
    pred = model.fit(X, y).predict(X)
    assert np.mean(np.abs(pred - 3.0 * (X[:, 2] > 0.5))) < 0.15
    roots = {t.feature[0] for t in model.trees_}
    assert roots == {2}


def test_conformal_quantile_refuses_too_few_points():
    assert M.conformal_quantile(np.array([1.0, 2.0, 3.0, 4.0]), 0.2) == 4.0
    assert M.conformal_quantile(np.array([1.0, 2.0, 3.0]), 0.2) == math.inf


def test_split_conformal_covers_iid_data_and_splits_by_track():
    rng = np.random.default_rng(7)
    covered = []
    for rep in range(60):
        n_tracks, per = 30, 5
        groups = np.repeat(np.arange(n_tracks), per)
        X = rng.normal(size=(n_tracks * per, 2))
        y = X[:, 0] + rng.normal(0, 1.0, len(X))
        Xt = rng.normal(size=(50, 2))
        yt = Xt[:, 0] + rng.normal(0, 1.0, 50)
        lo, hi, info = M.split_conformal(lambda: M.RidgeRegressor(alpha=1.0), X, y, groups, Xt,
                                         alpha=0.2, seed=rep)
        assert not set(info["fit_tracks"]) & set(info["cal_tracks"])
        covered.append(np.mean((yt >= lo) & (yt <= hi)))
    assert np.mean(covered) == pytest.approx(0.8, abs=0.04)


def test_group_kfold_never_cuts_a_group():
    groups = np.repeat(np.arange(17), np.arange(1, 18))
    folds = group_kfold(groups, 5, seed=0)
    assert len(folds) == 5
    seen = np.zeros(len(groups), dtype=int)
    for tr, va in folds:
        assert not set(groups[tr]) & set(groups[va])
        seen[va] += 1
    assert np.all(seen == 1)


def test_autoencoder_is_tiny_seeded_and_deterministic():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    crops = np.zeros((12, 64, 64), dtype=np.float32)
    for i in range(12):
        h = int(rng.integers(8, 30))
        crops[i, 32 - h:32 + h, 28:36] = 1.0
    a = M.MaskAutoencoder(epochs=15, seed=3).fit(crops)
    b = M.MaskAutoencoder(epochs=15, seed=3).fit(crops)
    za, zb = a.transform(crops), b.transform(crops)
    assert za.shape == (12, 8)
    np.testing.assert_allclose(za, zb, rtol=1e-5, atol=1e-6)
    assert a.n_params_ < 50_000
    assert a.loss_history_[-1] < a.loss_history_[0]
