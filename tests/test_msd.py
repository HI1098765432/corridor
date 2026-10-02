"""Mean squared displacement, its exponent, and directional persistence (§6).

Every random input is drawn from a fixed seed, so these tests are
deterministic: a tolerance here is chosen against the estimator's measured
bias (see ``measurements.msd_fit``), not against luck.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from corridor.core.config import MeasurementConfig, Scale
from corridor.core.export import MSD_COLUMNS
from corridor.core.measurements import (
    directional_autocorrelation,
    msd_fit,
    msd_rows,
    summarise,
)


@dataclass
class Obs:
    frame: int
    x: float
    y: float
    area_px: float = 100.0


@dataclass
class Trk:
    id: int
    observations: list[Obs]
    flags: set[str] = field(default_factory=set)
    channel: int = 0


def track(points, frames=None, tid=1) -> Trk:
    frames = list(range(len(points))) if frames is None else list(frames)
    return Trk(tid, [Obs(f, float(p[0]), float(p[1])) for f, p in zip(frames, points)])


UNIT = Scale.from_values(1.0, 1.0)


def by_lag(rows):
    return {r["lag_frames"]: r for r in rows}


def test_a_stationary_cell_has_zero_msd_and_no_exponent():
    rows = msd_rows([track([(5.0, 5.0)] * 10)], UNIT)
    assert rows and all(r["msd_um2"] == 0.0 and r["msd_px2"] == 0.0 for r in rows)
    assert msd_fit(rows, MeasurementConfig()) is None
    (summary,) = summarise([track([(5.0, 5.0)] * 10)], UNIT)
    assert summary.msd_alpha is None and summary.msd_alpha_r2 is None


def test_constant_velocity_is_ballistic():
    """MSD = v^2 tau^2 exactly, so alpha = 2 with a perfect fit."""
    v_px = 3.0  # px per frame
    scale = Scale.from_values(0.5, 20.0)
    points = [(v_px * k, 0.0) for k in range(21)]
    rows = msd_rows([track(points)], scale)

    for r in rows:
        lag = r["lag_frames"]
        assert r["msd_px2"] == pytest.approx((v_px * lag) ** 2)
        assert r["msd_um2"] == pytest.approx((v_px * 0.5 * lag) ** 2)
        assert r["lag_time_min"] == pytest.approx(20.0 * lag)
        assert r["lag_time_hr"] == pytest.approx(20.0 * lag / 60.0)
        assert r["n_pairs"] == 21 - lag
    alpha, r2, n_lags = msd_fit(rows, MeasurementConfig())
    assert alpha == pytest.approx(2.0, abs=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-12)
    assert n_lags == 10  # lags 1..10: at most half of the 20-frame span


def test_msd_is_in_square_micrometres():
    rows = msd_rows([track([(0, 0), (10, 0), (10, 10)])], Scale.from_values(0.5, 1.0))
    for r in rows:
        assert r["msd_um2"] == pytest.approx(0.25 * r["msd_px2"])


def test_a_random_walk_is_diffusive():
    """Many seeded 60-step walks; the median exponent is near 1.

    Measured: 0.905-0.979 across 20 seeds (the time-averaged estimator of a
    short track is biased low), 0.929 for this seed.  The bounds hold that
    measurement with margin and still separate diffusion from both
    confinement (alpha well below 1) and ballistic motion (2).
    """
    rng = np.random.default_rng(20261002)
    tracks = [
        track(np.vstack([[0, 0], np.cumsum(rng.normal(size=(60, 2)), axis=0)]), tid=t + 1)
        for t in range(200)
    ]
    alphas = [s.msd_alpha for s in summarise(tracks, UNIT)]
    assert all(a is not None for a in alphas)
    assert 0.85 < float(np.median(alphas)) < 1.1


def test_lags_are_actual_frame_differences():
    """Frames 1, 2, 5 contribute lags 1, 3 and 4 -- not 1, 2 and 3."""
    rows = msd_rows([track([(0, 0), (1, 0), (4, 0)], frames=[1, 2, 5])], UNIT)
    lags = by_lag(rows)
    assert sorted(lags) == [1, 3, 4]
    assert [lags[k]["n_pairs"] for k in (1, 3, 4)] == [1, 1, 1]
    assert lags[3]["msd_px2"] == pytest.approx(9.0)
    assert lags[4]["msd_px2"] == pytest.approx(16.0)


def test_pairs_are_grouped_across_gaps():
    # Frames 0,1,2,4,5: lag 1 comes from (0,1), (1,2) and (4,5).
    rows = msd_rows([track([(k, 0) for k in (0, 1, 2, 4, 5)], frames=[0, 1, 2, 4, 5])], UNIT)
    lags = by_lag(rows)
    assert {k: lags[k]["n_pairs"] for k in lags} == {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}


def test_the_fit_is_refused_with_too_few_lags():
    short = msd_rows([track([(k * 2.0, 0) for k in range(5)])], UNIT)
    # Span 4 frames: only lags 1 and 2 are within half of it.
    assert msd_fit(short, MeasurementConfig()) is None
    assert msd_fit(short, MeasurementConfig(msd_min_lags_for_fit=2)) is not None

    rows = msd_rows([track([(k * 2.0, 0) for k in range(21)])], UNIT)
    strict = MeasurementConfig(msd_min_pairs=19)  # only lags 1 and 2 have 19+ pairs
    assert msd_fit(rows, strict) is None


def test_the_fit_honours_the_lag_fraction():
    rows = msd_rows([track([(k * 2.0, 0) for k in range(41)])], UNIT)
    _, _, n_lags = msd_fit(rows, MeasurementConfig(msd_max_lag_fraction=0.25))
    assert n_lags == 10


def test_rows_carry_exactly_the_schema_columns():
    rows = msd_rows([track([(0, 0), (1, 1), (3, 1)])], UNIT)
    assert all(set(r) == set(MSD_COLUMNS) for r in rows)


def test_uncalibrated_msd_keeps_pixels():
    rows = msd_rows([track([(0, 0), (3, 4)])], Scale.from_values(None, None))
    assert rows[0]["msd_px2"] == 25.0
    assert rows[0]["msd_um2"] is None and rows[0]["lag_time_min"] is None


def test_single_observation_tracks_have_no_msd():
    assert msd_rows([track([(0, 0)])], UNIT) == []


# --------------------------------------------------------------------------
# Directional persistence
# --------------------------------------------------------------------------


def test_a_straight_track_keeps_its_heading():
    obs = track([(k, 0) for k in range(10)]).observations
    curve = directional_autocorrelation(obs, UNIT)
    assert all(c == pytest.approx(1.0) for c, _ in curve.values())
    (summary,) = summarise([track([(k, 0) for k in range(10)])], UNIT)
    assert summary.directional_autocorrelation == pytest.approx(1.0)
    # No decay observed: an infinite persistence time is not a measurement.
    assert summary.persistence_time_min is None and summary.persistence_fit_r2 is None


def test_a_back_and_forth_track_is_anticorrelated():
    (summary,) = summarise([track([(k % 2, 0) for k in range(10)])], UNIT)
    assert summary.directional_autocorrelation == pytest.approx(-1.0)
    assert summary.persistence_time_min is None


def test_steps_across_missed_frames_are_left_out():
    obs = track([(0, 0), (1, 0), (5, 0), (6, 0)], frames=[0, 1, 4, 5]).observations
    # Only steps 0->1 and 4->5 are one frame long; they are 4 frames apart.
    assert directional_autocorrelation(obs, UNIT) == {4: (pytest.approx(1.0), 1)}


def test_persistence_time_of_a_heading_diffusion_walk():
    """Heading increments N(0, sigma^2) give C(lag) = exp(-sigma^2 lag / 2).

    True persistence 2 / sigma^2 = 22.2 frames.  Measured on this seed: lag-1
    correlation 0.9562 against 0.9560, median fitted P 21.4.
    """
    sigma = 0.3
    truth = 2.0 / sigma**2
    rng = np.random.default_rng(7)
    tracks = []
    for t in range(30):
        theta = np.cumsum(rng.normal(scale=sigma, size=400))
        steps = np.column_stack([np.cos(theta), np.sin(theta)])
        tracks.append(track(np.vstack([[0, 0], np.cumsum(steps, axis=0)]), tid=t + 1))
    summaries = summarise(tracks, Scale.from_values(1.0, 10.0))

    lag1 = np.mean([s.directional_autocorrelation for s in summaries])
    assert lag1 == pytest.approx(math.exp(-(sigma**2) / 2.0), abs=0.005)
    persistence_frames = [s.persistence_time_min / 10.0 for s in summaries]
    assert np.median(persistence_frames) == pytest.approx(truth, rel=0.25)
    # Reported only with its fit quality.
    assert all(s.persistence_fit_r2 is not None and s.persistence_fit_lags for s in summaries)
    assert all(
        s.persistence_time_hr == pytest.approx(s.persistence_time_min / 60.0) for s in summaries
    )
