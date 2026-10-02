"""The self-test's own checks, without Cellpose.

``corridor --self-test`` is what an installed copy is judged by, so its checks
must fail when the thing they check is broken -- a check that passes on a
broken installation certifies nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corridor import selftest
from corridor.core import measurements, model_registry
from corridor.core.model_registry import MODEL_UNAVAILABLE_MESSAGE, ModelUnavailable, ResolvedModel


def test_the_tracking_check_passes_and_names_both_units():
    detail = selftest._check_tracking()
    assert "um/min" in detail and "um/hr" in detail and "MSD" in detail


def test_the_tracking_check_has_no_axis():
    import inspect

    source = inspect.getsource(selftest)
    for gone in ("ConfinementAxis", "resolve_axis", "sigma_along", "net_along"):
        assert gone not in source, gone


def test_the_tracking_check_catches_a_wrong_speed(monkeypatch):
    real = measurements.frame_rows

    def doubled(*args, **kwargs):
        rows = real(*args, **kwargs)
        for row in rows:
            if row.get("speed_um_per_hr") is not None:
                row["speed_um_per_hr"] *= 2
        return rows

    monkeypatch.setattr(measurements, "frame_rows", doubled)
    with pytest.raises(RuntimeError, match="um/hr"):
        selftest._check_tracking()


def test_the_tracking_check_catches_a_wrong_msd(monkeypatch):
    real = measurements.msd_rows

    def in_um(*args, **kwargs):
        # The classic unit error: an MSD reported in um instead of um^2.
        rows = real(*args, **kwargs)
        for row in rows:
            row["msd_um2"] = row["msd_um2"] ** 0.5
        return rows

    monkeypatch.setattr(measurements, "msd_rows", in_um)
    with pytest.raises(RuntimeError, match="MSD"):
        selftest._check_tracking()


def test_the_outputs_round_trip_including_msd_and_xlsx():
    detail = selftest._check_outputs()
    assert "track_msd" in detail and "XLSX" in detail and "schema 2" in detail


def test_the_model_check_goes_through_the_registry(monkeypatch):
    def unavailable(dimensionality="2D"):
        raise ModelUnavailable(MODEL_UNAVAILABLE_MESSAGE, [(Path("C:/absent"), "missing")])

    monkeypatch.setattr(model_registry, "resolve_model", unavailable)
    report = selftest.Report()
    assert not report.add("validated model", selftest._check_model)
    assert MODEL_UNAVAILABLE_MESSAGE in report.checks[0].error


def test_a_developer_override_fails_the_model_check(monkeypatch):
    spec = model_registry.production_spec("2D")
    override = ResolvedModel(spec=spec, path=Path("other.pt"), sha256="0" * 64,
                             developer_override=True)
    monkeypatch.setattr(model_registry, "resolve_model", lambda dimensionality="2D": override)
    with pytest.raises(RuntimeError, match="validated model"):
        selftest._check_model()


def test_run_reports_a_hard_failure_and_tolerates_an_optional_one(capsys):
    def ok() -> str:
        return "fine"

    def broken() -> str:
        raise RuntimeError("nope")

    assert selftest.run(verbose=False, checks=[("a", ok), ("napari (optional)", broken)]) == 0
    assert selftest.run(verbose=True, checks=[("a", ok), ("b", broken)]) == 1
    assert "1 check(s) failed" in capsys.readouterr().out
