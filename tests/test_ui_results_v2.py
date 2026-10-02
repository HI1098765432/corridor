"""The 2.0 results screen on synthetic v2 and v1 analyses.

What is pinned here, and why each matters:

*   **The detail panel reports what the files hold**, in the units they hold
    it: Len and D2S from the per-observation columns, speeds per hour, the
    MSD exponent or "not enough lags". A v1 run, which has none of the v2
    columns, still shows per-hour speeds (derived x 60, never recomputed).
*   **No provenance row ever says "None"** (critique C5), for a v2 run, a v1
    run, and a manifest with almost nothing in it.
*   **Lanes are drawn only when the run recorded them.** A run with no
    geometry must not get a fabricated vertical axis, which is what 1.x did.
*   **Exports go through the store's writers** with the selected track and
    the reference point; the writers are replaced here, because what they
    write is the store's contract, tested there.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6", reason="the interface is not installed")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import ui_synthetic as syn  # noqa: E402
from corridor.ui import analysis_view, workers  # noqa: E402
from corridor.ui.analysis_view import MsdPoint, MsdSeries  # noqa: E402
from corridor.ui.lanes import lanes_for_manifest  # noqa: E402
from corridor.ui.screens.results import ResultsScreen  # noqa: E402
from corridor.ui.widgets.msd_plot import MsdPlot  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def v2(tmp_path):
    return syn.load_saved(syn.write_v2_analysis(tmp_path / "v2"))


@pytest.fixture
def v1(tmp_path):
    return syn.load_saved(syn.write_v1_analysis(tmp_path / "v1"))


def _screen(qt_app, analysis, stack=None) -> ResultsScreen:
    screen = ResultsScreen()
    screen.resize(1300, 860)
    screen.load(analysis, syn.stack_2d() if stack is None else stack)
    qt_app.processEvents()
    return screen


def detail(screen, key: str) -> str:
    return screen.detail_fields[key]._value.full_text()


def run_rows(screen) -> dict[str, tuple[str, str]]:
    return {
        key: (field._value.full_text(), field._value.toolTip() + field.toolTip())
        for key, field in screen.run_fields.items()
    }


# --------------------------------------------------------------------------
# Detail panel
# --------------------------------------------------------------------------


def test_v2_track_detail_shows_len_d2s_per_hour_and_alpha(qt_app, v2):
    screen = _screen(qt_app, v2)
    screen.select_track(1)

    assert detail(screen, "len") == f"{syn.expected_len_um(1):.1f} µm"
    assert detail(screen, "d2s") == f"{syn.expected_d2s_um(1):.1f} µm"
    summary = v2.summary_for(1)
    assert detail(screen, "mean") == f"{summary['mean_speed_um_per_hr']:.1f} µm/h"
    assert detail(screen, "alpha") == "2.00  (r² 1.00)"
    assert screen.export_track_button.text() == "Export track 1"
    # Every MSD lag of track 1 has a positive MSD, so every one is plotted.
    assert screen.msd_plot.n_points == len(analysis_view.msd_rows_for(v2, 1)) == 5


def test_a_track_whose_alpha_was_not_fitted_says_so(qt_app, v2):
    screen = _screen(qt_app, v2)
    screen.select_track(2)
    assert detail(screen, "alpha") == "not enough lags"
    # Track 2 skips frame 3: its lags are the actual frame differences 1..5.
    lags = sorted(int(r["lag_frames"]) for r in analysis_view.msd_rows_for(v2, 2))
    assert lags == [1, 2, 3, 4, 5]
    assert detail(screen, "gaps") == "1 in 1 gap(s)"


def test_v1_analysis_reports_per_hour_speeds_and_no_invented_msd(qt_app, v1):
    screen = _screen(qt_app, v1)
    screen.select_track(1)
    summary = v1.summary_for(1)
    assert detail(screen, "mean") == f"{summary['mean_speed_um_per_min'] * 60:.1f} µm/h"
    assert detail(screen, "len") == f"{syn.expected_len_um(1):.1f} µm"
    # A 1.x run has no MSD at all (until the store upgrades it): an em dash,
    # not "not enough lags", which would claim a fit was attempted.
    v1.msd = []
    screen.select_track(2)
    assert detail(screen, "alpha") == "—"
    assert screen.msd_plot.is_empty


def test_the_track_list_reads_per_hour(qt_app, v2):
    screen = _screen(qt_app, v2)
    texts = [screen.track_list.item(i).text() for i in range(screen.track_list.count())]
    assert all("µm/h" in t and "µm/min" not in t for t in texts)


# --------------------------------------------------------------------------
# Provenance: never "None"
# --------------------------------------------------------------------------


def _assert_no_none(rows):
    for key, (text, tooltip) in rows.items():
        assert "None" not in text, f"{key} shows {text!r}"
        assert "None" not in tooltip, f"{key} tooltip {tooltip!r}"


def test_no_provenance_row_says_none_for_v2(qt_app, v2):
    screen = _screen(qt_app, v2)
    rows = run_rows(screen)
    _assert_no_none(rows)
    assert rows["schema"][0] == "2"
    assert rows["dimensionality"][0] == "2D"
    assert rows["model"][0] == "jhu_confined_cp3_combi"
    assert rows["model_version"][0] == "1.0.0"
    assert rows["model_hash"][0] == f"{syn.MODEL_SHA[:12]}…"
    assert rows["geometry"][0].startswith("yes")
    assert "2 lane(s)" in rows["geometry"][0]


def test_no_provenance_row_says_none_for_v1(qt_app, v1):
    screen = _screen(qt_app, v1)
    rows = run_rows(screen)
    _assert_no_none(rows)
    assert rows["schema"][0] == "1"
    assert rows["model"][0] == "cyto2_phase_microfluidic_KK1KK2_combi"
    assert rows["geometry"][0] == "—"
    assert "1.x" in rows["geometry"][1]


@pytest.mark.parametrize(
    "manifest",
    [
        {},
        {"input": {}, "segmentation": {}, "tracking": {}, "calibration": {}},
        {"tracking": {"max_gap": 3}, "segmentation": {"cellprob_threshold": 0.0}},
        {"segmentation": {"normalisation_mode": "local"}, "channel_geometry": {"lanes": []}},
    ],
)
def test_sparse_manifests_show_dashes_not_none(qt_app, manifest):
    """The C5 leaks: 'None px', 'prob None · flow None', 'None frames (gap ≤ None)'."""
    screen = ResultsScreen()
    screen.analysis = SimpleNamespace(manifest=manifest)
    screen._populate_run()
    rows = run_rows(screen)
    _assert_no_none(rows)
    if manifest.get("tracking") == {"max_gap": 3}:
        assert rows["max_gap"][0] == "3 frames"
        assert rows["thresholds"][0] == "prob 0"
        assert rows["min_extent"][0] == "—"


# --------------------------------------------------------------------------
# Lanes
# --------------------------------------------------------------------------


def test_v2_lanes_are_read_in_every_spelling(qt_app, v2):
    lanes = lanes_for_manifest(v2.manifest, (64, 48))
    assert [lane.index for lane in lanes] == [0, 1]
    assert lanes[0].centre == ((20.0, 0.0), (20.0, 48.0))
    assert lanes[1].centre == ((44.0, 0.0), (50.0, 48.0))
    screen = _screen(qt_app, v2)
    assert len(screen.canvas.lanes) == 2
    assert screen.toggle_lanes.isEnabled()
    screen.toggle_lanes.setChecked(True)
    assert screen.canvas.layers.lanes
    screen.canvas.grab()  # paints the lanes without error


def test_v1_channel_lines_become_lanes_clipped_to_the_image(qt_app, v1):
    lanes = lanes_for_manifest(v1.manifest, (64, 48))
    assert len(lanes) == 1
    (x0, y0), (x1, y1) = lanes[0].centre
    assert (x0, x1) == (20.0, 20.0) and {y0, y1} == {0.0, 48.0}
    assert lanes[0].half_width_px == 10.0


def test_no_recorded_geometry_draws_nothing(qt_app, v2):
    """1.x defaulted a missing block to a vertical axis; 2.0 draws nothing."""
    assert lanes_for_manifest({}, (64, 48)) == []
    assert lanes_for_manifest({"confinement": {}}, (64, 48)) == []
    assert lanes_for_manifest({"confinement": {"channels": [{"origin_x": 3}]}}, (64, 48)) == []
    v2.manifest.pop("channel_geometry")
    screen = _screen(qt_app, v2)
    assert screen.canvas.lanes == []
    assert not screen.toggle_lanes.isEnabled()
    assert not hasattr(screen.canvas, "axis_vector")


# --------------------------------------------------------------------------
# MSD widget
# --------------------------------------------------------------------------


def test_msd_plot_empty_and_normal(qt_app):
    plot = MsdPlot()
    plot.resize(320, 150)
    plot.set_series(MsdSeries((), "h", "µm²"), "#FF6B35")
    assert plot.is_empty
    assert not plot.grab().isNull()

    points = tuple(
        MsdPoint(lag=lag / 3.0, msd=2.0 * lag**1.5, n_pairs=n)
        for lag, n in ((1, 9), (2, 7), (3, 4), (4, 1))
    )
    plot.set_series(MsdSeries(points, "h", "µm²"), "#FF6B35")
    assert plot.n_points == 4 and not plot.is_empty
    radii = plot.point_radii()
    assert radii == sorted(radii, reverse=True), "more pairs must draw a larger point"
    assert not plot.grab().isNull()


def test_msd_series_units_never_mix():
    rows = [
        {"lag_frames": 1, "lag_time_hr": 1 / 3, "n_pairs": 4, "msd_um2": 2.0, "msd_px2": 8.0},
        {"lag_frames": 2, "lag_time_hr": 2 / 3, "n_pairs": 3, "msd_um2": None, "msd_px2": 30.0},
    ]
    series = analysis_view.msd_series(rows)
    # One row lacks µm²: the whole curve falls back to px², not a mixture.
    assert series.msd_unit == "px²"
    assert [p.msd for p in series.points] == [8.0, 30.0]
    # Lags with no time are shown in frames, with no invented interval.
    series = analysis_view.msd_series([{"lag_frames": 1, "n_pairs": 2, "msd_um2": 1.0}])
    assert series.lag_unit == "frames"
    series = analysis_view.msd_series(
        [{"lag_frames": 1, "n_pairs": 2, "msd_um2": 1.0}], frame_interval_min=30.0
    )
    assert series.lag_unit == "h" and series.points[0].lag == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Reference point
# --------------------------------------------------------------------------


def test_reference_point_is_set_by_clicking_the_image_and_persists(qt_app, v2):
    screen = _screen(qt_app, v2)
    canvas = screen.canvas
    canvas.resize(640, 480)
    canvas.fit_to_view()
    seen = []
    screen.reference_point_changed.connect(seen.append)

    screen.set_reference_mode(True)
    assert canvas.picking
    target = canvas._to_overlay(30.0, 12.0)
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(round(target.x()), round(target.y())))

    # Within one screen pixel of the clicked pixel centre, in data coordinates.
    tolerance = 1.0 / canvas._scale
    x, y = screen.reference_point_px
    assert x == pytest.approx(30.0, abs=tolerance) and y == pytest.approx(12.0, abs=tolerance)
    assert not canvas.picking, "one click sets the point and leaves picking mode"
    assert screen.selected_track is None, "setting a point must not select a track"
    assert seen and seen[-1] == screen.reference_point_px
    assert analysis_view.load_reference_point(v2.directory) == screen.reference_point_px

    # Stored per analysis: reopening restores it.
    reopened = _screen(qt_app, syn.load_saved(v2.directory))
    assert reopened.reference_point_px == screen.reference_point_px
    assert reopened.clear_reference_button.isVisibleTo(reopened)

    reopened.clear_reference_point()
    assert reopened.reference_point_px is None
    assert analysis_view.load_reference_point(v2.directory) is None


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------


@pytest.fixture
def writers(monkeypatch):
    """Replace every store writer with a recorder that writes a stub file."""
    from corridor.store import project

    calls: list[tuple[str, tuple, dict]] = []

    def recorder(name):
        def write(*args, **kwargs):
            calls.append((name, args, kwargs))
            target = Path(args[2] if name == "export_track" else args[1])
            if name == "export_bundle":
                target.mkdir(parents=True, exist_ok=True)
                out = target / "tracks.csv"
                out.write_text("track_id\n", encoding="utf-8")
                return [out]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("track_id\n", encoding="utf-8")
            return target

        return write

    for name in (
        "export_track", "export_tracks_csv", "export_summaries_csv",
        "export_msd_csv", "export_bundle",
    ):
        monkeypatch.setattr(project, name, recorder(name), raising=False)
    return calls


def test_every_export_kind_reaches_its_writer(v2, tmp_path, writers):
    reference = (30.0, 12.0)
    for kind in workers.EXPORT_LABELS:
        destination = tmp_path / "out" / f"{kind}.out"
        written = workers.run_export(
            v2, destination, kind=kind, track_id=1, reference_point_px=reference
        )
        assert written and all(Path(p).exists() for p in written)
    by_name = {name: (args, kwargs) for name, args, kwargs in writers}
    args, kwargs = by_name["export_track"]
    assert args[1] == 1 and kwargs["reference_point_px"] == reference
    assert kwargs["fmt"] == "xlsx"  # the XLSX export ran after the CSV one
    assert [c for c in writers if c[0] == "export_track"][0][2]["fmt"] == "csv"
    assert by_name["export_bundle"][0][0] is v2
    assert set(by_name) == {
        "export_track", "export_tracks_csv", "export_summaries_csv",
        "export_msd_csv", "export_bundle",
    }


def test_a_track_export_without_a_track_is_refused(v2, tmp_path, writers):
    with pytest.raises(ValueError):
        workers.run_export(v2, tmp_path / "x.csv", kind=workers.EXPORT_TRACK_CSV)
    assert writers == []


def test_export_menu_offers_every_export_and_tracks_need_a_selection(qt_app, v2):
    screen = _screen(qt_app, v2)
    labels = [a.text() for a in screen.export_menu.actions() if not a.isSeparator()]
    assert labels == [
        "Selected track CSV", "Selected track XLSX", "All tracks CSV",
        "All track summaries CSV", "MSD curves CSV", "Full analysis bundle",
    ]
    assert not screen.export_actions[workers.EXPORT_TRACK_CSV].isEnabled()
    assert screen.export_actions[workers.EXPORT_MSD_CSV].isEnabled()

    requested = []
    screen.export_requested.connect(requested.append)
    screen.select_track(2)
    assert screen.export_actions[workers.EXPORT_TRACK_XLSX].isEnabled()
    assert screen.export_actions[workers.EXPORT_TRACK_XLSX].text() == "Track 2 XLSX"
    screen.export_actions[workers.EXPORT_TRACK_XLSX].trigger()
    screen.export_track_button.click()
    screen.export_actions[workers.EXPORT_BUNDLE].trigger()
    assert requested == [workers.EXPORT_TRACK_XLSX, workers.EXPORT_TRACK_CSV, workers.EXPORT_BUNDLE]


def test_switching_analyses_clears_the_previous_selection(qt_app, v2, v1):
    screen = _screen(qt_app, v2)
    screen.select_track(1)
    screen.load(v1, syn.stack_2d())
    assert screen.selected_track is None
    assert not screen.export_actions[workers.EXPORT_TRACK_CSV].isEnabled()
    assert all(f._value.full_text() == "—" for f in screen.detail_fields.values())
    assert screen.msd_plot.is_empty


def test_three_d_analysis_shows_the_orthogonal_viewer(qt_app, tmp_path):
    analysis = syn.load_saved(syn.write_v2_analysis(tmp_path / "v3d", three_d=True))
    screen = _screen(qt_app, analysis, syn.stack_3d())
    assert screen.three_d
    assert screen.view_stack.currentWidget() is screen.ortho
    assert screen.ortho.anisotropy == pytest.approx(1.5 / syn.PIXEL_UM)
    screen.timeline.step(2)
    assert screen.ortho.frame == 2
    screen.select_track(1)
    x, y, z = screen.ortho.cursor
    row = [r for r in analysis.rows_for_track(1) if r["frame"] == 2][0]
    assert (x, y, z) == (row["x_px"], row["y_px"], int(row["z"]))
    _assert_no_none(run_rows(screen))
    assert run_rows(screen)["dimensionality"][0] == "3D"
    assert run_rows(screen)["z_step"][0] == "1.5 µm"
