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
    # Track 2 skips frame 3. What the plot draws is what the UI derived from
    # the rows: lags in hours, one per actual frame difference 1..5 at the
    # run's 20 min interval -- not five index lags, and not left in frames.
    assert screen.msd_plot.lag_unit == "h"
    assert screen.msd_plot.lags == pytest.approx(
        [lag * syn.INTERVAL_MIN / 60.0 for lag in (1, 2, 3, 4, 5)]
    )
    assert detail(screen, "gaps") == "1 in 1 gap(s)"


def test_v1_analysis_reports_per_hour_speeds_and_no_invented_msd(qt_app, v1):
    screen = _screen(qt_app, v1)
    screen.select_track(1)
    summary = v1.summary_for(1)
    assert detail(screen, "mean") == f"{summary['mean_speed_um_per_min'] * 60:.1f} µm/h"
    assert detail(screen, "len") == f"{syn.expected_len_um(1):.1f} µm"
    # The 2.0 store derives a 1.x run's MSD from its positions when it loads
    # it. An analysis with no MSD at all -- nothing derived, nothing fitted --
    # must read as an em dash, not "not enough lags", which would claim a fit
    # was attempted.
    v1.msd = []
    v1.msd_for_track = lambda tid: []
    for row in v1.summaries:
        for key in [k for k in row if k.startswith("msd")]:
            row.pop(key)
    screen.select_track(2)
    assert detail(screen, "alpha") == "—"
    assert screen.msd_plot.is_empty
    # ...and the plot below it must not claim a fit was attempted either.
    assert screen.msd_plot.empty_message == "no MSD in this analysis"


def test_an_msd_curve_too_short_to_plot_says_not_enough_lags(qt_app, v2):
    # The analysis has MSD curves, this track's is just empty.
    v2.msd = [r for r in v2.msd if int(r["track_id"]) != 2]
    # The 2.0 store's accessor, if it has one, must see the same rows.
    v2.msd_for_track = lambda tid: [r for r in v2.msd if int(r["track_id"]) == int(tid)]
    screen = _screen(qt_app, v2)
    screen.select_track(2)
    assert screen.msd_plot.is_empty
    assert screen.msd_plot.empty_message == "not enough lags"


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
    # The format ChannelGeometry.to_dict writes: origin plus unit direction.
    lane0 = v2.manifest["channel_geometry"]["lanes"][0]
    assert {"origin_x", "origin_y", "direction_x", "direction_y"} <= set(lane0)
    lanes = lanes_for_manifest(v2.manifest, (64, 48))
    assert [lane.index for lane in lanes] == [0, 1]
    for lane in lanes:
        (s0, s1), (e0, e1) = syn.LANE_ENDS[lane.index]
        assert lane.centre[0] == pytest.approx((s0, s1), abs=1e-9)
        assert lane.centre[1] == pytest.approx((e0, e1), abs=1e-9)
        assert lane.half_width_px == 8.0
    # Other spellings of the same lanes read to the same lines, exactly.
    spelled = {"channel_geometry": {"lanes": [
        {"index": 0, "centre_line": [[20.0, 0.0], [20.0, 48.0]], "half_width_px": 8.0},
        {"index": 1, "x0": 44.0, "y0": 0.0, "x1": 50.0, "y1": 48.0, "half_width_px": 8.0},
    ]}}
    other = lanes_for_manifest(spelled, (64, 48))
    assert [lane.index for lane in other] == [0, 1]
    assert other[0].centre == ((20.0, 0.0), (20.0, 48.0))
    assert other[1].centre == ((44.0, 0.0), (50.0, 48.0))
    # A lane in no readable spelling is skipped, not drawn down the left edge.
    assert lanes_for_manifest({"channel_geometry": {"lanes": [{"index": 0}]}}, (64, 48)) == []
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


def test_a_single_time_point_volume_is_shown_in_three_d(qt_app, tmp_path):
    """A ZYX stack (ndim 3) of a 3-D run must not play its slices as frames."""
    analysis = syn.load_saved(syn.write_v2_analysis(tmp_path / "zyx", three_d=True))
    volume = syn.stack_3d()[0]  # Z, Y, X
    screen = _screen(qt_app, analysis, volume)
    assert screen.three_d
    assert screen.view_stack.currentWidget() is screen.ortho
    assert screen.ortho.n_frames == 1 and screen.ortho.n_slices == volume.shape[0]


# --------------------------------------------------------------------------
# Len / D2S in 3-D: never an XY-only number labelled µm
# --------------------------------------------------------------------------


def _bare(rows):
    """Track rows without the store's µm path columns (the 3-D, no-Z-step case)."""
    drop = {"cumulative_path_um", "distance_from_start_um"}
    return [{k: v for k, v in r.items() if k not in drop} for r in rows]


def test_a_cell_moving_only_in_z_is_not_reported_as_still():
    rows = [
        {"frame": 0, "x_px": 10.0, "y_px": 10.0, "z": 0.0},
        {"frame": 1, "x_px": 10.0, "y_px": 10.0, "z": 5.0},
    ]
    # Z step unknown: no µm path can be measured, so nothing is shown.
    metrics = analysis_view.path_metrics(rows, {}, 0.5)
    assert (metrics.length, metrics.from_start) == (None, None)
    # Uncalibrated: slices and pixels have no common unit either.
    metrics = analysis_view.path_metrics(rows, {}, None, 2.0)
    assert (metrics.length, metrics.from_start) == (None, None)
    # With the real Z step, the 5 slices x 2.0 µm are the whole path.
    metrics = analysis_view.path_metrics(rows, {}, 0.5, 2.0)
    assert metrics.unit == "µm"
    assert metrics.length == pytest.approx(10.0)
    assert metrics.from_start == pytest.approx(10.0)


def test_three_d_path_uses_x_y_and_z_in_microns():
    rows = [
        {"frame": 0, "x_px": 0.0, "y_px": 0.0, "z_px": 0.0},
        {"frame": 1, "x_px": 6.0, "y_px": 0.0, "z_px": 0.0},  # 3 µm in x
        {"frame": 2, "x_px": 6.0, "y_px": 0.0, "z_px": 2.0},  # 4 µm in z
    ]
    metrics = analysis_view.path_metrics(rows, {}, 0.5, 2.0)
    assert metrics.length == pytest.approx(7.0)
    assert metrics.from_start == pytest.approx(5.0)  # the 3-4-5 triangle


def test_three_d_detail_shows_a_dash_when_the_z_step_is_unknown(qt_app, tmp_path):
    analysis = syn.load_saved(syn.write_v2_analysis(tmp_path / "v3d", three_d=True))
    analysis.tracks = _bare(analysis.tracks)
    for summary in analysis.summaries:
        summary.pop("path_length_um", None)
        summary.pop("net_displacement_um", None)
    analysis.manifest["calibration"].pop("z_step_um")
    screen = _screen(qt_app, analysis, syn.stack_3d())
    screen.select_track(1)
    assert detail(screen, "len") == "—"
    assert detail(screen, "d2s") == "—"

    # With the Z step back, the fallback measures in 3-D: track 1 alternates
    # z 2, 3, 2, ... so each step adds 1.5 µm of Z to its 3 µm in y.
    analysis.manifest["calibration"]["z_step_um"] = 1.5
    screen.select_track(1)
    step = (3.0**2 + 1.5**2) ** 0.5
    assert detail(screen, "len") == f"{5 * step:.1f} µm"


# --------------------------------------------------------------------------
# D2R reaches every export that carries tracks
# --------------------------------------------------------------------------


def _tracks_writer(analysis_rows):
    """A stand-in for the agreed ``export_tracks_csv(saved, path)``: no D2R."""
    import csv

    def write(saved, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = ["track_id", "frame", "x_px", "y_px", "speed_um_per_hr"]
        with open(path, "w", encoding="utf-8", newline="") as fh:
            out = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
            out.writeheader()
            for row in analysis_rows:
                out.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
        return path

    return write


def _read(path):
    import csv

    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_all_tracks_csv_gets_d2r_from_the_reference_point(v2, tmp_path, monkeypatch):
    from corridor.store import project

    monkeypatch.setattr(project, "export_tracks_csv", _tracks_writer(v2.tracks), raising=False)
    destination = tmp_path / "all.csv"
    written = workers.run_export(
        v2, destination, kind=workers.EXPORT_TRACKS_CSV, reference_point_px=(20.0, 5.0)
    )
    assert written == [destination]
    rows = _read(destination)
    assert "distance_from_reference_um" in rows[0]
    for row in rows:
        expected = (
            (float(row["x_px"]) - 20.0) ** 2 + (float(row["y_px"]) - 5.0) ** 2
        ) ** 0.5 * syn.PIXEL_UM
        assert float(row["distance_from_reference_um"]) == pytest.approx(expected)
    # Every other cell is the writer's own text, untouched.
    before = _read(_tracks_writer(v2.tracks)(v2, tmp_path / "plain.csv"))
    for a, b in zip(before, rows):
        assert {k: a[k] for k in a} == {k: b[k] for k in a}
    # Track 1 starts at the reference point: D2R 0 there.
    assert float(rows[0]["distance_from_reference_um"]) == 0.0


def test_no_reference_point_leaves_the_tracks_csv_alone(v2, tmp_path, monkeypatch):
    from corridor.store import project

    monkeypatch.setattr(project, "export_tracks_csv", _tracks_writer(v2.tracks), raising=False)
    destination = tmp_path / "all.csv"
    workers.run_export(v2, destination, kind=workers.EXPORT_TRACKS_CSV)
    assert "distance_from_reference_um" not in _read(destination)[0]


def test_the_bundle_carries_d2r_and_the_reference_point(v2, tmp_path, monkeypatch):
    from corridor.store import project

    def export_bundle(saved, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        tracks = _tracks_writer(saved.tracks)(saved, destination / "tracks.csv")
        return [tracks]

    monkeypatch.setattr(project, "export_bundle", export_bundle)
    destination = tmp_path / "bundle"
    written = workers.run_export(
        v2, destination, kind=workers.EXPORT_BUNDLE, reference_point_px=(44.0, 40.0)
    )
    assert destination / "reference_point.json" in written
    assert analysis_view.load_reference_point(destination) == (44.0, 40.0)
    rows = _read(destination / "tracks.csv")
    first_of_track_2 = [r for r in rows if r["track_id"] == "2"][0]
    assert float(first_of_track_2["distance_from_reference_um"]) == 0.0
    assert all(r["distance_from_reference_um"] != "" for r in rows)


def test_a_writer_that_takes_the_reference_point_is_given_it(v2, tmp_path, monkeypatch):
    from corridor.store import project

    seen = []

    def export_tracks_csv(saved, path, *, reference_point_px=None):
        seen.append(reference_point_px)
        Path(path).write_text("track_id,x_px,y_px\n1,0,0\n", encoding="utf-8")
        return Path(path)

    monkeypatch.setattr(project, "export_tracks_csv", export_tracks_csv, raising=False)
    destination = tmp_path / "all.csv"
    workers.run_export(
        v2, destination, kind=workers.EXPORT_TRACKS_CSV, reference_point_px=(1.0, 2.0)
    )
    assert seen == [(1.0, 2.0)]
    # The writer owns D2R then; its file is not rewritten.
    assert "distance_from_reference_um" not in _read(destination)[0]


def test_three_d_d2r_needs_the_points_z_and_the_z_step():
    row = {"x_px": 6.0, "y_px": 0.0, "z": 2.0}
    assert analysis_view.reference_distance_um(row, (0.0, 0.0, 0.0), 0.5, 2.0) == pytest.approx(5.0)
    assert analysis_view.reference_distance_um(row, (0.0, 0.0, 0.0), 0.5, None) is None
    assert analysis_view.reference_distance_um(row, (0.0, 0.0), 0.5, 2.0) is None
    assert analysis_view.reference_distance_um(row, (0.0, 0.0, 0.0), None, 2.0) is None
    assert analysis_view.reference_distance_um(
        {"x_px": 6.0, "y_px": 8.0}, (0.0, 0.0), 0.5
    ) == pytest.approx(5.0)


# --------------------------------------------------------------------------
# Reference point: shown in 3-D, and a failed save is said out loud
# --------------------------------------------------------------------------


def test_the_reference_point_is_drawn_in_the_orthogonal_panes(qt_app, tmp_path):
    analysis = syn.load_saved(syn.write_v2_analysis(tmp_path / "v3d", three_d=True))
    screen = _screen(qt_app, analysis, syn.stack_3d())
    ortho = screen.ortho
    cx, cy, cz = ortho.cursor
    screen._reference_picked(cx, cy, float(cz))
    assert ortho.xy.reference == (cx, cy, True)
    assert ortho.xz.reference == (cx, float(cz), True)
    assert ortho.yz.reference == (float(cz), cy, True)
    ortho.xy.grab()  # paints the marker without error

    # Another slice: still findable on XY (faint), gone from planes far from it.
    ortho.set_z(0 if cz else 1)
    ortho.set_cursor(cx + 20.0, cy + 20.0)
    assert ortho.xy.reference is not None and ortho.xy.reference[2] is False
    assert ortho.xz.reference is None and ortho.yz.reference is None

    screen.clear_reference_point()
    assert ortho.xy.reference is None


def test_a_reference_point_that_cannot_be_saved_says_so(qt_app, v2, monkeypatch):
    from corridor.ui.screens import results

    def refuse(directory, point):
        raise PermissionError(13, "Access is denied", str(directory))

    monkeypatch.setattr(results, "save_reference_point", refuse)
    screen = _screen(qt_app, v2)
    screen._reference_picked(30.0, 12.0)
    assert screen.reference_point_px == (30.0, 12.0), "kept for this session's exports"
    assert screen.reference_warning.isVisibleTo(screen)
    assert "could not be saved" in screen.reference_warning.toolTip()

    monkeypatch.setattr(results, "save_reference_point", lambda d, p: Path(d))
    screen._reference_picked(31.0, 12.0)
    assert not screen.reference_warning.isVisibleTo(screen)
