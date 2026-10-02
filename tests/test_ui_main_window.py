"""The window's 2.0 rules, through a real MainWindow on a throwaway store.

*   A new project never inherits calibration, model keys, an axis order or a
    reference point from the last run; a reopened project keeps its own.
*   With no validated model, Analyse shows the contract's sentence and starts
    nothing -- there is no fallback model to start instead.
*   A file whose T and Z cannot be told apart is asked about once, and read
    again with the chosen order.
*   Exports run the store's writers with the selected track and the reference
    point, off the UI thread, and report without a blocking dialog.

Modal dialogs are replaced by recorders: an ``exec()`` would hang an
offscreen run.
"""

from __future__ import annotations

import os
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6", reason="the interface is not installed")

from PySide6.QtWidgets import QApplication  # noqa: E402

import ui_synthetic as syn  # noqa: E402
from corridor.core import imaging, model_registry, updates  # noqa: E402
from corridor.core.config import RunConfig  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def window(qt_app, tmp_path, monkeypatch):
    from corridor.store import db
    from corridor.ui import main_window
    from corridor.ui.main_window import MainWindow

    monkeypatch.setenv("CORRIDOR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(updates, "check", lambda *a, **k: None)
    monkeypatch.setattr(main_window, "_gpu_available", lambda: False)
    store = db.Store(tmp_path / "projects.db")
    win = MainWindow(store=store)
    errors: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []
    # Bound to the window, as _start_job requires of every worker slot: a bare
    # lambda has no thread affinity and would run on the worker thread.
    monkeypatch.setattr(
        win, "_show_error", types.MethodType(lambda self, m, d="": errors.append((m, d)), win)
    )
    monkeypatch.setattr(
        win, "_warn", types.MethodType(lambda self, m, d="": warnings.append((m, d)), win)
    )
    yield SimpleNamespace(win=win, store=store, errors=errors, warnings=warnings)
    win.close()


def pump_until(app, predicate, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


# --------------------------------------------------------------------------
# New-project configuration
# --------------------------------------------------------------------------


def test_a_new_project_inherits_tuning_but_no_facts_about_another_file(window, tmp_path):
    saved = RunConfig()
    saved.calibration.pixel_size_um = 0.639
    saved.calibration.frame_interval_min = 10.0
    saved.segmentation.model_path = "C:/elsewhere/cyto3"
    saved.segmentation.use_custom_model = False
    saved.import_.axes = "ZYX"
    saved.measurement.reference_point_px = (3.0, 4.0)
    saved.tracking.max_gap = 7
    window.store.set_setting("default_config", saved.to_dict())

    source = tmp_path / "new.tif"
    source.write_bytes(b"")
    project = window.store.create_project(source)
    config = window.win._fresh_config(source, project)

    assert config.calibration.pixel_size_um is None
    assert config.calibration.frame_interval_min is None
    assert config.segmentation.model_path is None
    assert config.segmentation.use_custom_model is True
    assert config.import_.axes is None
    assert config.measurement.reference_point_px is None
    assert config.tracking.max_gap == 7, "tuning carries over"
    assert config.input_path == str(source)


def test_a_reopened_project_keeps_its_own_configuration(window, tmp_path):
    source = tmp_path / "old.tif"
    source.write_bytes(b"")
    project = window.store.create_project(source)
    own = RunConfig()
    own.calibration.pixel_size_um = 0.467
    project.config = own.to_dict()
    config = window.win._fresh_config(source, project)
    assert config.calibration.pixel_size_um == 0.467


# --------------------------------------------------------------------------
# Model check before analysis
# --------------------------------------------------------------------------


def test_a_missing_model_stops_the_analysis_with_the_contract_text(window, tmp_path, monkeypatch):
    def missing(dimensionality="2D"):
        raise model_registry.ModelUnavailable(
            model_registry.MODEL_UNAVAILABLE_MESSAGE, [(Path("C:/nowhere/model"), "missing")]
        )

    monkeypatch.setattr(model_registry, "resolve_model", missing)
    win = window.win
    source = tmp_path / "movie.tif"
    source.write_bytes(b"")
    win._project = window.store.create_project(source)
    win._metadata = SimpleNamespace(path=source, axes="TYX")
    win._stack = np.zeros((2, 8, 8), dtype=np.float32)

    win.start_analysis()

    assert win._analysis_job is None, "nothing may start without the validated model"
    assert len(window.errors) == 1
    message = window.errors[0][0]
    assert message.startswith(model_registry.MODEL_UNAVAILABLE_MESSAGE)
    assert "nowhere" in message
    assert "cyto3" not in message
    assert window.store.get_project(win._project.id).status != "running"


def test_imported_labels_need_no_model(window, tmp_path, monkeypatch):
    from corridor.core import segmentation

    def forbidden(dimensionality="2D"):
        raise AssertionError("the model must not be resolved for imported labels")

    monkeypatch.setattr(model_registry, "resolve_model", forbidden)
    # The 2.0 segmentation module (work package D) can read label images.
    monkeypatch.setattr(segmentation, "load_label_stack", lambda *a, **k: None, raising=False)
    win = window.win
    win._config.import_.labels_path = str(tmp_path / "labels.tif")
    win._metadata = SimpleNamespace(path=tmp_path / "x.tif", axes="TZYX")
    assert win._model_ready()


def test_labels_are_refused_by_a_pipeline_that_would_segment_anyway(window, tmp_path, monkeypatch):
    """The 1.x pipeline ignores labels_path and would run cyto3 instead."""
    from corridor.core import segmentation
    from corridor.ui import workers

    monkeypatch.delattr(segmentation, "load_label_stack", raising=False)
    monkeypatch.setattr(
        model_registry, "resolve_model", lambda d="2D": pytest.fail("no model for labels")
    )
    win = window.win
    win._config.import_.labels_path = str(tmp_path / "labels.tif")
    assert not win._model_ready()
    assert window.errors == [(workers.LABELS_UNSUPPORTED, "")]


def _verified(monkeypatch, tmp_path):
    spec = model_registry.production_spec("2D")
    resolved = model_registry.ResolvedModel(
        spec=spec, path=tmp_path / "models" / spec.filename, sha256=spec.sha256
    )
    monkeypatch.setattr(model_registry, "resolve_model", lambda dimensionality="2D": resolved)
    return resolved


def test_the_run_is_pinned_to_the_verified_model(window, tmp_path, monkeypatch):
    """No silent cyto3: a new project's config has no model_path, and a
    reopened 1.x one may name a file chosen with the old picker."""
    resolved = _verified(monkeypatch, tmp_path)
    win = window.win
    win._metadata = SimpleNamespace(path=tmp_path / "x.tif", axes="TYX")
    seg = win._config.segmentation
    seg.model_path = "C:/old-picker/some_other_model"  # a 1.x leftover
    seg.use_custom_model = False
    seg.ensemble_model_paths = ("C:/old-picker/sibling",)

    assert win._model_ready()
    assert seg.model_path == str(resolved.path)
    assert seg.use_custom_model is True
    assert seg.ensemble_model_paths == ()
    assert seg.resolved_model() == str(resolved.path), "the 1.x service loads this file"
    assert win._resolved_model is resolved


def test_the_worker_hands_the_verified_model_to_a_pipeline_that_takes_it(
    tmp_path, monkeypatch
):
    from corridor.core import pipeline
    from corridor.ui.workers import AnalysisWorker

    resolved = _verified(monkeypatch, tmp_path)
    calls = []

    def run_analysis(config, progress=None, *, save=True, keep_raw_masks=True, model=None):
        calls.append(model)
        return "result"

    monkeypatch.setattr(pipeline, "run_analysis", run_analysis)
    worker = AnalysisWorker(RunConfig(), model=resolved)
    results = []
    worker.finished.connect(results.append)
    worker.run()
    assert calls == [resolved] and results == ["result"]

    # A 1.x pipeline has no model= parameter: it is not passed (the config
    # carries the verified path instead).
    def legacy(config, progress=None, *, save=True, keep_raw_masks=True):
        calls.append("legacy")
        return "old"

    monkeypatch.setattr(pipeline, "run_analysis", legacy)
    AnalysisWorker(RunConfig(), model=resolved).run()
    assert calls[-1] == "legacy"


def test_start_analysis_passes_the_verified_model_to_the_worker(window, tmp_path, monkeypatch):
    from corridor.ui import main_window

    resolved = _verified(monkeypatch, tmp_path)
    created = []

    class Recorder(main_window.AnalysisWorker):
        def __init__(self, config, *, model=None):
            super().__init__(config, model=model)
            created.append((config, model))

        def run(self):  # nothing heavy in a test
            self.cancelled_signal.emit()

    monkeypatch.setattr(main_window, "AnalysisWorker", Recorder)
    win = window.win
    source = tmp_path / "movie.tif"
    source.write_bytes(b"")
    win._project = window.store.create_project(source)
    win._metadata = SimpleNamespace(path=source, axes="TYX")
    win._stack = np.zeros((2, 8, 8), dtype=np.float32)
    win.start_analysis()
    job = win._analysis_job
    assert created and created[0][1] is resolved
    assert created[0][0].segmentation.model_path == str(resolved.path)
    assert job is not None
    job.wait()


# --------------------------------------------------------------------------
# Reopening: only a user-chosen axis order is replayed
# --------------------------------------------------------------------------


def test_a_canonical_input_axes_is_not_forced_on_the_importer():
    from corridor.ui.workers import import_config_for

    # v2 writes input.axes for every run (canonical, C reduced).
    assert import_config_for({"input": {"axes": "TYX"}}) is None
    config = import_config_for({"input": {"axes": "TYX", "channel_index": 2}})
    assert config.axes is None and config.channel_index == 2


def test_a_user_chosen_axis_order_is_replayed():
    from corridor.core.config import ImportConfig
    from corridor.ui.workers import import_config_for

    manifest = {"input": {"axes": "ZYX", "axes_source": "user"}}
    assert import_config_for(manifest).axes == "ZYX"
    assert import_config_for({"import": {"axes": "TZCYX", "channel_index": 1}}).axes == "TZCYX"
    assert import_config_for({"config": {"import": {"axes": "ZYX"}}}).axes == "ZYX"
    # The project's own recorded choice (the raw file order) outranks all.
    recorded = ImportConfig(axes="TZCYX", channel_index=1)
    config = import_config_for({"input": {"axes": "TZYX"}}, recorded)
    assert (config.axes, config.channel_index) == ("TZCYX", 1)


def test_reopening_reads_with_the_projects_own_axis_order(window, tmp_path):
    win = window.win
    source = tmp_path / "planes.tif"
    source.write_bytes(b"")
    project = window.store.create_project(source)
    own = RunConfig()
    own.import_.axes = "ZYX"
    project.config = own.to_dict()
    win._project = project
    recorded = win._recorded_import(project.path)
    assert recorded is not None and recorded.axes == "ZYX"
    # Another project's directory, or a project with no choice: nothing.
    assert win._recorded_import(tmp_path / "elsewhere") is None
    project.config = RunConfig().to_dict()
    assert win._recorded_import(project.path) is None


# --------------------------------------------------------------------------
# Label image chooser
# --------------------------------------------------------------------------


def test_a_label_image_can_be_chosen_and_cleared(window, tmp_path, monkeypatch):
    win = window.win
    labels = tmp_path / "labels.tif"
    monkeypatch.setattr(win, "_ask_open_path", lambda title, start, f: str(labels))
    win.dataset.labels_button.click()
    assert win._config.import_.labels_path == str(labels)
    assert win.dataset.field_model._value.full_text() == "imported labels: labels.tif"
    assert win.dataset.labels_button.text() == "Segment with the model instead"

    win.dataset.labels_button.click()
    assert win._config.import_.labels_path is None
    assert win.dataset.labels_button.text() == "Use a label image…"

    # Cancelling the chooser changes nothing.
    monkeypatch.setattr(win, "_ask_open_path", lambda title, start, f: "")
    win.dataset.labels_button.click()
    assert win._config.import_.labels_path is None


# --------------------------------------------------------------------------
# Ambiguous axes
# --------------------------------------------------------------------------


class AmbiguousAxes(ValueError):
    """Stands in for imaging.AmbiguousAxes (work package D)."""

    def __init__(self, choices, message):
        super().__init__(message)
        self.choices = choices
        self.message = message


def _metadata(path: Path, axes: str):
    meta = imaging.StackMetadata(
        path=path, n_frames=1, height=8, width=10, dtype="float32",
        axes_raw="QYX", axes_interpretation="chosen by the user",
        pixel_size_um=imaging.Calibrated(None, "missing"),
        frame_interval_min=imaging.Calibrated(None, "missing"),
    )
    meta.axes = axes
    return meta


def test_ambiguous_axes_are_asked_once_and_the_file_is_read_again(
    window, tmp_path, monkeypatch, qt_app
):
    reads: list[str | None] = []

    def read_metadata(path, import_config=None):
        axes = getattr(import_config, "axes", None)
        reads.append(axes)
        if not axes:
            raise AmbiguousAxes(["TYX", "ZYX"], "The file has 5 planes and no T/Z labels.")
        return _metadata(Path(path), axes)

    monkeypatch.setattr(imaging, "AmbiguousAxes", AmbiguousAxes, raising=False)
    monkeypatch.setattr(imaging, "read_metadata", read_metadata)
    monkeypatch.setattr(
        imaging, "load_stack", lambda path, metadata=None: np.zeros((5, 8, 10), np.float32)
    )
    asked: list[tuple[list[str], str]] = []

    def ask(choices, message):
        asked.append((list(choices), message))
        return "ZYX"

    win = window.win
    monkeypatch.setattr(win, "_ask_axis_order", ask)
    source = tmp_path / "planes.tif"
    source.write_bytes(b"")

    win.open_file(str(source))
    assert pump_until(qt_app, lambda: win._metadata is not None)

    assert asked == [(["TYX", "ZYX"], "The file has 5 planes and no T/Z labels.")]
    assert reads == [None, "ZYX"]
    assert win._config.import_.axes == "ZYX"
    assert window.errors == []


def test_declining_the_axis_question_opens_nothing(window, tmp_path, monkeypatch, qt_app):
    def read_metadata(path, import_config=None):
        raise AmbiguousAxes(["TYX", "ZYX"], "ambiguous")

    monkeypatch.setattr(imaging, "read_metadata", read_metadata)
    asked = []
    win = window.win
    monkeypatch.setattr(win, "_ask_axis_order", lambda c, m: asked.append(c) or None)
    source = tmp_path / "planes.tif"
    source.write_bytes(b"")
    win.open_file(str(source))
    assert pump_until(qt_app, lambda: bool(asked))
    for _ in range(20):
        qt_app.processEvents()
    assert win._metadata is None
    assert win._config.import_.axes is None
    assert window.errors == []


# --------------------------------------------------------------------------
# Exports through the window
# --------------------------------------------------------------------------


def _open_v2(window, tmp_path):
    win = window.win
    analysis = syn.load_saved(syn.write_v2_analysis(tmp_path / "v2"))
    win._analysis = analysis
    win.results.load(analysis, syn.stack_2d())
    return win, analysis


def test_selected_track_xlsx_export_passes_track_format_and_reference(
    window, tmp_path, monkeypatch, qt_app
):
    from corridor.store import project

    calls = []

    def export_track(saved, track_id, path, *, fmt="csv", reference_point_px=None):
        calls.append((saved, track_id, Path(path), fmt, reference_point_px))
        Path(path).write_bytes(b"PK")
        return Path(path)

    monkeypatch.setattr(project, "export_track", export_track, raising=False)
    win, analysis = _open_v2(window, tmp_path)
    win.results.select_track(2)
    win.results._reference_picked(30.0, 12.0)

    asked = []
    target = tmp_path / "exports" / "chosen.xlsx"
    target.parent.mkdir()
    monkeypatch.setattr(
        win, "_ask_save_path", lambda title, suggested, f: asked.append((title, suggested, f)) or str(target)
    )
    win.results.export_actions["track_xlsx"].trigger()
    assert pump_until(qt_app, lambda: bool(win._last_export))

    assert calls == [(analysis, 2, target, "xlsx", (30.0, 12.0))]
    assert asked[0][1].endswith("synthetic_track_2.xlsx")
    assert win._last_export == [target]
    assert window.errors == []
    # The reference point also reached the project's configuration path.
    assert win._config.measurement.reference_point_px == (30.0, 12.0)


def test_bundle_export_goes_to_a_named_folder(window, tmp_path, monkeypatch, qt_app):
    from corridor.store import project

    seen = []

    def export_bundle(saved, destination):
        seen.append((saved, Path(destination)))
        Path(destination).mkdir(parents=True, exist_ok=True)
        return [Path(destination) / "tracks.csv"]

    monkeypatch.setattr(project, "export_bundle", export_bundle)
    win, analysis = _open_v2(window, tmp_path)
    monkeypatch.setattr(win, "_ask_directory", lambda title, start: str(tmp_path / "out"))
    win.export_results("bundle")
    assert pump_until(qt_app, lambda: bool(win._last_export))
    assert seen == [(analysis, tmp_path / "out" / "synthetic_corridor")]
    assert window.store.get_setting("export_dir") == str(tmp_path / "out")


def test_export_reports_do_not_pile_up(window, tmp_path, monkeypatch, qt_app):
    from PySide6.QtCore import QCoreApplication, QEvent, Qt
    from PySide6.QtWidgets import QMessageBox

    from corridor.store import project

    def export_msd_csv(saved, path):
        Path(path).write_text("track_id\n", encoding="utf-8")
        return Path(path)

    monkeypatch.setattr(project, "export_msd_csv", export_msd_csv, raising=False)
    win, _ = _open_v2(window, tmp_path)
    for index in range(3):
        target = tmp_path / f"msd_{index}.csv"
        monkeypatch.setattr(win, "_ask_save_path", lambda *a, t=target: str(t))
        win._last_export = []
        win.export_results("msd_csv")
        assert pump_until(qt_app, lambda: bool(win._last_export))
    for _ in range(10):
        qt_app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    boxes = [b for b in win.findChildren(QMessageBox) if b.windowTitle() == "Exported"]
    assert len(boxes) == 1, "each export replaces the last report instead of adding one"
    assert boxes[0].testAttribute(Qt.WA_DeleteOnClose)


def test_a_track_export_with_no_selection_warns_and_writes_nothing(window, tmp_path, monkeypatch):
    win, _ = _open_v2(window, tmp_path)
    monkeypatch.setattr(win, "_ask_save_path", lambda *a: pytest.fail("no dialog expected"))
    win.export_results("track_csv")
    assert window.warnings and "Select a track" in window.warnings[0][0]


def test_cancelling_the_save_dialog_writes_nothing(window, tmp_path, monkeypatch, qt_app):
    from corridor.store import project

    monkeypatch.setattr(
        project, "export_msd_csv", lambda *a, **k: pytest.fail("nothing to write"), raising=False
    )
    win, _ = _open_v2(window, tmp_path)
    monkeypatch.setattr(win, "_ask_save_path", lambda *a: "")
    win.export_results("msd_csv")
    for _ in range(10):
        qt_app.processEvents()
    assert win._last_export == []
