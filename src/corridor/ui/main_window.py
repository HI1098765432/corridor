"""Application shell: screens, navigation and the jobs behind them."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import app_meta, resources
from ..core import pipeline
from ..core.config import RunConfig
from ..core.imaging import StackMetadata
from ..core.segmentation import cellpose_version, gpu_available
from ..store import db
from ..store.project import SavedAnalysis, analysis_is_complete, load_analysis
from .dialogs import AboutDialog, ErrorDialog, SettingsDialog
from .screens.dataset import DatasetScreen
from .screens.home import HomeScreen
from .screens.results import ResultsScreen
from .theme import PALETTE, SPACE, stylesheet
from .workers import AnalysisWorker, DatasetWorker, ExportWorker, Job, ResultsWorker

SCREEN_HOME = 0
SCREEN_DATASET = 1
SCREEN_RESULTS = 2


class MainWindow(QMainWindow):
    def __init__(self, store: db.Store | None = None) -> None:
        super().__init__()
        self.store = store or db.Store()
        self.setWindowTitle(app_meta.APP_NAME)
        self.setMinimumSize(940, 620)
        self.resize(1320, 860)
        self.setAcceptDrops(True)

        self._jobs: list[Job] = []
        self._analysis_job: Job | None = None
        self._project: db.ProjectRecord | None = None
        self._metadata: StackMetadata | None = None
        self._stack: np.ndarray | None = None
        self._config = RunConfig()
        self._analysis: SavedAnalysis | None = None

        root = QWidget()
        root.setObjectName("Root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.stack = QStackedWidget()
        self.home = HomeScreen()
        self.dataset = DatasetScreen()
        self.results = ResultsScreen()
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.dataset)
        self.stack.addWidget(self.results)
        layout.addWidget(self.stack)
        self.setCentralWidget(root)

        self._connect()
        self._install_shortcuts()
        self.refresh_recent()

    # ------------------------------------------------------------------ wiring
    def _connect(self) -> None:
        self.home.file_chosen.connect(self.open_file)
        self.home.project_opened.connect(self.open_project)
        self.home.project_removed.connect(self.remove_project)
        self.home.settings_requested.connect(self.show_settings)

        self.dataset.back_requested.connect(self.go_home)
        self.dataset.analyse_requested.connect(self.start_analysis)
        self.dataset.cancel_requested.connect(self.cancel_analysis)
        self.dataset.advanced.changed.connect(self._config_edited)

        self.results.back_requested.connect(self.go_home)
        self.results.export_requested.connect(self.export_results)
        self.results.napari_requested.connect(self.open_in_napari)
        self.results.open_folder_requested.connect(self.open_results_folder)

    def _install_shortcuts(self) -> None:
        def shortcut(sequence: str, handler) -> None:
            action = QShortcut(QKeySequence(sequence), self)
            action.activated.connect(handler)

        shortcut("Ctrl+O", self._browse)
        shortcut("Ctrl+E", self._export_if_possible)
        shortcut("Escape", self._escape)
        shortcut("Ctrl+,", self.show_settings)
        shortcut("F1", self.show_about)
        shortcut("Space", self._toggle_play)
        shortcut("Left", lambda: self._step(-1))
        shortcut("Right", lambda: self._step(1))

    def _escape(self) -> None:
        if self.stack.currentIndex() == SCREEN_RESULTS and self.results._selected_track is not None:
            self.results.select_track(None)
        elif self.stack.currentIndex() != SCREEN_HOME and not self._busy:
            self.go_home()

    def _toggle_play(self) -> None:
        if self.stack.currentIndex() == SCREEN_RESULTS:
            self.results.timeline.toggle_play()

    def _step(self, delta: int) -> None:
        if self.stack.currentIndex() == SCREEN_RESULTS:
            self.results.timeline.step(delta)

    @property
    def _busy(self) -> bool:
        return self._analysis_job is not None and self._analysis_job.running

    # --------------------------------------------------------------- navigation
    def go_home(self) -> None:
        self.results.timeline.stop()
        self.refresh_recent()
        self.stack.setCurrentIndex(SCREEN_HOME)

    def refresh_recent(self) -> None:
        self.home.set_recent(self.store.recent_projects())

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a time-lapse", "", "TIFF stacks (*.tif *.tiff);;All files (*)"
        )
        if path:
            self.open_file(path)

    # ------------------------------------------------------------------- open
    def open_file(self, path: str) -> None:
        if self._busy:
            self._warn("An analysis is already running.", "Stop it before opening another file.")
            return
        source = Path(path)
        if not source.exists():
            self._warn("That file no longer exists.", str(source))
            return

        existing = self.store.find_by_source(source)
        if existing and existing.has_results and analysis_is_complete(existing.path):
            self.open_project(existing.id)
            return

        self._project = existing or self.store.create_project(source)
        self._config = self._fresh_config(source, self._project)
        self._start_job(
            DatasetWorker(source),
            finished=self._dataset_ready,
            failed=self._show_error,
        )

    def _fresh_config(self, source: Path, project: db.ProjectRecord) -> RunConfig:
        saved = project.config or self.store.get_setting("default_config", {})
        config = RunConfig.from_dict(saved) if saved else RunConfig()
        config.input_path = str(source)
        config.output_dir = str(project.path)
        if not config.segmentation.model_path:
            model = resources.bundled_model_path()
            config.segmentation.model_path = str(model) if model else None
            config.segmentation.use_custom_model = bool(model)
        config.segmentation.use_gpu = bool(
            self.store.get_setting("use_gpu", False) and gpu_available()
        )
        return config

    def _dataset_ready(self, metadata: StackMetadata, stack: np.ndarray) -> None:
        self._metadata = metadata
        self._stack = stack
        if self._project is not None:
            self._project.n_frames = metadata.n_frames
            self._project.height = metadata.height
            self._project.width = metadata.width
            self._project.pixel_size_um = metadata.pixel_size_um.value
            self._project.frame_interval_min = metadata.frame_interval_min.value
            self.store.update_project(self._project)
        self.stack.setCurrentIndex(SCREEN_DATASET)
        self.dataset.set_dataset(metadata, stack, self._config)
        self.dataset.set_busy(False)
        self._save_preview(stack)

    def _save_preview(self, stack: np.ndarray) -> None:
        """A small thumbnail so the home screen can show what a project is."""
        if self._project is None or stack.size == 0:
            return
        try:
            from PIL import Image

            from ..core.imaging import frame_to_display

            middle = frame_to_display(stack[stack.shape[0] // 2])
            image = Image.fromarray(middle).convert("L")
            image.thumbnail((256, 256))
            self._project.preview_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(self._project.preview_path)
        except Exception:  # noqa: BLE001 - a missing thumbnail is not an error
            pass

    def _config_edited(self) -> None:
        self.dataset.advanced.apply_to(self._config)

    # ------------------------------------------------------------------ analyse
    def start_analysis(self) -> None:
        if self._metadata is None or self._project is None:
            return
        self.dataset.advanced.apply_to(self._config)
        self._config.input_path = str(self._metadata.path)
        self._config.output_dir = str(self._project.path)

        self._project.config = self._config.to_dict()
        self._project.status = db.STATUS_RUNNING
        self._project.error = None
        self.store.update_project(self._project)

        self.dataset.set_busy(True, "Starting")
        worker = AnalysisWorker(self._config)
        # Explicitly queued: these arrive from the analysis thread.
        worker.stage_changed.connect(self.dataset.set_stage, Qt.QueuedConnection)
        worker.progressed.connect(self.dataset.set_progress, Qt.QueuedConnection)
        self._analysis_job = self._start_job(
            worker,
            finished=self._analysis_done,
            failed=self._analysis_failed,
            cancelled=self._analysis_cancelled,
        )

    def cancel_analysis(self) -> None:
        if self._analysis_job is not None:
            worker = self._analysis_job.worker
            if isinstance(worker, AnalysisWorker):
                worker.cancel()
            self.dataset.set_stage("Stopping", "finishing the current frame")

    def _analysis_done(self, result: pipeline.AnalysisResult) -> None:
        self.dataset.set_busy(False)
        self._analysis_job = None
        if self._project is not None:
            self._project.status = db.STATUS_COMPLETE
            self._project.n_detections = result.n_detections
            self._project.n_tracks = result.n_tracks
            self._project.n_warnings = sum(
                1 for i in result.issues if i.severity in ("warning", "critical")
            )
            self._project.manifest = result.manifest
            self._project.config = result.config.to_dict()
            self._project.pixel_size_um = result.scale.pixel_size_um if result.scale.calibrated_space else None
            self._project.frame_interval_min = (
                result.scale.frame_interval_min if result.scale.calibrated_time else None
            )
            self.store.update_project(self._project)
            self.store.set_setting("default_config", result.config.to_dict())
        self._open_results(Path(result.output_dir or ""))

    def _analysis_failed(self, message: str, detail: str) -> None:
        self.dataset.set_busy(False)
        self._analysis_job = None
        if self._project is not None:
            self._project.status = db.STATUS_FAILED
            self._project.error = message
            self.store.update_project(self._project)
        self._show_error(message, detail)

    def _analysis_cancelled(self) -> None:
        self.dataset.set_busy(False)
        self._analysis_job = None
        if self._project is not None:
            self.store.set_status(self._project.id, db.STATUS_CANCELLED)

    # ------------------------------------------------------------------ results
    def open_project(self, project_id: str) -> None:
        record = self.store.get_project(project_id)
        if record is None:
            return
        self._project = record
        if record.config:
            self._config = RunConfig.from_dict(record.config)
        if not analysis_is_complete(record.path):
            if record.source_exists:
                self.open_file(record.source_path)
            else:
                self._warn(
                    "That file has moved.",
                    f"Corridor last saw it at:\n{record.source_path}",
                )
            return
        self._open_results(record.path)

    def _open_results(self, directory: Path) -> None:
        """Load a saved analysis. Every file read happens off the UI thread."""
        self._start_job(
            ResultsWorker(directory),
            finished=self._results_ready,
            failed=self._show_error,
        )

    def _results_ready(
        self, analysis: SavedAnalysis, metadata: StackMetadata, stack: np.ndarray
    ) -> None:
        self._analysis = analysis
        self._metadata = metadata
        self._stack = stack
        self.stack.setCurrentIndex(SCREEN_RESULTS)
        self.results.load(analysis, stack)

    # ------------------------------------------------------------------- export
    def _export_if_possible(self) -> None:
        if self.stack.currentIndex() == SCREEN_RESULTS:
            self.export_results()

    def export_results(self) -> None:
        if self._analysis is None:
            return
        default = self.store.get_setting("export_dir", str(Path.home() / "Documents"))
        name = (self._analysis.source_path.stem if self._analysis.source_path else "corridor")
        target = QFileDialog.getExistingDirectory(self, "Export results to", default)
        if not target:
            return
        destination = Path(target) / f"{name}_corridor"
        self.store.set_setting("export_dir", target)
        self._export_destination = destination
        self._start_job(
            ExportWorker(self._analysis, destination),
            finished=self._export_done,
            failed=self._show_error,
        )

    def _export_done(self, written: list[Path]) -> None:
        destination = getattr(self, "_export_destination", None)
        if destination is None:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Exported")
        box.setIcon(QMessageBox.NoIcon)
        box.setText(f"{len(written)} files written.")
        box.setInformativeText(str(destination))
        open_button = box.addButton("Open folder", QMessageBox.AcceptRole)
        box.addButton("Done", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is open_button:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(destination)))

    def open_results_folder(self) -> None:
        if self._analysis is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._analysis.directory)))

    def open_in_napari(self) -> None:
        if self._analysis is None:
            return
        try:
            from ..viz.napari_qc import open_saved_in_napari
        except Exception:  # noqa: BLE001
            self._napari_missing()
            return
        try:
            open_saved_in_napari(self._analysis)
        except ImportError:
            self._napari_missing()
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Napari could not be opened.\n\n{exc}", "")

    def _napari_missing(self) -> None:
        QMessageBox.information(
            self,
            "Napari is not installed",
            "Napari is an optional extra for deep inspection. Corridor's own "
            "viewer shows the same layers.\n\nTo add it, install the 'napari' "
            "package into the environment Corridor runs in.",
        )

    # ------------------------------------------------------------------ dialogs
    def show_settings(self) -> None:
        dialog = SettingsDialog(self.store, self)
        dialog.exec()

    def show_about(self) -> None:
        AboutDialog(self).exec()

    def remove_project(self, project_id: str) -> None:
        record = self.store.get_project(project_id)
        if record is None:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Remove from Corridor")
        box.setIcon(QMessageBox.NoIcon)
        box.setText(f"Remove “{record.name}” from Corridor?")
        box.setInformativeText(
            "The analysis results Corridor created will be deleted.\n"
            "Your original microscopy file is never touched."
        )
        remove = box.addButton("Remove", QMessageBox.DestructiveRole)
        box.addButton("Keep", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is remove:
            self.store.delete_project(project_id, remove_files=True)
            self.refresh_recent()

    def _warn(self, message: str, detail: str = "") -> None:
        box = QMessageBox(self)
        box.setWindowTitle(app_meta.APP_NAME)
        box.setIcon(QMessageBox.NoIcon)
        box.setText(message)
        if detail:
            box.setInformativeText(detail)
        box.exec()

    def _show_error(self, message: str, detail: str = "") -> None:
        ErrorDialog(message, detail, self).exec()

    # --------------------------------------------------------------------- jobs
    def _start_job(self, worker, *, finished=None, failed=None, cancelled=None) -> Job:
        """Run a worker, delivering every callback on the UI thread.

        The slots passed in must be bound methods of this window. A bare
        function or lambda has no thread affinity, so Qt would call it directly
        on the worker thread -- which for anything that touches a widget is
        undefined behaviour.
        """
        job = Job(worker)
        for name, slot in (
            ("finished", finished), ("failed", failed), ("cancelled_signal", cancelled)
        ):
            signal = getattr(worker, name, None)
            if slot is not None and signal is not None:
                if not hasattr(slot, "__self__"):
                    raise TypeError(
                        f"{name} slot must be a bound method so it runs on the UI thread"
                    )
                signal.connect(slot, Qt.QueuedConnection)
        self._jobs.append(job)
        job.thread.finished.connect(self._retire_finished, Qt.QueuedConnection)
        job.start()
        return job

    def _retire_finished(self) -> None:
        """Drop jobs whose threads have stopped. Runs on the UI thread."""
        self._jobs = [job for job in self._jobs if job.running]

    # ------------------------------------------------------------ window events
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        from .screens.home import first_tiff

        if first_tiff(event.mimeData()) and not self._busy:
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        from .screens.home import first_tiff

        path = first_tiff(event.mimeData())
        if path:
            event.acceptProposedAction()
            self.open_file(path)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._busy:
            box = QMessageBox(self)
            box.setWindowTitle("Analysis in progress")
            box.setIcon(QMessageBox.NoIcon)
            box.setText("An analysis is still running.")
            box.setInformativeText("Closing now will stop it. Finished stages stay saved.")
            stop = box.addButton("Stop and close", QMessageBox.DestructiveRole)
            box.addButton("Keep running", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is not stop:
                event.ignore()
                return
            self.cancel_analysis()
        for job in list(self._jobs):
            job.wait(3000)
        self.store.close()
        event.accept()
