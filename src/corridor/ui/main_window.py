"""Application shell: screens, navigation and the jobs behind them.

Three 2.0 rules are enforced here, because this is where a run starts:

*   **A new project inherits tuning, never facts about another file**
    (``RunConfig.for_new_project``): no calibration override, model choice,
    axis order or reference point carries from the last run into the next.
    KK1 and KK2 are 0.639 and 0.467 µm/px; a silent carry-over is a wrong
    answer with nothing on screen to show it.
*   **No validated model, no analysis.** The model is resolved and
    hash-checked before the worker starts, and the run is pinned to the file
    that was checked; a ModelUnavailable is shown with the contract's own
    sentence and the paths tried, and nothing falls back to another model.
*   **An ambiguous file is asked about, not guessed.** When the importer
    cannot tell T from Z it refuses (AmbiguousAxes); the window asks which
    order the file has, records it in ``ImportConfig.axes`` and reads again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import app_meta
from ..core import pipeline, updates
from ..core.config import ImportConfig, RunConfig
from ..store import db
from ..store.project import SavedAnalysis, analysis_is_complete
from .dialogs import AboutDialog, AxisOrderDialog, ErrorDialog, SettingsDialog, _gpu_available
from .model_status import verified_model
from .screens.dataset import DatasetScreen, preview_stack
from .screens.home import HomeScreen
from .screens.results import ResultsScreen
from .widgets.update_banner import UpdateBanner, UpdateConsent
from .workers import (
    EXPORT_BUNDLE,
    EXPORT_LABELS,
    EXPORT_MSD_CSV,
    EXPORT_SUMMARIES_CSV,
    EXPORT_TRACK_CSV,
    EXPORT_TRACK_XLSX,
    EXPORT_TRACKS_CSV,
    LABELS_UNSUPPORTED,
    TRACK_EXPORTS,
    AnalysisWorker,
    DatasetWorker,
    ExportWorker,
    Job,
    ResultsWorker,
    labels_supported,
)

SCREEN_HOME = 0
SCREEN_DATASET = 1
SCREEN_RESULTS = 2

#: Default file name (after the source's stem) and save-dialog filter per export.
_EXPORT_FILES: dict[str, tuple[str, str]] = {
    EXPORT_TRACK_CSV: ("_track_{track}.csv", "CSV (*.csv)"),
    EXPORT_TRACK_XLSX: ("_track_{track}.xlsx", "Excel workbook (*.xlsx)"),
    EXPORT_TRACKS_CSV: ("_tracks.csv", "CSV (*.csv)"),
    EXPORT_SUMMARIES_CSV: ("_track_summary.csv", "CSV (*.csv)"),
    EXPORT_MSD_CSV: ("_track_msd.csv", "CSV (*.csv)"),
}


def _exec_once(dialog: QDialog) -> int:
    """Run a modal dialog, then schedule it for deletion.

    A dialog parented to the window lives as long as the window unless it is
    deleted, so each one exec'd and dropped was a small leak per use. The
    deletion is deferred (``deleteLater``), so the caller can still read
    ``clickedButton()`` or a chosen value straight after this returns.
    """
    try:
        return dialog.exec()
    finally:
        dialog.deleteLater()


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
        self._metadata: Any = None
        self._stack: np.ndarray | None = None
        self._config = RunConfig()
        self._analysis: SavedAnalysis | None = None
        self._opening: Path | None = None
        self._export_destination: Path | None = None
        self._last_export: list[Path] = []
        self._export_box: QMessageBox | None = None
        #: The model ``_model_ready`` verified for the run being started.
        self._resolved_model: Any = None

        root = QWidget()
        root.setObjectName("Root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Update notices sit above everything and are never modal: a banner
        # over somebody's analysis is an interruption, not a service.
        self.update_consent = UpdateConsent()
        self.update_banner = UpdateBanner()
        self.update_consent.hide()
        layout.addWidget(self.update_consent)
        layout.addWidget(self.update_banner)

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
        self._offer_update_check()

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
        self.dataset.labels_requested.connect(self._labels_requested)

        self.results.back_requested.connect(self.go_home)
        self.results.export_requested.connect(self.export_results)
        self.results.napari_requested.connect(self.open_in_napari)
        self.results.open_folder_requested.connect(self.open_results_folder)
        self.results.reference_point_changed.connect(self._reference_changed)

        self.update_consent.answered.connect(self._update_consent_given)
        self.update_banner.dismissed.connect(self._update_dismissed)
        self.update_banner.open_requested.connect(self._open_release_page)

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
        if self.stack.currentIndex() == SCREEN_RESULTS and self.results.reference_button.isChecked():
            self.results.set_reference_mode(False)
        elif self.stack.currentIndex() == SCREEN_RESULTS and self.results.selected_track is not None:
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
        self._read_dataset(source)

    def _read_dataset(self, source: Path) -> None:
        self._opening = source
        self._start_job(
            DatasetWorker(source, self._config.import_),
            finished=self._dataset_ready,
            failed=self._show_error,
            ambiguous=self._dataset_ambiguous,
        )

    def _fresh_config(self, source: Path, project: db.ProjectRecord) -> RunConfig:
        """The configuration for opening ``source`` in ``project``.

        A project's own saved configuration is loaded as it was saved:
        reopening a project is not starting one. A *new* project starts from
        the saved preferences through ``for_new_project``, which drops every
        calibration override, model key, import setting and reference point.
        """
        if project.config:
            config = RunConfig.from_dict(project.config)
        else:
            config = RunConfig.for_new_project(self.store.get_setting("default_config", {}))
        config.input_path = str(source)
        config.output_dir = str(project.path)
        config.segmentation.use_gpu = bool(
            self.store.get_setting("use_gpu", False) and _gpu_available()
        )
        return config

    def _dataset_ambiguous(self, exc: Any) -> None:
        """The file's T and Z cannot be told apart: ask, record, read again."""
        source = self._opening
        if source is None:
            return
        choices = [str(c) for c in (getattr(exc, "choices", None) or [])]
        message = str(getattr(exc, "message", None) or exc)
        if self._config.import_.axes:
            # An explicit order was already given and the file still cannot be
            # read with it. Asking again would loop; say what happened instead.
            self._show_error(
                f"This file could not be read with the axis order {self._config.import_.axes}.\n\n"
                f"{message}",
                "",
            )
            self._config.import_.axes = None
            return
        if not choices:
            self._show_error(message, "")
            return
        choice = self._ask_axis_order(choices, message)
        if not choice:
            return
        self._config.import_.axes = choice
        self._read_dataset(source)

    def _ask_axis_order(self, choices: Sequence[str], message: str) -> str | None:
        """Modal question; replaced in tests."""
        dialog = AxisOrderDialog(choices, message, self)
        if _exec_once(dialog) == QDialog.Accepted:
            return dialog.choice
        return None

    def _dataset_ready(self, metadata, stack: np.ndarray) -> None:
        self._metadata = metadata
        self._stack = stack
        if self._project is not None:
            self._project.n_frames = getattr(metadata, "n_frames", self._project.n_frames)
            self._project.height = getattr(metadata, "height", self._project.height)
            self._project.width = getattr(metadata, "width", self._project.width)
            pixel = getattr(metadata, "pixel_size_um", None)
            interval = getattr(metadata, "frame_interval_min", None)
            self._project.pixel_size_um = getattr(pixel, "value", pixel)
            self._project.frame_interval_min = getattr(interval, "value", interval)
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

            flat = preview_stack(stack)
            middle = frame_to_display(flat[flat.shape[0] // 2])
            image = Image.fromarray(middle).convert("L")
            image.thumbnail((256, 256))
            self._project.preview_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(self._project.preview_path)
        except Exception:  # noqa: BLE001 - a missing thumbnail is not an error
            pass

    def _config_edited(self) -> None:
        self.dataset.advanced.apply_to(self._config)

    def _ask_open_path(self, title: str, start: str, file_filter: str) -> str:
        """Modal file chooser; replaced in tests."""
        path, _ = QFileDialog.getOpenFileName(self, title, start, file_filter)
        return path

    def _labels_requested(self, use_labels: bool) -> None:
        """Set or clear ``ImportConfig.labels_path`` for the open file.

        Whether this build can actually analyse the labels is checked when
        Analyse is pressed (``_model_ready``), not here: the choice is
        recorded either way, and refused with a reason rather than ignored.
        """
        if use_labels:
            start = str(Path(self._config.input_path).parent) if self._config.input_path else ""
            path = self._ask_open_path(
                "Choose a label image",
                start,
                "Label images (*.tif *.tiff *.npy *.npz);;All files (*)",
            )
            if not path:
                return
            self._config.import_.labels_path = str(path)
        else:
            self._config.import_.labels_path = None
        self.dataset.show_model(self._config, self._dimensionality())

    def _dimensionality(self) -> str:
        """'3D' when the open file has a Z axis, by its metadata or its pixels."""
        axes = str(getattr(self._metadata, "axes", "") or "")
        if "Z" in axes.upper():
            return "3D"
        if self._stack is not None and self._stack.ndim == 4:
            return "3D"
        return "2D"

    # ------------------------------------------------------------------ analyse
    def _model_ready(self) -> bool:
        """Resolve and hash-check the validated model, and pin the run to it.

        Imported labels need no model, but only a pipeline that can read them
        may be given them (see :func:`labels_supported`). Otherwise a
        ModelUnavailable is shown verbatim -- the contract's sentence, then
        the paths tried -- and the analysis does not start. There is no
        fallback to try.

        Checking the model is not enough on its own: the run has to load the
        file that was checked. A new project's configuration has no
        ``model_path`` (``for_new_project`` drops it), and the 1.x
        segmentation service turns that into Cellpose's built-in ``cyto3``;
        a reopened 1.x project may still name a model picked with the old
        picker. So the verified ResolvedModel goes to the worker (for a
        pipeline that takes ``model=``, the 2.0 seam), and its path is written
        over the legacy fields (for one that still reads them). Whichever
        pipeline runs, it loads the hash-checked file and nothing else.
        """
        self._resolved_model = None
        if self._config.import_.labels_path:
            if labels_supported():
                return True
            self._show_error(LABELS_UNSUPPORTED, "")
            return False
        status = verified_model(self._dimensionality())
        if not status.verified or status.path is None:
            self._show_error(status.message, "")
            return False
        seg = self._config.segmentation
        seg.model_path = str(status.path)
        seg.use_custom_model = True
        # Companion models were the 1.x "models" rungs; 2.0 runs one model.
        seg.ensemble_model_paths = ()
        self._resolved_model = status.resolved
        return True

    def start_analysis(self) -> None:
        if self._metadata is None or self._project is None:
            return
        self.dataset.advanced.apply_to(self._config)
        if not self._model_ready():
            return
        self._config.input_path = str(self._metadata.path)
        self._config.output_dir = str(self._project.path)

        self._project.config = self._config.to_dict()
        self._project.status = db.STATUS_RUNNING
        self._project.error = None
        self.store.update_project(self._project)

        self.dataset.set_busy(True, "Starting")
        worker = AnalysisWorker(self._config, model=self._resolved_model)
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
            # Saved whole; ``for_new_project`` strips what must not carry over
            # when the next project reads it.
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
            ResultsWorker(directory, self._recorded_import(directory)),
            finished=self._results_ready,
            failed=self._show_error,
        )

    def _recorded_import(self, directory: Path) -> ImportConfig | None:
        """The axis order this project's user chose, to read its source again.

        Read from the project's own saved configuration, never from
        ``self._config``, which can still hold the previous project's
        settings. Only an explicit order is returned: without one the
        metadata decides (see ``workers.import_config_for``).
        """
        project = self._project
        if project is None or not project.config:
            return None
        if Path(project.path) != Path(directory):
            return None
        recorded = RunConfig.from_dict(project.config).import_
        return recorded if recorded.axes else None

    def _results_ready(self, analysis: SavedAnalysis, metadata, stack: np.ndarray) -> None:
        self._analysis = analysis
        self._metadata = metadata
        self._stack = stack
        self.stack.setCurrentIndex(SCREEN_RESULTS)
        self.results.load(analysis, stack)

    def _reference_changed(self, point: tuple[float, ...] | None) -> None:
        """Keep the project's configuration in step with the analysis's point.

        The point itself is stored beside the analysis (it is read from there
        when the results open); the project configuration records it too, so
        re-analysing this project measures D2R from the same place.
        """
        value = [float(v) for v in point] if point else None
        self._config.measurement.reference_point_px = tuple(value) if value else None
        if self._project is None or self._analysis is None:
            return
        if Path(self._project.path) != Path(self._analysis.directory):
            return
        config = dict(self._project.config or {})
        measurement = dict(config.get("measurement") or {})
        measurement["reference_point_px"] = value
        config["measurement"] = measurement
        self._project.config = config
        self.store.update_project(self._project)

    # ------------------------------------------------------------------- export
    def _export_if_possible(self) -> None:
        if self.stack.currentIndex() == SCREEN_RESULTS:
            self.export_results(EXPORT_BUNDLE)

    def _ask_directory(self, title: str, start: str) -> str:
        """Modal folder chooser; replaced in tests."""
        return QFileDialog.getExistingDirectory(self, title, start)

    def _ask_save_path(self, title: str, suggested: str, file_filter: str) -> str:
        """Modal save dialog; replaced in tests."""
        path, _ = QFileDialog.getSaveFileName(self, title, suggested, file_filter)
        return path

    def export_results(self, kind: str = EXPORT_BUNDLE) -> None:
        """Ask where, then write one export off the UI thread."""
        if self._analysis is None:
            return
        analysis = self._analysis
        stem = analysis.source_path.stem if analysis.source_path else analysis.directory.name
        default_dir = Path(self.store.get_setting("export_dir", str(Path.home() / "Documents")))
        track_id = self.results.selected_track
        if kind in TRACK_EXPORTS and track_id is None:
            self._warn("Select a track first.", "Click a track on the image or in the list.")
            return

        if kind == EXPORT_BUNDLE:
            target = self._ask_directory("Export results to", str(default_dir))
            if not target:
                return
            destination = Path(target) / f"{stem}_corridor"
            self.store.set_setting("export_dir", target)
        else:
            suffix, file_filter = _EXPORT_FILES[kind]
            suggested = default_dir / (stem + suffix.format(track=track_id))
            target = self._ask_save_path(EXPORT_LABELS[kind], str(suggested), file_filter)
            if not target:
                return
            destination = Path(target)
            self.store.set_setting("export_dir", str(destination.parent))

        self._export_destination = destination
        self._start_job(
            ExportWorker(
                analysis,
                destination,
                kind=kind,
                track_id=track_id,
                reference_point_px=self.results.reference_point_px,
            ),
            finished=self._export_done,
            failed=self._show_error,
        )

    def _export_done(self, written: list[Path]) -> None:
        """Report the export without blocking: the window stays usable."""
        self._last_export = list(written)
        destination = self._export_destination
        if destination is None:
            return
        folder = destination if destination.is_dir() else destination.parent
        previous = self._export_box
        if previous is not None:
            # A newer export supersedes the last report. Without this every
            # export left one more hidden QMessageBox parented to the window.
            try:
                previous.close()
                previous.deleteLater()
            except RuntimeError:  # already deleted by Qt (closed by the user)
                pass
        box = QMessageBox(self)
        box.setAttribute(Qt.WA_DeleteOnClose, True)
        box.setWindowTitle("Exported")
        box.setIcon(QMessageBox.NoIcon)
        box.setText(f"{len(written)} file{'s' if len(written) != 1 else ''} written.")
        box.setInformativeText(str(destination))
        open_button = box.addButton("Open folder", QMessageBox.AcceptRole)
        box.addButton("Done", QMessageBox.RejectRole)
        box.setProperty("folder", str(folder))
        self._export_box = box
        self._export_open_button = open_button
        box.buttonClicked.connect(self._export_box_clicked)
        box.open()

    def _export_box_clicked(self, button) -> None:
        box = self._export_box
        if box is not None and button is self._export_open_button:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(box.property("folder"))))

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
            # The stack already in memory: reading the TIFF again would block
            # the UI thread for as long as the file takes to load.
            open_saved_in_napari(self._analysis, stack=self._stack)
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
        _exec_once(SettingsDialog(self.store, self))

    def show_about(self) -> None:
        _exec_once(AboutDialog(self))

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
        _exec_once(box)
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
        _exec_once(box)

    def _show_error(self, message: str, detail: str = "") -> None:
        _exec_once(ErrorDialog(message, detail, self))

    # ------------------------------------------------------------ updates
    def _offer_update_check(self) -> None:
        """Ask on first run; check only if the user has already said yes.

        Nothing here touches the network until consent exists. A user who has
        never been asked is treated as having said no.
        """
        if not updates.has_been_asked(self.store):
            self.update_consent.show()
            return
        if updates.should_check(self.store):
            self._check_for_updates()

    def _update_consent_given(self, enabled: bool) -> None:
        updates.record_choice(self.store, enabled)
        if enabled:
            self._check_for_updates()

    def _check_for_updates(self) -> None:
        from .workers import UpdateWorker

        self._start_job(UpdateWorker(), finished=self._update_checked)

    def _update_checked(self, release) -> None:
        """Runs on the UI thread. A None release means no information, not an error."""
        if release is None or not release.is_newer:
            return
        if updates.already_seen(self.store, release.version):
            return
        self.update_banner.show_release(release)

    def _update_dismissed(self, version: str) -> None:
        # Remembered per version, so dismissing 1.3.0 does not silence 1.4.0.
        if version:
            updates.remember_seen(self.store, version)

    def _open_release_page(self, url: str) -> None:
        """Open the page in a browser. Corridor downloads and runs nothing."""
        if not url:
            return
        QDesktopServices.openUrl(QUrl(url))

    # --------------------------------------------------------------------- jobs
    def _start_job(
        self, worker, *, finished=None, failed=None, cancelled=None, ambiguous=None
    ) -> Job:
        """Run a worker, delivering every callback on the UI thread.

        The slots passed in must be bound methods of this window. A bare
        function or lambda has no thread affinity, so Qt would call it directly
        on the worker thread -- which for anything that touches a widget is
        undefined behaviour.
        """
        job = Job(worker)
        for name, slot in (
            ("finished", finished),
            ("failed", failed),
            ("cancelled_signal", cancelled),
            ("ambiguous", ambiguous),
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
            _exec_once(box)
            if box.clickedButton() is not stop:
                event.ignore()
                return
            self.cancel_analysis()
        for job in list(self._jobs):
            job.wait(3000)
        self.store.close()
        event.accept()
