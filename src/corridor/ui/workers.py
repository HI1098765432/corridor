"""Background work.

Segmentation takes seconds per frame on a CPU, so it cannot run on the UI
thread.  It runs in a QThread, reports progress through signals, and checks a
cancellation flag between frames -- the only point where stopping is safe and
leaves the partial result consistent.
"""

from __future__ import annotations

import csv
import inspect
import io
import traceback
from pathlib import Path
from typing import Any, Sequence

from PySide6.QtCore import QObject, Qt, QThread, Signal

from ..core import imaging, pipeline
from ..core.config import ImportConfig, RunConfig
from .analysis_view import finite, reference_distance_um, row_z, save_reference_point, z_step_um_of

#: The D2R column of tracks.csv v2 (contract §6).
REFERENCE_COLUMN = "distance_from_reference_um"

#: Every export the results screen offers, with the label it shows. The keys
#: are what ``ResultsScreen.export_requested`` carries.
EXPORT_TRACK_CSV = "track_csv"
EXPORT_TRACK_XLSX = "track_xlsx"
EXPORT_TRACKS_CSV = "tracks_csv"
EXPORT_SUMMARIES_CSV = "summaries_csv"
EXPORT_MSD_CSV = "msd_csv"
EXPORT_BUNDLE = "bundle"

EXPORT_LABELS: dict[str, str] = {
    EXPORT_TRACK_CSV: "Selected track CSV",
    EXPORT_TRACK_XLSX: "Selected track XLSX",
    EXPORT_TRACKS_CSV: "All tracks CSV",
    EXPORT_SUMMARIES_CSV: "All track summaries CSV",
    EXPORT_MSD_CSV: "MSD curves CSV",
    EXPORT_BUNDLE: "Full analysis bundle",
}
TRACK_EXPORTS = (EXPORT_TRACK_CSV, EXPORT_TRACK_XLSX)


def is_ambiguous_axes(exc: BaseException) -> bool:
    """Whether ``exc`` is the importer's "cannot tell T from Z" refusal.

    Matched on the class the imaging module defines when it defines one, and
    on the name otherwise, so the UI keeps working while the importer that
    raises it is merged.
    """
    cls = getattr(imaging, "AmbiguousAxes", None)
    if isinstance(cls, type) and isinstance(exc, cls):
        return True
    return type(exc).__name__ == "AmbiguousAxes"


#: Shown when a label image is set but this build's pipeline cannot read one.
LABELS_UNSUPPORTED = (
    "This build of Corridor cannot analyse an imported label image yet.\n\n"
    "Remove the label image to segment with the validated model instead."
)


def labels_supported() -> bool:
    """Whether the analysis pipeline can measure an imported label image.

    The 1.x pipeline ignores ``ImportConfig.labels_path`` and segments
    anyway -- with whatever its legacy model fields say, which for a new
    project is Cellpose's built-in ``cyto3``. Handing it a label image would
    therefore run an unvalidated model while every screen said "imported
    labels". The 2.0 segmentation module defines ``load_label_stack``; its
    presence is the seam that says labels will actually be read.
    """
    from ..core import segmentation  # noqa: PLC0415 - looked up at call time

    return callable(getattr(segmentation, "load_label_stack", None))


def read_metadata_for(path: str | Path, import_config: ImportConfig | None = None):
    """``imaging.read_metadata`` with the import settings, when it takes them.

    The 2.0 importer accepts ``import_config`` (an explicit axis order and the
    channel to analyse); the 1.x one does not, and is called without it.
    """
    reader = imaging.read_metadata
    try:
        accepts = "import_config" in inspect.signature(reader).parameters
    except (TypeError, ValueError):
        accepts = False
    if accepts:
        return reader(path, import_config=import_config)
    return reader(path)


class _SignalProgress:
    """Adapts the pipeline's progress protocol onto Qt signals."""

    def __init__(self, worker: "AnalysisWorker") -> None:
        self._worker = worker

    def stage(self, name: str, detail: str = "") -> None:
        self._worker.stage_changed.emit(name, detail)

    def step(self, done: int, total: int) -> None:
        self._worker.progressed.emit(int(done), int(total))

    def cancelled(self) -> bool:
        return self._worker.is_cancelled


class AnalysisWorker(QObject):
    """Runs one complete analysis."""

    stage_changed = Signal(str, str)
    progressed = Signal(int, int)
    finished = Signal(object)  # AnalysisResult
    failed = Signal(str, str)  # message, technical detail
    cancelled_signal = Signal()

    def __init__(self, config: RunConfig, *, model: Any = None) -> None:
        super().__init__()
        self.config = config
        #: The ``ResolvedModel`` the window verified, handed on to a pipeline
        #: that accepts ``model=`` so it loads exactly that file.
        self.model = model
        self._cancelled = False

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            options: dict[str, Any] = {}
            if self.model is not None and _accepts(pipeline.run_analysis, "model"):
                options["model"] = self.model
            result = pipeline.run_analysis(self.config, _SignalProgress(self), **options)
        except pipeline.Cancelled:
            self.cancelled_signal.emit()
        except KeyboardInterrupt:
            self.cancelled_signal.emit()
        except Exception as exc:  # noqa: BLE001 - turned into a readable message
            self.failed.emit(friendly_error(exc), traceback.format_exc())
        else:
            self.finished.emit(result)


class DatasetWorker(QObject):
    """Opens a file: reads its metadata and its pixels, off the UI thread.

    Both happen here because a researcher who has just dropped a file expects
    to see it, and an interface that reads the header instantly and then
    freezes on the pixels is worse than one that takes a moment and stays live.

    A file whose T and Z cannot be told apart is not an error: it emits
    ``ambiguous`` with the importer's exception (its ``choices`` and
    ``message``), so the window can ask and retry with an explicit order.
    """

    finished = Signal(object, object)  # StackMetadata, np.ndarray
    failed = Signal(str, str)
    ambiguous = Signal(object)  # AmbiguousAxes

    def __init__(self, path: str | Path, import_config: ImportConfig | None = None) -> None:
        super().__init__()
        self.path = Path(path)
        self.import_config = import_config

    def run(self) -> None:
        try:
            metadata = read_metadata_for(self.path, self.import_config)
            stack = imaging.load_stack(self.path, metadata)
        except Exception as exc:  # noqa: BLE001
            if is_ambiguous_axes(exc):
                self.ambiguous.emit(exc)
            else:
                self.failed.emit(friendly_error(exc), traceback.format_exc())
        else:
            self.finished.emit(metadata, stack)


class ResultsWorker(QObject):
    """Loads a saved analysis, its image and its masks, off the UI thread.

    The mask stack is deliberately forced into memory here. Leaving it lazy
    would move a multi-megabyte decompression onto whichever thread first
    touched it, which is how this used to run inside a paint path.
    """

    finished = Signal(object, object, object)  # SavedAnalysis, StackMetadata, stack
    failed = Signal(str, str)

    def __init__(self, directory: str | Path, import_config: ImportConfig | None = None) -> None:
        super().__init__()
        self.directory = Path(directory)
        #: The project's own import settings, when it recorded an axis order
        #: the user chose. Outranks what the manifest can tell.
        self.import_config = import_config

    def run(self) -> None:
        try:
            from ..store.project import load_analysis

            analysis = load_analysis(self.directory)
            _ = analysis.masks  # warm the cache here, not on the UI thread
            source = analysis.source_path
            if source is None or not Path(source).exists():
                raise FileNotFoundError(
                    "The original image could not be found:\n"
                    f"{source}\n\nThe result files are still available."
                )
            import_config = import_config_for(analysis.manifest, self.import_config)
            metadata = read_metadata_for(source, import_config)
            stack = imaging.load_stack(source, metadata)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(friendly_error(exc), traceback.format_exc())
        else:
            self.finished.emit(analysis, metadata, stack)


#: ``input.axes_source`` values meaning the order came from the user (the
#: axis question, or ``--axes``) rather than from the file's metadata.
USER_AXES_SOURCES = frozenset({"user", "user choice", "chosen by the user", "import_config", "cli"})


def import_config_for(
    manifest: dict[str, Any] | None, recorded: ImportConfig | None = None
) -> ImportConfig | None:
    """The import settings to read a saved run's source with again.

    Only an axis order the *user* chose is replayed, so reopening never asks
    the same question twice. ``input.axes`` on its own is not that: v2 writes
    it for every run, as the canonical order after a C axis was reduced
    (``TYX`` for a ``TCYX`` file), whereas ``ImportConfig.axes`` means the
    file's own order. Forcing the canonical order onto a multichannel or
    singleton-dimension file would make the importer refuse or misread it,
    and the results could not be opened. The order is replayed from, in turn:

    1. ``recorded`` -- the project's own ImportConfig, saved when the run
       started with the user's answer in it;
    2. the run's import block (``run.json["import"]`` or
       ``run.json["config"]["import"]``), the setting as the run saw it;
    3. ``input.axes`` only when ``input.axes_source`` says the user chose it.

    Otherwise only ``channel_index`` is passed and the metadata decides.
    """
    manifest = manifest or {}
    inp = manifest.get("input") or {}
    run_import = manifest.get("import") or (manifest.get("config") or {}).get("import") or {}

    axes = getattr(recorded, "axes", None) or run_import.get("axes")
    if not axes and str(inp.get("axes_source") or "").strip().lower() in USER_AXES_SOURCES:
        axes = inp.get("axes")

    channel = inp.get("channel_index")
    if channel is None:
        channel = run_import.get("channel_index")
    if channel is None and recorded is not None and recorded.axes:
        channel = recorded.channel_index
    if not axes and channel is None:
        return None
    try:
        channel_index = int(channel) if channel is not None else 0
    except (TypeError, ValueError):
        channel_index = 0
    return ImportConfig(axes=str(axes) if axes else None, channel_index=channel_index)


class ExportWorker(QObject):
    """Writes one export of a finished analysis.

    ``kind`` is one of the ``EXPORT_*`` keys. Every writer lives in the store
    (``store.project``); this only chooses which, so a script and the window
    produce byte-identical files.
    """

    finished = Signal(object)  # list[Path]
    failed = Signal(str, str)

    def __init__(
        self,
        analysis,
        destination: str | Path,
        *,
        kind: str = EXPORT_BUNDLE,
        track_id: int | None = None,
        reference_point_px: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.analysis = analysis
        self.destination = Path(destination)
        self.kind = kind
        self.track_id = track_id
        self.reference_point_px = (
            tuple(float(v) for v in reference_point_px) if reference_point_px else None
        )

    def run(self) -> None:
        try:
            written = run_export(
                self.analysis,
                self.destination,
                kind=self.kind,
                track_id=self.track_id,
                reference_point_px=self.reference_point_px,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(friendly_error(exc), traceback.format_exc())
        else:
            self.finished.emit(written)


def _writer(name: str):
    from ..store import project  # noqa: PLC0415 - looked up at call time

    function = getattr(project, name, None)
    if function is None:
        raise RuntimeError(
            f"This build of Corridor cannot write this export yet (store.project.{name} "
            "is missing)."
        )
    return function


def _accepts(function, keyword: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    return keyword in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


def _as_paths(result: Any, fallback: Path) -> list[Path]:
    if result is None:
        return [fallback] if fallback.exists() else []
    if isinstance(result, (str, Path)):
        return [Path(result)]
    try:
        return [Path(p) for p in result]
    except TypeError:
        return [fallback]


def run_export(
    analysis,
    destination: Path,
    *,
    kind: str,
    track_id: int | None = None,
    reference_point_px: Sequence[float] | None = None,
) -> list[Path]:
    """Dispatch one export to the store's writer. Returns the files written."""
    destination = Path(destination)
    if kind in TRACK_EXPORTS:
        if track_id is None:
            raise ValueError("Select a track to export it.")
        fmt = "xlsx" if kind == EXPORT_TRACK_XLSX else "csv"
        result = _writer("export_track")(
            analysis, int(track_id), destination, fmt=fmt,
            reference_point_px=reference_point_px,
        )
        return _as_paths(result, destination)
    if kind == EXPORT_TRACKS_CSV:
        writer = _writer("export_tracks_csv")
        # D2R needs the reference point. The agreed signature is (saved, path),
        # so a writer that does not declare the keyword writes the file and
        # D2R is filled in afterwards, from the same rows.
        if reference_point_px is not None and _accepts(writer, "reference_point_px"):
            return _as_paths(
                writer(analysis, destination, reference_point_px=reference_point_px),
                destination,
            )
        written = _as_paths(writer(analysis, destination), destination)
        if reference_point_px is not None:
            for path in written:
                fill_reference_distance(path, reference_point_px, analysis)
        return written
    if kind == EXPORT_SUMMARIES_CSV:
        return _as_paths(_writer("export_summaries_csv")(analysis, destination), destination)
    if kind == EXPORT_MSD_CSV:
        return _as_paths(_writer("export_msd_csv")(analysis, destination), destination)
    if kind == EXPORT_BUNDLE:
        return _export_bundle(analysis, destination, reference_point_px)
    raise ValueError(f"Unknown export {kind!r}.")


def _export_bundle(
    analysis, destination: Path, reference_point_px: Sequence[float] | None
) -> list[Path]:
    """The full bundle, with the reference point the user set.

    The store's bundle copies the run's files, whose tracks.csv was written
    before any reference point existed, so its D2R column is empty. A bundle
    is what gets archived and shared, so it must carry the same D2R as the
    per-track exports and the point itself (``reference_point.json``) --
    otherwise the numbers cannot be reproduced from the bundle alone.
    """
    writer = _writer("export_bundle")
    source = _bundle_source(writer, analysis)
    if reference_point_px is not None and _accepts(writer, "reference_point_px"):
        written = _as_paths(
            writer(source, destination, reference_point_px=reference_point_px), destination
        )
    else:
        written = _as_paths(writer(source, destination), destination)
        if reference_point_px is not None:
            tracks = Path(destination) / getattr(pipeline, "F_TRACKS", "tracks.csv")
            fill_reference_distance(tracks, reference_point_px, analysis)
    if reference_point_px is not None and Path(destination).is_dir():
        sidecar = save_reference_point(destination, reference_point_px)
        if sidecar not in written:
            written.append(sidecar)
    return written


def fill_reference_distance(path: str | Path, point: Sequence[float], analysis) -> bool:
    """Write D2R into a tracks CSV that was written without it.

    Only the D2R column changes: every other cell is written back as the
    exact text that was read, so a script and the window still produce the
    same numbers in every other column. The column is appended when the
    file has none (a v1 run). Returns False when the file is not a tracks
    table (no ``x_px``/``y_px``) and was left alone.
    """
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".csv":
        return False
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if "x_px" not in columns or "y_px" not in columns:
        return False
    if REFERENCE_COLUMN not in columns:
        columns.append(REFERENCE_COLUMN)
    from ..core.export import _clean, atomic_write_text  # noqa: PLC0415 - same formatting as the store

    for row, distance in zip(rows, reference_distances(rows, point, analysis)):
        row[REFERENCE_COLUMN] = _clean(distance)
    buffer = io.StringIO()
    out = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    out.writeheader()
    out.writerows(rows)
    atomic_write_text(path, buffer.getvalue())
    return True


def reference_distances(
    rows: Sequence[dict[str, Any]], point: Sequence[float], analysis
) -> list[float | None]:
    """D2R per row, in µm, or None where it cannot be measured.

    The measurement package's ``add_reference_distance`` is used when it
    exists and the rows are 2-D, so these values are the ones its per-track
    export writes; its signature takes no Z step, so a 3-D run is measured
    here with the run's real Z step (:func:`analysis_view.reference_distance_um`).
    """
    pixel = finite(getattr(analysis, "pixel_size_um", None))
    z_step = z_step_um_of(analysis)
    three_d = any(row_z(r) is not None for r in rows)
    try:
        from ..core import measurements  # noqa: PLC0415

        helper = getattr(measurements, "add_reference_distance", None)
    except ImportError:  # pragma: no cover - the measurement package may move
        helper = None
    if helper is not None and not three_d and pixel is not None:
        typed = [{k: _typed(v) for k, v in r.items()} for r in rows]
        try:
            measured = list(helper(typed, (float(point[0]), float(point[1])), pixel))
        except Exception:  # noqa: BLE001 - fall back to the same formula below
            measured = []
        if len(measured) == len(rows):
            return [finite(r.get(REFERENCE_COLUMN)) for r in measured]
    return [reference_distance_um(r, point, pixel, z_step) for r in rows]


def _typed(value: Any) -> Any:
    """A CSV cell as the store would load it: number, text, or None."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _bundle_source(writer, analysis):
    """What ``export_bundle`` takes first: the analysis, or its directory.

    1.x took the SavedAnalysis; the 2.0 store may take the saved directory
    (``export_bundle(saved_dir, dest)``). The parameter's name decides.
    """
    try:
        first = next(iter(inspect.signature(writer).parameters))
    except (StopIteration, TypeError, ValueError):
        return analysis
    if first in ("saved_dir", "directory", "source_dir", "src", "path"):
        return Path(getattr(analysis, "directory", analysis))
    return analysis


class Job:
    """Owns a QObject worker and the thread it runs on.

    Every signal a worker emits is connected with an explicit
    ``Qt.QueuedConnection``.  Qt's default AutoConnection decides between a
    direct and a queued call from the *receiver's* thread affinity, and a plain
    Python callable has none -- so connecting a lambda to a worker signal runs
    it on the worker thread.  For slots that touch widgets that is undefined
    behaviour, so the choice is made explicit here rather than left to luck.
    """

    #: Signals after which a worker has nothing more to say.
    TERMINAL = ("finished", "failed", "cancelled_signal", "ambiguous")

    def __init__(self, worker: QObject) -> None:
        self.worker = worker
        self.thread = QThread()
        worker.moveToThread(self.thread)
        self.thread.started.connect(worker.run)  # type: ignore[attr-defined]
        for name in self.TERMINAL:
            signal = getattr(worker, name, None)
            if signal is not None:
                signal.connect(self._stop, Qt.QueuedConnection)

    def start(self) -> None:
        self.thread.start()

    def _stop(self, *_args: Any) -> None:
        self.thread.quit()

    def wait(self, milliseconds: int = 8000) -> None:
        self.thread.quit()
        self.thread.wait(milliseconds)

    @property
    def running(self) -> bool:
        return self.thread.isRunning()


def friendly_error(exc: BaseException) -> str:
    """Turn an exception into something a researcher can act on.

    A traceback tells a user nothing they can use. What they need is which of
    their inputs is wrong, and what to do about it.
    """
    from ..core.model_registry import ModelUnavailable

    if isinstance(exc, ModelUnavailable):
        # Verbatim: the contract's sentence first, then every path tried.
        # Never a suggestion to use another model.
        return str(exc)
    if is_ambiguous_axes(exc):
        return str(getattr(exc, "message", None) or exc)
    if isinstance(exc, imaging.UnsupportedStackError):
        return str(exc)
    try:
        from ..core.segmentation import CellposeUnavailableError  # noqa: PLC0415
    except ImportError:  # pragma: no cover - the segmentation package may rename it
        CellposeUnavailableError = ()  # type: ignore[assignment]
    if CellposeUnavailableError and isinstance(exc, CellposeUnavailableError):
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        missing = getattr(exc, "filename", None) or str(exc)
        return f"This file could not be found:\n{missing}"
    if isinstance(exc, PermissionError):
        target = getattr(exc, "filename", None) or ""
        return (
            "Corridor is not allowed to write here"
            + (f":\n{target}" if target else ".")
            + "\n\nChoose a different output folder, or close the file if it is open "
            "in another program."
        )
    if isinstance(exc, MemoryError):
        return (
            "There was not enough memory to analyse this stack.\n\n"
            "Try a smaller crop, or close other applications."
        )
    if isinstance(exc, OSError) and getattr(exc, "winerror", None) == 112:
        return "The disk is full. Free some space and try again."
    message = str(exc).strip()
    return message or f"Something went wrong ({type(exc).__name__})."


class UpdateWorker(QObject):
    """Asks once whether a newer release exists, off the UI thread.

    A network call on the interface thread freezes the window for as long as
    the socket takes, which on a captive-portal wifi is however long the
    timeout is. Nobody should watch Corridor hang because a hotel router
    swallowed a packet.

    It never fails loudly: no network, a proxy, a rate limit and a malformed
    reply all arrive as ``finished(None)``.
    """

    finished = Signal(object)  # Release | None

    def run(self) -> None:
        from ..core import updates

        try:
            self.finished.emit(updates.check())
        except Exception:  # noqa: BLE001 - an update check may never break a run
            self.finished.emit(None)
