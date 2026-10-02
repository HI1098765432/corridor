"""Background work.

Segmentation takes seconds per frame on a CPU, so it cannot run on the UI
thread.  It runs in a QThread, reports progress through signals, and checks a
cancellation flag between frames -- the only point where stopping is safe and
leaves the partial result consistent.
"""

from __future__ import annotations

import inspect
import traceback
from pathlib import Path
from typing import Any, Sequence

from PySide6.QtCore import QObject, Qt, QThread, Signal

from ..core import imaging, pipeline
from ..core.config import ImportConfig, RunConfig

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

    def __init__(self, config: RunConfig) -> None:
        super().__init__()
        self.config = config
        self._cancelled = False

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            result = pipeline.run_analysis(self.config, _SignalProgress(self))
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

    def __init__(self, directory: str | Path) -> None:
        super().__init__()
        self.directory = Path(directory)

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
            metadata = read_metadata_for(source, import_config_for(analysis.manifest))
            stack = imaging.load_stack(source, metadata)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(friendly_error(exc), traceback.format_exc())
        else:
            self.finished.emit(analysis, metadata, stack)


def import_config_for(manifest: dict[str, Any] | None) -> ImportConfig | None:
    """The import settings a saved run used, so reopening never asks again.

    A run whose axis order had to be chosen recorded it (``input.axes``,
    contract §7); reading the source again without it would raise the same
    ambiguity the user already answered.
    """
    manifest = manifest or {}
    inp = manifest.get("input") or {}
    axes = inp.get("axes")
    channel = inp.get("channel_index")
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
        # D2R needs the reference point; the agreed signature is (saved, path),
        # so it is passed only to a writer that declares it can take it.
        if reference_point_px is not None and _accepts(writer, "reference_point_px"):
            result = writer(analysis, destination, reference_point_px=reference_point_px)
        else:
            result = writer(analysis, destination)
        return _as_paths(result, destination)
    if kind == EXPORT_SUMMARIES_CSV:
        return _as_paths(_writer("export_summaries_csv")(analysis, destination), destination)
    if kind == EXPORT_MSD_CSV:
        return _as_paths(_writer("export_msd_csv")(analysis, destination), destination)
    if kind == EXPORT_BUNDLE:
        writer = _writer("export_bundle")
        return _as_paths(writer(_bundle_source(writer, analysis), destination), destination)
    raise ValueError(f"Unknown export {kind!r}.")


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
