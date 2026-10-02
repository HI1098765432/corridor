"""Background work.

Segmentation takes seconds per frame on a CPU, so it cannot run on the UI
thread.  It runs in a QThread, reports progress through signals, and checks a
cancellation flag between frames -- the only point where stopping is safe and
leaves the partial result consistent.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, Signal

from ..core import pipeline
from ..core.config import RunConfig
from ..core.imaging import StackMetadata, read_metadata


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
    """

    finished = Signal(object, object)  # StackMetadata, np.ndarray
    failed = Signal(str, str)

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)

    def run(self) -> None:
        try:
            from ..core.imaging import load_stack

            metadata = read_metadata(self.path)
            refuse_z_stack_for_display(metadata)
            stack = load_stack(self.path, metadata)
        except Exception as exc:  # noqa: BLE001
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
            from ..core.imaging import load_stack

            from ..store.project import load_analysis

            analysis = load_analysis(self.directory)
            _ = analysis.masks  # warm the cache here, not on the UI thread
            source = analysis.source_path
            if source is None or not Path(source).exists():
                raise FileNotFoundError(
                    "The original image could not be found:\n"
                    f"{source}\n\nThe result files are still available."
                )
            from ..core.imaging import saved_import_config

            # Re-read the image exactly as the saved run read it: a v1 run read
            # Z/Q/I planes as time, and 2.0 would otherwise refuse the file or
            # return a Z stack its saved (T, Y, X) masks do not fit.
            metadata = read_metadata(
                source, saved_import_config(source, analysis.manifest.get("input"))
            )
            refuse_z_stack_for_display(metadata)
            stack = load_stack(source, metadata)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(friendly_error(exc), traceback.format_exc())
        else:
            self.finished.emit(analysis, metadata, stack)


class ExportWorker(QObject):
    """Copies a finished analysis into a folder the user chose."""

    finished = Signal(object)  # list[Path]
    failed = Signal(str, str)

    def __init__(self, analysis, destination: str | Path) -> None:
        super().__init__()
        self.analysis = analysis
        self.destination = Path(destination)

    def run(self) -> None:
        try:
            from ..store.project import export_bundle

            written = export_bundle(self.analysis, self.destination)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(friendly_error(exc), traceback.format_exc())
        else:
            self.finished.emit(written)


class Job:
    """Owns a QObject worker and the thread it runs on.

    Every signal a worker emits is connected with an explicit
    ``Qt.QueuedConnection``.  Qt's default AutoConnection decides between a
    direct and a queued call from the *receiver's* thread affinity, and a plain
    Python callable has none -- so connecting a lambda to a worker signal runs
    it on the worker thread.  For slots that touch widgets that is undefined
    behaviour, so the choice is made explicit here rather than left to luck.
    """

    def __init__(self, worker: QObject) -> None:
        self.worker = worker
        self.thread = QThread()
        worker.moveToThread(self.thread)
        self.thread.started.connect(worker.run)  # type: ignore[attr-defined]
        for name in ("finished", "failed", "cancelled_signal"):
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


def refuse_z_stack_for_display(metadata: StackMetadata) -> None:
    """Refuse a Z stack before its pixels reach a (T, Y, X)-only canvas.

    Transitional (work package D): ``load_stack`` returns ``(T, Z, Y, X)``
    for a Z stack, and ``ImageCanvas._frame_pixmap`` unpacks ``h, w`` from one
    frame, so a Z stack raised inside the paint path. Remove this once the
    canvas has a Z slider; until then the refusal names the route that works.
    """
    from ..core.imaging import UnsupportedStackError

    if metadata.dimensionality == "3D":
        raise UnsupportedStackError(
            f"{metadata.path.name} is a {metadata.axes_interpretation}. The image view "
            "cannot show Z stacks yet. If its planes are really time points, analyse it "
            "from the command line with --axes TYX."
        )


def friendly_error(exc: BaseException) -> str:
    """Turn an exception into something a researcher can act on.

    A traceback tells a user nothing they can use. What they need is which of
    their inputs is wrong, and what to do about it.
    """
    from ..core.imaging import UnsupportedStackError
    from ..core.segmentation import CellposeUnavailableError

    if isinstance(exc, UnsupportedStackError):
        return str(exc)
    if isinstance(exc, CellposeUnavailableError):
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
