# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Qt adapter for the pipeline: runs it on a worker thread, reports by signals.

The pipeline delivers plain Python events on the thread that runs it. This
adapter runs it on a QThread and re-emits the events as Qt signals, which Qt
queues to the receiver's (GUI) thread. Backend output can be thousands of
lines a second, so output lines are batched and progress is coalesced, at
most every FLUSH_S seconds, to keep the GUI responsive.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from ez2digitize.core.project import Project
from ez2digitize.core.runner import CancelToken, Output, Progress
from ez2digitize.pipeline import (
    MeshResult,
    MeshSettings,
    Notice,
    PipelineCancelled,
    PipelineError,
    PipelineEvent,
    SparseResult,
    SplatResult,
    StageFailed,
    StageFinished,
    StageOutput,
    StageStarted,
    Tools,
    run_mesh,
)

FLUSH_S = 0.1
TRACEBACK_LINES = 12

PipelineFunction = Callable[..., SparseResult | MeshResult | SplatResult]


@dataclass(frozen=True)
class Failure:
    """Why a run stopped, ready to show: message, log tail, full log if any.

    `trace`: for an unexpected error (a bug), where in the code it happened.
    """

    message: str
    tail: tuple[str, ...] = ()
    log: Path | None = None
    trace: tuple[str, ...] = ()


class _Worker(QThread):
    def __init__(
        self,
        runner: PipelineRunner,
        function: PipelineFunction,
        project: Project,
        tools: Tools,
        settings: MeshSettings,
        cancel: CancelToken,
    ) -> None:
        super().__init__()
        self.runner = runner
        self.function = function
        self.project = project
        self.tools = tools
        self.settings = settings
        self.cancel = cancel
        self._lines: list[str] = []
        self._progress: tuple[str, str, float] | None = None
        self._last_flush = 0.0

    def run(self) -> None:
        r = self.runner
        try:
            result = self.function(
                self.project, self.tools, self.settings, on_event=self._on_event, cancel=self.cancel
            )
        except PipelineCancelled:
            self._flush()
            r.cancelled.emit()
        except StageFailed as exc:
            self._flush()
            r.failed.emit(Failure(str(exc), tuple(exc.tail), exc.log))
        except PipelineError as exc:
            self._flush()
            r.failed.emit(Failure(str(exc)))
        except Exception as exc:  # a bug, a full disk...: report it, don't kill the thread silently
            self._flush()
            # Where it happened, for a bug report.
            where = tuple(traceback.format_exc().rstrip().splitlines()[-TRACEBACK_LINES:])
            r.failed.emit(Failure(f"unexpected error: {exc!r}", trace=where))
        else:
            self._flush()
            r.succeeded.emit(result)

    def _on_event(self, event: PipelineEvent) -> None:
        r = self.runner
        if isinstance(event, StageOutput):
            inner = event.event
            if isinstance(inner, Output):
                self._lines.append(inner.line)
            elif isinstance(inner, Progress):
                fraction = -1.0 if inner.fraction is None else inner.fraction
                self._progress = (event.stage, inner.message, fraction)
            if time.monotonic() - self._last_flush >= FLUSH_S:
                self._flush()
            return
        self._flush()
        if isinstance(event, StageStarted):
            r.stage_started.emit(event.stage, event.index, event.count)
        elif isinstance(event, StageFinished):
            m = event.manifest
            r.stage_finished.emit(event.stage, m.status, event.reused, m.wall_s)
        elif isinstance(event, Notice):
            r.notice.emit(event.message)

    def _flush(self) -> None:
        self._last_flush = time.monotonic()
        if self._lines:
            self.runner.output.emit(self._lines)
            self._lines = []
        if self._progress is not None:
            self.runner.progress.emit(*self._progress)
            self._progress = None


class PipelineRunner(QObject):
    """Runs one pipeline at a time and reports it through signals.

    `progress` carries a fraction of -1 when the stage doesn't know how far
    it is. Exactly one of `succeeded`, `failed` or `cancelled` ends a run,
    followed by `running_changed(False)`.
    """

    running_changed = Signal(bool)
    stage_started = Signal(str, int, int)  # stage, index (1-based), count
    progress = Signal(str, str, float)  # stage, message, fraction or -1
    output = Signal(list)  # lines of tool output
    notice = Signal(str)
    stage_finished = Signal(str, str, bool, float)  # stage, status, reused, seconds
    succeeded = Signal(object)  # SparseResult, MeshResult or SplatResult
    failed = Signal(object)  # Failure
    cancelled = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._worker: _Worker | None = None
        self._cancel: CancelToken | None = None

    @property
    def running(self) -> bool:
        return self._worker is not None

    def start(
        self,
        project: Project,
        tools: Tools,
        settings: MeshSettings,
        function: PipelineFunction = run_mesh,
    ) -> None:
        if self._worker is not None:
            raise RuntimeError("a pipeline is already running")
        self._cancel = CancelToken()
        self._worker = _Worker(self, function, project, tools, settings, self._cancel)
        self._worker.finished.connect(self._on_thread_finished)
        self.running_changed.emit(True)
        self._worker.start()

    def cancel(self) -> None:
        """Ask the running stage to stop; `cancelled` follows once it has."""
        if self._cancel is not None:
            self._cancel.cancel()

    def wait(self, timeout_ms: int = 30_000) -> bool:
        """Block until the worker thread ends (for shutdown); True if it did."""
        worker = self._worker
        return worker is None or worker.wait(timeout_ms)

    def _on_thread_finished(self) -> None:
        worker, self._worker, self._cancel = self._worker, None, None
        if worker is not None:
            worker.deleteLater()
        self.running_changed.emit(False)
