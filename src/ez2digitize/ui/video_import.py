# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Qt adapter for video import: runs it on a worker thread, reports by signals."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from ez2digitize.backends.ffmpeg import FFmpeg
from ez2digitize.core.capture import CaptureError
from ez2digitize.core.project import Project
from ez2digitize.core.runner import CancelToken, Event, Progress
from ez2digitize.video import VideoImportCancelled, import_video

# Progress is coalesced to at most one signal per this many seconds.
FLUSH_S = 0.1


class _Worker(QThread):
    def __init__(
        self,
        importer: VideoImporter,
        project: Project,
        video: Path,
        ffmpeg: FFmpeg,
        frames: int,
        cancel: CancelToken,
    ) -> None:
        super().__init__()
        self.importer = importer
        self.project = project
        self.video = video
        self.ffmpeg = ffmpeg
        self.frames = frames
        self.cancel = cancel
        self._last = 0.0

    def run(self) -> None:
        r = self.importer
        try:
            bundle = import_video(
                self.project,
                self.video,
                self.ffmpeg,
                frames=self.frames,
                on_event=self._on_event,
                cancel=self.cancel,
            )
        except VideoImportCancelled:
            r.cancelled.emit()
        except CaptureError as exc:
            r.failed.emit(str(exc))
        except Exception as exc:  # a bug, a full disk...: report it, don't kill the thread silently
            r.failed.emit(f"unexpected error: {exc!r}")
        else:
            r.succeeded.emit(bundle)

    def _on_event(self, event: Event) -> None:
        if not isinstance(event, Progress) or event.fraction is None:
            return
        now = time.monotonic()
        if now - self._last >= FLUSH_S or event.fraction >= 1.0:
            self._last = now
            self.importer.progress.emit(event.message, event.fraction)


class VideoImporter(QObject):
    """Imports one video at a time; `succeeded`, `failed` or `cancelled` ends it."""

    running_changed = Signal(bool)
    progress = Signal(str, float)  # phase, fraction
    succeeded = Signal(object)  # CaptureBundle
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._worker: _Worker | None = None
        self._cancel: CancelToken | None = None

    @property
    def running(self) -> bool:
        return self._worker is not None

    def start(self, project: Project, video: Path, ffmpeg: FFmpeg, frames: int) -> None:
        if self._worker is not None:
            raise RuntimeError("a video is already being imported")
        self._cancel = CancelToken()
        self._worker = _Worker(self, project, video, ffmpeg, frames, self._cancel)
        self._worker.finished.connect(self._on_thread_finished)
        self.running_changed.emit(True)
        self._worker.start()

    def cancel(self) -> None:
        if self._cancel is not None:
            self._cancel.cancel()

    def wait(self, timeout_ms: int = 30_000) -> bool:
        worker = self._worker
        return worker is None or worker.wait(timeout_ms)

    def _on_thread_finished(self) -> None:
        worker, self._worker, self._cancel = self._worker, None, None
        if worker is not None:
            worker.deleteLater()
        self.running_changed.emit(False)
