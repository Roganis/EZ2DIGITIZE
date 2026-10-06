# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The masks panel: make automatic masks, review them, import masks.

A grid of thumbnails shows every photo with what its mask removes tinted
red. Unchecking a photo drops its mask (the photo is then used whole);
"To look at" shows the masks the checks flagged. Masks are made by
`MaskMaker` on a worker thread (download of the model the first time, then
the masking stage per capture); thumbnails are drawn by `_Thumbnails`, also
off the GUI thread.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ez2digitize import masks
from ez2digitize.core.capture import CaptureBundle, CaptureError, list_bundles
from ez2digitize.core.project import Project
from ez2digitize.core.runner import CancelToken, Event, Progress
from ez2digitize.masks import MaskEntry

THUMB = QSize(160, 120)  # the most a thumbnail takes
FLUSH_S = 0.1
KEY_ROLE = Qt.ItemDataRole.UserRole  # (capture id, file name)

STATE_TEXT = {
    "automatic": "",
    "imported": "imported",
    "dropped": "not used",
    "none": "no mask",
}
FILTERS = (("All photos", "all"), ("To look at", "flagged"), ("Not used", "dropped"))

# Threads outlive the panel that started them (see photo_checks._inspectors).
_threads: list[QThread] = []


def _keep(thread: QThread) -> None:
    _threads[:] = [t for t in _threads if t.isRunning()]
    _threads.append(thread)


class _MakeWorker(QThread):
    def __init__(self, maker: MaskMaker, project: Project, cancel: CancelToken) -> None:
        super().__init__()
        self.maker = maker
        self.project = project
        self.cancel = cancel
        self._last = 0.0

    def run(self) -> None:
        m = self.maker
        try:
            model = masks.find_model()
            if model is None:
                model = masks.download_model(on_progress=self._on_download, cancel=self.cancel)
            runs = masks.make_masks(
                self.project, list_bundles(self.project), model,
                on_event=self._on_event, cancel=self.cancel,
            )  # fmt: skip
        except masks.MaskingCancelled:
            m.cancelled.emit()
        except (masks.MaskingError, CaptureError, OSError) as exc:
            m.failed.emit(str(exc))
        except Exception as exc:  # a bug: report it, don't kill the thread silently
            m.failed.emit(f"unexpected error: {exc!r}")
        else:
            m.succeeded.emit(runs)

    def _emit(self, message: str, fraction: float) -> None:
        now = time.monotonic()
        if now - self._last >= FLUSH_S or fraction >= 1.0:
            self._last = now
            self.maker.progress.emit(message, fraction)

    def _on_download(self, received: int, total: int) -> None:
        mb = masks.MODEL.size / 1e6
        self._emit(f"Downloading the masking model ({mb:.0f} MB)", received / max(total, 1))

    def _on_event(self, event: Event) -> None:
        if isinstance(event, Progress) and event.fraction is not None:
            self._emit(event.message[:1].upper() + event.message[1:], event.fraction)


class MaskMaker(QObject):
    """Makes the project's automatic masks; `succeeded`, `failed` or `cancelled` ends it."""

    running_changed = Signal(bool)
    progress = Signal(str, float)
    succeeded = Signal(object)  # list[masks.MaskRun]
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._worker: _MakeWorker | None = None
        self._cancel: CancelToken | None = None

    @property
    def running(self) -> bool:
        return self._worker is not None

    def start(self, project: Project) -> None:
        if self._worker is not None:
            raise RuntimeError("masks are already being made")
        self._cancel = CancelToken()
        self._worker = _MakeWorker(self, project, self._cancel)
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


class _Thumbnails(QThread):
    """Draws previews (as JPEG bytes: QPixmaps can only be made on the GUI thread)."""

    ready = Signal(object, bytes)  # key, JPEG

    def __init__(self, jobs: list[tuple[tuple[str, str], Path, Path | None]]) -> None:
        super().__init__()
        self.jobs = jobs
        self.stopped = False

    def run(self) -> None:
        for key, image, mask in self.jobs:
            if self.stopped:
                return
            try:
                thumb = masks.preview(image, mask, (THUMB.width(), THUMB.height()))
            except OSError:
                continue
            buffer = io.BytesIO()
            thumb.save(buffer, "JPEG", quality=85)
            self.ready.emit(key, buffer.getvalue())


class MasksPanel(QWidget):
    """Shows every photo's mask; makes, drops, restores and imports masks."""

    # Masks were made, dropped, restored, imported or removed.
    masks_changed = Signal()
    # Making masks started or ended (the page disables runs meanwhile).
    busy_changed = Signal(bool)

    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.maker = MaskMaker(self)
        self.entries: list[MaskEntry] = []
        self._locked = False
        self._filling = False
        self._thumbs: dict[tuple[str, str], QIcon] = {}
        self._thumb_sources: dict[tuple[str, str], tuple[Path, Path | None, float]] = {}
        self._loader: _Thumbnails | None = None

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.make_button = QPushButton("Make masks")
        self.make_button.setToolTip(
            "Separate the object from the background in every photo (about a second per "
            "photo). The first time, the model (179 MB) is downloaded."
        )
        self.make_button.clicked.connect(self.make_or_cancel)
        self.import_button = QPushButton("Add masks…")
        self.import_button.setToolTip(
            "Import a folder of masks made elsewhere: one PNG per photo, named like the "
            "photo plus .png, white for the object"
        )
        self.import_button.clicked.connect(self.choose_masks_to_import)
        self.clear_button = QPushButton("Remove automatic masks")
        self.clear_button.clicked.connect(self.clear_automatic)
        self.filter = QComboBox()
        for label, value in FILTERS:
            self.filter.addItem(label, value)
        self.filter.currentIndexChanged.connect(lambda _i: self._fill())
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setVisible(False)

        self.grid = QListWidget()
        self.grid.setViewMode(QListView.ViewMode.IconMode)
        self.grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.grid.setMovement(QListView.Movement.Static)
        self.grid.setIconSize(THUMB)
        # Room for the check box, and three lines of text under the thumbnail.
        self.grid.setGridSize(QSize(THUMB.width() + 28, THUMB.height() + 56))
        self.grid.setUniformItemSizes(True)
        self.grid.setWordWrap(True)
        self.grid.itemChanged.connect(self._on_item_changed)

        buttons = QHBoxLayout()
        buttons.addWidget(self.make_button)
        buttons.addWidget(self.import_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)
        buttons.addWidget(QLabel("Show:"))
        buttons.addWidget(self.filter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.summary)
        layout.addLayout(buttons)
        layout.addWidget(self.progress)
        layout.addWidget(self.grid, 1)

        self.maker.running_changed.connect(self._on_running_changed)
        self.maker.progress.connect(self._on_progress)
        self.maker.succeeded.connect(self._on_succeeded)
        self.maker.failed.connect(self._on_failed)
        self.maker.cancelled.connect(self._on_cancelled)

    # --- state ------------------------------------------------------------------

    def set_locked(self, locked: bool) -> None:
        """No changes while a reconstruction reads the masks."""
        self._locked = locked
        self._update_buttons()

    def refresh(self) -> None:
        bundles = list_bundles(self.project)
        self.entries = masks.review(self.project, bundles)
        self._fill()
        self._show_summary()
        self._update_buttons()

    def wait(self, timeout_ms: int = 30_000) -> bool:
        """Block until making masks and drawing thumbnails end (for shutdown)."""
        loader = self._loader
        if loader is not None:
            loader.stopped = True
            loader.wait(timeout_ms)
        return self.maker.wait(timeout_ms)

    def _update_buttons(self) -> None:
        making = self.maker.running
        has_photos = bool(self.entries)
        self.make_button.setText("Cancel" if making else "Make masks")
        self.make_button.setEnabled(making or (has_photos and not self._locked))
        idle = not making and not self._locked
        self.import_button.setEnabled(idle and has_photos)
        automatic = any(e.state == "automatic" or e.stats for e in self.entries)
        self.clear_button.setEnabled(idle and automatic)
        self.grid.setEnabled(idle)

    def _show_summary(self) -> None:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.state] = counts.get(entry.state, 0) + 1
        flagged = sum(1 for e in self.entries if e.flags and e.state != "none")
        if not self.entries:
            text = "Import photos to make masks."
        elif counts.get("none") == len(self.entries):
            text = (
                "Masks keep only the object: the background (and the table) is left out of "
                "the reconstruction, which is faster and cleaner, and needed for turntables "
                "and for joining both sides of an object. Make masks to start."
            )
        else:
            used = counts.get("automatic", 0) + counts.get("imported", 0)
            text = f"{used} of {len(self.entries)} photos have a mask"
            if counts.get("dropped"):
                text += f", {counts['dropped']} not used"
            text += "."
            if flagged:
                text += f" {flagged} to look at (Show: To look at)."
            text += " Uncheck a photo to use it whole, without its mask."
        self.summary.setText(text)

    # --- grid -------------------------------------------------------------------

    def _visible(self) -> list[MaskEntry]:
        mode = self.filter.currentData()
        if mode == "flagged":
            return [e for e in self.entries if e.flags and e.state != "none"]
        if mode == "dropped":
            return [e for e in self.entries if e.state == "dropped"]
        return self.entries

    def _fill(self) -> None:
        self._filling = True
        self.grid.clear()
        jobs: list[tuple[tuple[str, str], Path, Path | None]] = []
        placeholder = QPixmap(THUMB.width(), THUMB.width() * 2 // 3)
        placeholder.fill(Qt.GlobalColor.darkGray)
        for entry in self._visible():
            key = (entry.capture, entry.name)
            lines = [entry.name]
            state = STATE_TEXT[entry.state]
            if entry.excluded:
                state = ", ".join(s for s in (state, "photo left out") if s)
            if state:
                lines.append(f"({state})")
            if entry.flags and entry.state != "none":
                lines.append("⚠ " + ", ".join(entry.flags))
            item = QListWidgetItem("\n".join(lines))
            item.setData(KEY_ROLE, key)
            tip = f"{entry.capture}/{entry.name}"
            if entry.stats:
                tip += f"\nobject: {entry.stats.get('coverage', 0):.0%} of the photo"
            item.setToolTip(tip)
            if entry.state == "none":
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            else:
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                used = entry.state != "dropped"
                item.setCheckState(Qt.CheckState.Checked if used else Qt.CheckState.Unchecked)
            # Dropped masks are shown without the tint: the photo is used whole.
            shown_mask = entry.mask if entry.state != "dropped" else None
            source = (entry.image, shown_mask, _mtime(shown_mask))
            icon = self._thumbs.get(key) if self._thumb_sources.get(key) == source else None
            if icon is None:
                self._thumb_sources[key] = source
                jobs.append((key, entry.image, shown_mask))
                icon = QIcon(placeholder)
            item.setIcon(icon)
            self.grid.addItem(item)
        self._filling = False
        self._load_thumbnails(jobs)

    def _load_thumbnails(self, jobs: list[tuple[tuple[str, str], Path, Path | None]]) -> None:
        if self._loader is not None:
            self._loader.stopped = True
            self._loader = None
        if not jobs:
            return
        loader = _Thumbnails(jobs)
        loader.ready.connect(self._on_thumbnail)
        _keep(loader)
        self._loader = loader
        loader.start()

    def _on_thumbnail(self, key: tuple[str, str], data: bytes) -> None:
        image = QImage.fromData(data)
        if image.isNull():
            return
        icon = QIcon(QPixmap.fromImage(image))
        self._thumbs[key] = icon
        for row in range(self.grid.count()):
            item = self.grid.item(row)
            if item.data(KEY_ROLE) == key:
                # Setting the icon also emits itemChanged; it isn't a check.
                self._filling = True
                item.setIcon(icon)
                self._filling = False
                break

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        key = item.data(KEY_ROLE)
        if self._filling or not key:
            return
        capture, name = key
        entry = next((e for e in self.entries if (e.capture, e.name) == key), None)
        used = item.checkState() == Qt.CheckState.Checked
        bundle = _bundle(self.project, capture)
        if entry is None or bundle is None or used == (entry.state != "dropped"):
            return  # not a change of the check box
        try:
            (masks.restore if used else masks.drop)(self.project, bundle, [name])
        except (CaptureError, OSError) as exc:
            QMessageBox.warning(self, "Masks", str(exc))
        self.masks_changed.emit()
        # Not now: refreshing rebuilds the grid, deleting the item being changed.
        QTimer.singleShot(0, self.refresh)

    # --- actions ----------------------------------------------------------------

    def make_or_cancel(self) -> None:
        if self.maker.running:
            self.maker.cancel()
            self.make_button.setEnabled(False)
            self.summary.setText("Cancelling…")
        else:
            self.make_masks()

    def make_masks(self) -> None:
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.summary.setText("Making masks…")
        self.maker.start(self.project)

    def choose_masks_to_import(self) -> None:
        bundles = list_bundles(self.project)
        if not bundles:
            return
        bundle = bundles[0]
        if len(bundles) > 1:
            labels = [f"{b.id} ({len(b.images)} photos)" for b in bundles]
            choice, ok = QInputDialog.getItem(
                self, "Add masks", "Masks for which import?", labels, len(labels) - 1, False
            )
            if not ok:
                return
            bundle = bundles[labels.index(choice)]
        folder = QFileDialog.getExistingDirectory(self, "Folder of masks")
        if folder:
            self.import_masks(bundle, Path(folder))

    def import_masks(self, bundle: CaptureBundle, folder: Path) -> list[str]:
        try:
            names = masks.import_folder_masks(self.project, bundle, folder)
        except (CaptureError, OSError) as exc:
            QMessageBox.warning(self, "Add masks", str(exc))
            return []
        if not names:
            QMessageBox.information(
                self,
                "Add masks",
                f"No masks found in {folder.name} for these photos. Masks are PNG files "
                "named like the photo plus .png (IMG_0001.JPG.png) or with .png instead of "
                "its extension (IMG_0001.png).",
            )
        self.masks_changed.emit()
        self.refresh()
        return names

    def clear_automatic(self) -> None:
        answer = QMessageBox.question(
            self,
            "Remove automatic masks",
            "Remove the automatic masks? Imported masks stay. Making them again reuses "
            "the earlier results, as long as the photos are the same.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        for bundle in list_bundles(self.project):
            masks.clear_auto(self.project, bundle)
        self.masks_changed.emit()
        self.refresh()

    # --- maker signals ----------------------------------------------------------

    def _on_running_changed(self, running: bool) -> None:
        self._update_buttons()
        self.busy_changed.emit(running)

    def _on_progress(self, message: str, fraction: float) -> None:
        self.summary.setText(f"{message}…")
        self.progress.setValue(int(fraction * 1000))

    def _on_succeeded(self, runs: list[masks.MaskRun]) -> None:
        self.progress.setVisible(False)
        self.masks_changed.emit()
        self.refresh()
        dropped = sum(r.dropped for r in runs)
        if dropped:
            self.filter.setCurrentIndex(1)

    def _on_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.refresh()
        QMessageBox.warning(self, "Masks could not be made", message)

    def _on_cancelled(self) -> None:
        self.progress.setVisible(False)
        self.refresh()


def _bundle(project: Project, capture: str) -> CaptureBundle | None:
    return next((b for b in list_bundles(project) if b.id == capture), None)


def _mtime(path: Path | None) -> float:
    try:
        return path.stat().st_mtime if path is not None else 0.0
    except OSError:
        return 0.0
