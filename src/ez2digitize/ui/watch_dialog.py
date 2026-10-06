# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The "From a synced folder" dialog: photos arriving in a folder the phone syncs to.

See ez2digitize.watch. The dialog polls the folder each second, lists the
new photos as they arrive and says when the set looks complete; Import
makes a capture bundle of them. The folder is remembered for next time.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ez2digitize.core.capture import CaptureBundle, CaptureError
from ez2digitize.core.project import Project
from ez2digitize.watch import FolderWatch, import_ready

FOLDER_KEY = "watch/folder"
POLL_MS = 1000
# Which photos already in the folder count too: label, how far back (seconds; 0: none).
SINCE = (
    ("Only photos that arrive from now", 0),
    ("Also those from the last 30 minutes", 30 * 60),
    ("Also those from the last 2 hours", 2 * 3600),
    ("Also those from today", -1),  # since midnight
)


class WatchFolderDialog(QDialog):
    def __init__(
        self,
        project: Project,
        settings: QSettings | None = None,
        parent: QWidget | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add photos from a synced folder")
        self.project = project
        self.settings = settings or QSettings()
        self.clock = clock
        self.watch: FolderWatch | None = None
        self.bundle: CaptureBundle | None = None

        steps = QLabel(
            "For a phone that already syncs its photos to this computer (Syncthing, iCloud "
            "Drive, Google Drive, Dropbox…). Choose the folder they arrive in, take the "
            "photos, and import them once they are all here."
        )
        steps.setWordWrap(True)
        self.folder = QLineEdit(str(self.settings.value(FOLDER_KEY, "") or ""))
        self.folder.setPlaceholderText("The folder the phone's photos sync to")
        self.folder.editingFinished.connect(self._restart)
        choose = QPushButton("Choose…")
        choose.clicked.connect(self.choose_folder)
        self.since = QComboBox()
        for label, seconds in SINCE:
            self.since.addItem(label, seconds)
        self.since.currentIndexChanged.connect(lambda _i: self._restart())
        self.files = QListWidget()
        self.summary = QLabel()
        self.summary.setWordWrap(True)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.import_button = QPushButton("Import")
        self.import_button.setEnabled(False)
        self.import_button.setDefault(True)
        self.buttons.addButton(self.import_button, QDialogButtonBox.ButtonRole.AcceptRole)
        self.buttons.accepted.connect(self.import_files)
        self.buttons.rejected.connect(self.reject)

        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(choose)
        layout = QVBoxLayout(self)
        layout.addWidget(steps)
        layout.addLayout(folder_row)
        layout.addWidget(self.since)
        layout.addWidget(self.files, 1)
        layout.addWidget(self.summary)
        layout.addWidget(self.buttons)
        self.resize(560, 480)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self._restart()

    def choose_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "The folder the photos sync to")
        if chosen:
            self.set_folder(Path(chosen))

    def set_folder(self, folder: Path) -> None:
        self.folder.setText(str(folder))
        self._restart()

    def _since(self) -> float | None:
        seconds = int(self.since.currentData())
        now = self.clock()
        if seconds < 0:
            midnight = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0)
            return midnight.timestamp()
        return now - seconds if seconds else None

    def _restart(self) -> None:
        """Start watching the folder as it is now set (or say why not)."""
        self.timer.stop()
        self.watch = None
        self.files.clear()
        self.import_button.setEnabled(False)
        text = self.folder.text().strip()
        if not text:
            self.summary.setText("Choose the folder the phone's photos sync to.")
            return
        folder = Path(text).expanduser()
        try:
            self.watch = FolderWatch(folder, since=self._since(), clock=self.clock)
        except CaptureError as exc:
            self.summary.setText(f"Can't watch it: {exc}.")
            return
        self.settings.setValue(FOLDER_KEY, str(folder))
        self.refresh()
        self.timer.start(POLL_MS)

    def refresh(self) -> None:
        if self.watch is None:
            return
        state = self.watch.poll()
        names = [p.name for p in state.ready]
        if names != [self.files.item(i).text() for i in range(self.files.count())]:
            self.files.clear()
            self.files.addItems(names)
            self.files.scrollToBottom()
        self.summary.setText(state.describe())
        self.import_button.setEnabled(bool(state.ready))
        self.import_button.setText(f"Import {len(state.ready)}" if state.ready else "Import")

    def import_files(self) -> None:
        if self.watch is None:
            return
        self.timer.stop()
        state = self.watch.poll()
        if state.arriving:
            answer = QMessageBox.question(
                self,
                "Still arriving",
                f"{state.arriving} file(s) are still arriving and will be left out. "
                "Import the others now?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.timer.start(POLL_MS)
                return
        try:
            self.bundle = import_ready(self.project, state, self.watch.folder)
        except CaptureError as exc:
            QMessageBox.warning(self, "Nothing imported", str(exc))
            self.timer.start(POLL_MS)
            return
        self.accept()

    def done(self, result: int) -> None:
        self.timer.stop()
        super().done(result)
