# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The "Add photos from phone" dialog: a QR code and the files arriving."""

from __future__ import annotations

import io

import segno
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ez2digitize.core.capture import CaptureError
from ez2digitize.core.project import Project
from ez2digitize.upload import Received, UploadSession

POLL_MS = 500


def qr_pixmap(text: str, scale: int = 6) -> QPixmap:
    buffer = io.BytesIO()
    segno.make(text, error="m").save(buffer, kind="png", scale=scale, border=2)
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue())  # PNG, detected by Qt
    return pixmap


class PhoneUploadDialog(QDialog):
    """Serves the upload page while open; Import makes a capture bundle of the photos.

    Videos come back in `received.videos`, for the project page to import
    as frames.
    """

    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add photos from phone")
        self.session = UploadSession(project)
        self.received: Received | None = None
        url = self.session.start()

        qr = QLabel()
        qr.setPixmap(qr_pixmap(url))
        qr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        steps = QLabel(
            "1. Connect the phone to the same Wi-Fi network as this computer.\n"
            "2. Scan this code with the phone's camera and open the page.\n"
            "3. Choose the photos, then tap “I'm done”.\n\n"
            "Photos arrive unchanged. Only phones on this network can reach the page, "
            "and only while this window is open."
        )
        steps.setWordWrap(True)
        self.address = QLabel(f"Or type: {url}")
        self.address.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if self.session.host.startswith("127."):
            self.address.setText(
                "This computer doesn't seem to be on a network; the phone can't reach it."
            )

        self.files = QTreeWidget()
        self.files.setHeaderLabels(["Photo", "Received"])
        self.files.setRootIsDecorated(False)
        self.files.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.summary = QLabel("Waiting for the phone…")

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.import_button = QPushButton("Import")
        self.import_button.setEnabled(False)
        self.import_button.setDefault(True)
        self.buttons.addButton(self.import_button, QDialogButtonBox.ButtonRole.AcceptRole)
        self.buttons.accepted.connect(self.import_files)
        self.buttons.rejected.connect(self.reject)

        top = QHBoxLayout()
        top.addWidget(qr)
        top.addWidget(steps, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.address)
        layout.addWidget(self.files, 1)
        layout.addWidget(self.summary)
        layout.addWidget(self.buttons)
        self.resize(620, 560)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(POLL_MS)

    def refresh(self) -> None:
        status = self.session.status()
        while self.files.topLevelItemCount() < len(status):
            self.files.addTopLevelItem(QTreeWidgetItem(["", ""]))
        complete = 0
        for i, upload in enumerate(status):
            item = self.files.topLevelItem(i)
            if item is None:
                continue
            item.setText(0, upload.name)
            item.setText(1, f"{100 * upload.received // upload.size} %")
            complete += upload.complete
        waiting = len(status) - complete
        if not status:
            text = "Waiting for the phone…"
        elif waiting:
            text = f"{complete} of {len(status)} received…"
        else:
            text = f"{complete} received."
        if self.session.phone_done and not waiting:
            text += " The phone has finished."
        self.summary.setText(text)
        self.import_button.setEnabled(complete > 0)
        self.import_button.setText(f"Import {complete}" if complete else "Import")

    def import_files(self) -> None:
        self.timer.stop()
        waiting = [f for f in self.session.status() if not f.complete]
        if waiting:
            answer = QMessageBox.question(
                self,
                "Still receiving",
                f"{len(waiting)} file(s) haven't fully arrived and will be left out. "
                "Import the others now?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.timer.start(POLL_MS)
                return
        try:
            self.received = self.session.finish()
        except CaptureError as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
            self.reject()
            return
        self.accept()

    def reject(self) -> None:
        self.timer.stop()
        self.session.close()
        super().reject()

    def done(self, result: int) -> None:
        # Closing the window any way stops the server.
        if result != QDialog.DialogCode.Accepted:
            self.session.close()
        super().done(result)
