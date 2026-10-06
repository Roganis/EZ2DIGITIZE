# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Help -> Licenses: this program's license, third-party components, sources."""

from __future__ import annotations

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ez2digitize import licenses


class LicensesDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Licenses")
        self.summary = QLabel(licenses.summary())
        self.summary.setWordWrap(True)
        self.tabs = QTabWidget()
        for title, name in (
            ("Third-party components", licenses.THIRD_PARTY),
            ("EZ2DIGITIZE (GPL-3.0)", licenses.LICENSE),
        ):
            text = QPlainTextEdit(licenses.license_text(name) or f"{name} is missing.")
            text.setReadOnly(True)
            text.setFont(QFont("monospace"))
            self.tabs.addTab(text, title)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        folder = licenses.backends_dir()
        if folder is not None:
            open_folder = QPushButton("Open the backends' license files")
            open_folder.clicked.connect(
                lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder / "licenses")))
            )
            buttons.addButton(open_folder, QDialogButtonBox.ButtonRole.ActionRole)
        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(buttons)
        self.resize(760, 600)
