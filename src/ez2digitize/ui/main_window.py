# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QWidget

from ez2digitize import __version__

ABOUT_TEXT = f"""<h3>EZ2DIGITIZE {__version__}</h3>
<p>Turn photos or video of small objects into textured meshes and
Gaussian splats.</p>
<p>This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version. It comes with ABSOLUTELY NO WARRANTY.</p>
<p>Third-party components and their licenses are listed in
THIRD_PARTY_LICENSES.</p>"""


class MainWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("EZ2DIGITIZE")
        self.resize(1024, 700)

        placeholder = QLabel("No project open")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(placeholder)

        file_menu = self.menuBar().addMenu("&File")
        quit_action = file_menu.addAction("&Quit")
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = help_menu.addAction("&About EZ2DIGITIZE")
        about_action.triggered.connect(self.show_about)

    def show_about(self) -> None:
        QMessageBox.about(self, "About EZ2DIGITIZE", ABOUT_TEXT)
