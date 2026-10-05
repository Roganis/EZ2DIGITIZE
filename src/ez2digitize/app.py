# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""GUI entry point."""

import sys

from PySide6.QtWidgets import QApplication

from ez2digitize import __version__
from ez2digitize.ui.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    app.setOrganizationName("EZ2DIGITIZE")
    app.setApplicationName("EZ2DIGITIZE")
    app.setApplicationVersion(__version__)
    window = MainWindow()
    window.show()
    return app.exec()
