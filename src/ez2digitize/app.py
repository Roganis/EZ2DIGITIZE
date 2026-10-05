# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""GUI entry point."""

import json
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ez2digitize import __version__
from ez2digitize.ui.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    args = sys.argv if argv is None else argv
    self_test = "--self-test" in args[1:]
    app = QApplication([a for a in args if a != "--self-test"])
    app.setOrganizationName("EZ2DIGITIZE")
    app.setApplicationName("EZ2DIGITIZE")
    app.setApplicationVersion(__version__)
    window = MainWindow()
    window.show()
    if self_test:
        # For packaging checks: the window came up; report and quit.
        def report() -> None:
            info = {"self-test": "gui", "version": __version__, "visible": window.isVisible()}
            print(json.dumps(info), flush=True)
            app.quit()

        QTimer.singleShot(0, report)
    return app.exec()
