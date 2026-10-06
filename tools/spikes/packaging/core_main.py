# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Packaging spike, variant "core": the app window without the web viewer.

ez2digitize-core               open the main window
ez2digitize-core --self-test   show the window, print a JSON report, exit
"""

from __future__ import annotations

import sys

import bundle  # first, so its start time is close to process start
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ez2digitize.ui.main_window import MainWindow


def main() -> int:
    self_test = "--self-test" in sys.argv
    app = QApplication(sys.argv[:1])
    window = MainWindow()
    window.show()
    if self_test:

        def report() -> None:
            bundle.print_report("core")
            app.quit()

        QTimer.singleShot(0, report)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
