# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""GUI entry point."""

import json
import sys
from pathlib import Path

from PySide6.QtCore import QTemporaryDir, QTimer
from PySide6.QtWidgets import QApplication

from ez2digitize import __version__
from ez2digitize.ui import viewer
from ez2digitize.ui.main_window import MainWindow


def _heif_works() -> bool:
    """Whether HEIC decoding loads (its native library must be in the package)."""
    try:
        from ez2digitize.core import heic

        heic._register()
    except (ImportError, OSError):
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    args = sys.argv if argv is None else argv
    self_test = "--self-test" in args[1:]
    viewer.prepare()  # QtWebEngine: before the QApplication exists
    app = QApplication([a for a in args if a != "--self-test"])
    app.setOrganizationName("EZ2DIGITIZE")
    app.setApplicationName("EZ2DIGITIZE")
    app.setApplicationVersion(__version__)
    window = MainWindow()
    window.show()
    if self_test:
        # For packaging checks: the window came up, and the viewer's page
        # loads (QtWebEngine starts in the bundle); report and quit.
        info: dict[str, object] = {
            "self-test": "gui",
            "version": __version__,
            "visible": window.isVisible(),
            "heif": _heif_works(),
            "viewer": False,
        }

        def report() -> None:
            print(json.dumps(info), flush=True)
            app.quit()

        if viewer.usable():
            scratch = QTemporaryDir()
            probe = viewer.ViewerWidget(Path(scratch.path()))

            def page_ready(event: dict[str, object]) -> None:
                info["viewer"] = True  # the page loaded; drawing needs WebGL too
                info["webgl"] = event.get("gpu") if event.get("webgl") else False
                report()

            probe.ready.connect(page_ready)
            QTimer.singleShot(60_000, report)
        else:
            QTimer.singleShot(0, report)
    return app.exec()
