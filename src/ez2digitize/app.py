# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""GUI entry point."""

import json
import sys
from pathlib import Path

from PySide6.QtCore import QTemporaryDir, QTimer
from PySide6.QtWidgets import QApplication

from ez2digitize import __version__, models
from ez2digitize.ui import viewer
from ez2digitize.ui.main_window import MainWindow
from ez2digitize.ui.model_window import open_model


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
    # `--view FILE` (ez2d view): only the model's window, no project window.
    view_only = "--view" in args[1:]
    viewer.prepare()  # QtWebEngine: before the QApplication exists
    app = QApplication([a for a in args if a not in ("--self-test", "--view")])
    app.setOrganizationName("EZ2DIGITIZE")
    app.setApplicationName("EZ2DIGITIZE")
    app.setApplicationVersion(__version__)
    # Model files named on the command line (or opened with the app) open in their windows.
    files = [
        Path(a)
        for a in args[1:]
        if not a.startswith("-") and Path(a).suffix.lower() in models.OPENABLE
    ]
    if view_only:
        shown = [w for w in (open_model(f) for f in files) if w is not None]
        return app.exec() if shown else 1
    window = MainWindow()
    window.show()
    for path in files:
        window.open_model(path)
    if self_test:
        # For packaging checks: the window came up, and the viewer's page
        # loads (QtWebEngine starts in the bundle); report and quit, with exit
        # code 1 if any of it failed (all a program without a console can say).
        info: dict[str, object] = {
            "self-test": "gui",
            "version": __version__,
            "visible": window.isVisible(),
            "heif": _heif_works(),
            "viewer": False,
        }

        def report() -> None:
            print(json.dumps(info), flush=True)
            app.exit(0 if all(info[k] for k in ("visible", "heif", "viewer")) else 1)

        if viewer.usable():
            scratch = QTemporaryDir()
            probe = viewer.ViewerWidget(Path(scratch.path()))

            def page_ready(event: dict[str, object]) -> None:
                info["viewer"] = True  # the page loaded; drawing needs WebGL too
                info["webgl"] = event.get("gpu") if event.get("webgl") else False
                QTimer.singleShot(0, report)  # not from inside the page's callback

            probe.ready.connect(page_ready)
            QTimer.singleShot(60_000, report)
        else:
            QTimer.singleShot(0, report)
    return app.exec()
