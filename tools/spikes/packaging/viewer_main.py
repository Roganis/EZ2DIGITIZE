# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Packaging spike, variant "viewer": the web viewer spike, packaged.

ez2digitize-viewer MODEL.ply [--measure]   the viewer spike (see its README)
ez2digitize-viewer --self-test             start QtWebEngine, print a JSON report, exit
"""

from __future__ import annotations

import sys

import bundle  # first, so its start time is close to process start
import viewer_spike


def self_test() -> int:
    from PySide6.QtCore import QTimer
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication

    viewer_spike.register_scheme()
    app = QApplication(sys.argv[:1])
    view = QWebEngineView()
    view.setHtml("<p>EZ2DIGITIZE packaging self-test</p>")
    view.show()
    result = {"web_ok": False}

    def loaded(ok: bool) -> None:
        result["web_ok"] = ok
        bundle.print_report("viewer", {"web_engine_page_loaded": ok,
                                       "web_assets": viewer_spike.WEB_DIR.is_dir()})  # fmt: skip
        app.quit()

    view.loadFinished.connect(loaded)
    QTimer.singleShot(60_000, app.quit)
    app.exec()
    return 0 if result["web_ok"] else 1


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    return viewer_spike.main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
