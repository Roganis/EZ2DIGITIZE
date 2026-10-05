# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Viewer spike: a web 3D viewer (three.js + Spark) inside QWebEngineView.

    uv run python tools/spikes/viewer/fetch_vendor.py      # once
    uv run python tools/spikes/viewer/viewer_spike.py MODEL.ply

MODEL can be a Gaussian splat (Brush export), a textured mesh (OpenMVS
scene_textured.ply) or a point cloud (COLMAP sparse.ply, OpenMVS
scene_dense.ply); the kind is detected from the PLY header.

The page and the model are served through a custom `ez2d://` URL scheme
rather than file:// URLs, which Chromium restricts for ES modules. Only the
web/ folder and the files explicitly registered (model and its texture) can
be fetched.

--measure prints load time and frame rate as JSON lines and exits;
--screenshot also saves an image of the window.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import time
from pathlib import Path

from PySide6.QtCore import QByteArray, QFile, QIODevice, QTimer, QUrl, QUrlQuery
from PySide6.QtWebEngineCore import (
    QWebEnginePage,
    QWebEngineProfile,
    QWebEngineUrlRequestJob,
    QWebEngineUrlScheme,
    QWebEngineUrlSchemeHandler,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication, QMainWindow

STARTED = time.monotonic()
if getattr(sys, "frozen", False):  # inside a PyInstaller bundle (packaging spike)
    WEB_DIR = Path(getattr(sys, "_MEIPASS", ".")) / "viewer_web"
else:
    WEB_DIR = Path(__file__).resolve().parent / "web"
SCHEME = b"ez2d"
MIME = {".js": b"text/javascript", ".html": b"text/html", ".ply": b"application/octet-stream"}


def ply_header(path: Path) -> str:
    with path.open("rb") as fh:
        head = fh.read(64 * 1024)
    end = head.find(b"end_header")
    if not head.startswith(b"ply") or end < 0:
        raise SystemExit(f"{path} is not a PLY file")
    return head[:end].decode("ascii", errors="replace")


def detect(path: Path) -> tuple[str, str | None]:
    """Return (kind, texture file name) for a PLY file."""
    header = ply_header(path)
    if re.search(r"property \w+ f_dc_0\b", header):
        return "splat", None
    faces = re.search(r"element face (\d+)", header)
    if faces and int(faces.group(1)) > 0:
        texture = re.search(r"comment TextureFile (\S+)", header)
        return "mesh", texture.group(1) if texture else None
    return "points", None


class FileSchemeHandler(QWebEngineUrlSchemeHandler):
    """Serves ez2d://app/<path> from web/ and ez2d://model/<name> from a whitelist."""

    def __init__(self, models: dict[str, Path]) -> None:
        super().__init__()
        self.models = models

    def requestStarted(self, job: QWebEngineUrlRequestJob) -> None:  # noqa: N802 - Qt API
        url = job.requestUrl()
        host, rel = url.host(), url.path().lstrip("/")
        if host == "app":
            path = (WEB_DIR / rel).resolve()
            if not path.is_relative_to(WEB_DIR) or not path.is_file():
                job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
                return
        elif host == "model" and rel in self.models:
            path = self.models[rel]
        else:
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return
        file = QFile(str(path), job)  # parented to the job, freed with it
        if not file.open(QIODevice.OpenModeFlag.ReadOnly):
            job.fail(QWebEngineUrlRequestJob.Error.RequestFailed)
            return
        guessed = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        mime = MIME.get(path.suffix.lower(), guessed.encode())
        job.reply(QByteArray(mime), file)


class ReportingPage(QWebEnginePage):
    """Forwards the page's "EZ2D {...}" console lines (and errors) to stdout."""

    def __init__(self, profile: QWebEngineProfile, on_event: object) -> None:
        super().__init__(profile)
        self.on_event = on_event

    def javaScriptConsoleMessage(  # noqa: N802 - Qt API
        self,
        level: QWebEnginePage.JavaScriptConsoleMessageLevel,
        message: str,
        line: int,
        source: str,
    ) -> None:
        if message.startswith("EZ2D "):
            event = json.loads(message[5:])
            event["t_s"] = round(time.monotonic() - STARTED, 2)
            print(json.dumps(event), flush=True)
            assert callable(self.on_event)
            self.on_event(event)
        elif level == QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            print(json.dumps({"event": "console-error", "message": message,
                              "source": f"{source}:{line}"}), flush=True)  # fmt: skip


def register_scheme() -> None:
    scheme = QWebEngineUrlScheme(SCHEME)
    scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
    scheme.setFlags(
        QWebEngineUrlScheme.Flag.SecureScheme
        | QWebEngineUrlScheme.Flag.LocalAccessAllowed
        | QWebEngineUrlScheme.Flag.CorsEnabled
        | QWebEngineUrlScheme.Flag.FetchApiAllowed
    )
    QWebEngineUrlScheme.registerScheme(scheme)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--measure", action="store_true", help="print load time and fps as JSON lines, then exit"
    )
    parser.add_argument("--screenshot", type=Path, help="save a screenshot (implies --measure)")
    parser.add_argument("--timeout", type=float, default=120, help="seconds before giving up")
    parser.add_argument("--size", default="1280x800", help="window size, WIDTHxHEIGHT")
    args = parser.parse_args(argv)
    model = args.model.expanduser().resolve()
    kind, texture = detect(model)
    measure = args.measure or args.screenshot is not None

    register_scheme()  # must happen before QApplication exists
    app = QApplication(sys.argv[:1])

    models = {model.name: model}
    query = QUrlQuery()
    query.addQueryItem("model", f"ez2d://model/{model.name}")
    query.addQueryItem("kind", kind)
    if texture and (model.parent / texture).is_file():
        models[texture] = model.parent / texture
        query.addQueryItem("texture", f"ez2d://model/{texture}")
    url = QUrl("ez2d://app/index.html")
    url.setQuery(query)

    profile = QWebEngineProfile.defaultProfile()
    handler = FileSchemeHandler(models)
    profile.installUrlSchemeHandler(SCHEME, handler)

    window = QMainWindow()
    window.setWindowTitle(f"EZ2DIGITIZE viewer spike: {model.name} ({kind})")
    width, height = (int(v) for v in args.size.split("x"))
    window.resize(width, height)
    view = QWebEngineView(window)
    exit_code = 0

    def finish(code: int) -> None:
        nonlocal exit_code
        exit_code = code
        if args.screenshot:
            view.grab().save(str(args.screenshot))
        app.quit()

    def on_event(event: dict[str, object]) -> None:
        if not measure:
            return
        if event["event"] == "fps":
            finish(0)
        elif event["event"] == "error":
            finish(1)

    page = ReportingPage(profile, on_event)
    view.setPage(page)
    window.setCentralWidget(view)
    view.loadFinished.connect(
        lambda ok: print(json.dumps({"event": "page-loaded", "ok": ok,
                                     "t_s": round(time.monotonic() - STARTED, 2)}), flush=True)
    )  # fmt: skip
    view.load(url)
    window.show()

    if measure:

        def timed_out() -> None:
            print(json.dumps({"event": "timeout", "after_s": args.timeout}), flush=True)
            finish(2)

        QTimer.singleShot(int(args.timeout * 1000), timed_out)
    app.exec()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
