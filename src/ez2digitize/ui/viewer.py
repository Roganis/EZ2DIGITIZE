# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The 3D viewer: a web page (three.js + Spark, viewer_web/) in QWebEngineView.

The page and the files it shows are served through a custom `ez2d://`
scheme: `ez2d://app/...` from viewer_web/, `ez2d://data/<token>/<name>` from
the files of the view being shown (nothing else on disk is reachable).
Every other request (http, https, file, ...) is blocked: the viewer only
ever shows the app's own content.

Python asks the page to draw a view with `ez2d.show(spec)`; the page
reports back through console lines "EZ2D " + JSON (ready, loading, loaded,
error), turned into `ViewerWidget` signals.

`prepare()` must run before the QApplication is created: it registers the
scheme and, where Chromium's sandbox can't work (running as root; an
AppImage on a system that restricts unprivileged user namespaces, e.g.
Ubuntu 24.04), turns the sandbox off. The page loads nothing from outside.
"""

from __future__ import annotations

import atexit
import json
import mimetypes
import os
import secrets
import sys
from functools import partial
from pathlib import Path
from typing import Any

import shiboken6
from PySide6.QtCore import (
    QBuffer,
    QByteArray,
    QCoreApplication,
    QFile,
    QIODevice,
    QUrl,
    Signal,
)
from PySide6.QtWidgets import QLabel, QStackedLayout, QWidget

try:  # QtWebEngine is optional: the rest of the app works without it.
    from PySide6.QtWebEngineCore import (
        QWebEnginePage,
        QWebEngineProfile,
        QWebEngineUrlRequestInfo,
        QWebEngineUrlRequestInterceptor,
        QWebEngineUrlRequestJob,
        QWebEngineUrlScheme,
        QWebEngineUrlSchemeHandler,
    )
    from PySide6.QtWebEngineWidgets import QWebEngineView

    AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the Qt build
    AVAILABLE = False

from ez2digitize import views

WEB_DIR = Path(__file__).resolve().parent / "viewer_web"
SCHEME = b"ez2d"
MIME = {
    ".js": b"text/javascript",
    ".html": b"text/html",
    ".json": b"application/json",
    ".ply": b"application/octet-stream",
    ".glb": b"model/gltf-binary",
}
USERNS_RESTRICTED = Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")


def prepare() -> None:
    """Set up QtWebEngine; call before creating the QApplication."""
    if not AVAILABLE:
        return
    if sandbox_unusable():
        os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
    if QWebEngineUrlScheme.schemeByName(QByteArray(SCHEME)).name().isEmpty():
        scheme = QWebEngineUrlScheme(QByteArray(SCHEME))
        scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
        scheme.setFlags(
            QWebEngineUrlScheme.Flag.SecureScheme
            | QWebEngineUrlScheme.Flag.LocalAccessAllowed
            | QWebEngineUrlScheme.Flag.CorsEnabled
            | QWebEngineUrlScheme.Flag.FetchApiAllowed
        )
        QWebEngineUrlScheme.registerScheme(scheme)


def usable() -> bool:
    """QtWebEngine loads and the page with its libraries is there (packaging check)."""
    return AVAILABLE and (WEB_DIR / "index.html").is_file() and (WEB_DIR / "vendor").is_dir()


def sandbox_unusable() -> bool:
    """True where Chromium's sandbox fails to start (Linux only)."""
    if not sys.platform.startswith("linux"):
        return False
    if os.geteuid() == 0:
        return True
    if os.environ.get("APPIMAGE"):
        try:
            return USERNS_RESTRICTED.read_text().strip() == "1"
        except OSError:
            return False
    return False


if AVAILABLE:

    class _Handler(QWebEngineUrlSchemeHandler):
        """Serves the page and the registered files of the view being shown."""

        def __init__(self) -> None:
            super().__init__()
            self.files: dict[str, Path | bytes] = {}

        def requestStarted(self, job: QWebEngineUrlRequestJob) -> None:  # noqa: N802 - Qt API
            url = job.requestUrl()
            host, rel = url.host(), url.path().lstrip("/")
            source: Path | bytes | None = None
            if host == "app":
                path = (WEB_DIR / rel).resolve()
                if path.is_relative_to(WEB_DIR) and path.is_file():
                    source = path
            elif host == "data":
                source = self.files.get(rel)
            if source is None:
                job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
                return
            mime = MIME.get(Path(rel).suffix.lower())
            if mime is None:
                guessed = mimetypes.guess_type(rel)[0] or "application/octet-stream"
                mime = guessed.encode()
            device: QIODevice
            if isinstance(source, bytes):
                device = QBuffer(job)  # parented to the job, freed with it
                device.setData(QByteArray(source))
            else:
                device = QFile(str(source), job)
            if not device.open(QIODevice.OpenModeFlag.ReadOnly):
                job.fail(QWebEngineUrlRequestJob.Error.RequestFailed)
                return
            job.reply(QByteArray(mime), device)

    class _BlockOutside(QWebEngineUrlRequestInterceptor):
        """Only ez2d:// (and the page's own blob:/data: URLs) may load."""

        def interceptRequest(self, info: QWebEngineUrlRequestInfo) -> None:  # noqa: N802
            if info.requestUrl().scheme() not in ("ez2d", "blob", "data"):
                info.block(True)

    class _Page(QWebEnginePage):
        def __init__(self, profile: QWebEngineProfile, widget: ViewerWidget) -> None:
            super().__init__(profile, widget)
            self.widget = widget

        def javaScriptConsoleMessage(  # noqa: N802 - Qt API
            self,
            level: QWebEnginePage.JavaScriptConsoleMessageLevel,
            message: str,
            line: int,
            source: str,
        ) -> None:
            if message.startswith("EZ2D "):
                try:
                    event = json.loads(message[5:])
                except json.JSONDecodeError:
                    return
                if isinstance(event, dict):
                    self.widget.on_page_event(event)
            elif level == QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
                self.widget.on_page_event(
                    {"event": "console-error", "message": f"{message} ({source}:{line})"}
                )


class ViewerWidget(QWidget):
    """Shows one `views.View` at a time.

    `loaded` carries the page's report (count, unit, load_ms); `failed` a
    message. `cache` is where files made for the viewer go (see views.files).
    """

    ready = Signal(dict)  # the page is up: {"gpu": ..., "webgl2": ...}
    loaded = Signal(dict)
    failed = Signal(str)
    # The user dragged the crop box: {"centre", "half_size", "yaw"} (upright frame).
    crop_changed = Signal(dict)
    # Two points picked for the scale: {"points": [[x, y, z], [x, y, z]]} (upright frame).
    measured = Signal(dict)

    def __init__(self, cache: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cache = cache
        self.is_ready = False
        self.shown: views.View | None = None
        self._pending: views.View | None = None
        self._crop: tuple[dict[str, Any] | None, bool] = (None, False)
        self._measure: tuple[list[list[float]] | None, str] = (None, "")
        self.message = QLabel()
        self.message.setWordWrap(True)
        layout = QStackedLayout(self)
        layout.addWidget(self.message)
        if not AVAILABLE:
            self.message.setText("The 3D viewer needs QtWebEngine, which this Qt lacks.")
            return
        # A private, off-the-record profile: nothing is written to disk. It
        # must outlive its page, so it is kept apart and deleted only once
        # the page is gone (as a child of either, it could go first).
        self.profile = QWebEngineProfile()
        self.handler = _Handler()
        self.handler.setParent(self.profile)
        self.profile.installUrlSchemeHandler(QByteArray(SCHEME), self.handler)
        self.interceptor = _BlockOutside()
        self.interceptor.setParent(self.profile)
        self.profile.setUrlRequestInterceptor(self.interceptor)
        self.view = QWebEngineView(self)
        self.page = _Page(self.profile, self)
        _profiles[self.profile] = self.page
        self.page.destroyed.connect(partial(_release, self.profile))
        global _quit_connected
        app = QCoreApplication.instance()
        if app is not None and not _quit_connected:
            app.aboutToQuit.connect(_shutdown)
            _quit_connected = True
        self.view.setPage(self.page)
        layout.addWidget(self.view)
        layout.setCurrentWidget(self.view)
        self.view.load(QUrl("ez2d://app/index.html"))

    def show_view(self, view: views.View) -> None:
        """Draw `view` (once the page is ready); errors go to `failed`."""
        if not AVAILABLE:
            return
        if not self.is_ready:
            self._pending = view
            return
        try:
            files = views.files(view, self.cache)
        except views.ViewError as exc:
            self.failed.emit(str(exc))
            return
        token = secrets.token_hex(8)
        self.handler.files = {}
        urls = {}
        for name, source in files.items():
            suffix = source.suffix if isinstance(source, Path) else _suffix(name, view)
            key = f"{token}/{name}{suffix}"
            self.handler.files[key] = source
            urls[name] = f"ez2d://data/{key}"
        self.shown = view
        spec = views.spec(view, urls)
        self.page.runJavaScript(f"ez2d.show({json.dumps(spec)})")

    def set_crop_box(self, box: dict[str, Any] | None, *, editable: bool = True) -> None:
        """Show the crop box ({centre, half_size, yaw}, upright frame) or hide it (None)."""
        self._crop = (box, editable)
        if AVAILABLE and self.is_ready:
            self.page.runJavaScript(f"ez2d.setCropBox({json.dumps(box)}, {json.dumps(editable)})")

    def set_measuring(self, on: bool) -> None:
        """Let the user pick two points (reported by `measured`), or stop."""
        if AVAILABLE and self.is_ready:
            self.page.runJavaScript(f"ez2d.setMeasuring({json.dumps(on)})")

    def set_measure(self, points: list[list[float]] | None, label: str = "") -> None:
        """Show two points (upright frame) joined by a line, with `label`; None hides them."""
        self._measure = (points, label)
        if AVAILABLE and self.is_ready:
            self.page.runJavaScript(f"ez2d.setMeasure({json.dumps(points)}, {json.dumps(label)})")

    def frame_crop_box(self) -> None:
        if AVAILABLE and self.is_ready:
            self.page.runJavaScript("ez2d.frameCropBox()")

    def clear(self) -> None:
        self.shown = None
        self._pending = None
        if AVAILABLE and self.is_ready:
            self.page.runJavaScript("ez2d.clear()")

    def on_page_event(self, event: dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == "ready":
            self.is_ready = True
            self.ready.emit(event)
            if self._crop[0] is not None:
                self.set_crop_box(self._crop[0], editable=self._crop[1])
            if self._measure[0] is not None:
                self.set_measure(*self._measure)
            if event.get("webgl") is False:
                self.failed.emit(
                    "The 3D view needs WebGL, which this graphics driver doesn't offer "
                    "(see Troubleshooting)."
                )
                return
            pending, self._pending = self._pending, None
            if pending is not None:
                self.show_view(pending)
        elif kind == "loaded":
            self.loaded.emit(event)
        elif kind == "cropbox":
            self._crop = (event, self._crop[1])
            self.crop_changed.emit(event)
        elif kind == "measure":
            self.measured.emit(event)
        elif kind in ("error", "console-error"):
            self.failed.emit(str(event.get("message", "unknown error")))


# Live profiles and their pages. A profile must be deleted after its page:
# when a viewer is deleted (closing a project) its page goes first and
# `_release` follows; at exit, when Python would tear them down in any
# order, `_shutdown` deletes all pages, then all profiles.
_profiles: dict[Any, Any] = {}
_quit_connected = False


def _release(profile: Any, *_args: object) -> None:
    if _profiles.pop(profile, None) is not None:
        profile.deleteLater()


def _shutdown() -> None:
    pages = [page for page in _profiles.values() if shiboken6.isValid(page)]
    profiles = [p for p in _profiles if shiboken6.isValid(p)]
    _profiles.clear()
    for page in pages:
        shiboken6.delete(page)
    for profile in profiles:
        shiboken6.delete(profile)


atexit.register(_shutdown)


def _suffix(name: str, view: views.View) -> str:
    # Generated files: the sparse points (PLY) and the cameras (JSON).
    return ".json" if name == "cameras" else ".ply"
