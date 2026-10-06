# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phone upload over Wi-Fi: a one-time web page the phone sends photos to.

`UploadSession` serves a small upload page while it is open. The phone
scans a QR code with the session's URL, picks photos (or videos) taken
with its camera app, and the page sends each file in chunks:

    GET  /<token>/                  the page
    GET  /<token>/files/<id>        bytes received so far, to resume
    PUT  /<token>/files/<id>?offset=N&name=...&size=...   one chunk
    POST /<token>/done              the phone says it has finished

Files are stored bit for bit (no re-encoding, EXIF kept). A chunk must
start where the stored part ends, so after a Wi-Fi drop the page asks how
much arrived and continues from there. `finish` turns the complete files
into a capture bundle (source "upload"); incomplete ones are dropped.

Safety: the URL carries a random token and every other path is refused;
the server listens on the local network address only, refuses clients
outside private address ranges, and runs only while the session is open.
"""

from __future__ import annotations

import ipaddress
import json
import re
import secrets
import shutil
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from ez2digitize.core.capture import (
    CaptureBundle,
    CaptureError,
    CaptureFile,
    add_file,
    add_jpeg_copies,
    assemble_bundle,
    classify,
)
from ez2digitize.core.project import Project

CHUNK_LIMIT = 16 * 1024 * 1024  # the page sends 4 MB; refuse anything above this
FILE_LIMIT = 8 * 1024**3
_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


@dataclass
class UploadFile:
    id: str
    name: str
    size: int
    received: int = 0

    @property
    def complete(self) -> bool:
        return self.received == self.size


def local_address() -> str:
    """This machine's address on the local network (the one a phone can reach)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No packet is sent: connecting a UDP socket only picks the route.
        probe.connect(("192.0.2.1", 9))
        address = str(probe.getsockname()[0])
    except OSError:
        address = "127.0.0.1"
    finally:
        probe.close()
    return address


class UploadSession:
    """One upload page, open from `start` until `close` (or `finish`)."""

    def __init__(self, project: Project, *, host: str | None = None, port: int = 0) -> None:
        self.project = project
        self.token = secrets.token_urlsafe(18)
        self.host = host or local_address()
        self.port = port
        self.folder = project.captures_dir / f".uploading-{secrets.token_hex(4)}"
        self.files: dict[str, UploadFile] = {}
        self.phone_done = False
        self.user_agent: str | None = None
        self.lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/{self.token}/"

    def start(self) -> str:
        self.folder.mkdir(parents=True)
        handler = type("Handler", (_Handler,), {"session": self})
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._server.daemon_threads = True
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, name="upload")
        self._thread.daemon = True
        self._thread.start()
        return self.url

    def status(self) -> list[UploadFile]:
        with self.lock:
            return [UploadFile(f.id, f.name, f.size, f.received) for f in self.files.values()]

    def close(self) -> None:
        """Stop serving and delete anything received but not imported."""
        self._stop()
        shutil.rmtree(self.folder, ignore_errors=True)

    def finish(self, *, now: datetime | None = None) -> CaptureBundle:
        """Stop serving and turn the complete files into a capture bundle."""
        self._stop()
        complete = [f for f in self.status() if f.complete]
        if not complete:
            self.close()
            raise CaptureError("no photo was received completely")
        used: set[str] = set()

        def fill(staging: Path) -> list[CaptureFile]:
            entries = []
            for upload in sorted(complete, key=lambda f: f.name.lower()):
                name = _unique(upload.name, used)
                (self.folder / upload.id).rename(staging / name)
                kind = classify(Path(name))
                assert kind is not None  # checked when the upload started
                entries.append(add_file(staging, name, original_name=upload.name, kind=kind))
            return add_jpeg_copies(staging, entries, used)

        try:
            return assemble_bundle(
                self.project,
                fill,
                source="upload",
                device={"user_agent": self.user_agent} if self.user_agent else None,
                source_info={"files": len(complete)},
                now=now,
            )
        finally:
            shutil.rmtree(self.folder, ignore_errors=True)

    def _stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # --- request handling (called on the server's threads) -------------------------

    def _register(self, file_id: str, name: str, size: int) -> UploadFile:
        clean = Path(name.replace("\\", "/")).name.strip()
        if not clean or clean.startswith(".") or classify(Path(clean)) is None:
            raise ValueError(f"{name!r} is not a photo or video")
        if not 0 < size <= FILE_LIMIT:
            raise ValueError("file too large or empty")
        with self.lock:
            upload = self.files.get(file_id)
            if upload is None:
                upload = UploadFile(file_id, clean, size)
                self.files[file_id] = upload
            elif upload.size != size:
                raise ValueError("size changed")
            return upload

    def _append(self, upload: UploadFile, offset: int, data: bytes) -> tuple[int, bool]:
        """Store a chunk if it starts where the file ends; (received, stored)."""
        with self.lock:
            if offset != upload.received:
                return upload.received, False  # the page resends from here
            if upload.received + len(data) > upload.size:
                raise ValueError("more data than announced")
            with (self.folder / upload.id).open("ab") as part:
                part.write(data)
            upload.received += len(data)
            return upload.received, True


def _unique(name: str, used: set[str]) -> str:
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate, n = name, 1
    while candidate.lower() in used:
        n += 1
        candidate = f"{stem}-{n}{suffix}"
    used.add(candidate.lower())
    return candidate


class _Handler(BaseHTTPRequestHandler):
    session: UploadSession
    server_version = "EZ2DIGITIZE"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib name
        pass  # no console noise from the phone's requests

    def _route(self) -> tuple[list[str], dict[str, list[str]]] | None:
        address = ipaddress.ip_address(self.client_address[0])
        if not (address.is_private or address.is_loopback or address.is_link_local):
            self._reply(403, {"error": "local network only"})
            return None
        parts = urlsplit(self.path)
        segments = [unquote(s) for s in parts.path.split("/") if s]
        if not segments or not secrets.compare_digest(segments[0], self.session.token):
            self._reply(404, {"error": "not found"})
            return None
        if self.headers.get("User-Agent"):
            self.session.user_agent = self.headers["User-Agent"][:200]
        return segments[1:], parse_qs(parts.query)

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        route = self._route()
        if route is None:
            return
        segments, _ = route
        if not segments:
            body = UPLOAD_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif len(segments) == 2 and segments[0] == "files" and _ID.match(segments[1]):
            upload = self.session.files.get(segments[1])
            self._reply(200, {"received": upload.received if upload else 0})
        else:
            self._reply(404, {"error": "not found"})

    def do_PUT(self) -> None:  # noqa: N802 - stdlib name
        route = self._route()
        if route is None:
            return
        segments, query = route
        if len(segments) != 2 or segments[0] != "files" or not _ID.match(segments[1]):
            self._reply(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 <= length <= CHUNK_LIMIT:
                raise ValueError("chunk too large")
            upload = self.session._register(
                segments[1], query.get("name", [""])[0], int(query.get("size", ["0"])[0])
            )
            offset = int(query.get("offset", ["0"])[0])
            data = self.rfile.read(length)
            if len(data) != length:
                raise ValueError("connection closed during the chunk")
            received, stored = self.session._append(upload, offset, data)
        except ValueError as exc:
            self._reply(400, {"error": str(exc)})
            return
        status = 200 if stored else 409
        self._reply(status, {"received": received, "complete": received == upload.size})

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        route = self._route()
        if route is None:
            return
        segments, _ = route
        if segments == ["done"]:
            self.session.phone_done = True
            self._reply(200, {"ok": True})
        else:
            self._reply(404, {"error": "not found"})

    def _reply(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def wait_for_phone(session: UploadSession, *, poll_s: float = 0.5) -> None:
    """Block until the phone presses Done (for the CLI)."""
    while not session.phone_done:
        time.sleep(poll_s)


UPLOAD_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Send photos to EZ2DIGITIZE</title>
<style>
:root { --bg: #fafafa; --fg: #1b1b1b; --muted: #666; --accent: #2a62c9; --bar: #e3e3e3; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161616; --fg: #eee; --muted: #aaa; --accent: #6f9bf0; --bar: #333; }
}
body { font: 16px/1.4 system-ui, sans-serif; margin: 0; padding: 16px; background: var(--bg);
       color: var(--fg); }
h1 { font-size: 1.3em; margin: 0 0 8px; }
p { color: var(--muted); margin: 0 0 16px; }
label.pick, button { display: block; width: 100%; padding: 14px; font-size: 1.05em;
  border-radius: 10px; border: 0; background: var(--accent); color: #fff; text-align: center;
  margin-bottom: 12px; box-sizing: border-box; }
button.secondary { background: var(--bar); color: var(--fg); }
input[type=file] { display: none; }
.row { margin: 8px 0; font-size: .9em; }
.name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar { height: 6px; background: var(--bar); border-radius: 3px; overflow: hidden; }
.bar > div { height: 100%; width: 0; background: var(--accent); }
#summary { font-weight: 600; margin: 12px 0; }
</style></head><body>
<h1>Send photos to EZ2DIGITIZE</h1>
<p>Pick the photos (or a video) of your object. Keep this page open until every file is
sent. iPhone HEIC photos are fine: a JPEG copy is made on the computer.</p>
<label class="pick">Choose photos<input id="pick" type="file" multiple
  accept="image/*,video/*"></label>
<div id="summary"></div>
<div id="list"></div>
<button id="done" class="secondary">I'm done</button>
<script>
const CHUNK = 4 * 1024 * 1024;
const base = location.pathname.endsWith("/") ? location.pathname : location.pathname + "/";
const list = document.getElementById("list"), summary = document.getElementById("summary");
let total = 0, sent = 0, queue = Promise.resolve();
const sleep = ms => new Promise(r => setTimeout(r, ms));

// Same file, same id: a reload or a second pick resumes instead of restarting.
// (crypto.subtle needs HTTPS, which a local page doesn't have: two FNV-1a hashes.)
function fileId(file) {
  const text = `${file.name}|${file.size}|${file.lastModified}`;
  let a = 0x811c9dc5, b = 0x01000193;
  for (let i = 0; i < text.length; i++) {
    a = Math.imul(a ^ text.charCodeAt(i), 0x01000193) >>> 0;
    b = Math.imul(b ^ text.charCodeAt(text.length - 1 - i), 0x811c9dc5) >>> 0;
  }
  return a.toString(16).padStart(8, "0") + b.toString(16).padStart(8, "0") + "-" + file.size;
}

function show() { summary.textContent = total ? `${Math.round(100 * sent / total)} % sent` : ""; }

async function upload(file, bar) {
  const id = fileId(file);
  const q = `name=${encodeURIComponent(file.name)}&size=${file.size}`;
  let offset = 0, delay = 1000;
  for (;;) {
    try {
      offset = (await (await fetch(`${base}files/${id}`)).json()).received;
      while (offset < file.size) {
        const chunk = file.slice(offset, offset + CHUNK);
        const r = await fetch(`${base}files/${id}?offset=${offset}&${q}`,
                              { method: "PUT", body: chunk });
        const answer = await r.json();
        if (r.status === 400) throw new Error(answer.error);
        sent += answer.received - offset;
        offset = answer.received;
        bar.style.width = `${100 * offset / file.size}%`;
        show();
        delay = 1000;
      }
      return;
    } catch (e) {
      // A failed fetch is a TypeError (wording differs per browser); anything
      // else is the server refusing the file.
      if (!(e instanceof TypeError)) {
        bar.parentElement.previousElementSibling.textContent += ` — ${e.message}`;
        return;
      }
      await sleep(delay);  // Wi-Fi dropped: ask what arrived, then go on
      delay = Math.min(delay * 2, 15000);
    }
  }
}

document.getElementById("pick").addEventListener("change", e => {
  for (const file of e.target.files) {
    total += file.size;
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = '<div class="name"></div><div class="bar"><div></div></div>';
    row.firstChild.textContent = file.name;
    list.appendChild(row);
    const bar = row.querySelector(".bar > div");
    queue = queue.then(() => upload(file, bar));
  }
  show();
  e.target.value = "";
});

document.getElementById("done").addEventListener("click", async () => {
  await queue;
  await fetch(`${base}done`, { method: "POST" });
  summary.textContent = "Done. You can close this page.";
});
</script></body></html>
"""
