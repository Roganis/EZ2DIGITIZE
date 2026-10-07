#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Send a capture to EZ2DIGITIZE the way the companion app will.

A reference client for the upload protocol (docs/COMPANION.md), standard
library only: it checks the API version, describes the capture, sends
every file in resumable chunks and says it is done. Use it to try the
desktop side without the app, or as the model the app follows.

    python tools/companion/send_capture.py URL FILE... [--source android]
        [--make Google --model "Pixel 8"] [--app "name version"] [--flipped]

URL is the one the desktop shows (and puts in its QR code). FILE can be
photos, videos and their motion logs (`<name>.motion.json`).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API_VERSION = 1  # the one this client speaks
CHUNK = 4 * 1024 * 1024
RETRIES = 5


class SendError(Exception):
    pass


def call(url: str, method: str = "GET", data: bytes | None = None) -> tuple[int, Any]:
    headers = {"Content-Type": "application/json"} if method == "POST" else {}
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        body = error.read()
        try:
            return error.code, json.loads(body)
        except ValueError:
            return error.code, {"error": body.decode(errors="replace")}


def check_api(base: str) -> dict[str, Any]:
    """The server's description; SendError if this client can't talk to it."""
    status, api = call(f"{base}api")
    if status == 404:
        raise SendError("no api endpoint: a wrong address, or a desktop app too old for this")
    if status != 200 or not isinstance(api, dict) or api.get("app") != "EZ2DIGITIZE":
        raise SendError(f"not an EZ2DIGITIZE upload page (HTTP {status})")
    if api.get("api") != API_VERSION:
        raise SendError(f"the desktop speaks API {api.get('api')}, this client {API_VERSION}")
    return api


def accepted(api: dict[str, Any], path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for kind in api["accepts"].values() for suffix in kind)


def send_file(base: str, path: Path, chunk: int = CHUNK) -> None:
    """Send `path` in chunks, resuming from what the server has after any failure."""
    file_id = secrets.token_hex(8)
    size = path.stat().st_size
    query = urllib.parse.urlencode({"name": path.name, "size": size})
    failures = 0
    with path.open("rb") as fh:
        offset = 0
        while offset < size:
            fh.seek(offset)
            data = fh.read(chunk)
            try:
                status, answer = call(f"{base}files/{file_id}?offset={offset}&{query}", "PUT", data)
            except OSError:  # the connection dropped: ask where to resume
                status, answer = 0, None
            if status in (200, 409) and isinstance(answer, dict):
                offset = int(answer["received"])
                failures = 0 if status == 200 else failures + 1
            elif status == 400:
                raise SendError(f"{path.name}: refused: {answer.get('error')}")
            else:
                failures += 1
                time.sleep(min(2**failures, 30))
                with contextlib.suppress(OSError, KeyError, TypeError):
                    offset = int(call(f"{base}files/{file_id}")[1]["received"])
            if failures > RETRIES:
                raise SendError(f"{path.name}: gave up after {RETRIES} failures")


def send_capture(url: str, files: list[Path], capture: dict[str, Any]) -> None:
    base = url if url.endswith("/") else url + "/"
    api = check_api(base)
    refused = [f.name for f in files if not accepted(api, f)]
    if refused:
        raise SendError(f"not accepted by the desktop: {', '.join(refused)}")
    if capture:
        status, answer = call(f"{base}capture", "POST", json.dumps(capture).encode())
        if status != 200:
            raise SendError(f"capture description refused: {answer.get('error')}")
    for path in files:
        print(f"sending {path.name}", flush=True)
        send_file(base, path, min(CHUNK, int(api["chunk_limit"])))
    call(f"{base}done", "POST", b"")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("url")
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--source", default="android")
    parser.add_argument("--make")
    parser.add_argument("--model")
    parser.add_argument("--app")
    parser.add_argument("--flipped", action="store_true")
    args = parser.parse_args(argv)
    capture: dict[str, Any] = {"source": args.source, "flipped": args.flipped}
    device = {k: v for k, v in (("make", args.make), ("model", args.model)) if v}
    if device:
        capture["device"] = device
    if args.app:
        capture["app"] = args.app
    try:
        send_capture(args.url, args.files, capture)
    except (SendError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("done: import them on the desktop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
