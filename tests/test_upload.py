# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ez2digitize.core.capture import CaptureError, list_bundles
from ez2digitize.core.project import Project
from ez2digitize.upload import UploadSession


@pytest.fixture
def session(tmp_path: Path) -> Iterator[UploadSession]:
    upload = UploadSession(Project.create(tmp_path / "project"), host="127.0.0.1")
    upload.start()
    yield upload
    upload.close()


def call(url: str, method: str = "GET", data: bytes | None = None) -> tuple[int, Any]:
    request = urllib.request.Request(url, data=data, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        body, status = error.read(), error.code
    try:
        return status, json.loads(body)
    except ValueError:
        return status, body.decode()


def put(session: UploadSession, file_id: str, name: str, size: int, offset: int, data: bytes):  # type: ignore[no-untyped-def]
    url = f"{session.url}files/{file_id}?offset={offset}&name={name}&size={size}"
    return call(url, "PUT", data)


def test_page_and_token(session: UploadSession) -> None:
    status, page = call(session.url)
    assert status == 200 and "Send photos to EZ2DIGITIZE" in page
    base = session.url.rsplit("/", 2)[0]
    assert call(f"{base}/wrong-token/")[0] == 404
    assert call(f"{base}/")[0] == 404


def test_chunked_resumable_upload_becomes_a_bundle(session: UploadSession) -> None:
    photo = bytes(range(256)) * 40  # 10 kB, stored bit for bit
    assert put(session, "f1", "IMG_0001.JPG", len(photo), 0, photo[:4000]) == (
        200, {"received": 4000, "complete": False},
    )  # fmt: skip
    # A chunk resent after a drop, from the wrong place: told where to resume.
    assert put(session, "f1", "IMG_0001.JPG", len(photo), 0, photo[:4000])[0] == 409
    assert call(f"{session.url}files/f1") == (200, {"received": 4000})
    assert put(session, "f1", "IMG_0001.JPG", len(photo), 4000, photo[4000:])[1]["complete"]
    # A second file that never finishes is left out.
    put(session, "f2", "IMG_0002.JPG", 100, 0, b"x" * 10)
    assert call(f"{session.url}done", "POST")[0] == 200 and session.phone_done
    assert [(f.name, f.complete) for f in session.status()] == [
        ("IMG_0001.JPG", True), ("IMG_0002.JPG", False),
    ]  # fmt: skip

    received = session.finish()
    assert received.videos == [] and received.folder is None
    bundle = received.bundle
    assert bundle is not None and bundle.source == "upload"
    assert [f.name for f in bundle.files] == ["IMG_0001.JPG"]
    assert (bundle.root / "IMG_0001.JPG").read_bytes() == photo
    assert bundle.verify() == []
    assert bundle.device is not None and "Python-urllib" in bundle.device["user_agent"]
    assert [b.id for b in list_bundles(session.project)] == [bundle.id]
    assert not session.folder.exists()


@pytest.mark.parametrize(
    ("name", "size"), [("notes.txt", 10), ("../../evil.jpg", 0), (".hidden.jpg", 10)]
)
def test_refused_files(session: UploadSession, name: str, size: int) -> None:
    status, answer = put(session, "f1", name, size, 0, b"x")
    assert status == 400 and "error" in answer


def test_path_names_are_reduced_to_the_file_name(session: UploadSession) -> None:
    put(session, "f1", "..%2F..%2Fphoto.jpg", 3, 0, b"abc")
    bundle = session.finish().bundle
    assert bundle is not None
    assert [f.name for f in bundle.files] == ["photo.jpg"]
    assert bundle.root.parent == session.project.captures_dir


def test_nothing_received(session: UploadSession) -> None:
    with pytest.raises(CaptureError, match="no photo"):
        session.finish()
    assert list_bundles(session.project) == []


def test_api_for_apps(session: UploadSession) -> None:
    status, api = call(f"{session.url}api")
    assert status == 200 and api["api"] == 1 and api["app"] == "EZ2DIGITIZE"
    assert ".motion.json" in api["accepts"]["motion"] and ".mp4" in api["accepts"]["video"]
    assert api["chunk_limit"] >= 4 * 1024 * 1024


def _describe(session: UploadSession, capture: object) -> tuple[int, Any]:
    return call(f"{session.url}capture", "POST", json.dumps(capture).encode())


def test_an_app_describes_the_capture(session: UploadSession) -> None:
    """A capture app's photos with their motion log, from the second side."""
    capture = {
        "source": "android",
        "device": {"make": "Google", "model": "Pixel 8"},
        "app": "EZ2DIGITIZE Capture 0.1",
        "flipped": True,
    }
    assert _describe(session, capture) == (200, {"ok": True})
    log = {
        "format": "ez2digitize-motion", "version": 1, "metric": True,
        "gravity": [[10.0, 0.0, -9.81, 0.0]],
        "frames": [{"file": "IMG_1.jpg", "t": 10.0}],
    }  # fmt: skip
    data = json.dumps(log).encode()
    put(session, "f1", "IMG_1.jpg", 3, 0, b"abc")
    put(session, "f2", "capture.motion.json", len(data), 0, data)
    bundle = session.finish().bundle
    assert bundle is not None and bundle.source == "android" and bundle.flipped
    assert bundle.device == {"make": "Google", "model": "Pixel 8"}
    assert bundle.source_info["app"] == "EZ2DIGITIZE Capture 0.1"
    kinds = {f.name: f.kind for f in bundle.files}
    assert kinds == {"IMG_1.jpg": "image", "capture.motion.json": "motion"}
    photo = next(f for f in bundle.files if f.kind == "image")
    assert photo.metadata["motion"]["down"] == [0, 1, 0]


@pytest.mark.parametrize(
    "capture",
    [[], {"source": "Android!"}, {"device": {"serial": "x"}}, {"flipped": "yes"}, {"app": 3}],
)
def test_bad_capture_descriptions(session: UploadSession, capture: object) -> None:
    status, answer = _describe(session, capture)
    assert status == 400 and "error" in answer
    assert call(f"{session.url}capture", "POST", b"{")[0] == 400


def test_videos_come_back_with_their_logs(session: UploadSession) -> None:
    """Videos are imported as frames later; each keeps the log named after it."""
    put(session, "f1", "VID_1.mp4", 5, 0, b"video")
    put(session, "f2", "VID_1.motion.json", 2, 0, b"{}")
    put(session, "f3", "IMG_1.jpg", 3, 0, b"abc")
    received = session.finish()
    assert received.bundle is not None
    assert [f.name for f in received.bundle.files] == ["IMG_1.jpg"]
    assert [p.name for p in received.videos] == ["VID_1.mp4"]
    assert received.folder is not None
    assert sorted(p.name for p in received.folder.iterdir()) == ["VID_1.motion.json", "VID_1.mp4"]
    assert received.folder.parent == session.project.captures_dir
    assert list_bundles(session.project) == [received.bundle]  # the folder is hidden
    received.discard()
    assert not received.folder.exists()


def test_only_a_video(session: UploadSession) -> None:
    put(session, "f1", "VID_1.mp4", 5, 0, b"video")
    received = session.finish()
    assert received.bundle is None and [p.name for p in received.videos] == ["VID_1.mp4"]
    received.discard()
