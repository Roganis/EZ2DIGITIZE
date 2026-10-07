# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The reference capture client (tools/companion) against the real upload server."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import send_capture

from ez2digitize.core.project import Project
from ez2digitize.upload import UploadSession


@pytest.fixture
def session(tmp_path: Path) -> Iterator[UploadSession]:
    upload = UploadSession(Project.create(tmp_path / "project"), host="127.0.0.1")
    upload.start()
    yield upload
    upload.close()


def test_send_a_capture(session: UploadSession, tmp_path: Path) -> None:
    photo = tmp_path / "IMG_1.jpg"
    photo.write_bytes(bytes(range(256)) * 50)  # several chunks of 3000 bytes
    log = tmp_path / "IMG.motion.json"
    log.write_text(
        json.dumps({
            "format": "ez2digitize-motion", "version": 1,
            "gravity": [[5.0, 0.0, -9.81, 0.0]], "frames": [{"file": "IMG_1.jpg", "t": 5.0}],
        })
    )  # fmt: skip
    video = tmp_path / "VID_1.mp4"
    video.write_bytes(b"video")
    send_capture.CHUNK = 3000
    capture = {"source": "android", "device": {"model": "Pixel 8"}, "flipped": True}
    send_capture.send_capture(session.url, [photo, log, video], capture)
    assert session.phone_done

    received = session.finish()
    bundle = received.bundle
    assert bundle is not None and bundle.source == "android" and bundle.flipped
    assert (bundle.root / "IMG_1.jpg").read_bytes() == photo.read_bytes()
    assert next(f for f in bundle.files if f.kind == "image").metadata["motion"]["down"] == [
        0, 1, 0,
    ]  # fmt: skip
    assert [p.name for p in received.videos] == ["VID_1.mp4"] and received.flipped
    received.discard()


def test_refusals(session: UploadSession, tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("x")
    with pytest.raises(send_capture.SendError, match="not accepted by the desktop: notes.txt"):
        send_capture.send_capture(session.url, [notes], {})
    with pytest.raises(send_capture.SendError, match="capture description refused"):
        send_capture.send_capture(session.url, [], {"source": "Not OK"})
    base = session.url.rsplit("/", 2)[0]
    with pytest.raises(send_capture.SendError, match="no api endpoint"):
        send_capture.check_api(f"{base}/wrong-token/")
