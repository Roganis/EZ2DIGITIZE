# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The 3D viewer: its setup, its vendored libraries, and (on Linux) the real page."""

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from ez2digitize import views
from ez2digitize.ui import viewer

REPO = Path(__file__).parents[2]
TIMEOUT_MS = 60_000


def test_vendored_libraries_match_their_pins() -> None:
    vendor = viewer.WEB_DIR / "vendor"
    manifest = json.loads((vendor / "VENDOR.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((vendor / name).read_bytes()).hexdigest() == digest, name
    script = (REPO / "tools" / "viewer" / "fetch_vendor.py").read_text()
    for package in manifest["packages"].values():
        assert package["integrity"][:40] in script and package["version"] in script
    licenses = (REPO / "THIRD_PARTY_LICENSES").read_text()
    assert "three.js" in licenses and "Spark" in licenses


def test_sandbox_is_kept_where_it_works(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    if not sys.platform.startswith("linux"):
        assert not viewer.sandbox_unusable()
        return
    monkeypatch.setattr("os.geteuid", lambda: 1000)
    monkeypatch.delenv("APPIMAGE", raising=False)
    assert not viewer.sandbox_unusable()
    restricted = tmp_path / "restrict"
    monkeypatch.setattr(viewer, "USERNS_RESTRICTED", restricted)
    monkeypatch.setenv("APPIMAGE", "/x/EZ2DIGITIZE.AppImage")
    restricted.write_text("1\n")
    assert viewer.sandbox_unusable()
    restricted.write_text("0\n")
    assert not viewer.sandbox_unusable()
    monkeypatch.setattr("os.geteuid", lambda: 0)
    assert viewer.sandbox_unusable()


def _camera_view(tmp_path: Path) -> views.View:
    model = tmp_path / "sparse"
    model.mkdir()
    (model / "cameras.bin").write_bytes(struct.pack("<QIiQQ4d", 1, 1, 1, 4, 3, 2.0, 2.0, 2.0, 1.5))
    images = struct.pack("<Q", 1) + struct.pack("<I7dI", 1, 1, 0, 0, 0, 0, 0, 0, 1)
    (model / "images.bin").write_bytes(images + b"c/a.jpg\0" + struct.pack("<Q", 0))
    points = struct.pack("<Q", 2)
    for i, z in enumerate((4.0, 5.0), start=1):
        points += struct.pack("<Q3d3BdQ", i, 0.0, 0.0, z, 255, 0, 0, 0.1, 0)
    (model / "points3D.bin").write_bytes(points)
    return views.View("cameras", "cameras", "run1", model, None)


# The real page needs WebGL: on Linux CI Chromium falls back to its software
# renderer (SwiftShader / llvmpipe); elsewhere that path isn't set up.
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="software WebGL on Linux")
@pytest.mark.skipif(not viewer.AVAILABLE, reason="no QtWebEngine")
def test_page_shows_a_view(qtbot: QtBot, tmp_path: Path) -> None:
    widget = viewer.ViewerWidget(tmp_path / "cache")
    qtbot.addWidget(widget)
    widget.resize(400, 300)
    widget.show()
    errors: list[str] = []
    widget.failed.connect(errors.append)
    widget.show_view(_camera_view(tmp_path))  # queued until the page is ready
    with qtbot.waitSignal(widget.loaded, timeout=TIMEOUT_MS) as loaded:
        pass
    assert errors == []
    assert loaded.args is not None
    event = loaded.args[0]
    assert event["kind"] == "cameras" and event["count"] == 2
    assert event["unit"] == "points, 1 cameras"
