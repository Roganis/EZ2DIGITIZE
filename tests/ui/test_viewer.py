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


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="software WebGL on Linux")
@pytest.mark.skipif(not viewer.AVAILABLE, reason="no QtWebEngine")
def test_dragging_a_crop_box_face(qtbot: QtBot, tmp_path: Path) -> None:
    widget = viewer.ViewerWidget(tmp_path / "cache")
    qtbot.addWidget(widget)
    widget.resize(500, 400)
    widget.show()
    widget.show_view(_camera_view(tmp_path))
    with qtbot.waitSignal(widget.loaded, timeout=TIMEOUT_MS):
        pass
    box = {"centre": [0.0, 0.0, 4.5], "half_size": [0.5, 0.5, 0.5], "yaw": 0.0}
    widget.set_crop_box(box)
    widget.frame_crop_box()

    def call(script: str) -> object:
        with qtbot.waitCallback(timeout=TIMEOUT_MS) as callback:
            widget.page.runJavaScript(script, 0, callback)
        assert callback.args is not None
        return callback.args[0]

    qtbot.wait(300)  # a frame or two with the new camera
    # Arrays don't cross into Python as they are: JSON.
    x, y = json.loads(str(call("JSON.stringify(ez2d.handleOnScreen(0, 1))")))  # the +X face
    events = (
        f"const c = document.querySelector('canvas');"
        f"const ev = (type, x) => c.dispatchEvent(new PointerEvent(type, {{clientX: x, "
        f"clientY: {y}, button: 0, pointerId: 1, bubbles: true}}));"
        f"ev('pointerdown', {x}); ev('pointermove', {x} + 60); ev('pointerup', {x} + 60); 1"
    )
    with qtbot.waitSignal(widget.crop_changed, timeout=TIMEOUT_MS) as changed:
        call(events)
    assert changed.args is not None
    moved = changed.args[0]
    # The +X face moved out; the -X face stayed: the box grew and its centre followed.
    assert moved["half_size"][0] > 0.55
    assert moved["centre"][0] - moved["half_size"][0] == pytest.approx(-0.5, abs=1e-3)
    assert moved["half_size"][1:] == [0.5, 0.5] and moved["centre"][1:] == [0.0, 4.5]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="software WebGL on Linux")
@pytest.mark.skipif(not viewer.AVAILABLE, reason="no QtWebEngine")
def test_picking_two_points(qtbot: QtBot, tmp_path: Path) -> None:
    widget = viewer.ViewerWidget(tmp_path / "cache")
    qtbot.addWidget(widget)
    widget.resize(500, 400)
    widget.show()
    errors: list[str] = []
    widget.failed.connect(errors.append)
    widget.show_view(_camera_view(tmp_path))
    with qtbot.waitSignal(widget.loaded, timeout=TIMEOUT_MS):
        pass

    def call(script: str) -> object:
        with qtbot.waitCallback(timeout=TIMEOUT_MS) as callback:
            widget.page.runJavaScript(script, 0, callback)
        assert callback.args is not None
        return callback.args[0]

    def click(point: list[float]) -> str:
        x, y = json.loads(str(call(f"JSON.stringify(ez2d.pointOnScreen({point}))")))
        return (
            "(() => { const c = document.querySelector('canvas');"
            "for (const type of ['pointerdown', 'pointerup']) c.dispatchEvent(new PointerEvent("
            f"type, {{clientX: {x}, clientY: {y}, button: 0, pointerId: 1, bubbles: true}}));"
            "return 1; })()"
        )

    widget.set_measuring(True)
    qtbot.wait(300)
    call(click([0.0, 0.0, 4.0]))  # the two sparse points
    with qtbot.waitSignal(widget.measured, timeout=TIMEOUT_MS) as measured:
        call(click([0.0, 0.0, 5.0]))
    assert measured.args is not None
    first, second = measured.args[0]["points"]
    assert first == pytest.approx([0.0, 0.0, 4.0], abs=1e-4)
    assert second == pytest.approx([0.0, 0.0, 5.0], abs=1e-4)
    # Measuring ends after two: a further click picks nothing.
    widget.set_measure([first, second], "25 mm")
    label = call("document.getElementById('measure').textContent")
    assert label == "25 mm"

    # Three points to level: reported as "level", the measured pair hidden meanwhile.
    widget.set_picking(3, "level")
    qtbot.wait(100)
    call(click([0.0, 0.0, 4.0]))
    call(click([0.0, 0.0, 5.0]))
    with qtbot.waitSignal(widget.level_picked, timeout=TIMEOUT_MS) as picked:
        call(click([0.0, 0.0, 4.0]))
    assert picked.args is not None and len(picked.args[0]["points"]) == 3
    assert errors == []
