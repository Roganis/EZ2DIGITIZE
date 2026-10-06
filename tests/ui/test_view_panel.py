# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The 3D view panel's crop box and scale controls (the viewer itself: test_viewer.py)."""

import struct
from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from ez2digitize import crop, scale, views
from ez2digitize.backends.colmap import Colmap
from ez2digitize.core.files import write_json_atomic
from ez2digitize.core.project import Project
from ez2digitize.core.stage import MANIFEST_FILE, StageManifest
from ez2digitize.ui.view_panel import ViewPanel

BACKEND = Colmap(Path("colmap"), "4.2.1").backend


def _succeed(project: Project, stage: str, run_id: str) -> Path:
    folder = project.stage_dir(stage)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = StageManifest(
        stage=stage, run_id=run_id, status="succeeded", cache_key="k", backend=BACKEND,
        command=[], parameters={}, inputs={}, started="", finished="", wall_s=1.0,
        cpu_s=1.0, peak_rss_mb=None, exit_code=0, host={},
    )  # fmt: skip
    write_json_atomic(folder / MANIFEST_FILE, manifest.to_dict())
    return folder


@pytest.fixture
def project(tmp_path: Path) -> Project:
    project = Project.create(tmp_path / "p")
    _succeed(project, "mapping", "m1")
    model = _succeed(project, "undistort", "u1") / "sparse"
    model.mkdir()
    (model / "cameras.bin").write_bytes(struct.pack("<QIiQQ4d", 1, 1, 1, 4, 3, 2.0, 2.0, 2.0, 1.5))
    images = struct.pack("<Q", 1) + struct.pack("<I7dI", 1, 1, 0, 0, 0, 0, 0, 0, 1)
    (model / "images.bin").write_bytes(images + b"c/a.jpg\0" + struct.pack("<Q", 0))
    points = struct.pack("<Q", 4)
    for i, (x, y, z) in enumerate(((0, 0, 4), (1, 0, 4), (0, 1, 5), (1, 1, 5)), start=1):
        points += struct.pack("<Q3d3BdQ", i, x, y, z, 255, 0, 0, 0.1, 0)
    (model / "points3D.bin").write_bytes(points)
    return project


def test_crop_controls(qtbot: QtBot, project: Project) -> None:
    panel = ViewPanel(project)
    qtbot.addWidget(panel)
    panel.refresh(prefer="cameras")
    assert not panel.crop_row.isHidden()
    assert not panel.use_crop.isChecked() and not panel.yaw.isEnabled()

    panel.use_crop.setChecked(True)  # starts from the sparse points
    box = crop.current(project)
    assert box is not None and box.camera_run == "m1"
    assert panel.yaw.isEnabled() and "Drag the yellow handles" in panel.crop_hint.text()

    upright = views.upright_rotation(project)
    panel.yaw.setValue(45.0)
    box = crop.current(project)
    assert box is not None
    assert crop.to_upright(box, upright).yaw == pytest.approx(45.0, abs=1e-6)

    # What the viewer reports after a drag is saved.
    panel._on_crop_dragged({"centre": [0.5, 0.0, 4.5], "half_size": [2.0, 1.0, 1.0], "yaw": 0})
    box = crop.current(project)
    assert box is not None
    assert crop.to_upright(box, upright).half_size == (2.0, 1.0, 1.0)

    panel.set_locked(True)  # a build is running
    assert not panel.use_crop.isEnabled()
    panel.set_locked(False)
    panel.use_crop.setChecked(False)
    assert crop.stored(project) is None


def test_no_crop_controls_for_the_mesh(qtbot: QtBot, project: Project) -> None:
    panel = ViewPanel(project)
    qtbot.addWidget(panel)
    panel.refresh(prefer="mesh")  # not there: falls back to the first view, cameras
    assert panel.choice.currentData() == "cameras"
    panel.choice.clear()
    panel._sync_crop()
    assert panel.crop_row.isHidden()
    assert panel.scale_row.isHidden()


def test_scale_controls(qtbot: QtBot, project: Project) -> None:
    panel = ViewPanel(project)
    qtbot.addWidget(panel)
    panel.refresh(prefer="cameras")
    assert not panel.scale_row.isHidden() and panel.pick.isEnabled()
    assert not panel.set_scale.isEnabled() and "Not set" in panel.scale_hint.text()

    # The viewer reports two picked points (upright frame), 1 unit apart.
    panel._on_measured({"points": [[0.0, 0.0, 4.0], [0.0, 1.0, 4.0]]})
    assert panel.set_scale.isEnabled() and "Type their real distance" in panel.scale_hint.text()
    assert scale.stored(project) is None  # not saved before Set scale
    panel.distance.setValue(50.0)
    panel.set_scale.click()
    saved = scale.current(project)
    assert saved is not None and saved.camera_run == "m1"
    assert saved.mm_per_unit == pytest.approx(50.0)
    upright = views.upright_rotation(project)
    assert scale.to_upright(saved.points, upright)[1] == pytest.approx((0.0, 1.0, 4.0))
    assert "Set: 50 mm" in panel.scale_hint.text()

    # Measured again more carefully: the same points, a new distance.
    panel.distance.setValue(40.0)
    panel.set_scale.click()
    again = scale.current(project)
    assert again is not None and again.points == saved.points and again.distance_mm == 40.0

    panel.set_locked(True)  # a build is running
    assert not panel.pick.isEnabled() and not panel.set_scale.isEnabled()
    panel.set_locked(False)
    panel.clear_scale.click()
    assert scale.stored(project) is None and "Not set" in panel.scale_hint.text()
