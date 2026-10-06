# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import ExifTags, Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeWidgetItem
from pytestqt.qtbot import QtBot

from ez2digitize.core.capture import import_folder, list_bundles
from ez2digitize.core.project import Project
from ez2digitize.ui.photo_checks import PhotoChecks

TIMEOUT_MS = 20_000


@pytest.fixture
def project(tmp_path: Path) -> Project:
    folder = tmp_path / "photos"
    folder.mkdir()
    noise = Image.effect_noise((1200, 1000), 60).convert("RGB")
    exif = Image.Exif()
    exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.FocalLength] = 35.0
    for i in range(25):
        noise.rotate(i * 3).save(folder / f"{i:02}.jpg", exif=exif)
    Image.new("RGB", (300, 200)).save(folder / "Preview.jpg")
    project = Project.create(tmp_path / "project")
    import_folder(project, folder)
    return project


@pytest.fixture
def checks(qtbot: QtBot, project: Project) -> Iterator[PhotoChecks]:
    widget = PhotoChecks(project)
    qtbot.addWidget(widget)
    yield widget
    assert widget.wait()


def _wait(qtbot: QtBot, checks: PhotoChecks) -> None:
    qtbot.waitUntil(lambda: not checks.inspecting, timeout=TIMEOUT_MS)


def _top(checks: PhotoChecks) -> list[QTreeWidgetItem]:
    tree = checks.tree
    items = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    return [item for item in items if item is not None]


def _first_child(item: QTreeWidgetItem) -> QTreeWidgetItem:
    child = item.child(0)
    assert child is not None
    return child


def test_inspects_then_shows_findings(qtbot: QtBot, checks: PhotoChecks) -> None:
    checks.refresh()
    assert checks.inspecting and not checks.tree.isEnabled()
    _wait(qtbot, checks)
    assert checks.tree.isEnabled()
    assert [f.code for f in checks.findings] == ["odd-size"]
    (item,) = _top(checks)
    assert item.text(0) == "Different size (1 photo)"
    assert checks.details.text().startswith("Preview.jpg is a different size")
    assert _first_child(item).text(0) == "Preview.jpg"
    assert checks.summary.text().startswith("Checked 26 photos: 1 warning.")


def test_unchecking_leaves_a_photo_out_and_back(
    qtbot: QtBot, checks: PhotoChecks, project: Project
) -> None:
    changes: list[bool] = []
    checks.exclusions_changed.connect(lambda: changes.append(True))
    checks.refresh()
    _wait(qtbot, checks)
    _first_child(_top(checks)[0]).setCheckState(0, Qt.CheckState.Unchecked)
    assert [f.name for f in list_bundles(project)[0].excluded] == ["Preview.jpg"]
    qtbot.waitUntil(lambda: checks.findings == [])
    (group,) = _top(checks)
    assert group.text(0) == "Left out of the reconstruction (1)"
    assert checks.summary.text() == "No problems found in 25 photos."

    _first_child(group).setCheckState(0, Qt.CheckState.Checked)
    assert list_bundles(project)[0].excluded == []
    qtbot.waitUntil(lambda: len(checks.findings) == 1)
    assert changes == [True, True]


def test_locked_while_running(qtbot: QtBot, checks: PhotoChecks) -> None:
    checks.refresh()
    _wait(qtbot, checks)
    checks.set_locked(True)
    assert not checks.tree.isEnabled()
    checks.set_locked(False)
    assert checks.tree.isEnabled()
