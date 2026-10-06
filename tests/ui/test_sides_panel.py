# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Both sides panel, and the project page's two-sided workflow."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtWidgets import QComboBox, QFileDialog, QMessageBox
from pytestqt.qtbot import QtBot

from ez2digitize.core.capture import import_folder, list_bundles
from ez2digitize.core.project import Project
from ez2digitize.pipeline import Tools
from ez2digitize.ui.project_page import ProjectPage
from ez2digitize.ui.sides_panel import SidesPanel


def _folder(path: Path, *names: str) -> Path:
    path.mkdir()
    for n, name in enumerate(names):
        Image.new("RGB", (40, 30), (n * 50, 0, 0)).save(path / name)
    return path


@pytest.fixture
def project(tmp_path: Path) -> Project:
    project = Project.create(tmp_path / "project")
    import_folder(project, _folder(tmp_path / "top", "a.jpg", "b.jpg"))
    return project


def _combo(panel: SidesPanel, row: int) -> QComboBox:
    item = panel.captures.topLevelItem(row)
    assert item is not None
    widget = panel.captures.itemWidget(item, 2)
    assert isinstance(widget, QComboBox)
    return widget


def test_choosing_sides(qtbot: QtBot, project: Project, tmp_path: Path) -> None:
    panel = SidesPanel(project)
    qtbot.addWidget(panel)
    panel.refresh()
    assert "one-sided" in panel.checklist.text()
    import_folder(project, _folder(tmp_path / "under", "c.jpg"))
    panel.refresh()
    assert panel.captures.topLevelItemCount() == 2

    with qtbot.waitSignal(panel.sides_changed):
        _combo(panel, 1).setCurrentIndex(1)  # Turned over
    assert [b.flipped for b in list_bundles(project)] == [False, True]
    text = panel.checklist.text()
    assert "✔ First side: 2 photos" in text and "✔ Turned over: 1 photos" in text
    assert "✘ Masks: 0 of 3 photos have one" in text
    assert panel.problems(use_masks=True)

    for bundle in list_bundles(project):
        (project.masks_dir / bundle.id).mkdir(parents=True)
        for entry in bundle.files:
            (project.masks_dir / bundle.id / f"{entry.name}.png").write_bytes(b"m")
    panel.refresh()
    assert "Ready to build" in panel.checklist.text()
    assert panel.problems(use_masks=True) == []


@pytest.fixture
def page(qtbot: QtBot, project: Project, fake_tools: Tools) -> Iterator[ProjectPage]:
    page = ProjectPage(project, lambda: fake_tools)
    qtbot.addWidget(page)
    yield page
    assert page.photo_checks.wait()
    assert page.masks_panel.wait()


def test_other_side_import_and_run_warning(
    page: ProjectPage, project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    under = _folder(tmp_path / "under", "c.jpg", "d.jpg")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(under))
    page.other_side_button.click()
    assert [b.flipped for b in list_bundles(project)] == [False, True]
    assert page.tabs.currentWidget() is page.sides_panel
    assert "of the turned-over side" in page.status.text()

    # Building without masks asks first; No stays on the Both sides tab.
    asked: list[str] = []

    def no(_parent: object, _title: str, text: str) -> QMessageBox.StandardButton:
        asked.append(text)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", no)
    page.start_run()
    assert asked and "4 of 4 photos have no mask" in asked[0]
    assert not page.runner.running
    assert page.tabs.currentWidget() is page.sides_panel
