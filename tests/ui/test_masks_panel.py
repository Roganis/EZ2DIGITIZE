# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The masks panel, making masks with the fake masking worker."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidgetItem, QMessageBox
from pytestqt.qtbot import QtBot

from ez2digitize import masks
from ez2digitize.core.capture import import_folder, list_bundles
from ez2digitize.core.project import Project
from ez2digitize.ui.masks_panel import MasksPanel

TIMEOUT_MS = 20_000


@pytest.fixture
def project(tmp_path: Path) -> Project:
    folder = tmp_path / "photos"
    folder.mkdir()
    for n, name in enumerate(("a.jpg", "b.jpg", "c.jpg")):
        Image.new("RGB", (60, 40), (n * 60, 90, 0)).save(folder / name)
    project = Project.create(tmp_path / "project")
    import_folder(project, folder)
    return project


@pytest.fixture
def panel(qtbot: QtBot, project: Project) -> Iterator[MasksPanel]:
    widget = MasksPanel(project)
    qtbot.addWidget(widget)
    widget.refresh()
    yield widget
    assert widget.wait()


def _items(panel: MasksPanel) -> list[QListWidgetItem]:
    items = [panel.grid.item(i) for i in range(panel.grid.count())]
    return [item for item in items if item is not None]


def _make(qtbot: QtBot, panel: MasksPanel) -> None:
    with qtbot.waitSignal(panel.maker.running_changed, timeout=TIMEOUT_MS):
        panel.make_button.click()
    assert panel.make_button.text() == "Cancel"
    qtbot.waitUntil(lambda: not panel.maker.running, timeout=TIMEOUT_MS)


def test_without_masks(panel: MasksPanel) -> None:
    assert "Make masks to start" in panel.summary.text()
    items = _items(panel)
    assert [i.text() for i in items] == ["a.jpg\n(no mask)", "b.jpg\n(no mask)", "c.jpg\n(no mask)"]
    assert not items[0].flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert not panel.clear_button.isEnabled()


def test_make_review_and_clear(
    qtbot: QtBot,
    panel: MasksPanel,
    project: Project,
    fake_mask_worker: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_COVERAGE", "c.jpg=0")
    with qtbot.waitSignal(panel.masks_changed, timeout=TIMEOUT_MS):
        _make(qtbot, panel)

    # c.jpg's mask is empty: it starts out dropped, shown under "To look at".
    assert panel.filter.currentData() == "flagged"
    assert [i.text() for i in _items(panel)] == ["c.jpg\n(not used)\n⚠ nothing found"]
    panel.filter.setCurrentIndex(0)
    items = _items(panel)
    assert [i.checkState() for i in items] == [
        Qt.CheckState.Checked, Qt.CheckState.Checked, Qt.CheckState.Unchecked,
    ]  # fmt: skip
    assert "2 of 3 photos have a mask, 1 not used" in panel.summary.text()
    qtbot.waitUntil(lambda: all(not i.icon().isNull() for i in _items(panel)))

    # Unchecking drops the mask; the photo is then used whole.
    with qtbot.waitSignal(panel.masks_changed):
        items[0].setCheckState(Qt.CheckState.Unchecked)
    bundle = list_bundles(project)[0]
    qtbot.waitUntil(lambda: [e.state for e in panel.entries][0] == "dropped")  # refreshed
    assert _items(panel)[0].checkState() == Qt.CheckState.Unchecked
    assert {e.name: e.state for e in masks.review(project, [bundle])}["a.jpg"] == "dropped"
    panel.filter.setCurrentIndex(2)
    assert [i.text().split("\n")[0] for i in _items(panel)] == ["a.jpg", "c.jpg"]

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    panel.filter.setCurrentIndex(0)
    panel.clear_automatic()
    assert not masks.has_masks(project)
    assert all(i.text().endswith("(no mask)") for i in _items(panel))


def test_failure_is_reported(
    qtbot: QtBot, panel: MasksPanel, fake_mask_worker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_FAIL", "1")
    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _p, _t, text: shown.append(text))
    _make(qtbot, panel)
    assert shown and "failed (exit code 1)" in shown[0]
    assert not panel.progress.isVisible()


def test_import_masks(panel: MasksPanel, project: Project, tmp_path: Path) -> None:
    folder = tmp_path / "masks"
    folder.mkdir()
    Image.new("L", (60, 40), 255).save(folder / "b.png")
    bundle = list_bundles(project)[0]
    assert panel.import_masks(bundle, folder) == ["b.jpg"]
    texts = [i.text() for i in _items(panel)]
    assert texts[1] == "b.jpg\n(imported)"
    assert "1 of 3 photos have a mask" in panel.summary.text()


def test_locked_while_reconstructing(panel: MasksPanel, fake_mask_worker: Path) -> None:
    panel.set_locked(True)
    assert not panel.make_button.isEnabled() and not panel.grid.isEnabled()
    panel.set_locked(False)
    assert panel.make_button.isEnabled() and panel.import_button.isEnabled()
