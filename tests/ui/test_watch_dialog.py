# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

from PySide6.QtCore import QSettings
from pytestqt.qtbot import QtBot

from ez2digitize.core.capture import list_bundles
from ez2digitize.core.project import Project
from ez2digitize.ui.watch_dialog import FOLDER_KEY, WatchFolderDialog


class Clock:
    def __init__(self) -> None:
        self.now = 2_000_000_000.0

    def __call__(self) -> float:
        return self.now


def test_watch_and_import(qtbot: QtBot, tmp_path: Path) -> None:
    project = Project.create(tmp_path / "p")
    synced = tmp_path / "Camera"
    synced.mkdir()
    (synced / "before.jpg").write_bytes(b"there before")
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    clock = Clock()
    dialog = WatchFolderDialog(project, settings, clock=clock)
    qtbot.addWidget(dialog)
    assert "Choose the folder" in dialog.summary.text()

    dialog.set_folder(tmp_path / "missing")
    assert "Can't watch it" in dialog.summary.text()
    dialog.set_folder(synced)
    assert settings.value(FOLDER_KEY) == str(synced)  # remembered for next time
    assert dialog.summary.text() == "No new photos yet." and not dialog.import_button.isEnabled()

    (synced / "IMG_1.jpg").write_bytes(b"one")
    dialog.refresh()
    assert "1 still arriving" in dialog.summary.text()
    clock.now += 6
    dialog.refresh()
    assert dialog.files.count() == 1 and dialog.files.item(0).text() == "IMG_1.jpg"
    assert dialog.import_button.text() == "Import 1"
    clock.now += 30
    dialog.refresh()
    assert "the set looks complete" in dialog.summary.text()

    dialog.import_button.click()
    assert dialog.result() == dialog.DialogCode.Accepted
    assert dialog.bundle is not None and [f.name for f in dialog.bundle.files] == ["IMG_1.jpg"]
    assert [b.source for b in list_bundles(project)] == ["watch"]


def test_since_takes_recent_photos_already_there(qtbot: QtBot, tmp_path: Path) -> None:
    synced = tmp_path / "Camera"
    synced.mkdir()
    (synced / "IMG_1.jpg").write_bytes(b"one")  # modified just now (real time)
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.setValue(FOLDER_KEY, str(synced))
    dialog = WatchFolderDialog(Project.create(tmp_path / "p"), settings)
    qtbot.addWidget(dialog)
    assert dialog.files.count() == 0  # only new ones by default
    dialog.since.setCurrentIndex(1)  # the last 30 minutes
    assert dialog.files.count() == 1
