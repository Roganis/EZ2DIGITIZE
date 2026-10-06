# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox
from pytestqt.qtbot import QtBot

from ez2digitize.pipeline import Tools
from ez2digitize.ui.backends_dialog import COLMAP_KEY, OPENMVS_KEY, BackendsDialog
from ez2digitize.ui.main_window import LAST_PROJECT_KEY, MainWindow


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def window(qtbot: QtBot, settings: QSettings, fake_tools: Tools) -> MainWindow:
    window = MainWindow(settings=settings, tools_factory=lambda: fake_tools)
    qtbot.addWidget(window)
    return window


def test_starts_on_welcome_page(window: MainWindow) -> None:
    assert window.stack.currentWidget() is window.welcome
    assert window.page is None
    assert not window.import_action.isEnabled()


def test_new_and_open_project(window: MainWindow, tmp_path: Path, settings: QSettings) -> None:
    assert window.new_project(tmp_path / "skull")
    assert window.page is not None and window.stack.currentWidget() is window.page
    assert window.windowTitle() == "skull – EZ2DIGITIZE"
    assert window.import_action.isEnabled()
    assert settings.value(LAST_PROJECT_KEY) == str((tmp_path / "skull").absolute())

    assert window.close_project()
    assert window.page is None and window.stack.currentWidget() is window.welcome
    assert window.open_project(tmp_path / "skull")
    assert window.page is not None and window.page.project.name == "skull"


def test_open_errors_are_shown(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _p, title, text: warnings.append(title))
    assert not window.open_project(tmp_path / "nothing-here")
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "file.txt").write_text("x")
    assert not window.new_project(tmp_path / "full")
    assert warnings == ["Cannot open project", "Cannot create project"]
    assert window.page is None


def test_closing_while_running_asks_and_cancels(
    qtbot: QtBot, window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SLEEP", "global_mapper")
    window.new_project(tmp_path / "p")
    assert window.page is not None
    page = window.page
    (tmp_path / "photos").mkdir()
    (tmp_path / "photos" / "a.jpg").write_bytes(b"a")
    page.import_folder(tmp_path / "photos")
    page.start_run()
    qtbot.waitUntil(lambda: "mapping" in page._stage_items, timeout=20_000)

    answers = [QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes]
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: answers.pop(0))
    assert not window.close_project()  # said No: still running
    assert page.runner.running
    assert window.close_project()  # said Yes: cancelled and closed
    assert window.page is None


def test_backends_dialog_saves_and_checks(
    qtbot: QtBot, settings: QSettings, fake_tools: Tools, tmp_path: Path
) -> None:
    dialog = BackendsDialog(settings)
    qtbot.addWidget(dialog)
    dialog.colmap_edit.setText(str(tmp_path / "missing-colmap"))
    dialog.check()
    assert dialog.check_label.text().startswith("Problem:")
    assert settings.value(COLMAP_KEY, "") == ""  # check doesn't save
    dialog.openmvs_edit.setText(str(fake_tools.openmvs.bin_dir))
    dialog.save()
    assert settings.value(COLMAP_KEY) == str(tmp_path / "missing-colmap")
    assert settings.value(OPENMVS_KEY) == str(fake_tools.openmvs.bin_dir)


def test_licenses_dialog(qtbot: QtBot) -> None:
    from ez2digitize.ui.licenses_dialog import LicensesDialog

    dialog = LicensesDialog()
    qtbot.addWidget(dialog)
    assert dialog.tabs.count() == 2
    third_party = dialog.tabs.widget(0)
    assert "OpenMVS v2.4.0  AGPL-3.0" in third_party.toPlainText()  # type: ignore[union-attr]
    assert "No bundled backends" in dialog.summary.text()
