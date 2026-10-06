# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox
from pytestqt.qtbot import QtBot
from scripts import FAKE_POSES, make_plugin

from ez2digitize import plugins
from ez2digitize.ui.plugins_dialog import PluginLicenseDialog, PluginsDialog


def _source(tmp_path: Path) -> Path:
    return make_plugin(
        tmp_path / "src" / "vggt",
        provides="poses",
        script=FAKE_POSES,
        command=["bin/run"],
        licenses=(("code", "Apache-2.0"), ("model weights", "CC-BY-NC-4.0")),
    )


def test_license_dialog(qtbot: QtBot, tmp_path: Path) -> None:
    plugin = plugins.read_plugin(_source(tmp_path))
    dialog = PluginLicenseDialog(plugin)
    qtbot.addWidget(dialog)
    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == [
        "code: Apache-2.0",
        "model weights: CC-BY-NC-4.0 ⚠",
    ]


def test_install_accept_and_choose(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = [False, True]  # decline at install, then accept when choosing it
    shown: list[str] = []

    def exec_license(self: PluginLicenseDialog) -> int:
        shown.append(self.plugin.id)
        return 1 if answers.pop(0) else 0

    monkeypatch.setattr(PluginLicenseDialog, "exec", exec_license)
    dialog = PluginsDialog()
    qtbot.addWidget(dialog)
    poses = dialog.choices["poses"]
    assert [poses.itemText(i) for i in range(poses.count())] == ["COLMAP (built in)"]

    plugin = dialog.install(_source(tmp_path))
    assert plugin is not None and shown == ["vggt"]
    assert not plugins.accepted(plugin)
    item = dialog.table.topLevelItem(0)
    assert item is not None and item.text(3) == "Licenses not accepted"
    assert poses.itemText(1) == "Test poses 1.0 (plugin)"

    # Choosing it shows the licenses again; accepted, it is used.
    poses.setCurrentIndex(1)
    poses.activated.emit(1)
    assert shown == ["vggt", "vggt"] and plugins.chosen_id("poses") == "vggt"
    assert poses.currentData() == "vggt"
    item = dialog.table.topLevelItem(0)
    assert item is not None and item.text(3) == "Licenses accepted"

    # Back to the built-in one.
    poses.setCurrentIndex(0)
    poses.activated.emit(0)
    assert plugins.chosen_id("poses") is None

    # Installing it again offers to replace it; removing it asks first.
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_a: QMessageBox.StandardButton.Yes,
    )
    assert dialog.install(_source(tmp_path)) is not None
    assert shown == ["vggt", "vggt"]  # same licenses, still accepted
    dialog.table.topLevelItem(0).setSelected(True)  # type: ignore[union-attr]
    assert dialog.remove_button.isEnabled()
    dialog.remove_button.click()
    assert plugins.installed().plugins == () and dialog.table.topLevelItemCount() == 0


def test_declining_keeps_the_built_in(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugins.install(_source(tmp_path))
    monkeypatch.setattr(PluginLicenseDialog, "exec", lambda _self: 0)
    dialog = PluginsDialog()
    qtbot.addWidget(dialog)
    poses = dialog.choices["poses"]
    poses.setCurrentIndex(1)
    poses.activated.emit(1)
    assert plugins.chosen_id("poses") is None and poses.currentIndex() == 0


def test_bad_source_is_reported(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _p, _t, text: warnings.append(text))
    dialog = PluginsDialog()
    qtbot.addWidget(dialog)
    assert dialog.install(tmp_path) is None
    assert warnings and "not a plugin" in warnings[0]
