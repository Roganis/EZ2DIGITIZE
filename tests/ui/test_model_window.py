# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import pytest
import shiboken6
from model_files import random_splats, textured_square
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog, QMessageBox
from pytestqt.qtbot import QtBot

from ez2digitize import models, views
from ez2digitize.core import gltf, meshfiles
from ez2digitize.core import splats as sp
from ez2digitize.ui import model_window
from ez2digitize.ui.main_window import MainWindow
from ez2digitize.ui.model_window import ModelWindow
from ez2digitize.ui.viewer import ViewerWidget


@pytest.fixture
def shown(monkeypatch: pytest.MonkeyPatch) -> list[views.View]:
    """The views windows ask their viewer to draw (the page itself isn't needed)."""
    calls: list[views.View] = []
    monkeypatch.setattr(ViewerWidget, "show_view", lambda _self, view: calls.append(view))
    return calls


def test_shows_the_file_and_turns_it(qtbot: QtBot, tmp_path: Path, shown: list[views.View]) -> None:
    stl = meshfiles.write_stl(textured_square(), tmp_path / "part.stl")[0]
    window = ModelWindow(models.identify(stl))
    qtbot.addWidget(window)
    assert window.windowTitle() == "part.stl – EZ2DIGITIZE"
    ups: list[models.Up] = [window.chosen_up]
    assert window.convert_button.isEnabled()
    assert shown[-1].key == "file" and shown[-1].upright == models.UP_MATRICES["z"]
    window.up.setCurrentIndex(0)  # Y up
    ups.append(window.chosen_up)
    assert ups == ["z", "y"] and shown[-1].upright is None


def test_convert(
    qtbot: QtBot, tmp_path: Path, shown: list[views.View], monkeypatch: pytest.MonkeyPatch
) -> None:
    glb = gltf.write_glb(textured_square(), tmp_path / "scan.glb")[0]
    window = ModelWindow(models.identify(glb))
    qtbot.addWidget(window)
    files = window.convert(tmp_path / "scan.usdz")
    assert files == [tmp_path / "scan.usdz"] and "Saved scan.usdz" in window.status.text()

    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_args: (str(tmp_path / "chosen"), "OBJ (with MTL and images) (*.obj)"),
    )
    files = window.convert()
    assert files is not None and files[0] == tmp_path / "chosen.obj"

    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _p, _title, text: warnings.append(text))
    assert window.convert(glb) is None
    assert warnings and "replace the original" in warnings[0]


def test_files_that_only_show(qtbot: QtBot, tmp_path: Path, shown: list[views.View]) -> None:
    (tmp_path / "scene.sog").write_bytes(b"PK")
    window = ModelWindow(models.identify(tmp_path / "scene.sog"))
    qtbot.addWidget(window)
    assert not window.convert_button.isEnabled() and window.convert() is None
    assert shown[-1].kind == "splat" and shown[-1].upright == models.UP_MATRICES["-y"]


def test_main_window_opens_model_files(
    qtbot: QtBot, tmp_path: Path, shown: list[views.View], monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow(settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(window)
    spz = sp.write_spz(random_splats(4, k=0), tmp_path / "s.spz")
    opened = window.open_model(spz)
    assert opened is not None and opened in window.model_windows and opened.isVisible()
    assert shown[-1].source == spz
    opened.close()  # it deletes itself
    qtbot.waitUntil(lambda: not shiboken6.isValid(opened))
    again = window.open_model(spz)
    assert window.model_windows == [again]

    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _p, title, _text: warnings.append(title))
    (tmp_path / "notes.txt").write_text("x")
    assert window.open_model(tmp_path / "notes.txt") is None
    assert warnings == ["Cannot open the file"]
    assert "*.glb" in model_window.open_dialog_filter()
