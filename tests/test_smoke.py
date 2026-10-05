# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from importlib.metadata import version

from pytestqt.qtbot import QtBot

import ez2digitize
from ez2digitize.ui.main_window import MainWindow


def test_package_version_matches_metadata() -> None:
    assert version("ez2digitize") == ez2digitize.__version__


def test_main_window_opens(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    assert window.isVisible()
    assert window.windowTitle() == "EZ2DIGITIZE"
