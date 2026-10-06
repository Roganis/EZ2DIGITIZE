# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""No modal dialogs in the UI tests unless a test expects one.

A message box nobody closes waits forever: a test that hit an unexpected
error would hang CI instead of failing. Tests that expect a dialog patch
it themselves (their monkeypatch comes after this one). Information boxes
are only recorded: some flows end with one.
"""

from typing import NoReturn

import pytest
from PySide6.QtWidgets import QMessageBox


@pytest.fixture(autouse=True)
def _no_dialogs(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(kind: str) -> object:
        def show(_parent: object, title: str, text: str, *_args: object) -> NoReturn:
            # Raised in a Qt slot, pytest-qt fails the test with it.
            raise AssertionError(f"unexpected {kind} dialog {title!r}: {text}")

        return show

    for kind in ("warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, kind, unexpected(kind))
    monkeypatch.setattr(QMessageBox, "information", lambda *_a, **_k: QMessageBox.StandardButton.Ok)
