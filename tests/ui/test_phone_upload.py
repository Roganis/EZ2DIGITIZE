# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import urllib.request
from pathlib import Path

import pytest
from PySide6.QtWidgets import QDialog
from pytestqt.qtbot import QtBot

from ez2digitize import upload
from ez2digitize.core.capture import list_bundles
from ez2digitize.core.project import Project
from ez2digitize.ui.phone_upload import PhoneUploadDialog, qr_pixmap


@pytest.fixture
def dialog(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PhoneUploadDialog:
    monkeypatch.setattr(upload, "local_address", lambda: "127.0.0.1")
    widget = PhoneUploadDialog(Project.create(tmp_path / "p"))
    qtbot.addWidget(widget)
    return widget


def _send(dialog: PhoneUploadDialog, name: str, data: bytes) -> None:
    file_id = name.replace(".", "_")
    url = f"{dialog.session.url}files/{file_id}?offset=0&name={name}&size={len(data)}"
    request = urllib.request.Request(url, data=data, method="PUT")
    with urllib.request.urlopen(request, timeout=10):
        pass


def test_qr_code(qtbot: QtBot) -> None:  # QPixmap needs the QApplication
    pixmap = qr_pixmap("http://192.168.1.20:41234/token/")
    assert not pixmap.isNull() and pixmap.width() > 100


def test_receive_and_import(dialog: PhoneUploadDialog) -> None:
    assert not dialog.import_button.isEnabled()
    _send(dialog, "a.jpg", b"photo a")
    _send(dialog, "b.jpg", b"photo b")
    dialog.refresh()
    assert dialog.import_button.text() == "Import 2"
    assert dialog.files.topLevelItemCount() == 2
    dialog.import_button.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    received = dialog.received
    assert received is not None and received.bundle is not None
    assert [f.name for f in received.bundle.files] == ["a.jpg", "b.jpg"]
    assert list_bundles(dialog.session.project)[0].source == "upload"


def test_cancel_discards(dialog: PhoneUploadDialog) -> None:
    _send(dialog, "a.jpg", b"photo a")
    dialog.reject()
    assert list_bundles(dialog.session.project) == []
    assert not dialog.session.folder.exists()
