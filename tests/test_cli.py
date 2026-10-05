# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import pytest

from ez2digitize.cli import main


def test_new_import_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    photos, masks = tmp_path / "photos", tmp_path / "masks"
    photos.mkdir()
    masks.mkdir()
    for name in ("a.jpg", "b.jpg", "notes.txt"):
        (photos / name).write_bytes(name.encode())
    (masks / "a.jpg.png").write_bytes(b"m")

    assert main(["new", str(tmp_path / "scan"), "--name", "Skull"]) == 0
    assert main(["import", str(tmp_path / "scan"), str(photos), "--masks", str(masks)]) == 0
    assert main(["status", str(tmp_path / "scan")]) == 0
    out = capsys.readouterr().out
    assert "created project 'Skull'" in out
    assert "2 files -> capture" in out and "skipped (not an image or video): notes.txt" in out
    assert "1 of 2 masks imported" in out
    assert "folder   2 images, 0 videos" in out
    assert "features    -" in out


def test_errors_are_reported_not_raised(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status", str(tmp_path / "missing")]) == 1
    assert "error:" in capsys.readouterr().err
    (tmp_path / "x").mkdir()
    assert main(["import", str(tmp_path / "p"), "a", "b", "--masks", str(tmp_path)]) == 2
