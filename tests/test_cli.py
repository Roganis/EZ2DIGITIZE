# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import sys
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


def test_export_without_mesh_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["new", str(tmp_path / "scan")]) == 0
    assert main(["export", str(tmp_path / "scan"), "--formats", "glb"]) == 1
    assert "build the mesh first" in capsys.readouterr().err


def test_format_list_parsing() -> None:
    import argparse

    from ez2digitize.cli import _formats

    assert _formats("obj,GLB") == ("obj", "glb")
    assert _formats("none") == ()
    with pytest.raises(argparse.ArgumentTypeError, match="unknown format 'stl'"):
        _formats("obj,stl")


def _tool(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho '{text}'\n")
    path.chmod(0o755)
    return path


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executables")
def test_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    colmap = _tool(tmp_path / "colmap", "COLMAP 4.2.1 (Commit abc)")
    for name in ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh",
                 "TextureMesh"):  # fmt: skip
        _tool(tmp_path / "mvs" / name, "12:00:00 [App     ] OpenMVS x64 v2.4.0")
    monkeypatch.setenv("EZ2D_COLMAP", str(colmap))
    monkeypatch.setenv("EZ2D_OPENMVS_DIR", str(tmp_path / "mvs"))
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert f"COLMAP 4.2.1: ok, {colmap}" in out and "OpenMVS 2.4.0: ok" in out

    _tool(colmap, "COLMAP 3.9.1 -- SfM")
    monkeypatch.setenv("EZ2D_OPENMVS_DIR", str(tmp_path / "nothing"))
    assert main(["check"]) == 1
    out = capsys.readouterr().out
    assert "not the tested version 4.2.1" in out and "OpenMVS: OpenMVS not found" in out


def test_commands_list_matches_parser() -> None:
    from ez2digitize.cli import commands

    assert set(commands()) == {"new", "import", "run", "export", "check", "status"}
