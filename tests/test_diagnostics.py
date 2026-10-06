# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import json
import zipfile
from pathlib import Path

import pytest

from ez2digitize import diagnostics
from ez2digitize.cli import main
from ez2digitize.core.capture import import_folder
from ez2digitize.core.project import Project


@pytest.fixture
def project(tmp_path: Path) -> Project:
    project = Project.create(tmp_path / "scan")
    photos = tmp_path / "photos"
    photos.mkdir()
    (photos / "a.jpg").write_bytes(b"secret photo")
    import_folder(project, photos)
    stage = project.stage_dir("features")
    stage.mkdir(parents=True)
    (stage / "stage.json").write_text("{}")
    (stage / "log.txt").write_text("x" * 100 + "the end\n", newline="\n")  # no CRLF
    (stage / "database.db").write_bytes(b"big")
    texture = project.stage_dir("texture")
    texture.mkdir(parents=True)
    (texture / "TextureMesh-123.log").write_text("openmvs log")
    (texture / "scene_textured.ply").write_bytes(b"mesh")
    return project


def test_zip_has_logs_and_settings_but_no_photos(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagnostics, "LOG_LIMIT", 20)
    path = diagnostics.write_diagnostics(
        project, tmp_path / "out" / "d.zip", tools=lambda: {"colmap": {"version": "4.2.1"}}
    )
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        capture = next(n for n in names if n.endswith("capture.json"))
        assert names == {
            "project.json", capture, "stages/features/stage.json", "stages/features/log.txt",
            "stages/texture/TextureMesh-123.log", "report.json",
        }  # fmt: skip
        log = archive.read("stages/features/log.txt").decode()
        assert log.startswith("[first 88 bytes left out]") and log.endswith("the end\n")
        report = json.loads(archive.read("report.json"))
    assert report["tools"] == {"colmap": {"version": "4.2.1"}}
    assert report["system"]["cpu_threads"] >= 1 and "gpus" in report["system"]
    # Only photos already inspected are reported; exporting never inspects.
    assert report["photo_checks"] == []
    assert not path.with_name("d.zip.partial").exists()


def test_cli(project: Project, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "diag.zip"
    assert main(["diagnostics", str(project.root), "-o", str(target)]) == 0
    assert zipfile.is_zipfile(target)
    assert "no photos" in capsys.readouterr().out
