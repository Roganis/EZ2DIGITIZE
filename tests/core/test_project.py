# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import json
from pathlib import Path
from typing import Any

import pytest

from ez2digitize.core import project as project_mod
from ez2digitize.core.project import SCHEMA_VERSION, Project, ProjectError, migrate


def test_create_writes_layout_and_project_json(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "scan", name="Goat skull")
    for sub in ("captures", "masks", "stages", "exports"):
        assert (tmp_path / "scan" / sub).is_dir()
    data = json.loads((tmp_path / "scan" / "project.json").read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["name"] == "Goat skull"
    assert data["settings"] == {}
    assert project.root == (tmp_path / "scan").absolute()


def test_create_defaults_name_to_folder(tmp_path: Path) -> None:
    assert Project.create(tmp_path / "my-scan").name == "my-scan"


def test_create_refuses_non_empty_folder(tmp_path: Path) -> None:
    (tmp_path / "stuff.txt").write_text("x")
    with pytest.raises(ProjectError, match="not an empty folder"):
        Project.create(tmp_path)


def test_create_accepts_empty_existing_folder(tmp_path: Path) -> None:
    Project.create(tmp_path)
    assert (tmp_path / "project.json").is_file()


def test_save_and_open_round_trip(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "p")
    project.preset = "low-memory"
    project.settings["openmvs"] = {"resolution_level": 2}
    project.save()
    reopened = Project.open(tmp_path / "p")
    assert reopened.preset == "low-memory"
    assert reopened.settings == {"openmvs": {"resolution_level": 2}}
    assert reopened.created == project.created
    assert not list((tmp_path / "p").glob("*.bak"))


def test_open_missing_project(tmp_path: Path) -> None:
    with pytest.raises(ProjectError, match="does not exist"):
        Project.open(tmp_path)


def test_open_corrupt_project(tmp_path: Path) -> None:
    (tmp_path / "project.json").write_text("{not json")
    with pytest.raises(ProjectError, match="cannot read"):
        Project.open(tmp_path)


def test_open_rejects_newer_format(tmp_path: Path) -> None:
    Project.create(tmp_path / "p")
    path = tmp_path / "p" / "project.json"
    data = json.loads(path.read_text())
    data["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(data))
    with pytest.raises(ProjectError, match="newer version"):
        Project.open(tmp_path / "p")


@pytest.mark.parametrize("bad", [None, "1", 0, True])
def test_migrate_rejects_invalid_version(bad: Any) -> None:
    with pytest.raises(ProjectError, match="invalid schema_version"):
        migrate({"schema_version": bad})


def test_migrate_applies_steps_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    def v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
        data["title"] = data.pop("name")
        return data

    def v2_to_v3(data: dict[str, Any]) -> dict[str, Any]:
        data["name"] = data.pop("title").upper()
        return data

    monkeypatch.setattr(project_mod, "MIGRATIONS", {1: v1_to_v2, 2: v2_to_v3})
    assert migrate({"schema_version": 1, "name": "a"}, target=3) == {
        "schema_version": 3,
        "name": "A",
    }


def test_migrate_missing_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(project_mod, "MIGRATIONS", {})
    with pytest.raises(ProjectError, match="no migration from format 1"):
        migrate({"schema_version": 1}, target=2)


def test_open_migrates_and_keeps_backup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    Project.create(tmp_path / "p")
    old_text = (tmp_path / "p" / "project.json").read_text()

    def add_preset(data: dict[str, Any]) -> dict[str, Any]:
        data["preset"] = "migrated"
        return data

    monkeypatch.setattr(project_mod, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(project_mod, "MIGRATIONS", {1: add_preset})
    project = Project.open(tmp_path / "p")
    assert project.preset == "migrated"
    assert (tmp_path / "p" / "project.json.v1.bak").read_text() == old_text
    assert json.loads((tmp_path / "p" / "project.json").read_text())["schema_version"] == 2


def test_stage_dir_validates_names(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "p")
    assert project.stage_dir("01-features") == project.stages_dir / "01-features"
    for bad in ("", "../x", "a/b", ".hidden", "Features"):
        with pytest.raises(ProjectError):
            project.stage_dir(bad)
