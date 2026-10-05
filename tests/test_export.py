# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from ez2digitize import pipeline
from ez2digitize.core.capture import import_files
from ez2digitize.core.project import Project
from ez2digitize.export import ExportError, _file_stem, export_mesh
from ez2digitize.pipeline import MeshSettings, Tools

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX fake backends")
NOW = datetime(2026, 10, 5, 18, 0, 0)


@pytest.fixture
def built(tmp_path: Path, fake_tools: Tools) -> Project:
    """A project whose texture stage has run (fake backends), not yet exported."""
    project = Project.create(tmp_path / "My Skull")
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(name.encode())
    import_files(project, [tmp_path / "a.jpg", tmp_path / "b.jpg"], source="folder")
    pipeline.run_mesh(project, fake_tools, MeshSettings(export_formats=()))
    return project


def test_no_mesh_yet(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="build the mesh first"):
        export_mesh(Project.create(tmp_path / "p"))


def test_formats_are_checked(built: Project) -> None:
    with pytest.raises(ExportError, match="choose formats"):
        export_mesh(built, [])
    with pytest.raises(ExportError, match="choose formats"):
        export_mesh(built, ["stl"])  # type: ignore[list-item]


def test_export_all_formats(built: Project) -> None:
    files = export_mesh(built, ["obj", "glb", "ply"], now=NOW)
    folder = built.exports_dir / "20261005-180000"
    assert sorted(str(f.relative_to(folder)) for f in files) == [
        "My_Skull.glb",
        "obj/My_Skull.mtl",
        "obj/My_Skull.obj",
        "obj/My_Skull_texture0.png",
        "ply/My_Skull.ply",
        "ply/scene_textured0.png",
    ]
    info = json.loads((folder / "export.json").read_text())
    texture_run = json.loads((built.stage_dir("texture") / "stage.json").read_text())["run_id"]
    assert info["source"] == {"stage": "texture", "run_id": texture_run}
    assert info["formats"] == ["obj", "glb", "ply"]
    assert (info["vertices"], info["faces"]) == (3, 1)


def test_same_export_is_reused_other_formats_are_not(built: Project) -> None:
    first = export_mesh(built, ["glb"], now=NOW)
    assert export_mesh(built, ["glb"], now=NOW) == first
    other = export_mesh(built, ["obj"], now=NOW)
    assert other[0].parent.parent.name == "20261005-180000-2"
    (first[0]).unlink()  # a deleted export is made again
    assert export_mesh(built, ["glb"], now=NOW)[0].exists()


@pytest.mark.parametrize(
    ("name", "stem"),
    [("My Skull", "My_Skull"), ("crâne/v2", "crâne_v2"), ("...", "mesh"), ("a.b-c", "a.b-c")],
)
def test_file_stem(name: str, stem: str) -> None:
    assert _file_stem(name) == stem
