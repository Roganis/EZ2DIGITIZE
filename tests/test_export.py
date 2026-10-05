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
        export_mesh(built, ["step"])  # type: ignore[list-item]


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


def test_print_and_point_cloud_exports(built: Project) -> None:
    import struct
    import xml.etree.ElementTree as ET
    import zipfile

    from ez2digitize.export import export_notes

    files = export_mesh(built, ["stl", "3mf", "points"], now=NOW)
    folder = built.exports_dir / "20261005-180000"
    assert sorted(f.name for f in files) == ["My_Skull.3mf", "My_Skull.stl", "My_Skull_points.ply"]
    stl = (folder / "My_Skull.stl").read_bytes()
    assert struct.unpack("<I", stl[80:84]) == (1,) and len(stl) == 84 + 50
    with zipfile.ZipFile(folder / "My_Skull.3mf") as package:
        assert "[Content_Types].xml" in package.namelist()
        model = ET.fromstring(package.read("3D/3dmodel.model"))
    ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    assert len(model.findall(".//m:vertex", ns)) == 3
    assert model.find(".//m:triangle", ns).attrib == {"v1": "0", "v2": "1", "v3": "2"}  # type: ignore[union-attr]
    info = json.loads((folder / "export.json").read_text())
    # One triangle: three open edges.
    assert (info["watertight"], info["open_edges"]) == (False, 3)
    (note,) = export_notes(files)
    assert "not closed (3 open" in note


def test_watertight_check() -> None:
    from array import array

    from ez2digitize.core.meshio import TexturedMesh, check_watertight

    # A tetrahedron is closed; remove a face and three edges are open.
    faces = [0, 2, 1, 0, 1, 3, 1, 2, 3, 0, 3, 2]

    def mesh(f: list[int]) -> TexturedMesh:
        positions = array("f", [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
        return TexturedMesh(positions, array("I", f), array("f"), array("I"), [])

    assert check_watertight(mesh(faces)).watertight
    opened = check_watertight(mesh(faces[3:]))
    assert (opened.open_edges, opened.non_manifold_edges) == (3, 0)
    fan = check_watertight(mesh([*faces, 0, 1, 2]))
    assert fan.non_manifold_edges == 3
