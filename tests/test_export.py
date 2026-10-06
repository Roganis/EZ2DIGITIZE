# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import json
from datetime import datetime
from pathlib import Path

import pytest

from ez2digitize import pipeline
from ez2digitize.core.capture import import_files
from ez2digitize.core.project import Project
from ez2digitize.export import ExportError, _file_stem, export_mesh
from ez2digitize.pipeline import MeshSettings, Tools

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
    assert sorted(f.relative_to(folder).as_posix() for f in files) == [
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
    closed, unscaled = export_notes(files)
    assert "not closed (3 open" in closed
    assert "no scale is set" in unscaled and info["scale_mm_per_unit"] is None


def _stl_vertices(path: Path) -> list[float]:
    import struct

    data = path.read_bytes()
    (count,) = struct.unpack("<I", data[80:84])
    values: list[float] = []
    for i in range(count):
        values += struct.unpack("<9f", data[84 + 50 * i + 12 : 84 + 50 * i + 48])
    return values


def test_scale_gives_real_units(built: Project) -> None:
    from ez2digitize import crop, scale
    from ez2digitize.export import export_notes

    plain = export_mesh(built, ["stl", "glb"], align=False, now=NOW)
    run = crop.camera_run(built)
    assert run is not None
    # 2 reconstruction units are 10 mm: 5 mm per unit.
    scale.save(built, scale.make(((0, 0, 0), (0, 2, 0)), 10.0, run))
    scaled = export_mesh(built, ["stl", "glb"], align=False, now=NOW)
    assert scaled[0].parent != plain[0].parent  # not the unscaled export again
    stl = _stl_vertices(scaled[1])
    assert stl == pytest.approx([v * 5 for v in _stl_vertices(plain[1])])  # millimetres
    info = json.loads((scaled[0].parent / "export.json").read_text())
    assert info["scale_mm_per_unit"] == 5.0 and info["units"]["stl"] == "mm"
    assert export_notes(scaled) == [
        "the mesh is not closed (3 open and 0 non-manifold edges): "
        "a slicer may need to repair it before printing"
    ]
    assert export_mesh(built, ["stl", "glb"], align=False) == scaled  # reused


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


def test_alignment_is_recorded_and_part_of_reuse(built: Project) -> None:
    aligned = export_mesh(built, ["glb"], now=NOW)
    info = json.loads((aligned[0].parent / "export.json").read_text())
    assert info["align"] is True and "placement" in info
    raw = export_mesh(built, ["glb"], align=False, now=NOW)
    assert raw[0].parent != aligned[0].parent
    assert export_mesh(built, ["glb"], align=False) == raw


def test_point_cloud_round_trip(tmp_path: Path) -> None:
    from array import array

    from ez2digitize.core.meshio import PointCloud, read_point_cloud, write_point_cloud
    from ez2digitize.orientation import Placement

    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property list uchar uint view_indices\nend_header\n"
    )
    import struct

    body = b"".join(
        struct.pack("<3f3B3fBII", x, 0, 0, 255, 128, 0, 0, 1, 0, 2, 7, 9) for x in (1.0, 2.0)
    )
    (tmp_path / "dense.ply").write_bytes(header.encode() + body)
    cloud = read_point_cloud(tmp_path / "dense.ply")
    assert list(cloud.positions) == [1, 0, 0, 2, 0, 0]
    assert cloud.colors == bytes([255, 128, 0] * 2) and cloud.normals is not None
    flip = Placement(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), (0.0, 5.0, 0.0))
    cloud.positions = flip.apply(cloud.positions)
    write_point_cloud(cloud, tmp_path / "out.ply")
    again = read_point_cloud(tmp_path / "out.ply")
    assert list(again.positions) == [1, 5, 0, 2, 5, 0] and again.colors == cloud.colors
    assert isinstance(again, PointCloud) and list(again.normals or array("f")) == [0, 1, 0] * 2


def test_orientation_correction_is_used(built: Project) -> None:
    from ez2digitize import upright

    before = export_mesh(built, ["stl"], now=NOW)
    upright.change(built, upright.tilted(upright.starting_point(built), "x"))
    rotation = upright.rotation(built)
    assert rotation is not None
    after = export_mesh(built, ["stl"], now=NOW)
    assert after[0].parent != before[0].parent  # not the export stood up the old way
    info = json.loads((after[0].parent / "export.json").read_text())
    assert info["upright"] == [list(row) for row in rotation]
    assert info["placement"]["rotation"] == info["upright"]
    assert export_mesh(built, ["stl"]) == after  # reused while nothing changes
