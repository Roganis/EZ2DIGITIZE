# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import io
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest
from model_files import SQUARE, png, textured_square
from PIL import Image as PILImage

from ez2digitize.core import meshfiles as mf
from ez2digitize.core.meshio import MeshFormatError

# --- PLY ----------------------------------------------------------------------------


def test_ascii_ply_with_a_quad_and_colours(tmp_path: Path) -> None:
    path = tmp_path / "quad.ply"
    path.write_text(
        "ply\nformat ascii 1.0\ncomment made by hand\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
        "0 0 0 255 0 0\n1 0 0 0 255 0\n1 1 0 0 0 255\n0 1 0 10 20 30\n4 0 1 2 3\n"
    )
    mesh = mf.read_ply(path)
    assert mesh.face_count == 2 and mesh.faces.tolist() == [[0, 1, 2], [0, 2, 3]]
    assert mesh.colors is not None and mesh.colors[3].tolist() == [10, 20, 30]
    assert mesh.uvs is None and not mesh.textured


def test_binary_big_endian_ply_with_mixed_polygons_and_float_colours(tmp_path: Path) -> None:
    header = (
        "ply\nformat binary_big_endian 1.0\nelement vertex 5\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property float red\nproperty float green\nproperty float blue\n"
        "element face 2\nproperty list uchar uint vertex_indices\n"
        "element extra 1\nproperty int something\nend_header\n"
    )
    body = b"".join(struct.pack(">3d3f", *p, 0.5, 0.25, 1.0) for p in [*SQUARE, (2, 0, 0)])
    body += struct.pack(">B4I", 4, 0, 1, 2, 3) + struct.pack(">B3I", 3, 1, 4, 2)
    body += struct.pack(">i", 7)
    (tmp_path / "mixed.ply").write_bytes(header.encode() + body)
    mesh = mf.read_ply(tmp_path / "mixed.ply")
    assert mesh.faces.tolist() == [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
    assert mesh.colors is not None and mesh.colors[0].tolist() == [128, 64, 255]


def test_ply_with_vertex_texture_coordinates(tmp_path: Path) -> None:
    (tmp_path / "skin.png").write_bytes(png((0, 0, 255)))
    header = (
        "ply\nformat binary_little_endian 1.0\ncomment TextureFile skin.png\n"
        "element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
        "property float s\nproperty float t\n"
        "element face 2\nproperty list uchar int vertex_indices\nend_header\n"
    )
    body = b"".join(struct.pack("<5f", *p, p[0], p[1]) for p in SQUARE)
    body += struct.pack("<B3i", 3, 0, 1, 2) + struct.pack("<B3i", 3, 0, 2, 3)
    (tmp_path / "uv.ply").write_bytes(header.encode() + body)
    mesh = mf.read_ply(tmp_path / "uv.ply")
    assert mesh.textured and mesh.uvs is not None
    assert mesh.uvs[1].tolist() == [[0, 0], [1, 1], [0, 1]]
    assert mf.vertex_colors(mesh)[2].tolist() == [0, 0, 255]


def test_ply_round_trip_keeps_texture_and_colours(tmp_path: Path) -> None:
    source = textured_square(((200, 10, 10), (10, 200, 10)))
    files = mf.write_ply(source, tmp_path / "out.ply")
    assert [f.name for f in files] == ["out.ply", "out_texture0.png", "out_texture1.png"]
    back = mf.read_ply(files[0])
    assert back.textured and back.face_materials is not None
    assert back.face_materials.tolist() == [0, 1]
    assert back.uvs is not None and np.allclose(back.uvs, source.uvs)  # type: ignore[arg-type]

    colored = mf.mesh(SQUARE, [(0, 1, 2), (0, 2, 3)], colors=[(1, 2, 3)] * 4)
    again = mf.read_ply(mf.write_ply(colored, tmp_path / "colors.ply")[0])
    assert again.colors is not None and again.colors.tolist() == [[1, 2, 3]] * 4


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
         "property float z\nend_header\n0 0 0\n", "no faces"),
        ("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
         "property float z\nelement face 1\nproperty list uchar int vertex_indices\n"
         "end_header\n0 0 0\n3 0 1 2\n", "doesn't exist"),
        ("ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\n"
         "property float z\nend_header\n0 0 0\n", "file ends"),
        ("ply\nformat binary_middle_endian 1.0\nend_header\n", "unknown PLY format"),
        ("not a ply\n", "not a PLY file"),
    ],
)  # fmt: skip
def test_ply_errors(tmp_path: Path, text: str, message: str) -> None:
    (tmp_path / "bad.ply").write_text(text)
    with pytest.raises(MeshFormatError, match=message):
        mf.read_ply(tmp_path / "bad.ply")


# --- OBJ ----------------------------------------------------------------------------


def test_obj_with_materials_polygons_and_negative_indices(tmp_path: Path) -> None:
    (tmp_path / "wood.png").write_bytes(png((120, 80, 40)))
    (tmp_path / "scan.mtl").write_text(
        "newmtl wood\nKd 1 1 1\nmap_Kd -s 1 1 1 -o 0 0 wood.png\n\nnewmtl red\nKd 1 0 0\nd 0.5\n"
    )
    (tmp_path / "scan.obj").write_text(
        "# a square and a triangle\nmtllib scan.mtl\n"
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 2 0 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
        "usemtl wood\nf 1/1 2/2 3/3 4/4\n"
        "usemtl red\nf -4 -1 -3\n"
    )
    mesh = mf.read_obj(tmp_path / "scan.obj")
    assert mesh.faces.tolist() == [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
    assert [m.name for m in mesh.materials] == ["wood", "red"]
    assert mesh.materials[0].image is not None and mesh.materials[0].image.name == "wood.png"
    assert mesh.materials[1].color == (1.0, 0.0, 0.0, 0.5) and mesh.materials[1].image is None
    assert mesh.face_materials is not None and mesh.face_materials.tolist() == [0, 0, 1]
    assert mesh.uvs is not None and mesh.uvs[2].tolist() == [[0, 0]] * 3  # no vt: zeros
    colors = mf.vertex_colors(mesh)
    assert colors[3].tolist() == [120, 80, 40]  # only on the textured square
    assert colors[4].tolist() == [255, 0, 0]  # only on the red triangle


def test_obj_vertex_colours_and_faces_without_material(tmp_path: Path) -> None:
    (tmp_path / "c.obj").write_text(
        "v 0 0 0 1 0 0\nv 1 0 0 0 1 0\nv 0 1 0 0 0 1\nf 1//1 2//1 3//1\nusemtl x\n"
    )
    mesh = mf.read_obj(tmp_path / "c.obj")
    assert mesh.colors is not None and mesh.colors.tolist()[0] == [255, 0, 0]
    assert mesh.uvs is None and mesh.materials == []


def test_obj_triangles_read_in_one_go(tmp_path: Path) -> None:
    (tmp_path / "t.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 1\nvn 0 0 1\n"
        "f 1/1/1 2/2/1 3/2/1\nf -4/-2/-1 -2/-1/-1 -1/-1/-1\n"
    )
    mesh = mf.read_obj(tmp_path / "t.obj")
    assert mesh.faces.tolist() == [[0, 1, 2], [0, 2, 3]]
    assert mesh.uvs is not None and mesh.uvs[1].tolist() == [[0, 0], [1, 1], [1, 1]]


def test_obj_round_trip(tmp_path: Path) -> None:
    source = textured_square(((200, 10, 10), (10, 200, 10)))
    files = mf.write_obj(source, tmp_path / "out" / "scan.obj")
    assert [f.name for f in files] == ["scan.obj", "scan.mtl", "scan_texture0.png",
                                       "scan_texture1.png"]  # fmt: skip
    back = mf.read_obj(files[0])
    assert back.faces.tolist() == source.faces.tolist()
    assert back.textured and back.face_materials is not None
    assert back.face_materials.tolist() == [0, 1]
    assert np.allclose(back.uvs, source.uvs)  # type: ignore[arg-type]


def test_obj_errors(tmp_path: Path) -> None:
    (tmp_path / "empty.obj").write_text("v 0 0 0\n")
    with pytest.raises(MeshFormatError, match="no faces"):
        mf.read_obj(tmp_path / "empty.obj")
    (tmp_path / "bad.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 x\n")
    with pytest.raises(MeshFormatError, match="line 4"):
        mf.read_obj(tmp_path / "bad.obj")
    (tmp_path / "far.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n")
    with pytest.raises(MeshFormatError, match="doesn't exist"):
        mf.read_obj(tmp_path / "far.obj")


# --- STL, OFF, 3MF -------------------------------------------------------------------


def test_stl_binary_round_trip_joins_corners(tmp_path: Path) -> None:
    source = mf.mesh(SQUARE, [(0, 1, 2), (0, 2, 3)])
    path = mf.write_stl(source, tmp_path / "s.stl")[0]
    assert path.stat().st_size == 84 + 2 * 50
    back = mf.read_stl(path)
    assert back.vertex_count == 4 and back.face_count == 2
    assert sorted(map(tuple, back.positions.tolist())) == sorted(SQUARE)


def test_stl_ascii_and_binary_starting_with_solid(tmp_path: Path) -> None:
    (tmp_path / "a.stl").write_text(
        "solid square\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\n"
        "vertex 1 1 0\nendloop\nendfacet\nendsolid square\n"
    )
    assert mf.read_stl(tmp_path / "a.stl").face_count == 1
    binary = tmp_path / "b.stl"
    mf.write_stl(mf.mesh(SQUARE, [(0, 1, 2)]), binary)
    data = bytearray(binary.read_bytes())
    data[:5] = b"solid"  # some exporters do this
    binary.write_bytes(bytes(data))
    assert mf.read_stl(binary).face_count == 1
    (tmp_path / "c.stl").write_bytes(b"\0" * 90)
    with pytest.raises(MeshFormatError, match="not an STL file"):
        mf.read_stl(tmp_path / "c.stl")


def test_off_reading_and_writing(tmp_path: Path) -> None:
    (tmp_path / "q.off").write_text(
        "COFF\n# a comment\n4 1 0\n0 0 0 255 0 0 255\n1 0 0 0 255 0 255\n"
        "1 1 0 0 0 255 255\n0 1 0 9 9 9 255\n4 0 1 2 3\n"
    )
    mesh = mf.read_off(tmp_path / "q.off")
    assert mesh.face_count == 2 and mesh.colors is not None
    assert mesh.colors[3].tolist() == [9, 9, 9]
    (tmp_path / "plain.off").write_text("OFF 3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
    assert mf.read_off(tmp_path / "plain.off").colors is None

    written = mf.write_off(textured_square(), tmp_path / "out.off")[0]
    text = written.read_text().splitlines()
    assert text[:2] == ["COFF", "4 2 0"] and text[2].endswith("200 10 10 255")
    assert mf.read_off(written).colors is not None
    (tmp_path / "bad.off").write_text("OFF\n3 1 0\n0 0 0\n")
    with pytest.raises(MeshFormatError, match="ends early"):
        mf.read_off(tmp_path / "bad.off")


def test_3mf_items_components_and_transforms(tmp_path: Path) -> None:
    model = """<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
<resources>
<object id="1" type="model"><mesh><vertices>
<vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/>
</vertices><triangles><triangle v1="0" v2="1" v3="2"/></triangles></mesh></object>
<object id="2" type="model"><components>
<component objectid="1" transform="1 0 0 0 1 0 0 0 1 10 0 0"/>
</components></object>
</resources>
<build><item objectid="1"/><item objectid="2" transform="1 0 0 0 1 0 0 0 1 0 0 5"/></build>
</model>"""
    with zipfile.ZipFile(tmp_path / "two.3mf", "w") as package:
        package.writestr("3D/3dmodel.model", model)
    mesh = mf.read_3mf(tmp_path / "two.3mf")
    assert mesh.face_count == 2 and mesh.faces.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert mesh.positions[3].tolist() == [10.0, 0.0, 5.0]

    written = mf.write_3mf(mf.mesh(SQUARE, [(0, 1, 2)]), tmp_path / "out.3mf")[0]
    assert mf.read_3mf(written).face_count == 1
    (tmp_path / "bad.3mf").write_bytes(b"not a zip")
    with pytest.raises(MeshFormatError, match="not a 3MF file"):
        mf.read_3mf(tmp_path / "bad.3mf")


# --- helpers ------------------------------------------------------------------------


def test_transformed_keeps_faces_facing_out_under_a_mirror() -> None:
    source = textured_square()
    mirror = np.diag([1.0, 1.0, -1.0])
    turned = mf.transformed(source, mirror, 2.0)
    assert turned.positions[2].tolist() == [2.0, 2.0, 0.0]
    assert turned.faces.tolist() == [[2, 1, 0], [3, 2, 0]]
    assert turned.uvs is not None and turned.uvs[0, 0].tolist() == [1, 1]


def test_web_image_converts_other_types() -> None:
    out = io.BytesIO()
    PILImage.new("RGB", (2, 2), (1, 2, 3)).save(out, "BMP")
    image = mf.web_image(mf.Image("old.bmp", out.getvalue()))
    assert image.name == "old.png" and image.mime == "image/png"
    with pytest.raises(MeshFormatError, match="cannot read the texture"):
        mf.web_image(mf.Image("broken.tga", b"nothing"))


def test_mesh_check_and_size(tmp_path: Path) -> None:
    with pytest.raises(MeshFormatError, match="no triangles"):
        mf.mesh(SQUARE, np.zeros((0, 3))).check(tmp_path)
    with pytest.raises(MeshFormatError, match="material that doesn't exist"):
        mf.mesh(SQUARE, [(0, 1, 2)], face_materials=[0]).check(tmp_path)
    assert mf.size_text(mf.mesh(SQUARE, [(0, 1, 2)])) == "1 x 1 x 0"
