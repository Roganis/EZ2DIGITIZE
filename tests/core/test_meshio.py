# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import json
import struct
from pathlib import Path
from typing import Any

import pytest

from ez2digitize.core.meshio import MeshFormatError, read_openmvs_ply, write_glb, write_obj

# Two triangles sharing an edge, on two textures; vertex 1 and 2 are shared,
# and vertex 1 has the same UV in both faces (merged in GLB), vertex 2 not.
VERTICES = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
FACES = [
    ((0, 1, 2), (0.0, 0.0, 1.0, 0.0, 0.0, 1.0), 0),
    ((1, 3, 2), (1.0, 0.0, 1.0, 1.0, 0.5, 1.0), 1),
]


def write_ply(
    path: Path,
    *,
    textures: tuple[str, ...] = ("scene_textured0.png", "scene_textured1.png"),
    texnumber: bool = True,
    normals: bool = True,
    fmt: str = "binary_little_endian",
    face_size: int = 3,
) -> Path:
    lines = ["ply", f"format {fmt} 1.0", *(f"comment TextureFile {t}" for t in textures)]
    lines += [f"element vertex {len(VERTICES)}", "property float x", "property float y",
              "property float z"]  # fmt: skip
    if normals:
        lines += ["property float nx", "property float ny", "property float nz"]
    lines += [f"element face {len(FACES)}", "property list uchar uint vertex_indices",
              "property list uchar float texcoord"]  # fmt: skip
    if texnumber:
        lines.append("property uchar texnumber")
    lines.append("end_header")
    body = b""
    for v in VERTICES:
        body += struct.pack("<3f", *v) + (struct.pack("<3f", 0, 0, 1) if normals else b"")
    for indices, uv, tex in FACES:
        idx = (*indices, 0)[:face_size] if face_size <= 3 else (*indices, 0)
        body += struct.pack(f"<B{face_size}I", face_size, *idx)
        body += struct.pack("<B6f", 6, *uv) + (struct.pack("<B", tex) if texnumber else b"")
    path.write_bytes(("\n".join(lines) + "\n").encode() + body)
    for t in textures:
        (path.parent / t).write_bytes(b"\x89PNG " + t.encode())
    return path


def test_read(tmp_path: Path) -> None:
    mesh = read_openmvs_ply(write_ply(tmp_path / "m.ply"))
    assert mesh.vertex_count == 4 and mesh.face_count == 2
    assert list(mesh.positions[3:6]) == [1.0, 0.0, 0.0]
    assert list(mesh.faces) == [0, 1, 2, 1, 3, 2]
    assert list(mesh.uvs[6:]) == [1.0, 0.0, 1.0, 1.0, 0.5, 1.0]
    assert list(mesh.texture_ids) == [0, 1]
    assert [t.name for t in mesh.textures] == ["scene_textured0.png", "scene_textured1.png"]


def test_read_single_texture_without_texnumber(tmp_path: Path) -> None:
    path = write_ply(tmp_path / "m.ply", textures=("t.png",), texnumber=False, normals=False)
    mesh = read_openmvs_ply(path)
    assert list(mesh.texture_ids) == [0, 0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fmt": "ascii"}, "binary little-endian"),
        ({"textures": ()}, "no texture"),
        ({"textures": ("only-one.png",)}, "uses texture 1"),
        ({"face_size": 4}, "only triangles"),
    ],
)
def test_read_rejects(tmp_path: Path, kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(MeshFormatError, match=message):
        read_openmvs_ply(write_ply(tmp_path / "m.ply", **kwargs))


def test_read_truncated(tmp_path: Path) -> None:
    path = write_ply(tmp_path / "m.ply")
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(MeshFormatError, match="ends inside the face list"):
        read_openmvs_ply(path)


def test_write_obj(tmp_path: Path) -> None:
    mesh = read_openmvs_ply(write_ply(tmp_path / "m.ply"))
    files = write_obj(mesh, tmp_path / "out", "skull")
    assert [f.name for f in files] == [
        "skull.obj", "skull.mtl", "skull_texture0.png", "skull_texture1.png"
    ]  # fmt: skip
    assert (tmp_path / "out" / "skull_texture1.png").read_bytes() == b"\x89PNG scene_textured1.png"
    mtl = (tmp_path / "out" / "skull.mtl").read_text()
    assert "newmtl material_1" in mtl and "map_Kd skull_texture1.png" in mtl
    lines = (tmp_path / "out" / "skull.obj").read_text().splitlines()
    assert lines[1] == "mtllib skull.mtl"
    assert sum(line.startswith("v ") for line in lines) == 4
    assert [line for line in lines if line.startswith("vt ")][3:] == [
        "vt 1 0", "vt 1 1", "vt 0.5 1"
    ]  # fmt: skip
    faces = lines[lines.index("usemtl material_0") :]
    assert faces == ["usemtl material_0", "f 1/1 2/2 3/3", "usemtl material_1", "f 2/4 4/5 3/6"]


def _read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    magic, version, length = struct.unpack("<4sII", data[:12])
    assert (magic, version, length) == (b"glTF", 2, len(data))
    json_len, json_type = struct.unpack("<I4s", data[12:20])
    assert json_type == b"JSON" and json_len % 4 == 0
    gltf = json.loads(data[20 : 20 + json_len])
    bin_len, bin_type = struct.unpack("<I4s", data[20 + json_len : 28 + json_len])
    assert bin_type == b"BIN\0" and bin_len % 4 == 0
    return gltf, data[28 + json_len : 28 + json_len + bin_len]


def _accessor(gltf: dict[str, Any], blob: bytes, index: int) -> list[Any]:
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[acc["type"]]
    code = {5126: "f", 5125: "I"}[acc["componentType"]]
    raw = blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
    values = struct.unpack(f"<{acc['count'] * width}{code}", raw)
    return [values[i : i + width] for i in range(0, len(values), width)]


def test_write_glb(tmp_path: Path) -> None:
    mesh = read_openmvs_ply(write_ply(tmp_path / "m.ply"))
    gltf, blob = _read_glb(write_glb(mesh, tmp_path / "skull.glb"))
    assert gltf["asset"]["version"] == "2.0"
    assert gltf["extensionsUsed"] == ["KHR_materials_unlit"]
    positions = _accessor(gltf, blob, 0)
    uvs = _accessor(gltf, blob, 1)
    # 6 corners; vertex 1 shares its UV across faces (merged), vertex 2 doesn't (split).
    assert len(positions) == 5
    assert gltf["accessors"][0]["min"] == [0, 0, 0] and gltf["accessors"][0]["max"] == [1, 1, 0]
    primitives = gltf["meshes"][0]["primitives"]
    assert [p["material"] for p in primitives] == [0, 1]
    first = [i for (i,) in _accessor(gltf, blob, primitives[0]["indices"])]
    second = [i for (i,) in _accessor(gltf, blob, primitives[1]["indices"])]
    assert first == [0, 1, 2]
    assert second[0] == 1  # vertex 1, same UV (1, 0): reused
    assert second[2] not in first  # vertex 2 with another UV: a new glTF vertex
    # V is flipped: glTF's texture origin is the top left.
    assert uvs[0] == (0.0, 1.0) and uvs[2] == (0.0, 0.0)
    assert [positions[i] for i in second] == [(1, 0, 0), (1, 1, 0), (0, 1, 0)]
    image = gltf["images"][1]
    view = gltf["bufferViews"][image["bufferView"]]
    assert image["mimeType"] == "image/png"
    assert blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]] == (
        b"\x89PNG scene_textured1.png"
    )
    assert all(v["byteOffset"] % 4 == 0 for v in gltf["bufferViews"])
