# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import base64
import io
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from model_files import SQUARE, png, textured_square
from PIL import Image as PILImage

from ez2digitize.core import gltf
from ez2digitize.core import meshfiles as mf
from ez2digitize.core.meshio import MeshFormatError


def read_json(glb: Path) -> dict[str, Any]:
    data = glb.read_bytes()
    size = struct.unpack_from("<I", data, 12)[0]
    document: dict[str, Any] = json.loads(data[20 : 20 + size])
    return document


def write_gltf_json(path: Path, document: dict[str, Any], buffer: bytes) -> Path:
    uri = "data:application/octet-stream;base64," + base64.b64encode(buffer).decode()
    document = {"asset": {"version": "2.0"}, "buffers": [{"uri": uri, "byteLength": len(buffer)}],
                **document}  # fmt: skip
    path.write_text(json.dumps(document))
    return path


def test_glb_round_trip_splits_vertices_on_texture_seams(tmp_path: Path) -> None:
    source = textured_square(((200, 10, 10), (10, 200, 10)))
    path = gltf.write_glb(source, tmp_path / "s.glb")[0]
    document = read_json(path)
    assert [len(m["primitives"]) for m in document["meshes"]] == [2]
    assert document["extensionsUsed"] == ["KHR_materials_unlit"]
    # Vertex 0 and 2 have the same coordinates in both faces: 4 glTF vertices.
    assert document["accessors"][0]["count"] == 4
    back = gltf.read_gltf(path)
    assert back.face_count == 2 and back.textured
    assert back.face_materials is not None and back.face_materials.tolist() == [0, 1]
    assert back.materials[1].image is not None and back.materials[1].image.mime == "image/png"
    corners = back.positions[back.faces].tolist()
    assert corners == source.positions[source.faces].tolist()
    assert np.allclose(back.uvs, source.uvs)  # type: ignore[arg-type]


def test_gltf_with_files_beside_it(tmp_path: Path) -> None:
    files = gltf.write_gltf(textured_square(), tmp_path / "out" / "scan.gltf")
    assert [f.name for f in files] == ["scan.gltf", "scan.bin", "scan_texture0.png"]
    document = json.loads(files[0].read_text())
    assert document["buffers"][0]["uri"] == "scan.bin"
    assert document["images"] == [{"uri": "scan_texture0.png"}]
    back = gltf.read_gltf(files[0])
    assert back.textured and back.face_count == 2


def test_vertex_colours_and_plain_meshes(tmp_path: Path) -> None:
    colored = mf.mesh(SQUARE, [(0, 1, 2), (0, 2, 3)], colors=[(10, 20, 30)] * 4)
    document = read_json(gltf.write_glb(colored, tmp_path / "c.glb")[0])
    assert "COLOR_0" in document["meshes"][0]["primitives"][0]["attributes"]
    back = gltf.read_gltf(tmp_path / "c.glb")
    assert back.colors is not None and back.colors[0].tolist() == [10, 20, 30]

    plain = mf.mesh(SQUARE, [(0, 1, 2)])
    document = read_json(gltf.write_glb(plain, tmp_path / "p.glb")[0])
    assert "extensionsUsed" not in document  # lit: no colours of its own


def test_nodes_strips_strides_and_sparse_accessors(tmp_path: Path) -> None:
    # Interleaved positions (stride 16), a strip of 4 indices, a sparse change,
    # and a node that moves and scales the mesh.
    vertices = b"".join(struct.pack("<3f4x", *p) for p in SQUARE)
    indices = struct.pack("<4H", 0, 1, 3, 2)
    sparse = struct.pack("<B3x", 3) + struct.pack("<3f", 0.0, 2.0, 0.0)
    buffer = vertices + indices + sparse
    document = {
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"children": [1], "translation": [10, 0, 0]}, {"mesh": 0, "scale": [2, 2, 2]}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 5}]}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 64, "byteStride": 16},
            {"buffer": 0, "byteOffset": 64, "byteLength": 8},
            {"buffer": 0, "byteOffset": 72, "byteLength": 4},
            {"buffer": 0, "byteOffset": 76, "byteLength": 12},
        ],
        "accessors": [
            {
                "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
                "sparse": {"count": 1, "indices": {"bufferView": 2, "componentType": 5121},
                           "values": {"bufferView": 3}},
            },
            {"bufferView": 1, "componentType": 5123, "count": 4, "type": "SCALAR"},
        ],
    }  # fmt: skip
    mesh = gltf.read_gltf(write_gltf_json(tmp_path / "strip.gltf", document, buffer))
    assert mesh.faces.tolist() == [[0, 1, 3], [1, 2, 3]]  # both counter-clockwise
    assert mesh.positions[3].tolist() == [10.0, 4.0, 0.0]  # sparse (0, 2, 0), scaled, moved
    assert mesh.positions[1].tolist() == [12.0, 0.0, 0.0]


def test_embedded_texture_from_a_data_uri(tmp_path: Path) -> None:
    source = textured_square()
    path = gltf.write_gltf(source, tmp_path / "t.gltf")[0]
    document = json.loads(path.read_text())
    image = (tmp_path / "t_texture0.png").read_bytes()
    document["images"] = [{"uri": "data:image/png;base64," + base64.b64encode(image).decode()}]
    path.write_text(json.dumps(document))
    (tmp_path / "t_texture0.png").unlink()
    back = gltf.read_gltf(path)
    assert back.materials[0].image is not None and back.materials[0].image.data == image


def test_missing_texture_still_reads(tmp_path: Path) -> None:
    path = gltf.write_gltf(textured_square(), tmp_path / "t.gltf")[0]
    (tmp_path / "t_texture0.png").unlink()
    back = gltf.read_gltf(path)
    assert back.face_count == 2 and not back.textured


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"extensionsRequired": ["KHR_draco_mesh_compression"]}, "KHR_draco_mesh_compression"),
        ({"asset": {"version": "1.0"}}, "not a glTF 2.0 file"),
        ({"meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 1}]}]},
         "no triangles"),
    ],
)  # fmt: skip
def test_errors(tmp_path: Path, change: dict[str, Any], message: str) -> None:
    path = gltf.write_gltf(mf.mesh(SQUARE, [(0, 1, 2)]), tmp_path / "e.gltf")[0]
    document = json.loads(path.read_text())
    document.update(change)
    path.write_text(json.dumps(document))
    with pytest.raises(MeshFormatError, match=message):
        gltf.read_gltf(path)


def test_buffers_outside_the_folder_are_refused(tmp_path: Path) -> None:
    path = gltf.write_gltf(mf.mesh(SQUARE, [(0, 1, 2)]), tmp_path / "e.gltf")[0]
    document = json.loads(path.read_text())
    document["buffers"][0]["uri"] = "../../etc/passwd"
    path.write_text(json.dumps(document))
    with pytest.raises(MeshFormatError, match="only files beside"):
        gltf.read_gltf(path)


def test_webp_textures_become_png(tmp_path: Path) -> None:
    out = io.BytesIO()
    PILImage.new("RGB", (2, 2), (5, 6, 7)).save(out, "WEBP")
    source = textured_square()
    source.materials[0].image = mf.Image("skin.webp", out.getvalue())
    document = read_json(gltf.write_glb(source, tmp_path / "w.glb")[0])
    assert document["images"][0]["mimeType"] == "image/png"
    assert png((1, 1, 1))[:4] == b"\x89PNG"
