# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""glTF 2.0, as GLB (one file) or .gltf (JSON, a .bin and the images beside
it): read into a `meshfiles.Mesh`, and written from one.

Reading takes the default scene's triangles (strips and fans too) with
their nodes' transforms, the first set of texture coordinates, vertex
colours, and each material's base colour and base colour texture. Meshes
compressed with Draco or meshopt can't be read; other extensions are
ignored unless the file says they are required.

Writing splits a vertex where its faces use different texture coordinates
(glTF keeps them per vertex), makes one primitive per material, and marks
the materials unlit (KHR_materials_unlit) when the mesh has a texture or
vertex colours: a scan's colours already contain its lighting. Vertex
colours are stored as they are, like the project's own exports.
"""

from __future__ import annotations

import base64
import binascii
import json
import struct
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ez2digitize.core.meshfiles import (
    Image,
    Material,
    Mesh,
    mesh,
    web_image,
)
from ez2digitize.core.meshio import MeshFormatError

FLOAT, UINT32 = 5126, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
COMPONENTS = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
# Extensions a file may require that change nothing this reader needs.
UNDERSTOOD = frozenset({"KHR_materials_unlit", "KHR_texture_transform", "KHR_mesh_quantization",
                        "KHR_materials_emissive_strength", "KHR_texture_basisu"})  # fmt: skip
GLB_MAGIC = b"glTF"
JSON_CHUNK, BIN_CHUNK = 0x4E4F534A, 0x004E4942


# --- reading ------------------------------------------------------------------------


@dataclass
class _Document:
    path: Path
    gltf: dict[str, Any]
    buffers: list[bytes]

    def list(self, key: str) -> list[dict[str, Any]]:
        value = self.gltf.get(key, [])
        return value if isinstance(value, list) else []

    def accessor(self, index: int) -> NDArray[np.float64]:
        """An accessor's values as floats (n, width), normalised integers scaled to 0..1."""
        try:
            info = self.list("accessors")[index]
            dtype = np.dtype(COMPONENTS[info["componentType"]])
            width = WIDTHS[info["type"]]
            count = int(info["count"])
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise MeshFormatError(f"{self.path}: accessor {index} can't be read") from exc
        if "bufferView" in info:
            values = self._view_values(info, dtype, width, count)
        else:
            values = np.zeros((count, width))
        if info.get("normalized") and dtype.kind in "iu":
            values = values / float(np.iinfo(dtype).max)
            values = np.maximum(values, -1.0)
        sparse = info.get("sparse")
        if isinstance(sparse, dict):
            values = values.copy()
            n = int(sparse["count"])
            at = self._sparse(sparse["indices"], n, 1).astype(np.int64).reshape(-1)
            new = self._sparse(sparse["values"], n, width, dtype)
            if info.get("normalized") and dtype.kind in "iu":
                new = new / float(np.iinfo(dtype).max)
            values[at] = new
        return values

    def _view_values(
        self, info: dict[str, Any], dtype: np.dtype[Any], width: int, count: int
    ) -> NDArray[np.float64]:
        try:
            view = self.list("bufferViews")[info["bufferView"]]
            data = self.buffers[view["buffer"]]
            start = int(view.get("byteOffset", 0)) + int(info.get("byteOffset", 0))
            element = dtype.itemsize * width
            stride = int(view.get("byteStride", 0)) or element
            if count and start + stride * (count - 1) + element > len(data):
                raise MeshFormatError(f"{self.path}: an accessor runs past its buffer")
            if stride == element:
                flat = np.frombuffer(data, dtype, count * width, start)
                return flat.astype(np.float64).reshape(count, width)
            rows = np.frombuffer(
                data, np.uint8, stride * (count - 1) + element if count else 0, start
            )
            table = np.lib.stride_tricks.as_strided(
                rows, shape=(count, element), strides=(stride, 1)
            )
            return np.ascontiguousarray(table).view(dtype).astype(np.float64).reshape(count, width)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise MeshFormatError(f"{self.path}: a buffer view can't be read") from exc

    def _sparse(
        self, part: dict[str, Any], count: int, width: int, dtype: np.dtype[Any] | None = None
    ) -> NDArray[np.float64]:
        kind = np.dtype(COMPONENTS[part.get("componentType", 0)]) if dtype is None else dtype
        info = {"bufferView": part["bufferView"], "byteOffset": part.get("byteOffset", 0)}
        return self._view_values(info, kind, width, count)

    def image(self, index: int) -> Image | None:
        try:
            info = self.list("images")[index]
        except IndexError:
            return None
        mime = str(info.get("mimeType", ""))
        suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(mime, "")
        if "bufferView" in info:
            view = self.list("bufferViews")[info["bufferView"]]
            start = int(view.get("byteOffset", 0))
            data = self.buffers[view["buffer"]][start : start + int(view["byteLength"])]
            return Image(f"image{index}{suffix or '.png'}", data)
        uri = info.get("uri")
        if not isinstance(uri, str):
            return None
        try:
            data = _load_uri(uri, self.path.parent)
        except MeshFormatError:
            return None  # a missing texture: the mesh still shows
        name = Path(urllib.parse.unquote(uri)).name if not uri.startswith("data:") else ""
        return Image(name or f"image{index}{suffix or '.png'}", data)


def read_gltf(path: Path) -> Mesh:
    """The triangles of a .glb or .gltf file's default scene."""
    document = _open(path)
    gltf = document.gltf
    required = set(gltf.get("extensionsRequired", []) or [])
    if required - UNDERSTOOD:
        names = ", ".join(sorted(required - UNDERSTOOD))
        raise MeshFormatError(f"{path}: needs {names}, which can't be read (compressed mesh?)")
    scenes = document.list("scenes")
    scene_index = gltf.get("scene", 0)
    nodes = document.list("nodes")
    if scenes and isinstance(scene_index, int) and scene_index < len(scenes):
        roots = scenes[scene_index].get("nodes", [])
    else:  # no scene: every node nobody has as a child
        children = {c for node in nodes for c in node.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]

    parts: list[_Part] = []
    stack = [(int(r), np.eye(4), 0) for r in roots]
    while stack:
        index, parent, depth = stack.pop()
        if depth > 64 or index >= len(nodes):
            raise MeshFormatError(f"{path}: the node tree is broken")
        node = nodes[index]
        world = parent @ _node_matrix(node)
        if "mesh" in node:
            meshes = document.list("meshes")
            if node["mesh"] >= len(meshes):
                raise MeshFormatError(f"{path}: a node refers to a mesh that doesn't exist")
            for primitive in meshes[node["mesh"]].get("primitives", []):
                part = _primitive(document, primitive, world)
                if part is not None:
                    parts.append(part)
        stack += [(int(c), world, depth + 1) for c in node.get("children", [])]
    if not parts:
        raise MeshFormatError(f"{path}: no triangles")
    return _merge(document, parts).check(path)


@dataclass
class _Part:
    positions: NDArray[np.float64]
    faces: NDArray[np.int64]
    colors: NDArray[np.float64] | None
    uvs: NDArray[np.float64] | None  # per vertex, glTF's origin (top left)
    material: int | None


def _primitive(
    document: _Document, primitive: dict[str, Any], world: NDArray[np.float64]
) -> _Part | None:
    mode = primitive.get("mode", 4)
    attributes = primitive.get("attributes", {})
    if mode not in (4, 5, 6) or "POSITION" not in attributes:
        return None  # points and lines
    positions = document.accessor(attributes["POSITION"])[:, :3]
    positions = positions @ world[:3, :3].T + world[:3, 3]
    if "indices" in primitive:
        indices = document.accessor(primitive["indices"]).reshape(-1).astype(np.int64)
    else:
        indices = np.arange(len(positions), dtype=np.int64)
    if mode == 4:
        faces = indices[: len(indices) // 3 * 3].reshape(-1, 3)
    elif mode == 5:  # strip: every other triangle turned round
        k = np.arange(max(len(indices) - 2, 0))
        faces = np.stack([indices[k], indices[k + 1 + k % 2], indices[k + 2 - k % 2]], axis=1)
    else:  # fan
        k = np.arange(1, max(len(indices) - 1, 1))
        faces = np.stack([np.full(len(k), indices[0]), indices[k], indices[k + 1]], axis=1)
    if np.linalg.det(world[:3, :3]) < 0:
        faces = faces[:, ::-1]
    if len(faces) and (faces.max() >= len(positions) or faces.min() < 0):
        raise MeshFormatError(f"{document.path}: a face refers to a vertex that doesn't exist")
    colors = None
    if "COLOR_0" in attributes:
        values = document.accessor(attributes["COLOR_0"])
        colors = values[:, :3] if values.shape[1] >= 3 else None
    uvs = document.accessor(attributes["TEXCOORD_0"])[:, :2] if "TEXCOORD_0" in attributes else None
    material = primitive.get("material")
    return _Part(positions, faces, colors, uvs, material if isinstance(material, int) else None)


def _merge(document: _Document, parts: list[_Part]) -> Mesh:
    gltf_materials = document.list("materials")
    used = sorted({p.material for p in parts if p.material is not None})
    index_of: dict[int | None, int] = {m: i for i, m in enumerate(used)}
    materials = [_material(document, gltf_materials, m) for m in used]
    if any(p.material is None for p in parts):
        index_of[None] = len(materials)
        materials.append(Material("default"))
    textured = any(p.uvs is not None for p in parts)
    colored = any(p.colors is not None for p in parts)
    positions, faces, colors, uvs, face_materials = [], [], [], [], []
    offset = 0
    for part in parts:
        positions.append(part.positions)
        faces.append(part.faces + offset)
        offset += len(part.positions)
        if colored:
            if part.colors is not None:
                colors.append(part.colors)
            else:
                tint = np.array(materials[index_of[part.material]].color[:3])
                colors.append(np.tile(tint, (len(part.positions), 1)))
        if textured:
            corner = (
                part.uvs[part.faces] if part.uvs is not None else np.zeros((len(part.faces), 3, 2))
            )
            uvs.append(np.stack([corner[..., 0], 1 - corner[..., 1]], axis=-1))
        face_materials.append(np.full(len(part.faces), index_of[part.material]))
    return mesh(
        np.vstack(positions),
        np.vstack(faces),
        colors=np.clip(np.round(np.vstack(colors) * 255), 0, 255) if colored else None,
        uvs=np.vstack(uvs) if textured else None,
        materials=materials,
        face_materials=np.concatenate(face_materials),
    )


def _material(document: _Document, materials: list[dict[str, Any]], index: int) -> Material:
    info = materials[index] if index < len(materials) else {}
    pbr = info.get("pbrMetallicRoughness", {}) or {}
    factor = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
    color = tuple(float(v) for v in (list(factor) + [1.0] * 4)[:4])
    image = None
    texture_info = pbr.get("baseColorTexture")
    if isinstance(texture_info, dict):
        textures = document.list("textures")
        texture_index = texture_info.get("index", -1)
        if 0 <= texture_index < len(textures):
            texture = textures[texture_index]
            source = texture.get("source")
            if source is None:  # an image only an extension names (WebP, KTX2)
                for extension in (texture.get("extensions") or {}).values():
                    if isinstance(extension, dict) and "source" in extension:
                        source = extension["source"]
            if isinstance(source, int):
                image = document.image(source)
    name = str(info.get("name") or f"material_{index}")
    return Material(name, (color[0], color[1], color[2], color[3]), image)


def _node_matrix(node: dict[str, Any]) -> NDArray[np.float64]:
    if "matrix" in node:
        return np.array(node["matrix"], np.float64).reshape(4, 4).T  # column-major
    matrix = np.eye(4)
    x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, :3] *= np.array(node.get("scale", [1.0, 1.0, 1.0]), np.float64)
    matrix[:3, 3] = node.get("translation", [0.0, 0.0, 0.0])
    return matrix


def _open(path: Path) -> _Document:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise MeshFormatError(f"{path}: {exc}") from exc
    try:
        if data[:4] == GLB_MAGIC:
            _magic, version, _length = struct.unpack_from("<4sII", data)
            if version != 2:
                raise MeshFormatError(f"{path}: glTF version {version}, only 2 is read")
            offset = 12
            chunks: dict[int, bytes] = {}
            while offset + 8 <= len(data):
                size, kind = struct.unpack_from("<II", data, offset)
                chunks.setdefault(kind, data[offset + 8 : offset + 8 + size])
                offset += 8 + size
            gltf = json.loads(chunks[JSON_CHUNK])
            binary = chunks.get(BIN_CHUNK)
        else:
            gltf = json.loads(data)
            binary = None
    except (KeyError, struct.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeshFormatError(f"{path}: not a glTF file") from exc
    if not isinstance(gltf, dict) or not str(gltf.get("asset", {}).get("version", "")).startswith(
        "2"
    ):
        raise MeshFormatError(f"{path}: not a glTF 2.0 file")
    buffers = []
    for i, info in enumerate(gltf.get("buffers", [])):
        uri = info.get("uri")
        if uri is None:
            if binary is None or i != 0:
                raise MeshFormatError(f"{path}: buffer {i} has no data")
            buffers.append(binary)
        else:
            buffers.append(_load_uri(str(uri), path.parent))
    return _Document(path, gltf, buffers)


def _load_uri(uri: str, folder: Path) -> bytes:
    if uri.startswith("data:"):
        _header, _, payload = uri.partition(",")
        try:
            return base64.b64decode(payload) if ";base64" in _header else payload.encode()
        except binascii.Error as exc:
            raise MeshFormatError(f"a data URI can't be decoded: {exc}") from exc
    relative = Path(urllib.parse.unquote(uri))
    if relative.is_absolute() or ".." in relative.parts or "://" in uri:
        raise MeshFormatError(f"{uri}: only files beside the glTF are read")
    try:
        return (folder / relative).read_bytes()
    except OSError as exc:
        raise MeshFormatError(f"{folder / relative}: {exc}") from exc


# --- writing ------------------------------------------------------------------------


def write_glb(source: Mesh, path: Path) -> list[Path]:
    """One binary glTF file, the texture images inside."""
    gltf, blob, _images = _build(source, path.stem, embed=True)
    header = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 4)
    blob += b"\0" * (-len(blob) % 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(struct.pack("<4sII", GLB_MAGIC, 2, 12 + 8 + len(header) + 8 + len(blob)))
        out.write(struct.pack("<I4s", len(header), b"JSON") + header)
        out.write(struct.pack("<I4s", len(blob), b"BIN\0") + bytes(blob))
    return [path]


def write_gltf(source: Mesh, path: Path) -> list[Path]:
    """`<name>.gltf`, its buffer `<name>.bin` and the texture images beside it."""
    gltf, blob, images = _build(source, path.stem, embed=False)
    binary = path.with_suffix(".bin")
    gltf["buffers"][0]["uri"] = urllib.parse.quote(binary.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = [path, binary]
    for name, image in images:
        (path.parent / name).write_bytes(image.data)
        written.append(path.parent / name)
    path.write_text(json.dumps(gltf, indent=1), encoding="utf-8")
    binary.write_bytes(bytes(blob))
    return written


def _build(
    source: Mesh, name: str, *, embed: bool
) -> tuple[dict[str, Any], bytearray, list[tuple[str, Image]]]:
    """The glTF JSON, its binary buffer, and (not embedded) the images to write beside it."""
    face_material = (
        source.face_materials.astype(np.int64)
        if source.face_materials is not None and source.materials
        else np.zeros(source.face_count, np.int64)
    )
    materials = source.materials or [Material("default")]
    corners = source.faces.reshape(-1).astype(np.int64)
    if source.uvs is not None:
        # One glTF vertex per distinct (vertex, texture coordinates) pair.
        quantised = np.round(source.uvs.reshape(-1, 2).astype(np.float64) * (1 << 20))
        keys = np.column_stack([corners, quantised.astype(np.int64)])
        unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        vertex = unique[:, 0]
        uv = source.uvs.reshape(-1, 2)[first].astype(np.float32)
        uv[:, 1] = 1 - uv[:, 1]  # glTF puts the texture origin at the top left
        indices = inverse.reshape(-1, 3).astype(np.uint32)
    else:
        vertex = np.arange(source.vertex_count)
        uv = None
        indices = source.faces.astype(np.uint32)
    positions = np.ascontiguousarray(source.positions[vertex], "<f4")

    blob = bytearray()
    views: list[dict[str, int]] = []

    def add_view(data: bytes, target: int | None = None) -> int:
        blob.extend(b"\0" * (-len(blob) % 4))
        view = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        blob.extend(data)
        views.append(view)
        return len(views) - 1

    accessors: list[dict[str, Any]] = [
        {
            "bufferView": add_view(positions.tobytes(), ARRAY_BUFFER),
            "componentType": FLOAT,
            "count": len(positions),
            "type": "VEC3",
            "min": positions.min(axis=0).tolist(),
            "max": positions.max(axis=0).tolist(),
        }
    ]
    attributes = {"POSITION": 0}
    if uv is not None:
        attributes["TEXCOORD_0"] = len(accessors)
        accessors.append(
            {
                "bufferView": add_view(np.ascontiguousarray(uv, "<f4").tobytes(), ARRAY_BUFFER),
                "componentType": FLOAT,
                "count": len(uv),
                "type": "VEC2",
            }
        )
    if source.colors is not None:
        rgba = np.column_stack([source.colors[vertex], np.full(len(vertex), 255, np.uint8)]).astype(
            np.uint8
        )
        attributes["COLOR_0"] = len(accessors)
        accessors.append(
            {
                "bufferView": add_view(rgba.tobytes(), ARRAY_BUFFER),
                "componentType": 5121,
                "normalized": True,
                "count": len(rgba),
                "type": "VEC4",
            }
        )

    unlit = source.colors is not None or any(m.image is not None for m in materials)
    images: list[dict[str, Any]] = []
    textures: list[dict[str, int]] = []
    beside: list[tuple[str, Image]] = []
    gltf_materials: list[dict[str, Any]] = []
    for material in materials:
        pbr: dict[str, Any] = {"metallicFactor": 0.0, "roughnessFactor": 1.0}
        if material.color != (1.0, 1.0, 1.0, 1.0):
            pbr["baseColorFactor"] = list(material.color)
        if material.image is not None and uv is not None:
            image = web_image(material.image)
            if embed:
                images.append({"bufferView": add_view(image.data), "mimeType": image.mime})
            else:
                file_name = f"{name}_texture{len(beside)}{image.suffix}"
                beside.append((file_name, image))
                images.append({"uri": urllib.parse.quote(file_name)})
            textures.append({"source": len(images) - 1})
            pbr["baseColorTexture"] = {"index": len(textures) - 1}
        entry: dict[str, Any] = {"name": material.name, "pbrMetallicRoughness": pbr}
        if material.color[3] < 1.0:
            entry["alphaMode"] = "BLEND"
        if unlit:
            entry["extensions"] = {"KHR_materials_unlit": {}}
        gltf_materials.append(entry)

    primitives = []
    for index in range(len(materials)):
        chosen = indices[face_material == index]
        if not len(chosen):
            continue
        accessors.append(
            {
                "bufferView": add_view(
                    np.ascontiguousarray(chosen, "<u4").tobytes(), ELEMENT_ARRAY_BUFFER
                ),
                "componentType": UINT32,
                "count": int(chosen.size),
                "type": "SCALAR",
            }
        )
        primitives.append(
            {"attributes": dict(attributes), "indices": len(accessors) - 1, "material": index}
        )
    blob.extend(b"\0" * (-len(blob) % 4))
    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "EZ2DIGITIZE"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": primitives}],
        "materials": gltf_materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    if unlit:
        gltf["extensionsUsed"] = ["KHR_materials_unlit"]
    if images:
        gltf["images"] = images
        gltf["textures"] = textures
    return gltf, blob, beside
