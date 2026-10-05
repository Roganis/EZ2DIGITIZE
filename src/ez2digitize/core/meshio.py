# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Textured triangle meshes: read OpenMVS's PLY, write OBJ, GLB, STL and 3MF.

Standard library only. The heavy loops run in `struct` and `array`, so a
mesh of a few million faces converts in seconds.

OpenMVS writes a binary little-endian PLY: vertices with float x, y, z (and
normals if it has them), faces with `vertex_indices` (3), `texcoord` (6
floats: u, v per corner, origin at the bottom left) and, when the texture
was split over several images, `texnumber`. The texture images are named in
`comment TextureFile <name>` lines, in texture-number order.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import zipfile
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

PLY_TYPES = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}  # fmt: skip


class MeshFormatError(Exception):
    """A mesh file can't be read or doesn't have what an export needs."""


@dataclass
class TexturedMesh:
    """Triangles with per-corner texture coordinates.

    `positions`: x, y, z per vertex. `faces`: three vertex indices per face.
    `uvs`: u, v per face corner (six per face), origin at the bottom left as
    in OBJ. `texture_ids`: the texture image of each face. `textures`: image
    files, indexed by texture id.
    """

    positions: array[float]
    faces: array[int]
    uvs: array[float]
    texture_ids: array[int]
    textures: list[Path]

    @property
    def vertex_count(self) -> int:
        return len(self.positions) // 3

    @property
    def face_count(self) -> int:
        return len(self.faces) // 3


# --- reading -------------------------------------------------------------------


@dataclass
class _Property:
    name: str
    type: str
    count_type: str | None = None  # set for list properties


@dataclass
class _Element:
    name: str
    count: int
    properties: list[_Property]


def read_openmvs_ply(path: Path) -> TexturedMesh:
    """Read a textured mesh written by OpenMVS TextureMesh (PLY export)."""
    with path.open("rb") as fh:
        elements, comments = _read_header(fh, path)
        data = fh.read()
    textures = [path.parent / c.split(" ", 1)[1] for c in comments if c.startswith("TextureFile ")]
    if not textures:
        raise MeshFormatError(f"{path}: no texture (no 'comment TextureFile' line)")
    offset = 0
    positions = array("f")
    faces, uvs, texture_ids = array("I"), array("f"), array("I")
    for element in elements:
        if element.name == "vertex":
            offset = _read_vertices(data, offset, element, positions, path)
        elif element.name == "face":
            offset = _read_faces(data, offset, element, faces, uvs, texture_ids, path)
        else:
            raise MeshFormatError(f"{path}: unexpected element {element.name!r}")
    if not faces:
        raise MeshFormatError(f"{path}: no faces")
    if max(texture_ids) >= len(textures):
        raise MeshFormatError(f"{path}: a face uses texture {max(texture_ids)}, not listed")
    if max(faces) >= len(positions) // 3:
        raise MeshFormatError(f"{path}: a face refers to a vertex that doesn't exist")
    return TexturedMesh(positions, faces, uvs, texture_ids, textures)


def _read_header(fh: BinaryIO, path: Path) -> tuple[list[_Element], list[str]]:
    if fh.readline().strip() != b"ply":
        raise MeshFormatError(f"{path}: not a PLY file")
    elements: list[_Element] = []
    comments: list[str] = []
    while True:
        raw = fh.readline()
        if not raw:
            raise MeshFormatError(f"{path}: header has no end_header")
        words = raw.decode("ascii", errors="replace").split()
        if not words:
            continue
        keyword = words[0]
        if keyword == "end_header":
            return elements, comments
        if keyword == "format":
            if words[1:2] != ["binary_little_endian"]:
                raise MeshFormatError(f"{path}: only binary little-endian PLY is supported")
        elif keyword == "comment":
            comments.append(raw.decode("utf-8", errors="replace").strip()[len("comment ") :])
        elif keyword == "element":
            elements.append(_Element(words[1], int(words[2]), []))
        elif keyword == "property":
            if not elements:
                raise MeshFormatError(f"{path}: property before any element")
            if words[1] == "list":
                prop = _Property(words[4], _ply_type(words[3], path), _ply_type(words[2], path))
            else:
                prop = _Property(words[2], _ply_type(words[1], path))
            elements[-1].properties.append(prop)


def ply_element_counts(path: Path) -> dict[str, int]:
    """Number of each element (vertex, face) from a binary PLY's header alone."""
    with path.open("rb") as fh:
        elements, _ = _read_header(fh, path)
    return {element.name: element.count for element in elements}


def _ply_type(name: str, path: Path) -> str:
    try:
        return PLY_TYPES[name]
    except KeyError:
        raise MeshFormatError(f"{path}: unknown PLY type {name!r}") from None


def _read_vertices(
    data: bytes, offset: int, element: _Element, positions: array[float], path: Path
) -> int:
    if any(p.count_type for p in element.properties):
        raise MeshFormatError(f"{path}: list properties in vertices are not supported")
    names = [p.name for p in element.properties]
    if names[:3] != ["x", "y", "z"]:
        raise MeshFormatError(f"{path}: vertices must start with x, y, z")
    record = struct.Struct("<" + "".join(p.type for p in element.properties))
    end = offset + record.size * element.count
    if end > len(data):
        raise MeshFormatError(f"{path}: file ends inside the vertex list")
    for values in record.iter_unpack(data[offset:end]):
        positions.extend(values[:3])
    return end


def _read_faces(
    data: bytes,
    offset: int,
    element: _Element,
    faces: array[int],
    uvs: array[float],
    texture_ids: array[int],
    path: Path,
) -> int:
    # OpenMVS writes every face with 3 indices and 6 texture coordinates, so
    # one fixed-size record fits all; the counts are checked as they are read.
    layout = []
    columns: dict[str, slice] = {}
    position = 0
    for prop in element.properties:
        if prop.count_type is not None:
            expected = {"vertex_indices": 3, "vertex_index": 3, "texcoord": 6}.get(prop.name)
            if expected is None:
                raise MeshFormatError(f"{path}: unexpected face list {prop.name!r}")
            layout.append(prop.count_type + prop.type * expected)
            columns[prop.name + "#count"] = slice(position, position + 1)
            columns[prop.name] = slice(position + 1, position + 1 + expected)
            position += 1 + expected
        else:
            layout.append(prop.type)
            columns[prop.name] = slice(position, position + 1)
            position += 1
    index_name = "vertex_indices" if "vertex_indices" in columns else "vertex_index"
    if index_name not in columns or "texcoord" not in columns:
        raise MeshFormatError(f"{path}: faces need vertex_indices and texcoord")
    record = struct.Struct("<" + "".join(layout))
    end = offset + record.size * element.count
    if end > len(data):
        raise MeshFormatError(f"{path}: file ends inside the face list")
    idx, tex = columns[index_name], columns["texcoord"]
    idx_count, tex_count = columns[index_name + "#count"], columns["texcoord#count"]
    number = columns.get("texnumber")
    for values in record.iter_unpack(data[offset:end]):
        if values[idx_count] != (3,) or values[tex_count] != (6,):
            raise MeshFormatError(f"{path}: only triangles with 6 texture coordinates")
        faces.extend(values[idx])
        uvs.extend(values[tex])
        texture_ids.append(values[number][0] if number is not None else 0)
    return end


# --- OBJ -----------------------------------------------------------------------


def write_obj(mesh: TexturedMesh, folder: Path, name: str) -> list[Path]:
    """`<name>.obj` + `<name>.mtl` + the texture images, all in `folder`.

    Returns the files written. The texture coordinates are written per face
    corner, as in the source, so no vertex is split.
    """
    folder.mkdir(parents=True, exist_ok=True)
    textures = _copy_textures(mesh, folder, name)
    mtl = folder / f"{name}.mtl"
    mtl.write_text(
        "".join(
            f"newmtl material_{i}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd {tex.name}\n\n"
            for i, tex in enumerate(textures)
        ),
        encoding="utf-8",
    )
    obj = folder / f"{name}.obj"
    p, uv = mesh.positions, mesh.uvs
    with obj.open("w", encoding="utf-8", newline="\n") as out:
        out.write(f"# EZ2DIGITIZE\nmtllib {mtl.name}\n")
        out.writelines(f"v {p[i]:.6g} {p[i + 1]:.6g} {p[i + 2]:.6g}\n" for i in range(0, len(p), 3))
        out.writelines(f"vt {uv[i]:.6g} {uv[i + 1]:.6g}\n" for i in range(0, len(uv), 2))
        f = mesh.faces
        for texture_id in range(len(textures)):
            out.write(f"usemtl material_{texture_id}\n")
            out.writelines(
                # OBJ indices are 1-based; texture coordinate j belongs to corner j.
                f"f {f[3 * k] + 1}/{3 * k + 1} {f[3 * k + 1] + 1}/{3 * k + 2} "
                f"{f[3 * k + 2] + 1}/{3 * k + 3}\n"
                for k in range(mesh.face_count)
                if mesh.texture_ids[k] == texture_id
            )
    return [obj, mtl, *textures]


# --- GLB -----------------------------------------------------------------------

_FLOAT, _UINT32 = 5126, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963


def write_glb(mesh: TexturedMesh, path: Path) -> Path:
    """A binary glTF 2.0 file with the textures embedded.

    glTF stores texture coordinates per vertex, so a vertex is split where
    neighbouring faces use different coordinates for it. One primitive per
    texture image; the material is unlit (KHR_materials_unlit) because the
    photos already contain the lighting.
    """
    positions, uvs = array("f"), array("f")
    # Corners with the same vertex and texture coordinates become one glTF
    # vertex. Inside a texture patch, a vertex has the same coordinates in all
    # its faces, so each vertex remembers its first coordinates in two flat
    # arrays; only corners on patch seams go to a dict. Coordinates are
    # compared quantised to 2^-20, far finer than a texel.
    first_uv = array("q", [-1]) * mesh.vertex_count
    first_index = array("I", [0]) * mesh.vertex_count
    seams: dict[int, int] = {}
    indices_by_texture: list[array[int]] = [array("I") for _ in mesh.textures]
    src_p, src_uv, faces = mesh.positions, mesh.uvs, mesh.faces

    def new_vertex(v: int, u: float, w: float) -> int:
        index = len(positions) // 3
        positions.extend(src_p[3 * v : 3 * v + 3])
        uvs.append(u)
        uvs.append(1.0 - w)  # glTF puts the texture origin at the top left
        return index

    for k in range(mesh.face_count):
        out = indices_by_texture[mesh.texture_ids[k]]
        for c in range(3):
            v = faces[3 * k + c]
            u, w = src_uv[6 * k + 2 * c], src_uv[6 * k + 2 * c + 1]
            uv_key = (_quantise(u) << 21) | _quantise(w)
            if first_uv[v] == uv_key:
                index = first_index[v]
            elif first_uv[v] < 0:
                index = first_index[v] = new_vertex(v, u, w)
                first_uv[v] = uv_key
            else:
                key = (v << 42) | uv_key
                found = seams.get(key)
                index = seams[key] = new_vertex(v, u, w) if found is None else found
            out.append(index)

    blob = bytearray()
    buffer_views: list[dict[str, int]] = []

    def add_view(data: bytes, target: int | None = None) -> int:
        while len(blob) % 4:
            blob.append(0)
        view = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        blob.extend(data)
        buffer_views.append(view)
        return len(buffer_views) - 1

    count = len(positions) // 3
    xs, ys, zs = positions[0::3], positions[1::3], positions[2::3]
    accessors: list[dict[str, object]] = [
        {
            "bufferView": add_view(_le_bytes(positions), _ARRAY_BUFFER),
            "componentType": _FLOAT,
            "count": count,
            "type": "VEC3",
            "min": [min(xs), min(ys), min(zs)],
            "max": [max(xs), max(ys), max(zs)],
        },
        {
            "bufferView": add_view(_le_bytes(uvs), _ARRAY_BUFFER),
            "componentType": _FLOAT,
            "count": count,
            "type": "VEC2",
        },
    ]
    images, textures, materials, primitives = [], [], [], []
    for texture_id, (indices, image) in enumerate(
        zip(indices_by_texture, mesh.textures, strict=True)
    ):
        mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
        images.append({"bufferView": add_view(image.read_bytes()), "mimeType": mime})
        textures.append({"source": texture_id})
        materials.append(
            {
                "name": f"material_{texture_id}",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": texture_id},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "extensions": {"KHR_materials_unlit": {}},
            }
        )
        if not indices:
            continue
        accessors.append(
            {
                "bufferView": add_view(_le_bytes(indices), _ELEMENT_ARRAY_BUFFER),
                "componentType": _UINT32,
                "count": len(indices),
                "type": "SCALAR",
            }
        )
        primitives.append(
            {
                "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                "indices": len(accessors) - 1,
                "material": texture_id,
            }
        )
    while len(blob) % 4:
        blob.append(0)
    gltf = {
        "asset": {"version": "2.0", "generator": "EZ2DIGITIZE"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": path.stem}],
        "meshes": [{"primitives": primitives}],
        "materials": materials,
        "textures": textures,
        "images": images,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob)}],
    }
    header_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    header_json += b" " * (-len(header_json) % 4)
    total = 12 + 8 + len(header_json) + 8 + len(blob)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as glb:
        glb.write(struct.pack("<4sII", b"glTF", 2, total))
        glb.write(struct.pack("<I4s", len(header_json), b"JSON") + header_json)
        glb.write(struct.pack("<I4s", len(blob), b"BIN\0") + bytes(blob))
    return path


def _quantise(value: float) -> int:
    """A texture coordinate in [0, 1] as a 21-bit integer (clamped)."""
    return min(max(int(value * 1048576.0 + 0.5), 0), 2_097_151)


def _le_bytes(values: array[float] | array[int]) -> bytes:
    if struct.pack("=H", 1) != struct.pack("<H", 1):  # big-endian host
        values = array(values.typecode, values)
        values.byteswap()
    return values.tobytes()


def _copy_textures(mesh: TexturedMesh, folder: Path, name: str) -> list[Path]:
    copies = []
    for i, texture in enumerate(mesh.textures):
        target = folder / f"{name}_texture{i}{texture.suffix.lower()}"
        shutil.copyfile(texture, target)
        copies.append(target)
    return copies


# --- printing --------------------------------------------------------------------


@dataclass(frozen=True)
class Watertightness:
    """Edges not shared by exactly two faces. Zero of both: closed and manifold."""

    open_edges: int  # used by one face: a hole or the mesh's border
    non_manifold_edges: int  # used by three or more faces

    @property
    def watertight(self) -> bool:
        return self.open_edges == 0 and self.non_manifold_edges == 0


def check_watertight(mesh: TexturedMesh) -> Watertightness:
    """Count edges by how many faces use them (a few seconds per million faces)."""
    n = mesh.vertex_count
    faces = mesh.faces
    keys = array("q")
    for corner in range(3):
        a = faces[corner::3]
        b = faces[(corner + 1) % 3 :: 3] if corner < 2 else faces[0::3]
        keys.extend(min(x, y) * n + max(x, y) for x, y in zip(a, b, strict=True))
    ordered = sorted(keys)
    open_edges = non_manifold = 0
    run = 1
    for i in range(1, len(ordered) + 1):
        if i < len(ordered) and ordered[i] == ordered[i - 1]:
            run += 1
            continue
        if run == 1:
            open_edges += 1
        elif run > 2:
            non_manifold += 1
        run = 1
    return Watertightness(open_edges, non_manifold)


def write_stl(mesh: TexturedMesh, path: Path) -> Path:
    """Binary STL of the geometry (no texture)."""
    pos = mesh.positions
    faces = mesh.faces
    record = struct.Struct("<12fH")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stl:
        stl.write(b"EZ2DIGITIZE binary STL".ljust(80, b" "))
        stl.write(struct.pack("<I", mesh.face_count))
        chunk = bytearray()
        for f in range(0, len(faces), 3):
            a, b, c = faces[f] * 3, faces[f + 1] * 3, faces[f + 2] * 3
            ax, ay, az = pos[a], pos[a + 1], pos[a + 2]
            bx, by, bz = pos[b], pos[b + 1], pos[b + 2]
            cx, cy, cz = pos[c], pos[c + 1], pos[c + 2]
            ux, uy, uz = bx - ax, by - ay, bz - az
            vx, vy, vz = cx - ax, cy - ay, cz - az
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            chunk += record.pack(
                nx / length, ny / length, nz / length, ax, ay, az, bx, by, bz, cx, cy, cz, 0
            )
            if len(chunk) >= 1 << 22:
                stl.write(chunk)
                chunk.clear()
        stl.write(chunk)
    return path


_3MF_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"""
_3MF_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Target="/3D/3dmodel.model" Id="rel0" \
Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""


def write_3mf(
    mesh: TexturedMesh, path: Path, *, name: str = "scan", unit: str = "millimeter"
) -> Path:
    """3MF (core spec) of the geometry: one object, no texture.

    `unit` is what one model unit means; until the scale is set from a
    known distance the reconstruction's units are arbitrary.
    """
    pos = mesh.positions
    faces = mesh.faces
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<model unit="{unit}" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
        f'<metadata name="Title">{_xml_text(name)}</metadata>\n'
        '<metadata name="Application">EZ2DIGITIZE</metadata>\n'
        '<resources><object id="1" type="model"><mesh><vertices>\n'
    ]
    parts += [
        f'<vertex x="{pos[i]:.6g}" y="{pos[i + 1]:.6g}" z="{pos[i + 2]:.6g}"/>\n'
        for i in range(0, len(pos), 3)
    ]
    parts.append("</vertices><triangles>\n")
    parts += [
        f'<triangle v1="{faces[i]}" v2="{faces[i + 1]}" v3="{faces[i + 2]}"/>\n'
        for i in range(0, len(faces), 3)
    ]
    parts.append(
        '</triangles></mesh></object></resources>\n<build><item objectid="1"/></build>\n</model>\n'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", _3MF_CONTENT_TYPES)
        package.writestr("_rels/.rels", _3MF_RELS)
        package.writestr("3D/3dmodel.model", "".join(parts))
    return path


def _xml_text(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --- point clouds ------------------------------------------------------------------


@dataclass
class PointCloud:
    """Points with optional colours (r, g, b bytes) and normals."""

    positions: array[float]
    colors: bytes | None = None
    normals: array[float] | None = None

    @property
    def count(self) -> int:
        return len(self.positions) // 3


def read_point_cloud(path: Path) -> PointCloud:
    """The vertices of a binary little-endian PLY (e.g. OpenMVS's dense cloud).

    Reads x, y, z, red, green, blue and nx, ny, nz when present; other
    properties, including lists like OpenMVS's view_indices, are skipped.
    """
    with path.open("rb") as fh:
        elements, _ = _read_header(fh, path)
        data = fh.read()
    vertex = next((e for e in elements if e.name == "vertex"), None)
    if vertex is None or elements[0] is not vertex:
        raise MeshFormatError(f"{path}: the vertex element must come first")
    names = [p.name for p in vertex.properties]
    if not {"x", "y", "z"} <= set(names):
        raise MeshFormatError(f"{path}: vertices have no x, y, z")
    has_color = {"red", "green", "blue"} <= set(names)
    has_normal = {"nx", "ny", "nz"} <= set(names)
    positions, normals = array("f"), array("f")
    colors = bytearray()
    scalar = {p.name: struct.Struct("<" + p.type) for p in vertex.properties if not p.count_type}
    offset = 0
    try:
        for _ in range(vertex.count):
            values: dict[str, float] = {}
            for prop in vertex.properties:
                if prop.count_type:
                    count_struct = struct.Struct("<" + prop.count_type)
                    (count,) = count_struct.unpack_from(data, offset)
                    offset += count_struct.size + count * struct.calcsize("<" + prop.type)
                else:
                    reader = scalar[prop.name]
                    (values[prop.name],) = reader.unpack_from(data, offset)
                    offset += reader.size
            positions.extend((values["x"], values["y"], values["z"]))
            if has_color:
                colors += bytes((int(values["red"]), int(values["green"]), int(values["blue"])))
            if has_normal:
                normals.extend((values["nx"], values["ny"], values["nz"]))
    except struct.error as exc:
        raise MeshFormatError(f"{path}: file ends early") from exc
    return PointCloud(
        positions, bytes(colors) if has_color else None, normals if has_normal else None
    )


def write_point_cloud(cloud: PointCloud, path: Path) -> Path:
    """Binary PLY: float x, y, z, then uchar colours and float normals if present."""
    header = ["ply", "format binary_little_endian 1.0", "comment written by EZ2DIGITIZE"]
    header += [f"element vertex {cloud.count}"]
    header += [f"property float {axis}" for axis in "xyz"]
    fmt = "3f"
    if cloud.colors is not None:
        header += [f"property uchar {c}" for c in ("red", "green", "blue")]
        fmt += "3B"
    if cloud.normals is not None:
        header += [f"property float {n}" for n in ("nx", "ny", "nz")]
        fmt += "3f"
    header.append("end_header")
    record = struct.Struct("<" + fmt)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as ply:
        ply.write(("\n".join(header) + "\n").encode("ascii"))
        chunk = bytearray()
        p, c, n = cloud.positions, cloud.colors, cloud.normals
        for i in range(cloud.count):
            values: list[float | int] = [p[3 * i], p[3 * i + 1], p[3 * i + 2]]
            if c is not None:
                values += [c[3 * i], c[3 * i + 1], c[3 * i + 2]]
            if n is not None:
                values += [n[3 * i], n[3 * i + 1], n[3 * i + 2]]
            chunk += record.pack(*values)
            if len(chunk) >= 1 << 22:
                ply.write(chunk)
                chunk.clear()
        ply.write(chunk)
    return path
