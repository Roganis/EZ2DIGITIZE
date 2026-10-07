# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mesh files from anywhere: one model (`Mesh`), read from and written to the
usual formats.

meshio reads what OpenMVS writes and writes the exports of a project; this
reads the mesh files people already have, so they can be looked at and
converted (see ez2digitize.models). Here: PLY (ASCII or binary; vertex
colours, per-vertex or OpenMVS's per-corner texture coordinates), OBJ (with
its MTL and texture images), STL (binary or ASCII), OFF and 3MF (core
specification). glTF and GLB are in core.gltf, USDZ (written only) in
core.usdz.

A `Mesh` is triangles: polygons are split into fans as they are read.
Texture coordinates are kept per face corner, origin at the bottom left (as
in OBJ and PLY); each face has a material, which has a colour and maybe a
texture image. Normals are not kept: every program recomputes them.
"""

from __future__ import annotations

import io
import math
import re
import struct
import warnings
import zipfile
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO
from xml.etree import ElementTree

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from ez2digitize.core.meshio import (
    PLY_TYPES,
    MeshFormatError,
    TexturedMesh,
    _Element,
    _Property,
)
from ez2digitize.core.meshio import (
    write_3mf as _write_geometry_3mf,
)

Floats = NDArray[np.float32]
Indices = NDArray[np.uint32]
Bytes = NDArray[np.uint8]

MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tga": "image/x-tga",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}
# Image types every viewer and format takes; others are turned into PNG.
WEB_IMAGES = frozenset({".png", ".jpg", ".jpeg"})
GREY = 180  # the colour of vertices nothing says the colour of


@dataclass(frozen=True)
class Image:
    """A texture image: its file name (the suffix says its type) and its bytes."""

    name: str
    data: bytes

    @property
    def suffix(self) -> str:
        return Path(self.name).suffix.lower()

    @property
    def mime(self) -> str:
        return MIME.get(self.suffix, "application/octet-stream")


@dataclass
class Material:
    """A colour (RGBA, 0..1) and maybe a texture image, multiplied together."""

    name: str
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    image: Image | None = None


@dataclass
class Mesh:
    """Triangles, with vertex colours, texture coordinates and materials if the file had them.

    `positions` (n, 3) float32; `faces` (m, 3) vertex indices; `colors`
    (n, 3) RGB bytes or None; `uvs` (m, 3, 2): u, v per face corner, origin
    at the bottom left, or None; `face_materials` (m,): an index into
    `materials` per face, or None when there are no materials.
    """

    positions: Floats
    faces: Indices
    colors: Bytes | None = None
    uvs: Floats | None = None
    materials: list[Material] = field(default_factory=list)
    face_materials: Indices | None = None

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def textured(self) -> bool:
        return self.uvs is not None and any(m.image is not None for m in self.materials)

    def check(self, path: Path) -> Mesh:
        """This mesh, after checking it is whole; raises MeshFormatError if not."""
        if not self.face_count:
            raise MeshFormatError(f"{path}: no triangles")
        if int(self.faces.max()) >= self.vertex_count:
            raise MeshFormatError(f"{path}: a face refers to a vertex that doesn't exist")
        if self.face_materials is not None and (
            not self.materials or int(self.face_materials.max()) >= len(self.materials)
        ):
            raise MeshFormatError(f"{path}: a face refers to a material that doesn't exist")
        if not np.isfinite(self.positions).all():
            raise MeshFormatError(f"{path}: some vertices are not numbers")
        return self


def mesh(
    positions: object,
    faces: object,
    *,
    colors: object = None,
    uvs: object = None,
    materials: list[Material] | None = None,
    face_materials: object = None,
) -> Mesh:
    """A Mesh from array-likes, in the types `Mesh` keeps."""
    return Mesh(
        np.ascontiguousarray(np.asarray(positions, np.float32).reshape(-1, 3)),
        np.ascontiguousarray(np.asarray(faces, np.int64).reshape(-1, 3).astype(np.uint32)),
        None if colors is None else np.asarray(colors, np.uint8).reshape(-1, 3),
        None if uvs is None else np.asarray(uvs, np.float32).reshape(-1, 3, 2),
        materials or [],
        None if face_materials is None else np.asarray(face_materials, np.uint32).reshape(-1),
    )


def from_textured(source: TexturedMesh) -> Mesh:
    """The textured mesh of a project (meshio) as a `Mesh`."""
    materials = [
        Material(f"material_{i}", image=Image(path.name, path.read_bytes()))
        for i, path in enumerate(source.textures)
    ]
    return mesh(
        np.frombuffer(source.positions, np.float32),
        np.frombuffer(source.faces, np.uint32),
        uvs=np.frombuffer(source.uvs, np.float32),
        materials=materials,
        face_materials=np.frombuffer(source.texture_ids, np.uint32),
    )


def transformed(source: Mesh, matrix: NDArray[np.float64], scale: float = 1.0) -> Mesh:
    """The mesh with its vertices turned by a 3x3 `matrix` and scaled."""
    if np.linalg.det(matrix) < 0:  # a mirror would turn the faces inside out
        faces = np.ascontiguousarray(source.faces[:, ::-1])
        uvs = None if source.uvs is None else np.ascontiguousarray(source.uvs[:, ::-1])
    else:
        faces, uvs = source.faces, source.uvs
    positions = (source.positions.astype(np.float64) @ matrix.T * scale).astype(np.float32)
    return Mesh(positions, faces, source.colors, uvs, source.materials, source.face_materials)


# --- colours from the texture --------------------------------------------------


def vertex_colors(source: Mesh) -> Bytes:
    """A colour per vertex: the file's, else the average of the texture under its
    corners (or of its faces' material colours), else grey."""
    if source.colors is not None:
        return source.colors
    n = source.vertex_count
    sums = np.zeros((n, 3), np.float64)
    counts = np.zeros(n, np.float64)
    corners = source.faces.reshape(-1)
    face_material = (
        source.face_materials
        if source.face_materials is not None
        else np.zeros(source.face_count, np.uint32)
    )
    for index, material in enumerate(source.materials or [Material("default")]):
        chosen = np.repeat(face_material == index, 3)
        if not chosen.any():
            continue
        tint = np.array(material.color[:3], np.float64)
        vertices = corners[chosen]
        picked = None
        if material.image is not None and source.uvs is not None:
            picked = _texels(material.image, source.uvs, chosen)
        rgb = picked * tint if picked is not None else np.tile(tint * 255, (len(vertices), 1))
        for channel in range(3):
            sums[:, channel] += np.bincount(vertices, rgb[:, channel], minlength=n)
        counts += np.bincount(vertices, minlength=n)
    colors = np.full((n, 3), float(GREY))
    seen = counts > 0
    colors[seen] = sums[seen] / counts[seen, None]
    return np.clip(np.round(colors), 0, 255).astype(np.uint8)


def _texels(image: Image, uvs: Floats, chosen: NDArray[np.bool_]) -> NDArray[np.float64]:
    """The texture's colour (n, 3) at the chosen corners (nearest texel)."""
    try:
        with PILImage.open(io.BytesIO(image.data)) as opened:
            pixels = np.asarray(opened.convert("RGB"))
    except (OSError, UnidentifiedImageError, PILImage.DecompressionBombError) as exc:
        raise MeshFormatError(f"cannot read the texture {image.name}: {exc}") from exc
    height, width = pixels.shape[:2]
    uv = uvs.reshape(-1, 2)[chosen].astype(np.float64)
    u = uv[:, 0] - np.floor(uv[:, 0])  # textures repeat
    v = uv[:, 1] - np.floor(uv[:, 1])
    x = np.clip((u * width).astype(np.int64), 0, width - 1)
    y = np.clip(((1 - v) * height).astype(np.int64), 0, height - 1)
    texels: NDArray[np.float64] = pixels[y, x].astype(np.float64)
    return texels


def web_image(image: Image) -> Image:
    """The image as PNG or JPEG (what glTF and USDZ take), converted if it is another type."""
    if image.suffix in WEB_IMAGES:
        return image
    try:
        with PILImage.open(io.BytesIO(image.data)) as opened:
            out = io.BytesIO()
            opened.convert("RGBA" if "A" in opened.getbands() else "RGB").save(out, "PNG")
    except (OSError, UnidentifiedImageError, PILImage.DecompressionBombError) as exc:
        raise MeshFormatError(f"cannot read the texture {image.name}: {exc}") from exc
    return Image(str(Path(image.name).with_suffix(".png")), out.getvalue())


def texture_names(stem: str, images: list[Image]) -> list[str]:
    """File names for a mesh's texture images when they are written beside it."""
    return [f"{stem}_texture{i}{image.suffix or '.png'}" for i, image in enumerate(images)]


# --- triangles from polygons ------------------------------------------------------


def _fan(polygons: list[list[int]]) -> tuple[list[int], list[int]]:
    """Triangles from polygons as fans: (corner positions into the flat polygon
    list, the polygon of each triangle). Degenerate polygons (< 3) are dropped."""
    corners: list[int] = []
    owners: list[int] = []
    start = 0
    for index, polygon in enumerate(polygons):
        for k in range(1, len(polygon) - 1):
            corners += (start, start + k, start + k + 1)
            owners.append(index)
        start += len(polygon)
    return corners, owners


def _uniform_fan(table: NDArray[np.int64]) -> NDArray[np.int64]:
    """Fans of polygons that all have k corners: (m, k) to (m * (k - 2), 3)."""
    k = table.shape[1]
    if k == 3:
        return table
    pick = np.array([[0, i, i + 1] for i in range(1, k - 1)])
    return table[:, pick].reshape(-1, 3)


# --- PLY ----------------------------------------------------------------------------


@dataclass
class _Ply:
    format: str
    elements: list[_Element]
    comments: list[str]


def ply_header(path: Path) -> _Ply:
    """A PLY file's header: its format, elements and comments."""
    with path.open("rb") as fh:
        return _ply_header(fh, path)


def _ply_header(fh: BinaryIO, path: Path) -> _Ply:
    if fh.readline().strip() != b"ply":
        raise MeshFormatError(f"{path}: not a PLY file")
    header = _Ply("", [], [])
    while True:
        raw = fh.readline()
        if not raw:
            raise MeshFormatError(f"{path}: header has no end_header")
        words = raw.decode("ascii", errors="replace").split()
        if not words:
            continue
        keyword = words[0]
        try:
            if keyword == "end_header":
                if header.format not in ("ascii", "binary_little_endian", "binary_big_endian"):
                    raise MeshFormatError(f"{path}: unknown PLY format {header.format!r}")
                return header
            if keyword == "format":
                header.format = words[1]
            elif keyword in ("comment", "obj_info"):
                header.comments.append(raw.decode("utf-8", "replace").strip()[len(keyword) + 1 :])
            elif keyword == "element":
                header.elements.append(_Element(words[1], int(words[2]), []))
            elif keyword == "property":
                if not header.elements:
                    raise MeshFormatError(f"{path}: property before any element")
                if words[1] == "list":
                    prop = _Property(words[4], _ply_type(words[3], path), _ply_type(words[2], path))
                else:
                    prop = _Property(words[2], _ply_type(words[1], path))
                header.elements[-1].properties.append(prop)
        except (IndexError, ValueError) as exc:
            raise MeshFormatError(f"{path}: bad header line {raw.strip()!r}") from exc


def _ply_type(name: str, path: Path) -> str:
    try:
        return PLY_TYPES[name]
    except KeyError:
        raise MeshFormatError(f"{path}: unknown PLY type {name!r}") from None


# A list property's values: a table when every record has as many, else one array each.
ListValues = NDArray[np.float64] | list[NDArray[np.float64]]
Element = dict[str, NDArray[np.float64] | ListValues]


def read_ply_elements(path: Path) -> tuple[_Ply, dict[str, Element]]:
    """Every element of a PLY file, property by property (lists as described by ListValues)."""
    with path.open("rb") as fh:
        header = _ply_header(fh, path)
        data = fh.read()
    out: dict[str, Element] = {}
    if header.format == "ascii":
        tokens = data.split()
        position = 0
        for element in header.elements:
            out[element.name], position = _ascii_element(tokens, position, element, path)
    else:
        order = "<" if header.format == "binary_little_endian" else ">"
        offset = 0
        for element in header.elements:
            out[element.name], offset = _binary_element(data, offset, element, order, path)
    return header, out


def _binary_element(
    data: bytes, offset: int, element: _Element, order: str, path: Path
) -> tuple[Element, int]:
    lists = [p for p in element.properties if p.count_type is not None]
    if not lists:
        dtype = np.dtype([(p.name, order + p.type) for p in element.properties])
        end = offset + dtype.itemsize * element.count
        if end > len(data):
            raise MeshFormatError(f"{path}: file ends inside the {element.name} list")
        table = np.frombuffer(data, dtype, element.count, offset)
        return {p.name: table[p.name].astype(np.float64) for p in element.properties}, end
    # Usually every record has lists as long as the first's (triangles): one table.
    sizes = _first_list_sizes(data, offset, element, order, path) if element.count else {}
    fields: list[tuple[str, str] | tuple[str, str, int]] = []
    for p in element.properties:
        if p.count_type is None:
            fields.append((p.name, order + p.type))
        else:
            fields.append((p.name + "#count", order + p.count_type))
            if sizes[p.name]:
                fields.append((p.name, order + p.type, sizes[p.name]))
    dtype = np.dtype(fields)
    end = offset + dtype.itemsize * element.count
    if end <= len(data):
        table = np.frombuffer(data, dtype, element.count, offset)
        if all(np.all(table[p.name + "#count"] == sizes[p.name]) for p in lists):
            values: Element = {}
            for p in element.properties:
                if p.count_type is None:
                    values[p.name] = table[p.name].astype(np.float64)
                elif sizes[p.name]:
                    values[p.name] = table[p.name].astype(np.float64).reshape(element.count, -1)
                else:
                    values[p.name] = np.zeros((element.count, 0))
            return values, end
    return _binary_records(data, offset, element, order, path)


def _first_list_sizes(
    data: bytes, offset: int, element: _Element, order: str, path: Path
) -> dict[str, int]:
    sizes = {}
    try:
        for p in element.properties:
            if p.count_type is None:
                offset += struct.calcsize(order + p.type)
            else:
                (count,) = struct.unpack_from(order + p.count_type, data, offset)
                sizes[p.name] = int(count)
                offset += struct.calcsize(order + p.count_type) + count * struct.calcsize(
                    order + p.type
                )
    except struct.error as exc:
        raise MeshFormatError(f"{path}: file ends inside the {element.name} list") from exc
    return sizes


def _binary_records(
    data: bytes, offset: int, element: _Element, order: str, path: Path
) -> tuple[Element, int]:
    """Record by record, for lists of varying length (polygons of mixed sizes)."""
    scalars: dict[str, list[float]] = {}
    lists: dict[str, list[NDArray[np.float64]]] = {}
    readers = {
        p.name: (
            struct.Struct(order + p.type),
            struct.Struct(order + p.count_type) if p.count_type else None,
        )
        for p in element.properties
    }
    for p in element.properties:
        if p.count_type:
            lists[p.name] = []
        else:
            scalars[p.name] = []
    try:
        for _ in range(element.count):
            for p in element.properties:
                value, counter = readers[p.name]
                if counter is None:
                    scalars[p.name].append(value.unpack_from(data, offset)[0])
                    offset += value.size
                else:
                    (count,) = counter.unpack_from(data, offset)
                    offset += counter.size
                    items = np.frombuffer(data, order + p.type, count, offset)
                    lists[p.name].append(items.astype(np.float64))
                    offset += value.size * count
    except (struct.error, ValueError) as exc:
        raise MeshFormatError(f"{path}: file ends inside the {element.name} list") from exc
    values: Element = {name: np.array(v, np.float64) for name, v in scalars.items()}
    values.update(lists)
    return values, offset


def _ascii_element(
    tokens: list[bytes], position: int, element: _Element, path: Path
) -> tuple[Element, int]:
    try:
        if not any(p.count_type for p in element.properties):
            width = len(element.properties)
            end = position + width * element.count
            if end > len(tokens):
                raise MeshFormatError(f"{path}: file ends inside the {element.name} list")
            table = np.array(tokens[position:end], np.float64).reshape(element.count, width)
            return {p.name: table[:, i] for i, p in enumerate(element.properties)}, end
        scalars: dict[str, list[float]] = {}
        lists: dict[str, list[NDArray[np.float64]]] = {}
        for p in element.properties:
            if p.count_type:
                lists[p.name] = []
            else:
                scalars[p.name] = []
        for _ in range(element.count):
            for p in element.properties:
                if p.count_type is None:
                    scalars[p.name].append(float(tokens[position]))
                    position += 1
                else:
                    count = int(float(tokens[position]))
                    words = tokens[position + 1 : position + 1 + count]
                    if len(words) < count:
                        raise IndexError
                    lists[p.name].append(np.array(words, np.float64))
                    position += 1 + count
    except (IndexError, ValueError) as exc:
        raise MeshFormatError(f"{path}: file ends inside the {element.name} list") from exc
    values: Element = {name: np.array(v, np.float64) for name, v in scalars.items()}
    for name, rows in lists.items():
        sizes = {len(r) for r in rows}
        values[name] = np.array(rows).reshape(len(rows), -1) if len(sizes) <= 1 else rows
    return values, position


def _polygon_triangles(
    indices: ListValues, extra: ListValues | None, per_corner: int
) -> tuple[NDArray[np.int64], NDArray[np.float64] | None]:
    """Triangles (and their corners' extra values) from a face list property."""
    if not isinstance(indices, list):
        table = indices.astype(np.int64)
        triangles = _uniform_fan(table)
        if (
            extra is None
            or isinstance(extra, list)
            or extra.shape[1] != per_corner * table.shape[1]
        ):
            return triangles, None
        corner_values = extra.reshape(len(table), table.shape[1], per_corner)
        k = table.shape[1]
        pick = np.array([[0, i, i + 1] for i in range(1, k - 1)]) if k > 3 else np.arange(3)[None]
        return triangles, corner_values[:, pick].reshape(-1, 3, per_corner)
    polygons = [list(p.astype(np.int64)) for p in indices]
    corners, owners = _fan(polygons)
    flat = np.concatenate([np.asarray(p, np.int64) for p in polygons]) if polygons else np.zeros(0)
    triangles = flat[np.array(corners, np.int64)].reshape(-1, 3).astype(np.int64)
    if extra is None:
        return triangles, None
    rows = list(extra) if isinstance(extra, list) else list(extra)
    if any(len(r) != per_corner * len(p) for r, p in zip(rows, polygons, strict=False)):
        return triangles, None
    values = np.concatenate([np.asarray(r).reshape(-1, per_corner) for r in rows])
    return triangles, values[np.array(corners, np.int64)].reshape(-1, 3, per_corner)


def read_ply(path: Path) -> Mesh:
    """A PLY triangle mesh: positions, faces (polygons become triangles), vertex
    colours, and texture coordinates per vertex (u, v / s, t) or per corner
    (texcoord, OpenMVS) with the images named in `comment TextureFile` lines."""
    header, elements = read_ply_elements(path)
    vertex, face = elements.get("vertex"), elements.get("face")
    if vertex is None or not {"x", "y", "z"} <= vertex.keys():
        raise MeshFormatError(f"{path}: vertices have no x, y, z")
    if face is None:
        raise MeshFormatError(f"{path}: no faces (a point cloud or splats, not a mesh)")
    positions = np.stack([np.asarray(vertex[a]) for a in "xyz"], axis=1)
    index_name = next((n for n in ("vertex_indices", "vertex_index") if n in face), None)
    if index_name is None:
        raise MeshFormatError(f"{path}: faces have no vertex_indices")
    triangles, corner_uvs = _polygon_triangles(face[index_name], face.get("texcoord"), 2)
    if not len(triangles):
        raise MeshFormatError(f"{path}: no triangles")
    face_owner = _face_owners(face[index_name])

    colors = _ply_colors(vertex)
    uvs = corner_uvs
    if uvs is None:
        names = next(
            (pair for pair in (("u", "v"), ("s", "t"), ("texture_u", "texture_v")) if
             set(pair) <= vertex.keys()),
            None,
        )  # fmt: skip
        if names is not None:
            per_vertex = np.stack([np.asarray(vertex[n]) for n in names], axis=1)
            uvs = per_vertex[triangles]
    textures = [c.split(" ", 1)[1].strip() for c in header.comments if c.startswith("TextureFile ")]
    materials: list[Material] = []
    face_materials = None
    if textures and uvs is not None:
        materials = [Material(f"material_{i}", image=_image_file(path.parent / name))
                     for i, name in enumerate(textures)]  # fmt: skip
        number = face.get("texnumber")
        ids = np.asarray(number, np.int64)[face_owner] if number is not None else None
        face_materials = (
            np.clip(ids, 0, len(materials) - 1) if ids is not None else np.zeros(len(triangles))
        )
    return mesh(
        positions,
        triangles,
        colors=colors,
        uvs=uvs,
        materials=materials,
        face_materials=face_materials,
    ).check(path)


def _face_owners(indices: ListValues) -> NDArray[np.int64]:
    """The polygon each triangle of `_polygon_triangles` comes from."""
    if isinstance(indices, list):
        return np.array(_fan([list(p) for p in indices])[1], np.int64)
    k = indices.shape[1]
    return np.repeat(np.arange(len(indices)), max(k - 2, 0))


def _ply_colors(vertex: Element) -> Bytes | None:
    for names in (("red", "green", "blue"), ("r", "g", "b"),
                  ("diffuse_red", "diffuse_green", "diffuse_blue")):  # fmt: skip
        if set(names) <= vertex.keys():
            rgb = np.stack([np.asarray(vertex[n], np.float64) for n in names], axis=1)
            if len(rgb) and rgb.max() <= 1.0 and not np.equal(np.mod(rgb, 1), 0).all():
                rgb = rgb * 255  # float colours, 0..1
            return np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    return None


def _image_file(path: Path) -> Image | None:
    try:
        return Image(path.name, path.read_bytes())
    except OSError:
        return None  # a missing texture: the mesh still shows, in its colours


def write_ply(source: Mesh, path: Path) -> list[Path]:
    """Binary PLY: float x, y, z, uchar colours; faces as int index lists, with
    OpenMVS's per-corner `texcoord` (and `texnumber`) and its `comment
    TextureFile` lines when textured, the images written beside it."""
    images = _face_images(source)
    names = texture_names(path.stem, images) if images else []
    header = ["ply", "format binary_little_endian 1.0", "comment written by EZ2DIGITIZE"]
    header += [f"comment TextureFile {name}" for name in names]
    header += [f"element vertex {source.vertex_count}"]
    header += [f"property float {a}" for a in "xyz"]
    vertex_fields: list[tuple[str, str]] = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if source.colors is not None:
        header += [f"property uchar {c}" for c in ("red", "green", "blue")]
        vertex_fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    header += [f"element face {source.face_count}", "property list uchar int vertex_indices"]
    face_fields: list[tuple[str, str] | tuple[str, str, int]] = [
        ("n", "u1"), ("i", "<i4", 3)
    ]  # fmt: skip
    if images:
        header.append("property list uchar float texcoord")
        face_fields += [("m", "u1"), ("uv", "<f4", 6)]
        if len(images) > 1:
            header.append("property int texnumber")
            face_fields.append(("t", "<i4"))
    header.append("end_header")

    vertices = np.zeros(source.vertex_count, np.dtype(vertex_fields))
    for i, axis in enumerate("xyz"):
        vertices[axis] = source.positions[:, i]
    if source.colors is not None:
        for i, channel in enumerate(("red", "green", "blue")):
            vertices[channel] = source.colors[:, i]
    faces = np.zeros(source.face_count, np.dtype(face_fields))
    faces["n"] = 3
    faces["i"] = source.faces
    if images:
        assert source.uvs is not None and source.face_materials is not None
        faces["m"] = 6
        faces["uv"] = source.uvs.reshape(-1, 6)
        if len(images) > 1:
            faces["t"] = source.face_materials
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(("\n".join(header) + "\n").encode("utf-8"))
        out.write(vertices.tobytes())
        out.write(faces.tobytes())
    written = [path]
    for name, image in zip(names, images, strict=True):
        (path.parent / name).write_bytes(image.data)
        written.append(path.parent / name)
    return written


def _face_images(source: Mesh) -> list[Image]:
    """The images, one per material, if every material has one (PLY and OBJ export)."""
    if source.uvs is None or not source.materials or source.face_materials is None:
        return []
    images = [m.image for m in source.materials]
    if any(image is None for image in images):
        return []
    return [image for image in images if image is not None]


# --- OBJ ----------------------------------------------------------------------------


def read_obj(path: Path) -> Mesh:
    """A Wavefront OBJ: v (with r g b if present), vt, f (any polygon; negative
    indices), usemtl, and the materials of its mtllib files (Kd, d, map_Kd)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise MeshFormatError(f"{path}: {exc}") from exc
    # Statements found by pattern rather than line by line: OBJ files of
    # scans run to millions of lines. A row is what follows the keyword.
    vertex_rows = OBJ_VERTEX.findall(text)
    texcoord_rows = OBJ_TEXCOORD.findall(text)
    face_rows: list[str] = []
    face_starts: list[int] = []
    polygon_material: list[int] = []
    library: dict[str, Material] = {}
    used: dict[str, int] = {}
    current = -1
    for match in OBJ_STATEMENT.finditer(text):
        keyword, rest = match.group(1), match.group(2).strip()
        if keyword == "f":
            face_rows.append(rest)
            face_starts.append(match.start())
            polygon_material.append(current)
        elif keyword == "usemtl" and rest:
            current = used.setdefault(rest, len(used))
        elif keyword == "mtllib" and rest:
            for name in _mtl_names(rest):
                library.update(read_mtl(path.parent / name))
    if not vertex_rows or not face_rows:
        raise MeshFormatError(f"{path}: no faces")
    positions, colors = _obj_vertices(vertex_rows, path)
    texcoords = _obj_texcoords(texcoord_rows, path)
    triangles, uv_index, owners = _obj_faces(
        face_rows, face_starts, text, len(positions), len(texcoords), path
    )
    if (triangles < 0).any() or (triangles >= len(positions)).any():
        raise MeshFormatError(f"{path}: a face refers to a vertex that doesn't exist")
    uvs = None
    if len(texcoords) and uv_index is not None:
        if (uv_index >= len(texcoords)).any():
            raise MeshFormatError(
                f"{path}: a face refers to a texture coordinate that doesn't exist"
            )
        table = np.vstack([texcoords.astype(np.float32), np.zeros((1, 2), np.float32)])
        uvs = table[np.where(uv_index < 0, len(texcoords), uv_index)].reshape(-1, 3, 2)

    materials: list[Material] = []
    face_materials = None
    owner_material = np.array(polygon_material, np.int64)[owners]
    if (owner_material >= 0).any():
        # Only the materials faces use, in the order the file named them.
        by_index = sorted(used, key=used.__getitem__)
        kept, face_materials = np.unique(owner_material, return_inverse=True)
        materials = [
            Material("default") if i < 0 else library.get(by_index[i], Material(by_index[i]))
            for i in kept.tolist()
        ]
    rgb = None
    if colors is not None:
        rgb = colors * 255 if colors.max() <= 1.0 else colors
    return mesh(
        positions,
        triangles,
        colors=None if rgb is None else np.clip(np.round(rgb), 0, 255),
        uvs=uvs,
        materials=materials,
        face_materials=face_materials,
    ).check(path)


# A statement's keyword at the start of a line, and the rest of the line up to a comment.
OBJ_VERTEX = re.compile(r"^[ \t]*v[ \t]+([^\n#]*)", re.MULTILINE)
OBJ_TEXCOORD = re.compile(r"^[ \t]*vt[ \t]+([^\n#]*)", re.MULTILINE)
OBJ_STATEMENT = re.compile(r"^[ \t]*(f|usemtl|mtllib)[ \t]+([^\n#]*)", re.MULTILINE)


def _numbers(rows: list[str], dtype: type[np.float64] | type[np.int64]) -> NDArray[Any] | None:
    """The rows' numbers as a table, parsed in one go, when every row has as
    many; else (or if a word isn't a number) None."""
    width = _width(rows)
    if width is None:
        return None
    # A word that isn't a number ends the parse early, with a warning or an
    # error depending on NumPy's version.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            values = np.fromstring(" ".join(rows), dtype=dtype, sep=" ")
        except ValueError:
            return None
    if values.size != len(rows) * width:
        return None
    return values.reshape(len(rows), width)


def _width(rows: list[str]) -> int | None:
    """How many words each row has, if all have as many (else None)."""
    if not rows:
        return None
    words = ("| " + " | ".join(rows)).split()  # a marker before each row
    width, extra = divmod(len(words), len(rows))
    if extra or width < 2 or set(words[::width]) != {"|"}:
        return None  # some row has more words, some fewer
    return width - 1


def _obj_vertices(
    rows: list[str], path: Path
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Positions, and colours if every vertex has them (v x y z r g b)."""
    try:
        table = _numbers(rows, np.float64)
        if table is None:
            padded = [(r.split()[:6] + ["nan"] * 6)[:6] for r in rows]
            table = np.array(padded, np.float64)
    except ValueError as exc:
        raise MeshFormatError(f"{path}: a vertex can't be read") from exc
    if table.shape[1] < 3 or np.isnan(table[:, :3]).any():
        raise MeshFormatError(f"{path}: a vertex has fewer than three coordinates")
    colors = table[:, 3:6] if table.shape[1] >= 6 and not np.isnan(table[:, 3:6]).any() else None
    return table[:, :3], colors


def _obj_texcoords(rows: list[str], path: Path) -> NDArray[np.float64]:
    try:
        table = _numbers(rows, np.float64)
        if table is not None and table.shape[1] >= 2:
            return table[:, :2]
        return np.array([(r.split()[:2] + ["0"])[:2] for r in rows], np.float64).reshape(-1, 2)
    except ValueError as exc:
        raise MeshFormatError(f"{path}: a texture coordinate can't be read") from exc


def _obj_faces(
    rows: list[str], starts: list[int], text: str, vertices: int, texcoords: int, path: Path
) -> tuple[NDArray[np.int64], NDArray[np.int64] | None, NDArray[np.int64]]:
    """Triangles (0-based vertex indices), their corners' texture coordinate
    indices (-1: none; None if no face has any), and the face each comes from."""
    fast = _obj_triangles(rows, vertices, texcoords)
    if fast is not None:
        return fast
    polygons: list[list[int]] = []
    polygon_uvs: list[list[int]] = []
    for row, start in zip(rows, starts, strict=True):
        try:
            corners = [w.split("/") for w in row.split()]
            polygons.append([_obj_index(c[0], vertices) for c in corners])
            polygon_uvs.append(
                [_obj_index(c[1], texcoords) for c in corners]
                if all(len(c) > 1 and c[1] for c in corners)
                else [-1] * len(corners)
            )
        except (ValueError, IndexError) as exc:
            number = text.count("\n", 0, start) + 1
            raise MeshFormatError(f"{path}: line {number} can't be read") from exc
    fans, owners = _fan(polygons)
    pick = np.array(fans, np.int64)
    triangles = np.array([v for p in polygons for v in p], np.int64)[pick].reshape(-1, 3)
    uv_index = np.array([t for u in polygon_uvs for t in u], np.int64)[pick].reshape(-1, 3)
    return triangles, uv_index if (uv_index >= 0).any() else None, np.array(owners, np.int64)


def _obj_triangles(
    rows: list[str], vertices: int, texcoords: int
) -> tuple[NDArray[np.int64], NDArray[np.int64] | None, NDArray[np.int64]] | None:
    """The common case in one go: only triangles, every corner written alike
    (v, v/vt, v/vt/vn or v//vn). None if the faces aren't like that."""
    if _width(rows) != 3:
        return None
    parts = rows[0].split()[0].split("/")
    has_uv = len(parts) > 1 and parts[1] != ""
    joined = " ".join(rows)
    corners = 3 * len(rows)
    if joined.count("/") != (len(parts) - 1) * corners:
        return None
    if joined.count("//") != (0 if has_uv or len(parts) < 3 else corners):
        return None
    numbers = _numbers([joined.replace("//", "/0/").replace("/", " ")], np.int64)
    if numbers is None or numbers.size != corners * len(parts):
        return None
    table = numbers.reshape(len(rows), 3, len(parts))

    def zero_based(index: NDArray[np.int64], count: int) -> NDArray[np.int64]:
        return np.where(index > 0, index - 1, np.where(index < 0, count + index, -1))

    triangles = zero_based(table[..., 0], vertices)
    uv_index = zero_based(table[..., 1], texcoords) if has_uv else None
    return triangles, uv_index, np.arange(len(rows), dtype=np.int64)


def _obj_index(text: str, count: int) -> int:
    index = int(text)
    return index - 1 if index > 0 else count + index if index < 0 else -1


def _mtl_names(text: str) -> list[str]:
    """The file names after mtllib: usually one, which may contain spaces."""
    text = text.strip()
    if text.lower().endswith(".mtl") and text.lower().count(".mtl") == 1:
        return [text]
    return text.split()


def read_mtl(path: Path) -> dict[str, Material]:
    """The materials of an MTL file: diffuse colour (Kd), opacity (d, Tr) and map_Kd."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}  # a missing MTL: the faces keep their material names, uncoloured
    materials: dict[str, Material] = {}
    current: Material | None = None
    for line in text.splitlines():
        words = line.split("#", 1)[0].split()
        if not words:
            continue
        keyword = words[0].lower()
        try:
            if keyword == "newmtl" and len(words) > 1:
                current = materials[line.split(None, 1)[1].strip()] = Material(words[1])
            elif current is None:
                continue
            elif keyword == "kd" and len(words) >= 4:
                r, g, b = (float(w) for w in words[1:4])
                current.color = (r, g, b, current.color[3])
            elif keyword == "d" and len(words) >= 2:
                current.color = (*current.color[:3], float(words[-1]))
            elif keyword == "tr" and len(words) >= 2:
                current.color = (*current.color[:3], 1.0 - float(words[-1]))
            elif keyword == "map_kd" and len(words) >= 2:
                current.image = _image_file(path.parent / _map_file(line.split(None, 1)[1]))
        except ValueError:
            continue
    return materials


def _map_file(text: str) -> str:
    """The file name of a map_Kd statement, after its options (-o 0 0, -s 1 1 1...)."""
    words = text.split()
    i = 0
    while i < len(words) and words[i].startswith("-"):
        i += 1
        while i < len(words) and re.fullmatch(r"[-+]?[\d.]+(e[-+]?\d+)?|on|off", words[i]):
            i += 1
    name = " ".join(words[i:]) or words[-1]
    return name.replace("\\", "/")


def write_obj(source: Mesh, path: Path) -> list[Path]:
    """OBJ (vertex colours as `v x y z r g b`), its MTL and texture images beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    materials = source.materials or ([Material("default")] if source.uvs is not None else [])
    images = [m.image for m in materials if m.image is not None]
    names = iter(texture_names(stem, images))
    written = [path]
    lines = ["# EZ2DIGITIZE\n"]
    if materials:
        mtl = path.with_suffix(".mtl")
        entries = []
        for material in materials:
            r, g, b, a = material.color
            entry = f"newmtl {material.name}\nKa 1 1 1\nKd {r:.6g} {g:.6g} {b:.6g}\nKs 0 0 0\n"
            entry += f"d {a:.6g}\nillum 1\n"
            if material.image is not None:
                name = next(names)
                (path.parent / name).write_bytes(material.image.data)
                written.append(path.parent / name)
                entry += f"map_Kd {name}\n"
            entries.append(entry + "\n")
        mtl.write_text("".join(entries), encoding="utf-8")
        written.insert(1, mtl)
        lines.append(f"mtllib {mtl.name}\n")
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.writelines(lines)
        p = source.positions
        if source.colors is not None:
            c = (source.colors.astype(np.float64) / 255).tolist()
            out.writelines(
                f"v {x:.6g} {y:.6g} {z:.6g} {r:.4g} {g:.4g} {b:.4g}\n"
                for (x, y, z), (r, g, b) in zip(p.tolist(), c, strict=True)
            )
        else:
            out.writelines(f"v {x:.6g} {y:.6g} {z:.6g}\n" for x, y, z in p.tolist())
        if source.uvs is not None:
            out.writelines(f"vt {u:.6g} {v:.6g}\n" for u, v in source.uvs.reshape(-1, 2).tolist())
        f = (source.faces.astype(np.int64) + 1).tolist()
        groups = (
            source.face_materials
            if source.face_materials is not None and source.materials
            else np.zeros(source.face_count, np.uint32)
        )
        for index in range(max(len(materials), 1)):
            chosen = np.flatnonzero(groups == index)
            if not len(chosen):
                continue
            if materials:
                out.write(f"usemtl {materials[index].name}\n")
            if source.uvs is not None:
                out.writelines(
                    f"f {f[k][0]}/{3 * k + 1} {f[k][1]}/{3 * k + 2} {f[k][2]}/{3 * k + 3}\n"
                    for k in chosen.tolist()
                )
            else:
                out.writelines(f"f {f[k][0]} {f[k][1]} {f[k][2]}\n" for k in chosen.tolist())
    return written


# --- STL ----------------------------------------------------------------------------

STL_RECORD = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attribute", "<u2")])


def read_stl(path: Path) -> Mesh:
    """Binary or ASCII STL; corners at the same place are joined into one vertex."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise MeshFormatError(f"{path}: {exc}") from exc
    count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else -1
    if count >= 0 and len(data) == 84 + 50 * count:
        corners = np.frombuffer(data, STL_RECORD, count, 84)["corners"].reshape(-1, 3)
    elif data.lstrip()[:5].lower() == b"solid":
        found = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)
        try:
            corners = np.array(found, np.float64).reshape(-1, 3)
        except ValueError as exc:
            raise MeshFormatError(f"{path}: a vertex can't be read") from exc
        if len(corners) % 3:
            raise MeshFormatError(f"{path}: a facet doesn't have three vertices")
    else:
        raise MeshFormatError(f"{path}: not an STL file (wrong size for a binary STL)")
    if not len(corners):
        raise MeshFormatError(f"{path}: no triangles")
    positions, inverse = np.unique(corners.astype(np.float32), axis=0, return_inverse=True)
    return mesh(positions, inverse.reshape(-1, 3)).check(path)


def write_stl(source: Mesh, path: Path) -> list[Path]:
    """Binary STL of the geometry (no colour)."""
    corners = source.positions[source.faces.astype(np.int64)].astype(np.float64)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    records = np.zeros(source.face_count, STL_RECORD)
    records["normal"] = normals / np.where(lengths > 0, lengths, 1)
    records["corners"] = corners
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(b"EZ2DIGITIZE binary STL".ljust(80, b" "))
        out.write(struct.pack("<I", source.face_count))
        out.write(records.tobytes())
    return [path]


# --- OFF ----------------------------------------------------------------------------


def read_off(path: Path) -> Mesh:
    """OFF, with its prefixes: C (vertex colours), N (normals), ST (texture coordinates)."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise MeshFormatError(f"{path}: {exc}") from exc
    rows = iter(words for line in lines if (words := line.split("#", 1)[0].split()))
    try:
        first = next(rows)
        keyword = first[0].upper()
        match = re.fullmatch(r"(ST)?(C)?(N)?(4)?(n)?OFF", keyword, re.IGNORECASE)
        if match is None:
            raise MeshFormatError(f"{path}: not an OFF file")
        if match.group(4) or match.group(5):
            raise MeshFormatError(f"{path}: only 3D OFF files are read")
        counts = first[1:] or next(rows)
        n_vertices, n_faces = int(counts[0]), int(counts[1])
        has_st, has_color, has_normal = (bool(match.group(i)) for i in (1, 2, 3))
        positions = np.zeros((n_vertices, 3))
        colors = np.zeros((n_vertices, 3)) if has_color else None
        for i in range(n_vertices):
            words = next(rows)
            positions[i] = [float(w) for w in words[:3]]
            if colors is not None:
                at = 3 + (3 if has_normal else 0)
                colors[i] = [float(w) for w in words[at : at + 3]]
        polygons = []
        for _ in range(n_faces):
            words = next(rows)
            k = int(words[0])
            polygons.append([int(w) for w in words[1 : 1 + k]])
    except (StopIteration, ValueError, IndexError) as exc:
        raise MeshFormatError(f"{path}: the file ends early or a line can't be read") from exc
    del has_st  # texture coordinates in OFF are rare and untextured: not kept
    corners, _owners = _fan(polygons)
    flat = np.array([v for p in polygons for v in p], np.int64)
    triangles = flat[np.array(corners, np.int64)].reshape(-1, 3)
    rgb = None
    if colors is not None:
        rgb = colors * 255 if len(colors) and colors.max() <= 1.0 else colors
        rgb = np.clip(np.round(rgb), 0, 255)
    return mesh(positions, triangles, colors=rgb).check(path)


def write_off(source: Mesh, path: Path) -> list[Path]:
    """OFF; COFF with a colour per vertex (from the texture if the mesh has one)."""
    colors = vertex_colors(source) if source.colors is not None or source.materials else None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("COFF\n" if colors is not None else "OFF\n")
        out.write(f"{source.vertex_count} {source.face_count} 0\n")
        p = source.positions.tolist()
        if colors is not None:
            c = colors.tolist()
            out.writelines(
                f"{x:.6g} {y:.6g} {z:.6g} {r} {g} {b} 255\n"
                for (x, y, z), (r, g, b) in zip(p, c, strict=True)
            )
        else:
            out.writelines(f"{x:.6g} {y:.6g} {z:.6g}\n" for x, y, z in p)
        out.writelines(f"3 {a} {b} {c}\n" for a, b, c in source.faces.tolist())
    return [path]


# --- 3MF ----------------------------------------------------------------------------

CORE_3MF = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"
RELS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
# Metres per 3MF unit.
UNITS_3MF = {
    "micron": 1e-6, "millimeter": 1e-3, "centimeter": 1e-2, "inch": 0.0254, "foot": 0.3048,
    "meter": 1.0,
}  # fmt: skip


def read_3mf(path: Path) -> Mesh:
    """The build items of a 3MF file (core specification): objects and their
    components, placed by their transforms. Colours and textures are not read."""
    try:
        with zipfile.ZipFile(path) as package:
            model = _3mf_model_path(package)
            root = ElementTree.fromstring(package.read(model))
    except (OSError, zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise MeshFormatError(f"{path}: not a 3MF file ({exc})") from exc
    objects = {o.get("id"): o for o in root.iter(f"{CORE_3MF}object")}
    positions: list[NDArray[np.float64]] = []
    faces: list[NDArray[np.int64]] = []
    count = 0

    def add(object_id: str | None, matrix: NDArray[np.float64], depth: int) -> None:
        nonlocal count
        obj = objects.get(object_id)
        if obj is None or depth > 16:
            raise MeshFormatError(f"{path}: a build item refers to an unknown object")
        found = obj.find(f"{CORE_3MF}mesh")
        if found is not None:
            vertices = np.array(
                [[float(v.get(a, "0")) for a in "xyz"] for v in found.iter(f"{CORE_3MF}vertex")]
            ).reshape(-1, 3)
            triangles = np.array(
                [[int(t.get(a, "0")) for a in ("v1", "v2", "v3")]
                 for t in found.iter(f"{CORE_3MF}triangle")],
                np.int64,
            ).reshape(-1, 3)  # fmt: skip
            positions.append(vertices @ matrix[:3, :3] + matrix[3, :3])
            faces.append(triangles + count)
            count += len(vertices)
        for component in obj.iter(f"{CORE_3MF}component"):
            add(component.get("objectid"), _3mf_transform(component) @ matrix, depth + 1)

    try:
        build = root.find(f"{CORE_3MF}build")
        items = build.iter(f"{CORE_3MF}item") if build is not None else []
        for item in items:
            add(item.get("objectid"), _3mf_transform(item), 0)
    except ValueError as exc:
        raise MeshFormatError(f"{path}: a number can't be read ({exc})") from exc
    if not faces:
        raise MeshFormatError(f"{path}: no triangles")
    return mesh(np.vstack(positions), np.vstack(faces)).check(path)


def _3mf_model_path(package: zipfile.ZipFile) -> str:
    try:
        rels = ElementTree.fromstring(package.read("_rels/.rels"))
        for rel in rels.iter(f"{RELS}Relationship"):
            if rel.get("Type", "").endswith("/3dmodel"):
                return rel.get("Target", "").lstrip("/")
    except KeyError:
        pass
    return "3D/3dmodel.model"


def _3mf_transform(element: ElementTree.Element) -> NDArray[np.float64]:
    """A 3MF transform (12 numbers, row vectors: p' = p · M + t) as a 4x4 matrix."""
    matrix = np.eye(4)
    values = element.get("transform")
    if values:
        numbers = [float(v) for v in values.split()]
        if len(numbers) != 12:
            raise ValueError(f"transform {values!r}")
        matrix[:4, :3] = np.array(numbers).reshape(4, 3)
    return matrix


def write_3mf(source: Mesh, path: Path, *, name: str = "", unit: str = "millimeter") -> list[Path]:
    """3MF of the geometry (no colour), `unit` saying what one model unit is."""
    geometry = TexturedMesh(
        array("f", source.positions.astype(np.float32).tobytes()),
        array("I", source.faces.astype(np.uint32).tobytes()),
        array("f"),
        array("I"),
        [],
    )
    return [_write_geometry_3mf(geometry, path, name=name or path.stem, unit=unit)]


def finite_bounds(source: Mesh) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The smallest and largest x, y, z of the vertices the faces use."""
    used = source.positions[np.unique(source.faces)].astype(np.float64)
    return used.min(axis=0), used.max(axis=0)


def size_text(source: Mesh) -> str:
    """The mesh's extent, e.g. '1.2 x 0.8 x 0.5'."""
    low, high = finite_bounds(source)
    return " x ".join(f"{v:.3g}" if math.isfinite(v) else "?" for v in (high - low).tolist())
