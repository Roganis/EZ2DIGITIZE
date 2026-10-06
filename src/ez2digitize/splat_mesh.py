# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""A second mesh path: a surface from the trained splats, on the CPU.

Splats trained on a surface flatten onto it: a Gaussian's shortest axis is
close to the surface normal. So every solid splat becomes an oriented
point: its centre, the normal along its shortest axis (turned towards the
nearest camera, which sees the outside), and its colour (the constant
spherical-harmonic term). COLMAP's Poisson mesher (screened Poisson
surface reconstruction by Kazhdan and Hoppe, PoissonRecon, MIT, built
into COLMAP) makes a watertight surface through them and trims it where
the points are sparse.

This is the permissive, CPU-only take on meshing splats. The 2DGS-style
methods that train surface-aligned splats (2DGS, Gaussian Opacity Fields)
derive from Inria's non-commercial code and stay plugin material. The
result has vertex colours, not a texture: the OpenMVS path gives sharper
colour; this one follows what the splats learnt, which can be better on
thin or shiny parts.

    splat-mesh/  points.ply  (the oriented points)
                 mesh.ply    (Poisson's mesh, vertex colours)
"""

from __future__ import annotations

import json
import shutil
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ez2digitize import crop, upright
from ez2digitize.backends import brush, colmap
from ez2digitize.backends.colmap_model import read_images
from ez2digitize.backends.common import result_parameters
from ez2digitize.core.files import utc_now, write_json_atomic
from ez2digitize.core.meshio import MeshFormatError, _read_header
from ez2digitize.core.project import Project
from ez2digitize.core.splats import Splats, read_ply
from ez2digitize.core.stage import StageManifest, StageSpec, load_manifest, stage_input
from ez2digitize.orientation import quaternion_matrix
from ez2digitize.scale import current as current_scale

STAGE = "splat-mesh"
POINTS = "points.ply"
MESH = "mesh.ply"
SH_C0 = 0.28209479177387814  # the constant spherical harmonic
UNIT_MM = 1000.0  # GLB in metres once the scale is set, like the other exports
MIN_POINTS = 100  # fewer solid splats than this make no surface

Float = NDArray[np.float64]


@dataclass(frozen=True)
class SplatMeshOptions:
    # Octree depth: the finest detail is the model's size / 2^depth.
    depth: int = 10
    # Trim the surface where the points are sparser than this (Poisson's
    # density, log2-ish); higher trims more. COLMAP's default (10) removes
    # most of a splat surface, which is sparser than dense MVS points; on a
    # test sphere of 20 000 splats 7 removed all of it, 5 only the fringes.
    trim: float = 5.0
    # Splats less opaque than this are left out (floaters, haze).
    min_opacity: float = 0.5
    threads: int | None = None


class SplatMeshError(Exception):
    pass


# --- splats to oriented points ---------------------------------------------------


def oriented_points(
    splats: Splats,
    cameras: Float,
    *,
    min_opacity: float = 0.5,
    box: crop.CropBox | None = None,
) -> tuple[Float, Float, NDArray[np.uint8]]:
    """Positions, unit normals and RGB of the splats solid enough to be surface.

    Normals point along each splat's shortest axis, towards the nearest of
    `cameras` (their centres). With `box`, only splats inside it count.
    """
    keep = 1.0 / (1.0 + np.exp(-splats.opacities)) >= min_opacity
    # Huge splats are background fill, not surface: drop the largest 1%.
    size = np.exp(splats.scales).max(axis=1)
    if keep.any():
        keep &= size <= np.quantile(size[keep], 0.99)
    if box is not None:
        rotation = np.array(box.rotation)
        local = (splats.positions - np.array(box.centre)) @ rotation.T
        keep &= np.all(np.abs(local) <= np.array(box.half_size), axis=1)
    positions = splats.positions[keep]
    q = splats.rotations[keep]
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    # Rotation matrices (local to world), one per splat; column k is axis k.
    r = np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    )
    shortest = np.argmin(splats.scales[keep], axis=1)
    normals = r[np.arange(len(r)), :, shortest]
    if len(cameras) and len(positions):
        nearest = cameras[
            np.argmin(((positions[:, None, :] - cameras[None, :, :]) ** 2).sum(-1), axis=1)
        ]
        flip = np.einsum("ij,ij->i", normals, nearest - positions) < 0
        normals[flip] *= -1
    rgb = np.clip((0.5 + SH_C0 * splats.sh_dc[keep]) * 255 + 0.5, 0, 255).astype(np.uint8)
    return positions, normals, rgb


def write_points(path: Path, positions: Float, normals: Float, rgb: NDArray[np.uint8]) -> Path:
    """The oriented points as the PLY COLMAP's Poisson mesher reads (like fused.ply)."""
    dtype = np.dtype(
        [(n, "<f4") for n in ("x", "y", "z", "nx", "ny", "nz")]
        + [(n, "u1") for n in ("red", "green", "blue")]
    )
    table = np.empty(len(positions), dtype)
    for i, n in enumerate("xyz"):
        table[n] = positions[:, i]
        table["n" + n] = normals[:, i]
    for i, n in enumerate(("red", "green", "blue")):
        table[n] = rgb[:, i]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(table)}\n"
        + "".join(f"property float {n}\n" for n in ("x", "y", "z", "nx", "ny", "nz"))
        + "".join(f"property uchar {n}\n" for n in ("red", "green", "blue"))
        + "end_header\n"
    )
    path.write_bytes(header.encode() + table.tobytes())
    return path


def camera_centres(model_dir: Path) -> Float:
    """The photos' centres (world coordinates) in a COLMAP model."""
    centres = []
    for pose in read_images(model_dir).values():
        r = np.array(quaternion_matrix(pose.qvec))
        centres.append(-r.T @ np.array(pose.tvec))
    return np.array(centres).reshape(-1, 3)


# --- the stage ---------------------------------------------------------------------


def mesh_stage(
    sfm: colmap.Colmap,
    project: Project,
    splat: StageManifest,
    undistorted: StageManifest,
    *,
    options: SplatMeshOptions | None = None,
    stage: str = STAGE,
) -> StageSpec:
    """COLMAP's `poisson_mesher` on the splats' oriented points (made in prepare)."""
    options = options or SplatMeshOptions()
    stage_dir = project.stage_dir(stage)
    source = project.stage_dir(splat.stage) / brush.SPLAT_FILE
    model = project.stage_dir(undistorted.stage) / "sparse"
    box = crop.current(project)

    def prepare(folder: Path) -> None:
        positions, normals, rgb = oriented_points(
            read_ply(source), camera_centres(model), min_opacity=options.min_opacity, box=box
        )
        if len(positions) < MIN_POINTS:
            raise SplatMeshError(
                f"only {len(positions)} splats are solid enough to make a surface from"
            )
        write_points(folder / POINTS, positions, normals, rgb)

    argv: list[str | Path] = [
        sfm.path, "poisson_mesher",
        "--input_path", stage_dir / POINTS,
        "--output_path", stage_dir / MESH,
        "--PoissonMeshing.depth", str(options.depth),
        "--PoissonMeshing.trim", str(options.trim),
        "--PoissonMeshing.color", "1",
    ]  # fmt: skip
    if options.threads:
        argv += ["--PoissonMeshing.num_threads", str(options.threads)]
    return StageSpec(
        name=stage,
        backend=sfm.backend,
        argv=argv,
        parameters={**result_parameters(options), "crop": box.to_dict() if box else None},
        inputs={"splat": stage_input(splat), "undistorted": stage_input(undistorted)},
        prepare=prepare,
    )


# --- the mesh ----------------------------------------------------------------------


@dataclass
class ColoredMesh:
    positions: NDArray[np.float32]  # (n, 3)
    faces: NDArray[np.uint32]  # (m, 3)
    colors: NDArray[np.uint8]  # (n, 3)


def read_mesh(path: Path) -> ColoredMesh:
    """A binary PLY triangle mesh with vertex colours (as PoissonRecon writes it)."""
    with path.open("rb") as fh:
        elements, _comments = _read_header(fh, path)
        data = fh.read()
    offset = 0
    positions = faces = colors = None
    for element in elements:
        if element.name == "vertex":
            if any(p.count_type for p in element.properties):
                raise MeshFormatError(f"{path}: list properties in vertices")
            dtype = np.dtype([(p.name, "<" + p.type) for p in element.properties])
            table = np.frombuffer(data, dtype, element.count, offset)
            offset += dtype.itemsize * element.count
            positions = np.stack([table[n] for n in "xyz"], -1).astype(np.float32)
            names = dtype.names or ()
            if all(n in names for n in ("red", "green", "blue")):
                colors = np.stack([table[n] for n in ("red", "green", "blue")], -1)
                colors = colors.astype(np.uint8)
        elif element.name == "face":
            (prop,) = element.properties
            if prop.count_type is None:
                raise MeshFormatError(f"{path}: faces need a vertex index list")
            dtype = np.dtype([("n", "<" + prop.count_type), ("i", "<" + prop.type, 3)])
            table = np.frombuffer(data, dtype, element.count, offset)
            if element.count and not np.all(table["n"] == 3):
                raise MeshFormatError(f"{path}: only triangles are read")
            offset += dtype.itemsize * element.count
            faces = table["i"].astype(np.uint32)
        else:
            raise MeshFormatError(f"{path}: unexpected element {element.name!r}")
    if positions is None or faces is None or not len(faces):
        raise MeshFormatError(f"{path}: no triangles")
    if colors is None:
        colors = np.full((len(positions), 3), 180, np.uint8)
    return ColoredMesh(positions, faces, colors)


def write_glb(mesh: ColoredMesh, path: Path) -> Path:
    """A binary glTF 2.0 file with vertex colours (unlit, as the textured export)."""
    positions = np.ascontiguousarray(mesh.positions, "<f4")
    colors = np.ascontiguousarray(mesh.colors, "u1")
    rgba = np.concatenate([colors, np.full((len(colors), 1), 255, np.uint8)], axis=1)
    indices = np.ascontiguousarray(mesh.faces, "<u4")
    parts = [positions.tobytes(), rgba.tobytes(), indices.tobytes()]
    views, blob = [], b""
    for part, target in zip(parts, (34962, 34962, 34963), strict=True):
        views.append(
            {"buffer": 0, "byteOffset": len(blob), "byteLength": len(part), "target": target}
        )
        blob += part + b"\0" * (-len(part) % 4)
    gltf = {
        "asset": {"version": "2.0", "generator": "EZ2DIGITIZE"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": path.stem}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2,
                                    "material": 0}]}],
        "materials": [
            {
                "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0},
                "extensions": {"KHR_materials_unlit": {}},
            }
        ],
        "accessors": [
            {
                "bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
                "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist(),
            },
            {"bufferView": 1, "componentType": 5121, "normalized": True,
             "count": len(rgba), "type": "VEC4"},
            {"bufferView": 2, "componentType": 5125, "count": indices.size, "type": "SCALAR"},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }  # fmt: skip
    header = json.dumps(gltf, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(header) + 8 + len(blob)))
        fh.write(struct.pack("<I4s", len(header), b"JSON") + header)
        fh.write(struct.pack("<I4s", len(blob), b"BIN\0") + blob)
    return path


def placed(mesh: ColoredMesh, project: Project) -> tuple[ColoredMesh, dict[str, object]]:
    """Stood upright, centred, on the ground and in metres, like the other exports."""
    rotation = upright.rotation(project)
    scale = current_scale(project)
    info: dict[str, object] = {
        "upright": [list(row) for row in rotation] if rotation is not None else None,
        "scale_mm_per_unit": scale.mm_per_unit if scale is not None else None,
    }
    p = mesh.positions.astype(np.float64)
    if rotation is not None:
        p = p @ np.array(rotation).T
        lo, hi = p.min(axis=0), p.max(axis=0)
        p -= np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    if scale is not None:
        p *= scale.mm_per_unit / UNIT_MM
        info["units"] = {"glb": "m"}
    return ColoredMesh(p.astype(np.float32), mesh.faces, mesh.colors), info


def export(project: Project, *, stage: str = STAGE, now: datetime | None = None) -> list[Path]:
    """Export the mesh from the splats: `<name>_splat_mesh.glb` (placed) and the PLY as made."""
    from ez2digitize.export import ExportError, _file_stem, _new_folder

    manifest = load_manifest(project.stage_dir(stage))
    source = project.stage_dir(stage) / MESH
    if manifest is None or not manifest.succeeded or not source.is_file():
        raise ExportError("there is no mesh from the splats yet")
    try:
        mesh, info = placed(read_mesh(source), project)
    except (MeshFormatError, OSError) as exc:
        raise ExportError(f"cannot read the mesh from the splats: {exc}") from exc
    folder = _new_folder(project.exports_dir, now or datetime.now().astimezone())
    name = _file_stem(project.name)
    files = [folder / f"{name}_splat_mesh.glb", folder / f"{name}_splat_mesh.ply"]
    try:
        write_glb(mesh, files[0])
        shutil.copyfile(source, files[1])
    except OSError as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise ExportError(f"export failed: {exc}") from exc
    write_json_atomic(
        folder / "export.json",
        {
            "schema_version": 1,
            "created": utc_now(),
            "formats": ["glb", "ply"],
            "source": {"stage": stage, "run_id": manifest.run_id},
            "align": True,
            **info,
            "vertices": len(mesh.positions),
            "faces": len(mesh.faces),
            "files": [f.name for f in files],
        },
    )
    return files
