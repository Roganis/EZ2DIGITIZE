# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Model files from anywhere, not only this app's projects: what a file
holds, which way is up in it, and converting it to another format.

A file holds a mesh, Gaussian splats or a point cloud (`identify`; a PLY
can be any of the three, its header says which):

- meshes are read from PLY, OBJ, STL, OFF, glTF/GLB and 3MF, and written
  to OBJ, GLB, glTF, PLY, STL, 3MF, OFF and USDZ (core.meshfiles,
  core.gltf, core.usdz);
- splats are read from PLY, SPZ and .splat, and written to PLY and SPZ
  (core.splats); .ksplat and SOG files are only shown (the viewer reads
  them itself);
- point clouds are only shown.

Formats disagree on which way is up: glTF, OBJ, USDZ and most others are Y
up, STL and 3MF (3D printing) Z up, so a conversion between the two turns
the model. Splat PLY and .splat files keep the camera frame they were
trained in (COLMAP's: Y down), SPZ is Y up, so converting splats between
them turns them half a turn. Units are left as they are (`scale` changes
them: an STL in millimetres is 0.001 to a GLB in metres).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ez2digitize.core import gltf, meshfiles, splats, usdz
from ez2digitize.core.meshio import MeshFormatError
from ez2digitize.orientation import Matrix

Kind = Literal["mesh", "splat", "points"]
Up = Literal["y", "-y", "z", "-z"]

MESH_READ = (".ply", ".obj", ".stl", ".off", ".glb", ".gltf", ".3mf")
MESH_WRITE = (".glb", ".gltf", ".obj", ".ply", ".stl", ".3mf", ".off", ".usdz")
SPLAT_READ = (".ply", ".spz", ".splat")
SPLAT_SHOW = (*SPLAT_READ, ".ksplat", ".sog")
SPLAT_WRITE = (".ply", ".spz")
OPENABLE = tuple(dict.fromkeys((*MESH_READ, *SPLAT_SHOW)))
Z_UP = frozenset({".stl", ".3mf"})
# Splats in their training camera's frame (Y down).
CAMERA_FRAME = frozenset({".ply", ".splat", ".ksplat", ".sog"})
DESCRIPTIONS = {
    ".glb": "GLB (glTF, one file)",
    ".gltf": "glTF (with .bin and images)",
    ".obj": "OBJ (with MTL and images)",
    ".ply": "PLY",
    ".stl": "STL (3D printing, no colour)",
    ".3mf": "3MF (3D printing, no colour)",
    ".off": "OFF (vertex colours)",
    ".usdz": "USDZ (Apple AR Quick Look)",
    ".spz": "SPZ (compressed splats)",
    ".splat": ".splat",
    ".ksplat": ".ksplat",
    ".sog": "SOG",
}
# Turns that bring each up axis to Y (rows: the model's x, y, z to the view's).
UP_MATRICES: dict[Up, Matrix] = {
    "y": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "-y": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
    "z": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    "-z": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
}


class ModelError(Exception):
    """A model file can't be read, written or converted."""


@dataclass(frozen=True)
class ModelFile:
    path: Path
    kind: Kind

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()

    @property
    def up(self) -> Up:
        """Which axis is up in the file, by its format's convention."""
        if self.kind == "mesh":
            return "z" if self.suffix in Z_UP else "y"
        if self.kind == "splat":
            return "-y" if self.suffix in CAMERA_FRAME else "y"
        return "y"

    @property
    def outputs(self) -> tuple[str, ...]:
        """The formats (suffixes) this file can be converted to."""
        if self.kind == "mesh":
            return MESH_WRITE
        if self.kind == "splat" and self.suffix in SPLAT_READ:
            return SPLAT_WRITE
        return ()


def identify(path: Path) -> ModelFile:
    """What `path` holds, from its suffix (and for PLY, its header)."""
    suffix = path.suffix.lower()
    if not path.is_file():
        raise ModelError(f"{path}: no such file")
    if suffix == ".ply":
        try:
            header = meshfiles.ply_header(path)
        except (MeshFormatError, OSError) as exc:
            raise ModelError(str(exc)) from exc
        counts = {e.name: e.count for e in header.elements}
        vertex = next((e for e in header.elements if e.name == "vertex"), None)
        names = {p.name for p in vertex.properties} if vertex is not None else set()
        if counts.get("face", 0) > 0:
            return ModelFile(path, "mesh")
        if "f_dc_0" in names or {"opacity", "scale_0", "rot_0"} <= names:
            return ModelFile(path, "splat")
        return ModelFile(path, "points")
    if suffix in MESH_READ:
        return ModelFile(path, "mesh")
    if suffix in SPLAT_SHOW:
        return ModelFile(path, "splat")
    known = ", ".join(s.lstrip(".").upper() for s in OPENABLE)
    raise ModelError(f"{path.name}: not a model file this app reads ({known})")


def read_mesh(path: Path) -> meshfiles.Mesh:
    """The mesh in a mesh file, in the file's own frame and units."""
    readers = {
        ".ply": meshfiles.read_ply,
        ".obj": meshfiles.read_obj,
        ".stl": meshfiles.read_stl,
        ".off": meshfiles.read_off,
        ".glb": gltf.read_gltf,
        ".gltf": gltf.read_gltf,
        ".3mf": meshfiles.read_3mf,
    }
    reader = readers.get(path.suffix.lower())
    if reader is None:
        raise ModelError(f"{path.name}: meshes are read from {', '.join(MESH_READ)}")
    try:
        return reader(path)
    except (MeshFormatError, OSError, MemoryError) as exc:
        raise ModelError(str(exc)) from exc


def write_mesh(mesh: meshfiles.Mesh, path: Path) -> list[Path]:
    """Write `mesh` in the format `path`'s suffix names; returns the files written."""
    suffix = path.suffix.lower()
    writers = {
        ".glb": gltf.write_glb,
        ".gltf": gltf.write_gltf,
        ".obj": meshfiles.write_obj,
        ".ply": meshfiles.write_ply,
        ".stl": meshfiles.write_stl,
        ".3mf": meshfiles.write_3mf,
        ".off": meshfiles.write_off,
        ".usdz": usdz.write_usdz,
    }
    writer = writers.get(suffix)
    if writer is None:
        raise ModelError(f"{path.name}: meshes are written as {', '.join(MESH_WRITE)}")
    try:
        return writer(mesh, path)
    except (MeshFormatError, OSError) as exc:
        raise ModelError(f"cannot write {path}: {exc}") from exc


def read_splats(path: Path) -> splats.Splats:
    """The splats in a PLY, SPZ or .splat file, in the file's own frame."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".ply":
            return splats.read_ply(path)
        if suffix == ".spz":
            return splats.read_spz(path)
        if suffix == ".splat":
            return splats.read_splat(path)
    except (MeshFormatError, OSError, ValueError) as exc:
        raise ModelError(str(exc)) from exc
    raise ModelError(
        f"{path.name}: {suffix} splats can be shown but not converted "
        f"(convert from {', '.join(SPLAT_READ)})"
    )


def convert(source: Path, target: Path, *, scale: float = 1.0) -> list[Path]:
    """Convert `source` into the format of `target`'s suffix; returns the files written.

    The model is turned when the formats disagree on which way is up (see
    the module's description) and multiplied by `scale`.
    """
    if not scale > 0:
        raise ModelError("the scale must be more than 0")
    if source.resolve() == target.resolve():
        raise ModelError("the converted file would replace the original: choose another name")
    model = identify(source)
    suffix = target.suffix.lower()
    if model.kind == "points":
        raise ModelError(f"{source.name} is a point cloud: it can be shown, not converted")
    if suffix not in model.outputs:
        if model.kind == "splat" and suffix in MESH_WRITE:
            raise ModelError("splats can't become a mesh here: convert them to PLY or SPZ")
        if model.kind == "mesh" and suffix in SPLAT_WRITE:
            raise ModelError("a mesh can't become splats: choose a mesh format")
        options = ", ".join(model.outputs) or "nothing"
        raise ModelError(f"{source.name} can be converted to {options}")
    target_up: Up = "z" if suffix in Z_UP else "y"
    if model.kind == "mesh":
        mesh = read_mesh(source)
        turn = _between(model.up, target_up)
        if turn is not None or scale != 1.0:
            mesh = meshfiles.transformed(mesh, turn if turn is not None else np.eye(3), scale)
        return write_mesh(mesh, target)
    found = read_splats(source)
    from_camera = model.suffix in CAMERA_FRAME
    to_camera = suffix in CAMERA_FRAME
    if from_camera != to_camera:
        found = splats.turned(found, splats.PLY_TO_SPZ)
    if scale != 1.0:
        found = splats.Splats(
            found.positions * scale,
            found.scales + np.log(scale),
            found.rotations,
            found.opacities,
            found.sh_dc,
            found.sh_rest,
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".spz":
            return [splats.write_spz(found, target)]
        return [splats.write_ply(found, target)]
    except OSError as exc:
        raise ModelError(f"cannot write {target}: {exc}") from exc


def _between(source: Up, target: Up) -> NDArray[np.float64] | None:
    """The turn from a model with `source` up to one with `target` up (None: none needed)."""
    if source == target:
        return None
    to_view = np.array(UP_MATRICES[source])
    from_view = np.array(UP_MATRICES[target]).T
    return np.asarray(from_view @ to_view, np.float64)


def summary(model: ModelFile) -> str:
    """One line on what the file holds, e.g. '12,345 triangles, textured'."""
    if model.kind == "mesh":
        mesh = read_mesh(model.path)
        parts = [f"{mesh.face_count:,} triangles", f"{mesh.vertex_count:,} vertices"]
        if mesh.textured:
            parts.append("textured")
        elif mesh.colors is not None:
            parts.append("vertex colours")
        return ", ".join(parts) + f"; size {meshfiles.size_text(mesh)}"
    if model.kind == "splat" and model.suffix in SPLAT_READ:
        found = read_splats(model.path)
        return f"{found.count:,} splats, colour degree {found.sh_degree}"
    return {"splat": "splats", "points": "a point cloud", "mesh": "a mesh"}[model.kind]
