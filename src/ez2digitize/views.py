# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the 3D viewer can show of a project, and the files it loads.

Headless: the Qt side (ui.viewer) serves these files to the web page and
tells it which `View` to draw. A view is one result of the pipeline:

- `cameras`: the camera placement, COLMAP's sparse points and a frustum
  per photo (both generated from the mapping stage's binary model),
- `dense`: OpenMVS's dense point cloud,
- `mesh`: the textured mesh, as the GLB the export wrote for the current
  texture run, else converted from OpenMVS's PLY into a cached GLB,
- `splat`: Brush's Gaussian splats.

`available` only looks at which stages succeeded, so it is cheap; `files`
does the work (reading the sparse model, converting the mesh) when a view
is shown. Each view carries `upright`: the rotation that stands the model
up (see orientation), or None if it is already upright (the exported GLB)
or the photos don't say which way is up.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ez2digitize.backends import brush
from ez2digitize.backends.colmap_model import read_cameras, read_images
from ez2digitize.backends.common import BackendError
from ez2digitize.core.capture import list_bundles
from ez2digitize.core.files import FormatError, read_json_object
from ez2digitize.core.meshio import MeshFormatError, read_openmvs_ply, write_glb
from ez2digitize.core.photos import exif_orientations
from ez2digitize.core.project import Project
from ez2digitize.core.stage import StageManifest, load_manifest
from ez2digitize.orientation import Matrix, estimate_up, quaternion_matrix, rotation_between
from ez2digitize.sides import upright_names

ViewKey = Literal["cameras", "dense", "mesh", "splat"]
# How the page draws a view (viewer.js): GLB mesh, PLY points, splats, or
# sparse points with camera frustums.
Kind = Literal["glb", "points", "splat", "cameras"]

LABELS: dict[ViewKey, str] = {
    "cameras": "Camera placement",
    "dense": "Dense point cloud",
    "mesh": "Textured mesh",
    "splat": "Gaussian splats",
}
ORDER: tuple[ViewKey, ...] = ("mesh", "splat", "dense", "cameras")


class ViewError(Exception):
    """A view's files can't be read or made."""


@dataclass(frozen=True)
class View:
    key: ViewKey
    kind: Kind
    # The stage run the view shows: a new run means new files.
    run_id: str
    # The stage folder (and for the mesh, the GLB export if there is one).
    source: Path
    upright: Matrix | None
    finished: str = ""

    @property
    def label(self) -> str:
        return LABELS[self.key]


def available(project: Project) -> list[View]:
    """The views this project has results for, best first (see ORDER)."""
    views: dict[ViewKey, View] = {}
    up = _upright(project)

    texture = _succeeded(project, "texture")
    if texture is not None:
        glb = _exported_glb(project, texture.run_id)
        if glb is not None:
            views["mesh"] = View("mesh", "glb", texture.run_id, glb, None, texture.finished)
        else:
            stage = project.stage_dir("texture")
            views["mesh"] = View("mesh", "glb", texture.run_id, stage, up, texture.finished)
    splat = _succeeded(project, "splat")
    if splat is not None and (project.stage_dir("splat") / brush.SPLAT_FILE).is_file():
        views["splat"] = View(
            "splat", "splat", splat.run_id, project.stage_dir("splat"), up, splat.finished
        )
    dense = _succeeded(project, "densify")
    if dense is not None and _dense_ply(project).is_file():
        views["dense"] = View(
            "dense", "points", dense.run_id, project.stage_dir("densify"), up, dense.finished
        )
    undistorted = _succeeded(project, "undistort")
    model = project.stage_dir("undistort") / "sparse"
    if undistorted is not None and all(
        (model / f).is_file() for f in ("cameras.bin", "images.bin", "points3D.bin")
    ):
        views["cameras"] = View(
            "cameras",
            "cameras",
            undistorted.run_id,
            project.stage_dir("undistort") / "sparse",
            up,
            undistorted.finished,
        )
    return [views[k] for k in ORDER if k in views]


def files(view: View, cache: Path) -> dict[str, Path | bytes]:
    """The files the page loads for `view`, by name: `model` (and `cameras`).

    Files made for the viewer (the cached GLB) go into `cache`.
    """
    try:
        if view.key == "mesh":
            if view.source.suffix == ".glb":
                return {"model": view.source}
            return {"model": _cached_glb(view, cache)}
        if view.key == "splat":
            return {"model": view.source / brush.SPLAT_FILE}
        if view.key == "dense":
            return {"model": view.source / "scene_dense.ply"}
        points, cameras = sparse_scene(view.source)
        return {"model": points, "cameras": cameras}
    except (OSError, BackendError, MeshFormatError, FormatError) as exc:
        raise ViewError(f"{view.label} can't be shown: {exc}") from exc


def spec(view: View, urls: dict[str, str]) -> dict[str, object]:
    """What the page needs to draw `view` (JSON), given the URLs of its files."""
    return {
        "kind": view.kind,
        "label": view.label,
        "upright": [list(row) for row in view.upright] if view.upright else None,
        **urls,
    }


# --- the sparse model -------------------------------------------------------------


def read_points(model_dir: Path) -> tuple[list[tuple[float, float, float]], bytes]:
    """COLMAP's points3D.bin: positions and their RGB colours."""
    path = model_dir / "points3D.bin"
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise BackendError(f"cannot read {path}: {exc}") from exc
    record = struct.Struct("<Q3d3BdQ")
    try:
        (count,) = struct.unpack_from("<Q", data, 0)
        offset = 8
        positions = []
        colors = bytearray()
        for _ in range(count):
            _id, x, y, z, r, g, b, _error, track = record.unpack_from(data, offset)
            offset += record.size + 8 * track  # each track entry: image id, point index
            positions.append((x, y, z))
            colors += bytes((r, g, b))
    except struct.error as exc:
        raise BackendError(f"{path}: truncated") from exc
    return positions, bytes(colors)


def sparse_scene(model_dir: Path) -> tuple[bytes, bytes]:
    """The sparse points as a PLY and the cameras as JSON, for the page.

    Each camera: its centre, its rotation (camera to world, rows), the
    vertical field of view and aspect ratio of its image, and its name.
    """
    positions, colors = read_points(model_dir)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(positions)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode()
    body = bytearray()
    for i, (x, y, z) in enumerate(positions):
        body += struct.pack("<3f", x, y, z) + colors[3 * i : 3 * i + 3]

    cameras = read_cameras(model_dir)
    shown = []
    for name, pose in sorted(read_images(model_dir).items()):
        world_to_camera = quaternion_matrix(pose.qvec)
        to_world = tuple(tuple(world_to_camera[j][i] for j in range(3)) for i in range(3))
        t = pose.tvec
        centre = tuple(-sum(to_world[i][j] * t[j] for j in range(3)) for i in range(3))
        camera = cameras.get(pose.camera_id)
        if camera is None:
            continue
        focal_y = camera.params[1] if camera.model in ("PINHOLE", "OPENCV") else camera.params[0]
        fov = math.degrees(2 * math.atan(camera.height / (2 * focal_y))) if focal_y else 50.0
        shown.append(
            {
                "name": name,
                "centre": [round(c, 6) for c in centre],
                "rotation": [[round(v, 6) for v in row] for row in to_world],
                "fov": round(fov, 3),
                "aspect": round(camera.width / camera.height, 4),
            }
        )
    return header + bytes(body), json.dumps({"cameras": shown}).encode()


# --- helpers ----------------------------------------------------------------------


def _succeeded(project: Project, stage: str) -> StageManifest | None:
    manifest = load_manifest(project.stage_dir(stage))
    return manifest if manifest is not None and manifest.succeeded else None


def _dense_ply(project: Project) -> Path:
    return project.stage_dir("densify") / "scene_dense.ply"


def _upright(project: Project) -> Matrix | None:
    """The rotation that stands the reconstruction up (as the export does)."""
    model = project.stage_dir("undistort") / "sparse"
    if not (model / "images.bin").is_file():
        return None
    try:
        bundles = list_bundles(project)
        estimate = estimate_up(model, exif_orientations(bundles), upright_names(bundles))
    except (OSError, ValueError, BackendError, FormatError):
        return None
    if estimate is None:
        return None
    return rotation_between(estimate.up, (0.0, 1.0, 0.0))


def _exported_glb(project: Project, run_id: str) -> Path | None:
    """The newest GLB exported from this texture run, upright."""
    if not project.exports_dir.is_dir():
        return None
    for folder in sorted(project.exports_dir.iterdir(), reverse=True):
        try:
            info = read_json_object(folder / "export.json")
        except FormatError:
            continue
        source = info.get("source")
        names = info.get("files", [])
        if (
            isinstance(source, dict)
            and source.get("run_id") == run_id
            and isinstance(names, list)
            and info.get("align", False)
        ):
            for name in names:
                if str(name).endswith(".glb") and (folder / str(name)).is_file():
                    return folder / str(name)
    return None


def _cached_glb(view: View, cache: Path) -> Path:
    target = cache / f"mesh-{view.run_id}.glb"
    if not target.is_file():
        textured = sorted(view.source.glob("*.ply"))
        meshes = [p for p in textured if "textured" in p.name] or textured
        if not meshes:
            raise ViewError("the texture step left no mesh")
        cache.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        write_glb(read_openmvs_ply(meshes[0]), partial)
        partial.replace(target)
    return target
