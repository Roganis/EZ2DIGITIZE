# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export the textured mesh of a project to user-facing files.

The texture stage writes OpenMVS's textured PLY; exporting converts it into
`exports/<timestamp>/` and records where it came from in `export.json`
there:

- `obj` (+ MTL + textures) and `glb`, textured;
- `ply`, OpenMVS's textured PLY as is;
- `stl` and `3mf` for 3D printing: geometry only, after checking that the
  surface is closed (`export.json` records the open and non-manifold
  edges; `export_notes` says when it isn't printable as is);
- `points`: the dense point cloud (PLY with colours and normals).

Units: with the scale set (ez2digitize.scale), STL and 3MF are in
millimetres, as slicers expect, and OBJ, GLB and the point cloud in metres
(glTF's unit). Without it, everything is in the reconstruction's own
arbitrary units. The `ply` copy is always OpenMVS's file as it is.

Exporting is not a pipeline stage: it runs in-process (mesh export is
allowed there by architecture rule 1) and never changes the stage folders.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Literal

from ez2digitize.backends.common import BackendError
from ez2digitize.core.capture import list_bundles
from ez2digitize.core.files import FormatError, read_json_object, utc_now, write_json_atomic
from ez2digitize.core.meshio import (
    MeshFormatError,
    TexturedMesh,
    check_watertight,
    read_openmvs_ply,
    read_point_cloud,
    write_3mf,
    write_glb,
    write_obj,
    write_point_cloud,
    write_stl,
)
from ez2digitize.core.photos import exif_orientations
from ez2digitize.core.project import Project
from ez2digitize.core.stage import load_manifest
from ez2digitize.orientation import IDENTITY, Placement, estimate_up, place
from ez2digitize.scale import current as current_scale
from ez2digitize.sides import upright_names

ExportFormat = Literal["obj", "glb", "ply", "stl", "3mf", "points", "splat"]
FORMATS: tuple[ExportFormat, ...] = ("obj", "glb", "ply", "stl", "3mf", "points")
PRINT_FORMATS = frozenset({"stl", "3mf"})
# Millimetres per export unit: print formats in mm, the others in metres.
PRINT_UNIT_MM = 1.0
UNIT_MM = 1000.0
TEXTURED_PLY = "scene_textured.ply"
DENSE_PLY = "scene_dense.ply"
SPLAT_PLY = "splat.ply"


class ExportError(Exception):
    pass


def export_mesh(
    project: Project,
    formats: Iterable[ExportFormat] = ("obj", "glb"),
    *,
    stage: str = "texture",
    align: bool = True,
    now: datetime | None = None,
) -> list[Path]:
    """Export the latest textured mesh; returns the files written.

    With `align`, the mesh and point cloud are stood upright, centred and
    put on the ground (see `orientation`): Y up for OBJ and GLB, Z up for
    STL and 3MF. The `ply` copy stays as OpenMVS wrote it.

    If the same mesh (same texture run) was already exported in the same
    formats and alignment, nothing is written and that export's files are
    returned.
    """
    wanted = list(dict.fromkeys(formats))
    unknown = [f for f in wanted if f not in FORMATS]
    if not wanted or unknown:
        raise ExportError(f"choose formats from {', '.join(FORMATS)}")
    manifest = load_manifest(project.stage_dir(stage))
    source = project.stage_dir(stage) / TEXTURED_PLY
    if manifest is None or not manifest.succeeded or not source.is_file():
        raise ExportError("there is no textured mesh to export yet; build the mesh first")
    try:
        mesh = read_openmvs_ply(source)
    except (MeshFormatError, OSError) as exc:
        raise ExportError(f"cannot read the textured mesh: {exc}") from exc

    scale = current_scale(project)
    mm_per_unit = scale.mm_per_unit if scale is not None else None
    previous = _find_export(project.exports_dir, manifest.run_id, wanted, align, mm_per_unit)
    if previous is not None:
        return previous

    dense = project.stage_dir("densify") / DENSE_PLY
    if "points" in wanted and not dense.is_file():
        raise ExportError("there is no dense point cloud to export")

    folder = _new_folder(project.exports_dir, now or datetime.now().astimezone())
    name = _file_stem(project.name)
    files: list[Path] = []
    info: dict[str, object] = {"align": align, "scale_mm_per_unit": mm_per_unit}
    if mm_per_unit is not None:
        info["units"] = {"stl": "mm", "3mf": "mm", "obj": "m", "glb": "m", "points": "m"}
    placement = _placement(project, mesh) if align else None
    if placement is not None:
        info["placement"] = placement.to_dict()
    elif align:
        info["placement"] = None  # the photos' orientations disagree: model frame kept
    upright = _placed(mesh, _in_units(placement, mm_per_unit, UNIT_MM))
    if PRINT_FORMATS & set(wanted):
        closed = check_watertight(mesh)
        info["watertight"] = closed.watertight
        info["open_edges"] = closed.open_edges
        info["non_manifold_edges"] = closed.non_manifold_edges
    try:
        if "obj" in wanted:
            files += write_obj(upright, folder / "obj", name)
        if "glb" in wanted:
            files.append(write_glb(upright, folder / f"{name}.glb"))
        if "ply" in wanted:
            files += _copy_ply(source, mesh.textures, folder / "ply", name)
        if PRINT_FORMATS & set(wanted):
            in_mm = _in_units(placement, mm_per_unit, PRINT_UNIT_MM)
            # Z-up only once stood upright: an unaligned mesh keeps its frame.
            printable = _placed(mesh, in_mm, z_up=placement is not None)
            if "stl" in wanted:
                files.append(write_stl(printable, folder / f"{name}.stl"))
            if "3mf" in wanted:
                files.append(write_3mf(printable, folder / f"{name}.3mf", name=project.name))
        if "points" in wanted:
            in_m = _in_units(placement, mm_per_unit, UNIT_MM)
            files.append(_export_points(dense, folder / f"{name}_points.ply", in_m))
    except (OSError, MeshFormatError) as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise ExportError(f"export failed: {exc}") from exc
    write_json_atomic(
        folder / "export.json",
        {
            "schema_version": 1,
            "created": utc_now(),
            "formats": wanted,
            "source": {"stage": stage, "run_id": manifest.run_id},
            "vertices": mesh.vertex_count,
            "faces": mesh.face_count,
            **info,
            "files": [str(p.relative_to(folder)) for p in files],
        },
    )
    return files


def export_splat(
    project: Project, *, stage: str = "splat", now: datetime | None = None
) -> list[Path]:
    """Copy the trained splats to `exports/<timestamp>/<name>_splat.ply`.

    Splats keep the reconstruction's frame: standing them upright would mean
    rotating every splat's orientation and its spherical harmonics too.
    """
    manifest = load_manifest(project.stage_dir(stage))
    source = project.stage_dir(stage) / SPLAT_PLY
    if manifest is None or not manifest.succeeded or not source.is_file():
        raise ExportError("there are no splats to export yet; train them first")
    formats: list[ExportFormat] = ["splat"]
    previous = _find_export(project.exports_dir, manifest.run_id, formats, False, None)
    if previous is not None:
        return previous
    folder = _new_folder(project.exports_dir, now or datetime.now().astimezone())
    target = folder / f"{_file_stem(project.name)}_splat.ply"
    try:
        shutil.copyfile(source, target)
    except OSError as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise ExportError(f"export failed: {exc}") from exc
    write_json_atomic(
        folder / "export.json",
        {
            "schema_version": 1,
            "created": utc_now(),
            "formats": formats,
            "source": {"stage": stage, "run_id": manifest.run_id},
            "align": False,
            "files": [target.name],
        },
    )
    return [target]


def export_notes(files: list[Path]) -> list[str]:
    """What the user should know about an export (e.g. a mesh not closed for printing)."""
    if not files:
        return []
    folder = files[0].parent
    while folder.name in ("obj", "ply"):
        folder = folder.parent
    try:
        info = read_json_object(folder / "export.json")
    except FormatError:
        return []
    notes = []
    if info.get("watertight") is False:
        notes.append(
            f"the mesh is not closed ({info.get('open_edges')} open and "
            f"{info.get('non_manifold_edges')} non-manifold edges): a slicer may need to "
            "repair it before printing"
        )
    formats = info.get("formats")
    printing = isinstance(formats, list) and bool(PRINT_FORMATS & set(formats))
    if printing and info.get("scale_mm_per_unit") is None:
        notes.append(
            "no scale is set, so the size is arbitrary: set it in the 3D view (two points "
            "and the real distance between them), or scale the model in the slicer"
        )
    return notes


def _placement(project: Project, mesh: TexturedMesh) -> Placement | None:
    """Upright placement from the undistorted model's cameras; None if unsure."""
    model = project.stage_dir("undistort") / "sparse"
    if not (model / "images.bin").is_file():
        return None
    try:
        bundles = list_bundles(project)
        estimate = estimate_up(model, exif_orientations(bundles), upright_names(bundles))
    except (OSError, ValueError, FormatError, BackendError):
        return None
    return place(mesh.positions, estimate.up) if estimate is not None else None


def _in_units(
    placement: Placement | None, mm_per_unit: float | None, unit_mm: float
) -> Placement | None:
    """The placement, scaling reconstruction units to export units if the scale is set."""
    if mm_per_unit is None:
        return placement
    base = placement or Placement(IDENTITY, (0.0, 0.0, 0.0))
    return replace(base, scale=mm_per_unit / unit_mm)


def _placed(mesh: TexturedMesh, placement: Placement | None, *, z_up: bool = False) -> TexturedMesh:
    if placement is None:
        return mesh
    return replace(mesh, positions=placement.apply(mesh.positions, z_up=z_up))


def _export_points(dense: Path, target: Path, placement: Placement | None) -> Path:
    if placement is None:
        shutil.copyfile(dense, target)
        return target
    cloud = read_point_cloud(dense)
    cloud.positions = placement.apply(cloud.positions)
    if cloud.normals is not None:
        turn = Placement(placement.rotation, (0.0, 0.0, 0.0))
        cloud.normals = turn.apply(cloud.normals)
    return write_point_cloud(cloud, target)


def _find_export(
    exports: Path,
    run_id: str,
    formats: list[ExportFormat],
    align: bool,
    mm_per_unit: float | None,
) -> list[Path] | None:
    if not exports.is_dir():
        return None
    for folder in sorted(exports.iterdir(), reverse=True):
        try:
            info = read_json_object(folder / "export.json")
        except FormatError:
            continue
        source = info.get("source")
        if (
            isinstance(source, dict)
            and source.get("run_id") == run_id
            and info.get("formats") == formats
            and info.get("align", False) == align
            and info.get("scale_mm_per_unit") == mm_per_unit
        ):
            files = [folder / str(name) for name in info.get("files", [])]
            if files and all(f.is_file() for f in files):
                return files
    return None


def _copy_ply(source: Path, textures: list[Path], folder: Path, name: str) -> list[Path]:
    # The PLY names its textures in header comments, so keep their names.
    folder.mkdir(parents=True)
    files = [folder / f"{name}.ply"]
    shutil.copyfile(source, files[0])
    for texture in textures:
        shutil.copyfile(texture, folder / texture.name)
        files.append(folder / texture.name)
    return files


def _new_folder(exports: Path, now: datetime) -> Path:
    base = now.strftime("%Y%m%d-%H%M%S")
    folder, n = exports / base, 1
    while folder.exists():
        n += 1
        folder = exports / f"{base}-{n}"
    folder.mkdir(parents=True)
    return folder


def _file_stem(name: str) -> str:
    """A file name from the project name: letters, digits, '-', '_' and '.'."""
    stem = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("._")
    return stem or "mesh"
