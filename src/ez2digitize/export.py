# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export the textured mesh of a project to user-facing files.

The texture stage writes OpenMVS's textured PLY; exporting converts it into
`exports/<timestamp>/` as OBJ (+ MTL + textures) and/or GLB, or copies the
PLY, and records where it came from in `export.json` there. Exporting is
not a pipeline stage: it runs in-process (mesh export is allowed there by
architecture rule 1) and never changes the stage folders.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Literal

from ez2digitize.core.files import FormatError, read_json_object, utc_now, write_json_atomic
from ez2digitize.core.meshio import MeshFormatError, read_openmvs_ply, write_glb, write_obj
from ez2digitize.core.project import Project
from ez2digitize.core.stage import load_manifest

ExportFormat = Literal["obj", "glb", "ply"]
FORMATS: tuple[ExportFormat, ...] = ("obj", "glb", "ply")
TEXTURED_PLY = "scene_textured.ply"


class ExportError(Exception):
    pass


def export_mesh(
    project: Project,
    formats: Iterable[ExportFormat] = ("obj", "glb"),
    *,
    stage: str = "texture",
    now: datetime | None = None,
) -> list[Path]:
    """Export the latest textured mesh; returns the files written.

    If the same mesh (same texture run) was already exported in the same
    formats, nothing is written and that export's files are returned.
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

    previous = _find_export(project.exports_dir, manifest.run_id, wanted)
    if previous is not None:
        return previous

    folder = _new_folder(project.exports_dir, now or datetime.now().astimezone())
    name = _file_stem(project.name)
    files: list[Path] = []
    try:
        if "obj" in wanted:
            files += write_obj(mesh, folder / "obj", name)
        if "glb" in wanted:
            files.append(write_glb(mesh, folder / f"{name}.glb"))
        if "ply" in wanted:
            files += _copy_ply(source, mesh.textures, folder / "ply", name)
    except OSError as exc:
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
            "files": [str(p.relative_to(folder)) for p in files],
        },
    )
    return files


def _find_export(exports: Path, run_id: str, formats: list[ExportFormat]) -> list[Path] | None:
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
