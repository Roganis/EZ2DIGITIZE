# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""OpenMVS: import from COLMAP, densify, mesh, refine, texture. CPU only.

Stage layout:

    mvs-import/   scene.mvs                 (InterfaceCOLMAP)
    densify/      scene_dense.mvs/.ply      (DensifyPointCloud)
    mesh/         scene_mesh.mvs/.ply       (ReconstructMesh)
    refine/       scene_refined.mvs/.ply    (RefineMesh, optional)
    texture/      scene_textured.mvs/.<type> + texture images (TextureMesh)

Every tool runs with its own stage folder as working folder (`-w`), where it
also writes its log and temporary files. Scene files store image paths
relative to the working folder of the tool that wrote them; all stage
folders are siblings under `stages/`, so a path like
`../undistort/images/x.jpg` resolves the same from each of them.

The tools run under a pseudo-terminal: they write their console output
through C stdio, which is block-buffered on a pipe, so the progress lines
would otherwise only arrive when a tool exits.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from ez2digitize.backends.common import BackendError, BackendMissing, find_tool
from ez2digitize.core.project import Project
from ez2digitize.core.runner import ProcessStartError, Progress, run_quick
from ez2digitize.core.stage import Backend, StageManifest, StageSpec, stage_input

NAME = "openmvs"
PINNED_VERSION = "2.4.0"
ENV_VAR = "EZ2D_OPENMVS_DIR"
TOOLS = ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh")
# Where distribution and source builds install the tools (not on PATH).
SEARCH_DIRS = (
    Path("/usr/local/bin/OpenMVS"),
    Path("/usr/bin/OpenMVS"),
    Path("/opt/homebrew/bin/OpenMVS"),
)

ExportType = Literal["ply", "obj", "glb"]


@dataclass(frozen=True)
class OpenMVS:
    bin_dir: Path
    version: str

    @property
    def backend(self) -> Backend:
        return Backend(NAME, self.version)

    @property
    def supported(self) -> bool:
        return self.version == PINNED_VERSION

    def tool(self, name: str) -> Path:
        return self.bin_dir / name


def parse_version(text: str) -> str | None:
    """Version from the banner every tool prints ("OpenMVS x64 v2.4.0")."""
    match = re.search(r"OpenMVS\b[^\n]*?\bv(\d+\.\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def locate(explicit_dir: Path | None = None) -> OpenMVS:
    """Find the folder holding all OpenMVS tools and read their version."""
    if explicit_dir is None and (env_dir := os.environ.get(ENV_VAR)):
        explicit_dir = Path(env_dir).expanduser()
    densify = find_tool(
        "DensifyPointCloud",
        explicit=explicit_dir / "DensifyPointCloud" if explicit_dir else None,
        extra_dirs=SEARCH_DIRS,
    )
    if densify is None:
        raise BackendMissing(f"OpenMVS not found (looked in ${ENV_VAR}, the bundle, PATH)")
    bin_dir = densify.parent
    missing = [t for t in TOOLS if find_tool(t, explicit=bin_dir / t) is None]
    if missing:
        raise BackendMissing(f"OpenMVS in {bin_dir} is incomplete: no {', '.join(missing)}")
    try:
        text = run_quick([bin_dir / "InterfaceCOLMAP", "--help"])
    except ProcessStartError as exc:
        raise BackendMissing(f"OpenMVS in {bin_dir} can't be run: {exc}") from exc
    version = parse_version(text)
    if version is None:
        raise BackendMissing(f"no OpenMVS version in the output of {bin_dir / 'InterfaceCOLMAP'}")
    return OpenMVS(bin_dir=bin_dir, version=version)


# --- options -------------------------------------------------------------------


@dataclass(frozen=True)
class DensifyOptions:
    # Each level halves the image size. 1 is a guess until Phase 1 decides.
    resolution_level: int = 1
    max_resolution: int = 3200
    threads: int | None = None


@dataclass(frozen=True)
class MeshOptions:
    threads: int | None = None


@dataclass(frozen=True)
class RefineOptions:
    resolution_level: int = 1
    threads: int | None = None


@dataclass(frozen=True)
class TextureOptions:
    export_type: ExportType = "ply"
    resolution_level: int = 0
    threads: int | None = None


# --- stages --------------------------------------------------------------------


def import_colmap(
    mvs: OpenMVS, project: Project, undistorted: StageManifest, *, stage: str = "mvs-import"
) -> StageSpec:
    """InterfaceCOLMAP: the undistorted COLMAP model to an OpenMVS scene."""
    stage_dir = project.stage_dir(stage)
    source = project.stage_dir(undistorted.stage)
    return StageSpec(
        name=stage,
        backend=mvs.backend,
        argv=[
            mvs.tool("InterfaceCOLMAP"),
            "-i",
            source,
            "-o",
            stage_dir / "scene.mvs",
            "--image-folder",
            source / "images",
            "-w",
            stage_dir,
        ],  # fmt: skip
        inputs={"undistorted": stage_input(undistorted)},
        parse_line=OpenMVSProgress(),
        use_pty=True,
    )


def densify(
    mvs: OpenMVS,
    project: Project,
    imported: StageManifest,
    *,
    masks: StageManifest | None = None,
    stage: str = "densify",
    options: DensifyOptions | None = None,
) -> StageSpec:
    """DensifyPointCloud, optionally masked.

    `masks` is a `colmap.undistort_masks` stage: one `<stem>.mask.png` per
    undistorted image, where 0 marks background to ignore. The mask paths
    are saved into the dense scene relative to this stage's folder, which
    resolves the same from the later sibling stages.
    """
    options = options or DensifyOptions()
    stage_dir = project.stage_dir(stage)
    argv: list[str | Path] = [
        mvs.tool("DensifyPointCloud"),
        project.stage_dir(imported.stage) / "scene.mvs",
        "-o", stage_dir / "scene_dense.mvs",
        "-w", stage_dir,
        "--resolution-level", str(options.resolution_level),
        "--max-resolution", str(options.max_resolution),
    ]  # fmt: skip
    if options.threads:
        argv += ["--max-threads", str(options.threads)]
    inputs = {"scene": stage_input(imported)}
    if masks is not None:
        argv += [
            "--mask-path",
            project.stage_dir(masks.stage) / "masks",
            "--ignore-mask-label",
            "0",
        ]
        inputs["masks"] = stage_input(masks)
    return StageSpec(
        name=stage,
        backend=mvs.backend,
        argv=argv,
        parameters={**asdict(options), "masked": masks is not None},
        inputs=inputs,
        parse_line=OpenMVSProgress(),
        use_pty=True,
    )


def reconstruct_mesh(
    mvs: OpenMVS,
    project: Project,
    dense: StageManifest,
    *,
    stage: str = "mesh",
    options: MeshOptions | None = None,
) -> StageSpec:
    options = options or MeshOptions()
    stage_dir = project.stage_dir(stage)
    dense_dir = project.stage_dir(dense.stage)
    argv: list[str | Path] = [
        mvs.tool("ReconstructMesh"),
        dense_dir / "scene_dense.mvs",
        "--pointcloud-file", dense_dir / "scene_dense.ply",
        "-o", stage_dir / "scene_mesh.mvs",
        "-w", stage_dir,
    ]  # fmt: skip
    if options.threads:
        argv += ["--max-threads", str(options.threads)]
    return StageSpec(
        name=stage,
        backend=mvs.backend,
        argv=argv,
        parameters=asdict(options),
        inputs={"dense": stage_input(dense)},
        parse_line=OpenMVSProgress(),
        use_pty=True,
    )


def refine_mesh(
    mvs: OpenMVS,
    project: Project,
    dense: StageManifest,
    mesh: StageManifest,
    *,
    stage: str = "refine",
    options: RefineOptions | None = None,
) -> StageSpec:
    """RefineMesh (optional, slow): sharpens the mesh against the images."""
    options = options or RefineOptions()
    stage_dir = project.stage_dir(stage)
    argv: list[str | Path] = [
        mvs.tool("RefineMesh"),
        project.stage_dir(dense.stage) / "scene_dense.mvs",
        "--mesh-file", mesh_output(project, mesh),
        "-o", stage_dir / "scene_refined.mvs",
        "-w", stage_dir,
        "--resolution-level", str(options.resolution_level),
    ]  # fmt: skip
    if options.threads:
        argv += ["--max-threads", str(options.threads)]
    return StageSpec(
        name=stage,
        backend=mvs.backend,
        argv=argv,
        parameters=asdict(options),
        inputs={"dense": stage_input(dense), "mesh": stage_input(mesh)},
        parse_line=OpenMVSProgress(),
        use_pty=True,
    )


def texture_mesh(
    mvs: OpenMVS,
    project: Project,
    dense: StageManifest,
    mesh: StageManifest,
    *,
    stage: str = "texture",
    options: TextureOptions | None = None,
) -> StageSpec:
    """TextureMesh on the output of the mesh or refine stage."""
    options = options or TextureOptions()
    stage_dir = project.stage_dir(stage)
    argv: list[str | Path] = [
        mvs.tool("TextureMesh"),
        project.stage_dir(dense.stage) / "scene_dense.mvs",
        "--mesh-file", mesh_output(project, mesh),
        "-o", stage_dir / "scene_textured.mvs",
        "-w", stage_dir,
        "--export-type", options.export_type,
        "--resolution-level", str(options.resolution_level),
    ]  # fmt: skip
    if options.threads:
        argv += ["--max-threads", str(options.threads)]
    return StageSpec(
        name=stage,
        backend=mvs.backend,
        argv=argv,
        parameters=asdict(options),
        inputs={"dense": stage_input(dense), "mesh": stage_input(mesh)},
        parse_line=OpenMVSProgress(),
        use_pty=True,
    )


def mesh_output(project: Project, mesh: StageManifest) -> Path:
    """The mesh file written by a mesh or refine stage."""
    folder = project.stage_dir(mesh.stage)
    for name in ("scene_refined.ply", "scene_mesh.ply"):
        if (folder / name).is_file():
            return folder / name
    raise BackendError(f"stage {mesh.stage} has no mesh output")


# --- progress ------------------------------------------------------------------

_APP_PREFIX = re.compile(r"^\d\d:\d\d:\d\d \[\w+\s*\] ")
_COUNTED = re.compile(r"^([A-Z][A-Za-z -]*?) (\d+) \((\d+(?:\.\d+)?)%")
_COMPLETED = re.compile(r"^(.+?) completed\b")


class OpenMVSProgress:
    """Turns OpenMVS output into progress events.

    Long loops print `Estimated depth-maps 12 (37.50%, 1m24s, ETA 2m)...`;
    each phase restarts at 0 %, so the message names the phase. Finished
    steps (`... completed: ...`) become messages without a fraction.
    """

    def __call__(self, line: str) -> Progress | None:
        text = _APP_PREFIX.sub("", line).strip()
        if m := _COUNTED.match(text):
            return Progress(m.group(1), min(round(float(m.group(3)) / 100, 4), 1.0))
        if m := _COMPLETED.match(text):
            return Progress(f"{m.group(1)} completed")
        return None
