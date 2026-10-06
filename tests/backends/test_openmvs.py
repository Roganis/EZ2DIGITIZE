# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from collections.abc import Callable
from pathlib import Path

import pytest

from ez2digitize.backends import openmvs
from ez2digitize.backends.common import BackendError, BackendMissing
from ez2digitize.backends.openmvs import (
    TOOLS,
    DensifyOptions,
    OpenMVS,
    OpenMVSProgress,
    TextureOptions,
    parse_version,
)
from ez2digitize.core.project import Project
from ez2digitize.core.runner import Progress
from ez2digitize.core.stage import Backend, StageManifest, StageSpec

FakeTool = Callable[[Path, str], Path]
MVS = OpenMVS(bin_dir=Path("/opt/openmvs/bin"), version="2.4.0")
BANNER = "14:42:35 [App     ] OpenMVS x64 v2.4.0\n14:42:35 [App     ] Build date: Oct  5 2026"


def _manifest(stage: str, run_id: str = "abc") -> StageManifest:
    return StageManifest(
        stage=stage, run_id=run_id, status="succeeded", cache_key="k",
        backend=Backend("openmvs", "2.4.0"), command=[], parameters={}, inputs={}, started="",
        finished="", wall_s=0, cpu_s=0, peak_rss_mb=None, exit_code=0, host={},
    )  # fmt: skip


def _opt(spec: StageSpec, name: str) -> str:
    argv = [str(a) for a in spec.argv]
    return argv[argv.index(name) + 1]


def test_parse_version() -> None:
    assert parse_version(BANNER) == "2.4.0"
    assert parse_version("14:42:35 [App     ] OpenMVS x64 v2.3.0") == "2.3.0"
    assert parse_version("nothing") is None


def test_locate(tmp_path: Path, fake_tool: FakeTool, monkeypatch: pytest.MonkeyPatch) -> None:
    for tool in TOOLS:
        fake_tool(tmp_path / "bin" / tool, BANNER)
    found = openmvs.locate(tmp_path / "bin")
    assert found == OpenMVS(tmp_path / "bin", "2.4.0") and found.supported
    monkeypatch.setenv("EZ2D_OPENMVS_DIR", str(tmp_path / "bin"))
    assert openmvs.locate() == found
    (tmp_path / "bin" / "TextureMesh").unlink()
    with pytest.raises(BackendMissing, match="incomplete: no TextureMesh"):
        openmvs.locate()
    monkeypatch.setenv("EZ2D_OPENMVS_DIR", str(tmp_path / "nowhere"))
    with pytest.raises(BackendMissing, match="not found"):
        openmvs.locate()


def test_import_colmap(project: Project) -> None:
    spec = openmvs.import_colmap(MVS, project, _manifest("undistort", "u1"))
    undistort = project.stage_dir("undistort")
    assert spec.argv[0] == MVS.bin_dir / "InterfaceCOLMAP"
    assert _opt(spec, "-i") == str(undistort)
    assert _opt(spec, "--image-folder") == str(undistort / "images")
    assert _opt(spec, "-w") == str(project.stage_dir("mvs-import"))
    assert spec.inputs == {"undistorted": "run:u1"}


def test_densify(project: Project) -> None:
    spec = openmvs.densify(MVS, project, _manifest("mvs-import"))
    assert spec.argv[1] == project.stage_dir("mvs-import") / "scene.mvs"
    assert _opt(spec, "-w") == str(project.stage_dir("densify"))
    assert _opt(spec, "--resolution-level") == "1"
    assert "--mask-path" not in spec.argv and spec.parameters["masked"] is False

    masked = openmvs.densify(
        MVS,
        project,
        _manifest("mvs-import"),
        masks=_manifest("mask-undistort", "w1"),
        options=DensifyOptions(2, threads=3),
    )
    assert _opt(masked, "--mask-path") == str(project.stage_dir("mask-undistort") / "masks")
    assert masked.inputs["masks"] == "run:w1"
    assert _opt(masked, "--ignore-mask-label") == "0"
    assert _opt(masked, "--max-threads") == "3"
    assert masked.cache_key() != spec.cache_key()


def test_mesh_refine_texture(project: Project) -> None:
    dense, mesh = _manifest("densify", "d1"), _manifest("mesh", "m1")
    spec = openmvs.reconstruct_mesh(MVS, project, dense)
    assert _opt(spec, "--pointcloud-file") == str(project.stage_dir("densify") / "scene_dense.ply")

    with pytest.raises(BackendError, match="no mesh output"):
        openmvs.texture_mesh(MVS, project, dense, mesh)
    project.stage_dir("mesh").mkdir(parents=True)
    (project.stage_dir("mesh") / "scene_mesh.ply").write_text("ply")
    refine = openmvs.refine_mesh(MVS, project, dense, mesh)
    assert _opt(refine, "--mesh-file") == str(project.stage_dir("mesh") / "scene_mesh.ply")

    texture = openmvs.texture_mesh(MVS, project, dense, mesh, options=TextureOptions("glb"))
    assert _opt(texture, "--export-type") == "glb"
    assert texture.inputs == {"dense": "run:d1", "mesh": "run:m1"}


def test_texture_simplifies_to_target_faces(project: Project) -> None:
    dense, mesh = _manifest("densify", "d1"), _manifest("mesh", "m1")
    project.stage_dir("mesh").mkdir(parents=True)
    ply = project.stage_dir("mesh") / "scene_mesh.ply"
    ply.write_text(
        "ply\nformat binary_little_endian 1.0\nelement vertex 600\nproperty float x\n"
        "element face 1200\nproperty list uchar uint vertex_indices\nend_header\n"
    )
    spec = openmvs.texture_mesh(MVS, project, dense, mesh, options=TextureOptions(target_faces=300))
    assert _opt(spec, "--decimate") == "0.250000"
    assert spec.parameters["target_faces"] == 300
    # Already small enough: nothing to simplify.
    small = openmvs.texture_mesh(
        MVS, project, dense, mesh, options=TextureOptions(target_faces=5000)
    )
    assert "--decimate" not in [str(a) for a in small.argv]
    ply.write_text("not a ply")
    with pytest.raises(BackendError, match="simplify"):
        openmvs.texture_mesh(MVS, project, dense, mesh, options=TextureOptions(target_faces=300))


def test_progress() -> None:
    parse = OpenMVSProgress()
    lines = [
        "14:42:35 [App     ] Selecting images for dense reconstruction completed: 32 images (9ms)",
        "Estimated depth-maps 12 (37.50%, 1m24s, ETA 2m)...",
        "Estimated depth-maps 32 (100%, 3m10s382ms)          ",
        "Geometric-consistent estimated depth-maps 1 (3.12%, 1s, ETA 57s)...",
        "Points inserted 3976 (0.39%, 100ms, ETA 0ms)...",
        "\t32 images (32 calibrated) with a total of 9.31 MPixels (0.29 MPixels/image)",
        "14:47:43 [App     ] \tVmPeak:\t 1277692 kB",
    ]
    assert [parse(line) for line in lines] == [
        Progress("Selecting images for dense reconstruction completed"),
        Progress("Estimated depth-maps", 0.375),
        Progress("Estimated depth-maps", 1.0),
        Progress("Geometric-consistent estimated depth-maps", 0.0312),
        Progress("Points inserted", 0.0039),
        None,
        None,
    ]
