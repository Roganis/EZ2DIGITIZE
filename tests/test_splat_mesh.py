# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The mesh from the splats: oriented points, the Poisson stage, the mesh files."""

import json
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ez2digitize import crop, pipeline, presets, splat_mesh, views
from ez2digitize.backends.brush import Brush
from ez2digitize.core.capture import import_files
from ez2digitize.core.meshio import MeshFormatError, _read_header
from ez2digitize.core.project import Project
from ez2digitize.core.splats import Splats
from ez2digitize.pipeline import Notice, PipelineError, Tools


def sphere_splats(n: int = 4000, seed: int = 1) -> Splats:
    """Flat splats on a unit sphere: shortest axis (z, local) along the radius."""
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    z = np.array([0.0, 0.0, 1.0])
    q = np.concatenate([(1 + p @ z)[:, None], np.cross(z, p)], axis=1)  # w x y z
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    scales = np.log(np.tile([0.03, 0.03, 0.002], (n, 1)))
    opacities = np.full(n, 3.0)
    opacities[: n // 10] = -3.0  # a tenth nearly transparent
    return Splats(p, scales, q, opacities, np.full((n, 3), 0.8), np.zeros((n, 0, 3)))


CAMERAS = np.array([[4 * np.cos(a), 0.5, 4 * np.sin(a)] for a in np.linspace(0, 6.2, 12)])


def test_oriented_points() -> None:
    splats = sphere_splats()
    positions, normals, rgb = splat_mesh.oriented_points(splats, CAMERAS)
    # The transparent tenth is left out, and the largest 1% of the rest.
    assert 0.88 * splats.count <= len(positions) <= 0.9 * splats.count
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    radial = np.abs(np.einsum("ij,ij->i", normals, positions))
    assert np.all(radial > 0.999)  # along the shortest axis
    # Turned towards the nearest camera: outward wherever that camera sees the
    # point (on a unit sphere, camera c sees p if c·p > 1).
    nearest = CAMERAS[np.argmin(((positions[:, None] - CAMERAS[None]) ** 2).sum(-1), axis=1)]
    seen = np.einsum("ij,ij->i", nearest, positions) > 1
    assert seen.mean() > 0.5
    assert np.all(np.einsum("ij,ij->i", normals[seen], positions[seen]) > 0)
    assert np.all(rgb == round((0.5 + splat_mesh.SH_C0 * 0.8) * 255))

    # A crop box keeps what is inside it.
    box = crop.CropBox(((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)), (0.0, 0.0, 0.0), (2, 2, 0.5), "r")
    inside, _n, _c = splat_mesh.oriented_points(splats, CAMERAS, box=box)
    assert 0 < len(inside) < len(positions) and np.all(np.abs(inside[:, 2]) <= 0.5)


def test_points_file(tmp_path: Path) -> None:
    positions, normals, rgb = splat_mesh.oriented_points(sphere_splats(500), CAMERAS)
    path = splat_mesh.write_points(tmp_path / "points.ply", positions, normals, rgb)
    with path.open("rb") as fh:
        elements, _comments = _read_header(fh, path)
        data = fh.read()
    (vertex,) = elements
    assert [p.name for p in vertex.properties] == [
        "x", "y", "z", "nx", "ny", "nz", "red", "green", "blue",
    ]  # fmt: skip
    assert vertex.count == len(positions) and len(data) == vertex.count * (6 * 4 + 3)


def _tetrahedron(path: Path) -> Path:
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\nproperty float value\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "element face 4\nproperty list uchar int vertex_indices\nend_header\n"
    )
    body = b"".join(
        struct.pack("<4f3B", *v, 1.0, 200, 100, 50)
        for v in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    )
    body += b"".join(
        struct.pack("<B3i", 3, *f) for f in ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    )
    path.write_bytes(header.encode() + body)
    return path


def test_mesh_and_glb(tmp_path: Path) -> None:
    mesh = splat_mesh.read_mesh(_tetrahedron(tmp_path / "mesh.ply"))
    assert mesh.positions.shape == (4, 3) and mesh.faces.shape == (4, 3)
    assert mesh.colors.tolist()[0] == [200, 100, 50]
    glb = splat_mesh.write_glb(mesh, tmp_path / "mesh.glb").read_bytes()
    magic, version, total = struct.unpack_from("<4sII", glb)
    assert (magic, version, total) == (b"glTF", 2, len(glb))
    length, kind = struct.unpack_from("<I4s", glb, 12)
    gltf = json.loads(glb[20 : 20 + length])
    assert kind == b"JSON"
    primitive = gltf["meshes"][0]["primitives"][0]
    assert primitive["attributes"] == {"POSITION": 0, "COLOR_0": 1}
    counts = [a["count"] for a in gltf["accessors"]]
    assert counts == [4, 4, 12]
    assert gltf["accessors"][0]["max"] == [1.0, 1.0, 1.0]

    (tmp_path / "bad.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
    with pytest.raises(MeshFormatError):
        splat_mesh.read_mesh(tmp_path / "bad.ply")


# --- in the pipeline -----------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> Project:
    project = Project.create(tmp_path / "project")
    files = []
    for n, name in enumerate(("a.jpg", "b.jpg", "c.jpg")):
        Image.new("RGB", (8, 6), (n * 40, 0, 0)).save(tmp_path / name)
        files.append(tmp_path / name)
    import_files(project, files, source="folder")
    return project


def test_splats_and_their_mesh(
    project: Project, fake_tools: Tools, fake_brush: Brush, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ez2digitize.core.hardware import Gpu

    monkeypatch.setattr(pipeline, "detect_gpus", lambda: [Gpu("amd", "RX 7900 GRE")])
    monkeypatch.setattr(splat_mesh, "MIN_POINTS", 1)  # the fake Brush trains two splats
    tools = replace(fake_tools, brush=fake_brush)
    settings = presets.mesh_settings("fast", splat_mesh=True)
    assert settings.splat_mesh is not None and settings.splat_mesh.depth == 9
    events: list[pipeline.PipelineEvent] = []
    result = pipeline.run_splat(project, tools, settings, on_event=events.append)
    assert result.mesh == project.stage_dir("splat-mesh") / "mesh.ply"
    log = (project.stage_dir("splat-mesh") / "log.txt").read_text()
    assert "poisson_mesher" in log and "--PoissonMeshing.depth 9" in log
    assert (project.stage_dir("splat-mesh") / "points.ply").is_file()
    assert [f.suffix for f in result.mesh_exports] == [".glb", ".ply"]
    assert any(
        isinstance(e, Notice) and e.message.startswith("mesh from the splats exported")
        for e in events
    )
    shown = [v.key for v in views.available(project)]
    assert shown[0] == "splat-mesh" and "splat" in shown
    [view] = [v for v in views.available(project) if v.key == "splat-mesh"]
    assert view.source.suffix == ".glb"  # the export, already stood upright

    # Without it, no mesh; with too few solid splats, a clear error.
    plain = pipeline.run_splat(project, tools, presets.mesh_settings("fast"))
    assert plain.mesh is None
    monkeypatch.setattr(splat_mesh, "MIN_POINTS", 100)
    with pytest.raises(PipelineError, match="only 2 splats are solid enough"):
        pipeline.run_splat(project, tools, presets.mesh_settings("balanced", splat_mesh=True))
