# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import json
import struct
from pathlib import Path

import pytest
from PIL import Image

from ez2digitize import views
from ez2digitize.backends.colmap import Colmap
from ez2digitize.core.files import write_json_atomic
from ez2digitize.core.project import Project
from ez2digitize.core.stage import MANIFEST_FILE, StageManifest

BACKEND = Colmap(Path("colmap"), "4.2.1").backend


def _succeed(project: Project, stage: str, run_id: str) -> Path:
    folder = project.stage_dir(stage)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = StageManifest(
        stage=stage, run_id=run_id, status="succeeded", cache_key="k", backend=BACKEND,
        command=[], parameters={}, inputs={}, started="", finished="2026-10-06T08:00:00",
        wall_s=1.0, cpu_s=1.0, peak_rss_mb=None, exit_code=0, host={},
    )  # fmt: skip
    write_json_atomic(folder / MANIFEST_FILE, manifest.to_dict())
    return folder


def _sparse_model(folder: Path) -> None:
    """Two PINHOLE cameras 4x3 px (fx = fy = 2), looking along +z, and 3 points."""
    folder.mkdir(parents=True)
    (folder / "cameras.bin").write_bytes(struct.pack("<QIiQQ4d", 1, 1, 1, 4, 3, 2.0, 2.0, 2.0, 1.5))
    images = struct.pack("<Q", 2)
    for image_id, (name, tx) in enumerate((("c/a.jpg", 0.0), ("c/b.jpg", -1.0)), start=1):
        images += struct.pack("<I7dI", image_id, 1, 0, 0, 0, tx, 0, 0, 1)
        images += name.encode() + b"\0" + struct.pack("<Q", 0)
    (folder / "images.bin").write_bytes(images)
    points = struct.pack("<Q", 3)
    for point_id, (x, y, z) in enumerate(((0, 0, 5), (1, 0, 5), (0, 1, 6)), start=1):
        track = struct.pack("<II", 1, 0) * 2  # seen by two images
        points += struct.pack("<Q3d3BdQ", point_id, x, y, z, 200, 100, 50, 0.5, 2) + track
    (folder / "points3D.bin").write_bytes(points)


def _textured_ply(folder: Path) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\ncomment TextureFile scene_textured0.png\n"
        "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        "element face 1\nproperty list uchar uint vertex_indices\n"
        "property list uchar float texcoord\nend_header\n"
    )
    body = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    body += struct.pack("<B3IB6f", 3, 0, 1, 2, 6, 0, 0, 1, 0, 0, 1)
    (folder / "scene_textured.ply").write_bytes(header.encode() + body)
    Image.new("RGB", (4, 4), (200, 50, 50)).save(folder / "scene_textured0.png")


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "project")


def test_nothing_to_show(project: Project) -> None:
    assert views.available(project) == []


def test_camera_placement(project: Project, tmp_path: Path) -> None:
    undistort = _succeed(project, "undistort", "u1")
    _sparse_model(undistort / "sparse")
    [view] = views.available(project)
    assert (view.key, view.kind, view.run_id, view.label) == (
        "cameras", "cameras", "u1", "Camera placement",
    )  # fmt: skip
    # The photos (identity rotation) were held level: down is +y, so up is -y,
    # and standing the model upright flips y.
    assert view.upright is not None
    assert view.upright[1][1] == pytest.approx(-1)

    files = views.files(view, tmp_path / "cache")
    ply, cameras = files["model"], files["cameras"]
    assert isinstance(ply, bytes) and isinstance(cameras, bytes)
    assert b"element vertex 3\n" in ply
    body = ply[ply.index(b"end_header\n") + 11 :]
    assert len(body) == 3 * 15
    assert struct.unpack_from("<3f3B", body, 15) == (1.0, 0.0, 5.0, 200, 100, 50)
    data = json.loads(cameras)
    a, b = data["cameras"]
    assert a["name"] == "c/a.jpg" and a["centre"] == [0, 0, 0]
    assert b["centre"] == [1.0, 0, 0]  # t = -R·C, R = identity
    assert a["fov"] == pytest.approx(73.74, abs=0.01)  # 2·atan(1.5 / 2)
    assert a["aspect"] == pytest.approx(4 / 3, abs=1e-3)
    assert a["rotation"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    spec = views.spec(view, {"model": "ez2d://data/x/model.ply"})
    assert spec["kind"] == "cameras" and spec["upright"] == [list(r) for r in view.upright]


def test_all_results_best_first(project: Project, tmp_path: Path) -> None:
    _sparse_model(_succeed(project, "undistort", "u1") / "sparse")
    (_succeed(project, "densify", "d1") / "scene_dense.ply").write_bytes(b"ply")
    (_succeed(project, "splat", "s1") / "splat.ply").write_bytes(b"ply")
    _textured_ply(_succeed(project, "texture", "t1"))
    shown = views.available(project)
    assert [v.key for v in shown] == ["mesh", "splat", "dense", "cameras"]
    assert views.files(shown[1], tmp_path)["model"] == project.stage_dir("splat") / "splat.ply"

    # No GLB was exported: the viewer converts OpenMVS's PLY, once.
    mesh = shown[0]
    assert mesh.kind == "glb" and mesh.upright is not None
    cache = tmp_path / "cache"
    glb = views.files(mesh, cache)["model"]
    assert isinstance(glb, Path) and glb.read_bytes()[:4] == b"glTF"
    assert views.files(mesh, cache)["model"] == glb


def test_mesh_from_the_upright_export(project: Project, tmp_path: Path) -> None:
    _textured_ply(_succeed(project, "texture", "t1"))
    exported = project.exports_dir / "20261006-080000"
    exported.mkdir(parents=True)
    (exported / "skull.glb").write_bytes(b"glTF")
    info = {"source": {"stage": "texture", "run_id": "t1"}, "align": True, "files": ["skull.glb"]}
    (exported / "export.json").write_text(json.dumps(info))
    [mesh] = views.available(project)
    assert mesh.source == exported / "skull.glb" and mesh.upright is None
    assert views.files(mesh, tmp_path)["model"] == exported / "skull.glb"

    # An export stood up another way (the orientation was corrected since) isn't used,
    info["upright"] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    (exported / "export.json").write_text(json.dumps(info))
    assert views.available(project)[0].source == project.stage_dir("texture")
    # nor one of an older run, or one left unaligned.
    info["upright"] = None
    info["source"] = {"stage": "texture", "run_id": "t0"}
    (exported / "export.json").write_text(json.dumps(info))
    assert views.available(project)[0].source == project.stage_dir("texture")


def test_unreadable_files_are_reported(project: Project, tmp_path: Path) -> None:
    sparse = _succeed(project, "undistort", "u1") / "sparse"
    _sparse_model(sparse)
    (sparse / "points3D.bin").write_bytes(b"\x05\x00")
    [view] = views.available(project)
    with pytest.raises(views.ViewError, match="Camera placement can't be shown"):
        views.files(view, tmp_path)
