# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Refining a camera placement plugin's poses with COLMAP (backends.colmap_refine).

The pairs and the known-poses model are checked here directly; the whole
refinement runs with the real pinned COLMAP on the synthetic scene, from
poses 2° off with a focal length 7% too long, as a feed-forward network
might give them (skipped without the pinned COLMAP and OpenMVS, like
test_real_pipeline).
"""

import json
import math
import os
import sqlite3
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scripts import make_plugin

from ez2digitize import pipeline, plugins
from ez2digitize.backends import colmap, colmap_refine, openmvs
from ez2digitize.backends.colmap_model import Camera, read_cameras, read_images
from ez2digitize.backends.common import BackendError, BackendMissing
from ez2digitize.core.capture import import_folder
from ez2digitize.core.project import Project


def _quaternion(r: np.ndarray) -> tuple[float, float, float, float]:
    w = math.sqrt(max(0.0, 1 + np.trace(r))) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return (w, x, y, z)


def _look_at(centre: np.ndarray) -> np.ndarray:
    """World to camera rotation, looking at the origin, OpenCV axes (z up in the world)."""
    forward = -centre / np.linalg.norm(centre)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(forward, right), forward])


def _write_model(folder: Path, poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """A plugin's model: one PINHOLE camera per photo (640 x 480), the given poses."""
    folder.mkdir(parents=True)
    with (folder / "cameras.bin").open("wb") as f, (folder / "images.bin").open("wb") as g:
        f.write(struct.pack("<Q", len(poses)))
        g.write(struct.pack("<Q", len(poses)))
        for i, (name, (r, centre)) in enumerate(sorted(poses.items()), 1):
            f.write(struct.pack("<IiQQ4d", i, 1, 640, 480, 600.0, 600.0, 320.0, 240.0))
            t = -r @ centre
            g.write(struct.pack("<I4d3dI", i, *_quaternion(r), *t, i))
            g.write(name.encode() + b"\0" + struct.pack("<Q", 0))
    (folder / "points3D.bin").write_bytes(struct.pack("<Q", 0))


def test_pose_pairs(tmp_path: Path) -> None:
    """A ring of 12 cameras looking in: neighbours pair, opposite sides don't."""
    poses = {}
    for k in range(12):
        angle = 2 * math.pi * k / 12
        centre = np.array([3 * math.cos(angle), 3 * math.sin(angle), 1.0])
        poses[f"c/{k:02d}.jpg"] = (_look_at(centre), centre)
    _write_model(tmp_path / "model", poses)
    pairs = colmap_refine.pose_pairs(tmp_path / "model", neighbours=3, max_angle_deg=70)
    assert ("c/00.jpg", "c/01.jpg") in pairs and ("c/00.jpg", "c/11.jpg") in pairs
    assert ("c/00.jpg", "c/06.jpg") not in pairs  # looking at each other: 180° apart
    assert all(a < b for a, b in pairs) and len(set(pairs)) == len(pairs)
    # 30° apart each: 90° (three along) is past the 70° limit, however near.
    assert ("c/00.jpg", "c/03.jpg") not in pairs and ("c/00.jpg", "c/09.jpg") not in pairs


def _database(path: Path, *, extra_sensor: bool = False) -> None:
    """COLMAP 4.2.1's tables: two cameras (SIMPLE_RADIAL 1280 x 960), their rigs, a frame each.

    Ids are deliberately not in step: frames and rigs numbered apart from
    images and cameras, as after merging cameras.
    """
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER,"
            " height INTEGER, params BLOB, prior_focal_length INTEGER);"
            "CREATE TABLE rigs (rig_id INTEGER PRIMARY KEY, ref_sensor_id INTEGER,"
            " ref_sensor_type INTEGER);"
            "CREATE TABLE rig_sensors (rig_id INTEGER, sensor_id INTEGER, sensor_type INTEGER,"
            " sensor_from_rig BLOB);"
            "CREATE TABLE frames (frame_id INTEGER PRIMARY KEY, rig_id INTEGER);"
            "CREATE TABLE frame_data (frame_id INTEGER, data_id INTEGER, sensor_id INTEGER,"
            " sensor_type INTEGER);"
            "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER);"
        )
        for camera_id, rig_id in ((3, 7), (5, 8)):
            params = struct.pack("<4d", 1500.0, 640.0, 480.0, 0.0)
            db.execute("INSERT INTO cameras VALUES (?, 2, 1280, 960, ?, 0)", (camera_id, params))
            db.execute("INSERT INTO rigs VALUES (?, ?, 0)", (rig_id, camera_id))
        images = ((11, "a/1.jpg", 3, 21, 7), (12, "a/2.jpg", 3, 22, 7), (13, "b/1.jpg", 5, 23, 8))
        for image_id, name, camera_id, frame_id, rig_id in images:
            db.execute("INSERT INTO images VALUES (?, ?, ?)", (image_id, name, camera_id))
            db.execute("INSERT INTO frames VALUES (?, ?)", (frame_id, rig_id))
            data = (frame_id, image_id, camera_id)
            db.execute("INSERT INTO frame_data VALUES (?, ?, ?, 0)", data)
        if extra_sensor:
            db.execute("INSERT INTO rig_sensors VALUES (7, 5, 0, NULL)")


def test_known_poses_model_mirrors_the_database(tmp_path: Path) -> None:
    database = tmp_path / "database.db"
    _database(database)
    centre = np.array([0.0, 0.0, -3.0])
    poses = {name: (np.eye(3), centre + k) for k, name in enumerate(("a/1.jpg", "a/2.jpg"))}
    _write_model(tmp_path / "plugin", poses)  # b/1.jpg not placed by the plugin
    out = tmp_path / "known"
    assert colmap_refine.write_known_poses_model(database, tmp_path / "plugin", out) == 2

    cameras = read_cameras(out)
    # The database's cameras, models and sizes; the plugin's intrinsics at the
    # database's size (twice the plugin's): f 1200, centre (640, 480).
    assert cameras[3] == Camera("SIMPLE_RADIAL", 1280, 960, (1200.0, 640.0, 480.0, 0.0))
    assert cameras[5] == Camera("SIMPLE_RADIAL", 1280, 960, (1500.0, 640.0, 480.0, 0.0))
    images = read_images(out)
    assert sorted(images) == ["a/1.jpg", "a/2.jpg"]
    assert images["a/1.jpg"].camera_id == 3 and images["a/1.jpg"].tvec == pytest.approx((0, 0, 3))
    rigs = (out / "rigs.bin").read_bytes()
    assert struct.unpack_from("<Q", rigs)[0] == 2
    assert struct.unpack_from("<IIiI", rigs, 8) == (7, 1, 0, 3)  # rig 7: camera 3
    frames = (out / "frames.bin").read_bytes()
    assert struct.unpack_from("<Q", frames)[0] == 2
    frame_id, rig_id = struct.unpack_from("<II", frames, 8)
    assert (frame_id, rig_id) == (21, 7)
    assert struct.unpack_from("<IiIQ", frames, 8 + 8 + 56) == (1, 0, 3, 11)  # camera 3, image 11
    assert struct.unpack("<Q", (out / "points3D.bin").read_bytes()) == (0,)

    _database(tmp_path / "rigged.db", extra_sensor=True)
    with pytest.raises(BackendError, match="rigs of one camera"):
        colmap_refine.write_known_poses_model(tmp_path / "rigged.db", tmp_path / "plugin", out)


def test_params_for_other_models() -> None:
    assert colmap_refine.params_for("PINHOLE", 10, 12, 3, 4) == [10, 12, 3, 4]
    assert colmap_refine.params_for("SIMPLE_PINHOLE", 10, 12, 3, 4) == [11, 3, 4]
    assert colmap_refine.params_for("OPENCV", 10, 12, 3, 4) == [10, 12, 3, 4, 0, 0, 0, 0]
    assert colmap_refine.intrinsics(Camera("RADIAL", 8, 6, (5.0, 4.0, 3.0, 0.1, 0.2))) == (
        5.0,
        5.0,
        4.0,
        3.0,
    )


# --- the real thing -----------------------------------------------------------------

# A plugin that places the synthetic scene's cameras from the ground truth,
# 2° and 5 cm off, with a focal length 7% too long, in a world of its own.
NOISY_POSES = r"""
import json, math, os, struct, sys
from pathlib import Path
import numpy as np
args = sys.argv[1:]
opt = lambda name: args[args.index(name) + 1]
truth = json.loads(Path(os.environ["EZ2D_TEST_TRUTH"]).read_text())["cameras"]
names = Path(opt("--list")).read_text().split()
rng = np.random.default_rng(1)

def turn(axis, degrees):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = math.radians(degrees)
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * k @ k

def quaternion(r):
    w = math.sqrt(max(0, 1 + np.trace(r))) / 2
    x = math.copysign(math.sqrt(max(0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return w, x, y, z

world, scale, shift = turn([0.3, 1, 0.2], 40), 0.37, np.array([1.0, -2.0, 0.5])
model = Path(opt("--output")) / "sparse" / "0"
model.mkdir(parents=True)
cameras, images = struct.pack("<Q", len(names)), struct.pack("<Q", len(names))
for i, name in enumerate(names, 1):
    camera = truth[name.split("/")[-1]]
    r = turn(rng.normal(size=3), 2.0) @ np.array(camera["rotation"]) @ world.T
    centre = scale * world @ (np.array(camera["center"]) + rng.normal(size=3) * 0.05) + shift
    f = camera["focal_px"] * 1.07
    width, height = camera["size"]
    cameras += struct.pack("<IiQQ4d", i, 1, width, height, f, f, width / 2, height / 2)
    images += struct.pack("<I4d3dI", i, *quaternion(r), *(-r @ centre), i)
    images += name.encode() + b"\0" + struct.pack("<Q", 0)
(model / "cameras.bin").write_bytes(cameras)
(model / "images.bin").write_bytes(images)
(model / "points3D.bin").write_bytes(struct.pack("<Q", 0))
"""


def _relative_rotation_errors(model: Path, truth: dict[str, dict[str, object]]) -> list[float]:
    """Degrees between each pair's relative rotation and the true one: frame-free."""
    placed = {
        name: np.array(colmap_refine._rotation(pose.qvec))
        for name, pose in read_images(model).items()
    }
    names = sorted(placed)
    errors = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            true_a = np.array(truth[a.split("/")[-1]]["rotation"])
            true_b = np.array(truth[b.split("/")[-1]]["rotation"])
            difference = (placed[a] @ placed[b].T).T @ (true_a @ true_b.T)
            cosine = (np.trace(difference) - 1) / 2
            errors.append(math.degrees(math.acos(min(1.0, max(-1.0, cosine)))))
    return errors


def test_refine_on_synthetic_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    give_up = pytest.fail if os.environ.get("EZ2D_REQUIRE_BACKENDS") else pytest.skip
    try:
        sfm, mvs = colmap.locate(), openmvs.locate()
    except BackendMissing as exc:
        give_up(str(exc))
    if not sfm.supported:
        give_up(f"needs the pinned COLMAP, found {sfm.version}")
    synthetic = pytest.importorskip("ez2d_bench.synthetic", reason="needs --group feasibility")
    count = synthetic.generate(tmp_path / "synthetic")
    truth = json.loads((tmp_path / "synthetic" / "ground_truth.json").read_text())["cameras"]
    monkeypatch.setenv("EZ2D_TEST_TRUTH", str(tmp_path / "synthetic" / "ground_truth.json"))
    monkeypatch.setenv(plugins.ENV_VAR, str(tmp_path / "plugins"))
    source = make_plugin(
        tmp_path / "src" / "noisy-poses",
        provides="poses",
        script=NOISY_POSES,
        command=["bin/run", "--list", "{image_list}", "--output", "{output}"],
    )
    plugin = plugins.install(source)
    project = Project.create(tmp_path / "project")
    import_folder(project, tmp_path / "synthetic" / "images")
    tools = pipeline.Tools(colmap=sfm, openmvs=mvs, poses=plugin)
    settings = replace(pipeline.MeshSettings(), refine_poses=True, use_masks=False)

    result = pipeline.run_sparse(project, tools, settings)

    assert result.registered_images == count
    before = _relative_rotation_errors(project.stage_dir("poses") / "sparse" / "0", truth)
    after = _relative_rotation_errors(result.model, truth)
    assert np.median(before) > 2.0
    assert np.median(after) < 0.3 and max(after) < 1.0
    (focal,) = {c.params[0] for c in read_cameras(result.model).values()}
    assert focal == pytest.approx(truth["view_000.jpg"]["focal_px"], rel=0.01)
