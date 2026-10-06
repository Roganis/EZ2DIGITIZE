# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The VGGT camera placement plugin (tools/plugins/vggt-poses), without VGGT.

Its manifest, and the conversion of VGGT's predictions into a COLMAP model:
a sphere seen by a ring of cameras, rendered as the depth maps VGGT would
predict at its padded square resolution, must come back as the true
cameras and points on the sphere, seen by several photos each.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import vggt_colmap
from numpy.typing import NDArray
from vggt_colmap import Placement, Prediction

from ez2digitize import plugins
from ez2digitize.backends.colmap_model import read_cameras, read_images
from ez2digitize.views import read_points

PLUGIN = Path(__file__).parents[1] / "tools" / "plugins" / "vggt-poses"
RES = 96
RADIUS = 1.0


def test_manifest() -> None:
    plugin = plugins.read_plugin(PLUGIN)
    assert (plugin.id, plugin.slot, plugin.gpu) == ("vggt-poses", "poses", True)
    covers = {lic.covers: lic.spdx for lic in plugin.licenses}
    assert covers["plugin code"] == "GPL-3.0-or-later"
    assert covers["VGGT code"] == "LicenseRef-VGGT"
    assert "CC-BY-NC-4.0" in covers.values()
    assert not plugin.free  # shown as not known to be free before it can be used
    assert plugin.with_masks == ()  # VGGT has no use for masks


def _look_at(centre: NDArray[np.float64]) -> NDArray[np.float64]:
    """Camera from world [R|t], OpenCV axes (x right, y down, z forward), at the origin."""
    forward = -centre / np.linalg.norm(centre)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    r = np.stack([right, down, forward])
    return np.concatenate([r, (-r @ centre)[:, None]], axis=1)


def _scene(
    sizes: list[tuple[int, int]],
) -> tuple[Prediction, list[tuple[float, float, float, float]]]:
    """A sphere at the origin seen from a ring; the true photo intrinsics too."""
    extrinsic, intrinsic, depth, truth = [], [], [], []
    grid_v, grid_u = np.mgrid[0:RES, 0:RES].astype(np.float64) + 0.5
    for n, size in enumerate(sizes):
        angle = 2 * math.pi * n / len(sizes)
        centre = np.array([4 * math.cos(angle), 0.6, 4 * math.sin(angle)])
        rt = _look_at(centre)
        width, height = size
        focal = 1.1 * max(width, height)
        true = (focal, focal, width / 2 + 3.0, height / 2 - 2.0)  # off-centre a little
        place = Placement.of(size, RES)
        k = np.array(
            [
                [true[0] * place.scale, 0, (true[2] + place.left) * place.scale],
                [0, true[1] * place.scale, (true[3] + place.top) * place.scale],
                [0, 0, 1],
            ]
        )
        # Ray-cast the sphere: depth along the optical axis, 0 where it misses.
        rays = np.linalg.inv(k) @ np.stack([grid_u.ravel(), grid_v.ravel(), np.ones(RES * RES)])
        world_dirs = rt[:, :3].T @ rays
        b = 2 * (world_dirs.T @ centre)
        a = np.sum(world_dirs**2, axis=0)
        c = centre @ centre - RADIUS**2
        disc = b * b - 4 * a * c
        s = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), 0.0)
        depth.append(s.reshape(RES, RES))  # rays have z = 1, so s is the depth
        extrinsic.append(rt)
        intrinsic.append(k)
        truth.append(true)
    prediction = Prediction(
        names=[f"cap/{n:03d}.jpg" for n in range(len(sizes))],
        sizes=sizes,
        resolution=RES,
        extrinsic=np.array(extrinsic),
        intrinsic=np.array(intrinsic),
        depth=np.array(depth),
        confidence=np.ones((len(sizes), RES, RES)),
    )
    return prediction, truth


def test_predictions_become_a_colmap_model(tmp_path: Path) -> None:
    sizes = [(800, 600), (600, 800)] * 6  # landscape and portrait photos
    prediction, truth = _scene(sizes)
    points = vggt_colmap.sparse_points(prediction, per_photo=200)
    assert len(points.xyz) > 500
    assert all(len(track) >= 2 for track in points.tracks)
    assert np.allclose(np.linalg.norm(points.xyz, axis=1), RADIUS, atol=0.03)
    model = tmp_path / "sparse" / "0"
    vggt_colmap.write_model(model, prediction, points)

    cameras = read_cameras(model)
    for n, (width, height) in enumerate(sizes):
        camera = cameras[n + 1]
        assert (camera.model, camera.width, camera.height) == ("PINHOLE", width, height)
        assert camera.params == pytest.approx(truth[n], abs=1e-6)
    images = read_images(model)
    assert sorted(images) == prediction.names
    for n, name in enumerate(prediction.names):
        pose = images[name]
        r = prediction.extrinsic[n][:, :3]
        w, x, y, z = pose.qvec
        back = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        assert back == pytest.approx(r, abs=1e-9)
        assert pose.tvec == pytest.approx(prediction.extrinsic[n][:, 3], abs=1e-9)
    xyz, _colors = read_points(model)
    assert len(xyz) == len(points.xyz)

    # Each observation is where the point projects with the true intrinsics.
    for p in range(0, len(points.xyz), 97):
        for photo, x_obs, y_obs in points.tracks[p]:
            fx, fy, cx, cy = truth[photo]
            cam = (
                prediction.extrinsic[photo][:, :3] @ points.xyz[p]
                + prediction.extrinsic[photo][:, 3]
            )
            assert (fx * cam[0] / cam[2] + cx, fy * cam[1] / cam[2] + cy) == pytest.approx(
                (x_obs, y_obs), abs=1e-6
            )


def test_occluded_points_are_not_tracked() -> None:
    """The back of the sphere: a photo there sees other points first."""
    prediction, _truth = _scene([(640, 480)] * 2)  # two photos on opposite sides
    points = vggt_colmap.sparse_points(prediction, per_photo=500, min_views=1)
    assert points.tracks and all(len(track) == 1 for track in points.tracks)
    assert vggt_colmap.sparse_points(prediction, per_photo=500).xyz.shape == (0, 3)


def test_quaternion_branches() -> None:
    for axis in np.eye(3):
        for angle in (0.3, math.pi - 0.01, math.pi):
            k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            r = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)
            w, x, y, z = vggt_colmap.quaternion(r)
            assert w >= 0 and math.isclose(w * w + x * x + y * y + z * z, 1.0)
            assert abs(w) == pytest.approx(abs(math.cos(angle / 2)), abs=1e-9)
