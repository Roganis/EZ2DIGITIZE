# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import math
import struct
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy", reason="needs --group feasibility")

from ez2d_bench import synthetic  # noqa: E402
from ez2d_bench.evaluate import (  # noqa: E402
    Camera,
    EvalError,
    Mesh,
    evaluate,
    read_ground_truth_cameras,
    read_mesh,
    umeyama,
)


def _rotation(axis: list[float], degrees: float) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = math.radians(degrees)
    result: np.ndarray = np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * k @ k
    return result


@pytest.fixture
def truth(tmp_path: Path) -> tuple[Mesh, dict[str, Camera]]:
    cameras = {}
    for i in range(8):
        angle = 2 * math.pi * i / 8
        eye = np.array([2 * math.cos(angle), 2 * math.sin(angle), 1.0 + 0.2 * (i % 2)])
        right, up, forward = synthetic._look_at(eye, np.array([0.0, 0.0, 0.3]))
        cameras[f"view_{i:03d}.jpg"] = {
            "center": eye.tolist(), "rotation": np.stack([right, -up, forward]).tolist()
        }  # fmt: skip
    synthetic._write_ground_truth(tmp_path, cameras)
    _, cams = read_ground_truth_cameras(tmp_path / "ground_truth.json")
    return read_mesh(tmp_path / "ground_truth.ply"), cams


def _moved(
    mesh: Mesh, cameras: dict[str, Camera], scale: float, rotation: np.ndarray, t: np.ndarray
) -> tuple[Mesh, dict[str, Camera]]:
    """The same scene in another frame: x' = s R x + t (as a reconstruction would be)."""
    moved_mesh = Mesh(scale * mesh.vertices @ rotation.T + t, mesh.faces)
    moved_cams = {
        n: Camera(scale * rotation @ c.center + t, c.rotation @ rotation.T)
        for n, c in cameras.items()
    }
    return moved_mesh, moved_cams


def test_umeyama_recovers_a_similarity() -> None:
    rng = np.random.default_rng(1)
    src = rng.random((20, 3))
    r = _rotation([1, 2, 3], 40)
    dst = 2.5 * src @ r.T + np.array([1.0, -2.0, 0.5])
    s, r_est, t = umeyama(src, dst)
    assert s == pytest.approx(2.5) and np.allclose(r_est, r) and np.allclose(t, [1, -2, 0.5])


def test_perfect_reconstruction_in_another_frame(truth: tuple[Mesh, dict[str, Camera]]) -> None:
    mesh, cams = truth
    recon, recon_cams = _moved(mesh, cams, 0.37, _rotation([0, 1, 1], 70), np.array([5.0, 1, -3]))
    report = evaluate(recon, recon_cams, mesh, cams, samples=40_000)
    assert report.alignment["scale"] == pytest.approx(1 / 0.37, rel=1e-6)
    # ICP refines the exact camera alignment on sampled points: sampling noise.
    assert report.poses["center_max"] < 0.005 and report.poses["rotation_max_deg"] < 0.3
    # Only sampling noise remains: two independent samplings of the same surface.
    assert report.accuracy_median < 0.004 and report.completeness_median < 0.004
    assert all(t["f_score"] > 0.99 for t in report.thresholds[1:])
    assert report.outside_share == 0


def test_missing_face_lowers_completeness_only(truth: tuple[Mesh, dict[str, Camera]]) -> None:
    mesh, cams = truth
    top = mesh.faces[:2]  # the first quad is the top, 0.8 x 0.6 of 1.18 m2 total
    partial = Mesh(mesh.vertices, mesh.faces[2:])
    report = evaluate(partial, cams, mesh, cams, samples=40_000)
    lost = 0.8 * 0.6 / (0.8 * 0.6 + 2 * 0.6 * 0.7 + 2 * 0.8 * 0.7)
    assert len(top) == 2
    recall_1pct = report.thresholds[1]["recall"]
    assert recall_1pct == pytest.approx(1 - lost, abs=0.04)
    assert report.thresholds[1]["precision"] > 0.99


def test_background_is_left_out_of_accuracy(truth: tuple[Mesh, dict[str, Camera]]) -> None:
    mesh, cams = truth
    # A big floor far below the object, like a table or turntable.
    floor = np.array([[-5, -5, -3], [5, -5, -3], [5, 5, -3], [-5, 5, -3]], float)
    n = len(mesh.vertices)
    cluttered = Mesh(
        np.vstack([mesh.vertices, floor]),
        np.vstack([mesh.faces, [[n, n + 1, n + 2], [n, n + 2, n + 3]]]),
    )
    report = evaluate(cluttered, cams, mesh, cams, samples=40_000)
    assert report.outside_share > 0.9
    assert report.thresholds[1]["precision"] > 0.99


def test_too_few_matching_cameras(truth: tuple[Mesh, dict[str, Camera]]) -> None:
    mesh, cams = truth
    two = dict(list(cams.items())[:2])
    with pytest.raises(EvalError, match="only 2 cameras"):
        evaluate(mesh, two, mesh, cams, samples=1000)


def test_mesh_readers_agree(tmp_path: Path, truth: tuple[Mesh, dict[str, Camera]]) -> None:
    mesh, _ = truth
    obj = tmp_path / "m.obj"
    obj.write_text(
        "".join(f"v {x} {y} {z}\n" for x, y, z in mesh.vertices)
        + "".join(f"f {a + 1}/1 {b + 1}/1 {c + 1}/1\n" for a, b, c in mesh.faces)
    )
    binary = tmp_path / "m.ply"
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(mesh.vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(mesh.faces)}\nproperty list uchar uint vertex_indices\n"
        "end_header\n"
    )
    body = b"".join(struct.pack("<3f", *v) for v in mesh.vertices)
    body += b"".join(struct.pack("<B3I", 3, *f) for f in mesh.faces)
    binary.write_bytes(header.encode() + body)
    for path in (obj, binary):
        other = read_mesh(path)
        assert np.allclose(other.vertices, mesh.vertices, atol=1e-6)
        assert np.array_equal(other.faces, mesh.faces)
