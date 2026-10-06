# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import math
import struct
from array import array
from pathlib import Path

import pytest

from ez2digitize.orientation import (
    IDENTITY,
    Placement,
    estimate_up,
    place,
    quaternion_matrix,
    rotation_between,
    up_from_downs,
)


def _apply(m: tuple[tuple[float, ...], ...], v: tuple[float, float, float]) -> list[float]:
    return [sum(m[r][c] * v[c] for c in range(3)) for r in range(3)]


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ((0, 0, 1), (0, 1, 0)),
        ((1, 2, 3), (0, 1, 0)),
        ((0, -1, 0), (0, 1, 0)),
        ((0, 1, 0), (0, 1, 0)),
    ],
)
def test_rotation_between(
    source: tuple[float, float, float], target: tuple[float, float, float]
) -> None:
    rotation = rotation_between(source, target)
    n = math.sqrt(sum(c * c for c in source))
    turned = _apply(rotation, (source[0] / n, source[1] / n, source[2] / n))
    assert turned == pytest.approx(list(target), abs=1e-9)
    # A rotation: orthonormal rows.
    for i in range(3):
        for j in range(3):
            dot = sum(rotation[i][k] * rotation[j][k] for k in range(3))
            assert dot == pytest.approx(1.0 if i == j else 0.0, abs=1e-9)


def test_up_from_downs() -> None:
    estimate = up_from_downs([(0, 0, -1), (0.2, 0, -1), (-0.2, 0, -1)])
    assert estimate is not None and estimate.up == pytest.approx((0, 0, 1), abs=1e-9)
    assert estimate.images == 3 and estimate.agreement > 0.98
    # Photos at all angles: no estimate rather than a wrong one.
    assert up_from_downs([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]) is None
    assert up_from_downs([]) is None


def test_place_stands_the_object_on_the_ground() -> None:
    # A box from z=5 to z=7 whose up is +Z.
    corners = [(x, y, z) for x in (1, 3) for y in (-1, 1) for z in (5, 7)]
    positions = array("f", [c for p in corners for c in p])
    placement = place(positions, (0, 0, 1))
    placed = placement.apply(positions)
    xs, ys, zs = placed[0::3], placed[1::3], placed[2::3]
    assert min(ys) == pytest.approx(0) and max(ys) == pytest.approx(2)
    assert (min(xs) + max(xs)) == pytest.approx(0, abs=1e-6)
    assert (min(zs) + max(zs)) == pytest.approx(0, abs=1e-6)
    printable = placement.apply(positions, z_up=True)
    assert min(printable[2::3]) == pytest.approx(0) and max(printable[2::3]) == pytest.approx(2)
    assert place(positions, None).rotation == IDENTITY


def _images_bin(path: Path, poses: dict[str, tuple[float, float, float, float]]) -> None:
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(poses)))
        for i, (name, q) in enumerate(poses.items(), start=1):
            fh.write(struct.pack("<I7dI", i, *q, 0, 0, 0, 1))
            fh.write(name.encode() + b"\0")
            fh.write(struct.pack("<Q", 0))


def test_estimate_up_from_cameras_and_exif(tmp_path: Path) -> None:
    # Identity pose: the camera's down (+y) is the world's +y, so up is -y.
    _images_bin(tmp_path / "images.bin", {"c/a.jpg": (1, 0, 0, 0), "c/b.jpg": (1, 0, 0, 0)})
    estimate = estimate_up(tmp_path)
    assert estimate is not None and estimate.up == pytest.approx((0, -1, 0))
    # Phone photos stored sideways (EXIF 6): down is the camera's +x.
    sideways = estimate_up(tmp_path, {"c/a.jpg": 6, "c/b.jpg": 6})
    assert sideways is not None and sideways.up == pytest.approx((-1, 0, 0))


def test_quaternion_matrix() -> None:
    # 90° about Z.
    half = math.sqrt(0.5)
    assert _apply(quaternion_matrix((half, 0, 0, half)), (1, 0, 0)) == pytest.approx([0, 1, 0])


def test_placement_dict() -> None:
    assert Placement(IDENTITY, (1, 2, 3)).to_dict()["offset"] == [1, 2, 3]


def test_estimate_up_from_the_first_side_only(tmp_path: Path) -> None:
    # Two photos as usual, two of the object turned over: seen from the
    # object, those cameras are upside down (180° about the view axis).
    poses = {
        "top/a.jpg": (1.0, 0.0, 0.0, 0.0),
        "top/b.jpg": (1.0, 0.0, 0.0, 0.0),
        "under/c.jpg": (0.0, 0.0, 0.0, 1.0),
        "under/d.jpg": (0.0, 0.0, 0.0, 1.0),
    }
    _images_bin(tmp_path / "images.bin", poses)
    assert estimate_up(tmp_path) is None  # they cancel out
    first = estimate_up(tmp_path, only={"top/a.jpg", "top/b.jpg"})
    assert first is not None and first.up == pytest.approx((0, -1, 0)) and first.images == 2
