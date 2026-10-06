# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which way is up: stand the reconstruction upright for export.

A reconstruction comes out in an arbitrary frame. People hold the camera
roughly level, so the image "down" directions, averaged over every
registered photo, point along gravity (the same idea as COLMAP's
`model_orientation_aligner` with IMAGE-ORIENTATION, which however maps
gravity to +Y, upside down for glTF, and ignores EXIF rotation; COLMAP
reads pixels without applying it).

`Placement` turns model coordinates into export coordinates: rotate up to
+Y, then move the object so it is centred on the vertical axis and stands
on the ground plane (lowest point at height 0). Y-up suits OBJ and glTF;
`z_up` gives 3D-printing coordinates (STL, 3MF). With no usable estimate
(photos at all angles), the export keeps the model's own frame.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ez2digitize.backends.colmap_model import read_images

Vector = tuple[float, float, float]
Matrix = tuple[Vector, Vector, Vector]

# Image "down" in camera coordinates (x right, y down in the stored pixels)
# for each EXIF orientation: 6 and 8 are phones held upright, stored sideways.
EXIF_DOWN: dict[int, Vector] = {
    1: (0.0, 1.0, 0.0), 2: (0.0, 1.0, 0.0), 3: (0.0, -1.0, 0.0), 4: (0.0, -1.0, 0.0),
    5: (1.0, 0.0, 0.0), 6: (1.0, 0.0, 0.0), 7: (-1.0, 0.0, 0.0), 8: (-1.0, 0.0, 0.0),
}  # fmt: skip
# Below this, the photos' down directions disagree too much to trust the mean
# (1.0: all identical; a level orbit with ±30° tilt gives about 0.9).
MIN_AGREEMENT = 0.5
IDENTITY: Matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@dataclass(frozen=True)
class UpEstimate:
    up: Vector  # in model coordinates
    agreement: float  # length of the mean down vector, 0..1
    images: int


def estimate_up(
    model_dir: Path,
    orientations: Mapping[str, int] | None = None,
    only: Collection[str] | None = None,
) -> UpEstimate | None:
    """Up from the registered images of a COLMAP model; None if it can't tell.

    `only`: the images to go by (the first side of a two-sided scan, see
    sides.upright_names); None for all.
    """
    images = read_images(model_dir)
    downs = []
    for name, pose in images.items():
        if only is not None and name not in only:
            continue
        down_camera = EXIF_DOWN.get((orientations or {}).get(name, 1), EXIF_DOWN[1])
        rotation = quaternion_matrix(pose.qvec)  # world to camera
        downs.append(_mul_transposed(rotation, down_camera))
    return up_from_downs(downs)


def up_from_downs(downs: Iterable[Vector]) -> UpEstimate | None:
    items = [_normalise(d) for d in downs]
    if not items:
        return None
    mean = tuple(sum(d[i] for d in items) / len(items) for i in range(3))
    agreement = _length(mean)  # type: ignore[arg-type]
    if agreement < MIN_AGREEMENT:
        return None
    gravity = _normalise(mean)  # type: ignore[arg-type]
    return UpEstimate((-gravity[0], -gravity[1], -gravity[2]), round(agreement, 4), len(items))


@dataclass(frozen=True)
class Placement:
    """Export coordinates = scale · (rotation · model + offset) (then Z-up if asked).

    `scale` converts reconstruction units to the export's (see ez2digitize.scale).
    """

    rotation: Matrix
    offset: Vector
    scale: float = 1.0

    def apply(self, positions: array[float], *, z_up: bool = False) -> array[float]:
        (a, b, c), (d, e, f), (g, h, i) = self.rotation
        ox, oy, oz = self.offset
        s = self.scale
        out = array("f", positions)
        for k in range(0, len(out), 3):
            x, y, z = positions[k], positions[k + 1], positions[k + 2]
            nx = (a * x + b * y + c * z + ox) * s
            ny = (d * x + e * y + f * z + oy) * s
            nz = (g * x + h * y + i * z + oz) * s
            if z_up:  # rotate +90° about X: Y-up becomes Z-up
                ny, nz = -nz, ny
            out[k], out[k + 1], out[k + 2] = nx, ny, nz
        return out

    def rotate(self, vector: Vector, *, z_up: bool = False) -> Vector:
        (a, b, c), (d, e, f), (g, h, i) = self.rotation
        x, y, z = vector
        nx, ny, nz = a * x + b * y + c * z, d * x + e * y + f * z, g * x + h * y + i * z
        return (nx, -nz, ny) if z_up else (nx, ny, nz)

    def to_dict(self) -> dict[str, object]:
        return {
            "rotation": [list(row) for row in self.rotation],
            "offset": list(self.offset),
            "scale": self.scale,
        }


def place(positions: array[float], up: Vector | None) -> Placement:
    """Rotate `up` to +Y, centre on the vertical axis, lowest point at y = 0."""
    rotation = rotation_between(up, (0.0, 1.0, 0.0)) if up is not None else IDENTITY
    return place_rotated(positions, rotation)


def place_rotated(positions: array[float], rotation: Matrix) -> Placement:
    """Rotate by `rotation` (model to upright), then centre and put on the ground."""
    turned = Placement(rotation, (0.0, 0.0, 0.0)).apply(positions)
    if not turned:
        return Placement(rotation, (0.0, 0.0, 0.0))
    xs, ys, zs = turned[0::3], turned[1::3], turned[2::3]
    offset = (-(min(xs) + max(xs)) / 2, -min(ys), -(min(zs) + max(zs)) / 2)
    return Placement(rotation, offset)


# --- small linear algebra -----------------------------------------------------------


def quaternion_matrix(q: tuple[float, float, float, float]) -> Matrix:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def rotation_between(source: Vector, target: Vector) -> Matrix:
    """The rotation turning direction `source` onto `target` (Rodrigues)."""
    a, b = _normalise(source), _normalise(target)
    axis = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    sin = _length(axis)
    cos = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    if sin < 1e-9:
        if cos > 0:
            return IDENTITY
        # Opposite: half a turn about any axis perpendicular to a.
        helper = (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0)
        perpendicular = _normalise(
            (
                a[1] * helper[2] - a[2] * helper[1],
                a[2] * helper[0] - a[0] * helper[2],
                a[0] * helper[1] - a[1] * helper[0],
            )
        )
        u, v, w = perpendicular
        return (
            (2 * u * u - 1, 2 * u * v, 2 * u * w),
            (2 * u * v, 2 * v * v - 1, 2 * v * w),
            (2 * u * w, 2 * v * w, 2 * w * w - 1),
        )
    k = (axis[0] / sin, axis[1] / sin, axis[2] / sin)
    t = 1 - cos
    kx, ky, kz = k
    return (
        (cos + kx * kx * t, kx * ky * t - kz * sin, kx * kz * t + ky * sin),
        (ky * kx * t + kz * sin, cos + ky * ky * t, ky * kz * t - kx * sin),
        (kz * kx * t - ky * sin, kz * ky * t + kx * sin, cos + kz * kz * t),
    )


def _mul_transposed(m: Matrix, v: Vector) -> Vector:
    """mᵀ · v (camera to world for a world-to-camera rotation)."""
    return (
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    )


def _length(v: Vector) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _normalise(v: Vector) -> Vector:
    n = _length(v) or 1.0
    return (v[0] / n, v[1] / n, v[2] / n)
