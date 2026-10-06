# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small COLMAP models for tests: cameras on rings around a point."""

import math
import struct
from pathlib import Path

import numpy as np

Vector = tuple[float, float, float]


def ring(
    count: int, elevation: float, *, start: float = 0.0, span: float = 360.0, radius: float = 3.0
) -> list[Vector]:
    """Camera centres on a circle around the origin, Y up (COLMAP's frame, y down: see below)."""
    centres = []
    for i in range(count):
        azimuth = math.radians(start + span * i / count)
        up = math.radians(elevation)
        centres.append(
            (
                radius * math.cos(up) * math.cos(azimuth),
                -radius * math.sin(up),  # world "up" is -y, as photos held level put it
                radius * math.cos(up) * math.sin(azimuth),
            )
        )
    return centres


def write_images(path: Path, centres: list[Vector], target: Vector = (0.0, 0.0, 0.0)) -> None:
    """images.bin: one image per centre (c/<i>.jpg), held level, looking at `target`.

    Level: the image's down (+y) points along world +y, which stands the
    model up with -y up (see orientation.estimate_up).
    """
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(centres)))
        for i, centre in enumerate(centres, start=1):
            z = _unit(_sub(target, centre))
            x = _unit(_cross((0.0, 1.0, 0.0), z))
            y = _cross(z, x)
            rows = (x, y, z)  # world to camera
            t = tuple(-sum(rows[r][k] * centre[k] for k in range(3)) for r in range(3))
            fh.write(struct.pack("<I7dI", i, *_quaternion(rows), *t, 1))
            fh.write(f"c/{i:03d}.jpg".encode() + b"\0" + struct.pack("<Q", 0))


def _quaternion(m: tuple[Vector, Vector, Vector]) -> tuple[float, float, float, float]:
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        return (s / 4, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    i = max(range(3), key=lambda k: m[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * math.sqrt(1 + m[i][i] - m[j][j] - m[k][k])
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (m[k][j] - m[j][k]) / s
    q[1 + i] = s / 4
    q[1 + j] = (m[j][i] + m[i][j]) / s
    q[1 + k] = (m[k][i] + m[i][k]) / s
    return (q[0], q[1], q[2], q[3])


def _sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vector, b: Vector) -> Vector:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(v: Vector) -> Vector:
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) or 1.0
    return (v[0] / n, v[1] / n, v[2] / n)


def write_pinhole_camera(path: Path, width: int, height: int, focal: float) -> None:
    """cameras.bin: one PINHOLE camera (id 1), principal point in the middle."""
    path.write_bytes(
        struct.pack("<QIiQQ4d", 1, 1, 1, width, height, focal, focal, width / 2, height / 2)
    )


def look_at_rows(centre: Vector, target: Vector) -> tuple[Vector, Vector, Vector]:
    """World-to-camera rotation rows of a level camera at `centre` looking at `target`."""
    z = _unit(_sub(target, centre))
    x = _unit(_cross((0.0, 1.0, 0.0), z))
    return (x, _cross(z, x), z)


def render_sheet(
    texture: "np.ndarray",
    extent: float,
    centre: Vector,
    target: Vector,
    width: int,
    height: int,
    focal: float,
) -> bytes:
    """A grey PNG of the plane y = 0 seen by a pinhole camera (posed as write_images does).

    `texture` covers x and z from -extent/2 to extent/2 (rows along z); grey
    beyond it. Four samples a pixel.
    """
    import io

    from PIL import Image

    rows = np.array(look_at_rows(centre, target))
    n = texture.shape[0]
    image = np.zeros((height, width))
    for dx, dy in ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)):
        u, v = np.meshgrid(np.arange(width) + dx, np.arange(height) + dy)
        rays = np.stack([(u - width / 2) / focal, (v - height / 2) / focal, np.ones_like(u)], -1)
        world = rays @ rows  # camera to world: Rᵀ·d, as row vectors
        t = -centre[1] / world[..., 1]
        col = np.floor((centre[0] + t * world[..., 0] + extent / 2) / extent * n).astype(int)
        row = np.floor((centre[2] + t * world[..., 2] + extent / 2) / extent * n).astype(int)
        inside = (t > 0) & (col >= 0) & (col < n) & (row >= 0) & (row < n)
        image += np.where(inside, texture[row.clip(0, n - 1), col.clip(0, n - 1)], 128)
    buffer = io.BytesIO()
    Image.fromarray((image / 4).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def marker_scene(model: Path, images: Path, size_px: int = 120) -> float:
    """A COLMAP model and photos of 4 printed markers (ids 0-3) on a flat sheet.

    Eight level cameras on a ring above the sheet (the plane y = 0, in the
    world write_images uses). Returns the true millimetres per unit for 30 mm
    markers.
    """
    from ez2digitize import markers

    n, extent = 640, 16.0
    texture = np.full((n, n), 255, np.uint8)
    cell = size_px // 10
    for i, (cx, cz) in enumerate([(-5, -5), (5, -5), (-5, 5), (5, 5)]):
        square = np.kron(markers.marker_cells(i), np.ones((cell, cell), np.uint8))
        r0 = int((cz + extent / 2) * n / extent - square.shape[0] / 2)
        c0 = int((cx + extent / 2) * n / extent - square.shape[1] / 2)
        texture[r0 : r0 + square.shape[0], c0 : c0 + square.shape[1]] = square
    texture = np.ascontiguousarray(texture[:, ::-1])  # as seen from the cameras' side
    model.mkdir(parents=True, exist_ok=True)
    centres = ring(8, 55, radius=30)
    write_images(model / "images.bin", centres)
    write_pinhole_camera(model / "cameras.bin", 800, 600, 750)
    (model / "points3D.bin").write_bytes(struct.pack("<Q", 0))  # no sparse points
    for i, centre in enumerate(centres, start=1):
        path = images / f"c/{i:03d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(render_sheet(texture, extent, centre, (0, 0, 0), 800, 600, 750))
    edge_units = 8 * cell * extent / n
    return 30.0 / edge_units
