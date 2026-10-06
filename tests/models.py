# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small COLMAP models for tests: cameras on rings around a point."""

import math
import struct
from pathlib import Path

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
