# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which way the model stands: the automatic estimate, or the user's correction.

The rotation from reconstruction coordinates to the upright frame (Y up)
is what the 3D view and the export use (see orientation.Placement). By
default it comes from the photos (orientation.estimate_up). When that is
wrong, or the photos don't say, the user sets it in the 3D view:

- level: three points on the surface the object stands on; their plane
  becomes the ground, up being the side the cameras are on;
- tilt: a quarter turn about a horizontal axis (a model on its side);
- turn: degrees about the vertical, which way the model faces.

The correction is kept as a `base` rotation (levelling and tilts) and a
`turn`, in project.json (`settings["orientation"]`) with the run id of the
camera placement it was made on; placing the cameras again changes the
coordinates, so an earlier correction is ignored (the automatic estimate
applies again), as the crop box and the scale are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ez2digitize import crop
from ez2digitize.backends.colmap_model import read_images
from ez2digitize.backends.common import BackendError
from ez2digitize.core.capture import list_bundles
from ez2digitize.core.files import FormatError
from ez2digitize.core.photos import exif_orientations
from ez2digitize.core.project import Project
from ez2digitize.crop import camera_run
from ez2digitize.orientation import (
    IDENTITY,
    Matrix,
    Vector,
    estimate_up,
    quaternion_matrix,
    rotation_between,
)
from ez2digitize.sides import upright_names

SETTING = "orientation"
Y_UP: Vector = (0.0, 1.0, 0.0)


class OrientationError(ValueError):
    pass


@dataclass(frozen=True)
class Orientation:
    base: Matrix  # reconstruction to level (rows), before the turn
    turn: float  # degrees about the vertical
    camera_run: str  # the mapping run it was set on

    @property
    def rotation(self) -> Matrix:
        """Reconstruction to upright."""
        return matmul(turn_matrix(self.turn), self.base)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": [list(row) for row in self.base],
            "turn": self.turn,
            "camera_run": self.camera_run,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Orientation | None:
        try:
            base = tuple(tuple(float(v) for v in row) for row in data["base"])
            turn = float(data.get("turn", 0.0))
            run = str(data["camera_run"])
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
        if len(base) != 3 or any(len(row) != 3 for row in base) or not math.isfinite(turn):
            return None
        if not _is_rotation(base):  # type: ignore[arg-type]
            return None
        return cls(base, turn, run)  # type: ignore[arg-type]


# --- the rotation in use ------------------------------------------------------------


def automatic(project: Project) -> Matrix | None:
    """The rotation from the photos' down directions; None if they don't agree."""
    model = _model(project)
    if not (model / "images.bin").is_file():
        return None
    try:
        bundles = list_bundles(project)
        estimate = estimate_up(model, exif_orientations(bundles), upright_names(bundles))
    except (OSError, ValueError, BackendError, FormatError):
        return None
    if estimate is None:
        return None
    return rotation_between(estimate.up, Y_UP)


def stored(project: Project) -> Orientation | None:
    return Orientation.from_dict(project.settings.get(SETTING))


def current(project: Project) -> Orientation | None:
    """The user's correction, if made on the current camera placement."""
    orientation = stored(project)
    if orientation is None or orientation.camera_run != camera_run(project):
        return None
    return orientation


def rotation(project: Project) -> Matrix | None:
    """Reconstruction to upright: the user's correction, else the automatic estimate."""
    manual = current(project)
    return manual.rotation if manual is not None else automatic(project)


def save(project: Project, orientation: Orientation | None) -> None:
    if orientation is None:
        project.settings.pop(SETTING, None)
    else:
        project.settings[SETTING] = orientation.to_dict()
    project.save()


def change(project: Project, orientation: Orientation | None) -> None:
    """Save a correction (None: back to the automatic estimate).

    A crop box made in the old frame is re-fitted level in the new one (see
    crop.relevelled); the scale is in reconstruction coordinates, unchanged.
    """
    box = crop.current(project)
    save(project, orientation)
    if box is not None:
        crop.save(project, crop.relevelled(box, rotation(project)))


def starting_point(project: Project) -> Orientation:
    """What a correction starts from: the current one, else the automatic estimate."""
    run = camera_run(project)
    if run is None:
        raise OrientationError("place the cameras first")
    manual = current(project)
    if manual is not None:
        return manual
    return Orientation(automatic(project) or IDENTITY, 0.0, run)


# --- corrections ------------------------------------------------------------------


def levelled(
    start: Orientation, points: tuple[Vector, Vector, Vector], cameras: Vector
) -> Orientation:
    """The plane through three points (reconstruction coordinates) as the ground.

    Up is the plane's normal on the side of `cameras` (the mean camera
    centre: photos are taken from above the surface). The turn is kept.
    """
    a, b, c = points
    normal = _cross(_sub(b, a), _sub(c, a))
    length = math.sqrt(sum(v * v for v in normal))
    if not math.isfinite(length) or length < 1e-12:
        raise OrientationError("pick three points that are not in a line")
    normal = (normal[0] / length, normal[1] / length, normal[2] / length)
    if _dot(normal, _sub(cameras, a)) < 0:
        normal = (-normal[0], -normal[1], -normal[2])
    return Orientation(rotation_between(normal, Y_UP), start.turn, start.camera_run)


def tilted(start: Orientation, axis: str, degrees: float = 90.0) -> Orientation:
    """A turn of `degrees` about the upright frame's X or Z axis, applied after the turn.

    Composed in the upright frame, then split back into base and turn: the
    tilt is about the axis as the user sees it.
    """
    tilt = _axis_matrix(axis, degrees)
    base = matmul(_transpose(turn_matrix(start.turn)), matmul(tilt, start.rotation))
    return Orientation(_orthonormal(base), start.turn, start.camera_run)


def turned(start: Orientation, degrees: float) -> Orientation:
    """The same levelling, facing another way (degrees about the vertical)."""
    turn = (degrees + 180.0) % 360.0 - 180.0
    return Orientation(start.base, round(turn, 6), start.camera_run)


def camera_centre(project: Project) -> Vector | None:
    """The mean centre of the placed cameras (reconstruction coordinates)."""
    try:
        images = read_images(_model(project))
    except (OSError, BackendError):
        return None
    centres = []
    for pose in images.values():
        r = quaternion_matrix(pose.qvec)  # world to camera
        t = pose.tvec
        centres.append(tuple(-sum(r[j][i] * t[j] for j in range(3)) for i in range(3)))
    if not centres:
        return None
    return tuple(sum(c[i] for c in centres) / len(centres) for i in range(3))  # type: ignore[return-value]


# --- small linear algebra -----------------------------------------------------------


def turn_matrix(degrees: float) -> Matrix:
    """Rotation by `degrees` about +Y (counter-clockwise seen from above)."""
    return _axis_matrix("y", degrees)


def matmul(a: Matrix, b: Matrix) -> Matrix:
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def mul(m: Matrix, v: Vector) -> Vector:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def mul_transposed(m: Matrix, v: Vector) -> Vector:
    """`m`ᵀ·`v`: from the upright frame back to the reconstruction's, for a rotation."""
    return mul(_transpose(m), v)


def _axis_matrix(axis: str, degrees: float) -> Matrix:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    # Exact quarter turns: no 6e-17 noise in stored matrices.
    c, s = (round(c), round(s)) if abs(degrees) % 90 == 0 else (c, s)
    if axis == "x":
        return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))
    if axis == "y":
        return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))
    if axis == "z":
        return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    raise ValueError(f"axis {axis!r}")


def _model(project: Project) -> Path:
    return project.stage_dir("undistort") / "sparse"


def _transpose(m: Matrix) -> Matrix:
    return ((m[0][0], m[1][0], m[2][0]), (m[0][1], m[1][1], m[2][1]), (m[0][2], m[1][2], m[2][2]))


def _orthonormal(m: Matrix) -> Matrix:
    """`m` with rounding drift removed (Gram-Schmidt on its rows)."""
    x = _unit(m[0])
    y = _sub(m[1], _scaled(x, _dot(x, m[1])))
    y = _unit(y)
    return (x, y, _cross(x, y))


def _is_rotation(m: Matrix) -> bool:
    product = matmul(m, _transpose(m))
    return (
        all(
            abs(product[i][j] - (1.0 if i == j else 0.0)) < 1e-6 for i in range(3) for j in range(3)
        )
        and _dot(_cross(m[0], m[1]), m[2]) > 0
    )


def _unit(v: Vector) -> Vector:
    n = math.sqrt(_dot(v, v)) or 1.0
    return (v[0] / n, v[1] / n, v[2] / n)


def _scaled(v: Vector, k: float) -> Vector:
    return (v[0] * k, v[1] * k, v[2] * k)


def _sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector, b: Vector) -> Vector:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
