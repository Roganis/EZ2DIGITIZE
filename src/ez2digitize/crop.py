# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The crop box: what the dense reconstruction keeps.

Without one, OpenMVS estimates a region of interest from the sparse points
(`--estimate-roi`) and crops to it; that box often includes the table or
background around a small object. The user's box, drawn in the 3D view
after camera placement, replaces it (`DensifyPointCloud --import-roi-file`).

The box is an oriented box in the reconstruction's own coordinates, as
OpenMVS stores a region of interest: `rotation` (rows: the box's axes;
world to box), `centre`, and `half_size` along each axis. It is kept in
project.json (`settings["crop_box"]`) with the run id of the camera
placement it was drawn on: placing the cameras again changes the
coordinates, so a box from an earlier placement is ignored (the pipeline
says so).

The viewer edits it in the upright frame (see views.upright_rotation),
level and turned about the vertical axis by `yaw`; `from_upright` and
`to_upright` convert.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ez2digitize.backends.common import BackendError
from ez2digitize.core.project import Project
from ez2digitize.core.stage import load_manifest
from ez2digitize.orientation import IDENTITY, Matrix, Vector

SETTING = "crop_box"
# The automatic box: the sparse points between these percentiles, grown a bit.
LOW, HIGH = 0.02, 0.98
MARGIN = 0.1


@dataclass(frozen=True)
class CropBox:
    rotation: Matrix  # world to box: rows are the box's axes
    centre: Vector
    half_size: Vector
    camera_run: str  # the mapping run it was drawn on

    def to_dict(self) -> dict[str, Any]:
        return {
            "rotation": [list(row) for row in self.rotation],
            "centre": list(self.centre),
            "half_size": list(self.half_size),
            "camera_run": self.camera_run,
        }

    @classmethod
    def from_dict(cls, data: Any) -> CropBox | None:
        try:
            rotation = tuple(tuple(float(v) for v in row) for row in data["rotation"])
            centre = tuple(float(v) for v in data["centre"])
            half = tuple(float(v) for v in data["half_size"])
            run = str(data["camera_run"])
        except (KeyError, TypeError, ValueError):
            return None
        if len(rotation) != 3 or any(len(r) != 3 for r in rotation):
            return None
        if len(centre) != 3 or len(half) != 3 or min(half) <= 0:
            return None
        return cls(rotation, centre, half, run)  # type: ignore[arg-type]

    def roi_text(self) -> str:
        """The box as OpenMVS reads a region of interest (OBB: rotation rows,
        centre, half sizes; see libs/Common/OBB.h)."""
        rows = [" ".join(_num(v) for v in row) for row in self.rotation]
        return "\n".join([*rows, _join(self.centre), _join(self.half_size)]) + "\n"

    def contains(self, point: Vector) -> bool:
        local = _mul(self.rotation, _sub(point, self.centre))
        return all(abs(local[i]) <= self.half_size[i] for i in range(3))


@dataclass(frozen=True)
class UprightBox:
    """The box as the viewer edits it: in the upright frame, turned by `yaw`
    degrees about the vertical (+Y) axis."""

    centre: Vector
    half_size: Vector
    yaw: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"centre": list(self.centre), "half_size": list(self.half_size), "yaw": self.yaw}

    @classmethod
    def from_dict(cls, data: Any) -> UprightBox | None:
        try:
            centre = tuple(float(v) for v in data["centre"])
            half = tuple(float(v) for v in data["half_size"])
            yaw = float(data.get("yaw", 0.0))
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
        if len(centre) != 3 or len(half) != 3 or min(half) <= 0:
            return None
        if not all(math.isfinite(v) for v in (*centre, *half, yaw)):
            return None
        return cls(centre, half, yaw)


def yaw_matrix(degrees: float) -> Matrix:
    """World (upright) to box for a box turned by `degrees` about +Y."""
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return ((c, 0.0, -s), (0.0, 1.0, 0.0), (s, 0.0, c))


def from_upright(box: UprightBox, upright: Matrix | None, camera_run: str) -> CropBox:
    """The crop box in model coordinates; `upright` maps model to upright."""
    u = upright or IDENTITY
    rotation = _matmul(yaw_matrix(box.yaw), u)
    centre = _mul(_transpose(u), box.centre)
    return CropBox(rotation, centre, box.half_size, camera_run)


def to_upright(box: CropBox, upright: Matrix | None) -> UprightBox:
    """The box in the viewer's upright frame (its yaw about the vertical)."""
    u = upright or IDENTITY
    in_upright = _matmul(box.rotation, _transpose(u))  # rows: box axes, upright coords
    x_axis = in_upright[0]
    yaw = math.degrees(math.atan2(-x_axis[2], x_axis[0]))
    return UprightBox(_mul(u, box.centre), box.half_size, round(yaw, 3))


def relevelled(box: CropBox, upright: Matrix | None) -> CropBox:
    """The level box around `box` in another upright frame (the orientation changed).

    The viewer can only edit a box that is level; after the up direction
    changes, the old box is tilted, so it is replaced by the smallest level
    box that holds all of it.
    """
    u = upright or IDENTITY
    to_world = _transpose(box.rotation)  # box axes to model coordinates
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = (sx * box.half_size[0], sy * box.half_size[1], sz * box.half_size[2])
                model = _add(box.centre, _mul(to_world, local))
                corners.append(_mul(u, model))
    low = [min(c[i] for c in corners) for i in range(3)]
    high = [max(c[i] for c in corners) for i in range(3)]
    centre = tuple((lo + hi) / 2 for lo, hi in zip(low, high, strict=True))
    half = tuple(max((hi - lo) / 2, 1e-9) for lo, hi in zip(low, high, strict=True))
    level = UprightBox(centre, half)  # type: ignore[arg-type]
    return from_upright(level, upright, box.camera_run)


def automatic(points: Sequence[Vector], upright: Matrix | None) -> UprightBox | None:
    """A starting box: where most sparse points are (upright frame), grown by 10 %."""
    if not points:
        return None
    u = upright or IDENTITY
    turned = [_mul(u, p) for p in points]
    low, high = [], []
    for axis in range(3):
        values = sorted(p[axis] for p in turned)
        low.append(values[int(LOW * (len(values) - 1))])
        high.append(values[int(HIGH * (len(values) - 1))])
    centre = tuple((lo + hi) / 2 for lo, hi in zip(low, high, strict=True))
    half = tuple(max((hi - lo) / 2 * (1 + MARGIN), 1e-6) for lo, hi in zip(low, high, strict=True))
    return UprightBox(centre, half)  # type: ignore[arg-type]


# --- in the project ---------------------------------------------------------------


def camera_run(project: Project) -> str | None:
    """The run id of the current camera placement (mapping), if it succeeded."""
    manifest = load_manifest(project.stage_dir("mapping"))
    return manifest.run_id if manifest is not None and manifest.succeeded else None


def stored(project: Project) -> CropBox | None:
    return CropBox.from_dict(project.settings.get(SETTING))


def current(project: Project) -> CropBox | None:
    """The stored box, if it was drawn on the current camera placement."""
    box = stored(project)
    if box is None or box.camera_run != camera_run(project):
        return None
    return box


def automatic_for(project: Project) -> UprightBox | None:
    """The starting box for the current camera placement (None before it)."""
    model = project.stage_dir("undistort") / "sparse"
    if camera_run(project) is None or not (model / "points3D.bin").is_file():
        return None
    from ez2digitize import views  # views reads the model; avoid an import cycle

    try:
        points, _colors = views.read_points(model)
    except BackendError:
        return None
    return automatic(points, views.upright_rotation(project))


def save(project: Project, box: CropBox | None) -> None:
    if box is None:
        project.settings.pop(SETTING, None)
    else:
        project.settings[SETTING] = box.to_dict()
    project.save()


# --- small linear algebra ---------------------------------------------------------


def _num(value: float) -> str:
    return repr(float(value))


def _join(values: Iterable[float]) -> str:
    return " ".join(_num(v) for v in values)


def _mul(m: Matrix, v: Vector) -> Vector:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _transpose(m: Matrix) -> Matrix:
    return ((m[0][0], m[1][0], m[2][0]), (m[0][1], m[1][1], m[2][1]), (m[0][2], m[1][2], m[2][2]))


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )
