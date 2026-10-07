# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Real-world scale: a known distance between two points of the reconstruction.

Photos alone don't tell how big an object is, so a reconstruction comes out
in arbitrary units. The user picks two points in the 3D view (on the camera
placement or the dense cloud) and types the real distance between them;
the ratio is the scale, in millimetres per reconstruction unit. Exports
then come out in real units (see export).

The scale can also come from printed markers (ez2digitize.markers): then
the two points are the ends of a marker edge, stretched to the median of
all the markers' edges, `source` is "markers" and `detail` says how many
markers agreed how well.

Or from the phone's tracking (`from_tracking`): where a motion log
recorded the camera positions in metres (ARCore), the median ratio of the
distances between tracked cameras to those between the same cameras
placed by COLMAP; `source` is "tracking". ARCore drifts a few centimetres,
so this is a default for when nothing better is set: markers replace it,
and a scale set by hand wins over it.

The points are kept in the reconstruction's own coordinates, with the real
distance and the run id of the camera placement they were picked on, in
project.json (`settings["scale"]`). Placing the cameras again changes the
coordinates (and their scale), so a scale from an earlier placement is
ignored, as the crop box is. The viewer shows and picks points in the
upright frame; `from_upright` and `to_upright` convert.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ez2digitize.core.project import Project
from ez2digitize.crop import camera_run
from ez2digitize.orientation import IDENTITY, Matrix, Vector

SETTING = "scale"
# Points closer than this (in reconstruction units) can't give a scale.
MIN_SEPARATION = 1e-9


class ScaleError(ValueError):
    pass


@dataclass(frozen=True)
class Scale:
    points: tuple[Vector, Vector]  # reconstruction coordinates
    distance_mm: float  # the real distance between them
    camera_run: str  # the mapping run they were picked on
    source: str = "points"  # "points" (picked in the 3D view), "markers" or "tracking"
    detail: str = ""  # for markers: how many, and how well they agreed

    @property
    def model_distance(self) -> float:
        return _distance(*self.points)

    @property
    def mm_per_unit(self) -> float:
        return self.distance_mm / self.model_distance

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "points": [list(p) for p in self.points],
            "distance_mm": self.distance_mm,
            "camera_run": self.camera_run,
        }
        if self.source != "points":
            data["source"] = self.source
            data["detail"] = self.detail
        return data

    @classmethod
    def from_dict(cls, data: Any) -> Scale | None:
        try:
            a, b = (tuple(float(v) for v in p) for p in data["points"])
            distance = float(data["distance_mm"])
            run = str(data["camera_run"])
            source = str(data.get("source", "points"))
            detail = str(data.get("detail", ""))
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
        try:
            return make((a, b), distance, run, source=source, detail=detail)  # type: ignore[arg-type]
        except ScaleError:
            return None


def make(
    points: tuple[Vector, Vector],
    distance_mm: float,
    camera_run: str,
    *,
    source: str = "points",
    detail: str = "",
) -> Scale:
    """A scale from two points and their real distance; ScaleError if unusable."""
    a, b = points
    if len(a) != 3 or len(b) != 3 or not all(math.isfinite(v) for v in (*a, *b)):
        raise ScaleError("pick two points")
    if not (math.isfinite(distance_mm) and distance_mm > 0):
        raise ScaleError("the real distance must be more than 0 mm")
    if _distance(a, b) < MIN_SEPARATION:
        raise ScaleError("pick two different points")
    return Scale((a, b), distance_mm, camera_run, source, detail)


def from_upright(points: tuple[Vector, Vector], upright: Matrix | None) -> tuple[Vector, Vector]:
    """Points picked in the viewer (upright frame) in reconstruction coordinates."""
    u = upright or IDENTITY
    return (_mul_transposed(u, points[0]), _mul_transposed(u, points[1]))


def to_upright(points: tuple[Vector, Vector], upright: Matrix | None) -> tuple[Vector, Vector]:
    u = upright or IDENTITY
    return (_mul(u, points[0]), _mul(u, points[1]))


# --- in the project ---------------------------------------------------------------


def stored(project: Project) -> Scale | None:
    return Scale.from_dict(project.settings.get(SETTING))


def current(project: Project) -> Scale | None:
    """The stored scale, if it was picked on the current camera placement."""
    scale = stored(project)
    if scale is None or scale.camera_run != camera_run(project):
        return None
    return scale


def save(project: Project, scale: Scale | None) -> None:
    if scale is None:
        project.settings.pop(SETTING, None)
    else:
        project.settings[SETTING] = scale.to_dict()
    project.save()


def describe(scale: Scale) -> str:
    if scale.source in ("markers", "tracking"):
        return f"from {scale.detail}"
    return f"{_mm(scale.distance_mm)} between the two points ({scale.mm_per_unit:.4g} mm per unit)"


# --- from the phone's tracking -------------------------------------------------------

# Cameras with a metric pose needed, and how far apart the tracked ones must
# be at most: ARCore drifts a few centimetres, so a phone that hardly moved
# (over a turntable) can't give a scale.
MIN_TRACKED = 5
MIN_TRACKED_SPREAD_M = 0.15
# Pairs of cameras closer than this share of the farthest pair are left out:
# their ratio is mostly the tracking's noise.
MIN_PAIR_SHARE = 0.2
# Beyond this spread of the ratios (median absolute deviation over the
# median), the tracking is said to disagree with the camera placement.
TRACKING_DISAGREE = 0.05


@dataclass(frozen=True)
class Tracked:
    """The model's scale measured against tracked camera positions."""

    metres_per_unit: float
    spread: float  # relative median absolute deviation of the pair ratios
    cameras: int
    far_pair: tuple[Vector, Vector]  # the model centres farthest apart


def measure_tracking(placed: dict[str, Vector], tracked: dict[str, Vector]) -> Tracked | None:
    """Compare camera centres placed by COLMAP with tracked ones (metres), by name.

    None if too few cameras are in both or the tracked ones hardly moved.
    """
    names = sorted(set(placed) & set(tracked))
    if len(names) < MIN_TRACKED:
        return None
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    if max(math.dist(tracked[a], tracked[b]) for a, b in pairs) < MIN_TRACKED_SPREAD_M:
        return None
    far = max(pairs, key=lambda p: math.dist(placed[p[0]], placed[p[1]]))
    longest = math.dist(placed[far[0]], placed[far[1]])
    if longest < MIN_SEPARATION:
        return None
    ratios = sorted(
        math.dist(tracked[a], tracked[b]) / math.dist(placed[a], placed[b])
        for a, b in pairs
        if math.dist(placed[a], placed[b]) >= MIN_PAIR_SHARE * longest
    )
    median = _median(ratios)
    spread = _median([abs(r - median) for r in ratios]) / median
    return Tracked(median, spread, len(names), (placed[far[0]], placed[far[1]]))


def from_tracking(project: Project, measured: Tracked) -> str | None:
    """After camera placement: set the scale from the tracking, unless one is set.

    Returns what to tell the user, or None. A scale set by hand or from
    markers is kept and compared.
    """
    run = camera_run(project)
    if run is None:
        return None
    mm_per_unit = 1000 * measured.metres_per_unit
    current_scale = current(project)
    if current_scale is not None and current_scale.source != "tracking":
        differs = mm_per_unit / current_scale.mm_per_unit - 1
        return (
            f"the phone's tracking gives a scale {100 * differs:+.1f} % off the one set "
            f"{'from markers' if current_scale.source == 'markers' else 'by hand'}, "
            "which is kept"
        )
    a, b = measured.far_pair
    detail = (
        f"{measured.cameras} camera positions tracked by the phone "
        f"(they agree within {100 * measured.spread:.1f} %)"
    )
    save(
        project, make((a, b), mm_per_unit * math.dist(a, b), run, source="tracking", detail=detail)
    )
    note = f"scale from {detail}: exports are in real units; markers measure it more precisely"
    if measured.spread > TRACKING_DISAGREE:
        note += "; the tracking and the camera placement disagree, so check it"
    return note


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _mm(value: float) -> str:
    return f"{value:.4g} mm"


def _distance(a: Vector, b: Vector) -> float:
    return math.dist(a, b)


def _mul(m: Matrix, v: Vector) -> Vector:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _mul_transposed(m: Matrix, v: Vector) -> Vector:
    return (
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    )
