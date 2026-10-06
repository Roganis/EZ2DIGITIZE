# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""How well the placed cameras cover the object: gaps, heights, a still camera.

After camera placement, every registered photo has a position and a viewing
direction. The object is where the viewing axes meet (the point closest to
all of them, in the least-squares sense); "up" comes from `orientation`.
Each camera then has an angle around the object (azimuth) and a height
angle (elevation), which shows:

- a gap: no photos from part of the way around (the far side will be
  missing or guessed);
- a single ring: every photo from about the same height (the top and the
  underside are barely seen);
- a still camera: the camera hardly moved. With a turntable and no masks,
  COLMAP places the cameras from the static background, so they all end
  up in one spot and the object is lost.

The findings are advice for the next capture; nothing is stopped.
"""

from __future__ import annotations

import contextlib
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ez2digitize.backends.colmap_model import read_images
from ez2digitize.orientation import (
    Vector,
    _mul_transposed,
    estimate_up,
    quaternion_matrix,
)

# Larger gaps around the object are reported.
MAX_GAP_DEGREES = 90.0
# Below this spread of height angles, all photos are "one ring".
MIN_ELEVATION_SPREAD = 15.0
# A camera looking the same way (within this angle) in every photo stood still.
STILL_DEGREES = 10.0
MIN_CAMERAS = 6
# Cameras farther than this many median distances are misplaced: ignored.
OUTLIER_DISTANCE = 5.0


@dataclass(frozen=True)
class Coverage:
    cameras: int
    largest_gap_degrees: float | None  # None without an up direction
    elevation_degrees: tuple[float, float] | None  # lowest, highest
    camera_spread: float  # cameras' spread / median distance to the object
    view_spread_degrees: float  # widest angle between a view and the mean view
    findings: tuple[str, ...]


def analyse(model_dir: Path, orientations: Mapping[str, int] | None = None) -> Coverage | None:
    """Coverage of a COLMAP model's registered images; None if too few to judge."""
    images = read_images(model_dir)
    centres: list[Vector] = []
    axes: list[Vector] = []
    names = list(images)
    for pose in images.values():
        rotation = quaternion_matrix(pose.qvec)
        t = pose.tvec
        centre = _mul_transposed(rotation, t)
        centres.append((-centre[0], -centre[1], -centre[2]))
        axes.append(_mul_transposed(rotation, (0.0, 0.0, 1.0)))
    if len(centres) < MIN_CAMERAS:
        return None
    up_estimate = estimate_up(model_dir, orientations)
    return assess(centres, axes, up_estimate.up if up_estimate else None, names)


def assess(
    centres: Sequence[Vector],
    axes: Sequence[Vector],
    up: Vector | None,
    names: Sequence[str] | None = None,
) -> Coverage:
    # Scale-free: around an object the views turn; a fixed camera looks one way.
    units = [_unit(a) for a in axes]
    mean_view = _unit(tuple(sum(a[i] for a in units) for i in range(3)))  # type: ignore[arg-type]
    view_spread = max(
        math.degrees(math.acos(max(-1.0, min(1.0, _dot(a, mean_view))))) for a in units
    )
    if view_spread < STILL_DEGREES:
        return Coverage(
            len(centres), None, None, 0.0, round(view_spread, 1),
            (
                "The camera hardly moved between photos. If the object turned on a "
                "turntable, mask the background (or cover it in plain cloth), or the "
                "cameras are placed from the background and the object is lost.",
            ),
        )  # fmt: skip
    target = _closest_point(centres, axes)
    distances = [_norm(_sub(c, target)) for c in centres]
    median = sorted(distances)[len(distances) // 2] or 1.0
    # A camera COLMAP misplaced far away would dominate the analysis.
    far = [d > OUTLIER_DISTANCE * median for d in distances]
    kept = [c for c, out in zip(centres, far, strict=True) if not out]
    misplaced = [names[i] if names else str(i) for i, out in enumerate(far) if out]
    middle = _median_point(kept)
    from_middle = sorted(_norm(_sub(c, middle)) for c in kept)
    spread = from_middle[int(0.9 * (len(from_middle) - 1))] / median
    centres = kept
    findings: list[str] = []
    if misplaced:
        which = name_list(misplaced) if names else f"{len(misplaced)} photo(s)"
        findings.append(
            f"placed far from the others, probably wrongly: {which}. They can add noise; "
            "check them in the photo checks (blurry, or of something else?)."
        )
    gap = elevations = None
    if up is not None:
        u = _unit(up)
        e1 = _unit(_cross(u, (1.0, 0.0, 0.0) if abs(u[0]) < 0.9 else (0.0, 1.0, 0.0)))
        e2 = _cross(u, e1)
        azimuths, heights = [], []
        for centre in centres:
            v = _sub(centre, target)
            h = _dot(v, u)
            x, y = _dot(v, e1), _dot(v, e2)
            azimuths.append(math.degrees(math.atan2(y, x)) % 360.0)
            heights.append(math.degrees(math.atan2(h, math.hypot(x, y))))
        ordered = sorted(azimuths)
        gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False)]
        gaps.append(ordered[0] + 360.0 - ordered[-1])
        gap = round(max(gaps), 1)
        elevations = (round(min(heights), 1), round(max(heights), 1))
        if gap > MAX_GAP_DEGREES:
            findings.append(
                f"No photos from about {gap:.0f}° of the way around the object: that side "
                "will be missing or guessed. Walk all the way around (a photo every 10-15°)."
            )
        if elevations[1] - elevations[0] < MIN_ELEVATION_SPREAD:
            findings.append(
                f"All photos were taken from about the same height ({elevations[0]:.0f}° to "
                f"{elevations[1]:.0f}°). Add a ring from higher up (30-45°) to see the top, "
                "and one lower if the underside matters."
            )
    return Coverage(
        len(centres), gap, elevations, round(spread, 4), round(view_spread, 1), tuple(findings)
    )


# --- geometry --------------------------------------------------------------------------


def _closest_point(points: Sequence[Vector], directions: Sequence[Vector]) -> Vector:
    """The point nearest to all lines (point, direction), robustly.

    Least squares, reweighted a few times by 1/distance to each line, so a
    few cameras looking elsewhere (or misplaced) don't pull it away.
    """
    lines = [(p, _unit(d)) for p, d in zip(points, directions, strict=True)]
    weights = [1.0] * len(lines)
    point = _median_point(points)
    for _ in range(8):
        a = [[0.0] * 3 for _ in range(3)]
        b = [0.0, 0.0, 0.0]
        for (p, d), w in zip(lines, weights, strict=True):
            for i in range(3):
                for j in range(3):
                    m = w * ((1.0 if i == j else 0.0) - d[i] * d[j])
                    a[i][j] += m
                    b[i] += m * p[j]
        solved = _solve(a, b)
        if solved is None:
            return point
        point = solved
        misses = [_line_distance(point, p, d) for p, d in lines]
        floor = max(sorted(misses)[len(misses) // 2] * 0.1, 1e-9)
        weights = [1.0 / max(miss, floor) for miss in misses]
    return point


def _line_distance(point: Vector, origin: Vector, direction: Vector) -> float:
    v = _sub(point, origin)
    along = _dot(v, direction)
    return _norm(_sub(v, (direction[0] * along, direction[1] * along, direction[2] * along)))


def _median_point(points: Sequence[Vector]) -> Vector:
    middle = len(points) // 2
    return (
        sorted(p[0] for p in points)[middle],
        sorted(p[1] for p in points)[middle],
        sorted(p[2] for p in points)[middle],
    )


def _solve(a: list[list[float]], b: list[float]) -> Vector | None:
    def det(m: list[list[float]]) -> float:
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

    d = det(a)
    if abs(d) < 1e-12:
        return None  # parallel axes: no single meeting point
    result = []
    for k in range(3):
        m = [row[:] for row in a]
        for i in range(3):
            m[i][k] = b[i]
        result.append(det(m) / d)
    return (result[0], result[1], result[2])


def _sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector, b: Vector) -> Vector:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a: Vector) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Vector) -> Vector:
    n = _norm(a) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


# --- overlap -------------------------------------------------------------------------

# COLMAP pair ids: id1 * MAX + id2 (util/types.h, kMaxNumImages).
_MAX_IMAGES = 2147483647
# Verified two-view geometries that count as real overlap (calibrated,
# uncalibrated, planar, planar-or-panoramic); not degenerate or watermarks.
_GOOD_CONFIGS = (2, 3, 4, 6)
MIN_INLIERS = 15
MIN_NEIGHBOURS = 2


def weak_photos(database: Path) -> list[str]:
    """Photos sharing at least MIN_INLIERS verified matches with fewer than
    MIN_NEIGHBOURS other photos, from a COLMAP database after matching."""
    uri = f"{database.absolute().as_uri()}?mode=ro"
    with contextlib.closing(sqlite3.connect(uri, uri=True)) as db:
        names = dict(db.execute("SELECT image_id, name FROM images"))
        neighbours = dict.fromkeys(names, 0)
        rows = db.execute(
            "SELECT pair_id FROM two_view_geometries WHERE rows >= ? AND config IN (?, ?, ?, ?)",
            (MIN_INLIERS, *_GOOD_CONFIGS),
        )
        for (pair_id,) in rows:
            second = pair_id % _MAX_IMAGES
            first = (pair_id - second) // _MAX_IMAGES
            for image in (first, second):
                if image in neighbours:
                    neighbours[image] += 1
    return sorted(names[i] for i, n in neighbours.items() if n < MIN_NEIGHBOURS)


def name_list(names: Sequence[str], limit: int = 5) -> str:
    """`a, b, c and 4 more`, with capture folders dropped."""
    short = [name.rsplit("/", 1)[-1] for name in names]
    shown = ", ".join(short[:limit])
    return shown if len(short) <= limit else f"{shown} and {len(short) - limit} more"
