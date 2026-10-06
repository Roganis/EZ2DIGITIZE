# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Real-world scale from printed markers.

Print the marker sheet (`sheet_svg`: AprilTag tag36h11 squares of a known
size round the edge of an A4 page), put the object in the middle, and take
the photos as usual. After camera placement, the markers are found in the
placed photos (AprilTag, in-process: a small BSD-licensed library), each
corner seen from two or more photos is triangulated with the cameras'
poses, and the printed size over the markers' edge lengths in the
reconstruction is the scale (millimetres per unit): the median over every
edge of every marker, with their spread as a check.

The size is the black square's edge (the "8 cells" of a tag36h11 marker,
not counting its white margin). Printers scale pages ("fit to page"), so
the sheet carries a 100 mm line to measure, and `size_mm` is what the user
measured on a marker.

Detection runs on the undistorted photos, whose cameras are plain pinhole
ones (COLMAP's image_undistorter), so a corner projects as K [R | t] X.
"""

from __future__ import annotations

import math
import os
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ez2digitize import scale
from ez2digitize.backends.colmap_model import read_cameras, read_images
from ez2digitize.backends.common import BackendError
from ez2digitize.core.project import Project
from ez2digitize.crop import camera_run
from ez2digitize.orientation import Vector, quaternion_matrix

FAMILY = "tag36h11"
DEFAULT_SIZE_MM = 30.0
SIZE_SETTING = "marker_size_mm"
# The sheet: A4, a 4 x 5 grid of 45 mm places, markers round the edge only,
# so the middle (90 x 135 mm) is free for the object.
PAGE_MM = (210.0, 297.0)
GRID = (4, 5)
PITCH_MM = 45.0
SHEET_IDS = tuple(range(14))
# A detection must decode without corrected bits and stand out this clearly.
MIN_MARGIN = 30.0
# Corners triangulated with a worse reprojection error (pixels) are dropped.
MAX_ERROR_PX = 2.0
# Without a marker in this many photos (spread over the set), there are none.
PROBE_PHOTOS = 8
# Edges disagreeing more than this (median relative deviation): a warning.
DISAGREE = 0.02

Array = NDArray[np.float64]


class MarkerError(Exception):
    pass


@dataclass(frozen=True)
class MarkerScale:
    mm_per_unit: float
    markers: int  # markers with all four corners triangulated
    edges: int
    spread: float  # median relative deviation of the edges' estimates
    photos: int  # photos a marker was found in
    edge: tuple[Vector, Vector]  # the edge closest to the median, in model coordinates

    def describe(self, size_mm: float) -> str:
        return (
            f"{self.markers} marker(s) of {size_mm:g} mm in {self.photos} photo(s): "
            f"{self.mm_per_unit:.4g} mm per unit, edges agreeing within "
            f"{100 * self.spread:.1f} %"
        )


# --- the sheet -----------------------------------------------------------------------


def marker_cells(tag_id: int) -> NDArray[np.uint8]:
    """A tag36h11 marker as 10 x 10 cells, 255 white and 0 black (margin included).

    Drawn from the family's own code table and bit layout (the library's
    apriltag_to_image draws a shifted border in this version).
    """
    family = _detector().tag_families[FAMILY].contents
    if not 0 <= tag_id < family.ncodes:
        raise MarkerError(f"no marker {tag_id} in {FAMILY}")
    n, w = family.total_width, family.width_at_border
    cells = np.full((n, n), 255, np.uint8)
    start = (n - w) // 2
    cells[start : start + w, start : start + w] = 0
    code = int(family.codes[tag_id])
    for i in range(family.nbits):
        if code >> (family.nbits - i - 1) & 1:
            cells[start + family.bit_y[i], start + family.bit_x[i]] = 255
    return cells


def sheet_svg(size_mm: float = DEFAULT_SIZE_MM) -> str:
    """An A4 page of markers round the edge, at real size (print at 100 %)."""
    if not 10 <= size_mm <= PITCH_MM * 0.8:
        raise MarkerError(f"marker size {size_mm:g} mm: choose 10 to {PITCH_MM * 0.8:g} mm")
    cell = size_mm / 8
    columns, rows = GRID
    left = (PAGE_MM[0] - columns * PITCH_MM) / 2
    top = PAGE_MM[1] - rows * PITCH_MM - 20
    places = [
        (c, r) for r in range(rows) for c in range(columns)
        if r in (0, rows - 1) or c in (0, columns - 1)
    ]  # fmt: skip
    shapes = []
    for tag_id, (c, r) in zip(SHEET_IDS, places, strict=True):
        x0 = left + c * PITCH_MM + (PITCH_MM - 10 * cell) / 2
        y0 = top + r * PITCH_MM + (PITCH_MM - 10 * cell) / 2
        cells = marker_cells(tag_id)
        for y in range(10):
            x = 0
            while x < 10:  # runs of black cells: one rectangle each
                if cells[y, x]:
                    x += 1
                    continue
                end = x
                while end < 10 and not cells[y, end]:
                    end += 1
                shapes.append(
                    f'<rect x="{x0 + x * cell:.3f}" y="{y0 + y * cell:.3f}" '
                    f'width="{(end - x) * cell:.3f}" height="{cell:.3f}"/>'
                )
                x = end
        shapes.append(
            f'<text x="{x0 + 5 * cell:.2f}" y="{y0 + 10 * cell + 3:.2f}" font-size="2.5" '
            f'text-anchor="middle" fill="#888">{tag_id}</text>'
        )
    ruler_y = 38.0
    text = (
        f'<text x="15" y="14" font-size="6" font-weight="bold">EZ2DIGITIZE scale markers</text>'
        f'<text x="15" y="21" font-size="3.4">Print at 100 % (no "fit to page"): the black '
        f"squares should measure {size_mm:g} mm and the line below 100 mm.</text>"
        f'<text x="15" y="25.5" font-size="3.4">If they don\'t, measure a black square and '
        f"give that as the marker size.</text>"
        f'<text x="15" y="30" font-size="3.4">Put the object in the middle, and keep the '
        f"markers flat and in view of most photos.</text>"
        f'<line x1="15" y1="{ruler_y}" x2="115" y2="{ruler_y}" stroke="#000" stroke-width="0.4"/>'
        f'<line x1="15" y1="{ruler_y - 2}" x2="15" y2="{ruler_y + 2}" stroke="#000" '
        f'stroke-width="0.4"/>'
        f'<line x1="115" y1="{ruler_y - 2}" x2="115" y2="{ruler_y + 2}" stroke="#000" '
        f'stroke-width="0.4"/>'
        f'<text x="120" y="{ruler_y + 1.2}" font-size="3.4">100 mm</text>'
    )
    w, h = PAGE_MM
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:g}mm" height="{h:g}mm" '
        f'viewBox="0 0 {w:g} {h:g}" font-family="sans-serif">'
        f'<rect width="{w:g}" height="{h:g}" fill="#fff"/>{text}'
        f'<g fill="#000" shape-rendering="crispEdges">{"".join(shapes)}</g></svg>\n'
    )


# --- finding them ------------------------------------------------------------------


def find(image: Path) -> dict[int, Array]:
    """The markers in a photo: id to corners (4, 2), in pixels."""
    with Image.open(image) as photo:
        gray = np.asarray(photo.convert("L"))
    found: dict[int, Array] = {}
    for detection in _detector().detect(gray):
        if detection.hamming == 0 and detection.decision_margin >= MIN_MARGIN:
            found[int(detection.tag_id)] = np.asarray(detection.corners, dtype=np.float64)
    return found


def measure(model_dir: Path, images_dir: Path, size_mm: float) -> MarkerScale | None:
    """The scale from the markers in a placed photo set; None if there are none."""
    if not (math.isfinite(size_mm) and size_mm > 0):
        raise MarkerError("the marker size must be more than 0 mm")
    try:
        poses = read_images(model_dir)
        cameras = read_cameras(model_dir)
    except BackendError as exc:
        raise MarkerError(f"no camera placement to measure in: {exc}") from exc
    projections: dict[str, Array] = {}
    for name, pose in poses.items():
        camera = cameras.get(pose.camera_id)
        if camera is not None:
            projections[name] = _projection(camera.model, camera.params, pose.qvec, pose.tvec)
    names = sorted(n for n in projections if (images_dir / n).is_file())
    seen: dict[str, dict[int, Array]] = {}
    for name in _probe_first(names):
        seen[name] = _find_quietly(images_dir / name)
        if len(seen) == min(PROBE_PHOTOS, len(names)) and not any(seen.values()):
            return None  # none in a spread of photos: no markers in this set
    corners = _triangulate(seen, projections)
    return _scale(corners, size_mm, sum(1 for found in seen.values() if found))


def project_size(project: Project) -> float:
    try:
        value = float(project.settings.get(SIZE_SETTING, DEFAULT_SIZE_MM))
    except (TypeError, ValueError):
        return DEFAULT_SIZE_MM
    return value if math.isfinite(value) and value > 0 else DEFAULT_SIZE_MM


def measure_project(project: Project, size_mm: float | None = None) -> MarkerScale | None:
    """`measure` on the project's undistorted photos (after camera placement)."""
    undistorted = project.stage_dir("undistort")
    size = size_mm if size_mm is not None else project_size(project)
    return measure(undistorted / "sparse", undistorted / "images", size)


# --- geometry ----------------------------------------------------------------------


def _projection(model: str, params: Sequence[float], qvec: Any, tvec: Any) -> Array:
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif model == "SIMPLE_PINHOLE":
        fx, cx, cy = params[:3]
        fy = fx
    else:
        raise MarkerError(f"camera model {model}: expected the undistorted (pinhole) cameras")
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    rt = np.column_stack([np.array(quaternion_matrix(tuple(qvec))), np.asarray(tvec)])
    p: Array = k @ rt
    return p


def _triangulate(
    seen: dict[str, dict[int, Array]], projections: dict[str, Array]
) -> dict[int, list[Array | None]]:
    """Each marker's four corners in model coordinates (None where not enough views)."""
    observations: dict[tuple[int, int], list[tuple[Array, Array]]] = {}
    for name, found in seen.items():
        for tag_id, corners in found.items():
            for k in range(4):
                observations.setdefault((tag_id, k), []).append((projections[name], corners[k]))
    result: dict[int, list[Array | None]] = {}
    for (tag_id, k), views in observations.items():
        result.setdefault(tag_id, [None] * 4)[k] = _robust_point(views)
    return result


def _robust_point(views: list[tuple[Array, Array]]) -> Array | None:
    """DLT triangulation, then again without the views it reprojects badly in."""
    for _ in range(2):
        if len(views) < 2:
            return None
        point = _dlt(views)
        errors = [_reprojection_error(p, uv, point) for p, uv in views]
        good = [v for v, e in zip(views, errors, strict=True) if e <= MAX_ERROR_PX]
        if len(good) == len(views):
            return point
        views = good
    return None


def _dlt(views: list[tuple[Array, Array]]) -> Array:
    rows = []
    for p, (u, v) in views:
        rows.append(u * p[2] - p[0])
        rows.append(v * p[2] - p[1])
    a = np.array(rows)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    _u, _s, vt = np.linalg.svd(a)
    x = vt[-1]
    point: Array = x[:3] / x[3]
    return point


def _reprojection_error(p: Array, uv: Array, point: Array) -> float:
    x = p @ np.append(point, 1.0)
    if x[2] <= 0:
        return math.inf  # behind the camera
    return float(np.hypot(x[0] / x[2] - uv[0], x[1] / x[2] - uv[1]))


def _scale(
    corners: dict[int, list[Array | None]], size_mm: float, photos: int
) -> MarkerScale | None:
    estimates: list[tuple[float, tuple[Array, Array]]] = []
    markers = 0
    for points in corners.values():
        if any(p is None for p in points):
            continue
        markers += 1
        for k in range(4):
            a, b = points[k], points[(k + 1) % 4]
            assert a is not None and b is not None
            length = float(np.linalg.norm(b - a))
            if length > 0:
                estimates.append((size_mm / length, (a, b)))
    if not estimates:
        return None
    values = [e for e, _ in estimates]
    median = statistics.median(values)
    spread = statistics.median(abs(v / median - 1) for v in values)
    _value, (a, b) = min(estimates, key=lambda e: abs(e[0] - median))
    # The edge stretched or shrunk to the median: its two ends give exactly that scale.
    middle, half = (a + b) / 2, (b - a) / 2 * (_value / median)
    edge = (_vector(middle - half), _vector(middle + half))
    return MarkerScale(median, markers, len(values), spread, photos, edge)


def _vector(v: Array) -> Vector:
    return (float(v[0]), float(v[1]), float(v[2]))


def _probe_first(names: Sequence[str]) -> Iterable[str]:
    """`names` with PROBE_PHOTOS spread over the set first."""
    if not names:
        return []
    step = max(1, len(names) // PROBE_PHOTOS)
    probe = list(dict.fromkeys(names[i] for i in range(0, len(names), step)))[:PROBE_PHOTOS]
    return probe + [n for n in names if n not in set(probe)]


def _find_quietly(image: Path) -> dict[int, Array]:
    try:
        return find(image)
    except OSError:
        return {}  # unreadable: the photo checks report it


_DETECTOR: Any = None


def _detector() -> Any:
    global _DETECTOR
    if _DETECTOR is None:
        try:
            from pupil_apriltags import Detector
        except ImportError as exc:  # pragma: no cover - a broken install or package
            raise MarkerError(f"the marker detector is missing: {exc}") from exc

        # Quads found at half resolution (4x faster on 12 MP photos), their edges
        # then refined at full resolution, so the corners keep their accuracy.
        threads = min(4, os.cpu_count() or 1)
        try:
            _DETECTOR = Detector(
                families=FAMILY, nthreads=threads, quad_decimate=2.0, refine_edges=True
            )
        except (RuntimeError, OSError) as exc:  # its C library didn't load
            raise MarkerError(f"the marker detector couldn't load: {exc}") from exc
    return _DETECTOR


# --- in the project ------------------------------------------------------------------


def to_scale(found: MarkerScale, size_mm: float, camera_run: str) -> scale.Scale:
    """The project scale from markers: the median edge, with what it rests on."""
    return scale.make(
        found.edge, size_mm, camera_run, source="markers", detail=found.describe(size_mm)
    )


def auto_scale(project: Project) -> str | None:
    """After camera placement: the scale from the markers, if the photos have any.

    A scale set by hand on this camera placement is kept (the markers are
    compared with it); one from markers isn't measured again. Returns what
    to tell the user, or None (no markers, or nothing new). Advice only:
    problems reading the placement are not errors here.
    """
    run = camera_run(project)
    current = scale.current(project)
    if run is None or (current is not None and current.source == "markers"):
        return None
    size = project_size(project)
    try:
        found = measure_project(project, size)
    except (MarkerError, BackendError, OSError):
        return None
    if found is None:
        return None
    if current is not None:
        differs = current.mm_per_unit / found.mm_per_unit - 1
        return (
            f"scale markers found ({found.describe(size)}); the scale set by hand differs "
            f"by {100 * differs:+.1f} % and is kept"
        )
    scale.save(project, to_scale(found, size, run))
    note = f"scale from {found.describe(size)}: exports are in real units"
    if found.spread > DISAGREE:
        note += "; the markers disagree, check that they lie flat and the printed size"
    return note
