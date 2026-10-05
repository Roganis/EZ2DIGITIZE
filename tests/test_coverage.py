# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import math

from ez2digitize.coverage import Coverage, assess
from ez2digitize.orientation import Vector

Camera = tuple[Vector, Vector]

UP: Vector = (0.0, 0.0, 1.0)
OBJECT: Vector = (1.0, 2.0, 0.5)


def ring(count: int, elevation: float, start: float = 0.0, span: float = 360.0) -> list[Camera]:
    """Cameras on a circle around OBJECT, all looking at it."""
    cameras = []
    for i in range(count):
        azimuth = math.radians(start + span * i / count)
        up_angle = math.radians(elevation)
        offset = (
            3 * math.cos(up_angle) * math.cos(azimuth),
            3 * math.cos(up_angle) * math.sin(azimuth),
            3 * math.sin(up_angle),
        )
        centre = (OBJECT[0] + offset[0], OBJECT[1] + offset[1], OBJECT[2] + offset[2])
        cameras.append((centre, (-offset[0], -offset[1], -offset[2])))
    return cameras


def run(cameras: list[Camera]) -> Coverage:
    return assess([c for c, _ in cameras], [a for _, a in cameras], UP)


def test_two_rings_all_around() -> None:
    result = run(ring(24, 10) + ring(24, 40, start=7))
    assert result.findings == ()
    assert result.largest_gap_degrees is not None and result.largest_gap_degrees < 20
    assert result.elevation_degrees == (10.0, 40.0)


def test_gap() -> None:
    result = run(ring(12, 10, span=180) + ring(12, 40, span=180))
    assert any("way around" in f for f in result.findings)


def test_single_ring() -> None:
    (finding,) = run(ring(30, 20)).findings
    assert "same height" in finding


def test_still_camera() -> None:
    # A turntable without masks: every camera in the same place.
    still: list[Camera] = [((5.0, 0.0, 1.0 + 0.001 * i), (-1.0, 0.0, 0.0)) for i in range(20)]
    (finding,) = run(still).findings
    assert "hardly moved" in finding


def test_misplaced_camera_is_left_out() -> None:
    cameras = ring(24, 10) + ring(24, 40, start=7)
    cameras.append(((500.0, 400.0, 300.0), (0.0, 1.0, 0.0)))
    result = run(cameras)
    assert result.cameras == 48
    assert result.findings == (
        "1 photo(s) were placed far from the others, probably wrongly; they can add "
        "noise. Check them in the photo checks (blurry, or of something else?).",
    )
