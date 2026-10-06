# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import math
from pathlib import Path

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
        "placed far from the others, probably wrongly: 1 photo(s). They can add noise; "
        "check them in the photo checks (blurry, or of something else?).",
    )
    named = assess(
        [c for c, _ in cameras], [a for _, a in cameras], UP, [f"c/{i}.jpg" for i in range(49)]
    )
    assert "probably wrongly: 48.jpg." in named.findings[0]


def test_weak_photos_and_name_list(tmp_path: Path) -> None:
    import sqlite3

    from ez2digitize.coverage import name_list, weak_photos

    database = tmp_path / "database.db"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
        db.execute(
            "CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER, config INTEGER)"
        )
        db.executemany(
            "INSERT INTO images VALUES (?, ?)", [(i, f"c/{i}.jpg") for i in (1, 2, 3, 4)]
        )

        def pair(a: int, b: int) -> int:
            return a * 2147483647 + b

        db.executemany(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?)",
            [(pair(1, 2), 100, 2), (pair(2, 3), 100, 3), (pair(1, 3), 80, 2),
             (pair(3, 4), 9, 2), (pair(1, 4), 300, 7)],  # too few; a watermark
        )  # fmt: skip
    assert weak_photos(database) == ["c/4.jpg"]
    assert name_list([f"c/{i}.jpg" for i in range(8)], limit=3) == "0.jpg, 1.jpg, 2.jpg and 5 more"
