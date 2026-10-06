# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import math
from pathlib import Path

import pytest

from ez2digitize import coverage
from ez2digitize.coverage import Coverage, RingLayout, _rotate, assess, layout
from ez2digitize.orientation import Vector, rotation_between

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


# --- the rings, for the 3D view ---------------------------------------------------

Z_UP_TO_Y_UP = rotation_between(UP, (0.0, 1.0, 0.0))


def layout_of(cameras: list[Camera], names: list[str] | None = None) -> RingLayout | None:
    names = names or [f"c/{i}.jpg" for i in range(len(cameras))]
    return layout(names, [c for c, _ in cameras], [a for _, a in cameras], Z_UP_TO_Y_UP)


def test_rings_by_height() -> None:
    result = layout_of(ring(24, 10) + ring(18, 40, start=7))
    assert result is not None
    assert [(r.elevation, r.cameras) for r in result.rings] == [(10.0, 24), (40.0, 18)]
    low, high = result.rings
    assert low.radius == pytest.approx(3 * math.cos(math.radians(10)))
    assert high.height == pytest.approx(3 * math.sin(math.radians(40)))
    assert low.gaps == () and high.gaps == ()
    # The centre is the object's, in the upright frame (Z up became Y up).
    assert result.centre == pytest.approx(_rotate(Z_UP_TO_Y_UP, OBJECT))
    assert result.largest_gap == pytest.approx(15, abs=0.2)
    assert coverage.describe(result) == (
        "Photos by height: 24 at 10°, 18 at 40°. No gaps in the rings."
    )


def test_gaps_in_a_ring() -> None:
    # The low ring all round; the high one only half way, two photos missing.
    high = ring(12, 45, span=180)
    del high[4:6]
    result = layout_of(ring(24, 10) + high)
    assert result is not None
    low, top = result.rings
    assert low.gaps == ()
    assert [g.degrees for g in top.gaps] == pytest.approx([45.0, 195.0], abs=0.2)
    assert result.largest_gap < 90  # the low ring covers all around
    text = coverage.describe(result, weak=2)
    assert "Gaps in the rings (orange, red over 90°): 45° at 45°, 195° at 45°." in text
    assert "2 photo(s) with few matches" in text
    assert "Nothing from" not in text


def test_a_side_missing_and_a_low_top() -> None:
    result = layout_of(ring(12, 15, span=180))
    assert result is not None
    (only,) = result.rings
    assert only.gaps[0].degrees == pytest.approx(195, abs=0.2)
    text = coverage.describe(result)
    assert "Nothing from 195° of the way around" in text
    assert "The top is seen only from low down" in text


def test_stray_photos_join_a_ring() -> None:
    # Two photos from much higher are not a ring of their own.
    result = layout_of(ring(20, 10) + ring(2, 70))
    assert result is not None
    assert [r.cameras for r in result.rings] == [22]


def test_misplaced_cameras_are_named() -> None:
    cameras = ring(24, 10) + ring(24, 40, start=7)
    cameras.append(((500.0, 400.0, 300.0), (0.0, 1.0, 0.0)))
    result = layout_of(cameras)
    assert result is not None
    assert result.far == ("c/48.jpg",)
    assert sum(r.cameras for r in result.rings) == 48
    assert "1 placed far off (red cameras)." in coverage.describe(result)


def test_no_rings_without_up_or_movement(tmp_path: Path) -> None:
    assert coverage.rings(tmp_path, None) is None
    still: list[Camera] = [((5.0, 0.0, 1.0 + 0.001 * i), (-1.0, 0.0, 0.0)) for i in range(20)]
    assert layout_of(still) is None
    assert layout_of(ring(4, 10)) is None  # too few to say
    assert coverage.describe(None) == ""


def test_ring_layout_for_the_page() -> None:
    result = layout_of(ring(12, 15, span=180))
    assert result is not None
    data = result.to_dict()
    assert data["max_gap"] == 90.0
    shown_rings = data["rings"]
    assert isinstance(shown_rings, list)
    (shown,) = shown_rings
    assert shown["cameras"] == 12 and shown["elevation"] == 15.0
    assert shown["gaps"] == [{"start": result.rings[0].gaps[0].start, "degrees": 195.0}]
