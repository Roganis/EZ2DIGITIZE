# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import pytest

from ez2digitize import crop, upright
from ez2digitize.core.files import write_json_atomic
from ez2digitize.core.project import Project
from ez2digitize.core.stage import MANIFEST_FILE, Backend, StageManifest
from ez2digitize.orientation import IDENTITY, Matrix, Vector

START = upright.Orientation(IDENTITY, 0.0, "m1")


def _mapping(project: Project, run_id: str) -> None:
    folder = project.stage_dir("mapping")
    folder.mkdir(parents=True, exist_ok=True)
    manifest = StageManifest(
        stage="mapping", run_id=run_id, status="succeeded", cache_key="k",
        backend=Backend("colmap", "4.2.1"), command=[], parameters={}, inputs={},
        started="", finished="", wall_s=1.0, cpu_s=1.0, peak_rss_mb=None, exit_code=0, host={},
    )  # fmt: skip
    write_json_atomic(folder / MANIFEST_FILE, manifest.to_dict())


def _flat(m: Matrix) -> list[float]:
    return [v for row in m for v in row]


@pytest.mark.parametrize(("cameras", "up"), [((0, 0, 5), (0, 0, 1)), ((0, 0, -5), (0, 0, -1))])
def test_level_from_three_points(cameras: Vector, up: Vector) -> None:
    # A mat in the z = 0 plane; up is the side the cameras are on.
    points = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    turned = upright.turned(START, 30.0)
    level = upright.levelled(turned, points, cameras)
    assert upright.mul(level.base, up) == pytest.approx((0.0, 1.0, 0.0))
    assert level.turn == 30.0  # which way it faces is kept
    with pytest.raises(upright.OrientationError, match="not in a line"):
        upright.levelled(START, ((0, 0, 0), (1, 1, 1), (2, 2, 2)), cameras)


def test_tilts_are_quarter_turns_about_the_axis_shown() -> None:
    on_its_side = upright.tilted(START, "x")
    # A quarter turn about X: what pointed along -Z now points up.
    assert upright.mul(on_its_side.rotation, (0.0, 0.0, -1.0)) == pytest.approx((0.0, 1.0, 0.0))
    assert on_its_side.base == ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))  # exact
    # With a turn, the tilt is still about the X axis as seen.
    facing = upright.turned(START, 30.0)
    tilted = upright.tilted(facing, "z", -90.0)
    expected = upright.matmul(upright._axis_matrix("z", -90.0), facing.rotation)
    assert _flat(tilted.rotation) == pytest.approx(_flat(expected), abs=1e-9)
    assert tilted.turn == 30.0
    with pytest.raises(ValueError, match="axis"):
        upright.tilted(START, "w")


def test_turn_wraps() -> None:
    assert upright.turned(START, 190.0).turn == -170.0
    assert upright.turned(START, -180.0).turn == -180.0


def test_stored_with_its_camera_placement(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "p")
    assert upright.rotation(project) is None  # nothing placed: no estimate either
    with pytest.raises(upright.OrientationError, match="place the cameras"):
        upright.starting_point(project)
    _mapping(project, "m1")
    start = upright.starting_point(project)
    assert start == START  # no estimate: the reconstruction's frame
    upright.change(project, upright.tilted(start, "x"))
    reopened = Project.open(project.root)
    manual = upright.current(reopened)
    assert manual is not None and upright.rotation(reopened) == manual.rotation
    _mapping(project, "m2")  # placed again: other coordinates
    assert upright.current(reopened) is None and upright.rotation(reopened) is None
    upright.change(reopened, None)
    assert upright.stored(Project.open(project.root)) is None


def test_bad_stored_orientations_are_ignored() -> None:
    assert upright.Orientation.from_dict(None) is None
    squashed = {"base": [[2, 0, 0], [0, 1, 0], [0, 0, 1]], "turn": 0, "camera_run": "m"}
    assert upright.Orientation.from_dict(squashed) is None
    mirrored = {"base": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]], "turn": 0, "camera_run": "m"}
    assert upright.Orientation.from_dict(mirrored) is None
    good = {"base": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "turn": 15, "camera_run": "m"}
    assert upright.Orientation.from_dict(good) == upright.Orientation(IDENTITY, 15.0, "m")


def test_crop_box_is_refitted_level(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "p")
    _mapping(project, "m1")
    box = crop.from_upright(crop.UprightBox((0.0, 1.0, 0.0), (2.0, 1.0, 0.5)), None, "m1")
    crop.save(project, box)
    upright.change(project, upright.tilted(upright.starting_point(project), "x"))
    new = crop.current(project)
    assert new is not None
    level = crop.to_upright(new, upright.rotation(project))
    # Tilted a quarter turn about X: the box's height and depth swap.
    assert level.yaw == pytest.approx(0.0) and level.half_size == pytest.approx((2.0, 0.5, 1.0))
    for corner in ((2.0, 2.0, 0.5), (-2.0, 0.0, -0.5)):  # still holds the old box
        assert new.contains(_shrunk(corner))


def _shrunk(p: Vector) -> Vector:
    return (p[0] * 0.999, 1.0 + (p[1] - 1.0) * 0.999, p[2] * 0.999)
