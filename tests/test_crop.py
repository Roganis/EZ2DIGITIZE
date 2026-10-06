# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import math
from pathlib import Path

import pytest

from ez2digitize import crop
from ez2digitize.backends.colmap import Colmap
from ez2digitize.core.files import write_json_atomic
from ez2digitize.core.project import Project
from ez2digitize.core.stage import MANIFEST_FILE, StageManifest
from ez2digitize.orientation import Matrix, Vector, rotation_between

BACKEND = Colmap(Path("colmap"), "4.2.1").backend
# A reconstruction whose up is -Y (the usual for COLMAP with level photos).
UPRIGHT = rotation_between((0.0, -1.0, 0.0), (0.0, 1.0, 0.0))


def _mapping(project: Project, run_id: str) -> None:
    folder = project.stage_dir("mapping")
    folder.mkdir(parents=True, exist_ok=True)
    manifest = StageManifest(
        stage="mapping", run_id=run_id, status="succeeded", cache_key="k", backend=BACKEND,
        command=[], parameters={}, inputs={}, started="", finished="", wall_s=1.0,
        cpu_s=1.0, peak_rss_mb=None, exit_code=0, host={},
    )  # fmt: skip
    write_json_atomic(folder / MANIFEST_FILE, manifest.to_dict())


def test_upright_round_trip() -> None:
    upright_box = crop.UprightBox((1.0, 2.0, 3.0), (0.5, 1.0, 1.5), 30.0)
    box = crop.from_upright(upright_box, UPRIGHT, "m1")
    back = crop.to_upright(box, UPRIGHT)
    assert back.centre == pytest.approx(upright_box.centre)
    assert back.half_size == upright_box.half_size and back.yaw == pytest.approx(30.0)
    # In model coordinates the box sits where the upright one does.
    assert box.contains(_model(UPRIGHT, (1.0, 2.0, 3.0)))
    assert box.contains(_model(UPRIGHT, (1.0, 2.9, 3.0)))  # within the half height of 1
    assert not box.contains(_model(UPRIGHT, (1.0, 3.1, 3.0)))


def _model(upright: Matrix, point: Vector) -> Vector:
    # upright is a rotation: model = upright^T · point
    x, y, z = (sum(upright[j][i] * point[j] for j in range(3)) for i in range(3))
    return (x, y, z)


def test_yaw_turns_about_the_vertical() -> None:
    box = crop.from_upright(crop.UprightBox((0, 0, 0), (2.0, 1.0, 0.1), 90.0), None, "m1")
    # Turned a quarter: the long side now runs along Z.
    assert box.contains((0.0, 0.0, 1.9)) and not box.contains((1.9, 0.0, 0.0))
    assert crop.to_upright(box, None).yaw == pytest.approx(90.0)


def test_roi_text_is_what_openmvs_reads() -> None:
    box = crop.CropBox(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (1.0, 2.0, 3.0), (0.5, 0.5, 0.25), "m1")
    rows = box.roi_text().splitlines()
    assert rows == ["1.0 0.0 0.0", "0.0 1.0 0.0", "0.0 0.0 1.0", "1.0 2.0 3.0", "0.5 0.5 0.25"]


def test_stored_with_its_camera_placement(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "p")
    assert crop.current(project) is None and crop.camera_run(project) is None
    _mapping(project, "m1")
    box = crop.from_upright(crop.UprightBox((0, 0, 0), (1, 1, 1)), None, "m1")
    crop.save(project, box)
    reopened = Project.open(project.root)
    assert crop.current(reopened) == box
    # Cameras placed again: other coordinates, the box no longer applies.
    _mapping(project, "m2")
    assert crop.current(reopened) is None and crop.stored(reopened) == box
    crop.save(reopened, None)
    assert crop.stored(Project.open(project.root)) is None


def test_bad_boxes_are_ignored() -> None:
    assert crop.CropBox.from_dict({"rotation": [[1, 0, 0]], "centre": [0, 0, 0]}) is None
    good = {"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "centre": [0, 0, 0]}
    assert crop.CropBox.from_dict({**good, "half_size": [1, 0, 1], "camera_run": "m"}) is None
    assert crop.UprightBox.from_dict({"centre": [0, 0], "half_size": [1, 1, 1]}) is None
    nan = {"centre": [0, 0, math.nan], "half_size": [1, 1, 1]}
    assert crop.UprightBox.from_dict(nan) is None


def test_automatic_box_ignores_stray_points() -> None:
    points = [(float(i % 10), float(i % 7), float(i % 5)) for i in range(1000)]
    points.append((1000.0, 1000.0, 1000.0))  # an outlier far away
    box = crop.automatic(points, None)
    assert box is not None
    assert box.centre == pytest.approx((4.5, 3.0, 2.0))
    assert box.half_size == pytest.approx((4.95, 3.3, 2.2))  # grown by 10 %
    assert crop.automatic([], None) is None
