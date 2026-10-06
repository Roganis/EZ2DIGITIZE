# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import pytest

from ez2digitize import scale
from ez2digitize.core.files import write_json_atomic
from ez2digitize.core.project import Project
from ez2digitize.core.stage import MANIFEST_FILE, Backend, StageManifest
from ez2digitize.orientation import rotation_between

# A reconstruction whose up is -Y (the usual for COLMAP with level photos).
UPRIGHT = rotation_between((0.0, -1.0, 0.0), (0.0, 1.0, 0.0))


def _mapping(project: Project, run_id: str) -> None:
    folder = project.stage_dir("mapping")
    folder.mkdir(parents=True, exist_ok=True)
    manifest = StageManifest(
        stage="mapping", run_id=run_id, status="succeeded", cache_key="k",
        backend=Backend("colmap", "4.2.1"), command=[], parameters={}, inputs={},
        started="", finished="", wall_s=1.0, cpu_s=1.0, peak_rss_mb=None, exit_code=0, host={},
    )  # fmt: skip
    write_json_atomic(folder / MANIFEST_FILE, manifest.to_dict())


def test_mm_per_unit() -> None:
    picked = scale.make(((1.0, 0.0, 0.0), (1.0, 3.0, 4.0)), 25.0, "m1")
    assert picked.model_distance == 5.0 and picked.mm_per_unit == 5.0
    assert scale.describe(picked) == "25 mm between the two points (5 mm per unit)"


@pytest.mark.parametrize(
    ("points", "distance", "message"),
    [
        (((0, 0, 0), (0, 0, 0)), 10.0, "two different points"),
        (((0, 0, 0), (1, 0, 0)), 0.0, "more than 0 mm"),
        (((0, 0, 0), (1, 0, 0)), float("nan"), "more than 0 mm"),
        (((0, 0, float("inf")), (1, 0, 0)), 10.0, "pick two points"),
    ],
)
def test_unusable_scales(points: object, distance: float, message: str) -> None:
    with pytest.raises(scale.ScaleError, match=message):
        scale.make(points, distance, "m1")  # type: ignore[arg-type]


def test_upright_round_trip() -> None:
    picked = ((1.0, 2.0, 3.0), (-1.0, 0.5, 2.0))
    model = scale.from_upright(picked, UPRIGHT)
    assert model[0][1] == pytest.approx(-2.0)  # up is -Y in the reconstruction
    back = scale.to_upright(model, UPRIGHT)
    assert back[0] == pytest.approx(picked[0]) and back[1] == pytest.approx(picked[1])
    assert scale.from_upright(picked, None) == picked


def test_stored_with_its_camera_placement(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "p")
    _mapping(project, "m1")
    picked = scale.make(((0, 0, 0), (0, 0, 2)), 30.0, "m1")
    scale.save(project, picked)
    reopened = Project.open(project.root)
    assert scale.current(reopened) == picked
    # Cameras placed again: other coordinates and size, the scale no longer applies.
    _mapping(project, "m2")
    assert scale.current(reopened) is None and scale.stored(reopened) == picked
    scale.save(reopened, None)
    assert scale.stored(Project.open(project.root)) is None


def test_bad_stored_scales_are_ignored() -> None:
    assert scale.Scale.from_dict(None) is None
    assert (
        scale.Scale.from_dict({"points": [[0, 0, 0]], "distance_mm": 1, "camera_run": "m"}) is None
    )
    same = {"points": [[0, 0, 0], [0, 0, 0]], "distance_mm": 1, "camera_run": "m"}
    assert scale.Scale.from_dict(same) is None
