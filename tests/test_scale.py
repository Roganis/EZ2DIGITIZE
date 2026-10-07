# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import math
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


def _orbit(
    count: int, radius_m: float, rise_m: float = 0.1
) -> dict[str, tuple[float, float, float]]:
    """Camera centres on a ring, at three heights `rise_m` apart, in metres."""
    return {
        f"c/{k:02d}.jpg": (
            radius_m * math.cos(2 * math.pi * k / count),
            rise_m * (k % 3),
            radius_m * math.sin(2 * math.pi * k / count),
        )
        for k in range(count)
    }


def _placed(
    tracked: dict[str, tuple[float, float, float]], units_per_m: float
) -> dict[str, tuple[float, float, float]]:
    """The same cameras as COLMAP might place them: turned, moved and scaled."""
    out = {}
    for name, (x, y, z) in tracked.items():
        out[name] = (units_per_m * -z + 3.0, units_per_m * y - 1.0, units_per_m * x + 0.5)
    return out


def test_measure_tracking() -> None:
    tracked = _orbit(12, 0.4)
    measured = scale.measure_tracking(_placed(tracked, 2.5), tracked)
    assert measured is not None and measured.cameras == 12
    assert measured.metres_per_unit == pytest.approx(0.4) and measured.spread < 1e-9

    # Drift on a few cameras moves the median little.
    drifted = dict(tracked)
    for name in list(drifted)[:3]:
        x, y, z = drifted[name]
        drifted[name] = (x + 0.05, y, z - 0.04)
    measured = scale.measure_tracking(_placed(tracked, 2.5), drifted)
    assert measured is not None and measured.metres_per_unit == pytest.approx(0.4, rel=0.03)

    few = dict(list(tracked.items())[: scale.MIN_TRACKED - 1])
    assert scale.measure_tracking(_placed(tracked, 2.5), few) is None
    still = _orbit(12, 0.05, rise_m=0.01)  # a phone over a turntable: hardly moved
    assert scale.measure_tracking(_placed(still, 2.5), still) is None


def test_scale_from_tracking(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "p")
    tracked = _orbit(12, 0.4)
    measured = scale.measure_tracking(_placed(tracked, 2.5), tracked)
    assert measured is not None
    assert scale.from_tracking(project, measured) is None  # no camera placement yet
    _mapping(project, "m1")
    note = scale.from_tracking(project, measured)
    assert note is not None and note.startswith("scale from 12 camera positions tracked")
    current = scale.current(project)
    assert current is not None and current.source == "tracking"
    assert current.mm_per_unit == pytest.approx(400.0)
    assert scale.describe(current).startswith("from 12 camera positions")

    # A scale set by hand wins, and the tracking is compared with it.
    scale.save(project, scale.make(((0, 0, 0), (1, 0, 0)), 380.0, "m1"))
    note = scale.from_tracking(project, measured)
    assert note is not None and "+5.3 % off the one set by hand" in note
    assert scale.current(project) == scale.make(((0, 0, 0), (1, 0, 0)), 380.0, "m1")


def test_pipeline_scale_from_tracked_photos(tmp_path: Path) -> None:
    """Photos whose motion log tracked them in metres scale the camera placement."""
    from models import ring, write_images

    from ez2digitize import pipeline
    from ez2digitize.core.capture import CaptureBundle, CaptureFile

    project = Project.create(tmp_path / "p")
    _mapping(project, "m1")
    centres = ring(12, 20.0, radius=3.0)  # COLMAP's units
    model = tmp_path / "model"
    model.mkdir()
    write_images(model / "images.bin", centres)

    def bundle(metric: bool) -> CaptureBundle:
        files = []
        for i, (x, y, z) in enumerate(centres, start=1):
            pose = [[1, 0, 0, 0.1 * x], [0, 1, 0, 0.1 * y], [0, 0, 1, 0.1 * z]]  # 0.1 m a unit
            entry = {"camera_to_world": pose, **({"metric": True} if metric else {})}
            files.append(
                CaptureFile(f"{i:03d}.jpg", f"{i:03d}.jpg", "image", 0, "", {"motion": entry})
            )
        return CaptureBundle(tmp_path, "c", "android", "", files)

    assert pipeline._tracking_scale(project, model, [bundle(metric=False)]) is None
    note = pipeline._tracking_scale(project, model, [bundle(metric=True)])
    assert note is not None and note.startswith("scale from 12 camera positions tracked")
    current = scale.current(project)
    assert current is not None and current.mm_per_unit == pytest.approx(100.0)
