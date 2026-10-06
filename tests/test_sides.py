# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import pytest
from PIL import Image

from ez2digitize import sides
from ez2digitize.core.capture import CaptureBundle, import_files, list_bundles
from ez2digitize.core.project import Project


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "project")


def _capture(project: Project, folder: Path, *names: str, flipped: bool = False) -> CaptureBundle:
    folder.mkdir(parents=True)
    for n, name in enumerate(names):
        Image.new("RGB", (8, 6), (n * 40, 0, 0)).save(folder / name)
    return import_files(project, [folder / n for n in names], source="folder", flipped=flipped)


def _mask(project: Project, bundle: CaptureBundle, name: str) -> None:
    (project.masks_dir / bundle.id).mkdir(parents=True, exist_ok=True)
    (project.masks_dir / bundle.id / f"{name}.png").write_bytes(b"m")


def test_one_sided_projects_are_left_alone(project: Project, tmp_path: Path) -> None:
    _capture(project, tmp_path / "a", "1.jpg", "2.jpg")
    bundles = list_bundles(project)
    assert sides.sides(bundles) is None
    assert sides.upright_names(bundles) is None
    assert sides.check(project, bundles, use_masks=False) == []


def test_two_sides_and_what_is_missing(project: Project, tmp_path: Path) -> None:
    top = _capture(project, tmp_path / "a", "1.jpg", "2.jpg")
    under = _capture(project, tmp_path / "b", "3.jpg", flipped=True)
    under.set_excluded([])  # saving keeps the flag
    bundles = list_bundles(project)
    found = sides.sides(bundles)
    assert found == sides.Sides([f"{top.id}/1.jpg", f"{top.id}/2.jpg"], [f"{under.id}/3.jpg"])
    assert found.complete
    assert sides.upright_names(bundles) == {f"{top.id}/1.jpg", f"{top.id}/2.jpg"}

    problems = sides.check(project, bundles, use_masks=True)
    assert len(problems) == 1 and problems[0].startswith("3 of 3 photos have no mask")
    for bundle, name in ((top, "1.jpg"), (top, "2.jpg"), (under, "3.jpg")):
        _mask(project, bundle, name)
    assert sides.check(project, bundles, use_masks=True) == []
    assert "masks are switched off" in sides.check(project, bundles, use_masks=False)[0]

    # Left-out photos don't count; neither does a side without photos.
    top.set_excluded(["1.jpg", "2.jpg"])
    problems = sides.check(project, list_bundles(project), use_masks=True)
    assert problems[0].startswith("every capture is marked as turned over")
    assert sides.upright_names(list_bundles(project)) is None


def test_turned_side_without_photos(project: Project, tmp_path: Path) -> None:
    _capture(project, tmp_path / "a", "1.jpg")
    under = _capture(project, tmp_path / "b", "3.jpg", flipped=True)
    under.set_excluded(["3.jpg"])
    problems = sides.check(project, list_bundles(project), use_masks=False)
    assert "the turned-over side has no photos yet" in problems


def test_joined() -> None:
    found = sides.Sides(["t/1", "t/2"], ["u/3", "u/4"])
    both = sides.joined(found, ["t/1", "t/2", "u/3"])
    assert both == sides.Joined(2, 2, 1, 2) and both.ok
    assert sides.join_notice(both).startswith("both sides joined: 2 of 2 photos of the first")
    one = sides.joined(found, ["t/1", "t/2"])
    assert not one.ok
    assert "did not join" in sides.join_notice(one)
