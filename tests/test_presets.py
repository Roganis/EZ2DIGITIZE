# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

from ez2digitize import presets, subject
from ez2digitize.core.project import Project


def test_presets_get_slower_and_finer() -> None:
    fast, balanced, high = (presets.mesh_settings(q) for q in presets.QUALITIES)
    levels = [s.densify.resolution_level for s in (fast, balanced, high)]
    assert levels == [2, 1, 0]
    assert fast.features.max_image_size < balanced.features.max_image_size
    assert fast.refine is None and balanced.refine is None and high.refine is not None
    assert presets.mesh_settings() == balanced


def test_overrides() -> None:
    settings = presets.mesh_settings("fast", level=0, refine=True, max_image_size=2000)
    assert settings.densify.resolution_level == 0
    assert settings.features.max_image_size == 2000
    # Refining at full size needs far more memory for little gain.
    assert settings.refine is not None and settings.refine.resolution_level == 1
    assert presets.mesh_settings("high", refine=False).refine is None
    assert presets.mesh_settings("balanced", refine=True).refine is not None


def test_describe_and_parse() -> None:
    rows = dict(presets.describe(presets.mesh_settings("high")))
    assert rows["Dense point cloud"].startswith("full size photos")
    assert rows["Refine mesh"] == "yes, 1/2 size"
    assert presets.parse_quality("high") == "high"
    assert presets.parse_quality(None) == presets.parse_quality("ultra") == "balanced"


def test_learned_features() -> None:
    settings = presets.mesh_settings("fast", features="aliked")
    assert settings.features.kind == "aliked"
    rows = dict(presets.describe(settings))
    assert rows["Features"] == "ALIKED + LightGlue (learned)"
    assert rows["Features per photo"] == "up to 1024"  # a quarter of SIFT's
    assert presets.mesh_settings().features.kind == "sift"


def test_scene_subject(tmp_path: Path) -> None:
    obj, scene = presets.mesh_settings(), presets.mesh_settings("high", subject="scene")
    assert obj.subject == "object" and obj.use_masks and not obj.mesh.free_space_support
    assert scene.subject == "scene" and not scene.use_masks and scene.mesh.free_space_support
    assert scene.densify == presets.mesh_settings("high").densify  # the quality still decides
    assert dict(presets.describe(scene))["Subject"] == "Room or outdoor scene"

    project = Project.create(tmp_path / "p")
    assert subject.of(project) == "object"
    subject.store(project, "scene")
    assert subject.of(Project.open(project.root)) == "scene"
    assert subject.parse("garden") == "object"
