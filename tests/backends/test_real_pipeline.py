# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The whole mesh path on the synthetic scene, with the real pinned backends.

Skipped unless the pinned COLMAP and OpenMVS are found (EZ2D_COLMAP and
EZ2D_OPENMVS_DIR, or PATH). The Backends workflow runs it against the fresh
builds, with EZ2D_TEST_DOWNLOADS=1 for the learned features (their models
are downloaded); locally:

    EZ2D_COLMAP=.../bin/colmap EZ2D_OPENMVS_DIR=.../bin \\
        uv run pytest tests/backends/test_real_pipeline.py
"""

import os
import shutil
from pathlib import Path

import pytest

from ez2digitize.backends import colmap, openmvs
from ez2digitize.backends.common import BackendMissing
from ez2digitize.core import download
from ez2digitize.core.capture import import_folder
from ez2digitize.core.project import Project
from ez2digitize.core.runner import Event, Progress
from ez2digitize.core.stage import StageManifest, StageSpec, run_stage


def _pinned() -> tuple[colmap.Colmap, openmvs.OpenMVS]:
    # CI sets EZ2D_REQUIRE_BACKENDS so a missing or wrong backend fails the
    # job instead of skipping the one test that checks it.
    give_up = pytest.fail if os.environ.get("EZ2D_REQUIRE_BACKENDS") else pytest.skip
    try:
        sfm, mvs = colmap.locate(), openmvs.locate()
    except BackendMissing as exc:
        give_up(str(exc))
    if not (sfm.supported and mvs.supported):
        give_up(f"needs the pinned versions, found COLMAP {sfm.version}, OpenMVS {mvs.version}")
    return sfm, mvs


def test_mesh_path_on_synthetic_scene(tmp_path: Path) -> None:
    sfm, mvs = _pinned()
    synthetic = pytest.importorskip("ez2d_bench.synthetic", reason="needs --group feasibility")
    count = synthetic.generate(tmp_path / "synthetic")

    project = Project.create(tmp_path / "project")
    bundle, skipped = import_folder(project, tmp_path / "synthetic" / "images")
    assert skipped == [] and len(bundle.images) == count
    shutil.copytree(tmp_path / "synthetic" / "masks", project.masks_dir / bundle.id)

    progress: list[Progress] = []

    def run(spec: StageSpec) -> StageManifest:
        def on_event(event: Event) -> None:
            if isinstance(event, Progress):
                progress.append(event)

        manifest = run_stage(project, spec, on_event=on_event).manifest
        log = (project.stage_dir(spec.name) / "log.txt").read_text(errors="replace")
        assert manifest.succeeded, f"{spec.name} failed:\n{log[-3000:]}"
        return manifest

    features = run(
        colmap.extract_features(
            sfm, project, [bundle], masks=project.masks_dir,
            options=colmap.FeatureOptions(max_image_size=1600),
        )
    )  # fmt: skip
    log = (project.stage_dir("features") / "log.txt").read_text(errors="replace")
    assert any(p.fraction == 1.0 for p in progress), (
        "no feature progress parsed; the log starts:\n" + "\n".join(log.splitlines()[:40])
    )
    matching = run(colmap.match_features(sfm, project, features))
    mapping = run(colmap.map_sparse(sfm, project, matching, total_images=count))
    model = colmap.best_model(project.stage_dir("mapping") / "sparse")
    assert model is not None
    assert colmap.registered_images(model) >= count * 0.9

    undistorted = run(colmap.undistort(sfm, project, mapping, model=model))
    imported = run(openmvs.import_colmap(mvs, project, undistorted))
    dense = run(
        openmvs.densify(mvs, project, imported, options=openmvs.DensifyOptions(resolution_level=2))
    )
    mesh = run(openmvs.reconstruct_mesh(mvs, project, dense))
    textured = run(
        openmvs.texture_mesh(mvs, project, dense, mesh, options=openmvs.TextureOptions("obj"))
    )
    texture_dir = project.stage_dir(textured.stage)
    assert (texture_dir / "scene_textured.obj").stat().st_size > 0
    assert list(texture_dir.glob("scene_textured*.png")) or list(texture_dir.glob("*.jpg"))
    assert any("depth-maps" in p.message for p in progress), "no OpenMVS progress parsed"

    # Nothing changed: the first stage is reused, not re-run.
    again = run_stage(
        project,
        colmap.extract_features(
            sfm, project, [bundle], masks=project.masks_dir,
            options=colmap.FeatureOptions(max_image_size=1600),
        ),
    )  # fmt: skip
    assert again.reused


def test_learned_features_on_synthetic_scene(tmp_path: Path) -> None:
    """ALIKED features and LightGlue matching place the cameras (needs ONNX Runtime)."""
    sfm, _mvs = _pinned()
    if not os.environ.get("EZ2D_TEST_DOWNLOADS"):
        pytest.skip("downloads the ALIKED and LightGlue models (49 MB): EZ2D_TEST_DOWNLOADS=1")
    synthetic = pytest.importorskip("ez2d_bench.synthetic", reason="needs --group feasibility")
    count = synthetic.generate(tmp_path / "synthetic")
    project = Project.create(tmp_path / "project")
    bundle, _skipped = import_folder(project, tmp_path / "synthetic" / "images")
    # What colmap.fetch_pinned does (the tests stand it in, to stay offline).
    aliked, lightglue = (
        download.fetch(p.url, p.path, p.sha256, size=p.size)
        for p in (colmap.ALIKED_MODEL, colmap.LIGHTGLUE_MODEL)
    )

    def run(spec: StageSpec) -> StageManifest:
        manifest = run_stage(project, spec).manifest
        log = (project.stage_dir(spec.name) / "log.txt").read_text(errors="replace")
        assert manifest.succeeded, f"{spec.name} failed:\n{log[-3000:]}"
        return manifest

    features = run(
        colmap.extract_features(
            sfm, project, [bundle],
            options=colmap.FeatureOptions(max_image_size=1600, kind="aliked", model=aliked),
        )
    )  # fmt: skip
    matching = run(
        colmap.match_features(
            sfm, project, features,
            options=colmap.MatchOptions(features="aliked", lightglue=lightglue),
        )
    )  # fmt: skip
    run(colmap.map_sparse(sfm, project, matching, total_images=count))
    model = colmap.best_model(project.stage_dir("mapping") / "sparse")
    assert model is not None
    assert colmap.registered_images(model) >= count * 0.9
