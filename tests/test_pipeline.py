# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pipeline logic with fake backends that write the files the real ones do."""

import os
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from ez2digitize import pipeline
from ez2digitize.backends.brush import Brush
from ez2digitize.core.capture import import_files, list_bundles
from ez2digitize.core.project import Project
from ez2digitize.core.runner import CancelToken, Output, Progress
from ez2digitize.pipeline import (
    MeshSettings,
    Notice,
    PipelineBusy,
    PipelineCancelled,
    PipelineError,
    PipelineEvent,
    StageFailed,
    StageFinished,
    StageOutput,
    StageStarted,
    Tools,
)


@pytest.fixture
def project(tmp_path: Path) -> Project:
    project = Project.create(tmp_path / "project")
    files = []
    for n, name in enumerate(("a.jpg", "b.jpg", "c.jpg")):
        _jpeg(tmp_path / name, n)
        files.append(tmp_path / name)
    import_files(project, files, source="folder")
    return project


def _jpeg(path: Path, shade: int) -> None:
    # Real images: the features stage reads the size of those without a mask.
    Image.new("RGB", (8, 6), (shade * 40, 0, 0)).save(path)


@pytest.fixture
def tools(fake_tools: Tools) -> Tools:
    return fake_tools


def _collect() -> tuple[list[PipelineEvent], pipeline.PipelineHandler]:
    events: list[PipelineEvent] = []
    return events, events.append


def test_run_mesh_runs_every_stage_in_order(project: Project, tools: Tools) -> None:
    events, handler = _collect()
    result = pipeline.run_mesh(project, tools, on_event=handler)

    started = [e for e in events if isinstance(e, StageStarted)]
    unmasked = [s for s in pipeline.STAGES if s not in ("refine", "mask-undistort")]
    assert [e.stage for e in started] == unmasked
    assert [(e.index, e.count) for e in started][:2] == [(1, 8), (2, 8)]
    assert result.sparse.registered_images == 3 and result.sparse.total_images == 3
    assert [p.name for p in result.files] == ["scene_textured.ply", "scene_textured0.png"]
    notices = [e.message for e in events if isinstance(e, Notice)]
    assert notices[:2] == ["3 of 3 images registered (model 0)", "exporting OBJ, GLB"]
    assert notices[2].startswith(f"exported to {project.exports_dir}")
    assert any(isinstance(e, StageOutput) and isinstance(e.event, Output) for e in events)
    names = sorted(p.name for p in result.exports)
    assert names == ["project.glb", "project.mtl", "project.obj", "project_texture0.png"]

    # Unchanged: everything reused, including the export.
    events, handler = _collect()
    again = pipeline.run_mesh(project, tools, on_event=handler)
    finished = [e for e in events if isinstance(e, StageFinished)]
    assert len(finished) == 8 and all(e.reused for e in finished)
    assert again.exports == result.exports
    assert len(list(project.exports_dir.iterdir())) == 1


def test_export_can_be_skipped_and_needs_ply(project: Project, tools: Tools) -> None:
    from ez2digitize.backends.openmvs import TextureOptions

    result = pipeline.run_mesh(project, tools, MeshSettings(export_formats=()))
    assert result.exports == [] and not any(project.exports_dir.iterdir())
    with pytest.raises(PipelineError, match="PLY output"):
        pipeline.run_mesh(project, tools, MeshSettings(texture=TextureOptions("glb")))


def test_refine_is_optional(project: Project, tools: Tools) -> None:
    from ez2digitize.backends.openmvs import RefineOptions

    events, handler = _collect()
    settings = MeshSettings(refine=RefineOptions())
    pipeline.run_mesh(project, tools, settings, on_event=handler)
    stages = [e.stage for e in events if isinstance(e, StageStarted)]
    assert stages == [s for s in pipeline.STAGES if s != "mask-undistort"]
    texture_cmd = (project.stage_dir("texture") / "log.txt").read_text()
    assert "scene_refined.ply" in texture_cmd


def test_sparse_then_dense(project: Project, tools: Tools) -> None:
    with pytest.raises(PipelineError, match="sparse reconstruction first"):
        pipeline.run_dense(project, tools)
    sparse = pipeline.run_sparse(project, tools)
    assert sparse.model == project.stage_dir("mapping") / "sparse" / "0"
    events, handler = _collect()
    mesh = pipeline.run_dense(project, tools, on_event=handler)
    assert mesh.sparse.model == sparse.model
    assert [e.stage for e in events if isinstance(e, StageStarted)][0] == "mvs-import"


def test_force_from_reruns_that_stage_and_later(project: Project, tools: Tools) -> None:
    pipeline.run_mesh(project, tools)
    events, handler = _collect()
    pipeline.run_mesh(project, tools, on_event=handler, force_from="mapping")
    reran = [e.stage for e in events if isinstance(e, StageFinished) and not e.reused]
    assert reran == ["mapping", "undistort", "mvs-import", "densify", "mesh", "texture"]
    with pytest.raises(PipelineError, match="unknown stage"):
        pipeline.run_mesh(project, tools, force_from="nope")


def test_best_of_several_models_and_notices(
    project: Project, tools: Tools, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODELS", "1,2")
    events, handler = _collect()
    sparse = pipeline.run_sparse(project, tools, on_event=handler)
    assert sparse.model.name == "1" and sparse.registered_images == 2
    notices = " | ".join(e.message for e in events if isinstance(e, Notice))
    assert "2 separate groups" in notices
    assert "only" not in notices  # 2 of 3 is above the threshold

    monkeypatch.setenv("FAKE_MODELS", "1")
    events, handler = _collect()
    pipeline.run_sparse(project, tools, on_event=handler, force_from="mapping")
    assert any("only 1 of 3" in e.message for e in events if isinstance(e, Notice))


def test_no_model(project: Project, tools: Tools, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MODELS", "none")
    with pytest.raises(PipelineError, match="could not reconstruct any cameras"):
        pipeline.run_sparse(project, tools)


def test_stage_failure(project: Project, tools: Tools, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_FAIL", "exhaustive_matcher")
    with pytest.raises(StageFailed) as failure:
        pipeline.run_mesh(project, tools)
    assert failure.value.manifest.stage == "matching"
    assert "something went wrong" in failure.value.tail
    assert failure.value.log == project.stage_dir("matching") / "log.txt"
    assert not project.stage_dir("mapping").exists()


def test_no_captures(tmp_path: Path, tools: Tools) -> None:
    with pytest.raises(PipelineError, match="no captures"):
        pipeline.run_sparse(Project.create(tmp_path / "empty"), tools)


def test_video_only_capture_needs_frames(tmp_path: Path, tools: Tools) -> None:
    project = Project.create(tmp_path / "video")
    (tmp_path / "clip.mp4").write_bytes(b"video")
    import_files(project, [tmp_path / "clip.mp4"], source="video")
    with pytest.raises(PipelineError, match="no images"):
        pipeline.run_sparse(project, tools)


def test_matching_mode_follows_capture_source_and_size(
    project: Project, tools: Tools, monkeypatch: pytest.MonkeyPatch
) -> None:
    def matcher() -> str:
        pipeline.run_sparse(project, tools)
        log = (project.stage_dir("matching") / "log.txt").read_text()
        return "sequential" if "sequential_matcher" in log else "exhaustive"

    assert matcher() == "exhaustive"
    for bundle in list_bundles(project):
        bundle.source = "video"
        bundle.save()
    assert matcher() == "exhaustive"  # few frames: every pair, to close the loop
    monkeypatch.setattr(pipeline, "EXHAUSTIVE_MAX_IMAGES", 2)
    assert matcher() == "sequential"


def test_matching_many_photos(
    project: Project, tools: Tools, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Beyond EXHAUSTIVE_MAX_IMAGES: by GPS, by the vocabulary tree, else in order."""
    from ez2digitize.backends import colmap
    from ez2digitize.core import photos

    monkeypatch.setattr(pipeline, "EXHAUSTIVE_MAX_IMAGES", 2)

    def matched() -> tuple[str, list[str]]:
        events, handler = _collect()
        pipeline.run_sparse(project, tools, on_event=handler)
        log = (project.stage_dir("matching") / "log.txt").read_text().splitlines()[0]
        return log, [e.message for e in events if isinstance(e, Notice)]

    # No tree (the tests are offline): photos in the order they were taken.
    log, notices = matched()
    assert "sequential_matcher" in log and "--SequentialMatching.loop_detection 0" in log
    assert "downloading COLMAP's vocabulary tree (72 MB, once)" in notices
    assert any(n.startswith("could not download COLMAP's vocabulary tree") for n in notices)
    assert any("taken just before and after it" in n for n in notices)

    tree = colmap.VOCAB_TREES["sift"].path
    tree.parent.mkdir(parents=True, exist_ok=True)
    with tree.open("wb") as f:
        f.truncate(colmap.VOCAB_TREES["sift"].size)  # as if downloaded
    log, notices = matched()
    assert "vocab_tree_matcher" in log and str(tree) in log
    assert "3 photos: each is matched with the most similar ones" in notices

    monkeypatch.setattr(photos, "gps_share", lambda _bundles: 0.95)
    log, notices = matched()
    assert "spatial_matcher" in log
    assert "3 photos with GPS positions: each is matched with its neighbours" in notices

    # Video: frames in order, loops found with the tree.
    for bundle in list_bundles(project):
        bundle.source = "video"
        bundle.save()
    log, _notices = matched()
    assert "sequential_matcher" in log and "--SequentialMatching.loop_detection 1" in log


def test_learned_features(project: Project, tools: Tools) -> None:
    """ALIKED + LightGlue: the models are fetched first (here: already there)."""
    from ez2digitize import presets
    from ez2digitize.backends import colmap

    settings = presets.mesh_settings(features="aliked")
    events, handler = _collect()
    with pytest.raises(PipelineError, match="need the ALIKED and LightGlue models"):
        pipeline.run_sparse(project, tools, settings, on_event=handler)
    notices = [e.message for e in events if isinstance(e, Notice)]
    assert "downloading the ALIKED model (3 MB, once)" in notices

    for pinned in (colmap.ALIKED_MODEL, colmap.LIGHTGLUE_MODEL):
        pinned.path.parent.mkdir(parents=True, exist_ok=True)
        with pinned.path.open("wb") as f:
            f.truncate(pinned.size)  # as if downloaded
    pipeline.run_sparse(project, tools, settings)
    features = (project.stage_dir("features") / "log.txt").read_text()
    matching = (project.stage_dir("matching") / "log.txt").read_text()
    assert "ALIKED_N16ROT" in features and str(colmap.ALIKED_MODEL.path) in features
    assert "ALIKED_LIGHTGLUE" in matching and str(colmap.LIGHTGLUE_MODEL.path) in matching


def test_chosen_vocab_tree_matching_needs_the_tree(project: Project, tools: Tools) -> None:
    from ez2digitize.backends import colmap

    settings = MeshSettings(matching=colmap.MatchOptions(mode="vocab_tree"))
    with pytest.raises(PipelineError, match="needs COLMAP's vocabulary tree"):
        pipeline.run_sparse(project, tools, settings)


def test_scene(project: Project, tools: Tools, monkeypatch: pytest.MonkeyPatch) -> None:
    """A room or outdoor scene: no masks, no ring advice, walls kept when meshing."""
    from ez2digitize import coverage, presets

    bundle = list_bundles(project)[0]
    (project.masks_dir / bundle.id).mkdir()
    (project.masks_dir / bundle.id / "a.jpg.png").write_bytes(b"mask")
    analysed: list[object] = []
    monkeypatch.setattr(coverage, "analyse", lambda *args: analysed.append(args))
    monkeypatch.setattr(coverage, "weak_photos", lambda _db: [])  # the fake has no database
    pipeline.run_mesh(project, tools, presets.mesh_settings(subject="scene"))
    assert "--ImageReader.mask_path" not in (project.stage_dir("features") / "log.txt").read_text()
    assert "--free-space-support 1" in (project.stage_dir("mesh") / "log.txt").read_text()
    assert analysed == []
    pipeline.run_mesh(project, tools, presets.mesh_settings())
    assert len(analysed) == 1
    assert "--free-space-support" not in (project.stage_dir("mesh") / "log.txt").read_text()


def test_masks_are_used_when_present(project: Project, tools: Tools) -> None:
    pipeline.run_sparse(project, tools)
    assert "--ImageReader.mask_path" not in (project.stage_dir("features") / "log.txt").read_text()
    bundle = list_bundles(project)[0]
    (project.masks_dir / bundle.id).mkdir()
    (project.masks_dir / bundle.id / "a.jpg.png").write_bytes(b"mask")
    pipeline.run_sparse(project, tools)
    assert "--ImageReader.mask_path" in (project.stage_dir("features") / "log.txt").read_text()
    pipeline.run_sparse(project, tools, MeshSettings(use_masks=False))
    assert "--ImageReader.mask_path" not in (project.stage_dir("features") / "log.txt").read_text()


def test_cancel_and_busy(project: Project, tools: Tools, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SLEEP", "global_mapper")
    cancel = CancelToken()
    outcome: list[BaseException] = []

    def work() -> None:
        try:
            pipeline.run_mesh(project, tools, cancel=cancel)
        except BaseException as exc:
            outcome.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    deadline = time.monotonic() + 20
    while not (project.stage_dir("mapping") / "log.txt").exists():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    # Only one reconstruction at a time.
    with pytest.raises(PipelineBusy):
        pipeline.run_sparse(project, tools)
    cancel.cancel()
    worker.join(timeout=20)
    assert not worker.is_alive()
    assert isinstance(outcome[0], PipelineCancelled)
    assert "mapping" in str(outcome[0])
    # The lock is released afterwards.
    os.environ.pop("FAKE_SLEEP")
    pipeline.run_sparse(project, tools)


def test_masks_reach_openmvs_warped_and_named_by_stem(project: Project, tools: Tools) -> None:
    bundle = list_bundles(project)[0]
    (project.masks_dir / bundle.id).mkdir()
    (project.masks_dir / bundle.id / "a.jpg.png").write_bytes(b"mask of a")
    events, handler = _collect()
    result = pipeline.run_mesh(project, tools, on_event=handler)

    assert result.sparse.masks is not None
    started = [e.stage for e in events if isinstance(e, StageStarted)]
    assert started.index("mask-undistort") == started.index("undistort") + 1
    warped = project.stage_dir("mask-undistort") / "masks"
    assert sorted(p.name for p in warped.iterdir()) == ["a.mask.png", "b.mask.png", "c.mask.png"]
    assert (warped / "a.mask.png").read_bytes() == b"mask of a"
    assert (warped / "b.mask.png").read_bytes().startswith(b"\x89PNG")  # white: keep all
    camera_list = (project.stage_dir("mask-undistort") / "mask_cameras.txt").read_text()
    assert camera_list.splitlines()[0] == "a.mask.png SIMPLE_RADIAL 8 6 7.0 4.0 3.0 0.01"
    densify_log = (project.stage_dir("densify") / "log.txt").read_text()
    assert f"--mask-path {warped}" in densify_log

    # Dense-only runs pick the masks up from the manifests; unmasked runs don't.
    events, handler = _collect()
    pipeline.run_dense(project, tools, on_event=handler)
    assert all(e.reused for e in events if isinstance(e, StageFinished))
    pipeline.run_dense(project, tools, MeshSettings(use_masks=False))
    assert "--mask-path" not in (project.stage_dir("densify") / "log.txt").read_text()


def test_same_file_name_in_two_captures_skips_openmvs_masks(
    project: Project, tools: Tools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    _jpeg(other / "a.jpg", 5)
    second = import_files(project, [other / "a.jpg"], source="folder")
    (project.masks_dir / second.id).mkdir(parents=True)
    (project.masks_dir / second.id / "a.jpg.png").write_bytes(b"m")
    monkeypatch.setenv("FAKE_MODELS", "4")
    events, handler = _collect()
    result = pipeline.run_mesh(project, tools, on_event=handler)
    assert result.sparse.masks is None
    notices = [e.message for e in events if isinstance(e, Notice)]
    assert any("densifying without masks" in n and "same file name" in n for n in notices)


def test_feature_threads_are_capped_by_memory(
    project: Project, tools: Tools, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "cpu_threads", lambda: 16)
    monkeypatch.setattr(pipeline, "available_memory", lambda: 21 * 1024**3)
    events, handler = _collect()
    pipeline.run_sparse(project, tools, on_event=handler)
    log = (project.stage_dir("features") / "log.txt").read_text()
    assert "--FeatureExtraction.num_threads 7" in log
    notices = [e.message for e in events if isinstance(e, Notice)]
    assert any(n.startswith("finding features with 7 of 16 CPU threads") for n in notices)


def test_splats(
    project: Project, tools: Tools, fake_brush: Brush, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from ez2digitize.core.hardware import Gpu

    monkeypatch.setattr(pipeline, "detect_gpus", lambda: [Gpu("amd", "RX 7900 GRE")])
    events: list[pipeline.PipelineEvent] = []
    result = pipeline.run_splat(project, replace(tools, brush=fake_brush), on_event=events.append)
    assert result.file.read_bytes().startswith(b"ply\n")
    assert [f.suffix for f in result.exports] == [".ply", ".spz"]
    started = [e.stage for e in events if isinstance(e, pipeline.StageStarted)]
    assert started == ["features", "matching", "mapping", "undistort", "splat"]
    assert result.splat.host["gpu"]
    progress = [
        e.event.fraction
        for e in events
        if isinstance(e, pipeline.StageOutput) and e.stage == "splat"
        and isinstance(e.event, Progress) and e.event.fraction is not None
    ]  # fmt: skip
    assert progress == [0.5, 1.0]
    # Nothing changed: the splats are reused too.
    again = pipeline.run_splat(project, replace(tools, brush=fake_brush))
    assert again.splat.run_id == result.splat.run_id


def test_splats_use_the_masks(
    project: Project, tools: Tools, fake_brush: Brush, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from ez2digitize.core.hardware import Gpu

    monkeypatch.setattr(pipeline, "detect_gpus", lambda: [Gpu("amd", "RX 7900 GRE")])
    bundle = list_bundles(project)[0]
    (project.masks_dir / bundle.id).mkdir(parents=True)
    (project.masks_dir / bundle.id / "a.jpg.png").write_bytes(b"mask of a")
    with_brush = replace(tools, brush=fake_brush)
    events, handler = _collect()
    result = pipeline.run_splat(project, with_brush, on_event=handler)
    started = [e.stage for e in events if isinstance(e, StageStarted)]
    assert started[-2:] == ["mask-undistort", "splat"]
    assert result.sparse.masks is not None
    assert result.splat.inputs["masks"] == f"run:{result.sparse.masks.run_id}"
    # Every registered photo has a mask (white where there was none).
    assert "masks: 3" in (project.stage_dir("splat") / "log.txt").read_text()

    unmasked = pipeline.run_splat(project, with_brush, MeshSettings(use_masks=False))
    assert "masks" not in unmasked.splat.inputs
    assert "masks: 0" in (project.stage_dir("splat") / "log.txt").read_text()


def test_splats_skip_masks_brush_cannot_tell_apart(
    project: Project, tools: Tools, fake_brush: Brush, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    from dataclasses import replace

    from ez2digitize.core.hardware import Gpu

    monkeypatch.setattr(pipeline, "detect_gpus", lambda: [Gpu("amd", "RX 7900 GRE")])
    other = tmp_path / "other"
    other.mkdir()
    _jpeg(other / "A.JPG", 5)  # a.jpg's stem apart from case
    import_files(project, [other / "A.JPG"], source="folder")
    bundle = list_bundles(project)[0]
    (project.masks_dir / bundle.id).mkdir(parents=True)
    (project.masks_dir / bundle.id / "a.jpg.png").write_bytes(b"m")
    monkeypatch.setenv("FAKE_MODELS", "4")
    events, handler = _collect()
    result = pipeline.run_splat(project, replace(tools, brush=fake_brush), on_event=handler)
    notices = [e.message for e in events if isinstance(e, Notice)]
    assert any("training splats without masks" in n and "apart from case" in n for n in notices)
    assert "masks" not in result.splat.inputs


def test_splats_need_brush_and_a_real_gpu(
    project: Project, tools: Tools, fake_brush: Brush, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from ez2digitize.core.hardware import Gpu

    with pytest.raises(pipeline.PipelineError, match="need Brush"):
        pipeline.run_splat(project, tools)
    monkeypatch.setattr(pipeline, "detect_gpus", lambda: [Gpu("cpu", "llvmpipe")])
    with_brush = replace(tools, brush=fake_brush)
    with pytest.raises(pipeline.PipelineError, match="software renderer"):
        pipeline.run_splat(project, with_brush)
    assert pipeline.run_splat(project, with_brush, allow_software_gpu=True).file.is_file()


def test_two_sided_scan_notices(
    project: Project, tools: Tools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    under = tmp_path / "under"
    under.mkdir()
    _jpeg(under / "d.jpg", 3)
    _jpeg(under / "e.jpg", 4)
    second = import_files(
        project, [under / "d.jpg", under / "e.jpg"], source="folder", flipped=True
    )
    # The fake mapper places the first images in order: a, b, c only.
    monkeypatch.setenv("FAKE_MODELS", "3")
    events, handler = _collect()
    pipeline.run_sparse(project, tools, on_event=handler)
    notices = [e.message for e in events if isinstance(e, Notice)]
    assert any(n.startswith("two-sided scan: 5 of 5 photos have no mask") for n in notices)
    assert any("the two sides did not join (3 of 3 photos of the first side" in n for n in notices)

    for bundle in list_bundles(project):
        (project.masks_dir / bundle.id).mkdir(parents=True, exist_ok=True)
        for name in ("a.jpg", "b.jpg", "c.jpg") if bundle.id != second.id else ("d.jpg", "e.jpg"):
            (project.masks_dir / bundle.id / f"{name}.png").write_bytes(b"m")
    monkeypatch.setenv("FAKE_MODELS", "5")
    events, handler = _collect()
    pipeline.run_sparse(project, tools, on_event=handler)
    notices = [e.message for e in events if isinstance(e, Notice)]
    assert not any(n.startswith("two-sided scan") for n in notices)
    assert any(n.startswith("both sides joined: 3 of 3 photos") for n in notices)


def test_crop_box_reaches_densify(project: Project, tools: Tools) -> None:
    from ez2digitize import crop, scale

    pipeline.run_sparse(project, tools)
    run = crop.camera_run(project)
    assert run is not None
    upright_box = crop.UprightBox((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    crop.save(project, crop.from_upright(upright_box, None, run))
    scale.save(project, scale.make(((0, 0, 0), (1, 0, 0)), 10.0, run))
    events, handler = _collect()
    pipeline.run_dense(project, tools, on_event=handler)
    densify = project.stage_dir("densify")
    log = (densify / "log.txt").read_text()
    assert f"--import-roi-file {densify / 'crop_box.txt'} --crop-to-roi 1" in log
    assert (densify / "crop_box.txt").read_text().splitlines()[-1] == "1.0 2.0 3.0"
    notices = [e.message for e in events if isinstance(e, Notice)]
    assert "the dense cloud keeps what is inside the crop box" in notices

    # A new camera placement: the old box doesn't fit its coordinates.
    pipeline.run_sparse(project, tools, force_from="mapping")
    events, handler = _collect()
    pipeline.run_dense(project, tools, on_event=handler)
    assert "--import-roi-file" not in (densify / "log.txt").read_text()
    notices = [e.message for e in events if isinstance(e, Notice)]
    assert any("drawn on an earlier camera placement" in n for n in notices)
    assert any("scale was set on an earlier camera placement" in n for n in notices)
