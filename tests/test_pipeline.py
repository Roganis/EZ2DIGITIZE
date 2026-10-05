# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pipeline logic with fake backends that write the files the real ones do."""

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from ez2digitize import pipeline
from ez2digitize.core.capture import import_files, list_bundles
from ez2digitize.core.project import Project
from ez2digitize.core.runner import CancelToken, Output
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

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX executables")


@pytest.fixture
def project(tmp_path: Path) -> Project:
    project = Project.create(tmp_path / "project")
    files = []
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (tmp_path / name).write_bytes(name.encode())
        files.append(tmp_path / name)
    import_files(project, files, source="folder")
    return project


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
    monkeypatch.setenv("FAKE_MODELS", "")
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
    (other / "a.jpg").write_bytes(b"another a")
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
