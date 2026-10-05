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
from ez2digitize.backends.colmap import Colmap
from ez2digitize.backends.openmvs import TOOLS, OpenMVS
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

# Behaviour is steered through environment variables the fakes read:
# FAKE_FAIL=<command> makes that command exit 1, FAKE_MODELS="30,2" sets the
# registered images per model, FAKE_SLEEP=<command> makes it hang.
FAKE_COLMAP = """
import os, struct, sys, time
from pathlib import Path
cmd, args = sys.argv[1], sys.argv[2:]
opt = lambda name: args[args.index(name) + 1]
print(f"fake colmap {cmd}", flush=True)
if os.environ.get("FAKE_SLEEP") == cmd:
    time.sleep(60)
if os.environ.get("FAKE_FAIL") == cmd:
    print("something went wrong", flush=True)
    sys.exit(1)
if cmd == "feature_extractor":
    # The "database" is the image list, so the mapper knows the names.
    Path(opt("--database_path")).write_text(Path(opt("--image_list_path")).read_text())
    print("Processed file [1/1]")
elif cmd in ("mapper", "global_mapper"):
    names = Path(opt("--database_path")).read_text().split()
    for i, n in enumerate(os.environ.get("FAKE_MODELS", "3").split(",")):
        if not n:
            continue
        model = Path(opt("--output_path")) / str(i)
        model.mkdir(parents=True)
        # One SIMPLE_RADIAL camera, 8x6 pixels; images registered in order.
        cameras = struct.pack("<QIiQQ4d", 1, 1, 2, 8, 6, 7.0, 4.0, 3.0, 0.01)
        (model / "cameras.bin").write_bytes(cameras)
        images = struct.pack("<Q", int(n))
        for image_id, name in enumerate(names[: int(n)], 1):
            images += struct.pack("<I7dI", image_id, 1, 0, 0, 0, 0, 0, 0, 1)
            images += name.encode() + b"\\0" + struct.pack("<Q", 0)
        (model / "images.bin").write_bytes(images)
elif cmd == "image_undistorter":
    out = Path(opt("--output_path"))
    (out / "images").mkdir()
    (out / "sparse").mkdir()
elif cmd == "image_undistorter_standalone":
    src, out = Path(opt("--image_path")), Path(opt("--output_path"))
    for line in Path(opt("--input_file")).read_text().splitlines():
        name = line.split()[0]
        (out / name).write_bytes((src / name).read_bytes())
"""

FAKE_OPENMVS = """
import os, sys
from pathlib import Path
tool = Path(sys.argv[0]).name
args = sys.argv[1:]
print(f"fake {tool}", flush=True)
if os.environ.get("FAKE_FAIL") == tool:
    sys.exit(1)
out = Path(args[args.index("-o") + 1])
out.write_text("mvs")
out.with_suffix(".ply").write_text("ply")
if tool == "TextureMesh":
    (out.parent / "scene_textured0.png").write_bytes(b"png")
"""


def _script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)
    return path


@pytest.fixture
def tools(tmp_path: Path) -> Tools:
    exe = _script(tmp_path / "fake" / "colmap", FAKE_COLMAP)
    for tool in TOOLS:
        _script(tmp_path / "fake" / "mvs" / tool, FAKE_OPENMVS)
    return Tools(Colmap(exe, "4.2.1"), OpenMVS(tmp_path / "fake" / "mvs", "2.4.0"))


@pytest.fixture
def project(tmp_path: Path) -> Project:
    project = Project.create(tmp_path / "project")
    files = []
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (tmp_path / name).write_bytes(name.encode())
        files.append(tmp_path / name)
    import_files(project, files, source="folder")
    return project


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("FAKE_FAIL", "FAKE_MODELS", "FAKE_SLEEP"):
        monkeypatch.delenv(var, raising=False)


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
    assert notices == ["3 of 3 images registered (model 0)"]
    assert any(isinstance(e, StageOutput) and isinstance(e.event, Output) for e in events)

    # Unchanged: everything reused.
    events, handler = _collect()
    pipeline.run_mesh(project, tools, on_event=handler)
    finished = [e for e in events if isinstance(e, StageFinished)]
    assert len(finished) == 8 and all(e.reused for e in finished)


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


def test_matching_mode_follows_capture_source(project: Project, tools: Tools) -> None:
    pipeline.run_sparse(project, tools)
    assert "exhaustive_matcher" in (project.stage_dir("matching") / "log.txt").read_text()
    for bundle in list_bundles(project):
        bundle.source = "video"
        bundle.save()
    pipeline.run_sparse(project, tools)
    assert "sequential_matcher" in (project.stage_dir("matching") / "log.txt").read_text()


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
    monkeypatch.setenv("FAKE_SLEEP", "mapper")
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
