# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from ez2digitize.core.capture import import_files
from ez2digitize.core.project import Project
from ez2digitize.core.runner import CancelToken, Event, Output, ProcessStartError
from ez2digitize.core.stage import (
    Backend,
    StageSpec,
    capture_input,
    file_input,
    load_manifest,
    run_stage,
    stage_input,
    tree_input,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="runner is POSIX only")

TOOL = Backend("fake-tool", "1.0.0")


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "project")


def writer_spec(
    name: str = "01-write",
    text: str = "hello",
    *,
    backend: Backend = TOOL,
    parameters: dict[str, Any] | None = None,
    inputs: dict[str, str] | None = None,
) -> StageSpec:
    """A stage that writes `text` to out.txt in its folder (its working directory)."""
    code = f"open('out.txt', 'w').write({text!r}); print('wrote')"
    return StageSpec(
        name=name,
        backend=backend,
        argv=[sys.executable, "-c", code],
        parameters={"text": text} if parameters is None else parameters,
        inputs=inputs or {},
    )


def test_run_writes_manifest_log_and_outputs(project: Project) -> None:
    run = run_stage(project, writer_spec())
    stage_dir = project.stage_dir("01-write")
    assert not run.reused
    assert (stage_dir / "out.txt").read_text() == "hello"
    assert "wrote" in (stage_dir / "log.txt").read_text()
    data = json.loads((stage_dir / "stage.json").read_text())
    assert data["schema_version"] == 1
    assert data["status"] == "succeeded"
    assert data["backend"] == {"name": "fake-tool", "version": "1.0.0"}
    assert data["command"][0] == sys.executable
    assert data["parameters"] == {"text": "hello"}
    assert data["exit_code"] == 0
    assert set(data["host"]) == {"system", "release", "machine", "app_version"}
    assert load_manifest(stage_dir) == run.manifest


def test_identical_run_is_reused(project: Project) -> None:
    first = run_stage(project, writer_spec())
    second = run_stage(project, writer_spec())
    assert second.reused
    assert second.manifest.run_id == first.manifest.run_id


def test_force_reruns(project: Project) -> None:
    first = run_stage(project, writer_spec())
    second = run_stage(project, writer_spec(), force=True)
    assert not second.reused and second.manifest.run_id != first.manifest.run_id


@pytest.mark.parametrize(
    "changed",
    [
        {"parameters": {"text": "other"}},
        {"backend": Backend("fake-tool", "1.0.1")},
        {"inputs": {"images": "capture:abc"}},
    ],
)
def test_changes_invalidate(project: Project, changed: dict[str, Any]) -> None:
    run_stage(project, writer_spec())
    assert not run_stage(project, writer_spec(**changed)).reused


def test_command_paths_are_not_part_of_the_key(project: Project) -> None:
    spec = writer_spec()
    run_stage(project, spec)
    spec.argv = [*spec.argv, "--moved-project"]
    assert run_stage(project, spec).reused


def test_rerun_clears_stale_outputs(project: Project) -> None:
    run_stage(project, writer_spec())
    stale = project.stage_dir("01-write") / "stale.bin"
    stale.write_bytes(b"old")
    run_stage(project, writer_spec(text="new"))
    assert not stale.exists()
    assert (project.stage_dir("01-write") / "out.txt").read_text() == "new"


def test_failed_stage_is_recorded_and_not_reused(project: Project) -> None:
    spec = StageSpec(
        name="02-fail", backend=TOOL, argv=[sys.executable, "-c", "raise SystemExit(2)"]
    )
    first = run_stage(project, spec)
    assert first.manifest.status == "failed" and first.manifest.exit_code == 2
    with pytest.raises(ValueError, match="did not succeed"):
        stage_input(first.manifest)
    second = run_stage(project, spec)
    assert not second.reused


def test_cancelled_stage(project: Project) -> None:
    cancel = CancelToken()

    def on_event(event: Event) -> None:
        if isinstance(event, Output) and event.line == "ready":
            cancel.cancel()

    code = "import time; print('ready', flush=True); time.sleep(60)"
    spec = StageSpec(name="03-slow", backend=TOOL, argv=[sys.executable, "-c", code])
    run = run_stage(project, spec, on_event=on_event, cancel=cancel)
    assert run.manifest.status == "cancelled"
    assert not run_stage(project, spec, on_event=on_event, cancel=cancel).reused


def test_upstream_rerun_invalidates_downstream(project: Project) -> None:
    def downstream(upstream_input: str) -> StageSpec:
        return writer_spec("02-next", inputs={"previous": upstream_input})

    up = run_stage(project, writer_spec())
    assert not run_stage(project, downstream(stage_input(up.manifest))).reused
    # Same upstream run: downstream reused.
    up = run_stage(project, writer_spec())
    assert run_stage(project, downstream(stage_input(up.manifest))).reused
    # Upstream re-ran (same parameters, new outputs): downstream must re-run.
    up = run_stage(project, writer_spec(), force=True)
    assert not run_stage(project, downstream(stage_input(up.manifest))).reused


def test_missing_backend_leaves_no_manifest(project: Project, tmp_path: Path) -> None:
    spec = StageSpec(name="04-missing", backend=TOOL, argv=[tmp_path / "no-such-tool"])
    with pytest.raises(ProcessStartError):
        run_stage(project, spec)
    assert load_manifest(project.stage_dir("04-missing")) is None
    assert (project.stage_dir("04-missing") / "log.txt").exists()


def test_corrupt_manifest_means_rerun(project: Project) -> None:
    run_stage(project, writer_spec())
    (project.stage_dir("01-write") / "stage.json").write_text('{"schema_version": 1}')
    assert load_manifest(project.stage_dir("01-write")) is None
    assert not run_stage(project, writer_spec()).reused


def test_input_fingerprints(project: Project, tmp_path: Path) -> None:
    a = tmp_path / "a.jpg"
    a.write_bytes(b"one")
    assert file_input(a).startswith("sha256:")

    bundle = import_files(project, [a], source="folder")
    before = capture_input(bundle)
    bundle.files[0].sha256 = "0" * 64
    assert capture_input(bundle) != before

    masks = tmp_path / "masks"
    (masks / "cap").mkdir(parents=True)
    (masks / "cap" / "a.png").write_bytes(b"mask")
    first = tree_input(masks)
    assert first == tree_input(masks)
    (masks / "cap" / "a.png").write_bytes(b"MASK")
    assert tree_input(masks) != first


def test_prepare_runs_in_the_fresh_stage_folder(project: Project) -> None:
    seen: list[list[str]] = []

    def prepare(stage_dir: Path) -> None:
        seen.append(sorted(p.name for p in stage_dir.iterdir()))
        (stage_dir / "input.txt").write_text("from prepare")

    code = "print(open('input.txt').read())"
    spec = StageSpec(
        name="05-prep", backend=TOOL, argv=[sys.executable, "-c", code], prepare=prepare
    )
    run_stage(project, spec)
    run_stage(project, spec, force=True)
    assert seen == [[], []]
    assert "from prepare" in (project.stage_dir("05-prep") / "log.txt").read_text()
