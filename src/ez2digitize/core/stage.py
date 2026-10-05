# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pipeline stages: run one backend command and record it in `stage.json`.

A stage's cache key is a hash of what decides its output: the stage name,
the backend's name and exact version, the parameters, and its inputs. Inputs
are fingerprints:

- `sha256:...`  contents of a single file (`file_input`),
- `capture:...` a capture bundle, from the hashes in its capture.json,
- `tree:...`    a folder such as the masks (`tree_input`),
- `run:...`     the run of an earlier stage (`stage_input`).

An earlier stage is referenced by its run id, which is new on every run, so
re-running a stage invalidates every stage that used its outputs without
hashing those outputs. The command line itself is recorded but not part of
the key: it contains absolute paths, and moving a project must not
invalidate it.

A stage is reused when its stage.json says it succeeded with the same key.
Otherwise its folder is emptied and the command runs again, with the stage
folder as working directory (OpenMVS writes its log files there).
"""

from __future__ import annotations

import platform
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import ez2digitize
from ez2digitize.core.capture import CaptureBundle
from ez2digitize.core.files import (
    FormatError,
    fingerprint,
    read_json_object,
    sha256_file,
    write_json_atomic,
)
from ez2digitize.core.project import Project
from ez2digitize.core.runner import CancelToken, EventHandler, LineParser, run_process

MANIFEST_FILE = "stage.json"
LOG_FILE = "log.txt"
SCHEMA_VERSION = 1

Status = Literal["succeeded", "failed", "cancelled"]


@dataclass(frozen=True)
class Backend:
    name: str
    version: str


@dataclass
class StageSpec:
    """Everything needed to run a stage. Built by the backend modules."""

    name: str
    backend: Backend
    argv: Sequence[str | Path]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    inputs: Mapping[str, str] = field(default_factory=dict)
    env: Mapping[str, str] | None = None
    parse_line: LineParser | None = None
    # Run under a pseudo-terminal: for tools whose output is block-buffered
    # when it goes to a pipe (see runner.run_process).
    use_pty: bool = False
    # Called with the (new, empty) stage folder just before the command runs,
    # to write files the command reads (image lists, copies of a database).
    prepare: Callable[[Path], None] | None = None

    def cache_key(self) -> str:
        return fingerprint(
            {
                "stage": self.name,
                "backend": asdict(self.backend),
                "parameters": dict(self.parameters),
                "inputs": dict(self.inputs),
            }
        )


@dataclass
class StageManifest:
    stage: str
    run_id: str
    status: Status
    cache_key: str
    backend: Backend
    command: list[str]
    parameters: dict[str, Any]
    inputs: dict[str, str]
    started: str
    finished: str
    wall_s: float
    cpu_s: float
    peak_rss_mb: float | None
    exit_code: int
    host: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageManifest:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise FormatError(f"unsupported stage.json schema_version {data.get('schema_version')}")
        fields = {k: v for k, v in data.items() if k != "schema_version"}
        backend = fields.get("backend")
        if not isinstance(backend, dict):
            raise FormatError("stage.json: 'backend' must be an object")
        try:
            fields["backend"] = Backend(**backend)
            return cls(**fields)
        except TypeError as exc:
            raise FormatError(f"stage.json: {exc}") from exc

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"


@dataclass
class StageRun:
    manifest: StageManifest
    # True if the stage was skipped because an identical run had succeeded.
    reused: bool


def load_manifest(stage_dir: Path) -> StageManifest | None:
    """The stage's manifest, or None if it has none or it is unreadable."""
    try:
        return StageManifest.from_dict(read_json_object(stage_dir / MANIFEST_FILE))
    except FormatError:
        return None


def file_input(path: Path) -> str:
    return f"sha256:{sha256_file(path)}"


def capture_input(bundle: CaptureBundle) -> str:
    return "capture:" + fingerprint([(f.name, f.sha256) for f in bundle.files])


def tree_input(folder: Path) -> str:
    """Fingerprint of every file under `folder` (names and contents)."""
    entries = sorted(
        (p.relative_to(folder).as_posix(), sha256_file(p)) for p in folder.rglob("*") if p.is_file()
    )
    return "tree:" + fingerprint(entries)


def stage_input(manifest: StageManifest) -> str:
    if not manifest.succeeded:
        raise ValueError(f"stage {manifest.stage} did not succeed; its outputs can't be used")
    return f"run:{manifest.run_id}"


def run_stage(
    project: Project,
    spec: StageSpec,
    *,
    on_event: EventHandler | None = None,
    cancel: CancelToken | None = None,
    force: bool = False,
) -> StageRun:
    """Run `spec` in its stage folder, or reuse the previous identical run.

    Raises runner.ProcessStartError if the backend can't be started; the
    stage folder then has a log but no manifest, so it is never reused.
    """
    stage_dir = project.stage_dir(spec.name)
    key = spec.cache_key()
    previous = load_manifest(stage_dir)
    if not force and previous is not None and previous.succeeded and previous.cache_key == key:
        return StageRun(previous, reused=True)

    # Stale outputs from an older run must never be mistaken for new ones.
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    if spec.prepare is not None:
        spec.prepare(stage_dir)

    result = run_process(
        spec.argv,
        log_path=stage_dir / LOG_FILE,
        cwd=stage_dir,
        env=spec.env,
        on_event=on_event,
        parse_line=spec.parse_line,
        cancel=cancel,
        use_pty=spec.use_pty,
    )
    status: Status = "cancelled" if result.cancelled else "succeeded" if result.ok else "failed"
    manifest = StageManifest(
        stage=spec.name,
        run_id=uuid.uuid4().hex,
        status=status,
        cache_key=key,
        backend=spec.backend,
        command=result.argv,
        parameters=dict(spec.parameters),
        inputs=dict(spec.inputs),
        started=result.started,
        finished=result.finished,
        wall_s=result.wall_s,
        cpu_s=result.cpu_s,
        peak_rss_mb=result.peak_rss_mb,
        exit_code=result.exit_code,
        host=_host_info(),
    )
    write_json_atomic(stage_dir / MANIFEST_FILE, manifest.to_dict())
    return StageRun(manifest, reused=False)


def _host_info() -> dict[str, str]:
    # The GPU driver (Mesa version on Linux) is added with the first GPU
    # backend (Brush, Phase 3); the CPU-only backends don't depend on it.
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "app_version": ez2digitize.__version__,
    }
