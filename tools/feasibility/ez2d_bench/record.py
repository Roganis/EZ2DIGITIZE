# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-run bookkeeping: the steps a run executed and its run.json record."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ez2d_bench.runner import StepResult, run_step

RECORD_NAME = "run.json"


class RunFailed(Exception):
    """A run could not produce its result; the message says why."""


class StepFailed(RunFailed):
    def __init__(self, step: StepResult) -> None:
        super().__init__(f"step '{step.name}' exited with code {step.exit_code}")
        self.step = step


@dataclass
class RunRecord:
    kind: str  # colmap | openmvs | brush
    label: str
    dataset: str
    machine: str
    config: dict[str, Any]
    status: str = "pending"  # ok | failed | skipped | interrupted
    reason: str = ""
    started: str = ""
    finished: str = ""
    steps: list[StepResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    tool_versions: dict[str, str | None] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def total_wall_s(self) -> float:
        return round(sum(s.wall_s for s in self.steps), 1)

    def wall_s(self, prefix: str) -> float | None:
        matching = [s.wall_s for s in self.steps if s.name.startswith(prefix)]
        return round(sum(matching), 1) if matching else None

    def peak(self, attr: str) -> float | None:
        values = [getattr(s, attr) for s in self.steps if getattr(s, attr) is not None]
        return max(values) if values else None

    def save(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        (run_dir / RECORD_NAME).write_text(json.dumps(data, indent=2, default=str) + "\n")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        data = dict(data)
        data["steps"] = [StepResult.from_dict(s) for s in data.get("steps", [])]
        return cls(**data)

    @classmethod
    def load(cls, run_dir: Path) -> RunRecord | None:
        path = run_dir / RECORD_NAME
        if not path.is_file():
            return None
        return cls.from_dict(json.loads(path.read_text()))


class Recorder:
    """Runs the steps of one benchmark run and collects their results."""

    def __init__(self, run_dir: Path, record: RunRecord) -> None:
        self.run_dir = run_dir
        self.record = record

    def step(self, name: str, command: list[str | Path], cwd: Path | None = None) -> StepResult:
        index = len(self.record.steps) + 1
        log = self.run_dir / "logs" / f"{index:02d}-{name}.log"
        result = run_step(name, command, log, cwd=cwd)
        self.record.steps.append(result)
        if not result.ok:
            raise StepFailed(result)
        return result

    def note(self, text: str) -> None:
        print(f"  note: {text}", flush=True)
        self.record.notes.append(text)
