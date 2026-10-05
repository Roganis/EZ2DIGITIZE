# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run every variant of a plan and record the results.

Results layout:
    <results>/<machine>/sysinfo.json
    <results>/<machine>/<dataset>/<kind>/<label>/run.json   (+ outputs, logs/)

A run whose run.json says "ok" with the same config is skipped, so an
interrupted plan can simply be started again.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ez2d_bench import brush, colmap, openmvs, sysinfo
from ez2d_bench.plan import BrushRun, ColmapRun, OpenMVSRun, Plan, config_dict
from ez2d_bench.record import Recorder, RunFailed, RunRecord, StepFailed
from ez2d_bench.tools import Toolbox, ToolMissing

Body = Callable[[Recorder], dict[str, Any]]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Executor:
    def __init__(
        self,
        plan: Plan,
        results: Path,
        machine: str,
        toolbox: Toolbox,
        only: set[str] | None = None,
        force: bool = False,
    ) -> None:
        self.plan = plan
        self.machine_dir = results / machine
        self.dataset_dir = self.machine_dir / plan.dataset.name
        self.machine = machine
        self.tb = toolbox
        self.only = only
        self.force = force
        self.versions: dict[str, str | None] = {}

    def run_dir(self, kind: str, label: str) -> Path:
        return self.dataset_dir / kind / label

    def execute(self) -> list[RunRecord]:
        unknown = (self.only or set()) - set(self.plan.labels())
        if unknown:
            raise RunFailed(f"--only: unknown labels {sorted(unknown)}")
        info = sysinfo.collect(self.tb)
        self.versions = info["tools"]
        self.machine_dir.mkdir(parents=True, exist_ok=True)
        (self.machine_dir / "sysinfo.json").write_text(json.dumps(info, indent=2) + "\n")

        records: list[RunRecord] = []
        for c in self.plan.colmap:
            records += self._maybe(c, "colmap", self._colmap_body(c))
        for o in self.plan.openmvs:
            records += self._maybe(o, "openmvs", self._openmvs_body(o), source=o.source)
        for b in self.plan.brush:
            records += self._maybe(b, "brush", self._brush_body(b), source=b.source)
        return records

    # --- bodies ----------------------------------------------------------------

    def _colmap_body(self, cfg: ColmapRun) -> Body:
        return lambda rec: colmap.run(rec, self.tb, self.plan.dataset, cfg)

    def _openmvs_body(self, cfg: OpenMVSRun) -> Body:
        src = self.run_dir("colmap", cfg.source)
        return lambda rec: openmvs.run(rec, self.tb, self.plan.dataset, cfg, src)

    def _brush_body(self, cfg: BrushRun) -> Body:
        src = self.run_dir("colmap", cfg.source)
        return lambda rec: brush.run(rec, self.tb, cfg, src)

    # --- orchestration -------------------------------------------------------

    def _maybe(
        self,
        cfg: ColmapRun | OpenMVSRun | BrushRun,
        kind: str,
        body: Body,
        source: str | None = None,
    ) -> list[RunRecord]:
        if self.only is not None and cfg.label not in self.only:
            return []
        run_dir = self.run_dir(kind, cfg.label)
        config = config_dict(cfg)

        existing = RunRecord.load(run_dir)
        if existing and existing.ok and existing.config == config and not self.force:
            print(f"= {kind}/{cfg.label}: already done, skipping (use --force to rerun)")
            return [existing]

        record = RunRecord(
            kind=kind,
            label=cfg.label,
            dataset=self.plan.dataset.name,
            machine=self.machine,
            config=config,
            tool_versions=self.versions,
        )
        if source is not None:
            src_record = RunRecord.load(self.run_dir("colmap", source))
            if src_record is None or not src_record.ok:
                record.status = "skipped"
                record.reason = f"COLMAP run '{source}' has not completed successfully"
                print(f"- {kind}/{cfg.label}: skipped, {record.reason}")
                record.save(run_dir)
                return [record]

        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        print(f"> {kind}/{cfg.label}", flush=True)
        recorder = Recorder(run_dir, record)
        record.started = _now()
        try:
            record.metrics.update(body(recorder))
            record.status = "ok"
        except StepFailed as exc:
            record.status, record.reason = "failed", str(exc)
        except (RunFailed, ToolMissing) as exc:
            record.status, record.reason = "failed", str(exc)
        except KeyboardInterrupt:
            record.status, record.reason = "interrupted", "interrupted by user"
            record.finished = _now()
            record.save(run_dir)
            raise
        record.finished = _now()
        record.save(run_dir)
        summary = "ok" if record.ok else f"FAILED: {record.reason}"
        print(f"< {kind}/{cfg.label}: {summary} ({record.total_wall_s():.0f}s)\n", flush=True)
        return [record]


def load_all(results: Path) -> list[RunRecord]:
    return [
        RunRecord.from_dict(json.loads(path.read_text()))
        for path in sorted(results.rglob("run.json"))
    ]
