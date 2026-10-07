#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Time the app's pipeline with two or more backend builds on one machine.

    python tools/backends/compare.py IMAGES --backend main=PREFIX_A \\
        --backend branch=PREFIX_B [--rounds 2] [--markdown OUT.md] [--json OUT.json]

Each PREFIX is an unpacked backend archive (bin/colmap, the OpenMVS tools,
BUILDINFO.json). For every round and build it makes a fresh project of the
photos in IMAGES and runs `ez2d run` on it, then reads each stage's wall time
from its stage.json: the stages the user waits for, run the way the app runs
them. The builds take turns, in the opposite order every other round, so a
runner that slows down partway doesn't favour one of them. The first build is
the reference the others are compared with.

Used by the Backends workflow to check that a change in how the backends are
built (compiler, optimisation) doesn't make reconstructions slower. Run it
where `ez2digitize` is importable (`uv run python tools/backends/compare.py`).
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Build:
    name: str
    prefix: Path
    compiler: str = "?"
    stages: dict[str, list[float]] = field(default_factory=dict)
    totals: list[float] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def colmap(self) -> Path:
        exe = self.prefix / "bin" / "colmap.exe"
        return exe if exe.exists() else self.prefix / "bin" / "colmap"


def parse_build(text: str) -> Build:
    name, sep, prefix = text.partition("=")
    if not sep or not name or not prefix:
        raise argparse.ArgumentTypeError(f"expected NAME=PREFIX, got {text!r}")
    build = Build(name, Path(prefix))
    try:
        info = json.loads((build.prefix / "BUILDINFO.json").read_text(encoding="utf-8"))
        build.compiler = str(info.get("compiler", "?"))
    except (OSError, ValueError):
        pass
    return build


def ez2d(*args: str | Path) -> None:
    cli = [sys.executable, "-m", "ez2digitize.cli", *map(str, args)]
    subprocess.run(cli, check=True)  # noqa: S603 - our own CLI, arguments from the command line


def stage_times(project: Path) -> dict[str, float]:
    """Wall time of every stage that ran, in the order they ran.

    `started` has whole seconds; the manifest's own write time, at the end
    of the stage, breaks ties.
    """
    runs = []
    for path in (project / "stages").glob("*/stage.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") == "succeeded":
            key = (data["started"], path.stat().st_mtime)
            runs.append((key, data["stage"], float(data["wall_s"])))
    return {stage: wall for _key, stage, wall in sorted(runs)}


def run_once(build: Build, images: Path, work: Path, label: str, run_args: list[str]) -> None:
    project = work / label
    shutil.rmtree(project, ignore_errors=True)
    print(f"\n=== {label}: {build.prefix}", flush=True)
    start = time.perf_counter()
    try:
        ez2d("new", project)
        ez2d("import", project, images)
        ez2d(
            "run", project, "--colmap", build.colmap, "--openmvs-dir", build.prefix / "bin",
            *run_args,
        )  # fmt: skip
    except subprocess.CalledProcessError as exc:
        build.failures.append(f"{label}: exit code {exc.returncode}")
        return
    build.totals.append(time.perf_counter() - start)
    for stage, wall in stage_times(project).items():
        build.stages.setdefault(stage, []).append(wall)
    shutil.rmtree(project, ignore_errors=True)  # the dense outputs are large


def fmt(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    return f"{seconds / 60:.1f} min" if seconds >= 120 else f"{seconds:.1f} s"


def ratio(value: float | None, reference: float | None) -> str:
    if value is None or not reference:
        return "–"
    return f"{value / reference:.2f}×"


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def markdown(builds: list[Build], rounds: int, images: Path, run_args: list[str]) -> str:
    ref, others = builds[0], builds[1:]
    count = sum(1 for p in images.iterdir() if p.is_file())
    lines = [
        "## Backend builds compared",
        "",
        f"`ez2d run {' '.join(run_args)}` on {count} photos ({images.name}), "
        f"{rounds} round(s) per build, the builds taking turns; median wall time per "
        f"stage. Ratios are against **{ref.name}**: below 1 is faster.",
        "",
    ]
    for build in builds:
        lines.append(f"- **{build.name}**: {build.compiler}")
    lines.append("")
    header = ["Stage", ref.name]
    for other in others:
        header += [other.name, "ratio"]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    order = list(ref.stages) + [s for b in others for s in b.stages if s not in ref.stages]
    for stage in dict.fromkeys(order):
        row = [stage, fmt(median(ref.stages.get(stage, [])))]
        for other in others:
            value = median(other.stages.get(stage, []))
            row += [fmt(value), ratio(value, median(ref.stages.get(stage, [])))]
        lines.append("| " + " | ".join(row) + " |")
    row = ["**whole run**", f"**{fmt(median(ref.totals))}**"]
    for other in others:
        value = median(other.totals)
        row += [f"**{fmt(value)}**", f"**{ratio(value, median(ref.totals))}**"]
    lines.append("| " + " | ".join(row) + " |")
    failures = [f for b in builds for f in b.failures]
    if failures:
        lines += ["", "Failed runs (left out of the medians):", ""]
        lines += [f"- {f}" for f in failures]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("images", type=Path, help="a folder of photos")
    parser.add_argument(
        "--backend", dest="builds", type=parse_build, action="append", required=True,
        metavar="NAME=PREFIX", help="a build to time (twice or more; the first is the reference)",
    )  # fmt: skip
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--run-arg", dest="run_args", action="append", default=[],
        help="an argument for `ez2d run` (repeat; e.g. --run-arg=--quality=fast)",
    )  # fmt: skip
    parser.add_argument("--work", type=Path, help="where the projects go (default: a temp dir)")
    parser.add_argument("--markdown", type=Path, help="write the comparison table here")
    parser.add_argument("--json", type=Path, help="write every measurement here")
    args = parser.parse_args(argv)
    builds: list[Build] = args.builds
    if len(builds) < 2:
        parser.error("give at least two --backend builds")
    if len({b.name for b in builds}) != len(builds):
        parser.error("the builds need different names")
    for build in builds:
        if not build.colmap.exists():
            parser.error(f"{build.name}: no COLMAP in {build.prefix / 'bin'}")

    work = args.work or Path(tempfile.mkdtemp(prefix="ez2d-compare-"))
    work.mkdir(parents=True, exist_ok=True)
    for round_ in range(args.rounds):
        for build in builds if round_ % 2 == 0 else builds[::-1]:
            run_once(build, args.images, work, f"{build.name}-{round_ + 1}", args.run_args)

    table = markdown(builds, args.rounds, args.images, args.run_args)
    print("\n" + table)
    if args.markdown:
        args.markdown.write_text(table, encoding="utf-8")
    if args.json:
        data = {
            b.name: {
                "prefix": str(b.prefix),
                "compiler": b.compiler,
                "stages": b.stages,
                "totals": b.totals,
                "failures": b.failures,
            }
            for b in builds
        }
        args.json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return 1 if any(b.failures for b in builds) else 0


if __name__ == "__main__":
    sys.exit(main())
