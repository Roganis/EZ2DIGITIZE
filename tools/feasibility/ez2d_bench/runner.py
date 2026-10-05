# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run one external command and measure what it costs.

Measures wall time, CPU time and peak RSS of the process tree (through the
small _launch.py helper, see there why), and samples GPU memory and swap
while it runs (see memprobe).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ez2d_bench import memprobe

TAIL_LINES = 40
LAUNCHER = Path(__file__).with_name("_launch.py")


@dataclass
class StepResult:
    name: str
    command: list[str]
    exit_code: int
    wall_s: float
    cpu_s: float
    peak_rss_mb: float
    peak_vram_mb: float | None = None
    peak_gtt_mb: float | None = None
    peak_swap_mb: float | None = None
    log: str = ""
    tail: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepResult:
        return cls(**data)


class _Sampler(threading.Thread):
    """Polls GPU memory and swap; keeps the peak increase over the baseline."""

    def __init__(self, interval_s: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self._halt = threading.Event()
        self.baseline = memprobe.sample()
        self.peak = self.baseline

    def run(self) -> None:
        while not self._halt.wait(self.interval_s):
            self.peak = self.peak.max(memprobe.sample())

    def stop(self) -> memprobe.Usage:
        self._halt.set()
        self.join()
        self.peak = self.peak.max(memprobe.sample())
        return self.peak.minus(self.baseline)


def _maxrss_mb(ru_maxrss: int) -> float:
    # Linux reports kilobytes, macOS bytes.
    return ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else ru_maxrss / 1024


def _kill_group(proc: subprocess.Popen[str]) -> None:
    for sig, grace in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.1)


def run_step(
    name: str,
    command: list[str | Path],
    log_path: Path,
    cwd: Path | None = None,
    echo: bool = True,
) -> StepResult:
    """Run `command`, stream its output to `log_path`, and measure it.

    The child gets its own process group so Ctrl+C here kills the whole tree.
    """
    stats_path = log_path.with_suffix(".stats.json")
    argv = [str(part) for part in command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail: deque[str] = deque(maxlen=TAIL_LINES)

    if echo:
        print(f"  [{name}] {' '.join(argv)}", flush=True)

    sampler = _Sampler()
    sampler.start()
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(argv) + "\n\n")
        log.flush()
        proc = subprocess.Popen(  # noqa: S603 - argv is built by this tool, never a shell
            [sys.executable, "-I", "-S", str(LAUNCHER), str(stats_path), "--", *argv],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )

        def pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                tail.append(line.rstrip("\n"))

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            _, status, rusage = os.wait4(proc.pid, 0)
        except KeyboardInterrupt:
            print(f"\n  [{name}] interrupted, stopping process group", flush=True)
            _kill_group(proc)
            sampler.stop()
            raise
        proc.returncode = os.waitstatus_to_exitcode(status)
        reader.join()
    wall = time.monotonic() - start
    delta = sampler.stop()

    # Prefer the launcher's measurement of the command itself; fall back to the
    # launcher's own rusage if it died before writing it.
    try:
        stats = json.loads(stats_path.read_text())
        stats_path.unlink()
        exit_code, maxrss, cpu = stats["exit_code"], stats["ru_maxrss"], stats["cpu_s"]
    except (OSError, ValueError, KeyError):
        exit_code, maxrss = proc.returncode, rusage.ru_maxrss
        cpu = rusage.ru_utime + rusage.ru_stime

    result = StepResult(
        name=name,
        command=argv,
        exit_code=exit_code,
        wall_s=round(wall, 2),
        cpu_s=round(cpu, 2),
        peak_rss_mb=round(_maxrss_mb(maxrss), 1),
        peak_vram_mb=delta.vram_mb,
        peak_gtt_mb=delta.gtt_mb,
        peak_swap_mb=delta.swap_mb,
        log=str(log_path),
        tail=list(tail),
    )
    if echo:
        status_text = "ok" if result.ok else f"FAILED (exit {result.exit_code})"
        print(
            f"  [{name}] {status_text} in {result.wall_s:.1f}s, "
            f"peak RSS {result.peak_rss_mb:.0f} MB",
            flush=True,
        )
    return result


def capture(command: list[str | Path], timeout_s: float = 60.0) -> str:
    """Run a quick command (help, version) and return its combined output.

    The exit code is ignored: many tools exit non-zero after printing help.
    Runs in a temporary directory because OpenMVS tools write a log file into
    the current directory on every invocation.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="ez2d-probe-") as scratch:
            done = subprocess.run(  # noqa: S603 - argv is built by this tool, never a shell
                [str(part) for part in command],
                cwd=scratch,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout_s,
                stdin=subprocess.DEVNULL,
                check=False,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"<failed to run: {exc}>"
    return done.stdout + done.stderr
