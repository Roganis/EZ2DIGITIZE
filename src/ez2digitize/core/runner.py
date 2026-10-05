# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The process runner: the only place that starts backend processes.

- Commands are argument lists, never a shell string.
- Each process starts in its own process group (session), so cancelling
  kills the whole tree: SIGTERM, then SIGKILL after a grace period.
- stdout and stderr are merged, written to a log file, and handed line by
  line to an optional parser that turns them into progress events.
- Events are delivered on the calling thread, in order. A GUI calls
  `run_process` from a worker thread and forwards the events as signals.

POSIX only for now (Linux, macOS); Windows will need a job object for the
process tree and has no `wait4`.
"""

from __future__ import annotations

import contextlib
import os
import queue
import resource
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ez2digitize.core.files import utc_now

TAIL_LINES = 40
_POLL_S = 0.1
_MEMORY_SAMPLE_S = 0.5
# After the process exits, how long to wait for output still in the pipe
# (a grandchild that inherited stdout can keep it open forever).
_DRAIN_S = 2.0


@dataclass(frozen=True)
class Started:
    argv: list[str]
    pid: int


@dataclass(frozen=True)
class Output:
    line: str


@dataclass(frozen=True)
class Progress:
    """Parsed from output by a backend's line parser. `fraction` is 0..1 if known."""

    message: str
    fraction: float | None = None


Event = Started | Output | Progress
EventHandler = Callable[[Event], None]
LineParser = Callable[[str], Progress | None]


class ProcessStartError(Exception):
    """The command could not be started at all (missing executable, permissions)."""


class CancelToken:
    """Thread-safe flag a UI sets to stop a running process."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass
class ProcessResult:
    argv: list[str]
    exit_code: int
    cancelled: bool
    started: str
    finished: str
    wall_s: float
    cpu_s: float
    peak_rss_mb: float | None
    log_path: Path
    tail: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.cancelled


def run_process(
    argv: Sequence[str | Path],
    *,
    log_path: Path,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    on_event: EventHandler | None = None,
    parse_line: LineParser | None = None,
    cancel: CancelToken | None = None,
    term_grace_s: float = 10.0,
) -> ProcessResult:
    """Run `argv` to completion (or cancellation) and return what happened.

    `env` entries are added to the current environment. The log file is
    overwritten. Raises ProcessStartError if the command can't be started.
    """
    args = [str(a) for a in argv]
    if not args:
        raise ValueError("empty command")
    emit = on_event or (lambda _event: None)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    parent_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(args)}\n\n")
        log.flush()
        started_at, start = utc_now(), time.monotonic()
        try:
            proc = subprocess.Popen(  # noqa: S603 - argument list, never a shell
                args,
                cwd=cwd,
                env={**os.environ, **env} if env else None,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # Universal newlines also split the \r-only progress lines that
                # many CLI tools print.
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            log.write(f"cannot start {args[0]}: {exc}\n")
            raise ProcessStartError(f"cannot start {args[0]}: {exc}") from exc

        lines: queue.Queue[str | None] = queue.Queue()
        reader = threading.Thread(target=_pump, args=(proc, lines), daemon=True)
        reader.start()
        emit(Started(argv=args, pid=proc.pid))

        def handle(line: str) -> None:
            log.write(line + "\n")
            tail.append(line)
            emit(Output(line))
            if parse_line is not None and (progress := parse_line(line)) is not None:
                emit(progress)

        status: int | None = None
        rusage: resource.struct_rusage | None = None
        cancelled = False
        eof = False
        hwm_kb = 0
        next_sample = 0.0
        exited_at = 0.0
        while not eof:
            try:
                line = lines.get(timeout=_POLL_S)
            except queue.Empty:
                line = ""
            else:
                if line is None:
                    eof = True
                else:
                    handle(line)
            if status is None:
                now = time.monotonic()
                if now >= next_sample:
                    hwm_kb = max(hwm_kb, _vm_hwm_kb(proc.pid))
                    next_sample = now + _MEMORY_SAMPLE_S
                pid, wstatus, ru = os.wait4(proc.pid, os.WNOHANG)
                if pid == proc.pid:
                    status, rusage, exited_at = wstatus, ru, now
                    # Popen must not try to reap the process again.
                    proc.returncode = os.waitstatus_to_exitcode(wstatus)
                elif cancel is not None and cancel.cancelled and not cancelled:
                    cancelled = True
                    log.write("\n[cancelled, stopping the process group]\n")
                    log.flush()
                    status, rusage = _kill_group(proc.pid, term_grace_s)
                    exited_at = time.monotonic()
                    proc.returncode = os.waitstatus_to_exitcode(status)
            elif time.monotonic() - exited_at > _DRAIN_S:
                # The process is done but something it started still holds
                # the output pipe open. A finished stage leaves nothing behind.
                _signal_group(proc.pid, signal.SIGKILL)
                break
        if status is None:  # EOF arrived first: the process closed stdout
            _, status, rusage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
        # Anything left in the queue after EOF or the drain timeout.
        while True:
            try:
                rest = lines.get_nowait()
            except queue.Empty:
                break
            if rest is not None:
                handle(rest)
        wall = time.monotonic() - start
        exit_code = os.waitstatus_to_exitcode(status)
        log.write(f"\n[exit code {exit_code}{', cancelled' if cancelled else ''}]\n")

    assert rusage is not None
    return ProcessResult(
        argv=args,
        exit_code=exit_code,
        cancelled=cancelled,
        started=started_at,
        finished=utc_now(),
        wall_s=round(wall, 3),
        cpu_s=round(rusage.ru_utime + rusage.ru_stime, 3),
        peak_rss_mb=_peak_rss_mb(rusage.ru_maxrss, parent_maxrss, hwm_kb),
        log_path=log_path,
        tail=list(tail),
    )


def _pump(proc: subprocess.Popen[str], lines: queue.Queue[str | None]) -> None:
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            lines.put(line.rstrip("\n"))
    finally:
        lines.put(None)


def _kill_group(pid: int, term_grace_s: float) -> tuple[int, resource.struct_rusage]:
    """SIGTERM the process group, SIGKILL it if the leader is still alive later."""
    for sig, grace in ((signal.SIGTERM, term_grace_s), (signal.SIGKILL, None)):
        _signal_group(pid, sig)
        deadline = None if grace is None else time.monotonic() + grace
        while deadline is None or time.monotonic() < deadline:
            done, status, ru = os.wait4(pid, 0 if deadline is None else os.WNOHANG)
            if done == pid:
                if sig is signal.SIGTERM:
                    # The leader is gone; make sure no stragglers survive it.
                    _signal_group(pid, signal.SIGKILL)
                return status, ru
            time.sleep(_POLL_S)
    raise AssertionError("unreachable")


def _signal_group(pid: int, sig: signal.Signals) -> None:
    # ESRCH: the group is gone. macOS answers EPERM for a group of zombies.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)


def _vm_hwm_kb(pid: int) -> int:
    """Peak resident memory of a running process on Linux (0 where unknown)."""
    if sys.platform != "linux":
        return 0
    try:
        with Path(f"/proc/{pid}/status").open(encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _peak_rss_mb(child_maxrss: int, parent_maxrss: int, sampled_hwm_kb: int) -> float | None:
    """Best estimate of the child's peak RSS in MB.

    macOS reports ru_maxrss in bytes, per process. On Linux it is in KB and a
    child inherits its parent's high-water mark at fork, so a value not above
    the parent's own peak says nothing; then use the VmHWM samples instead
    (which can miss growth in the last half second).
    """
    if sys.platform == "darwin":
        return round(child_maxrss / (1024 * 1024), 1)
    if child_maxrss > parent_maxrss:
        return round(child_maxrss / 1024, 1)
    if sampled_hwm_kb > 0:
        return round(sampled_hwm_kb / 1024, 1)
    return None
