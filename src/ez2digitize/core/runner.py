# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The process runner: the only place that starts backend processes.

- Commands are argument lists, never a shell string.
- Each process starts in its own process group (session), so cancelling
  kills the whole tree: SIGTERM, then SIGKILL after a grace period. On
  Windows a Job Object holds the tree instead, and cancelling ends it.
- stdout and stderr are merged, split into lines (at \n, \r\n and the lone
  \r of redrawn progress lines), written to a log file, and handed line by
  line to an optional parser that turns them into progress events. Tools that
  buffer output written to a pipe can be given a pseudo-terminal instead.
- Events are delivered on the calling thread, in order. A GUI calls
  `run_process` from a worker thread and forwards the events as signals.

Platform differences live in `_PosixChild` and `_WindowsChild`: how the
tree is killed and how its CPU time and peak memory are measured. Windows
has no pseudo-terminals: `use_pty` falls back to a pipe there.
"""

from __future__ import annotations

import codecs
import contextlib
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ez2digitize.core.files import utc_now

if sys.platform == "win32":
    from ez2digitize.core.winjob import Job
else:
    import pty
    import resource

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
    use_pty: bool = False,
) -> ProcessResult:
    """Run `argv` to completion (or cancellation) and return what happened.

    `env` entries are added to the current environment. The log file is
    overwritten. Raises ProcessStartError if the command can't be started.

    `use_pty` gives the command a pseudo-terminal instead of a pipe. Tools
    that write through C stdio (OpenMVS) buffer their output in blocks when
    it isn't a terminal, so without it their progress arrives only at exit.
    """
    args = [str(a) for a in argv]
    if not args:
        raise ValueError("empty command")
    emit = on_event or (lambda _event: None)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    child_type = _child_type()

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {command_line(args)}\n\n")
        log.flush()
        started_at, start = utc_now(), time.monotonic()
        if use_pty and sys.platform != "win32":
            read_fd, child_out = pty.openpty()
        else:
            read_fd, child_out = os.pipe()
        try:
            child = child_type(args, cwd=cwd, env=env, output=child_out)
        except OSError as exc:
            os.close(read_fd)
            log.write(f"cannot start {args[0]}: {exc}\n")
            raise ProcessStartError(f"cannot start {args[0]}: {exc}") from exc
        finally:
            # The child has its own copy; EOF comes once it (and anything it
            # started) closes it.
            os.close(child_out)

        lines: queue.Queue[str | None] = queue.Queue()
        reader = threading.Thread(target=_pump, args=(read_fd, lines), daemon=True)
        reader.start()
        emit(Started(argv=args, pid=child.pid))

        def handle(line: str) -> None:
            log.write(line + "\n")
            if lines.empty():
                # Keep log.txt current: it is what the user reads when a step
                # hangs or the session dies. Flushing only when caught up keeps
                # bursts of output cheap.
                log.flush()
            tail.append(line)
            emit(Output(line))
            if parse_line is not None and (progress := parse_line(line)) is not None:
                emit(progress)

        exited = False
        cancelled = False
        eof = False
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
            if not exited:
                now = time.monotonic()
                if now >= next_sample:
                    child.sample()
                    next_sample = now + _MEMORY_SAMPLE_S
                if child.poll():
                    exited, exited_at = True, now
                elif cancel is not None and cancel.cancelled and not cancelled:
                    cancelled = True
                    log.write("\n[cancelled, stopping the process and what it started]\n")
                    log.flush()
                    child.stop(term_grace_s)
                    exited, exited_at = True, time.monotonic()
            elif time.monotonic() - exited_at > _DRAIN_S:
                # The process is done but something it started still holds
                # the output pipe open. A finished stage leaves nothing behind.
                child.kill_rest()
                break
        if not exited:  # EOF arrived first: the process closed stdout
            child.wait()
        # Anything left in the queue after EOF or the drain timeout.
        while True:
            try:
                rest = lines.get_nowait()
            except queue.Empty:
                break
            if rest is not None:
                handle(rest)
        wall = time.monotonic() - start
        exit_code = child.exit_code
        cpu_s, peak_rss_mb = child.usage()
        child.close()
        log.write(f"\n[exit code {exit_code}{', cancelled' if cancelled else ''}]\n")

    return ProcessResult(
        argv=args,
        exit_code=exit_code,
        cancelled=cancelled,
        started=started_at,
        finished=utc_now(),
        wall_s=round(wall, 3),
        cpu_s=round(cpu_s, 3),
        peak_rss_mb=peak_rss_mb,
        log_path=log_path,
        tail=list(tail),
    )


class _Child:
    """A started backend process and the tree under it (platform part)."""

    pid: int
    exit_code: int

    def __init__(
        self, args: list[str], *, cwd: Path | None, env: Mapping[str, str] | None, output: int
    ) -> None:
        raise NotImplementedError

    def poll(self) -> bool:
        """True once the process has exited (then `exit_code` is set)."""
        raise NotImplementedError

    def wait(self) -> None:
        raise NotImplementedError

    def sample(self) -> None:
        """Note the current memory use (where it can't be read at the end)."""

    def stop(self, grace_s: float) -> None:
        """Cancel: end the process and everything it started."""
        raise NotImplementedError

    def kill_rest(self) -> None:
        """End whatever the (exited) process left running."""
        raise NotImplementedError

    def usage(self) -> tuple[float, float | None]:
        """CPU seconds and peak memory in MB (None if unknown)."""
        raise NotImplementedError

    def close(self) -> None:
        pass


def _popen(
    args: list[str],
    cwd: Path | None,
    env: Mapping[str, str] | None,
    output: int,
    *,
    start_new_session: bool = False,
    creationflags: int = 0,
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(  # noqa: S603 - argument list, never a shell
        args,
        cwd=cwd,
        env={**os.environ, **env} if env else None,
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=output,
        start_new_session=start_new_session,
        creationflags=creationflags,
    )


if sys.platform != "win32":

    class _PosixChild(_Child):
        """Its own session (process group), reaped with wait4 for its rusage."""

        def __init__(
            self, args: list[str], *, cwd: Path | None, env: Mapping[str, str] | None,
            output: int,
        ) -> None:  # fmt: skip
            self.parent_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.proc = _popen(args, cwd, env, output, start_new_session=True)
            self.pid = self.proc.pid
            self.exit_code = 0
            self.rusage: resource.struct_rusage | None = None
            self.hwm_kb = 0

        def _reaped(self, status: int, rusage: resource.struct_rusage) -> None:
            self.rusage = rusage
            self.exit_code = os.waitstatus_to_exitcode(status)
            self.proc.returncode = self.exit_code  # Popen must not reap it again

        def poll(self) -> bool:
            pid, status, rusage = os.wait4(self.pid, os.WNOHANG)
            if pid == self.pid:
                self._reaped(status, rusage)
                return True
            return False

        def wait(self) -> None:
            _, status, rusage = os.wait4(self.pid, 0)
            self._reaped(status, rusage)

        def sample(self) -> None:
            self.hwm_kb = max(self.hwm_kb, _vm_hwm_kb(self.pid))

        def stop(self, grace_s: float) -> None:
            self._reaped(*_kill_group(self.pid, grace_s))

        def kill_rest(self) -> None:
            _signal_group(self.pid, signal.SIGKILL)

        def usage(self) -> tuple[float, float | None]:
            if self.rusage is None:
                return 0.0, None
            cpu = self.rusage.ru_utime + self.rusage.ru_stime
            return cpu, _peak_rss_mb(self.rusage.ru_maxrss, self.parent_maxrss, self.hwm_kb)

else:

    class _WindowsChild(_Child):
        """In a Job Object, which holds the tree and counts its CPU and memory."""

        def __init__(
            self, args: list[str], *, cwd: Path | None, env: Mapping[str, str] | None,
            output: int,
        ) -> None:  # fmt: skip
            no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            self.proc = _popen(args, cwd, env, output, creationflags=no_window | group)
            self.pid = self.proc.pid
            self.exit_code = 0
            try:
                self.job: Job | None = Job(self.pid)
            except OSError:
                self.job = None  # still runs; cancelling ends the process only

        def poll(self) -> bool:
            code = self.proc.poll()
            if code is None:
                return False
            self.exit_code = code
            return True

        def wait(self) -> None:
            self.exit_code = self.proc.wait()

        def stop(self, grace_s: float) -> None:
            # Windows tools get no polite signal: end the tree at once.
            if self.job is not None:
                self.job.terminate()
            else:
                self.proc.kill()
            self.wait()

        def kill_rest(self) -> None:
            if self.job is not None:
                self.job.terminate()

        def usage(self) -> tuple[float, float | None]:
            return self.job.usage() if self.job is not None else (0.0, None)

        def close(self) -> None:
            if self.job is not None:
                self.job.close()


def command_line(args: Sequence[str]) -> str:
    """The command as its platform's shell would take it (for logs)."""
    return subprocess.list2cmdline(args) if sys.platform == "win32" else shlex.join(args)


def run_quick(argv: Sequence[str | Path], *, timeout_s: float = 30.0) -> str:
    """Run a short command (help, version) and return its combined output.

    For probing backends, not for stages: no log, no events. The exit code is
    ignored because many tools exit non-zero after printing help. Runs in a
    temporary folder because OpenMVS tools write a log file into the current
    one. Raises ProcessStartError if the command can't be started or times out.
    """
    args = [str(a) for a in argv]
    try:
        with tempfile.TemporaryDirectory(prefix="ez2d-probe-") as scratch:
            done = subprocess.run(  # noqa: S603 - argument list, never a shell
                args,
                cwd=scratch,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise ProcessStartError(f"{args[0]} did not answer within {timeout_s:.0f} s") from exc
    except OSError as exc:
        raise ProcessStartError(f"cannot start {args[0]}: {exc}") from exc
    return done.stdout + done.stderr


def _pump(fd: int, lines: queue.Queue[str | None]) -> None:
    """Read output from `fd` until EOF and queue it line by line, then None."""
    splitter = _LineSplitter()
    try:
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError:  # Linux reports EIO on a pty once the child is gone
                break
            if not chunk:
                break
            for line in splitter.feed(chunk):
                lines.put(line)
        for line in splitter.close():
            lines.put(line)
    finally:
        os.close(fd)
        lines.put(None)


class _LineSplitter:
    """Splits a byte stream into lines at \n, \r\n and lone \r.

    Lone \r is how CLI tools redraw a progress line; each redraw becomes a
    line. A pseudo-terminal turns \n into \r\n, which must stay one break,
    so a trailing \r is held back until the next chunk shows what follows.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending = ""

    def feed(self, chunk: bytes) -> list[str]:
        return self._split(self._pending + self._decoder.decode(chunk), final=False)

    def close(self) -> list[str]:
        return self._split(self._pending + self._decoder.decode(b"", final=True), final=True)

    def _split(self, text: str, *, final: bool) -> list[str]:
        if not final and text.endswith("\r"):
            text, self._pending = text[:-1], "\r"
        else:
            self._pending = ""
        parts = re.split(r"\r\n|\r|\n", text)
        rest = parts.pop()
        if final:
            if rest:
                parts.append(rest)
        else:
            self._pending = rest + self._pending
        return parts


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


def _child_type() -> type[_Child]:
    if sys.platform == "win32":
        return _WindowsChild
    else:  # an else, which mypy reads as "not on Windows"
        return _PosixChild


if sys.platform != "win32":

    def _kill_group(pid: int, term_grace_s: float) -> tuple[int, Any]:
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
