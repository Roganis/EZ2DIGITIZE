# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import os
import re
import sys
import textwrap
import time
from pathlib import Path

import pytest

from ez2digitize.core.runner import (
    CancelToken,
    Event,
    Output,
    ProcessStartError,
    Progress,
    Started,
    command_line,
    run_process,
)

POSIX = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")


def py(code: str) -> list[str]:
    return [sys.executable, "-c", textwrap.dedent(code)]


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000 | 0x1000, False, pid)  # SYNCHRONIZE, query
        if not handle:
            return False
        try:
            return bool(kernel32.WaitForSingleObject(handle, 0) == 0x102)  # WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie (exited, not yet reaped by its new parent) counts as dead.
    status = Path(f"/proc/{pid}/status")
    return not (status.exists() and "\nState:\tZ" in status.read_text())


def _wait_dead(pid: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def test_success_logs_and_events(tmp_path: Path) -> None:
    events: list[Event] = []
    log = tmp_path / "logs" / "log.txt"
    result = run_process(
        py("""
            import sys
            print("one")
            print("two", file=sys.stderr, flush=True)
            print("three")
        """),
        log_path=log,
        on_event=events.append,
    )
    assert result.ok and result.exit_code == 0 and not result.cancelled
    assert isinstance(events[0], Started) and events[0].pid > 0
    lines = [e.line for e in events if isinstance(e, Output)]
    assert sorted(lines) == ["one", "three", "two"]
    assert result.tail == lines
    text = log.read_text()
    assert text.startswith(f"$ {command_line([sys.executable, '-c'])} ")
    assert "one\n" in text and "two\n" in text and text.endswith("[exit code 0]\n")
    assert result.wall_s > 0 and result.cpu_s >= 0
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", result.started)


def test_failure_exit_code(tmp_path: Path) -> None:
    result = run_process(py("raise SystemExit(3)"), log_path=tmp_path / "log.txt")
    assert result.exit_code == 3 and not result.ok


@POSIX
def test_killed_by_signal_reports_negative_code(tmp_path: Path) -> None:
    result = run_process(
        py("import os, signal; os.kill(os.getpid(), signal.SIGKILL)"),
        log_path=tmp_path / "log.txt",
    )
    assert result.exit_code == -9 and not result.ok


def test_carriage_return_progress_and_parser(tmp_path: Path) -> None:
    def parse(line: str) -> Progress | None:
        m = re.fullmatch(r"step (\d+)/4", line)
        return Progress(line, int(m.group(1)) / 4) if m else None

    events: list[Event] = []
    run_process(
        py("""
            import sys
            for i in range(1, 5):
                sys.stdout.write(f"step {i}/4\\r")
            sys.stdout.write("done\\n")
        """),
        log_path=tmp_path / "log.txt",
        on_event=events.append,
        parse_line=parse,
    )
    assert [e.fraction for e in events if isinstance(e, Progress)] == [0.25, 0.5, 0.75, 1.0]
    assert [e.line for e in events if isinstance(e, Output)][-1] == "done"


def test_env_and_cwd(tmp_path: Path) -> None:
    result = run_process(
        py("import os; print(os.environ['EZ2D_TEST'], os.getcwd(), 'PATH' in os.environ)"),
        log_path=tmp_path / "log.txt",
        cwd=tmp_path,
        env={"EZ2D_TEST": "hello"},
    )
    # getcwd() resolves symlinks (macOS: /var -> /private/var).
    assert result.tail == [f"hello {tmp_path.resolve()} True"]


def test_missing_executable(tmp_path: Path) -> None:
    with pytest.raises(ProcessStartError, match="cannot start"):
        run_process([str(tmp_path / "no-such-tool")], log_path=tmp_path / "log.txt")
    assert "cannot start" in (tmp_path / "log.txt").read_text()


def test_empty_command(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        run_process([], log_path=tmp_path / "log.txt")


def test_cancel_kills_whole_process_tree(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    cancel = CancelToken()

    def on_event(event: Event) -> None:
        if isinstance(event, Output) and event.line == "ready":
            cancel.cancel()

    started = time.monotonic()
    result = run_process(
        py(f"""
            import subprocess, sys, time
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            open({str(pid_file)!r}, "w").write(str(child.pid))
            print("ready", flush=True)
            time.sleep(60)
        """),
        log_path=tmp_path / "log.txt",
        on_event=on_event,
        cancel=cancel,
    )
    assert time.monotonic() - started < 15
    assert result.cancelled and not result.ok
    # POSIX: SIGTERM. Windows: the job is ended, exit code 1.
    assert result.exit_code == (1 if sys.platform == "win32" else -15)
    assert _wait_dead(int(pid_file.read_text()))
    assert "[cancelled" in (tmp_path / "log.txt").read_text()


@POSIX
def test_cancel_escalates_to_sigkill(tmp_path: Path) -> None:
    cancel = CancelToken()

    def on_event(event: Event) -> None:
        if isinstance(event, Output) and event.line == "ready":
            cancel.cancel()

    result = run_process(
        py("""
            import signal, time
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            print("ready", flush=True)
            time.sleep(60)
        """),
        log_path=tmp_path / "log.txt",
        on_event=on_event,
        cancel=cancel,
        term_grace_s=0.5,
    )
    assert result.cancelled and result.exit_code == -9


def test_leftover_child_holding_output_is_killed(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    started = time.monotonic()
    result = run_process(
        py(f"""
            import subprocess, sys
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            open({str(pid_file)!r}, "w").write(str(child.pid))
            print("leaving", flush=True)
        """),
        log_path=tmp_path / "log.txt",
    )
    assert time.monotonic() - started < 15
    assert result.ok and result.tail == ["leaving"]
    assert _wait_dead(int(pid_file.read_text()))


@pytest.mark.skipif(sys.platform not in ("linux", "darwin", "win32"), reason="no peak memory")
def test_peak_rss_is_the_childs(tmp_path: Path) -> None:
    result = run_process(
        py("""
            import time
            block = bytearray(300 * 1024 * 1024)
            for i in range(0, len(block), 4096):
                block[i] = 1
            time.sleep(0.6)
        """),
        log_path=tmp_path / "log.txt",
    )
    assert result.peak_rss_mb is not None
    assert 290 < result.peak_rss_mb < 600


@pytest.mark.skipif(sys.platform != "linux", reason="Linux reports KB and inherits maxrss")
def test_peak_rss_falls_back_to_samples_when_inherited() -> None:
    from ez2digitize.core.runner import _peak_rss_mb

    # Child's ru_maxrss above the parent's peak: it is the child's own.
    assert _peak_rss_mb(800 * 1024, 200 * 1024, 0) == 800.0
    # Not above: it may be the parent's, inherited at fork; use the samples.
    assert _peak_rss_mb(200 * 1024, 200 * 1024, 30 * 1024) == 30.0
    assert _peak_rss_mb(200 * 1024, 200 * 1024, 0) is None


def test_run_quick_returns_output_and_runs_elsewhere(tmp_path: Path) -> None:
    from ez2digitize.core.runner import run_quick

    out = run_quick(
        py("import os, sys; print(os.getcwd()); print('err', file=sys.stderr); exit(1)")
    )
    lines = out.splitlines()
    assert "ez2d-probe-" in lines[0] and "err" in lines


def test_run_quick_errors(tmp_path: Path) -> None:
    from ez2digitize.core.runner import run_quick

    with pytest.raises(ProcessStartError, match="cannot start"):
        run_quick([tmp_path / "missing"])
    with pytest.raises(ProcessStartError, match="did not answer"):
        run_quick(py("import time; time.sleep(5)"), timeout_s=0.2)


@pytest.mark.parametrize(
    ("chunks", "lines"),
    [
        ([b"a\nb\n"], ["a", "b"]),
        ([b"a\r\nb"], ["a", "b"]),
        ([b"a\r", b"\nb\n"], ["a", "b"]),  # \r\n split across reads
        ([b"10%\r20%\r", b"done\n"], ["10%", "20%", "done"]),
        ([b"\xc3", b"\xa9t\xc3\xa9\n"], ["été"]),  # UTF-8 split across reads
        ([b"no newline at end"], ["no newline at end"]),
        ([b"a\n\nb\n"], ["a", "", "b"]),
    ],
)
def test_line_splitter(chunks: list[bytes], lines: list[str]) -> None:
    from ez2digitize.core.runner import _LineSplitter

    splitter = _LineSplitter()
    out = [line for chunk in chunks for line in splitter.feed(chunk)]
    assert out + splitter.close() == lines


@POSIX
def test_pty_delivers_buffered_output_while_running(tmp_path: Path) -> None:
    # Python, like C stdio, buffers stdout in blocks unless it is a terminal.
    code = """
        import sys, time
        print("tty" if sys.stdout.isatty() else "pipe")
        time.sleep(1.5)
        print("end")
    """
    for use_pty, expected in ((False, "pipe"), (True, "tty")):
        seen: list[tuple[str, float]] = []
        start = time.monotonic()

        def on_event(event: Event) -> None:
            if isinstance(event, Output):
                seen.append((event.line, time.monotonic() - start))  # noqa: B023

        result = run_process(
            py(code),
            log_path=tmp_path / "log.txt",
            on_event=on_event,
            use_pty=use_pty,
            env={"PYTHONUNBUFFERED": ""},  # empty means unset
        )
        assert result.ok
        assert [line for line, _ in seen] == [expected, "end"]
        first_line_at = seen[0][1]
        assert (first_line_at < 1.0) == use_pty


def test_log_file_is_current_while_running(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    seen_in_file: list[bool] = []

    def on_event(event: Event) -> None:
        if isinstance(event, Output) and event.line == "first":
            # The process is still running (it sleeps next); the line must be on disk.
            time.sleep(0.3)
            seen_in_file.append("first" in log.read_text())

    run_process(
        py("""
            import time
            print("first", flush=True)
            time.sleep(1.5)
            print("second")
        """),
        log_path=log,
        on_event=on_event,
    )
    assert seen_in_file == [True]
