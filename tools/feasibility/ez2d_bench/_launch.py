# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal launcher that measures one command's own peak memory.

Usage: python -I -S _launch.py STATS_JSON -- COMMAND [ARGS...]

Why: on Linux a process inherits its parent's peak-RSS high-water mark when
it is forked, so measuring a backend spawned directly from a large Python
process reports the parent's size for small steps. This launcher is tiny
(~10 MB), forks the command, waits for it, and writes the command's
rusage to STATS_JSON. It runs with -I -S and imports only the stdlib.
"""

import json
import os
import sys


def main() -> int:
    stats_path, sep, *argv = sys.argv[1:]
    if sep != "--" or not argv:
        print("usage: _launch.py STATS_JSON -- COMMAND [ARGS...]", file=sys.stderr)
        return 2
    sys.stdout.flush()
    pid = os.fork()
    if pid == 0:
        try:
            os.execvp(argv[0], argv)
        except OSError as exc:
            print(f"cannot run {argv[0]}: {exc}", file=sys.stderr, flush=True)
        os._exit(127)
    _, status, ru = os.wait4(pid, 0)
    exit_code = os.waitstatus_to_exitcode(status)
    with open(stats_path, "w") as fh:  # noqa: PTH123 - keep the launcher import-light
        json.dump(
            {
                "exit_code": exit_code,
                "ru_maxrss": ru.ru_maxrss,
                "cpu_s": ru.ru_utime + ru.ru_stime,
            },
            fh,
        )
    return exit_code if exit_code >= 0 else 128 - exit_code


if __name__ == "__main__":
    sys.exit(main())
