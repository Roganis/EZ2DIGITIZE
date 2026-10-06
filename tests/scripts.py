# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stand-in executables for the tests, on every platform.

POSIX runs a script with a `#!` line directly. Windows doesn't: there the
script goes next to a `.cmd` launcher that runs it with this Python, and
the launcher is what tests run (the app finds `name.cmd` for `name`).
"""

import sys
from pathlib import Path


def python_script(path: Path, body: str) -> Path:
    """An executable at `path` running `body` with this Python; returns what to run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        source = path.with_name(path.name + ".py")
        source.write_text(body)
        launcher = path.with_name(path.name + ".cmd")
        launcher.write_text(f'@"{sys.executable}" "{source}" %*\r\n')
        return launcher
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)
    return path


def printing_script(path: Path, text: str) -> Path:
    """An executable that prints `text`, whatever its arguments."""
    return python_script(path, f"print({text!r})\n")
