# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stand-in executables for the tests, on every platform.

POSIX runs a script with a `#!` line directly. Windows doesn't: there the
script is a `name.cmd` (the app finds it for `name`) whose first line runs
the file itself with this Python, `-x` skipping that line. Like a POSIX
script, the file holds the whole program, so its hash changes with the
body and not with its folder (the stage tests rely on both).
"""

import sys
from pathlib import Path


def python_script(path: Path, body: str) -> Path:
    """An executable at `path` running `body` with this Python; returns what to run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        launcher = path if path.suffix == ".cmd" else path.with_name(path.name + ".cmd")
        # "exit /b" without a number keeps Python's exit code.
        launcher.write_text(f'@"{sys.executable}" -x "%~f0" %* & exit /b\n{body}')
        return launcher
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)
    return path


def printing_script(path: Path, text: str) -> Path:
    """An executable that prints `text`, whatever its arguments."""
    return python_script(path, f"print({text!r})\n")
