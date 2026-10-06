# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Locating backend executables, shared by the backend modules.

Search order for every tool:

1. an explicit path (from the user's settings),
2. an `EZ2D_*` environment variable (development, the benchmark harness),
3. the backends bundled with a packaged app (`<bundle>/backends/bin`),
4. `PATH`.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any


class BackendError(Exception):
    """A backend is missing, unusable, or was given inputs it can't handle."""


class BackendMissing(BackendError):
    pass


def bundled_bin_dir() -> Path | None:
    """`backends/bin` inside a PyInstaller bundle, if running from one.

    Linux (AppImage): `_internal/backends`. macOS (.app): `Contents/Resources/
    backends`, outside `Contents/Frameworks` (which _MEIPASS points to),
    because code signing treats everything there as nested code.
    """
    base = getattr(sys, "_MEIPASS", None)
    if base is None:
        return None
    for candidate in (
        Path(base) / "backends" / "bin",
        Path(base).parent / "Resources" / "backends" / "bin",
    ):
        if candidate.is_dir():
            return candidate
    return None


def find_tool(
    name: str,
    *,
    explicit: Path | None = None,
    env_var: str | None = None,
    extra_dirs: Iterable[Path] = (),
) -> Path | None:
    """Absolute path of executable `name` (see the module docstring for the order).

    Absolute because stages run with their own folder as working directory.
    """
    if explicit is not None:
        found = executable(explicit)
        return found.absolute() if found else None
    if env_var and (value := os.environ.get(env_var)):
        found = executable(Path(value).expanduser())
        return found.absolute() if found else None
    for folder in (bundled_bin_dir(), *extra_dirs):
        if folder is not None and (found := executable(folder / name)):
            return found.absolute()
    which = shutil.which(name)
    return Path(which).absolute() if which else None


# On Windows a tool is named without its suffix: COLMAP is colmap.exe (and
# the test suite's stand-ins are .cmd scripts).
WINDOWS_SUFFIXES = (".exe", ".cmd", ".bat")


def executable(path: Path) -> Path | None:
    """`path` if it is an executable file; on Windows also `path` + .exe etc.

    Windows has no executable bit (every file passes X_OK): there the
    suffix decides.
    """
    if sys.platform == "win32":
        if path.suffix.lower() in WINDOWS_SUFFIXES and path.is_file():
            return path
        for suffix in WINDOWS_SUFFIXES:
            candidate = path.with_name(path.name + suffix)
            if candidate.is_file():
                return candidate
        return None
    if path.is_file() and os.access(path, os.X_OK):
        return path
    return None


def result_parameters(options: Any) -> dict[str, Any]:
    """A stage's option dataclass as the parameters that decide its result.

    Thread counts are left out: they don't change the output, and they are
    picked per machine (see core.resources), so including them would make a
    stage re-run whenever the free memory changes.
    """
    return {k: v for k, v in asdict(options).items() if k != "threads"}
