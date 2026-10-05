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
from pathlib import Path


class BackendError(Exception):
    """A backend is missing, unusable, or was given inputs it can't handle."""


class BackendMissing(BackendError):
    pass


def bundled_bin_dir() -> Path | None:
    """`backends/bin` inside a PyInstaller bundle, if running from one."""
    base = getattr(sys, "_MEIPASS", None)
    if base is None:
        return None
    candidate = Path(base) / "backends" / "bin"
    return candidate if candidate.is_dir() else None


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
        return explicit.absolute() if _is_executable(explicit) else None
    if env_var and (value := os.environ.get(env_var)):
        candidate = Path(value).expanduser()
        return candidate.absolute() if _is_executable(candidate) else None
    for folder in (bundled_bin_dir(), *extra_dirs):
        if folder is not None and _is_executable(folder / name):
            return (folder / name).absolute()
    found = shutil.which(name)
    return Path(found).absolute() if found else None


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)
