# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where a packaged app finds its bundled files, and a self-test report.

PyInstaller puts bundled files under `sys._MEIPASS` (the `_internal` folder
of a onedir build, `Contents/Frameworks` in a macOS .app). Backends are
bundled under `backends/` there.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

STARTED = time.monotonic()


def bundle_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def backend(name: str) -> Path:
    return bundle_dir() / "backends" / name


def backend_report(name: str) -> dict[str, object]:
    exe = backend(name)
    if not exe.is_file():
        return {"name": name, "found": False}
    if not os.access(exe, os.X_OK):
        return {"name": name, "found": True, "executable": False}
    try:
        out = subprocess.run(  # noqa: S603 - our own bundled binary
            [str(exe), "--version"], capture_output=True, text=True, timeout=30, check=False
        )
    except OSError as exc:
        return {"name": name, "found": True, "error": str(exc)}
    return {
        "name": name,
        "found": True,
        "exit_code": out.returncode,
        "version": (out.stdout or out.stderr).strip().splitlines()[:1],
    }


def print_report(variant: str, extra: dict[str, object] | None = None) -> None:
    report: dict[str, object] = {
        "event": "self-test",
        "variant": variant,
        "frozen": bool(getattr(sys, "frozen", False)),
        "platform": platform.platform(),
        "bundle_dir": str(bundle_dir()),
        "window_shown_s": round(time.monotonic() - STARTED, 2),
        "backend": backend_report("brush_app"),
    }
    report.update(extra or {})
    print(json.dumps(report), flush=True)
