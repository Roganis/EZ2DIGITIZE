# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export diagnostics: what a bug report needs, zipped, without the photos.

The zip holds the project's description files (project.json, every
capture.json, stage.json and export.json), every stage's logs (the last
LOG_LIMIT bytes of each), the photo checks' findings, the tools found and
their versions, and system information. Photos, videos and reconstructed
models are never included. Logs and manifests do contain file paths, which
may include the user's name; the dialogs say so.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import sys
import zipfile
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import ez2digitize
from ez2digitize.backends import colmap, ffmpeg, openmvs
from ez2digitize.backends.common import BackendError
from ez2digitize.core import photos
from ez2digitize.core.capture import CAPTURE_FILE, CaptureError, list_bundles
from ez2digitize.core.hardware import detect_gpus
from ez2digitize.core.project import PROJECT_FILE, Project
from ez2digitize.core.resources import GIB, available_memory, cpu_threads

# Per log file: the end is what matters, and some logs reach hundreds of MB.
LOG_LIMIT = 2 * 1024 * 1024
DESCRIPTION_FILES = ("stage.json", "export.json")
LOG_SUFFIXES = (".txt", ".log")


def default_name(project: Project, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{project.root.name}-diagnostics-{stamp}.zip"


def write_diagnostics(
    project: Project,
    target: Path,
    *,
    tools: Callable[[], dict[str, Any]] | None = None,
    now: datetime | None = None,
) -> Path:
    """Write the diagnostics zip to `target` and return it.

    `tools` returns the tool report (default: look the tools up like the
    CLI does); the GUI passes one that uses its settings.
    """
    report: dict[str, Any] = {
        "created": (now or datetime.now()).astimezone().isoformat(timespec="seconds"),
        "app_version": ez2digitize.__version__,
        "system": system_info(),
        "tools": (tools or find_tools)(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
        root = project.root
        _add_file(archive, root / PROJECT_FILE, root)
        try:
            bundles = list_bundles(project)
        except CaptureError as exc:
            report["captures_error"] = str(exc)
            bundles = []
        for bundle in bundles:
            _add_file(archive, bundle.root / CAPTURE_FILE, root)
        report["photo_checks"] = [asdict(f) for f in photos.check_project(bundles)]
        for folder in (project.stages_dir, project.exports_dir):
            if not folder.is_dir():
                continue
            for path in sorted(folder.rglob("*")):
                if not path.is_file():
                    continue
                if path.name in DESCRIPTION_FILES:
                    _add_file(archive, path, root)
                elif path.suffix in LOG_SUFFIXES and folder == project.stages_dir:
                    _add_log(archive, path, root)
        archive.writestr("report.json", json.dumps(report, indent=2, default=str))
    partial.replace(target)
    return target


def system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_threads": cpu_threads(),
        "available_memory_gb": round(available_memory() / GIB, 1),
        "frozen": bool(getattr(sys, "frozen", False)),
    }
    if sys.platform != "win32":
        with contextlib.suppress(ValueError, OSError, AttributeError):
            info["total_memory_gb"] = round(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB, 1
            )
    info["gpus"] = [asdict(gpu) for gpu in detect_gpus()]
    return info


def find_tools() -> dict[str, Any]:
    """Each tool's version and location, or why it wasn't found."""
    report: dict[str, Any] = {}
    for name, locate in (
        ("colmap", lambda: colmap.locate()),
        ("openmvs", lambda: openmvs.locate()),
        ("ffmpeg", lambda: ffmpeg.locate()),
    ):
        try:
            tool = locate()
        except BackendError as exc:
            report[name] = {"error": str(exc)}
        else:
            report[name] = {k: str(v) for k, v in asdict(tool).items()}
    return report


def _add_file(archive: zipfile.ZipFile, path: Path, root: Path) -> None:
    if path.is_file():
        archive.write(path, path.relative_to(root).as_posix())


def _add_log(archive: zipfile.ZipFile, path: Path, root: Path) -> None:
    size = path.stat().st_size
    name = path.relative_to(root).as_posix()
    if size <= LOG_LIMIT:
        archive.write(path, name)
        return
    with path.open("rb") as log:
        log.seek(size - LOG_LIMIT)
        tail = log.read()
    note = f"[first {size - LOG_LIMIT} bytes left out]\n".encode()
    archive.writestr(name, note + tail)
