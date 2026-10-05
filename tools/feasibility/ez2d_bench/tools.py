# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Find backend executables and adapt to the options their version supports.

COLMAP, OpenMVS and Brush rename options between releases. Instead of
hard-coding one version's names, read `--help` and pick whichever candidate
the installed binary accepts.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ez2d_bench.runner import capture

OPENMVS_TOOLS = (
    "InterfaceCOLMAP",
    "DensifyPointCloud",
    "ReconstructMesh",
    "RefineMesh",
    "TextureMesh",
)
OPENMVS_SEARCH_DIRS = (
    Path("/usr/local/bin/OpenMVS"),
    Path("/usr/bin/OpenMVS"),
    Path("/opt/homebrew/bin/OpenMVS"),
)
BRUSH_NAMES = ("brush_app", "brush")


class ToolMissing(RuntimeError):
    pass


@dataclass
class Toolbox:
    colmap: Path | None = None
    glomap: Path | None = None
    openmvs_dir: Path | None = None
    brush: Path | None = None
    _help_cache: dict[tuple[str, ...], str] = field(default_factory=dict, repr=False)

    @classmethod
    def discover(
        cls,
        colmap: Path | None = None,
        openmvs_dir: Path | None = None,
        brush: Path | None = None,
    ) -> Toolbox:
        """Explicit arguments win, then EZ2D_* environment variables, then PATH."""
        colmap = colmap or _env_path("EZ2D_COLMAP") or _which("colmap")
        glomap = _env_path("EZ2D_GLOMAP") or _which("glomap")
        openmvs_dir = openmvs_dir or _env_path("EZ2D_OPENMVS_DIR") or _find_openmvs_dir()
        brush = (
            brush
            or _env_path("EZ2D_BRUSH")
            or next((p for name in BRUSH_NAMES if (p := _which(name))), None)
        )
        return cls(colmap=colmap, glomap=glomap, openmvs_dir=openmvs_dir, brush=brush)

    # --- locating -----------------------------------------------------------

    def openmvs(self, tool: str) -> Path:
        if self.openmvs_dir is None:
            raise ToolMissing(
                "OpenMVS not found. Pass --openmvs-dir or set EZ2D_OPENMVS_DIR to the "
                "folder containing DensifyPointCloud."
            )
        path = self.openmvs_dir / tool
        if not path.is_file():
            raise ToolMissing(f"{tool} not found in {self.openmvs_dir}")
        return path

    def require_colmap(self) -> Path:
        if self.colmap is None:
            raise ToolMissing("colmap not found on PATH. Pass --colmap or set EZ2D_COLMAP.")
        return self.colmap

    def require_brush(self) -> Path:
        if self.brush is None:
            raise ToolMissing(
                "Brush not found (looked for brush_app/brush on PATH). Pass --brush or set "
                "EZ2D_BRUSH."
            )
        return self.brush

    # --- option detection ---------------------------------------------------

    def help_text(self, exe: Path, *args: str) -> str:
        key = (str(exe), *args)
        if key not in self._help_cache:
            self._help_cache[key] = capture([exe, *args, "--help"])
        return self._help_cache[key]

    def has_option(self, exe: Path, option: str, *args: str) -> bool:
        return option_in_help(self.help_text(exe, *args), option)

    def pick_option(self, exe: Path, candidates: list[str], *args: str) -> str:
        text = self.help_text(exe, *args)
        for option in candidates:
            if option_in_help(text, option):
                return option
        raise ToolMissing(
            f"{exe.name} {' '.join(args)}: none of {candidates} is supported by this version"
        )

    def colmap_commands(self) -> str:
        return self.help_text(self.require_colmap(), "help")

    def has_colmap_global_mapper(self) -> bool:
        return re.search(r"^\s*global_mapper\b", self.colmap_commands(), re.MULTILINE) is not None

    # --- versions -----------------------------------------------------------

    def versions(self) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        if self.colmap:
            text = capture([self.colmap, "help"])
            match = re.search(r"COLMAP\s+(\S+).*?\n\s*\(([^)]*)\)", text, re.DOTALL)
            out["colmap"] = f"{match.group(1)} ({match.group(2)})" if match else _first_line(text)
        else:
            out["colmap"] = None
        out["glomap"] = str(self.glomap) if self.glomap else None
        if self.openmvs_dir and (self.openmvs_dir / "InterfaceCOLMAP").is_file():
            text = capture([self.openmvs_dir / "InterfaceCOLMAP", "--help"])
            match = re.search(r"OpenMVS[^\n]*?v?(\d+\.\d+(?:\.\d+)?)", text)
            out["openmvs"] = match.group(1) if match else _first_line(text)
        else:
            out["openmvs"] = None
        out["brush"] = _first_line(capture([self.brush, "--version"])) if self.brush else None
        ffmpeg = _which("ffmpeg")
        out["ffmpeg"] = _first_line(capture([ffmpeg, "-version"])) if ffmpeg else None
        return out


def option_in_help(help_text: str, option: str) -> bool:
    """True if `option` appears as a whole option name in help output."""
    return re.search(rf"(?<![\w.-]){re.escape(option)}(?![\w.-])", help_text) is not None


def _which(name: str) -> Path | None:
    found = shutil.which(name)
    return Path(found) if found else None


def _env_path(var: str) -> Path | None:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else None


def _find_openmvs_dir() -> Path | None:
    on_path = _which("DensifyPointCloud")
    if on_path:
        return on_path.parent
    for candidate in OPENMVS_SEARCH_DIRS:
        if (candidate / "DensifyPointCloud").is_file():
            return candidate
    return None


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""
