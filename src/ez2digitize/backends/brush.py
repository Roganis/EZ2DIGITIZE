# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Brush: Gaussian splat training on the GPU (Vulkan, or Metal on macOS).

Brush (Apache-2.0) trains on the undistorted COLMAP output, the same input
OpenMVS gets. It expects a COLMAP dataset folder with `images/` and
`sparse/0/`; the splat stage builds one in its own folder from relative
symlinks to the undistort stage, so nothing is copied.

    splat/   dataset/images -> ../../undistort/images
             dataset/sparse/0 -> ../../../undistort/sparse
             splat.ply (the trained splats)

Brush is pre-1.0 and its CLI changes between versions; this module is
written for PINNED_VERSION. It only prints progress to a terminal, so it
runs under a pseudo-terminal; its progress bar ("1200/30000 Steps") is
parsed after stripping the terminal colour codes.

It runs on whatever Vulkan device it picks. A software renderer (llvmpipe)
works but is hundreds of times slower; the pipeline refuses it unless asked.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from ez2digitize.backends.common import BackendMissing, find_tool, result_parameters
from ez2digitize.core.project import Project
from ez2digitize.core.runner import ProcessStartError, Progress, run_quick
from ez2digitize.core.stage import Backend, StageManifest, StageSpec, stage_input

NAME = "brush"
PINNED_VERSION = "0.3.0"
ENV_VAR = "EZ2D_BRUSH"
EXECUTABLE = "brush_app"
SPLAT_FILE = "splat.ply"


@dataclass(frozen=True)
class Brush:
    path: Path
    version: str

    @property
    def backend(self) -> Backend:
        return Backend(NAME, self.version)

    @property
    def supported(self) -> bool:
        return self.version == PINNED_VERSION


@dataclass(frozen=True)
class SplatOptions:
    # Brush's defaults: 30k steps, images up to 1920 px, up to 10M splats.
    total_steps: int = 30_000
    max_resolution: int = 1920
    max_splats: int = 10_000_000
    sh_degree: int = 3


def parse_version(text: str) -> str | None:
    match = re.search(r"brush(?:-cli|_app)?\s+v?(\d+\.\d+\.\d+)", text)
    return match.group(1) if match else None


def locate(explicit: Path | None = None) -> Brush:
    if explicit is None and (env := os.environ.get(ENV_VAR)):
        explicit = Path(env).expanduser()
    path = find_tool(EXECUTABLE, explicit=explicit)
    if path is None:
        raise BackendMissing(f"Brush not found (looked in ${ENV_VAR}, the bundle, PATH)")
    try:
        text = run_quick([path, "--version"])
    except ProcessStartError as exc:
        raise BackendMissing(f"Brush at {path} can't be run: {exc}") from exc
    version = parse_version(text)
    if version is None:
        raise BackendMissing(f"no Brush version in the output of {path} --version")
    return Brush(path=path, version=version)


def train(
    brush: Brush,
    project: Project,
    undistorted: StageManifest,
    *,
    stage: str = "splat",
    options: SplatOptions | None = None,
) -> StageSpec:
    """Train splats on an undistort stage's images and model."""
    options = options or SplatOptions()
    stage_dir = project.stage_dir(stage)
    source = project.stage_dir(undistorted.stage)

    def prepare(folder: Path) -> None:
        dataset = folder / "dataset"
        (dataset / "sparse").mkdir(parents=True)
        # Relative, so a moved project still works.
        (dataset / "images").symlink_to(Path("..") / ".." / source.name / "images")
        (dataset / "sparse" / "0").symlink_to(Path("..") / ".." / ".." / source.name / "sparse")

    argv: list[str | Path] = [
        brush.path,
        stage_dir / "dataset",
        "--total-steps", str(options.total_steps),
        "--max-resolution", str(options.max_resolution),
        "--max-splats", str(options.max_splats),
        "--sh-degree", str(options.sh_degree),
        "--export-every", str(options.total_steps),
        "--export-path", stage_dir,
        "--export-name", SPLAT_FILE,
    ]  # fmt: skip
    return StageSpec(
        name=stage,
        backend=brush.backend,
        argv=argv,
        parameters=result_parameters(options),
        inputs={"undistorted": stage_input(undistorted)},
        parse_line=BrushProgress(),
        use_pty=True,
        prepare=prepare,
        gpu=True,
    )


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_STEPS = re.compile(r"(\d+)/(\d+)\s+Steps\b")


class BrushProgress:
    """`[56s] ◍◍◍◍  1200/30000      Steps (1.3/s, 6h remaining)` -> 4 %."""

    def __init__(self) -> None:
        self._loaded = False

    def __call__(self, line: str) -> Progress | None:
        text = _ANSI.sub("", line)
        match = _STEPS.search(text)
        if match is None:
            # Brush redraws its status lines; report loading once.
            if "Completed loading" in text and not self._loaded:
                self._loaded = True
                return Progress("Loaded the photos")
            return None
        done, total = int(match.group(1)), int(match.group(2))
        if total <= 0:
            return None
        return Progress("Training splats", round(min(done / total, 1.0), 4))
