# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Brush Gaussian splat training on the undistorted COLMAP output.

Brush is pre-1.0 and its CLI changes, so only the dataset path and export
folder are passed by default. Put everything else (steps, export settings) in
the plan's `args`, after checking `brush_app --help` for the installed version.

Quality: with `--eval-split-every N --eval-save-to-disk`, Brush holds out every
Nth image and saves its renders of those views as eval_<step>/<name>.png.
Brush prints nothing when not attached to a terminal, so the PSNR against the
held-out photos is computed here from those renders.
"""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ez2d_bench.meshinfo import ply_counts
from ez2d_bench.plan import BrushRun
from ez2d_bench.record import Recorder, RunFailed
from ez2d_bench.tools import Toolbox


def prepare_dataset(colmap_dir: Path, dataset_dir: Path) -> None:
    """Lay out the standard `images/` + `sparse/0/` structure Brush expects."""
    src_model = colmap_dir / "dense" / "sparse"
    model_dir = dataset_dir / "sparse" / "0"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (src_model / name).exists():
            raise RunFailed(f"{src_model / name} missing; did the COLMAP run finish?")
        shutil.copy2(src_model / name, model_dir / name)
    images_link = dataset_dir / "images"
    if not images_link.exists():
        images_link.symlink_to(colmap_dir / "dense" / "images", target_is_directory=True)


def run(rec: Recorder, tb: Toolbox, cfg: BrushRun, colmap_dir: Path) -> dict[str, Any]:
    brush = tb.require_brush()
    dataset_dir = rec.run_dir / "dataset"
    prepare_dataset(colmap_dir, dataset_dir)

    export_dir = rec.run_dir / "export"
    export_dir.mkdir(exist_ok=True)
    cmd: list[str | Path] = [brush, dataset_dir]
    if tb.has_option(brush, "--export-path") and "--export-path" not in cfg.args:
        cmd += ["--export-path", export_dir]
    rec.step("brush", [*cmd, *cfg.args], cwd=rec.run_dir)

    exports = sorted(
        (p for p in rec.run_dir.rglob("*.ply") if dataset_dir not in p.parents),
        key=lambda p: p.stat().st_mtime,
    )
    if not exports:
        raise RunFailed(
            "Brush finished but wrote no .ply; check its export options in `brush_app --help` "
            "and add them to the plan's args"
        )
    final = exports[-1]
    metrics: dict[str, Any] = {
        "splats": ply_counts(final).get("vertex", 0),
        "export_mb": round(final.stat().st_size / 2**20, 1),
        "result": str(final),
    }
    metrics.update(eval_psnr(rec.run_dir, colmap_dir / "dense" / "images"))
    return metrics


def psnr(rendered: np.ndarray, reference: np.ndarray) -> float:
    mse = float(np.mean((rendered.astype(np.float64) - reference.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)


def eval_psnr(search_dir: Path, images_dir: Path) -> dict[str, Any]:
    """Mean PSNR of the latest eval renders against the (undistorted) photos."""
    eval_dirs = [
        (int(m.group(1)), d)
        for d in search_dir.rglob("eval_*")
        if d.is_dir() and (m := re.fullmatch(r"eval_(\d+)", d.name))
    ]
    if not eval_dirs:
        return {}
    step, latest = max(eval_dirs)
    references = {p.stem: p for p in images_dir.rglob("*") if p.is_file()}
    scores = []
    for render_path in sorted(latest.glob("*.png")):
        ref_path = references.get(render_path.stem)
        if ref_path is None:
            continue
        with Image.open(render_path) as r, Image.open(ref_path) as g:
            rendered = r.convert("RGB")
            reference = g.convert("RGB").resize(rendered.size, Image.Resampling.BOX)
        scores.append(psnr(np.asarray(rendered), np.asarray(reference)))
    if not scores:
        return {}
    return {
        "eval_step": step,
        "eval_views": len(scores),
        "eval_psnr": round(sum(scores) / len(scores), 2),
    }
