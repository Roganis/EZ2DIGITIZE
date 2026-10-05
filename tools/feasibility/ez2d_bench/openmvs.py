# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""OpenMVS stages on CPU: import, densify, mesh, (refine), texture.

Consumes the `dense/` folder of a COLMAP run. All outputs go into the run
directory: scene.mvs, scene_dense.ply, scene_mesh.ply, scene_textured.*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ez2d_bench import cameras
from ez2d_bench.meshinfo import mesh_counts, ply_counts
from ez2d_bench.plan import Dataset, OpenMVSRun
from ez2d_bench.record import Recorder, RunFailed
from ez2d_bench.tools import Toolbox


def run(
    rec: Recorder, tb: Toolbox, dataset: Dataset, cfg: OpenMVSRun, colmap_dir: Path
) -> dict[str, Any]:
    work = rec.run_dir
    dense = colmap_dir / "dense"
    if not (dense / "sparse").is_dir():
        raise RunFailed(f"{dense} has no undistorted model; did the COLMAP run finish?")
    metrics: dict[str, Any] = {}

    # --- import ---------------------------------------------------------------
    rec.step("InterfaceCOLMAP", [
        tb.openmvs("InterfaceCOLMAP"),
        "-i", dense,
        "-o", work / "scene.mvs",
        "--image-folder", dense / "images",
        "-w", work,
    ], cwd=work)  # fmt: skip

    # --- densify --------------------------------------------------------------
    densify = tb.openmvs("DensifyPointCloud")
    cmd: list[str | Path] = [
        densify, work / "scene.mvs",
        "-o", work / "scene_dense.mvs",
        "-w", work,
        "--resolution-level", str(cfg.resolution_level),
    ]  # fmt: skip
    if cfg.max_resolution:
        cmd += ["--max-resolution", str(cfg.max_resolution)]
    if cfg.threads:
        cmd += ["--max-threads", str(cfg.threads)]
    if cfg.masks:
        cmd += _prepare_masks(rec, tb, densify, dataset, colmap_dir)
    rec.step("DensifyPointCloud", [*cmd, *cfg.extra.get("DensifyPointCloud", [])], cwd=work)
    dense_ply = _expect(work, "scene_dense", ".ply")
    metrics["dense_points"] = ply_counts(dense_ply).get("vertex", 0)

    # --- mesh -----------------------------------------------------------------
    reconstruct = tb.openmvs("ReconstructMesh")
    cmd = [reconstruct, work / "scene_dense.mvs", "-o", work / "scene_mesh.mvs", "-w", work]
    if tb.has_option(reconstruct, "--pointcloud-file"):
        cmd += ["--pointcloud-file", dense_ply]
    rec.step("ReconstructMesh", [*cmd, *cfg.extra.get("ReconstructMesh", [])], cwd=work)
    mesh = _expect(work, "scene_mesh", ".ply")
    counts = mesh_counts(mesh)
    metrics["mesh_vertices"] = counts.get("vertex", 0)
    metrics["mesh_faces"] = counts.get("face", 0)

    # --- refine (optional, slow) ----------------------------------------------
    if cfg.refine:
        refine = tb.openmvs("RefineMesh")
        cmd = [
            refine,
            *_scene_and_mesh(tb, refine, work, mesh, "scene_mesh.mvs"),
            "-o",
            work / "scene_refined.mvs",
            "-w",
            work,
            "--resolution-level",
            str(cfg.refine_resolution_level),
        ]
        if cfg.threads:
            cmd += ["--max-threads", str(cfg.threads)]
        rec.step("RefineMesh", [*cmd, *cfg.extra.get("RefineMesh", [])], cwd=work)
        mesh = _expect(work, "scene_refined", ".ply")
        metrics["refined_faces"] = mesh_counts(mesh).get("face", 0)

    # --- texture --------------------------------------------------------------
    if cfg.texture:
        texture = tb.openmvs("TextureMesh")
        mesh_mvs = "scene_refined.mvs" if cfg.refine else "scene_mesh.mvs"
        cmd = [
            texture,
            *_scene_and_mesh(tb, texture, work, mesh, mesh_mvs),
            "-o",
            work / "scene_textured.mvs",
            "-w",
            work,
        ]
        suffix = ".ply"
        if tb.has_option(texture, "--export-type"):
            cmd += ["--export-type", cfg.texture_export]
            suffix = f".{cfg.texture_export}"
        if cfg.threads:
            cmd += ["--max-threads", str(cfg.threads)]
        rec.step("TextureMesh", [*cmd, *cfg.extra.get("TextureMesh", [])], cwd=work)
        textured = _expect(work, "scene_textured", suffix)
        metrics["textured_faces"] = mesh_counts(textured).get("face", 0)
        textures = [p for p in work.glob("scene_textured*") if p.suffix.lower() in (".png", ".jpg")]
        metrics["texture_files"] = len(textures)
        metrics["texture_mb"] = round(sum(p.stat().st_size for p in textures) / 2**20, 1)
        metrics["result"] = str(textured)
    else:
        metrics["result"] = str(mesh)
    return metrics


def _scene_and_mesh(
    tb: Toolbox, exe: Path, work: Path, mesh: Path, mesh_mvs: str
) -> list[str | Path]:
    """OpenMVS 2.x takes the mesh as a separate file; 1.x expects it inside the scene."""
    if tb.has_option(exe, "--mesh-file"):
        return [work / "scene_dense.mvs", "--mesh-file", mesh]
    return [work / mesh_mvs]


def _expect(work: Path, stem: str, suffix: str) -> Path:
    exact = work / f"{stem}{suffix}"
    if exact.exists():
        return exact
    candidates = sorted(work.glob(f"{stem}*{suffix}"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]
    raise RunFailed(f"expected output {exact.name} was not produced")


def _prepare_masks(
    rec: Recorder, tb: Toolbox, densify: Path, dataset: Dataset, colmap_dir: Path
) -> list[str | Path]:
    if not tb.has_option(densify, "--mask-path"):
        raise RunFailed("this DensifyPointCloud has no --mask-path option; masks unsupported")
    assert dataset.masks is not None
    out_dir = rec.run_dir / "masks"
    written, missing = undistort_masks(
        dataset.masks, colmap_dir / "sparse_txt", colmap_dir / "dense" / "sparse", out_dir
    )
    rec.note(f"undistorted {written} masks for OpenMVS ({missing} images without a mask)")
    args: list[str | Path] = ["--mask-path", out_dir]
    if tb.has_option(densify, "--ignore-mask-label"):
        args += ["--ignore-mask-label", "0"]
    return args


def undistort_masks(
    masks_dir: Path, distorted_model: Path, undistorted_model: Path, out_dir: Path
) -> tuple[int, int]:
    """Write `<image stem>.mask.png` (0 = background, 255 = object) per undistorted image.

    Images without a COLMAP mask (`<image name>.png`) get no OpenMVS mask, so
    OpenMVS uses the whole image for them.
    """
    src_cams = cameras.read_cameras_txt(distorted_model / "cameras.txt")
    src_images = cameras.read_image_cameras_txt(distorted_model / "images.txt")
    dst_cams = cameras.read_cameras_txt(undistorted_model / "cameras.txt")
    dst_images = cameras.read_image_cameras_txt(undistorted_model / "images.txt")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = missing = 0
    for name, dst_cam_id in dst_images.items():
        mask_file = masks_dir / f"{name}.png"
        if not mask_file.exists() or name not in src_images:
            missing += 1
            continue
        with Image.open(mask_file) as img:
            mask = (np.asarray(img.convert("L")) >= 128).astype(np.uint8) * 255
        warped = cameras.undistort_mask(mask, src_cams[src_images[name]], dst_cams[dst_cam_id])
        Image.fromarray(warped).save(out_dir / f"{Path(name).stem}.mask.png")
        written += 1
    return written, missing
