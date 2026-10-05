# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""COLMAP stages: features, matching, mapping, undistortion. CPU only.

Output layout of a run directory:
    database.db
    sparse/<n>/        one folder per reconstructed model
    sparse.ply         points of the best model, for viewing
    sparse_txt/        best model as text (used for mask undistortion)
    dense/             undistorted images + model: input for OpenMVS and Brush
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ez2d_bench.plan import ColmapRun, Dataset
from ez2d_bench.record import Recorder, RunFailed
from ez2d_bench.tools import Toolbox, ToolMissing

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

_ANALYZER_FIELDS = {
    "registered_images": r"Registered images:\s*(\d+)",
    "points": r"Points:\s*(\d+)",
    "observations": r"Observations:\s*(\d+)",
    "mean_track_length": r"Mean track length:\s*([\d.]+)",
    "mean_observations_per_image": r"Mean observations per image:\s*([\d.]+)",
    "mean_reprojection_error_px": r"Mean reprojection error:\s*([\d.]+)",
}


def list_images(folder: Path) -> list[Path]:
    """Images COLMAP will read (it recurses into subfolders)."""
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def parse_model_analyzer(text: str) -> dict[str, float]:
    stats: dict[str, float] = {}
    for key, pattern in _ANALYZER_FIELDS.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            stats[key] = float(value) if "." in value else int(value)
    return stats


def find_models(sparse_dir: Path) -> list[Path]:
    return sorted(
        d
        for d in sparse_dir.iterdir()
        if d.is_dir() and ((d / "cameras.bin").exists() or (d / "cameras.txt").exists())
    )


def run(rec: Recorder, tb: Toolbox, dataset: Dataset, cfg: ColmapRun) -> dict[str, Any]:
    colmap = tb.require_colmap()
    if not dataset.images.is_dir():
        raise RunFailed(f"image folder not found: {dataset.images}")
    images = list_images(dataset.images)
    if not images:
        raise RunFailed(f"no images in {dataset.images}")

    run_dir = rec.run_dir
    db = run_dir / "database.db"
    sparse = run_dir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)

    # --- features -------------------------------------------------------------
    fe = "feature_extractor"
    cmd: list[str | Path] = [
        colmap, fe,
        "--database_path", db,
        "--image_path", dataset.images,
        "--ImageReader.single_camera", "1" if cfg.single_camera else "0",
        tb.pick_option(colmap, ["--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"], fe),
        "0",
        tb.pick_option(
            colmap, ["--FeatureExtraction.max_image_size", "--SiftExtraction.max_image_size"], fe
        ),
        str(cfg.max_image_size),
    ]  # fmt: skip
    if cfg.camera_model:
        cmd += ["--ImageReader.camera_model", cfg.camera_model]
    if cfg.masks:
        assert dataset.masks is not None
        cmd += ["--ImageReader.mask_path", dataset.masks]
        _check_masks(rec, images, dataset)
    if cfg.threads:
        threads_opt = ["--FeatureExtraction.num_threads", "--SiftExtraction.num_threads"]
        cmd += [tb.pick_option(colmap, threads_opt, fe), str(cfg.threads)]
    rec.step(fe, [*cmd, *cfg.extra.get(fe, [])])

    # --- matching -------------------------------------------------------------
    matcher = f"{cfg.matcher}_matcher"
    gpu_opt = ["--FeatureMatching.use_gpu", "--SiftMatching.use_gpu"]
    cmd = [colmap, matcher, "--database_path", db, tb.pick_option(colmap, gpu_opt, matcher), "0"]
    if cfg.matcher == "sequential":
        cmd += [
            "--SequentialMatching.overlap", str(cfg.sequential_overlap),
            "--SequentialMatching.loop_detection", "0",
        ]  # fmt: skip
    if cfg.threads:
        threads_opt = ["--FeatureMatching.num_threads", "--SiftMatching.num_threads"]
        cmd += [tb.pick_option(colmap, threads_opt, matcher), str(cfg.threads)]
    rec.step(matcher, [*cmd, *cfg.extra.get(matcher, [])])

    # --- mapping --------------------------------------------------------------
    io_args: list[str | Path] = [
        "--database_path", db, "--image_path", dataset.images, "--output_path", sparse,
    ]  # fmt: skip
    if cfg.mapper == "incremental":
        cmd = [colmap, "mapper", *io_args]
        if cfg.threads:
            cmd += ["--Mapper.num_threads", str(cfg.threads)]
        rec.step("mapper", [*cmd, *cfg.extra.get("mapper", [])])
    elif tb.has_colmap_global_mapper():
        rec.step(
            "global_mapper",
            [colmap, "global_mapper", *io_args, *cfg.extra.get("global_mapper", [])],
        )
    elif tb.glomap is not None:
        rec.note("this COLMAP has no global_mapper; using standalone GLOMAP")
        rec.step(
            "global_mapper", [tb.glomap, "mapper", *io_args, *cfg.extra.get("global_mapper", [])]
        )
    else:
        raise ToolMissing("global mapper needs a COLMAP with 'global_mapper' or GLOMAP on PATH")

    models = find_models(sparse)
    metrics: dict[str, Any] = {"total_images": len(images), "models": len(models)}
    if not models:
        metrics["registered_images"] = 0
        rec.record.metrics = metrics
        raise RunFailed("mapper produced no model (no image pair could be initialized)")

    # --- analysis -------------------------------------------------------------
    per_model: dict[str, dict[str, float]] = {}
    for model in models:
        step = rec.step(f"model_analyzer-{model.name}", [colmap, "model_analyzer", "--path", model])
        per_model[model.name] = parse_model_analyzer(Path(step.log).read_text(errors="replace"))
    best_name = max(per_model, key=lambda name: per_model[name].get("registered_images", 0))
    best = sparse / best_name
    metrics.update(per_model[best_name])
    metrics["best_model"] = best_name
    metrics["model_sizes"] = {k: v.get("registered_images", 0) for k, v in per_model.items()}
    if len(models) > 1:
        rec.note(f"{len(models)} disconnected models; using model {best_name} (largest)")

    rec.step(
        "export_sparse_ply",
        [
            colmap,
            "model_converter",
            "--input_path",
            best,
            "--output_path",
            run_dir / "sparse.ply",
            "--output_type",
            "PLY",
        ],
    )
    sparse_txt = run_dir / "sparse_txt"
    sparse_txt.mkdir(exist_ok=True)
    rec.step(
        "export_sparse_txt",
        [
            colmap,
            "model_converter",
            "--input_path",
            best,
            "--output_path",
            sparse_txt,
            "--output_type",
            "TXT",
        ],
    )

    # --- undistortion (shared input for OpenMVS and Brush) --------------------
    dense = run_dir / "dense"
    cmd = [
        colmap,
        "image_undistorter",
        "--image_path",
        dataset.images,
        "--input_path",
        best,
        "--output_path",
        dense,
        "--output_type",
        "COLMAP",
    ]
    if cfg.undistort_max_image_size:
        cmd += ["--max_image_size", str(cfg.undistort_max_image_size)]
    rec.step("image_undistorter", [*cmd, *cfg.extra.get("image_undistorter", [])])
    # Text copies next to the binaries: older OpenMVS InterfaceCOLMAP only reads text.
    rec.step(
        "export_dense_txt",
        [
            colmap,
            "model_converter",
            "--input_path",
            dense / "sparse",
            "--output_path",
            dense / "sparse",
            "--output_type",
            "TXT",
        ],
    )
    return metrics


def _check_masks(rec: Recorder, images: list[Path], dataset: Dataset) -> None:
    assert dataset.masks is not None
    masks = dataset.masks
    missing = [
        img
        for img in images
        if not (masks / img.relative_to(dataset.images)).with_name(img.name + ".png").exists()
    ]
    if len(missing) == len(images):
        raise RunFailed(
            f"no masks found in {dataset.masks}; COLMAP expects '<image name>.png', "
            f"e.g. {images[0].name}.png"
        )
    if missing:
        rec.note(f"{len(missing)} of {len(images)} images have no mask")
