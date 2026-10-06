# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render a tiny synthetic dataset to smoke-test the pipeline.

A randomly textured box standing on a textured mat, photographed from two
camera rings, with exact object masks. It checks that the plumbing works
end to end; it says nothing about quality on real photos.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

F64 = NDArray[np.float64]

BOX_MIN = np.array([-0.4, -0.3, 0.0])
BOX_MAX = np.array([0.4, 0.3, 0.7])
MAT_HALF = 2.0


def _noise_texture(rng: np.random.Generator, size: int = 512) -> F64:
    """Multi-octave value noise, RGB in [0, 1]; gives SIFT plenty of blobs."""
    tex = np.zeros((size, size, 3))
    for cells, weight in ((4, 0.4), (16, 0.35), (64, 0.25)):
        grid = rng.random((cells + 1, cells + 1, 3))
        coords = np.linspace(0, cells, size, endpoint=False)
        i0 = coords.astype(int)
        t = (coords - i0)[:, None]
        rows = grid[i0] * (1 - t[:, :, None]) + grid[i0 + 1] * t[:, :, None]
        cols = rows[:, i0] * (1 - t[None, :, :]) + rows[:, i0 + 1] * t[None, :, :]
        tex += weight * cols
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    return np.clip((tex - 0.5) * 1.6 + 0.5, 0, 1)


def _sample(tex: F64, u: F64, v: F64) -> F64:
    size = tex.shape[0]
    iu = np.clip((u * size).astype(int), 0, size - 1)
    iv = np.clip((v * size).astype(int), 0, size - 1)
    sampled: F64 = tex[iv, iu]
    return sampled


def _look_at(eye: F64, target: F64) -> tuple[F64, F64, F64]:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up, forward


def render_view(
    eye: F64, textures: list[F64], width: int, height: int, focal: float, supersample: int = 2
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    right, up, forward = _look_at(eye, np.array([0.0, 0.0, 0.3]))
    w, h, f = width * supersample, height * supersample, focal * supersample
    jj, ii = np.mgrid[0:h, 0:w].astype(np.float64)
    x = (ii + 0.5 - w / 2) / f
    y = (jj + 0.5 - h / 2) / f
    dirs = x[..., None] * right - y[..., None] * up + forward
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    color = np.full((h, w, 3), 0.55)  # featureless grey background
    depth = np.full((h, w), np.inf)
    mask = np.zeros((h, w), dtype=bool)

    # Ground mat at z = 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ground = -eye[2] / dirs[..., 2]
    hit = (t_ground > 0) & np.isfinite(t_ground)
    px = eye[0] + t_ground * dirs[..., 0]
    py = eye[1] + t_ground * dirs[..., 1]
    hit &= (np.abs(px) < MAT_HALF) & (np.abs(py) < MAT_HALF)
    mat = _sample(textures[0], (px + MAT_HALF) / (2 * MAT_HALF), (py + MAT_HALF) / (2 * MAT_HALF))
    color[hit] = 0.85 * mat[hit]
    depth[hit] = t_ground[hit]

    # Box (slab method); the entry face is the axis with the largest near t.
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (BOX_MIN - eye) / dirs
        t2 = (BOX_MAX - eye) / dirs
    t_near = np.minimum(t1, t2)
    t_far = np.maximum(t1, t2)
    t_enter = np.nanmax(t_near, axis=-1)
    t_exit = np.nanmin(t_far, axis=-1)
    box_hit = (t_enter <= t_exit) & (t_enter > 0) & (t_enter < depth)
    axis = np.argmax(np.nan_to_num(t_near, nan=-np.inf), axis=-1)
    p = eye + t_enter[..., None] * dirs
    rel = (p - BOX_MIN) / (BOX_MAX - BOX_MIN)
    light = np.array([0.4, -0.5, 0.75])
    light /= np.linalg.norm(light)
    for ax, (a, b) in enumerate(((1, 2), (0, 2), (0, 1))):
        sel = box_hit & (axis == ax)
        side = dirs[..., ax] > 0  # entered through the min face
        for is_min in (True, False):
            face = sel & (side == is_min)
            normal = np.zeros(3)
            normal[ax] = -1.0 if is_min else 1.0
            shade = 0.45 + 0.55 * max(float(normal @ light), 0.0)
            tex = textures[1 + 2 * ax + (0 if is_min else 1)]
            color[face] = shade * _sample(tex, rel[..., a], rel[..., b])[face]
    mask |= box_hit

    rgb = (np.clip(color, 0, 1) * 255).reshape(height, supersample, width, supersample, 3)
    rgb = rgb.mean(axis=(1, 3))
    m = mask.reshape(height, supersample, width, supersample).mean(axis=(1, 3)) >= 0.5
    return rgb.astype(np.uint8), (m * 255).astype(np.uint8)


def generate(
    out_dir: Path,
    views_per_ring: int = 16,
    width: int = 640,
    height: int = 480,
    seed: int = 7,
) -> int:
    """Write images/, masks/ (COLMAP naming) and the ground truth; return image count.

    Ground truth for `bench.py eval`: `ground_truth.ply`, the box without its
    bottom face (no camera sees it), and `ground_truth.json`, every camera's
    centre and world-to-camera rotation (COLMAP convention: x right, y down,
    z forward), in scene units.
    """
    rng = np.random.default_rng(seed)
    textures = [_noise_texture(rng) for _ in range(7)]  # mat + 6 box faces
    images_dir, masks_dir = out_dir / "images", out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    focal = 0.9 * width
    # The focal length in EXIF, as a camera writes it: COLMAP's prior is
    # FocalLengthIn35mmFilm / 35 * the longer side. Without it COLMAP guesses
    # 1.2 * the longer side, a third too long, and its global mapper (which
    # needs good priors) lost every point on macOS with ALIKED features.
    exif = Image.Exif()
    exif[0x010F] = "EZ2DIGITIZE"  # Make
    exif[0x0110] = "synthetic scene"  # Model
    exif.get_ifd(0x8769)[0xA405] = round(focal / max(width, height) * 35)
    cameras: dict[str, dict[str, object]] = {}
    n = 0
    for radius, height_z, offset in ((2.6, 1.0, 0.0), (2.2, 2.0, 0.5)):
        for k in range(views_per_ring):
            angle = 2 * math.pi * (k + offset) / views_per_ring
            eye = np.array([radius * math.cos(angle), radius * math.sin(angle), height_z])
            rgb, mask = render_view(eye, textures, width, height, focal)
            name = f"view_{n:03d}.jpg"
            Image.fromarray(rgb).save(images_dir / name, quality=95, exif=exif)
            Image.fromarray(mask).save(masks_dir / f"{name}.png")
            right, up, forward = _look_at(eye, np.array([0.0, 0.0, 0.3]))
            cameras[name] = {
                "center": eye.tolist(),
                "rotation": np.stack([right, -up, forward]).tolist(),
                "focal_px": focal,
                "size": [width, height],
            }
            n += 1
    _write_ground_truth(out_dir, cameras)
    return n


def _write_ground_truth(out_dir: Path, cameras: dict[str, dict[str, object]]) -> None:
    lo, hi = BOX_MIN, BOX_MAX
    corners = np.array([[x, y, z] for z in (lo[2], hi[2]) for y in (lo[1], hi[1])
                        for x in (lo[0], hi[0])])  # fmt: skip
    # Corner index = 4*z + 2*y + x (0 = min, 1 = max). Outward-facing quads,
    # all sides but the bottom (z = 0, standing on the mat).
    quads = [(4, 5, 7, 6), (0, 4, 6, 2), (1, 3, 7, 5), (0, 1, 5, 4), (2, 6, 7, 3)]
    faces = [tri for a, b, c, d in quads for tri in ((a, b, c), (a, c, d))]
    lines = ["ply", "format ascii 1.0", f"element vertex {len(corners)}",
             "property float x", "property float y", "property float z",
             f"element face {len(faces)}", "property list uchar int vertex_indices",
             "end_header"]  # fmt: skip
    lines += [" ".join(f"{c:.6f}" for c in v) for v in corners]
    lines += [f"3 {a} {b} {c}" for a, b, c in faces]
    (out_dir / "ground_truth.ply").write_text("\n".join(lines) + "\n")
    info = {"units": "scene", "object": "box without its bottom face", "cameras": cameras}
    (out_dir / "ground_truth.json").write_text(json.dumps(info, indent=1))
