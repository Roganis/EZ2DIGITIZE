# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Place the cameras with VGGT, for EZ2DIGITIZE (run by bin/run, in the plugin's .venv).

All photos go through VGGT at once: it predicts every camera and a depth
map per photo in one pass, with no feature matching. vggt_colmap turns that
into the COLMAP model the camera placement slot asks for.

The photos are padded to a square and scaled to 518 px, as VGGT's own
COLMAP export does (load_and_preprocess_images_square), so no part of a
portrait or landscape photo is cropped away. The weights are the ones
chosen at install (weights.txt): VGGT-1B-Commercial (needs Hugging Face
access, approved by Meta, and HF_TOKEN) or VGGT-1B (non-commercial).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import vggt_colmap
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

RESOLUTION = 518  # VGGT's input size
WEIGHTS = {"commercial": "facebook/VGGT-1B-Commercial", "research": "facebook/VGGT-1B"}
STEPS = 4


def step(n: int, what: str) -> None:
    print(f"step {n} of {STEPS}: {what}", flush=True)


def device() -> str:
    if torch.cuda.is_available():  # NVIDIA, or AMD with a ROCm build of PyTorch
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    print("no GPU found: running VGGT on the CPU, which is very slow", flush=True)
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    choice = (here / "weights.txt").read_text().strip() if (here / "weights.txt").is_file() else ""
    if choice not in WEIGHTS:
        print("run install_vggt.py first: it sets up the plugin and the weights", flush=True)
        return 1
    names = args.image_list.read_text(encoding="utf-8").split("\n")
    names = [n for n in names if n.strip()]
    paths = [str(args.images / name) for name in names]

    where = device()
    step(1, f"loading {WEIGHTS[choice]} on {where}")
    model = VGGT.from_pretrained(WEIGHTS[choice]).to(where).eval()
    step(2, f"reading {len(paths)} photos")
    images, coords = load_and_preprocess_images_square(paths, RESOLUTION)
    images = images.to(where)
    sizes = [(int(c[4]), int(c[5])) for c in coords.tolist()]
    dtype = torch.float32
    if where == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    step(3, "predicting cameras and depth")
    with torch.no_grad():
        with torch.autocast(device_type=where, dtype=dtype, enabled=dtype != torch.float32):
            tokens, start = model.aggregator(images[None])
        pose = model.camera_head(tokens)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose, images.shape[-2:])
        depth, confidence = model.depth_head(tokens, images[None], start)
    prediction = vggt_colmap.Prediction(
        names=names,
        sizes=sizes,
        resolution=RESOLUTION,
        extrinsic=extrinsic[0].float().cpu().numpy().astype(np.float64),
        intrinsic=intrinsic[0].float().cpu().numpy().astype(np.float64),
        depth=depth[0, ..., 0].float().cpu().numpy().astype(np.float64),
        confidence=confidence[0].float().cpu().numpy().astype(np.float64),
        colors=(images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8),
    )
    step(4, "writing the COLMAP model")
    points = vggt_colmap.sparse_points(prediction)
    vggt_colmap.write_model(args.output / "sparse" / "0", prediction, points)
    print(f"{len(names)} cameras, {len(points.xyz)} points", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
