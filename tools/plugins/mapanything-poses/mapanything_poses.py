# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Place the cameras with MapAnything, for EZ2DIGITIZE (run by bin/run, in the plugin's .venv).

All photos go through MapAnything at once: it predicts every camera and a
depth map per photo in one pass, with no feature matching. Unlike VGGT it
also takes what is already known: here each photo's focal length from EXIF,
when every photo has one (settings.json can turn that off). feedforward_colmap
turns the prediction into the COLMAP model the camera placement slot asks
for.

The photos are scaled and cropped to the size MapAnything's own loader
would pick (518 px wide or high, at the supported aspect ratio nearest the
photos' average), but read as stored, not turned by EXIF, to match COLMAP
(mapanything_inputs). The weights are the ones chosen at install:
map-anything-apache (Apache-2.0) or map-anything (CC-BY-NC-4.0).
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import feedforward_colmap  # noqa: E402 - after the allocator setting
import mapanything_inputs  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mapanything.models import MapAnything  # noqa: E402
from mapanything.utils.image import find_closest_aspect_ratio  # noqa: E402
from uniception.models.encoders.image_normalizations import (  # noqa: E402
    IMAGE_NORMALIZATION_DICT,
)

RESOLUTION_SET = 518  # MapAnything's input sizes: 518 px on the long side
WEIGHTS = {"apache": "facebook/map-anything-apache", "research": "facebook/map-anything"}
STEPS = 4


def step(n: int, what: str) -> None:
    print(f"step {n} of {STEPS}: {what}", flush=True)


def device() -> str:
    if torch.cuda.is_available():  # NVIDIA, or AMD with a ROCm build of PyTorch
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    print("no GPU found: running MapAnything on the CPU, which is very slow", flush=True)
    return "cpu"


def settings(here: Path) -> dict[str, object]:
    path = here / "settings.json"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    chosen = settings(here)
    weights = chosen.get("weights")
    if weights not in WEIGHTS:
        print("run install_mapanything.py first: it sets up the plugin and the weights", flush=True)
        return 1
    names = args.image_list.read_text(encoding="utf-8").split("\n")
    names = [n for n in names if n.strip()]
    paths = [args.images / name for name in names]

    where = device()
    step(1, f"loading {WEIGHTS[str(weights)]} on {where}")
    model = MapAnything.from_pretrained(WEIGHTS[str(weights)]).to(where).eval()
    norm_type = model.encoder.data_norm_type
    norm = IMAGE_NORMALIZATION_DICT[norm_type]
    mean = torch.as_tensor(norm.mean, dtype=torch.float32)[:, None, None]
    std = torch.as_tensor(norm.std, dtype=torch.float32)[:, None, None]

    step(2, f"reading {len(paths)} photos")
    size = find_closest_aspect_ratio(mapanything_inputs.average_aspect(paths), RESOLUTION_SET)
    size = (int(size[0]), int(size[1]))
    photos = [mapanything_inputs.prepare(path, size) for path in paths]
    use_focal = chosen.get("exif_focal", True) and all(p.focal for p in photos)
    if chosen.get("exif_focal", True) and not use_focal:
        print("not every photo has a focal length in EXIF: MapAnything estimates it", flush=True)
    views = []
    for i, photo in enumerate(photos):
        pixels = torch.from_numpy(photo.pixels).permute(2, 0, 1).float() / 255
        view = {
            "img": ((pixels - mean) / std)[None],
            "data_norm_type": [norm_type],
            "true_shape": np.array([[size[1], size[0]]], dtype=np.int32),
            "idx": i,
            "instance": str(i),
        }
        if use_focal and photo.focal:
            place = photo.placement
            k = place.input_intrinsics(photo.focal, photo.focal, place.width / 2, place.height / 2)
            view["intrinsics"] = torch.from_numpy(k).float()[None]
        views.append(view)

    step(3, f"predicting cameras and depth at {size[0]} x {size[1]}")
    with torch.no_grad():
        outputs = model.infer(
            views,
            memory_efficient_inference=True,
            use_amp=where != "cpu",
            amp_dtype="bf16",  # MapAnything falls back to fp16 where bf16 isn't supported
            apply_mask=True,
            mask_edges=True,
        )

    def stacked(key: str) -> np.ndarray:
        return np.stack([out[key][0].float().cpu().numpy() for out in outputs]).astype(np.float64)

    cam_from_world = np.linalg.inv(stacked("camera_poses"))  # it predicts world from camera
    valid = stacked("mask")[..., 0] > 0.5  # leaves out ambiguous pixels and depth edges
    prediction = feedforward_colmap.Prediction(
        names=names,
        placements=[p.placement for p in photos],
        extrinsic=cam_from_world[:, :3, :],
        intrinsic=stacked("intrinsics"),
        depth=np.where(valid, stacked("depth_z")[..., 0], 0.0),
        confidence=stacked("conf"),
        colors=np.stack([p.pixels for p in photos]),
    )
    step(4, "writing the COLMAP model")
    points = feedforward_colmap.sparse_points(prediction)
    feedforward_colmap.write_model(args.output / "sparse" / "0", prediction, points)
    print(f"{len(names)} cameras, {len(points.xyz)} points", flush=True)
    scale = float(outputs[0]["metric_scaling_factor"].reshape(-1)[0])
    print(f"MapAnything's scale estimate: the model is in metres (factor {scale:.3g})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
