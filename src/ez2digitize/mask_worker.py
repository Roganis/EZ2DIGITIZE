# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The masking worker: background removal with an ONNX model, in its own process.

Started by `ez2digitize.masks` through the process runner, never imported by
the app itself: the model needs about 1 GB while it runs, and a crash
in ONNX Runtime must not take the app down.

    python -m ez2digitize.mask_worker --model isnet.onnx --jobs jobs.json

`jobs.json` lists `{"image": ..., "mask": ...}` pairs. Each mask is written
as an 8-bit PNG, white for the object, in the image's raw pixel layout (the
layout COLMAP reads, which ignores EXIF orientation). The model itself sees
the photo upright. Per image the worker prints `mask <n>/<total> <name>`
and finally writes `report.json` next to `jobs.json`: per mask the share
of the photo kept (`coverage`) and how much of the model's answer was
neither clearly object nor clearly background (`uncertain`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from PIL import Image, ImageOps

REPORT_FILE = "report.json"
# The transpose that undoes ImageOps.exif_transpose for each EXIF orientation.
UNDO_ORIENTATION = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_90,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_270,
}
ORIENTATION_TAG = 0x0112

FloatArray = npt.NDArray[np.float32]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mask-worker")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024, help="the model's input size")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--grow", type=float, default=0.003, help="share of the long side")
    parser.add_argument("--threads", type=int, default=0, help="0: ONNX Runtime decides")
    args = parser.parse_args(argv)

    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))["jobs"]
    options = ort.SessionOptions()
    # Without the arena ISNet peaks at 1.0 GB instead of 1.5 GB, at about the
    # same speed; one image at a time doesn't need it.
    options.enable_cpu_mem_arena = False
    if args.threads > 0:
        options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    print(f"model {args.model.name} loaded, {len(jobs)} images", flush=True)

    report: dict[str, dict[str, float]] = {}
    for n, job in enumerate(jobs, 1):
        image, out = Path(job["image"]), Path(job["mask"])
        print(f"mask {n}/{len(jobs)} {image.name}", flush=True)
        mask, stats = make_mask(
            session, image, size=args.size, threshold=args.threshold, grow=args.grow
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        mask.save(out)
        report[job["key"]] = stats
    (args.jobs.parent / REPORT_FILE).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("done", flush=True)
    return 0


def make_mask(
    session: ort.InferenceSession, path: Path, *, size: int, threshold: float, grow: float
) -> tuple[Image.Image, dict[str, float]]:
    """The mask of one photo, in its raw pixel layout, and its statistics."""
    with Image.open(path) as img:
        orientation = int(img.getexif().get(ORIENTATION_TAG, 1))
        raw_size = img.size
        # JPEGs decode much faster at a reduced scale; the model sees 1024 px.
        img.draft("RGB", (size * 2, size * 2))
        upright = ImageOps.exif_transpose(img).convert("RGB")
    probability = predict(session, upright, size)
    stats = {
        "coverage": float((probability >= threshold).mean()),
        "uncertain": float(((probability > 0.1) & (probability < 0.9)).mean()),
    }
    # Grow the object a little so its outline (and thin parts) stay in. At
    # the model's resolution: a max filter on the full-size photo is slow.
    radius = round(grow * size)
    if radius > 0:
        probability = grow_max(probability, radius)
    # Back to the photo's full size and raw layout, then threshold.
    prob_image = Image.fromarray(probability)  # float32: mode "F"
    if orientation in UNDO_ORIENTATION:
        prob_image = prob_image.transpose(UNDO_ORIENTATION[orientation])
    full = prob_image.resize(raw_size, Image.Resampling.BILINEAR)
    mask = Image.fromarray(np.where(np.asarray(full) >= threshold, 255, 0).astype(np.uint8))
    return mask, stats


def grow_max(values: FloatArray, radius: int) -> FloatArray:
    """Maximum over a (2 * radius + 1) square around each pixel (separable)."""
    window = 2 * radius + 1
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(values, pad, mode="edge")
        values = np.lib.stride_tricks.sliding_window_view(padded, window, axis=axis).max(axis=-1)
    return values


def predict(session: ort.InferenceSession, image: Image.Image, size: int) -> FloatArray:
    """Probability of 'object' per pixel, at size x size."""
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    # ISNet's normalisation: mean 0.5, std 1.
    batch = (pixels - 0.5).transpose(2, 0, 1)[np.newaxis]
    name = session.get_inputs()[0].name
    outputs: list[Any] = session.run(None, {name: batch})
    result: FloatArray = np.clip(outputs[0][0, 0], 0.0, 1.0).astype(np.float32)
    return result


if __name__ == "__main__":
    sys.exit(main())
