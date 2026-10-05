# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Background removal masks with rembg (MIT) on ONNX Runtime.

Masks are written in COLMAP's convention: `<image name>.png` (so
`IMG_0001.JPG` -> `IMG_0001.JPG.png`), white = object, black = ignored.

EXIF orientation is deliberately NOT applied: COLMAP reads the raw pixel
layout, so the mask must match the raw layout too.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ez2d_bench.colmap import list_images


def make_masks(
    images_dir: Path,
    masks_dir: Path,
    model: str = "birefnet-general",
    dilate_px: int = 4,
    threshold: int = 128,
) -> dict[str, object]:
    from rembg import new_session, remove  # heavy import, only needed here

    images = list_images(images_dir)
    if not images:
        raise RuntimeError(f"no images in {images_dir}")
    masks_dir.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    session = new_session(model)
    load_s = time.monotonic() - start

    coverage: list[float] = []
    per_image: list[float] = []
    for n, path in enumerate(images, 1):
        t0 = time.monotonic()
        with Image.open(path) as img:
            raw = img.convert("RGB")
        mask_img = remove(raw, session=session, only_mask=True)
        assert isinstance(mask_img, Image.Image)
        mask = mask_img.convert("L").point(lambda v: 255 if v >= threshold else 0)
        if dilate_px > 0:
            # Grow the object slightly so thin edges aren't cut off.
            mask = mask.filter(ImageFilter.MaxFilter(2 * dilate_px + 1))
        out = masks_dir / path.relative_to(images_dir)
        out = out.with_name(out.name + ".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        mask.save(out)
        coverage.append(float(np.asarray(mask).mean() / 255))
        per_image.append(time.monotonic() - t0)
        print(f"  mask {n}/{len(images)} {path.name}: {per_image[-1]:.1f}s, "
              f"object covers {coverage[-1]:.0%}", flush=True)  # fmt: skip

    info: dict[str, object] = {
        "model": model,
        "images": len(images),
        "model_load_s": round(load_s, 1),
        "mean_s_per_image": round(float(np.mean(per_image)), 2),
        "total_s": round(time.monotonic() - start, 1),
        "dilate_px": dilate_px,
        "coverage_min": round(min(coverage), 3),
        "coverage_max": round(max(coverage), 3),
        "suspicious": [images[i].name for i, c in enumerate(coverage) if c < 0.02 or c > 0.9],
    }
    (masks_dir.parent / f"{masks_dir.name}_info.json").write_text(json.dumps(info, indent=2))
    contact_sheet(images, images_dir, masks_dir, masks_dir.parent / f"{masks_dir.name}_preview.jpg")
    return info


def contact_sheet(
    images: list[Path], images_dir: Path, masks_dir: Path, out: Path, thumb: int = 240
) -> None:
    """Grid of thumbnails with the masked-out area tinted red, for quick review."""
    cols = min(8, len(images))
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (40, 40, 40))
    for i, path in enumerate(images):
        mask_path = masks_dir / path.relative_to(images_dir)
        mask_path = mask_path.with_name(mask_path.name + ".png")
        if not mask_path.exists():
            continue
        with Image.open(path) as img, Image.open(mask_path) as m:
            rgb = img.convert("RGB")
            rgb.thumbnail((thumb, thumb))
            mask = m.convert("L").resize(rgb.size)
        red = Image.new("RGB", rgb.size, (220, 30, 30))
        tinted = Image.blend(rgb, red, 0.6)
        composed = Image.composite(rgb, tinted, mask)
        sheet.paste(composed, ((i % cols) * thumb, (i // cols) * thumb))
    sheet.save(out, quality=85)
