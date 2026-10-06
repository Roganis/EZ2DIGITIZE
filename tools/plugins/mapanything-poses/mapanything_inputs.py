# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The photos as MapAnything's input, with their focal length from EXIF.

Pillow and numpy only, so it is tested without a GPU or MapAnything
(tests/test_mapanything_plugin.py).

MapAnything's own loader turns photos upright by their EXIF orientation;
COLMAP reads the pixels as stored, and the camera placement slot's model
must match COLMAP. So the photos are read here as stored, then scaled to
cover MapAnything's input size and cropped to it, centred, as its loader
does (feedforward_colmap.Placement.cover).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from feedforward_colmap import Placement
from numpy.typing import NDArray
from PIL import ExifTags, Image

# A 35 mm equivalent focal length gives the same angle of view across the
# diagonal as on a 36 x 24 mm frame (CIPA DC-008).
FILM_DIAGONAL_MM = math.hypot(36.0, 24.0)


def exif_focal(image: Image.Image) -> float | None:
    """The focal length in the photo's pixels, from EXIF's 35 mm equivalent.

    Phones and most cameras write it; the focal length in millimetres alone
    would need the sensor's size, which EXIF rarely has. Video frames have
    neither.
    """
    exif = image.getexif().get_ifd(ExifTags.IFD.Exif)
    value = exif.get(ExifTags.Base.FocalLengthIn35mmFilm)
    if not isinstance(value, int | float):
        return None
    equivalent = float(value)
    if not 0 < equivalent < 10000:  # absent (0) or nonsense
        return None
    return equivalent / FILM_DIAGONAL_MM * math.hypot(*image.size)


def average_aspect(paths: list[Path]) -> float:
    """Width over height, averaged over the photos, as stored."""
    ratios = []
    for path in paths:
        with Image.open(path) as image:
            ratios.append(image.width / image.height)
    return sum(ratios) / len(ratios)


@dataclass
class Photo:
    pixels: NDArray[np.uint8]  # (H, W, 3): the input, at MapAnything's size
    placement: Placement  # the photo in the input
    focal: float | None  # in the photo's pixels, from EXIF


def prepare(path: Path, size: tuple[int, int]) -> Photo:
    """The photo scaled and cropped to `size` (width, height), as stored."""
    with Image.open(path) as image:
        focal = exif_focal(image)
        rgb = image.convert("RGB")
    placement = Placement.cover(rgb.size, size)
    resized = rgb.resize(size, Image.Resampling.LANCZOS, box=placement.source_box(size))
    return Photo(np.array(resized, dtype=np.uint8), placement, focal)  # writable, for torch
