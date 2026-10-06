# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""HEIC/HEIF photos (iPhones): a JPEG copy the reconstruction can read.

COLMAP and OpenMVS don't read HEIC. On import, each HEIC photo is decoded
(pillow-heif, libheif) and saved as a JPEG next to it in the bundle; the
original stays, bit for bit, marked as left out (`excluded`). The two
entries name each other in their metadata (`converted_to`,
`converted_from`). EXIF is carried over with its orientation reset to 1,
because the decoder already turns the pixels upright.
"""

from __future__ import annotations

import functools
from pathlib import Path

from PIL import ExifTags, Image

HEIF_SUFFIXES = frozenset({".heic", ".heif"})
JPEG_QUALITY = 95


@functools.cache
def _register() -> None:
    import pillow_heif  # imported on first use: it loads libheif

    pillow_heif.register_heif_opener()


def is_heif(name: str) -> bool:
    return Path(name).suffix.lower() in HEIF_SUFFIXES


def to_jpeg(source: Path, target: Path) -> None:
    """Decode `source` (HEIC/HEIF) and write it as a JPEG with its EXIF."""
    _register()
    with Image.open(source) as image:
        exif = image.getexif()
        exif[ExifTags.Base.Orientation] = 1
        rgb = image.convert("RGB")
    rgb.save(target, "JPEG", quality=JPEG_QUALITY, exif=exif, subsampling=0)
