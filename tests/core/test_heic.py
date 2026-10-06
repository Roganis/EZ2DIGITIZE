# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import pillow_heif
import pytest
from PIL import ExifTags, Image

from ez2digitize.backends.colmap import image_names
from ez2digitize.core import photos
from ez2digitize.core.capture import CaptureBundle, import_files
from ez2digitize.core.project import Project


def write_heic(path: Path) -> Path:
    image = Image.effect_noise((320, 240), 60).convert("RGB")
    exif = Image.Exif()
    exif[ExifTags.Base.Make] = "Apple"
    exif[ExifTags.Base.Orientation] = 6
    exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.FocalLength] = 5.7
    heif = pillow_heif.from_pillow(image)
    heif.save(path, quality=90, exif=exif.tobytes())
    return path


def test_heic_gets_a_jpeg_copy(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "p")
    heic = write_heic(tmp_path / "IMG_0001.HEIC")
    other = tmp_path / "IMG_0002.jpg"
    Image.new("RGB", (8, 8)).save(other)
    bundle = import_files(project, [heic, other], source="upload")

    by_name = {f.name: f for f in bundle.files}
    original, copy = by_name["IMG_0001.HEIC"], by_name["IMG_0001.jpg"]
    assert original.excluded and original.metadata["converted_to"] == "IMG_0001.jpg"
    assert (bundle.root / original.name).read_bytes() == heic.read_bytes()  # untouched
    assert not copy.excluded and copy.metadata["converted_from"] == "IMG_0001.HEIC"
    assert bundle.verify() == []
    assert CaptureBundle.load(bundle.root).to_dict() == bundle.to_dict()

    with Image.open(bundle.root / copy.name) as jpeg:
        # Orientation 6 (phone held upright): decoded already turned, 240 wide.
        assert jpeg.format == "JPEG" and jpeg.size == (240, 320)
        exif = jpeg.getexif()
        assert exif[ExifTags.Base.Make] == "Apple"
        assert exif[ExifTags.Base.Orientation] == 1  # pixels already upright
    names = image_names([bundle])
    assert f"{bundle.id}/IMG_0001.jpg" in names and f"{bundle.id}/IMG_0001.HEIC" not in names
    info = photos.inspect_photo(bundle.root / copy.name)
    assert info.error is None and info.focal_mm == pytest.approx(5.7)


def test_undecodable_heic_is_kept_as_is(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "p")
    broken = tmp_path / "broken.heic"
    broken.write_bytes(b"not really heic")
    bundle = import_files(project, [broken], source="folder")
    assert [(f.name, f.excluded) for f in bundle.files] == [("broken.heic", False)]
