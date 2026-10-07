# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The MapAnything camera placement plugin (tools/plugins/mapanything-poses), without MapAnything.

Its manifest, the photos as its input (read as stored, scaled and cropped,
the focal length from EXIF), and its copy of feedforward_colmap, which
tests/test_vggt_plugin.py tests.
"""

import json
import math
from pathlib import Path

import mapanything_inputs
import numpy as np
import pytest
from feedforward_colmap import Placement
from PIL import ExifTags, Image

from ez2digitize import plugins

PLUGINS = Path(__file__).parents[1] / "tools" / "plugins"
PLUGIN = PLUGINS / "mapanything-poses"


def test_manifest() -> None:
    plugin = plugins.read_plugin(PLUGIN)
    assert (plugin.id, plugin.slot, plugin.gpu) == ("mapanything-poses", "poses", True)
    covers = {lic.covers: lic.spdx for lic in plugin.licenses}
    assert covers["plugin code"] == "GPL-3.0-or-later"
    assert covers["MapAnything code"] == "Apache-2.0"
    assert sorted(spdx for c, spdx in covers.items() if c.startswith("model weights")) == [
        "Apache-2.0",
        "CC-BY-NC-4.0",
    ]
    assert not plugin.free  # the research weights are non-commercial
    assert plugin.with_masks == ()  # MapAnything has no use for masks


def test_shared_conversion_is_the_same_in_both_plugins() -> None:
    """Each plugin is installed on its own, so each carries feedforward_colmap.py."""
    ours = (PLUGIN / "feedforward_colmap.py").read_bytes()
    assert ours == (PLUGINS / "vggt-poses" / "feedforward_colmap.py").read_bytes()


def _photo(path: Path, size: tuple[int, int], *, focal_35mm: int | None = None) -> None:
    """A gradient: red grows with x, green with y, so a pixel tells where it came from."""
    width, height = size
    x = (np.arange(width) + 0.5) / width
    y = (np.arange(height) + 0.5) / height
    pixels = np.zeros((height, width, 3), np.uint8)
    pixels[..., 0] = np.round(255 * x)[None, :]
    pixels[..., 1] = np.round(255 * y)[:, None]
    image = Image.fromarray(pixels)
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = 6  # "turn it", which must be ignored
    if focal_35mm is not None:
        exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.FocalLengthIn35mmFilm] = focal_35mm
    image.save(path, quality=100, exif=exif)


def test_exif_focal(tmp_path: Path) -> None:
    _photo(tmp_path / "phone.jpg", (400, 300), focal_35mm=26)
    _photo(tmp_path / "frame.jpg", (400, 300))
    with Image.open(tmp_path / "phone.jpg") as image:
        focal = mapanything_inputs.exif_focal(image)
    # 26 mm on 35 mm film, by the diagonal: 500 px here, 43.27 mm there.
    assert focal == pytest.approx(26 / math.hypot(36, 24) * 500)
    with Image.open(tmp_path / "frame.jpg") as image:
        assert mapanything_inputs.exif_focal(image) is None


def test_photos_are_read_as_stored_then_scaled_and_cropped(tmp_path: Path) -> None:
    path = tmp_path / "IMG_0001.jpg"
    _photo(path, (300, 400), focal_35mm=28)  # portrait as stored, cropped to landscape
    assert mapanything_inputs.average_aspect([path]) == pytest.approx(0.75)
    size = (98, 70)
    photo = mapanything_inputs.prepare(path, size)
    assert photo.pixels.shape == (70, 98, 3)  # not turned by the EXIF orientation
    assert photo.placement == Placement.cover((300, 400), size)
    assert photo.focal == pytest.approx(28 / math.hypot(36, 24) * 500)
    # Every input pixel shows the part of the photo the placement says.
    v, u = np.mgrid[0:70, 0:98].astype(np.float64) + 0.5
    x, y = photo.placement.to_photo(u, v)
    assert np.abs(photo.pixels[..., 0] - 255 * x / 300).max() < 4
    assert np.abs(photo.pixels[..., 1] - 255 * y / 400).max() < 4


def test_known_poses(tmp_path: Path) -> None:
    """Poses go to MapAnything only when every photo has one, from one recording."""
    priors = tmp_path / "priors.json"
    names = ["v/frame_0001.jpg", "v/frame_0002.jpg"]
    pose = [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]]

    def write(images: dict[str, dict[str, object]]) -> None:
        priors.write_text(json.dumps({"version": 1, "images": images}))

    write({n: {"camera_to_world": pose, "frame": "v", "metric": False} for n in names})
    poses, why = mapanything_inputs.known_poses(priors, names)
    assert poses is not None and why == "known poses for all 2 photos"
    assert not mapanything_inputs.poses_are_metric(priors, names)
    write({n: {"camera_to_world": pose, "frame": "v", "metric": True} for n in names})
    assert mapanything_inputs.poses_are_metric(priors, names)
    assert poses[1][:3, 3].tolist() == [0.5, 0.0, 2.0] and poses[1][3].tolist() == [0, 0, 0, 1]

    write({names[0]: {"camera_to_world": pose, "frame": "v", "metric": False}})
    assert mapanything_inputs.known_poses(priors, names) == (
        None,
        "1 of 2 photos have no known pose",
    )
    write({n: {"camera_to_world": pose, "frame": n[-5], "metric": False} for n in names})
    assert mapanything_inputs.known_poses(priors, names)[1] == (
        "the known poses come from separate recordings"
    )
    write({})
    assert mapanything_inputs.known_poses(priors, names) == (None, "no known poses")
    assert mapanything_inputs.known_poses(tmp_path / "none.json", names)[0] is None
