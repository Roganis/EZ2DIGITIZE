# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import numpy as np

from ez2d_bench.cameras import (
    Camera,
    read_cameras_txt,
    read_image_cameras_txt,
    undistort_mask,
)


def _column_index_image(width: int, height: int) -> np.ndarray:
    """Each pixel holds its own column index, so warps are easy to read back."""
    return np.tile(np.arange(width, dtype=np.uint8), (height, 1))


def test_identity_when_cameras_match() -> None:
    cam = Camera(1, "PINHOLE", 200, 100, (150.0, 150.0, 100.0, 50.0))
    mask = _column_index_image(200, 100)
    assert np.array_equal(undistort_mask(mask, cam, cam), mask)


def test_zero_distortion_matches_pinhole() -> None:
    radial = Camera(1, "OPENCV", 200, 100, (150.0, 150.0, 100.0, 50.0, 0.0, 0.0, 0.0, 0.0))
    pinhole = Camera(2, "PINHOLE", 200, 100, (150.0, 150.0, 100.0, 50.0))
    mask = _column_index_image(200, 100)
    assert np.array_equal(undistort_mask(mask, radial, pinhole), mask)


def test_positive_radial_samples_further_out() -> None:
    # With k > 0, a point away from the centre comes from further out in the
    # distorted image; the centre stays put.
    distorted = Camera(1, "SIMPLE_RADIAL", 200, 100, (150.0, 100.0, 50.0, 0.2))
    pinhole = Camera(2, "PINHOLE", 200, 100, (150.0, 150.0, 100.0, 50.0))
    out = undistort_mask(_column_index_image(200, 100), distorted, pinhole)
    assert out[50, 100] == 100
    assert out[50, 170] > 170
    assert out[50, 30] < 30


def test_mask_at_other_resolution_is_scaled() -> None:
    cam = Camera(1, "PINHOLE", 200, 100, (150.0, 150.0, 100.0, 50.0))
    half = _column_index_image(100, 50)  # mask stored at half resolution
    out = undistort_mask(half, cam, cam)
    assert out.shape == (100, 200)
    assert out[10, 120] == 60


def test_read_colmap_text_model(tmp_path: Path) -> None:
    (tmp_path / "cameras.txt").write_text(
        "# Camera list\n1 SIMPLE_RADIAL 4032 3024 3000 2016 1512 0.01\n"
    )
    (tmp_path / "images.txt").write_text(
        "# Image list\n"
        "1 1 0 0 0 0 0 0 1 IMG 0001.JPG\n"
        "10.0 20.0 -1 30.0 40.0 5\n"
        "2 1 0 0 0 0 0 0 1 sub/IMG_0002.JPG\n"
        "\n"
    )
    cams = read_cameras_txt(tmp_path / "cameras.txt")
    assert cams[1].model == "SIMPLE_RADIAL"
    assert cams[1].params == (3000.0, 2016.0, 1512.0, 0.01)
    assert read_image_cameras_txt(tmp_path / "images.txt") == {
        "IMG 0001.JPG": 1,
        "sub/IMG_0002.JPG": 1,
    }
