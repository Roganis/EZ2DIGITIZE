# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import struct
from pathlib import Path

import pytest

from ez2digitize.backends.colmap_model import (
    Camera,
    read_cameras,
    read_image_cameras,
    read_images,
)
from ez2digitize.backends.common import BackendError


def _write_model(folder: Path) -> None:
    folder.mkdir()
    cameras = struct.pack("<Q", 2)
    cameras += struct.pack("<IiQQ4d", 1, 2, 4000, 3000, 3200.0, 2000.0, 1500.0, -0.05)
    cameras += struct.pack("<IiQQ8d", 7, 4, 1920, 1080, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    (folder / "cameras.bin").write_bytes(cameras)
    images = struct.pack("<Q", 2)
    for image_id, camera_id, name, points in ((3, 1, "cap/a.jpg", 2), (9, 7, "cap/é.jpg", 0)):
        images += struct.pack("<I7dI", image_id, 1, 0, 0, 0, 0.5, 0.5, 0.5, camera_id)
        images += name.encode() + b"\0" + struct.pack("<Q", points)
        images += struct.pack("<ddQ", 1.0, 2.0, 5) * points
    (folder / "images.bin").write_bytes(images)


def test_read_model(tmp_path: Path) -> None:
    _write_model(tmp_path / "0")
    cameras = read_cameras(tmp_path / "0")
    assert cameras[1] == Camera("SIMPLE_RADIAL", 4000, 3000, (3200.0, 2000.0, 1500.0, -0.05))
    assert cameras[7].model == "OPENCV" and len(cameras[7].params) == 8
    assert cameras[1].to_text() == "SIMPLE_RADIAL 4000 3000 3200.0 2000.0 1500.0 -0.05"
    assert read_image_cameras(tmp_path / "0") == {"cap/a.jpg": 1, "cap/é.jpg": 7}
    poses = read_images(tmp_path / "0")
    assert poses["cap/a.jpg"].qvec == (1, 0, 0, 0)
    assert poses["cap/a.jpg"].tvec == (0.5, 0.5, 0.5)


def test_read_model_errors(tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="cannot read"):
        read_cameras(tmp_path)
    (tmp_path / "cameras.bin").write_bytes(struct.pack("<Q", 1) + b"\0" * 5)
    with pytest.raises(BackendError, match="truncated"):
        read_cameras(tmp_path)
    (tmp_path / "cameras.bin").write_bytes(struct.pack("<QIiQQ", 1, 1, 99, 1, 1))
    with pytest.raises(BackendError, match="unsupported camera model id 99"):
        read_cameras(tmp_path)
