# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading COLMAP's binary sparse models (cameras.bin, images.bin).

Only what the pipeline needs: each camera's model and parameters, and which
camera each registered image uses. The format is little-endian; see
`src/colmap/scene/reconstruction_io_binary.cc` in COLMAP.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from ez2digitize.backends.common import BackendError

# Model id -> (name, number of parameters), from src/colmap/sensor/models.h.
CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
    12: ("SIMPLE_DIVISION", 4),
    13: ("DIVISION", 5),
    14: ("SIMPLE_FISHEYE", 3),
    15: ("FISHEYE", 4),
    16: ("EUCM", 6),
}


@dataclass(frozen=True)
class Camera:
    model: str
    width: int
    height: int
    params: tuple[float, ...]

    def to_text(self) -> str:
        """`MODEL WIDTH HEIGHT PARAMS...`, as COLMAP's text formats write it."""
        values = " ".join(repr(p) for p in self.params)
        return f"{self.model} {self.width} {self.height} {values}"


def read_cameras(model_dir: Path) -> dict[int, Camera]:
    """Cameras of a binary model by camera id."""
    cameras = {}
    with _open(model_dir / "cameras.bin") as fh:
        for _ in range(int(_read(fh, "<Q")[0])):
            camera_id, model_id, width, height = (int(v) for v in _read(fh, "<IiQQ"))
            if model_id not in CAMERA_MODELS:
                raise BackendError(f"{model_dir}: unsupported camera model id {model_id}")
            name, num_params = CAMERA_MODELS[model_id]
            params = tuple(float(v) for v in _read(fh, f"<{num_params}d"))
            cameras[camera_id] = Camera(name, width, height, params)
    return cameras


@dataclass(frozen=True)
class ImagePose:
    """A registered image: world-to-camera rotation (unit quaternion w, x, y, z)
    and translation, as COLMAP stores them, and its camera."""

    camera_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]


def read_images(model_dir: Path) -> dict[str, ImagePose]:
    """Pose and camera of every registered image, by image name."""
    images = {}
    with _open(model_dir / "images.bin") as fh:
        for _ in range(int(_read(fh, "<Q")[0])):
            # image_id, rotation (qw, qx, qy, qz), translation (x, y, z), camera_id
            values = _read(fh, "<I7dI")
            name = _read_name(fh)
            num_points = int(_read(fh, "<Q")[0])
            fh.seek(num_points * 24, 1)  # x, y (double), point3D_id (uint64)
            q = tuple(float(v) for v in values[1:5])
            t = tuple(float(v) for v in values[5:8])
            images[name] = ImagePose(int(values[8]), (q[0], q[1], q[2], q[3]), (t[0], t[1], t[2]))
    return images


def read_image_cameras(model_dir: Path) -> dict[str, int]:
    """Camera id of every registered image, by image name."""
    return {name: pose.camera_id for name, pose in read_images(model_dir).items()}


def _open(path: Path) -> BinaryIO:
    try:
        return path.open("rb")
    except OSError as exc:
        raise BackendError(f"cannot read COLMAP model file {path}: {exc}") from exc


def _read(fh: BinaryIO, fmt: str) -> tuple[int | float, ...]:
    size = struct.calcsize(fmt)
    data = fh.read(size)
    if len(data) != size:
        raise BackendError(f"{getattr(fh, 'name', 'model file')}: truncated")
    values: tuple[int | float, ...] = struct.unpack(fmt, data)
    return values


def _read_name(fh: BinaryIO) -> str:
    chars = bytearray()
    while (byte := fh.read(1)) != b"\0":
        if not byte:
            raise BackendError(f"{getattr(fh, 'name', 'model file')}: truncated image name")
        chars += byte
    return chars.decode("utf-8")
