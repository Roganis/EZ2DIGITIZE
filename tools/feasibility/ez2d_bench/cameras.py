# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""COLMAP camera models, enough to undistort masks.

`colmap image_undistorter` undistorts the photos but not the masks. OpenMVS
needs masks that line up with the undistorted photos, so we warp each mask
with the same camera models: for every pixel of the undistorted (pinhole)
image, project its ray through the original distorted camera and sample the
original mask (nearest neighbour, outside = background).

Conventions follow COLMAP: pixel (i, j) has its centre at (i + 0.5, j + 0.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]

    def focal_center(self) -> tuple[float, float, float, float]:
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            return p[0], p[0], p[1], p[2]
        if self.model in ("PINHOLE", "OPENCV"):
            return p[0], p[1], p[2], p[3]
        raise ValueError(f"camera model {self.model} is not supported for mask undistortion")

    def distort(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Apply lens distortion to normalized image coordinates."""
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "PINHOLE"):
            return x, y
        r2 = x * x + y * y
        if self.model == "SIMPLE_RADIAL":
            radial = p[3] * r2
            return x * (1 + radial), y * (1 + radial)
        if self.model == "RADIAL":
            radial = p[3] * r2 + p[4] * r2 * r2
            return x * (1 + radial), y * (1 + radial)
        if self.model == "OPENCV":
            k1, k2, p1, p2 = p[4:8]
            radial = k1 * r2 + k2 * r2 * r2
            xd = x * (1 + radial) + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            yd = y * (1 + radial) + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            return xd, yd
        raise ValueError(f"camera model {self.model} is not supported for mask undistortion")


def read_cameras_txt(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        cam = Camera(
            camera_id=int(parts[0]),
            model=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=tuple(float(v) for v in parts[4:]),
        )
        cameras[cam.camera_id] = cam
    return cameras


def read_image_cameras_txt(path: Path) -> dict[str, int]:
    """Map image name -> camera id from images.txt (every other line is 2D points)."""
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    mapping: dict[str, int] = {}
    for line in lines[0::2]:
        parts = line.split()
        if len(parts) >= 10:
            mapping[" ".join(parts[9:])] = int(parts[8])
    return mapping


def undistort_mask(
    mask: NDArray[np.uint8], distorted: Camera, undistorted: Camera
) -> NDArray[np.uint8]:
    """Warp `mask` (taken with `distorted`) into the `undistorted` pinhole camera."""
    fx, fy, cx, cy = undistorted.focal_center()
    cols = np.arange(undistorted.width, dtype=np.float64) + 0.5
    rows = np.arange(undistorted.height, dtype=np.float64) + 0.5
    u, v = np.meshgrid(cols, rows)
    xd, yd = distorted.distort((u - cx) / fx, (v - cy) / fy)

    dfx, dfy, dcx, dcy = distorted.focal_center()
    # The mask may be stored at a different resolution than the camera model.
    scale_x = mask.shape[1] / distorted.width
    scale_y = mask.shape[0] / distorted.height
    src_u = np.floor((xd * dfx + dcx) * scale_x).astype(np.int64)
    src_v = np.floor((yd * dfy + dcy) * scale_y).astype(np.int64)

    inside = (src_u >= 0) & (src_u < mask.shape[1]) & (src_v >= 0) & (src_v < mask.shape[0])
    out = np.zeros((undistorted.height, undistorted.width), dtype=np.uint8)
    out[inside] = mask[src_v[inside], src_u[inside]]
    return out
