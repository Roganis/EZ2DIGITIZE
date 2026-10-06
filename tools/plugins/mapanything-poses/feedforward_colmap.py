# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""A feed-forward network's predictions as a COLMAP model, for EZ2DIGITIZE's camera placement slot.

Shared by the VGGT and MapAnything plugins: each folder holds an identical
copy, since a plugin is installed on its own (tests/test_mapanything_plugin.py
checks they match).

The networks predict, for every photo at once, its camera (extrinsics in
OpenCV's camera-from-world convention, intrinsics) and a depth map with a
confidence, all at the size they ran at: each photo scaled, then padded
(VGGT: to a square) or cropped (MapAnything: to a fixed aspect ratio). This
module turns that into the binary COLMAP model the slot asks for
(docs/PLUGINS.md):

- one PINHOLE camera per photo, its intrinsics mapped back to the photo's
  own pixels (undo the scale, then the padding or crop);
- the photo's pose, as COLMAP's world-to-camera quaternion and translation;
- sparse points: confident depth pixels lifted to 3D, each kept only where
  it is also seen by other photos, whose depth there agrees. OpenMVS picks
  each photo's neighbours by the points they share, so points need tracks
  across photos, not only the photo they came from.

Numpy and the standard library only, so it is tested without a GPU or the
networks (tests/test_vggt_plugin.py).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PINHOLE = 1  # COLMAP's camera model id
NO_POINT = 2**64 - 1  # a 2D point without a 3D point

Float = NDArray[np.float64]


@dataclass(frozen=True)
class Placement:
    """Where a photo sits in the network's input: its pixels scaled by `scale`
    and shifted by (`left`, `top`) original pixels, positive for padding and
    negative for a crop. Pixel coordinates have 0 at a pixel's edge."""

    scale: float
    left: float
    top: float
    width: int
    height: int

    @classmethod
    def of(cls, size: tuple[int, int], resolution: int) -> Placement:
        """Padded to a square, centred, then scaled to `resolution` (VGGT)."""
        width, height = size
        side = max(width, height)
        return cls(resolution / side, (side - width) // 2, (side - height) // 2, width, height)

    @classmethod
    def cover(cls, size: tuple[int, int], target: tuple[int, int]) -> Placement:
        """Scaled to cover `target` (width, height), then cropped to it, centred."""
        width, height = size
        scale = max(target[0] / width, target[1] / height)
        return cls(
            scale,
            -(width - target[0] / scale) / 2,
            -(height - target[1] / scale) / 2,
            width,
            height,
        )

    def source_box(self, target: tuple[int, int]) -> tuple[float, float, float, float]:
        """The part of the photo that becomes the `target`-sized input, in its
        own pixels (left, top, right, bottom): what PIL's resize(box=) takes."""
        return (
            -self.left,
            -self.top,
            target[0] / self.scale - self.left,
            target[1] / self.scale - self.top,
        )

    def to_photo(self, u: Float, v: Float) -> tuple[Float, Float]:
        """Input pixel coordinates to the photo's own."""
        return u / self.scale - self.left, v / self.scale - self.top

    def inside(self, u: Float, v: Float) -> NDArray[np.bool_]:
        """Which input pixel coordinates fall on the photo, not the padding."""
        x, y = self.to_photo(u, v)
        return (x >= 0) & (y >= 0) & (x < self.width) & (y < self.height)

    def input_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> Float:
        """K at the network's input, from the photo's own intrinsics."""
        s = self.scale
        return np.array(
            [[fx * s, 0, (cx + self.left) * s], [0, fy * s, (cy + self.top) * s], [0, 0, 1]]
        )


@dataclass
class Prediction:
    """A network's output for S photos at its input size H x W."""

    names: list[str]  # COLMAP image names, as in the image list
    placements: list[Placement]  # each photo in the input
    extrinsic: Float  # (S, 3, 4) camera from world
    intrinsic: Float  # (S, 3, 3) at the input size
    depth: Float  # (S, H, W), 0 where unknown
    confidence: Float  # (S, H, W)
    colors: NDArray[np.uint8] | None = None  # (S, H, W, 3)


def photo_intrinsics(k: Float, placement: Placement) -> tuple[float, float, float, float]:
    """fx, fy, cx, cy in the photo's own pixels, from K at the input size."""
    s = placement.scale
    return (
        float(k[0, 0] / s),
        float(k[1, 1] / s),
        float(k[0, 2] / s - placement.left),
        float(k[1, 2] / s - placement.top),
    )


def quaternion(r: Float) -> tuple[float, float, float, float]:
    """(w, x, y, z) of a rotation matrix, w >= 0 (COLMAP's qvec)."""
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    norm = float(np.sqrt(sum(c * c for c in q)))
    sign = 1.0 if q[0] >= 0 else -1.0
    w, x, y, z = (float(sign * c / norm) for c in q)
    return w, x, y, z


@dataclass
class Points:
    xyz: Float  # (N, 3)
    rgb: NDArray[np.uint8]  # (N, 3)
    # Per point, its observations: (photo index, x, y in the photo's pixels).
    tracks: list[list[tuple[int, float, float]]]


def sparse_points(
    prediction: Prediction,
    *,
    per_photo: int = 1500,
    min_views: int = 2,
    agreement: float = 0.03,
    seed: int = 0,
) -> Points:
    """Confident depth pixels lifted to 3D, with the photos that see them.

    Per photo, up to `per_photo` pixels from the more confident half (on
    the photo, not the padding). Each 3D point is projected into every
    photo; it is seen there if it lands on the photo, in front, and that
    photo's own depth there is within `agreement` (relative). Points seen
    by fewer than `min_views` photos are dropped.
    """
    rng = np.random.default_rng(seed)
    count = len(prediction.names)
    rows_in, cols_in = prediction.depth.shape[1:]
    placements = prediction.placements
    grid_v, grid_u = np.mgrid[0:rows_in, 0:cols_in].astype(np.float64) + 0.5
    xyz_all, rgb_all, origin = [], [], []
    for i in range(count):
        on_photo = placements[i].inside(grid_u, grid_v)
        conf = prediction.confidence[i]
        depth = prediction.depth[i]
        usable = on_photo & (depth > 0) & np.isfinite(depth)
        if not usable.any():
            continue
        usable &= conf >= np.median(conf[usable])
        vs, us = np.nonzero(usable)
        if len(vs) > per_photo:
            pick = rng.choice(len(vs), per_photo, replace=False)
            vs, us = vs[pick], us[pick]
        z = depth[vs, us]
        k_inv = np.linalg.inv(prediction.intrinsic[i])
        rays = k_inv @ np.stack([us + 0.5, vs + 0.5, np.ones_like(z)])
        in_camera = rays * z
        r, t = prediction.extrinsic[i][:, :3], prediction.extrinsic[i][:, 3]
        xyz_all.append((r.T @ (in_camera - t[:, None])).T)
        if prediction.colors is not None:
            rgb_all.append(prediction.colors[i][vs, us])
        else:
            rgb_all.append(np.full((len(vs), 3), 160, np.uint8))
        origin.append(np.full(len(vs), i))
    if not xyz_all:
        return Points(np.zeros((0, 3)), np.zeros((0, 3), np.uint8), [])
    xyz = np.concatenate(xyz_all)
    rgb = np.concatenate(rgb_all).astype(np.uint8)
    tracks: list[list[tuple[int, float, float]]] = [[] for _ in range(len(xyz))]
    for j in range(count):
        r, t = prediction.extrinsic[j][:, :3], prediction.extrinsic[j][:, 3]
        cam = (r @ xyz.T).T + t
        z = cam[:, 2]
        front = z > 1e-9
        proj = (prediction.intrinsic[j] @ cam.T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(front, proj[:, 0] / z, -1.0)
            v = np.where(front, proj[:, 1] / z, -1.0)
        seen = front & placements[j].inside(u, v)
        seen &= (u >= 0) & (v >= 0) & (u < cols_in) & (v < rows_in)  # cropped away
        cols = np.clip(np.floor(u).astype(int), 0, cols_in - 1)
        rows = np.clip(np.floor(v).astype(int), 0, rows_in - 1)
        there = prediction.depth[j][rows, cols]
        seen &= np.abs(there - z) <= agreement * z
        x, y = placements[j].to_photo(u, v)
        for p in np.nonzero(seen)[0]:
            tracks[p].append((j, float(x[p]), float(y[p])))
    keep = [p for p, track in enumerate(tracks) if len(track) >= min_views]
    return Points(xyz[keep], rgb[keep], [tracks[p] for p in keep])


def write_model(folder: Path, prediction: Prediction, points: Points) -> None:
    """cameras.bin, images.bin and points3D.bin (COLMAP's binary format)."""
    folder.mkdir(parents=True, exist_ok=True)
    count = len(prediction.names)
    placements = prediction.placements
    with (folder / "cameras.bin").open("wb") as f:
        f.write(struct.pack("<Q", count))
        for i in range(count):
            params = photo_intrinsics(prediction.intrinsic[i], placements[i])
            size = (placements[i].width, placements[i].height)
            f.write(struct.pack("<IiQQ4d", i + 1, PINHOLE, *size, *params))
    # The 2D points of each photo: its observations, in track order.
    per_photo: list[list[tuple[float, float, int]]] = [[] for _ in range(count)]
    point_tracks: list[list[tuple[int, int]]] = []
    for p, track in enumerate(points.tracks):
        refs = []
        for photo, x, y in track:
            refs.append((photo + 1, len(per_photo[photo])))
            per_photo[photo].append((x, y, p + 1))
        point_tracks.append(refs)
    with (folder / "images.bin").open("wb") as f:
        f.write(struct.pack("<Q", count))
        for i, name in enumerate(prediction.names):
            r, t = prediction.extrinsic[i][:, :3], prediction.extrinsic[i][:, 3]
            f.write(struct.pack("<I4d3dI", i + 1, *quaternion(r), *map(float, t), i + 1))
            f.write(name.encode() + b"\0")
            f.write(struct.pack("<Q", len(per_photo[i])))
            for x, y, point in per_photo[i]:
                f.write(struct.pack("<2dQ", x, y, point))
    with (folder / "points3D.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(point_tracks)))
        for p, refs in enumerate(point_tracks):
            x, y, z = map(float, points.xyz[p])
            red, green, blue = (int(c) for c in points.rgb[p])
            f.write(struct.pack("<Q3d3Bd", p + 1, x, y, z, red, green, blue, 0.0))
            f.write(struct.pack("<Q", len(refs)))
            for image_id, index in refs:
                f.write(struct.pack("<II", image_id, index))
