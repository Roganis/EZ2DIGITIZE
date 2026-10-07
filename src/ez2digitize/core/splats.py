# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gaussian splats: read Brush's PLY, stand them upright, write SPZ.

Brush writes the usual 3D Gaussian splatting PLY: per splat a centre, a
log scale per axis, a rotation quaternion (w, x, y, z), an opacity (logit)
and spherical-harmonics colour coefficients (`f_dc_*`, and `f_rest_*` up to
degree 3, stored channel by channel). About 250 bytes a splat.

SPZ (Niantic, MIT-licensed format) packs the same in about 25: positions as
24-bit fixed point, the rest as bytes, attribute by attribute, gzipped.
This writes version 2 (readers of later versions read it too): rotations as
the quaternion's x, y, z with w made positive, SH degree 1 at 5 bits and
higher degrees at 4. SPZ coordinates are Y up (right, up, back), so the
export stands the splats upright first (`placed`), turning each splat's
rotation and its colour coefficients with them, centres them and puts them
on the ground, as the mesh exports are.
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ez2digitize.core.meshio import MeshFormatError, _read_header
from ez2digitize.orientation import Matrix

Array = NDArray[np.float64]

# SH coefficients per colour channel beyond the constant one, by degree.
REST_COUNTS = {0: 0, 3: 1, 8: 2, 15: 3}
SPZ_MAGIC = 0x5053474E  # "NGSP"
SPZ_VERSION = 2
SPZ_COLOR_SCALE = 0.15  # SH DC coefficient to byte, as SPZ defines it
# Opaque enough to count when placing on the ground (sigmoid of the logit).
SOLID = 0.5


class SplatFormatError(MeshFormatError):
    pass


@dataclass
class Splats:
    positions: Array  # (n, 3)
    scales: Array  # (n, 3), natural log
    rotations: Array  # (n, 4), w x y z
    opacities: Array  # (n,), logit
    sh_dc: Array  # (n, 3)
    sh_rest: Array  # (n, k, 3): coefficient, then channel; k = 0, 3, 8 or 15

    @property
    def count(self) -> int:
        return len(self.positions)

    @property
    def sh_degree(self) -> int:
        return REST_COUNTS[self.sh_rest.shape[1]]


# --- PLY --------------------------------------------------------------------------


def read_ply(path: Path) -> Splats:
    """A 3D Gaussian splatting PLY (binary little-endian, properties in any order)."""
    with path.open("rb") as fh:
        try:
            elements, _comments = _read_header(fh, path)
        except SplatFormatError:
            raise
        except MeshFormatError as exc:
            raise SplatFormatError(str(exc)) from exc
        data = fh.read()
    if [e.name for e in elements] != ["vertex"]:
        raise SplatFormatError(f"{path}: expected only a vertex element")
    (vertex,) = elements
    if any(p.count_type for p in vertex.properties):
        raise SplatFormatError(f"{path}: list properties in splats")
    dtype = np.dtype([(p.name, "<" + p.type) for p in vertex.properties])
    if len(data) < dtype.itemsize * vertex.count:
        raise SplatFormatError(f"{path}: file ends inside the splats")
    table = np.frombuffer(data, dtype=dtype, count=vertex.count)
    names = set(dtype.names or ())

    def columns(*wanted: str) -> Array:
        missing = [n for n in wanted if n not in names]
        if missing:
            raise SplatFormatError(f"{path}: no {', '.join(missing)}")
        return np.stack([table[n].astype(np.float64) for n in wanted], axis=-1)

    rest = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    if len(rest) % 3 or len(rest) // 3 not in REST_COUNTS:
        raise SplatFormatError(f"{path}: {len(rest)} f_rest values: not an SH degree")
    k = len(rest) // 3
    # Stored channel by channel: f_rest_[c * k + j] is coefficient j of channel c.
    if k:
        sh_rest = columns(*rest).reshape(-1, 3, k).transpose(0, 2, 1)
    else:
        sh_rest = np.zeros((vertex.count, 0, 3))
    return Splats(
        positions=columns("x", "y", "z"),
        scales=columns("scale_0", "scale_1", "scale_2"),
        rotations=columns("rot_0", "rot_1", "rot_2", "rot_3"),
        opacities=columns("opacity")[:, 0],
        sh_dc=columns("f_dc_0", "f_dc_1", "f_dc_2"),
        sh_rest=np.ascontiguousarray(sh_rest),
    )


# --- standing them upright ----------------------------------------------------


def placed(
    splats: Splats,
    rotation: Matrix | None,
    scale: float = 1.0,
    anchor: Array | None = None,
) -> Splats:
    """Turned by `rotation` (model to upright), centred on the vertical axis
    and put on the ground (as orientation.place_rotated does for meshes), then
    scaled by `scale` (export units per model unit).

    Centre and ground come from `anchor` (n, 3) in model coordinates: the
    camera placement's sparse points, which are the object and what it stands
    on. Trained splats are a poor guide: floaters can sit thousands of units
    out. Without anchor points, the solid splats are used (all, if none is).
    Without a rotation (the photos don't say which way is up) the splats keep
    their frame and are only scaled.
    """
    positions, rotations, sh_rest = splats.positions, splats.rotations, splats.sh_rest
    if rotation is not None:
        r = np.array(rotation, dtype=np.float64)
        positions, rotations, sh_rest = _turned(splats, r)
        if anchor is not None and len(anchor):
            guide = anchor @ r.T
        else:
            guide = positions[_sigmoid(splats.opacities) >= SOLID]
            if not len(guide):
                guide = positions
        if len(guide):
            # Percentiles, not extremes: points and splats both have strays.
            low, high = np.percentile(guide, 2, axis=0), np.percentile(guide, 98, axis=0)
            offset = np.array([-(low[0] + high[0]) / 2, -low[1], -(low[2] + high[2]) / 2])
            positions = positions + offset
    return Splats(
        positions=positions * scale,
        scales=splats.scales + np.log(scale),
        rotations=rotations,
        opacities=splats.opacities,
        sh_dc=splats.sh_dc,
        sh_rest=sh_rest,
    )


def turned(splats: Splats, rotation: Array) -> Splats:
    """The splats turned about the origin by `rotation` (3x3), each splat's
    rotation and view-dependent colour with them."""
    positions, rotations, sh_rest = _turned(splats, np.asarray(rotation, np.float64))
    return Splats(positions, splats.scales, rotations, splats.opacities, splats.sh_dc, sh_rest)


def _turned(splats: Splats, r: Array) -> tuple[Array, Array, Array]:
    rotations = _quaternion_product(_matrix_quaternion(r), splats.rotations)
    return splats.positions @ r.T, rotations, rotate_sh(splats.sh_rest, r)


def rotate_sh(sh_rest: Array, rotation: Array) -> Array:
    """The view-dependent colour coefficients (n, k, 3) of splats turned by `rotation`.

    Each degree's coefficients mix among themselves. The matrix for a degree
    is found by least squares from sample directions: the turned splat's
    colour towards d is the original's towards Rᵀd.
    """
    k = sh_rest.shape[1]
    if not k:
        return sh_rest
    directions = _sample_directions(64)
    turned_back = directions @ rotation  # rows: Rᵀ·d
    out = np.empty_like(sh_rest)
    start = 0
    for degree in range(1, REST_COUNTS[k] + 1):
        size = 2 * degree + 1
        a = _sh_basis(degree, directions)
        b = _sh_basis(degree, turned_back)
        mix = np.linalg.lstsq(a, b, rcond=None)[0]  # a · mix = b
        out[:, start : start + size] = np.einsum(
            "ij,njc->nic", mix, sh_rest[:, start : start + size]
        )
        start += size
    return out


def _sh_basis(degree: int, d: Array) -> Array:
    """The real SH basis of one degree, as 3D Gaussian splatting defines it (sh_utils)."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    if degree == 1:
        c1 = 0.4886025119029199
        return np.stack([-c1 * y, c1 * z, -c1 * x], axis=1)
    xx, yy, zz = x * x, y * y, z * z
    if degree == 2:
        c2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
              -1.0925484305920792, 0.5462742152960396)  # fmt: skip
        return np.stack(
            [
                c2[0] * x * y,
                c2[1] * y * z,
                c2[2] * (2 * zz - xx - yy),
                c2[3] * x * z,
                c2[4] * (xx - yy),
            ],
            axis=1,
        )
    c = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435)  # fmt: skip
    return np.stack(
        [
            c[0] * y * (3 * xx - yy),
            c[1] * x * y * z,
            c[2] * y * (4 * zz - xx - yy),
            c[3] * z * (2 * zz - 3 * xx - 3 * yy),
            c[4] * x * (4 * zz - xx - yy),
            c[5] * z * (xx - yy),
            c[6] * x * (xx - 3 * yy),
        ],
        axis=1,
    )


def _sample_directions(n: int) -> Array:
    """`n` unit vectors spread over the sphere (a Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    phi = np.pi * (1 + 5**0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def _matrix_quaternion(m: Array) -> Array:
    """The unit quaternion (w, x, y, z) of a rotation matrix."""
    w = np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2
    x = np.copysign(x, m[2, 1] - m[1, 2])
    y = np.copysign(y, m[0, 2] - m[2, 0])
    z = np.copysign(z, m[1, 0] - m[0, 1])
    q: Array = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _quaternion_product(a: Array, b: Array) -> Array:
    """a ⊗ b for one quaternion `a` and rows `b` (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )


def _sigmoid(x: Array) -> Array:
    return 1 / (1 + np.exp(-x))


# --- SPZ -----------------------------------------------------------------------


def write_spz(splats: Splats, path: Path) -> Path:
    """Write `splats` as SPZ version 2."""
    n = splats.count
    # 24-bit fixed point: 12 fractional bits unless the scene is too large for them.
    extent = float(np.abs(splats.positions).max()) if n else 0.0
    bits = 12
    while bits > 0 and extent * (1 << bits) >= (1 << 23) - 1:
        bits -= 1
    fixed = np.round(splats.positions * (1 << bits)).astype(np.int64)
    fixed = np.clip(fixed, -(1 << 23), (1 << 23) - 1) & 0xFFFFFF
    positions = np.stack([fixed & 0xFF, (fixed >> 8) & 0xFF, (fixed >> 16) & 0xFF], axis=-1)

    q = splats.rotations / np.linalg.norm(splats.rotations, axis=1, keepdims=True).clip(1e-12)
    q = q * np.where(q[:, :1] < 0, -1.0, 1.0)  # w ≥ 0: x, y, z then say it all
    sh = splats.sh_rest.copy()
    if sh.shape[1]:
        sh = np.round(sh * 128) + 128
        buckets = np.where(np.arange(sh.shape[1]) < 3, 8, 16)[None, :, None]  # 5 and 4 bits
        sh = np.floor((sh + buckets // 2) / buckets) * buckets
    parts = [
        positions,
        _bytes(_sigmoid(splats.opacities) * 255),
        _bytes(splats.sh_dc * (SPZ_COLOR_SCALE * 255) + 127.5),
        _bytes((splats.scales + 10) * 16),
        _bytes(q[:, 1:] * 127.5 + 127.5),
        _bytes(sh),
    ]
    header = struct.pack("<IIIBBBB", SPZ_MAGIC, SPZ_VERSION, n, splats.sh_degree, bits, 0, 0)
    body = b"".join(np.ascontiguousarray(p, dtype=np.uint8).tobytes() for p in parts)
    path.write_bytes(gzip.compress(header + body, compresslevel=6))
    return path


def read_spz(path: Path) -> Splats:
    """Read an SPZ file (versions 1 to 3) back into splats, in SPZ's frame.

    Version 1 stores positions as half floats, version 2 as 24-bit fixed
    point with the rotation's x, y, z (w positive), version 3 the rotation's
    three smallest components and which one was left out.
    """
    try:
        data = gzip.decompress(path.read_bytes())
        magic, version, n, degree, bits, _flags, _ = struct.unpack_from("<IIIBBBB", data)
    except (OSError, EOFError, gzip.BadGzipFile, struct.error) as exc:
        raise SplatFormatError(f"{path}: not an SPZ file ({exc})") from exc
    if magic != SPZ_MAGIC or degree > 3:
        raise SplatFormatError(f"{path}: not an SPZ file")
    if version not in (1, 2, 3):
        raise SplatFormatError(f"{path}: SPZ version {version}; versions 1 to 3 are read")
    k = {0: 0, 1: 3, 2: 8, 3: 15}[degree]
    sizes = [n * (6 if version == 1 else 9), n, n * 3, n * 3, n * (4 if version >= 3 else 3)]
    sizes.append(n * k * 3)
    if len(data) < 16 + sum(sizes):
        raise SplatFormatError(f"{path}: file ends inside the splats")
    chunks, offset = [], 16
    for size in sizes:
        chunks.append(np.frombuffer(data, np.uint8, size, offset))
        offset += size
    if version == 1:
        positions = chunks[0].view("<f2").reshape(n, 3).astype(np.float64)
    else:
        raw = chunks[0].reshape(n, 3, 3).astype(np.int64)
        fixed = raw[..., 0] + (raw[..., 1] << 8) + (raw[..., 2] << 16)
        fixed = np.where(fixed >= 1 << 23, fixed - (1 << 24), fixed)
        positions = fixed / (1 << bits)
    if version >= 3:
        rotations = _smallest_three(chunks[4].reshape(n, 4))
    else:
        xyz = chunks[4].reshape(n, 3).astype(np.float64) / 127.5 - 1
        w = np.sqrt(np.clip(1 - (xyz**2).sum(axis=1), 0, None))
        rotations = np.column_stack([w, xyz])
    alpha = np.clip(chunks[1].astype(np.float64) / 255, 1e-6, 1 - 1e-6)
    return Splats(
        positions=positions,
        scales=chunks[3].reshape(n, 3).astype(np.float64) / 16 - 10,
        rotations=rotations,
        opacities=np.log(alpha / (1 - alpha)),
        sh_dc=(chunks[2].reshape(n, 3).astype(np.float64) / 255 - 0.5) / SPZ_COLOR_SCALE,
        sh_rest=(chunks[5].reshape(n, k, 3).astype(np.float64) - 128) / 128,
    )


def _smallest_three(packed: NDArray[np.uint8]) -> Array:
    """SPZ 3's rotations (4 bytes each) as quaternions (w, x, y, z).

    The 32 bits, high to low: the index (2 bits) of the largest component of
    (x, y, z, w), then for the others, last first, a sign bit and 9 bits of
    magnitude in units of 1/sqrt(2) / 511. The largest is positive.
    """
    word = packed.astype(np.uint32)
    bits = word[:, 0] | (word[:, 1] << 8) | (word[:, 2] << 16) | (word[:, 3] << 24)
    largest = (bits >> 30).astype(np.int64)
    xyzw = np.zeros((len(packed), 4))
    rows = np.arange(len(packed))
    for slot in (3, 2, 1, 0):
        # Each splat skips its largest component; the others are read last first.
        reading = slot != largest
        magnitude = (bits & 511).astype(np.float64) * (np.sqrt(0.5) / 511)
        negative = ((bits >> 9) & 1).astype(bool)
        xyzw[reading, slot] = np.where(negative, -magnitude, magnitude)[reading]
        bits = np.where(reading, bits >> 10, bits)
    xyzw[rows, largest] = np.sqrt(np.clip(1 - (xyzw**2).sum(axis=1), 0, None))
    return np.column_stack([xyzw[:, 3], xyzw[:, :3]])


# --- other formats ---------------------------------------------------------------

# 3D Gaussian splatting's PLY (and .splat files made from it) keep COLMAP's
# frame (x right, y down, z forward); SPZ is x right, y up, z back. A half
# turn about x goes from one to the other, both ways.
PLY_TO_SPZ: Array = np.diag([1.0, -1.0, -1.0])
SH_C0 = 0.28209479177387814  # the constant spherical harmonic
SPLAT_RECORD = np.dtype(
    [("position", "<f4", 3), ("scale", "<f4", 3), ("color", "u1", 4), ("rotation", "u1", 4)]
)


def read_splat(path: Path) -> Splats:
    """A `.splat` file (antimatter15's web viewer): 32 bytes a splat, position
    and scale as floats, colour and opacity as bytes, the rotation (w, x, y,
    z) as bytes; no view-dependent colour."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SplatFormatError(f"{path}: {exc}") from exc
    if len(data) % SPLAT_RECORD.itemsize:
        raise SplatFormatError(f"{path}: not a .splat file (its size isn't a multiple of 32)")
    table = np.frombuffer(data, SPLAT_RECORD)
    rgba = table["color"].astype(np.float64) / 255
    alpha = np.clip(rgba[:, 3], 1e-6, 1 - 1e-6)
    rotations = (table["rotation"].astype(np.float64) - 128) / 128
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotations = np.where(norms > 0, rotations / np.where(norms > 0, norms, 1), [1.0, 0, 0, 0])
    return Splats(
        positions=table["position"].astype(np.float64),
        scales=np.log(np.clip(table["scale"].astype(np.float64), 1e-12, None)),
        rotations=rotations,
        opacities=np.log(alpha / (1 - alpha)),
        sh_dc=(rgba[:, :3] - 0.5) / SH_C0,
        sh_rest=np.zeros((len(table), 0, 3)),
    )


def write_ply(splats: Splats, path: Path) -> Path:
    """The usual 3D Gaussian splatting PLY (what Brush and most tools read)."""
    k = splats.sh_rest.shape[1]
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * k)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    columns = [
        splats.positions,
        np.zeros((splats.count, 3)),
        splats.sh_dc,
        # Channel by channel: f_rest_[c * k + j] is coefficient j of channel c.
        splats.sh_rest.transpose(0, 2, 1).reshape(splats.count, 3 * k),
        splats.opacities[:, None],
        splats.scales,
        splats.rotations,
    ]
    table = np.ascontiguousarray(np.hstack(columns), "<f4")
    header = ["ply", "format binary_little_endian 1.0", "comment written by EZ2DIGITIZE"]
    header += [f"element vertex {splats.count}", *(f"property float {n}" for n in names)]
    header.append("end_header")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(("\n".join(header) + "\n").encode("ascii"))
        out.write(table.tobytes())
    return path


def _bytes(values: Array) -> NDArray[np.uint8]:
    return np.clip(np.round(values), 0, 255).astype(np.uint8)
