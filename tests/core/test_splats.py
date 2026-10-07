# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import gzip
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from ez2digitize.core import splats as sp
from ez2digitize.core.splats import SplatFormatError, Splats
from ez2digitize.orientation import rotation_between

BASE = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]  # fmt: skip


def write_ply(path: Path, names: list[str], rows: list[list[float]]) -> Path:
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(rows)}\n"
    header += "".join(f"property float {n}\n" for n in names) + "end_header\n"
    body = b"".join(struct.pack(f"<{len(names)}f", *row) for row in rows)
    path.write_bytes(header.encode() + body)
    return path


def random_splats(n: int, k: int = 15, seed: int = 1) -> Splats:
    rng = np.random.default_rng(seed)
    return Splats(
        positions=rng.uniform(-3, 3, (n, 3)),
        scales=rng.uniform(-6, -1, (n, 3)),
        rotations=rng.normal(size=(n, 4)),
        opacities=rng.normal(size=n),
        sh_dc=rng.uniform(-1, 1, (n, 3)),
        sh_rest=rng.uniform(-0.6, 0.6, (n, k, 3)),
    )


def quaternion_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    matrix: np.ndarray = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip
    return matrix


def test_read_brush_ply(tmp_path: Path) -> None:
    # Brush's order: colours first, position last; f_rest stored channel by channel.
    rest = [f"f_rest_{i}" for i in range(45)]
    names = ["f_dc_0", "f_dc_1", "f_dc_2", *rest, "opacity", "rot_0", "rot_1", "rot_2", "rot_3",
             "scale_0", "scale_1", "scale_2", "x", "y", "z"]  # fmt: skip
    row = [0.1, 0.2, 0.3, *range(45), 2.0, 1, 0, 0, 0, -3, -4, -5, 7, 8, 9]
    splats = sp.read_ply(write_ply(tmp_path / "s.ply", names, [row, row]))
    assert splats.count == 2 and splats.sh_degree == 3
    assert splats.positions[1].tolist() == [7, 8, 9]
    assert splats.scales[0].tolist() == [-3, -4, -5]
    assert splats.rotations[0].tolist() == [1, 0, 0, 0]
    assert splats.opacities.tolist() == [2.0, 2.0]
    assert splats.sh_dc[0].tolist() == pytest.approx([0.1, 0.2, 0.3])
    # Coefficient j of channel c is f_rest_(c * 15 + j).
    assert splats.sh_rest[0, 4].tolist() == [4, 19, 34]


def test_read_errors(tmp_path: Path) -> None:
    degree0 = sp.read_ply(write_ply(tmp_path / "a.ply", BASE, [[0.0] * 14]))
    assert degree0.sh_degree == 0 and degree0.sh_rest.shape == (1, 0, 3)
    with pytest.raises(SplatFormatError, match="no opacity"):
        sp.read_ply(
            write_ply(tmp_path / "b.ply", [n for n in BASE if n != "opacity"], [[0.0] * 13])
        )
    rest = [f"f_rest_{i}" for i in range(6)]  # 2 per channel: no SH degree has that
    with pytest.raises(SplatFormatError, match="not an SH degree"):
        sp.read_ply(write_ply(tmp_path / "c.ply", BASE + rest, [[0.0] * 20]))
    (tmp_path / "d.ply").write_text("ply splats")
    with pytest.raises(SplatFormatError):
        sp.read_ply(tmp_path / "d.ply")


def test_turned_colours_look_the_same_from_turned_directions() -> None:
    splats = random_splats(5)
    r = np.array(rotation_between((0.2, -0.9, 0.4), (0.0, 1.0, 0.0)))
    turned = sp.rotate_sh(splats.sh_rest, r)
    directions = sp._sample_directions(20)

    def colour(coefficients: np.ndarray, d: np.ndarray) -> np.ndarray:
        basis = np.concatenate([sp._sh_basis(degree, d) for degree in (1, 2, 3)], axis=1)
        seen: np.ndarray = np.einsum("dk,nkc->ndc", basis, coefficients)
        return seen

    # Seen from d, the turned splat looks as the original did from Rᵀd.
    assert colour(turned, directions) == pytest.approx(colour(splats.sh_rest, directions @ r))


def test_placed_upright_on_the_ground() -> None:
    splats = random_splats(400)
    splats.opacities[:] = 3.0  # all solid
    r = np.array(rotation_between((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))  # Z up to Y up
    out = sp.placed(splats, tuple(map(tuple, r)), scale=0.5)  # type: ignore[arg-type]
    expected = splats.positions @ r.T
    low, high = np.percentile(expected, 2, axis=0), np.percentile(expected, 98, axis=0)
    expected = (expected - [(low[0] + high[0]) / 2, low[1], (low[2] + high[2]) / 2]) * 0.5
    assert out.positions == pytest.approx(expected)
    assert out.scales == pytest.approx(splats.scales + math.log(0.5))
    # Each splat's axes turn with it.
    for before, after in zip(splats.rotations[:5], out.rotations[:5], strict=True):
        assert quaternion_matrix(after) == pytest.approx(r @ quaternion_matrix(before))
    # Placed by anchor points (the sparse points), not by floaters far out.
    splats.positions[0] = (5000.0, 5000.0, -5000.0)
    anchor = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 1.0]] * 10)
    by_anchor = sp.placed(splats, tuple(map(tuple, r)), anchor=anchor)  # type: ignore[arg-type]
    shift = by_anchor.positions[1] - splats.positions[1] @ r.T
    assert shift == pytest.approx([-1.0, 0.0, 1.0])  # anchor upright: x 0..2, y 0..1, z -2..0
    # Without an up direction: the same frame, only scaled.
    kept = sp.placed(splats, None, scale=2.0)
    assert kept.positions == pytest.approx(splats.positions * 2)
    assert kept.sh_rest is splats.sh_rest


def test_spz_round_trip(tmp_path: Path) -> None:
    splats = random_splats(300)
    path = sp.write_spz(splats, tmp_path / "s.spz")
    raw = gzip.decompress(path.read_bytes())
    assert struct.unpack_from("<IIIBBBB", raw) == (0x5053474E, 2, 300, 3, 12, 0, 0)
    assert len(raw) == 16 + 300 * (9 + 1 + 3 + 3 + 3 + 45)
    back = sp.read_spz(path)
    assert np.abs(back.positions - splats.positions).max() <= 0.5 / 4096 + 1e-9
    assert np.abs(back.scales - splats.scales).max() <= 1 / 32 + 1e-9
    assert np.abs(back.sh_dc - splats.sh_dc).max() <= 0.5 / (0.15 * 255) + 1e-9
    alpha = 1 / (1 + np.exp(-splats.opacities))
    assert np.abs(1 / (1 + np.exp(-back.opacities)) - alpha).max() <= 0.5 / 255 + 1e-6
    unit = splats.rotations / np.linalg.norm(splats.rotations, axis=1, keepdims=True)
    assert np.abs((unit * back.rotations).sum(axis=1)).min() > 0.99  # same rotation (± sign)
    # Degree 1 at 5 bits, the rest at 4: half a step of 8/128 and 16/128, plus rounding.
    error = np.abs(back.sh_rest - splats.sh_rest)
    assert error[:, :3].max() <= 4.5 / 128 and error[:, 3:].max() <= 8.5 / 128


def test_spz_far_positions_keep_their_range(tmp_path: Path) -> None:
    splats = random_splats(10, k=0)
    splats.positions *= 2000  # beyond 12 fractional bits' ±2048
    back = sp.read_spz(sp.write_spz(splats, tmp_path / "far.spz"))
    bits = struct.unpack_from("<IIIBB", gzip.decompress((tmp_path / "far.spz").read_bytes()))[4]
    assert bits < 12 and back.sh_degree == 0
    assert np.abs(back.positions - splats.positions).max() <= 0.5 / (1 << bits) + 1e-6


def test_read_spz_rejects_other_files(tmp_path: Path) -> None:
    (tmp_path / "a.spz").write_bytes(b"not gzip")
    with pytest.raises(SplatFormatError):
        sp.read_spz(tmp_path / "a.spz")
    (tmp_path / "b.spz").write_bytes(gzip.compress(struct.pack("<IIIBBBB", 1, 2, 0, 0, 12, 0, 0)))
    with pytest.raises(SplatFormatError, match="not an SPZ file"):
        sp.read_spz(tmp_path / "b.spz")
    header = struct.pack("<IIIBBBB", sp.SPZ_MAGIC, 9, 0, 0, 12, 0, 0)
    (tmp_path / "c.spz").write_bytes(gzip.compress(header))
    with pytest.raises(SplatFormatError, match="versions 1 to 3"):
        sp.read_spz(tmp_path / "c.spz")


# --- other formats -----------------------------------------------------------------


def test_write_ply_reads_back(tmp_path: Path) -> None:
    splats = random_splats(20, k=8)
    back = sp.read_ply(sp.write_ply(splats, tmp_path / "s.ply"))
    assert back.sh_degree == 2
    for name in ("positions", "scales", "rotations", "opacities", "sh_dc", "sh_rest"):
        assert np.allclose(getattr(back, name), getattr(splats, name), atol=1e-6), name


def test_read_splat_file(tmp_path: Path) -> None:
    record = np.zeros(2, sp.SPLAT_RECORD)
    record["position"] = [(1, 2, 3), (4, 5, 6)]
    record["scale"] = [(0.5, 1, 2), (1, 1, 1)]
    record["color"] = [(255, 128, 0, 192), (0, 0, 0, 255)]
    record["rotation"] = [(255, 128, 128, 128), (128, 255, 128, 128)]
    (tmp_path / "a.splat").write_bytes(record.tobytes())
    splats = sp.read_splat(tmp_path / "a.splat")
    assert splats.count == 2 and splats.sh_degree == 0
    assert np.allclose(splats.scales[0], np.log([0.5, 1, 2]))
    assert np.allclose(splats.rotations[0], [1, 0, 0, 0], atol=1e-6)
    assert np.allclose(splats.rotations[1], [0, 1, 0, 0], atol=1e-6)
    assert np.allclose(0.5 + sp.SH_C0 * splats.sh_dc[0], [1, 128 / 255, 0])
    assert np.isclose(1 / (1 + np.exp(-splats.opacities[0])), 192 / 255)
    (tmp_path / "b.splat").write_bytes(b"\0" * 33)
    with pytest.raises(SplatFormatError, match="multiple of 32"):
        sp.read_splat(tmp_path / "b.splat")


def spz_bytes(splats: Splats, version: int) -> bytes:
    """An SPZ file of `version` 1 or 3, written by hand (the module writes 2)."""
    n = splats.count
    header = struct.pack("<IIIBBBB", sp.SPZ_MAGIC, version, n, 0, 12, 0, 0)
    if version == 1:
        positions = splats.positions.astype("<f2").tobytes()
    else:
        fixed = np.round(splats.positions * 4096).astype(np.int64) & 0xFFFFFF
        triples = np.stack([fixed & 255, (fixed >> 8) & 255, fixed >> 16], -1)
        positions = triples.astype(np.uint8).tobytes()
    q = splats.rotations / np.linalg.norm(splats.rotations, axis=1, keepdims=True)
    if version == 1:
        q = q * np.where(q[:, :1] < 0, -1, 1)
        rotations = np.clip(np.round(q[:, 1:] * 127.5 + 127.5), 0, 255).astype(np.uint8).tobytes()
    else:
        packed = []
        for w, x, y, z in q:
            xyzw = np.array([x, y, z, w])
            largest = int(np.argmax(np.abs(xyzw)))
            xyzw *= np.sign(xyzw[largest])
            word = largest
            for i in range(4):
                if i != largest:
                    magnitude = round(abs(xyzw[i]) / math.sqrt(0.5) * 511)
                    word = (word << 10) | (int(xyzw[i] < 0) << 9) | magnitude
            packed.append(struct.pack("<I", word))
        rotations = b"".join(packed)
    rest = bytes(n) + bytes(3 * n) + bytes(3 * n)  # opacity, colour, scale
    return gzip.compress(header + positions + rest + rotations)


@pytest.mark.parametrize("version", [1, 3])
def test_read_spz_versions_1_and_3(tmp_path: Path, version: int) -> None:
    splats = random_splats(40, k=0)
    (tmp_path / "v.spz").write_bytes(spz_bytes(splats, version))
    back = sp.read_spz(tmp_path / "v.spz")
    tolerance = 0.01 if version == 1 else 0.5 / 4096
    assert np.abs(back.positions - splats.positions).max() <= tolerance
    q = splats.rotations / np.linalg.norm(splats.rotations, axis=1, keepdims=True)
    same = np.abs(np.sum(back.rotations * q, axis=1))  # q and -q are the same turn
    assert same.min() > (0.999 if version == 3 else 0.99)


def test_turned_keeps_colours_pointing_the_same_way() -> None:
    splats = random_splats(5, k=3)
    flipped = sp.turned(splats, sp.PLY_TO_SPZ)
    assert np.allclose(flipped.positions, splats.positions * [1, -1, -1])
    back = sp.turned(flipped, sp.PLY_TO_SPZ)
    assert np.allclose(back.positions, splats.positions)
    assert np.allclose(back.sh_rest, splats.sh_rest, atol=1e-9)
    q = splats.rotations / np.linalg.norm(splats.rotations, axis=1, keepdims=True)
    r = back.rotations / np.linalg.norm(back.rotations, axis=1, keepdims=True)
    assert np.allclose(np.abs(np.sum(q * r, axis=1)), 1)
