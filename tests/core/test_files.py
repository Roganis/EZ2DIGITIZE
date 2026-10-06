# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import struct
import zlib
from pathlib import Path

import pytest

from ez2digitize.core.files import write_uniform_png


def _decode_grey_png(data: bytes) -> tuple[int, int, bytes]:
    """Minimal reader for the PNGs write_uniform_png makes (checks CRCs)."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, {}
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        assert crc == zlib.crc32(kind + body)
        chunks[kind] = body
        pos += 12 + length
    width, height, depth, colour = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert (depth, colour) == (8, 0)
    raw = zlib.decompress(chunks[b"IDAT"])
    rows = [raw[i * (width + 1) : (i + 1) * (width + 1)] for i in range(height)]
    assert all(row[0] == 0 for row in rows)
    return width, height, b"".join(row[1:] for row in rows)


def test_write_uniform_png(tmp_path: Path) -> None:
    path = tmp_path / "white.png"
    write_uniform_png(path, 640, 3, 255)
    width, height, pixels = _decode_grey_png(path.read_bytes())
    assert (width, height) == (640, 3)
    assert pixels == b"\xff" * 640 * 3


@pytest.mark.parametrize(("w", "h", "v"), [(0, 1, 0), (1, 0, 0), (1, 1, 256), (1, 1, -1)])
def test_write_uniform_png_rejects(tmp_path: Path, w: int, h: int, v: int) -> None:
    with pytest.raises(ValueError, match="invalid PNG"):
        write_uniform_png(tmp_path / "x.png", w, h, v)
