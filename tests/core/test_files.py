# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import struct
import sys
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


def test_replace_retries_while_windows_has_the_file_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ez2digitize.core import files

    target = tmp_path / "capture.json"
    files.write_json_atomic(target, {"v": 1})
    real = Path.replace
    busy = [2]  # the first two renames find the file open

    def flaky(self: Path, other: Path) -> Path:
        if busy[0]:
            busy[0] -= 1
            raise PermissionError(13, "Access is denied")
        return real(self, other)

    monkeypatch.setattr(Path, "replace", flaky)
    monkeypatch.setattr(files, "REPLACE_WAIT_S", 0.0)
    monkeypatch.setattr(sys, "platform", "win32")
    files.write_json_atomic(target, {"v": 2})
    assert files.read_json_object(target) == {"v": 2}

    # Elsewhere a permission error is real: no retries.
    monkeypatch.setattr(sys, "platform", "linux")
    busy[0] = 1
    with pytest.raises(PermissionError):
        files.write_json_atomic(target, {"v": 3})
    assert files.read_json_object(target) == {"v": 2}
    assert not list(tmp_path.glob(".capture.json.*"))  # no temporary file left
