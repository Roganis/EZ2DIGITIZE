# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small file helpers shared by the project, capture and manifest code."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024


class FormatError(Exception):
    """A project file is missing, unreadable or not in the expected format."""


def utc_now() -> str:
    """Current time as an ISO 8601 UTC string, as stored in every JSON file."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON file whose top level must be an object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FormatError(f"{path} does not exist") from None
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise FormatError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise FormatError(f"{path}: expected a JSON object at the top level")
    return data


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write JSON so that readers see either the old file or the new one, never half.

    Writes to a temporary file in the same directory, fsyncs it and renames it
    over the target. A crash or a full disk leaves the old file in place.
    """
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's contents, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    """Hex SHA-256 of a JSON-serialisable value, independent of dict key order."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_uniform_png(path: Path, width: int, height: int, value: int) -> None:
    """Write an 8-bit greyscale PNG where every pixel is `value` (e.g. a blank mask)."""
    if not (0 <= value <= 255 and width > 0 and height > 0):
        raise ValueError(f"invalid PNG: {width}x{height}, value {value}")

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\0" + bytes([value]) * width  # filter type 0 (none), then the pixels
    compressor = zlib.compressobj(9)
    pixels = b"".join(compressor.compress(row) for _ in range(height)) + compressor.flush()
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # 8-bit greyscale
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")
    )


def link(target: Path, link_path: Path) -> None:
    """Make `link_path` point at `target` (relative to the link's folder, or absolute).

    A relative symlink where possible, so a moved project still works. On
    Windows, where symlinks need Developer Mode or admin rights, a folder
    becomes a directory junction (absolute) and a file a hard link, or a
    copy across drives.
    """
    try:
        link_path.symlink_to(target)
        return
    except (OSError, NotImplementedError):
        if sys.platform != "win32":
            raise
    resolved = (link_path.parent / target).resolve()
    if resolved.is_dir():
        import _winapi

        _winapi.CreateJunction(str(resolved), str(link_path))
        return
    try:
        os.link(resolved, link_path)
    except OSError:
        shutil.copyfile(resolved, link_path)
