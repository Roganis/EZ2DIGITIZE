# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small file helpers shared by the project, capture and manifest code."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
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
