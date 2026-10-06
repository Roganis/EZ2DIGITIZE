# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Files the app downloads on first use, checked against a pinned sha256.

The masking model and COLMAP's vocabulary tree are too large to bundle and
only some projects need them. They go into the user's cache
(`models_dir()`), never half-written under their final name.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

from ez2digitize.core.runner import CancelToken

CHUNK = 1 << 20


class DownloadError(Exception):
    """The file could not be fetched, or arrived damaged."""


class DownloadCancelled(DownloadError):
    pass


class DownloadDamaged(DownloadError):
    """It arrived, but not as pinned (sha256 mismatch)."""


def models_dir() -> Path:
    """Where downloaded files live: the user's cache, or `EZ2D_MODELS_DIR`."""
    if override := os.environ.get("EZ2D_MODELS_DIR"):
        return Path(override)
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "ez2digitize" / "models"


def fetch(
    url: str,
    target: Path,
    sha256: str,
    *,
    size: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    cancel: CancelToken | None = None,
) -> Path:
    """Download `url` to `target` and check its sha256; returns `target`.

    `on_progress(received, total)` is called as it arrives (total from the
    server, else `size`, else 0). Raises DownloadError (or its subclasses
    DownloadCancelled, DownloadDamaged) and leaves nothing at `target` then.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    received = 0
    try:
        with (
            urllib.request.urlopen(url, timeout=60) as response,  # noqa: S310 - pinned URLs
            partial.open("wb") as out,
        ):
            total = int(response.headers.get("Content-Length") or size or 0)
            while chunk := response.read(CHUNK):
                if cancel is not None and cancel.cancelled:
                    raise DownloadCancelled("download cancelled")
                out.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if on_progress is not None:
                    on_progress(received, total)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"could not download {url}: {exc}") from exc
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    if digest.hexdigest() != sha256:
        partial.unlink(missing_ok=True)
        raise DownloadDamaged(f"sha256 mismatch, {received} bytes")
    partial.replace(target)
    return target
