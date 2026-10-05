# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Capture bundles: how every photo set enters a project.

A bundle is a folder under `captures/` holding the original files, copied
byte for byte, plus a `capture.json`:

    {
      "schema_version": 1,
      "id": "20261005-203200",
      "source": "folder",            # folder, video, upload, android, ...
      "created": "2026-10-05T20:32:00+00:00",
      "device": {"make": "Google", "model": "Pixel 8"} or null,
      "source_info": {...},          # free-form, e.g. the imported folder
      "files": [
        {"name": "IMG_0001.jpg", "original_name": "IMG_0001.jpg",
         "kind": "image", "size": 4123456, "sha256": "...", "metadata": {}}
      ]
    }

Folder import, video, phone upload and the Android app all produce this
format; the pipeline only reads bundles. A bundle is assembled in a hidden
temporary folder and renamed into place when complete, so an interrupted
import never shows up as a bundle.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ez2digitize.core.files import (
    FormatError,
    read_json_object,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from ez2digitize.core.project import Project

CAPTURE_FILE = "capture.json"
SCHEMA_VERSION = 1

FileKind = Literal["image", "video"]
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif", ".webp"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".mkv", ".webm"})

_CHUNK = 1024 * 1024


class CaptureError(Exception):
    """A capture can't be imported or a bundle is invalid."""


def classify(path: Path) -> FileKind | None:
    """Kind of a capture file from its extension, or None if not supported."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return None


@dataclass
class CaptureFile:
    name: str
    original_name: str
    kind: FileKind
    size: int
    sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "original_name": self.original_name,
            "kind": self.kind,
            "size": self.size,
            "sha256": self.sha256,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Any) -> CaptureFile:
        if not isinstance(data, dict):
            raise CaptureError("file entry is not an object")
        name, kind = data.get("name"), data.get("kind")
        size, sha = data.get("size"), data.get("sha256")
        if not isinstance(name, str) or Path(name).name != name or name.startswith("."):
            raise CaptureError(f"invalid file name: {name!r}")
        if kind not in ("image", "video"):
            raise CaptureError(f"{name}: invalid kind {kind!r}")
        if not isinstance(size, int) or not isinstance(sha, str):
            raise CaptureError(f"{name}: 'size' and 'sha256' are required")
        original = data.get("original_name", name)
        metadata = data.get("metadata", {})
        return cls(
            name=name,
            original_name=original if isinstance(original, str) else name,
            kind=kind,
            size=size,
            sha256=sha,
            metadata=metadata if isinstance(metadata, dict) else {},
        )


@dataclass
class CaptureBundle:
    root: Path
    id: str
    source: str
    created: str
    files: list[CaptureFile]
    device: dict[str, str] | None = None
    source_info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path) -> CaptureBundle:
        try:
            data = read_json_object(root / CAPTURE_FILE)
        except FormatError as exc:
            raise CaptureError(str(exc)) from exc
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise CaptureError(f"{root / CAPTURE_FILE}: unsupported schema_version {version!r}")
        bundle_id, source, created = data.get("id"), data.get("source"), data.get("created")
        if not all(isinstance(v, str) for v in (bundle_id, source, created)):
            raise CaptureError(f"{root / CAPTURE_FILE}: 'id', 'source', 'created' are required")
        raw_files = data.get("files")
        if not isinstance(raw_files, list):
            raise CaptureError(f"{root / CAPTURE_FILE}: 'files' must be a list")
        device, source_info = data.get("device"), data.get("source_info", {})
        assert isinstance(bundle_id, str) and isinstance(source, str) and isinstance(created, str)
        return cls(
            root=root,
            id=bundle_id,
            source=source,
            created=created,
            files=[CaptureFile.from_dict(entry) for entry in raw_files],
            device=device if isinstance(device, dict) else None,
            source_info=source_info if isinstance(source_info, dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "source": self.source,
            "created": self.created,
            "device": self.device,
            "source_info": self.source_info,
            "files": [f.to_dict() for f in self.files],
        }

    def save(self) -> None:
        """Rewrite capture.json, e.g. after filling in per-file metadata."""
        write_json_atomic(self.root / CAPTURE_FILE, self.to_dict())

    @property
    def images(self) -> list[Path]:
        return [self.root / f.name for f in self.files if f.kind == "image"]

    @property
    def videos(self) -> list[Path]:
        return [self.root / f.name for f in self.files if f.kind == "video"]

    def verify(self) -> list[str]:
        """Re-hash every file; return a description of each problem found."""
        problems = []
        for entry in self.files:
            path = self.root / entry.name
            if not path.is_file():
                problems.append(f"{entry.name}: missing")
            elif path.stat().st_size != entry.size:
                problems.append(f"{entry.name}: size changed")
            elif sha256_file(path) != entry.sha256:
                problems.append(f"{entry.name}: contents changed")
        return problems


def list_bundles(project: Project) -> list[CaptureBundle]:
    """All complete bundles of a project, oldest first."""
    bundles = [
        CaptureBundle.load(child)
        for child in sorted(project.captures_dir.iterdir())
        if child.is_dir() and not child.name.startswith(".") and (child / CAPTURE_FILE).is_file()
    ]
    return sorted(bundles, key=lambda b: (b.created, b.id))


def import_files(
    project: Project,
    files: Sequence[Path],
    *,
    source: str,
    device: dict[str, str] | None = None,
    source_info: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> CaptureBundle:
    """Copy `files` into a new capture bundle of `project`, untouched.

    Files with the same name (e.g. IMG_0001.jpg from two folders) get a
    numeric suffix; the name they arrived with is kept as `original_name`.
    """
    if not files:
        raise CaptureError("nothing to import")
    sources = [Path(f) for f in files]
    seen: set[Path] = set()
    for path in sources:
        if classify(path) is None:
            raise CaptureError(f"{path}: not a supported image or video file")
        if not path.is_file():
            raise CaptureError(f"{path}: not a file")
        resolved = path.resolve()
        if resolved in seen:
            raise CaptureError(f"{path}: listed twice")
        seen.add(resolved)

    bundle_id = _new_bundle_id(project.captures_dir, now or datetime.now().astimezone())
    final = project.captures_dir / bundle_id
    staging = project.captures_dir / f".importing-{bundle_id}"
    staging.mkdir()
    try:
        entries = []
        used: set[str] = set()
        for path in sources:
            name = _unique_name(path.name, used)
            size, sha = _copy_and_hash(path, staging / name)
            kind = classify(path)
            assert kind is not None
            entries.append(
                CaptureFile(name=name, original_name=path.name, kind=kind, size=size, sha256=sha)
            )
        bundle = CaptureBundle(
            root=staging,
            id=bundle_id,
            source=source,
            created=utc_now(),
            files=entries,
            device=device,
            source_info=source_info or {},
        )
        bundle.save()
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    bundle.root = final
    return bundle


def import_folder(
    project: Project, folder: Path, *, now: datetime | None = None
) -> tuple[CaptureBundle, list[Path]]:
    """Import the supported files directly inside `folder` (not subfolders).

    Returns the bundle and the files that were skipped as unsupported.
    Hidden files (macOS `._*` and the like) are ignored silently.
    """
    if not folder.is_dir():
        raise CaptureError(f"{folder}: not a folder")
    candidates = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))
    accepted = [p for p in candidates if classify(p) is not None]
    skipped = [p for p in candidates if classify(p) is None]
    if not accepted:
        raise CaptureError(f"{folder}: no supported image or video files")
    bundle = import_files(
        project,
        accepted,
        source="folder",
        source_info={"folder": str(folder.absolute())},
        now=now,
    )
    return bundle, skipped


def _new_bundle_id(captures_dir: Path, now: datetime) -> str:
    base = now.strftime("%Y%m%d-%H%M%S")
    candidate, n = base, 1
    while (captures_dir / candidate).exists() or (
        captures_dir / f".importing-{candidate}"
    ).exists():
        n += 1
        candidate = f"{base}-{n}"
    return candidate


def _unique_name(name: str, used: set[str]) -> str:
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate, n = name, 1
    # Compare case-insensitively: macOS file systems are case-insensitive.
    while candidate.lower() in used or candidate.lower() == CAPTURE_FILE:
        n += 1
        candidate = f"{stem}-{n}{suffix}"
    used.add(candidate.lower())
    return candidate


def _copy_and_hash(src: Path, dst: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with src.open("rb") as fin, dst.open("xb") as fout:
        while chunk := fin.read(_CHUNK):
            digest.update(chunk)
            fout.write(chunk)
            size += len(chunk)
    shutil.copystat(src, dst)
    return size, digest.hexdigest()
