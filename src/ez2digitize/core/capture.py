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
         "kind": "image", "size": 4123456, "sha256": "...", "metadata": {},
         "excluded": false}
      ]
    }

`metadata` holds what was learned about a file: the photo checks store
their inspection under `metadata["photo"]` (see core.photos). `excluded`
marks a file the user left out of the reconstruction; the file itself stays
in the bundle, untouched, so it can be brought back. Readers older than
this field ignore it (and use every file).

Folder import, video, phone upload and the Android app all produce this
format; the pipeline only reads bundles. A bundle is assembled in a hidden
temporary folder and renamed into place when complete, so an interrupted
import never shows up as a bundle.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable, Iterable, Sequence
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
    excluded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "original_name": self.original_name,
            "kind": self.kind,
            "size": self.size,
            "sha256": self.sha256,
            "metadata": self.metadata,
            "excluded": self.excluded,
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
        excluded = data.get("excluded", False)
        if not isinstance(excluded, bool):
            raise CaptureError(f"{name}: 'excluded' must be true or false")
        return cls(
            name=name,
            original_name=original if isinstance(original, str) else name,
            kind=kind,
            size=size,
            sha256=sha,
            metadata=metadata if isinstance(metadata, dict) else {},
            excluded=excluded,
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
    def used(self) -> list[CaptureFile]:
        """The files the reconstruction uses: all but the excluded ones."""
        return [f for f in self.files if not f.excluded]

    @property
    def images(self) -> list[Path]:
        """Images the reconstruction uses."""
        return [self.root / f.name for f in self.used if f.kind == "image"]

    @property
    def videos(self) -> list[Path]:
        """Videos the reconstruction uses."""
        return [self.root / f.name for f in self.used if f.kind == "video"]

    @property
    def excluded(self) -> list[CaptureFile]:
        return [f for f in self.files if f.excluded]

    def set_excluded(self, names: Iterable[str], excluded: bool = True) -> None:
        """Leave files out of the reconstruction (or bring them back) and save."""
        wanted = set(names)
        known = {f.name for f in self.files}
        if unknown := sorted(wanted - known):
            raise CaptureError(f"capture {self.id} has no file {unknown[0]!r}")
        for entry in self.files:
            if entry.name in wanted:
                entry.excluded = excluded
        self.save()

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

    def fill(staging: Path) -> list[CaptureFile]:
        used: set[str] = set()
        return [copy_into(staging, path, used) for path in sources]

    return assemble_bundle(
        project, fill, source=source, device=device, source_info=source_info, now=now
    )


def assemble_bundle(
    project: Project,
    fill: Callable[[Path], list[CaptureFile]],
    *,
    source: str,
    device: dict[str, str] | None = None,
    source_info: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> CaptureBundle:
    """Create a bundle from the files `fill` puts into the (staging) folder it gets.

    The bundle appears under its final name only once complete; if `fill`
    raises, nothing is left behind. `fill` may add to `source_info`.
    """
    bundle_id = _new_bundle_id(project.captures_dir, now or datetime.now().astimezone())
    final = project.captures_dir / bundle_id
    staging = project.captures_dir / f".importing-{bundle_id}"
    staging.mkdir()
    info = source_info if source_info is not None else {}
    try:
        entries = fill(staging)
        if not entries:
            raise CaptureError("nothing to import")
        bundle = CaptureBundle(
            root=staging,
            id=bundle_id,
            source=source,
            created=utc_now(),
            files=entries,
            device=device,
            source_info=info,
        )
        bundle.save()
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    bundle.root = final
    return bundle


def copy_into(folder: Path, path: Path, used: set[str]) -> CaptureFile:
    """Copy `path` into a bundle folder under a name not in `used`, hashing it."""
    kind = classify(path)
    if kind is None:
        raise CaptureError(f"{path}: not a supported image or video file")
    name = _unique_name(path.name, used)
    size, sha = _copy_and_hash(path, folder / name)
    return CaptureFile(name=name, original_name=path.name, kind=kind, size=size, sha256=sha)


def add_file(folder: Path, name: str, *, original_name: str, kind: FileKind) -> CaptureFile:
    """Describe a file already written into a bundle folder (e.g. a video frame)."""
    path = folder / name
    return CaptureFile(
        name=name,
        original_name=original_name,
        kind=kind,
        size=path.stat().st_size,
        sha256=sha256_file(path),
    )


def import_folder(
    project: Project, folder: Path, *, now: datetime | None = None
) -> tuple[CaptureBundle, list[Path]]:
    """Import the photos directly inside `folder` (not subfolders).

    Returns the bundle and the files that were skipped: unsupported ones,
    and videos, which are imported one by one (`ez2digitize.video`).
    Hidden files (macOS `._*` and the like) are ignored silently.
    """
    if not folder.is_dir():
        raise CaptureError(f"{folder}: not a folder")
    candidates = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))
    accepted = [p for p in candidates if classify(p) == "image"]
    skipped = [p for p in candidates if classify(p) != "image"]
    if not accepted:
        videos = [p for p in candidates if classify(p) == "video"]
        hint = f"; import videos one by one, e.g. {videos[0].name}" if videos else ""
        raise CaptureError(f"{folder}: no supported photos{hint}")
    bundle = import_files(
        project,
        accepted,
        source="folder",
        source_info={"folder": str(folder.absolute())},
        now=now,
    )
    return bundle, skipped


def import_masks(project: Project, bundle: CaptureBundle, masks: Path) -> int:
    """Copy masks for `bundle`'s images into `masks/<bundle id>/`.

    Masks are looked up by the name each image arrived with, in COLMAP's
    naming (`IMG_0001.jpg.png`, or `IMG_0001.png`), and stored under the
    image's name in the bundle. Returns how many images got a mask.
    """
    target = project.masks_dir / bundle.id
    copied = 0
    for entry in bundle.files:
        if entry.kind != "image":
            continue
        original = Path(entry.original_name)
        for candidate in (masks / f"{original.name}.png", masks / f"{original.stem}.png"):
            if candidate.is_file():
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, target / f"{entry.name}.png")
                copied += 1
                break
    return copied


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
