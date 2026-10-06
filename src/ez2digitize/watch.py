# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Watch folder: photos a phone syncs to this computer.

For people whose phone already copies its photos here (Syncthing, iCloud
Drive, Google Drive, Dropbox, Nextcloud...): pick the folder, take the
photos, and they are imported as a capture bundle once they have arrived.

When is a synced set complete? There is no "done" signal, so:

- New photos are the ones that appear after watching starts (optionally
  also those modified in the last while). Not by modification time alone:
  sync tools keep the phone's capture time, so a photo taken ten minutes
  ago and synced now looks old.
- A photo is complete when its size and modification time have not changed
  for `quiet` seconds. Files a sync tool is still writing (Syncthing's
  `.syncthing.IMG_1.jpg.tmp`, iCloud's `.IMG_1.jpg.icloud` placeholders,
  `IMG_1.jpg.part` and the like) count as arriving, never as photos; one
  left unchanged for longer than `settle` is taken as abandoned.
- The set has settled when nothing new or changing has been seen for
  `settle` seconds. The GUI shows this and leaves the Import to the user;
  `ez2d watch` imports then.

Polling, not file system notifications: it needs nothing beyond the
standard library and works on synced and network folders alike. Photos
already in the project (same size and content) are not imported again.
Videos are counted but left to Import video, as in folder import.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ez2digitize.core.capture import (
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    CaptureBundle,
    CaptureError,
    classify,
    import_files,
    list_bundles,
)
from ez2digitize.core.project import Project

QUIET_S = 5.0  # a photo unchanged this long has arrived
SETTLE_S = 30.0  # nothing new or changing this long: the set looks complete
MAX_DEPTH = 4  # folders below the watched one (DCIM/Camera/2026/10, ...)
# What sync tools write while a file is on its way (besides hidden files).
TEMP_SUFFIXES = (".tmp", ".part", ".partial", ".crdownload", ".download", ".icloud", ".!sync")


@dataclass
class _Seen:
    size: int
    mtime_ns: int
    changed: float  # when size or time last changed (or it appeared)


@dataclass(frozen=True)
class WatchState:
    ready: tuple[Path, ...]  # complete new photos, oldest first
    arriving: int  # new photos still changing, and sync tools' temporary files
    videos: int  # complete new videos (left to Import video)
    quiet_for: float  # seconds since anything new or changing was seen
    settle: float

    @property
    def settled(self) -> bool:
        return bool(self.ready) and not self.arriving and self.quiet_for >= self.settle

    def describe(self) -> str:
        if not self.ready and not self.arriving:
            text = "No new photos yet."
        else:
            text = (
                f"{len(self.ready)} new photo(s) ready" if self.ready else "No new photos ready yet"
            )
            if self.arriving:
                text += f", {self.arriving} still arriving"
            text += "."
            if self.settled:
                text += f" Nothing new for {self.quiet_for:.0f} s: the set looks complete."
        if self.videos:
            text += f" {self.videos} video(s): import them with Import video."
        return text


@dataclass
class FolderWatch:
    """Polls `folder` for new photos (see the module's description).

    `since`: also count files already there whose modification time is at
    or after it (a time.time() value); None for only what appears from now.
    """

    folder: Path
    since: float | None = None
    quiet: float = QUIET_S
    settle: float = SETTLE_S
    clock: Callable[[], float] = time.time
    _baseline: set[Path] = field(default_factory=set, init=False)
    _seen: dict[Path, _Seen] = field(default_factory=dict, init=False)
    _activity: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not self.folder.is_dir():
            raise CaptureError(f"{self.folder}: not a folder")
        now = self.clock()
        self._activity = now
        for path, stat in _scan(self.folder):
            if self.since is None or stat.st_mtime < self.since:
                self._baseline.add(path)
            else:
                # Already there: arrived long enough ago unless it is still changing.
                self._seen[path] = _Seen(stat.st_size, stat.st_mtime_ns, now - self.quiet)

    def poll(self) -> WatchState:
        now = self.clock()
        present: set[Path] = set()
        for path, stat in _scan(self.folder):
            if path in self._baseline:
                continue
            present.add(path)
            seen = self._seen.get(path)
            if seen is None or (seen.size, seen.mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                self._seen[path] = _Seen(stat.st_size, stat.st_mtime_ns, now)
                self._activity = now
        for gone in set(self._seen) - present:  # renamed (a temporary file done) or removed
            del self._seen[gone]
            self._activity = now
        ready, arriving, videos = [], 0, 0
        for path, seen in self._seen.items():
            if _temporary(path):
                if now - seen.changed < self.settle:  # else abandoned by the sync tool
                    arriving += 1
                continue
            kind = classify(path)
            if seen.size == 0 or now - seen.changed < self.quiet:
                arriving += 1
            elif kind == "image":
                ready.append((seen.mtime_ns, path))
            else:
                videos += 1
        return WatchState(
            ready=tuple(path for _mtime, path in sorted(ready)),
            arriving=arriving,
            videos=videos,
            quiet_for=now - self._activity,
            settle=self.settle,
        )


def import_ready(project: Project, state: WatchState, folder: Path) -> CaptureBundle:
    """Import the state's ready photos into a new capture bundle (source "watch").

    Photos already in the project (same size and SHA-256) are left out.
    """
    known = {(f.size, f.sha256) for b in list_bundles(project) for f in b.files}
    new = [p for p in state.ready if (p.stat().st_size, _sha256(p)) not in known]
    if not new:
        raise CaptureError(
            f"all {len(state.ready)} photo(s) are already in the project"
            if state.ready
            else "no new photos to import"
        )
    return import_files(project, new, source="watch", source_info={"folder": str(folder)})


def _scan(folder: Path, depth: int = 0) -> Iterator[tuple[Path, os.stat_result]]:
    """Files under `folder` (not in hidden folders such as .stversions), with their stat."""
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                if depth < MAX_DEPTH and not entry.name.startswith("."):
                    yield from _scan(Path(entry.path), depth + 1)
            elif entry.is_file():
                path = Path(entry.path)
                if _temporary(path) or (
                    classify(path) is not None and not entry.name.startswith(".")
                ):
                    yield path, entry.stat()
        except OSError:
            continue  # vanished while looking: the next poll sees it


def _temporary(path: Path) -> bool:
    """A sync tool's file for a photo or video on its way (IMG_1.jpg.part, ...)."""
    name = path.name.lower()
    if not name.endswith(TEMP_SUFFIXES):
        return False
    return any(suffix + "." in name for suffix in IMAGE_SUFFIXES | VIDEO_SUFFIXES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
