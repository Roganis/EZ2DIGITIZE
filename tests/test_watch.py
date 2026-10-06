# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import os
from pathlib import Path

import pytest

from ez2digitize.core.capture import CaptureError, list_bundles
from ez2digitize.core.project import Project
from ez2digitize.watch import FolderWatch, import_ready


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def write(path: Path, data: bytes, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def clock() -> Clock:
    return Clock()


def test_new_photos_once_they_stop_changing(tmp_path: Path, clock: Clock) -> None:
    write(tmp_path / "old.jpg", b"old")  # there before: not new
    watch = FolderWatch(tmp_path, quiet=5, settle=30, clock=clock)
    assert watch.poll().describe() == "No new photos yet."

    # A photo synced now, keeping the phone's capture time (an hour ago).
    write(tmp_path / "Camera" / "IMG_1.jpg", b"one", mtime=clock.now - 3600)
    state = watch.poll()
    assert state.ready == () and state.arriving == 1  # just appeared
    clock.now += 3
    write(tmp_path / "Camera" / "IMG_1.jpg", b"one, more", mtime=clock.now)  # still growing
    assert watch.poll().arriving == 1  # a change is seen when polled
    clock.now += 4
    assert watch.poll().arriving == 1  # changed 4 s ago
    clock.now += 2
    state = watch.poll()
    assert [p.name for p in state.ready] == ["IMG_1.jpg"] and not state.settled
    assert state.describe() == "1 new photo(s) ready."
    clock.now += 30
    state = watch.poll()
    assert state.settled
    assert state.describe().endswith("Nothing new for 36 s: the set looks complete.")


def test_sync_tools_files_on_their_way(tmp_path: Path, clock: Clock) -> None:
    watch = FolderWatch(tmp_path, quiet=5, settle=30, clock=clock)
    write(tmp_path / ".syncthing.IMG_2.jpg.tmp", b"half")
    write(tmp_path / ".IMG_3.HEIC.icloud", b"placeholder")
    write(tmp_path / ".DS_Store", b"x")  # not a photo on its way: ignored
    write(tmp_path / "notes.txt", b"x")
    clock.now += 10
    state = watch.poll()
    assert state.ready == () and state.arriving == 2 and not state.settled
    # Syncthing renames its temporary file when done.
    (tmp_path / ".syncthing.IMG_2.jpg.tmp").rename(tmp_path / "IMG_2.jpg")
    clock.now += 6
    state = watch.poll()
    assert state.arriving == 2  # IMG_2 renamed just now (new to us), the placeholder
    clock.now += 6
    assert [p.name for p in watch.poll().ready] == ["IMG_2.jpg"]
    # A placeholder left unchanged for longer than `settle` is abandoned.
    clock.now += 30
    state = watch.poll()
    assert state.arriving == 0 and state.settled


def test_since_counts_recent_files_already_there(tmp_path: Path, clock: Clock) -> None:
    write(tmp_path / "a.jpg", b"a", mtime=clock.now - 7200)
    write(tmp_path / "b.jpg", b"b", mtime=clock.now - 600)
    write(tmp_path / "c.mp4", b"c", mtime=clock.now - 60)
    watch = FolderWatch(tmp_path, since=clock.now - 1800, clock=clock)
    state = watch.poll()
    assert [p.name for p in state.ready] == ["b.jpg"] and state.videos == 1
    assert "1 video(s): import them with Import video." in state.describe()


def test_hidden_and_deep_folders_are_skipped(tmp_path: Path, clock: Clock) -> None:
    watch = FolderWatch(tmp_path, quiet=0, clock=clock)
    write(tmp_path / ".stversions" / "IMG_9.jpg", b"old version")
    write(tmp_path / "a" / "b" / "c" / "d" / "IMG_4.jpg", b"4 deep")
    write(tmp_path / "a" / "b" / "c" / "d" / "e" / "IMG_5.jpg", b"5 deep: too far")
    clock.now += 1
    assert [p.name for p in watch.poll().ready] == ["IMG_4.jpg"]


def test_not_a_folder(tmp_path: Path) -> None:
    with pytest.raises(CaptureError, match="not a folder"):
        FolderWatch(tmp_path / "missing")


def test_import_leaves_out_photos_already_in_the_project(tmp_path: Path, clock: Clock) -> None:
    project = Project.create(tmp_path / "p")
    synced = tmp_path / "synced"
    synced.mkdir()
    watch = FolderWatch(synced, quiet=0, clock=clock)
    write(synced / "IMG_1.jpg", b"one")
    write(synced / "IMG_2.jpg", b"two")
    clock.now += 1
    bundle = import_ready(project, watch.poll(), synced)
    assert sorted(f.name for f in bundle.files) == ["IMG_1.jpg", "IMG_2.jpg"]
    assert bundle.source == "watch"

    write(synced / "IMG_3.jpg", b"three")
    clock.now += 1
    second = import_ready(project, watch.poll(), synced)  # 1 and 2 are still "ready"
    assert [f.name for f in second.files] == ["IMG_3.jpg"]
    with pytest.raises(CaptureError, match="all 3 photo"):
        import_ready(project, watch.poll(), synced)
    assert len(list_bundles(project)) == 2
