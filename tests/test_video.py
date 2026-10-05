# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from ez2digitize.backends import ffmpeg
from ez2digitize.backends.ffmpeg import FFmpeg, VideoInfo
from ez2digitize.core import photos
from ez2digitize.core.capture import CaptureBundle, CaptureError, list_bundles
from ez2digitize.core.project import Project
from ez2digitize.core.runner import CancelToken, Event, Progress
from ez2digitize.video import (
    FramePlan,
    VideoImportCancelled,
    import_video,
    plan_frames,
    select_frames,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX fake tools")


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "project")


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"pretend this is a video")
    return path


def info(duration: float = 60.0, rate: float = 30.0) -> VideoInfo:
    return VideoInfo(duration, rate, 1920, 1080, "h264")


def test_plan_frames() -> None:
    assert plan_frames(info(), 100) == FramePlan(pytest.approx(100 / 60), pytest.approx(400 / 60))  # type: ignore[arg-type]
    # Short clips: at most 5 kept frames a second, candidates capped by the video's rate.
    assert plan_frames(info(duration=4), 100) == FramePlan(5.0, 20.0)
    assert plan_frames(info(duration=4, rate=12), 100) == FramePlan(5.0, 12.0)
    with pytest.raises(ValueError, match="at least one"):
        plan_frames(info(), 0)


def test_select_sharpest_per_window() -> None:
    plan = FramePlan(rate=1.0, candidate_rate=4.0)
    scores: list[float | None] = [1, 5, 2, 3, 9, 1, 1, 1, None, None, None, None, 2]
    # Windows: [0-3] -> 1, [4-7] -> 4, [8-11] all unreadable, [12] -> 12.
    assert select_frames(scores, plan) == [1, 4, 12]


def test_select_when_candidates_are_capped() -> None:
    # 12 fps video, 5 frames a second wanted: 2.4 candidates per window.
    plan = FramePlan(rate=5.0, candidate_rate=12.0)
    chosen = select_frames([1.0] * 24, plan)
    assert len(chosen) == 10 and chosen == sorted(chosen)


def test_import_keeps_the_sharp_frames(project: Project, clip: Path, fake_ffmpeg: FFmpeg) -> None:
    events: list[Event] = []
    bundle = import_video(project, clip, fake_ffmpeg, frames=20, on_event=events.append)
    assert bundle.source == "video"
    assert [p.name for p in bundle.videos] == ["clip.mp4"]
    assert (bundle.root / "clip.mp4").read_bytes() == clip.read_bytes()
    names = [p.name for p in bundle.images]
    assert names == [f"frame_{i:04d}.jpg" for i in range(1, 21)]
    frames = [f for f in bundle.files if f.kind == "image"]
    # The fake makes every third of four candidates sharp: those are kept.
    assert [f.metadata["video"]["time_s"] for f in frames[:3]] == [0.25, 0.75, 1.25]
    assert frames[0].original_name == "clip.mp4 at 0.25 s"
    assert bundle.source_info | {"ffmpeg": "?"} == {
        "video": "clip.mp4",
        "ffmpeg": "?",
        "frames_wanted": 20,
        "frame_rate": 2.0,
        "candidate_rate": 8.0,
        "candidates": 80,
        "frames": 20,
    }
    assert not (bundle.root / ".candidates").exists()
    assert bundle.verify() == []
    assert CaptureBundle.load(bundle.root).to_dict() == bundle.to_dict()
    messages = {e.message for e in events if isinstance(e, Progress)}
    assert messages == {"Extracting frames", "Choosing the sharpest frames"}


def test_video_frames_get_no_focal_length_warning(
    project: Project, clip: Path, fake_ffmpeg: FFmpeg
) -> None:
    bundle = import_video(project, clip, fake_ffmpeg, frames=20)
    photos.inspect_bundle(bundle)
    codes = [f.code for f in photos.check_project([bundle])]
    assert "no-focal-length" not in codes and "blurry" not in codes


def test_failure_leaves_nothing(
    project: Project, clip: Path, fake_ffmpeg: FFmpeg, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_FAIL", "ffmpeg")
    with pytest.raises(CaptureError, match="Invalid data found"):
        import_video(project, clip, fake_ffmpeg)
    assert list(project.captures_dir.iterdir()) == []


def test_cancel(
    project: Project, clip: Path, fake_ffmpeg: FFmpeg, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SLEEP", "ffmpeg")
    cancel = CancelToken()
    threading.Timer(0.5, cancel.cancel).start()
    with pytest.raises(VideoImportCancelled):
        import_video(project, clip, fake_ffmpeg, cancel=cancel)
    assert list(project.captures_dir.iterdir()) == []


def test_rejects_non_videos(project: Project, tmp_path: Path, fake_ffmpeg: FFmpeg) -> None:
    with pytest.raises(CaptureError, match="not a supported video"):
        import_video(project, tmp_path / "a.jpg", fake_ffmpeg)
    with pytest.raises(CaptureError, match="not a file"):
        import_video(project, tmp_path / "missing.mp4", fake_ffmpeg)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs FFmpeg")
def test_real_ffmpeg(project: Project, tmp_path: Path) -> None:
    video = tmp_path / "real.mp4"
    subprocess.run(  # noqa: S603 - test helper, argument list
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=25:duration=4",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],  # fmt: skip
        check=True,
    )
    bundle = import_video(project, video, ffmpeg.locate(), frames=8)
    assert len(bundle.images) == 8
    assert [b.id for b in list_bundles(project)] == [bundle.id]
    info = photos.inspect_photo(bundle.images[0])
    assert info.size == (320, 240) and info.error is None
