# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import shutil
import subprocess
import threading
from pathlib import Path

import pytest
from mp4_files import TrackSpec, camm, write_mp4

from ez2digitize import motion
from ez2digitize.backends import ffmpeg
from ez2digitize.backends.ffmpeg import FFmpeg, VideoInfo
from ez2digitize.core import photos
from ez2digitize.core.capture import CaptureBundle, CaptureError, list_bundles
from ez2digitize.core.project import Project
from ez2digitize.core.runner import CancelToken, Event, Progress
from ez2digitize.video import (
    FramePlan,
    VideoImportCancelled,
    frame_progress,
    import_video,
    plan_frames,
    select_frames,
    steady,
)


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


def test_frame_progress() -> None:
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    # Turned 300° in the first two seconds, then held still.
    progress = frame_progress([0, 150, 300, 300, 300], times, 4.0)
    assert progress == pytest.approx([0, 0.375 + 0.0625, 0.75 + 0.125, 0.75 + 0.1875, 1.0])
    # A camera that hardly turned (a turntable): spaced by time.
    assert frame_progress([0, 10, 20, 30, 40], times, 4.0) is None
    assert frame_progress([], [], 4.0) is None


def test_select_by_progress() -> None:
    plan = FramePlan(1.0, 4.0)
    # Eight candidates; the first four cover three quarters of the progress.
    progress = [0.0, 0.25, 0.5, 0.74, 0.8, 0.85, 0.9, 0.95]
    scores = [1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 3.0, 1.0]
    assert select_frames(scores, plan, progress, windows=4) == [0, 1, 3, 6]


def test_steady() -> None:
    # The slowest turns at 20°/s: up to 20 * 1.5 + 10 = 40°/s is steady enough.
    assert steady([0, 1, 2, 3], [20.0, 40.0, 41.0, None]) == [0, 1, 3]
    assert steady([0, 1], [None, None]) == [0, 1]  # no gyroscope there
    # Tremor-level differences don't count.
    assert steady([0, 1], [0.5, 9.0]) == [0, 1]


def test_fast_turning_candidates_are_passed_over() -> None:
    plan = FramePlan(1.0, 4.0)
    scores = [1.0, 5.0, 2.0, 1.0, 1.0, 1.0, 1.0, 3.0]
    turning = [10.0, 80.0, 12.0, 11.0, 30.0, 30.0, 30.0, 30.0]
    # Window 0: the sharpest-scoring candidate was turning fast; window 1 is even.
    assert select_frames(scores, plan) == [1, 7]
    assert select_frames(scores, plan, turning=turning) == [2, 7]


def test_frames_follow_the_turning(project: Project, tmp_path: Path, fake_ffmpeg: FFmpeg) -> None:
    """Turning for the first 5 s of 10, then still: most frames come from the turning."""
    clip = tmp_path / "VID_0002.mp4"
    readings = []
    for i in range(1000):  # 10 ms apart
        turning = 1.0 if i < 500 else 0.0
        readings += [camm(3, 0.0, -9.81, 0.0), camm(2, 0.0, turning, 0.0)]
    write_mp4(clip, [TrackSpec("meta", "camm", 1000, [(r, 5) for r in readings])])
    bundle = import_video(project, clip, fake_ffmpeg, frames=20)
    times = [f.metadata["video"]["time_s"] for f in bundle.files if f.kind == "image"]
    assert bundle.source_info["spacing"] == {
        "by": "angle",
        "turned_deg": pytest.approx(286.5, abs=1),
    }
    assert bundle.source_info["fast_passed_over"] == 0  # the fake's sharp frames turn alike
    assert sum(t < 5 for t in times) >= 15 and len(times) <= 20
    assert times == sorted(times)


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


def test_import_records_the_motion_track(
    project: Project, tmp_path: Path, fake_ffmpeg: FFmpeg
) -> None:
    """A video with a CAMM track: each kept frame notes gravity and the turning rate."""
    clip = tmp_path / "VID_0001.mp4"
    readings = [camm(3, 0.0, -9.81, 0.0), camm(2, 0.0, 0.5, 0.0)] * 3000  # 10 ms apart
    write_mp4(clip, [TrackSpec("meta", "camm", 1000, [(r, 10) for r in readings])])
    bundle = import_video(project, clip, fake_ffmpeg, frames=20)
    assert bundle.source_info["motion"] == {
        "format": "camm",
        "device": "",
        "gravity": "accelerometer",
        "gravity_samples": 3000,
        "gyro_samples": 3000,
        "pose_samples": 0,
    }
    frames = [f for f in bundle.files if f.kind == "image"]
    assert all(f.metadata["motion"]["down"] == [0, 1, 0] for f in frames)
    assert frames[0].metadata["motion"]["turn_deg_s"] == pytest.approx(28.65, abs=0.01)
    downs = motion.measured_downs([bundle])
    assert downs[f"{bundle.id}/frame_0001.jpg"] == (0, 1, 0) and len(downs) == 20
    bundle.set_excluded(["frame_0001.jpg"])
    assert len(motion.measured_downs([bundle])) == 19  # only the photos in use


def test_unreadable_motion_track_is_noted(
    project: Project, tmp_path: Path, fake_ffmpeg: FFmpeg
) -> None:
    clip = tmp_path / "GX010001.MP4"
    broken = b"DEVC\0\x04\xff\xff" + b"\0" * 8
    write_mp4(clip, [TrackSpec("meta", "gpmd", 1000, [(broken, 1000)])])
    bundle = import_video(project, clip, fake_ffmpeg, frames=20)
    assert "unreadable motion track" in bundle.source_info["motion"]["error"]
    assert all("motion" not in f.metadata for f in bundle.files)


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
