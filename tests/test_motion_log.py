# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from ez2digitize import motion, motion_log, plugins
from ez2digitize.backends.ffmpeg import FFmpeg
from ez2digitize.core.capture import CAPTURE_FILE, CaptureBundle, import_folder
from ez2digitize.core.project import Project
from ez2digitize.motion_log import MotionLogError, parse_log
from ez2digitize.video import import_video


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "project")


def _log(frames: list[dict[str, Any]], start: float = 100.0, **extra: Any) -> dict[str, Any]:
    """A phone held upright, turning about its y axis at 0.5 rad/s for 60 s."""
    times = [start + i * 0.01 for i in range(6000)]
    return {
        "format": "ez2digitize-motion",
        "version": 1,
        "device": {"make": "Google", "model": "Pixel 8"},
        "gyroscope": [[t, 0.0, 0.5, 0.0] for t in times],
        "accelerometer": [[t, 0.0, -9.6, 0.0] for t in times],
        "gravity": [[t, 0.0, -9.81, 0.0] for t in times],
        "frames": frames,
        **extra,
    }


def _pose(t: float, x: float) -> list[float]:
    return [t, 1, 0, 0, 0, 1, 0, 0, 0, 1, x, 0.0, 0.0]


def test_parse_log() -> None:
    log = parse_log(_log([{"file": "a.jpg", "t": 101.5}], poses=[_pose(101.5, 0.2)]))
    assert log.frames == {"a.jpg": 101.5}
    assert log.track.device == "Google Pixel 8" and log.track.gravity == "gravity sensor"
    assert log.track.down_at(101.5) == (0.0, 1.0, 0.0)  # the sensor pushes up
    assert log.track.turn_at(101.5) == pytest.approx(28.65, abs=0.01)
    assert log.track.pose_at(101.5) == (((1, 0, 0), (0, 1, 0), (0, 0, 1)), (0.2, 0.0, 0.0))
    assert not log.track.metric

    accel_only = _log([{"file": "a.jpg", "t": 0}])
    del accel_only["gravity"]
    assert parse_log(accel_only).track.gravity == "accelerometer"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"format": "something else"}, "not an EZ2DIGITIZE motion log"),
        ({"version": 2}, "unsupported version 2"),
        ({"frames": []}, "'frames' must list"),
        ({"frames": [{"file": "a.jpg"}]}, "needs a 'file' and a time"),
        ({"frames": [{"file": "a.jpg", "t": True}]}, "needs a 'file' and a time"),
        ({"gyroscope": [[0, 1, 2]]}, "'gyroscope' must be rows of 4 numbers"),
        ({"poses": [[0.0] * 12]}, "'poses' must be rows of 13 numbers"),
    ],
)
def test_invalid_logs(change: dict[str, Any], message: str) -> None:
    with pytest.raises(MotionLogError, match=message):
        parse_log({**_log([{"file": "a.jpg", "t": 0}]), **change})


def test_photos_with_their_log(project: Project, tmp_path: Path) -> None:
    folder = tmp_path / "shoot"
    folder.mkdir()
    for name in ("IMG_1.jpg", "IMG_2.jpg", "IMG_3.jpg"):
        Image.new("RGB", (8, 6)).save(folder / name)
    frames = [{"file": "IMG_1.jpg", "t": 101.0}, {"file": "IMG_2.jpg", "t": 105.0}]
    poses = [_pose(101.0, 0.0), _pose(105.0, 0.3)]
    log = _log(frames, poses=poses, metric=True)
    (folder / "shoot.motion.json").write_text(json.dumps(log))

    bundle, skipped = import_folder(project, folder)
    assert skipped == []
    kinds = {f.name: f.kind for f in bundle.files}
    assert kinds["shoot.motion.json"] == "motion" and len(bundle.images) == 3
    entry = {f.name: f.metadata.get("motion") for f in bundle.files}
    assert entry["IMG_1.jpg"] is not None and entry["IMG_1.jpg"]["down"] == [0, 1, 0]
    assert entry["IMG_2.jpg"] is not None and entry["IMG_2.jpg"]["metric"] is True
    assert entry["IMG_2.jpg"]["camera_to_world"][0] == [1, 0, 0, 0.3]
    assert entry["IMG_3.jpg"] is None  # not in the log

    names = [f"{bundle.id}/IMG_{n}.jpg" for n in (1, 2, 3)]
    assert sorted(motion.known_poses([bundle])) == names[:2]
    priors = plugins.pose_priors([bundle], names)["images"]
    assert sorted(priors) == names[:2] and priors[names[0]]["metric"] is True


def test_unreadable_log_is_noted(project: Project, tmp_path: Path) -> None:
    folder = tmp_path / "shoot"
    folder.mkdir()
    Image.new("RGB", (8, 6)).save(folder / "IMG_1.jpg")
    (folder / "a.motion.json").write_text("{")
    bundle, _skipped = import_folder(project, folder)
    assert "a.motion.json" in bundle.source_info["motion_log_error"]
    assert all("motion" not in f.metadata for f in bundle.files)


def test_a_videos_log_stays_with_the_video(project: Project, tmp_path: Path) -> None:
    folder = tmp_path / "shoot"
    folder.mkdir()
    Image.new("RGB", (8, 6)).save(folder / "IMG_1.jpg")
    (folder / "VID_1.mp4").write_bytes(b"video")
    (folder / "VID_1.motion.json").write_text("{}")
    bundle, skipped = import_folder(project, folder)
    assert sorted(p.name for p in skipped) == ["VID_1.motion.json", "VID_1.mp4"]
    assert [f.kind for f in bundle.files] == ["image"]


def test_video_with_its_log(project: Project, tmp_path: Path, fake_ffmpeg: FFmpeg) -> None:
    """The log beside the video replaces its own track, moved onto the video's clock."""
    clip = tmp_path / "VID_1.mp4"
    clip.write_bytes(b"pretend this is a video")
    poses = [_pose(100.0 + i * 0.05, i * 0.01) for i in range(1200)]
    log = _log([{"file": "VID_1.mp4", "t": 100.0}], poses=poses, metric=True)
    (tmp_path / "VID_1.motion.json").write_text(json.dumps(log))
    bundle = import_video(project, clip, fake_ffmpeg, frames=20)
    assert bundle.source_info["motion"]["format"] == motion_log.LOG_FORMAT
    assert bundle.source_info["motion"]["metric"] is True
    assert [f.kind for f in bundle.files][:2] == ["video", "motion"]
    frames = [f for f in bundle.files if f.kind == "image"]
    assert all(f.metadata["motion"]["down"] == [0, 1, 0] for f in frames)
    for frame in frames:
        time_s = frame.metadata["video"]["time_s"]
        x = frame.metadata["motion"]["camera_to_world"][0][3]
        assert x == pytest.approx(time_s / 5, abs=0.006)  # 0.01 every 0.05 s

    (tmp_path / "VID_1.motion.json").write_text(json.dumps(_log([{"file": "x.mp4", "t": 0}])))
    other = import_video(project, clip, fake_ffmpeg, frames=20)
    assert "don't say when VID_1.mp4 started" in other.source_info["motion"]["error"]


def test_version_1_bundles_still_load(project: Project, tmp_path: Path) -> None:
    folder = tmp_path / "shoot"
    folder.mkdir()
    Image.new("RGB", (8, 6)).save(folder / "IMG_1.jpg")
    bundle, _skipped = import_folder(project, folder)
    data = json.loads((bundle.root / CAPTURE_FILE).read_text())
    assert data["schema_version"] == 2
    data["schema_version"] = 1
    (bundle.root / CAPTURE_FILE).write_text(json.dumps(data))
    loaded = CaptureBundle.load(bundle.root)
    assert [f.name for f in loaded.files] == ["IMG_1.jpg"]
    loaded.save()
    assert json.loads((bundle.root / CAPTURE_FILE).read_text())["schema_version"] == 2
