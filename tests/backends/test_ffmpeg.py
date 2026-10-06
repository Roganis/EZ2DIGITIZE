# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import json
from pathlib import Path

import pytest

from ez2digitize.backends import ffmpeg
from ez2digitize.backends.common import BackendError, BackendMissing
from ez2digitize.backends.ffmpeg import FFmpeg, FFmpegProgress, VideoInfo
from ez2digitize.core.runner import Progress


@pytest.mark.parametrize(
    ("text", "version"),
    [
        ("ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023", "6.1.1-3ubuntu5"),
        ("ffmpeg version n7.1 Copyright", "7.1"),
        (
            "ffmpeg version 2024-10-02-git-358fdf3083-full_build-www.gyan.dev",
            "2024-10-02-git-358fdf3083-full_build-www.gyan.dev",
        ),  # noqa: E501
        ("nothing here", None),
    ],
)
def test_parse_version(text: str, version: str | None) -> None:
    assert ffmpeg.parse_version(text) == version


@pytest.mark.parametrize(
    ("version", "supported"), [("6.1.1-3ubuntu5", True), ("7.1", True), ("4.4.2", False)]
)
def test_supported(version: str, supported: bool) -> None:
    assert FFmpeg(Path("ffmpeg"), Path("ffprobe"), version).supported is supported


def _probe(**stream: object) -> str:
    base = {"codec_name": "hevc", "width": 3840, "height": 2160, "avg_frame_rate": "30000/1001"}
    return json.dumps({"streams": [base | stream], "format": {"duration": "42.5"}})


def test_parse_probe() -> None:
    info = ffmpeg.parse_probe(_probe(), Path("v.mp4"))
    assert info == VideoInfo(42.5, pytest.approx(29.97, abs=0.01), 3840, 2160, "hevc")  # type: ignore[arg-type]


def test_parse_probe_applies_rotation() -> None:
    rotated = _probe(side_data_list=[{"rotation": -90}])
    assert ffmpeg.parse_probe(rotated, Path("v.mp4")).width == 2160


@pytest.mark.parametrize(
    "text",
    [
        json.dumps({"streams": []}),
        "not json",
        _probe(avg_frame_rate="0/0", r_frame_rate="0/0"),
    ],
)
def test_parse_probe_rejects(text: str) -> None:
    with pytest.raises(BackendError, match="v.mp4"):
        ffmpeg.parse_probe(text, Path("v.mp4"))


def test_progress() -> None:
    progress = FFmpegProgress(10.0)
    assert progress("out_time_us=2500000") == Progress("Extracting frames", 0.25)
    assert progress("out_time_us=-9") == Progress("Extracting frames", 0.0)
    assert progress("progress=end") == Progress("Extracting frames", 1.0)
    assert progress("frame=12") is None


def test_locate(fake_ffmpeg: FFmpeg, monkeypatch: pytest.MonkeyPatch) -> None:
    found = ffmpeg.locate(fake_ffmpeg.path)
    assert (found.path, found.probe, found.version) == (
        fake_ffmpeg.path,
        fake_ffmpeg.probe,
        "7.1",
    )
    monkeypatch.setenv(ffmpeg.ENV_VAR, str(fake_ffmpeg.path))
    assert ffmpeg.locate().path == fake_ffmpeg.path


def test_locate_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(BackendMissing, match="FFmpeg not found"):
        ffmpeg.locate(tmp_path / "nothing")


def test_probe_and_argv(fake_ffmpeg: FFmpeg, tmp_path: Path) -> None:
    info = ffmpeg.probe_video(fake_ffmpeg, tmp_path / "v.mp4")
    assert (info.duration_s, info.frame_rate, info.width) == (10.0, 30.0, 640)
    argv = ffmpeg.extract_frames_argv(fake_ffmpeg, tmp_path / "v.mp4", tmp_path, 8.0)
    assert argv[argv.index("-vf") + 1] == "fps=8"
    assert argv[-1] == str(tmp_path / "c_%06d.jpg")
