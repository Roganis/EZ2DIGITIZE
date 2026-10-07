# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The OpenCamera Sensors converter (tools/companion), read back by the desktop."""

from pathlib import Path

import opencamera_sensors as ocs
import pytest

from ez2digitize.motion_log import read_log, video_track

STAMP = "20261007_143000"
VIDEO = f"VID_{STAMP}.mp4"


def _recording(folder: Path, *, offset_s: float = 0.0) -> Path:
    """A phone upright in portrait, turning about its long axis at 0.5 rad/s for 10 s,
    written as OpenCamera Sensors writes it."""
    video = folder / VIDEO
    video.write_bytes(b"video")
    data = folder / STAMP
    data.mkdir()
    start_ns = 5_000_000_000_000  # times since boot
    sensors = [start_ns + i * 5_000_000 for i in range(2000)]  # 200 Hz
    gyro = "".join(f"0.0,0.5,0.0,{t}\n" for t in sensors)
    accel = "".join(f"0.01,9.81,-0.02,{t}\n" for t in sensors)
    offset = int(offset_s * 1e9)
    frames = "".join(f"{start_ns + offset + 100_000_000 + i * 33_333_333}\n" for i in range(290))
    (data / f"VID_{STAMP}gyro.csv").write_text(gyro)
    (data / f"VID_{STAMP}accel.csv").write_text(accel)
    (data / f"VID_{STAMP}magnetic.csv").write_text("1,2,3,4\n")
    (data / f"VID_{STAMP}_imu_timestamps.csv").write_text(frames)
    return video


def test_device_to_image_axes() -> None:
    push = (0.0, 9.81, 0.0)  # upright in portrait, at rest
    assert ocs.device_to_image(push, 90) == pytest.approx((-9.81, 0.0, 0.0))
    assert ocs.device_to_image(push, 0) == pytest.approx((0.0, -9.81, 0.0))
    assert ocs.device_to_image(push, 270) == pytest.approx((9.81, 0.0, 0.0))
    assert ocs.device_to_image((0.0, 0.0, 1.0), 90) == pytest.approx((0.0, 0.0, -1.0))


def test_convert_and_read_back(tmp_path: Path) -> None:
    video = _recording(tmp_path)
    out, warning = ocs.convert(video)
    assert out == tmp_path / f"VID_{STAMP}.motion.json" and warning is None

    log = read_log(out)
    assert log.frames == {VIDEO: pytest.approx(5000.1)}
    assert log.track.gravity == "accelerometer" and len(log.track.gyro) == 2000
    # As stored (landscape), gravity points along the image's +x ...
    down = log.track.down_at(5001.0)
    assert down is not None and down == pytest.approx((1.0, 0.0, 0.0), abs=0.01)
    # ... and in the frames FFmpeg extracts (the video's -90° display
    # rotation applied: portrait, upright), straight down, at the video's times.
    track = video_track(out, VIDEO, rotation_deg=-90)
    assert track is not None
    down = track.down_at(1.0)
    assert down is not None and down == pytest.approx((0.0, 1.0, 0.0), abs=0.01)
    assert track.turn_at(1.0) == pytest.approx(28.65, abs=0.01)


def test_found_next_to_the_video_or_given(tmp_path: Path) -> None:
    video = _recording(tmp_path)
    moved = tmp_path / "elsewhere"
    (tmp_path / STAMP).rename(moved)
    with pytest.raises(ocs.ConvertError, match="no OpenCamera Sensors data"):
        ocs.convert(video)
    out, _warning = ocs.convert(video, folder=moved)
    assert out.exists()


def test_other_clock_is_warned_about(tmp_path: Path) -> None:
    video = _recording(tmp_path, offset_s=1000.0)
    _out, warning = ocs.convert(video)
    assert warning is not None and "don't overlap" in warning


def test_bad_lines(tmp_path: Path) -> None:
    video = _recording(tmp_path)
    (tmp_path / STAMP / f"VID_{STAMP}gyro.csv").write_text("0,0.5,0,1\nnot,a,row\n")
    with pytest.raises(ocs.ConvertError, match="line 2: expected x,y,z,timestamp"):
        ocs.convert(video)


def test_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    video = _recording(tmp_path)
    assert ocs.main([str(video), "--sensor-orientation", "90"]) == 0
    assert f"{VIDEO} -> VID_{STAMP}.motion.json" in capsys.readouterr().out
    assert ocs.main([str(tmp_path / "VID_19990101_000000.mp4")]) == 1
