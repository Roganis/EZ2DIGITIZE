# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Motion tracks in videos: readings made from a known down direction and turning
rate, packed as a GoPro or a CAMM camera would, must come back in the camera's
axes (x right, y down, z forward).

The axis conventions were checked on GoPro's own sample files (gpmf-parser's
samples/, HERO5 to MAX): the down direction comes out along the image's y
for level shots, steeply forward where the camera looks at the floor, and
the MAX's fused gravity vector agrees with its accelerometer.
"""

import math
from collections.abc import Callable
from pathlib import Path

import pytest
from mp4_files import TrackSpec, camm, gpmf_payload, klv, nested, write_mp4

from ez2digitize import motion
from ez2digitize.motion import MotionTrack, Vector, read_motion

G = 9.80665
TURN_DEG_S = 30.0


def _unit(v: Vector) -> Vector:
    n = math.sqrt(sum(c * c for c in v))
    return (v[0] / n, v[1] / n, v[2] / n)


DOWN = _unit((0.2, 0.9, 0.3))  # camera axes, tilted and rolled a little


def _gopro_channels(camera: Vector, order: str) -> list[float]:
    """A camera-axes vector as GoPro stores it: IMU axes X left, Y back, Z up, in `order`."""
    imu = {"X": -camera[0], "Y": -camera[2], "Z": -camera[1]}
    return [-imu[c.upper()] if c.islower() else imu[c] for c in order]


def _gopro(
    path: Path,
    device: str,
    *,
    orin: str | None,
    gravity: Vector | None = None,
    seconds: int = 2,
) -> None:
    order = orin or motion.GPMF_ORDER_BY_DEVICE.get(device, "ZXY")
    accel = _gopro_channels(tuple(-G * c for c in DOWN), order)  # type: ignore[arg-type]
    gyro = _gopro_channels((math.radians(TURN_DEG_S), 0.0, 0.0), order)  # turning about x
    payloads = []
    for _ in range(seconds):
        orientation = [klv("ORIN", "c", 1, orin.encode() + b"\0")] if orin else []
        streams = [
            nested(
                "STRM",
                klv("SCAL", "s", 2, [418]),
                *orientation,
                klv("ACCL", "s", 6, [round(v * 418) for v in accel] * 200),
            ),
            nested(
                "STRM",
                klv("SCAL", "s", 2, [939]),
                *orientation,
                klv("GYRO", "s", 6, [round(v * 939) for v in gyro] * 200),
            ),
        ]
        if gravity is not None:
            values = [round(c * 32767) for c in gravity]
            streams.append(
                nested("STRM", klv("SCAL", "s", 2, [32767]), klv("GRAV", "s", 6, values * 30))
            )
        payloads.append((gpmf_payload(device, *streams), 1001))
    video = TrackSpec("vide", "avc1", 30000, [(b"frame", 1001)] * 60)
    write_mp4(path, [video, TrackSpec("meta", "gpmd", 1000, payloads)])


@pytest.mark.parametrize(
    ("device", "orin"),
    [("HERO8 Black", "zxY"), ("Hero6 Black", "YxZ"), ("Camera", None), ("Hero6 Black", None)],
    ids=["HERO8", "HERO6", "HERO5 (no ORIN)", "HERO6 (old firmware)"],
)
def test_gopro_imu(tmp_path: Path, device: str, orin: str | None) -> None:
    path = tmp_path / "GX010001.MP4"
    order = orin or motion.GPMF_ORDER_BY_DEVICE[device]
    _gopro(path, device, orin=order if orin else None)
    track = read_motion(path)
    assert track is not None
    assert (track.format, track.device, track.gravity) == ("gpmd", device, "accelerometer")
    for t in (0.5, 1.0, 1.7):
        assert track.down_at(t) == pytest.approx(DOWN, abs=2e-3)
        assert track.turn_at(t) == pytest.approx(TURN_DEG_S, abs=0.1)
    assert track.down_at(10.0) is None  # past the end


def test_gopro_prefers_fused_gravity(tmp_path: Path) -> None:
    path = tmp_path / "GX010002.MP4"
    fused = _unit((0.0, 1.0, -0.2))
    _gopro(path, "HERO9 Black", orin="zxY", gravity=fused)
    track = read_motion(path)
    assert track is not None and track.gravity == "gravity vector"
    assert track.down_at(1.0) == pytest.approx(fused, abs=1e-4)
    # Zeros (as before the camera's fusion settles) don't count.
    _gopro(path, "HERO8 Black", orin="zxY", gravity=(0.0, 0.0, 0.0))
    track = read_motion(path)
    assert track is not None and track.gravity == "accelerometer"


def test_unknown_gopro_axes_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "GOPR0001.MP4"
    _gopro(path, "Fusion", orin=None)  # a 360° camera, in no table
    assert read_motion(path) is None


def _camm_file(path: Path, samples: list[bytes]) -> None:
    video = TrackSpec("vide", "avc1", 30000, [(b"frame", 1001)] * 60)
    meta = TrackSpec("meta", "camm", 1000, [(data, 10) for data in samples])
    write_mp4(path, [video, meta])


def test_camm_orientation(tmp_path: Path) -> None:
    # Camera to world: 30° about the camera's x (pitched); world y is down.
    angle = math.radians(30)
    samples = [camm(0, angle, 0.0, 0.0), camm(2, 0.0, 0.0, math.radians(45))] * 100
    samples += [camm(3, 0.0, -G, 0.0)] * 10  # ignored: the orientation is exact
    path = tmp_path / "VID_0001.mp4"
    _camm_file(path, samples)
    track = read_motion(path)
    assert track is not None and (track.format, track.gravity) == ("camm", "orientation")
    # Down in camera axes = R^T (0, 1, 0) for R the rotation about x.
    assert track.down_at(0.5) == pytest.approx((0, math.cos(angle), -math.sin(angle)), abs=1e-6)
    assert track.turn_at(0.5) == pytest.approx(45.0, abs=1e-3)


def test_camm_accelerometer(tmp_path: Path) -> None:
    path = tmp_path / "VID_0002.mp4"
    _camm_file(path, [camm(3, *(-G * c for c in DOWN))] * 200)
    track = read_motion(path)
    assert track is not None and track.gravity == "accelerometer"
    assert track.down_at(1.0) == pytest.approx(DOWN, abs=1e-6)
    assert track.turn_at(1.0) is None  # no gyroscope samples


def test_display_rotation(tmp_path: Path) -> None:
    """A phone video stored sideways: FFmpeg turns the frames, the vectors turn with them."""
    path = tmp_path / "VID_0003.mp4"
    _camm_file(path, [camm(3, *(-G * c for c in DOWN))] * 200)
    # ffprobe's -90: shown turned 90° clockwise, so stored right is shown down.
    track = read_motion(path, rotation_deg=-90)
    assert track is not None
    assert track.down_at(1.0) == pytest.approx((-DOWN[1], DOWN[0], DOWN[2]), abs=1e-6)
    upside_down = read_motion(path, rotation_deg=180)
    assert upside_down is not None
    assert upside_down.down_at(1.0) == pytest.approx((-DOWN[0], -DOWN[1], DOWN[2]), abs=1e-6)


def test_no_motion_track(tmp_path: Path) -> None:
    path = tmp_path / "plain.mp4"
    write_mp4(path, [TrackSpec("vide", "avc1", 30000, [(b"frame", 1001)] * 10)])
    assert read_motion(path) is None
    other = tmp_path / "clip.webm"
    other.write_bytes(b"\x1aE\xdf\xa3 not an MP4")
    assert read_motion(other) is None


def test_corrupt_gpmf(tmp_path: Path) -> None:
    path = tmp_path / "GX010003.MP4"
    broken = b"DEVC\0\x04\xff\xff" + b"\0" * 8  # claims far more than it holds
    write_mp4(path, [TrackSpec("meta", "gpmd", 1000, [(broken, 1000)])])
    with pytest.raises(motion.MotionError):
        read_motion(path)


def test_frame_entry() -> None:
    track = MotionTrack("camm", gravity="orientation")
    track.down = [(t / 10, (0.0, 2.0, 0.0)) for t in range(20)]
    track.gyro = [(t / 10, (0.0, math.radians(10), 0.0)) for t in range(20)]
    assert motion.frame_motion(track, 1.0) == {"down": [0, 1, 0], "turn_deg_s": 10.0}
    assert motion.frame_motion(track, 50.0) == {}


def test_describe() -> None:
    assert motion.describe({}) is None
    gopro = {"motion": {"format": "gpmd", "device": "HERO8 Black", "gravity": "accelerometer"}}
    assert motion.describe(gopro) == (
        "Uses its GoPro motion data (HERO8 Black): gravity from its accelerometer."
    )
    gyro_only = {"motion": {"format": "camm", "device": "", "gravity": ""}}
    assert motion.describe(gyro_only) == "Has CAMM motion data, but no gravity in it."
    broken = {"motion": {"error": "x.mp4: unreadable motion track: corrupt box size"}}
    assert "can't be read" in (motion.describe(broken) or "")


def _gyro_track(rate_at: Callable[[float], Vector], seconds: float, hz: int = 400) -> MotionTrack:
    track = MotionTrack("camm")
    track.gyro = [(i / hz, rate_at(i / hz)) for i in range(int(seconds * hz) + 1)]
    return track


def test_angle_travelled() -> None:
    steady = _gyro_track(lambda _t: (0.0, math.radians(30), 0.0), 4.0)
    angles = steady.angle_travelled([0.0, 1.0, 2.0, 4.0, 9.0])
    assert angles == pytest.approx([0, 30, 60, 120, 120], abs=0.5)
    # Turning about a tilted axis is the same angle.
    tilted = _gyro_track(lambda _t: _scaled(_unit((1.0, 1.0, 0.0)), math.radians(30)), 2.0)
    assert tilted.angle_travelled([2.0]) == pytest.approx([60], abs=0.5)
    assert MotionTrack("camm").angle_travelled([1.0]) is None


def test_tremor_adds_little() -> None:
    """Shaking at 8 Hz, ±20°/s: |rate| adds up to about 50° in 4 s, the path to about 10°."""
    amplitude = math.radians(20)
    shaking = _gyro_track(lambda t: (amplitude * math.sin(2 * math.pi * 8 * t), 0.0, 0.0), 4.0)
    naive = sum(abs(amplitude * math.sin(2 * math.pi * 8 * i / 400)) / 400 for i in range(1600))
    assert math.degrees(naive) > 45
    travelled = shaking.angle_travelled([4.0])
    assert travelled is not None and travelled[0] < math.degrees(naive) / 4


def _scaled(v: Vector, k: float) -> Vector:
    return (v[0] * k, v[1] * k, v[2] * k)
