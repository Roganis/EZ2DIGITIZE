# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Motion data recorded inside videos: GoPro's GPMF and Google's CAMM tracks.

Many action cameras and some phones store their gyroscope and accelerometer
readings as a metadata track in the video file. This reads them, in the
camera's own axes as COLMAP uses them (x right, y down, z forward, in the
frames as FFmpeg extracts them), so the video import can note, for every
frame it keeps:

- `down`: the direction of gravity, averaged over a short window, so the
  upright direction is measured rather than guessed from how the camera
  was held (ez2digitize.orientation);
- `turn_deg_s`: how fast the camera was turning.

Formats, both documented by their makers and read here without their
libraries (standard library only):

- GPMF (GoPro, sample entry `gpmd`): https://github.com/gopro/gpmf-parser.
  Big-endian key-length-value records; a payload a second, holding each
  sensor's samples for that second. GoPro's IMU axes are X left, Y back
  (towards the screen), Z up; each model stores them in its own order,
  given by the stream's ORIN record or, before that existed, by model.
- CAMM (Google, sample entry `camm`):
  https://developers.google.com/streetview/publish/camm-spec. One reading
  a sample, little-endian, already in the camera axes above. Orientation
  samples (type 0) give gravity exactly (world Y is down); otherwise the
  accelerometer, which at rest measures the push opposing gravity.

The video's display rotation (phones store one; FFmpeg applies it when
extracting frames) is applied to the vectors too, so they match the frames.
"""

from __future__ import annotations

import bisect
import math
import struct
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ez2digitize.core.capture import CaptureBundle
from ez2digitize.core.mp4 import Mp4Error, Track, read_samples, read_tracks

Vector = tuple[float, float, float]

# A video frame's capture.json entry: {"down": [x, y, z], "turn_deg_s": n}.
MOTION_KEY = "motion"
GPMF_FORMAT = "gpmd"
CAMM_FORMAT = "camm"


class MotionError(ValueError):
    pass


# --- GPMF -----------------------------------------------------------------------

# struct format and size of each GPMF value type (big-endian).
GPMF_TYPES: dict[str, tuple[str, int]] = {
    "b": ("b", 1), "B": ("B", 1), "c": ("c", 1), "d": ("d", 8), "f": ("f", 4),
    "F": ("4s", 4), "j": ("q", 8), "J": ("Q", 8), "l": ("i", 4), "L": ("I", 4),
    "q": ("i", 4), "Q": ("q", 8), "s": ("h", 2), "S": ("H", 2), "U": ("16s", 16),
}  # fmt: skip
NESTED = "\0"


@dataclass
class Klv:
    key: str
    type: str
    size: int  # bytes per sample
    repeat: int  # samples
    data: bytes
    children: list[Klv] = field(default_factory=list)

    def values(self) -> list[float]:
        """The numbers, flattened (size // value size per sample)."""
        if self.type not in GPMF_TYPES or self.type in "cFU":
            return []
        code, width = GPMF_TYPES[self.type]
        count = self.size * self.repeat // width
        numbers = struct.unpack_from(f">{count}{code}", self.data)
        if self.type == "q":
            return [n / 65536 for n in numbers]
        if self.type == "Q":
            return [n / 2**32 for n in numbers]
        return [float(n) for n in numbers]

    def text(self) -> str:
        if self.type != "c":
            return ""
        return self.data[: self.size * self.repeat].split(b"\0")[0].decode("latin-1")


def parse_gpmf(data: bytes) -> list[Klv]:
    """The KLV records in a GPMF payload, nested ones parsed too."""
    records = []
    position = 0
    while position + 8 <= len(data):
        key_bytes, kind, size, repeat = struct.unpack_from(">4scBH", data, position)
        if key_bytes == b"\0\0\0\0":
            break
        length = size * repeat
        body = data[position + 8 : position + 8 + length]
        if len(body) < length:
            raise MotionError("GPMF record runs past its payload")
        record = Klv(key_bytes.decode("latin-1"), kind.decode("latin-1"), size, repeat, body)
        if record.type == NESTED:
            record.children = parse_gpmf(body)
        records.append(record)
        position += 8 + (length + 3) // 4 * 4
    return records


def gpmf_streams(payload: Sequence[Klv]) -> Iterator[tuple[str, list[Klv]]]:
    """(device name, the records of one STRM) for every stream in a payload."""
    for device in payload:
        if device.key != "DEVC":
            continue
        name = next((r.text() for r in device.children if r.key == "DVNM"), "")
        for stream in device.children:
            if stream.key == "STRM":
                yield name, stream.children


# Channel order of GoPro's IMU streams before ORIN records existed, by device name
# (gpmf-parser's README); lower case is a negated axis.
GPMF_ORDER_BY_DEVICE = {"Camera": "ZXY", "Hero6 Black": "YxZ", "Hero7 Black": "YxZ"}
GPMF_IMU = ("ACCL", "GYRO")


def _gopro_to_camera(v: Vector) -> Vector:
    """GoPro IMU axes (X left, Y back, Z up) to the camera's (x right, y down, z forward)."""
    return (-v[0], -v[2], -v[1])


def _reorder(values: Sequence[float], order: str) -> Vector:
    """Channels in `order` (e.g. "YxZ": Y, -X, Z) to (X, Y, Z)."""
    axes = [0.0, 0.0, 0.0]
    for value, letter in zip(values, order, strict=True):
        axes["XYZ".index(letter.upper())] = -value if letter.islower() else value
    return (axes[0], axes[1], axes[2])


def _scaled(record: Klv, scal: Klv | None, elements: int) -> list[list[float]]:
    """The record's samples, each `elements` numbers, divided by SCAL."""
    numbers = record.values()
    scale = scal.values() if scal is not None else [1.0]
    rows = []
    for start in range(0, len(numbers) - elements + 1, elements):
        row = numbers[start : start + elements]
        if len(scale) == elements:
            rows.append([v / s for v, s in zip(row, scale, strict=True)])
        else:
            rows.append([v / scale[0] for v in row])
    return rows


def _spread(start: float, duration: float, count: int) -> list[float]:
    """Times of `count` evenly spaced samples in a payload."""
    return [start + duration * (i + 0.5) / count for i in range(count)]


def read_gpmf(path: Path, track: Track) -> MotionTrack:
    """The gravity and turning rate in a GoPro GPMF track."""
    motion = MotionTrack(GPMF_FORMAT)
    accel_down: list[tuple[float, Vector]] = []
    gravity_down: list[tuple[float, Vector]] = []
    for sample, data in read_samples(path, track):
        for device, records in gpmf_streams(parse_gpmf(data)):
            scal: Klv | None = None
            orin: str | None = None
            for record in records:
                if record.key == "SCAL":
                    scal = record
                elif record.key == "ORIN":
                    orin = record.text()
                elif record.key in GPMF_IMU and record.size // _width(record) == 3:
                    order = orin or GPMF_ORDER_BY_DEVICE.get(device)
                    if order is None or len(order) != 3:
                        continue  # an axis order we don't know
                    motion.device = motion.device or device
                    rows = _scaled(record, scal, 3)
                    times = _spread(sample.time_s, sample.duration_s, len(rows))
                    for t, row in zip(times, rows, strict=True):
                        v = _gopro_to_camera(_reorder(row, order))
                        if record.key == "ACCL":  # the push against gravity
                            accel_down.append((t, (-v[0], -v[1], -v[2])))
                        else:
                            motion.turn.append((t, _length(v)))
                elif record.key == "GRAV" and record.size // _width(record) == 3:
                    # Fused, and already in the image's axes (checked against
                    # ACCL on GoPro MAX and HERO8 samples).
                    rows = _scaled(record, scal, 3)
                    times = _spread(sample.time_s, sample.duration_s, len(rows))
                    for t, row in zip(times, rows, strict=True):
                        if _length((row[0], row[1], row[2])) > 0.5:  # zeros until it settles
                            gravity_down.append((t, (row[0], row[1], row[2])))
    if gravity_down:
        motion.down, motion.gravity = gravity_down, "gravity vector"
    elif accel_down:
        motion.down, motion.gravity = accel_down, "accelerometer"
    return motion


def _width(record: Klv) -> int:
    return GPMF_TYPES.get(record.type, ("", 1))[1]


# --- CAMM -----------------------------------------------------------------------

# Payload sizes by CAMM sample type (after the 4-byte reserved and type).
CAMM_SIZES = {0: 12, 1: 8, 2: 12, 3: 12, 4: 12, 5: 24, 6: 60, 7: 12}


def read_camm(path: Path, track: Track) -> MotionTrack:
    """The gravity and turning rate in a CAMM track."""
    motion = MotionTrack(CAMM_FORMAT)
    oriented: list[tuple[float, Vector]] = []
    accel_down: list[tuple[float, Vector]] = []
    for sample, data in read_samples(path, track):
        if len(data) < 4:
            continue
        (kind,) = struct.unpack_from("<H", data, 2)
        if kind not in CAMM_SIZES or len(data) < 4 + CAMM_SIZES[kind]:
            continue
        t = sample.time_s
        if kind == 0:  # camera to world, world Y down along gravity
            axis_angle = struct.unpack_from("<3f", data, 4)
            oriented.append((t, _mul_transposed(_rodrigues(axis_angle), (0.0, 1.0, 0.0))))
        elif kind == 2:
            gyro = struct.unpack_from("<3f", data, 4)
            motion.turn.append((t, _length(gyro)))
        elif kind == 3:  # the push against gravity, as phones measure it
            a = struct.unpack_from("<3f", data, 4)
            accel_down.append((t, (-a[0], -a[1], -a[2])))
    if oriented:
        motion.down, motion.gravity = oriented, "orientation"
    elif accel_down:
        motion.down, motion.gravity = accel_down, "accelerometer"
    return motion


# --- both -----------------------------------------------------------------------

# Seconds to average over: the accelerometer also feels the camera's own
# movements, which mostly cancel over a second; fused gravity needs little.
ACCEL_WINDOW_S = 1.0
FUSED_WINDOW_S = 0.2
TURN_WINDOW_S = 0.1


@dataclass
class MotionTrack:
    format: str  # GPMF_FORMAT or CAMM_FORMAT
    device: str = ""
    gravity: str = ""  # what down comes from: "gravity vector", "orientation", "accelerometer"
    down: list[tuple[float, Vector]] = field(default_factory=list)  # (time, camera axes)
    turn: list[tuple[float, float]] = field(default_factory=list)  # (time, rad/s)

    def rotated(self, degrees: int) -> MotionTrack:
        """The same, in the axes of the frames turned by `degrees` clockwise."""
        turn_matrix = _image_turn(degrees)
        down = [(t, _mul(turn_matrix, v)) for t, v in self.down]
        return MotionTrack(self.format, self.device, self.gravity, down, self.turn)

    def down_at(self, time_s: float) -> Vector | None:
        """Gravity's direction (unit) in the camera's axes around `time_s`; None if unknown."""
        window = ACCEL_WINDOW_S if self.gravity == "accelerometer" else FUSED_WINDOW_S
        near = [v for _t, v in _between(self.down, time_s, window)]
        if not near:
            return None
        mean = (
            sum(v[0] for v in near) / len(near),
            sum(v[1] for v in near) / len(near),
            sum(v[2] for v in near) / len(near),
        )
        length = _length(mean)
        if length == 0:
            return None
        return (mean[0] / length, mean[1] / length, mean[2] / length)

    def turn_at(self, time_s: float) -> float | None:
        """How fast the camera turned around `time_s`, in degrees a second."""
        near = [rate for _t, rate in _between(self.turn, time_s, TURN_WINDOW_S)]
        if not near:
            return None
        return math.degrees(sum(near) / len(near))

    def summary(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "device": self.device,
            "gravity": self.gravity,
            "gravity_samples": len(self.down),
            "gyro_samples": len(self.turn),
        }


def read_motion(path: Path, rotation_deg: int = 0) -> MotionTrack | None:
    """The motion track of a video, in the axes of its frames as FFmpeg shows them.

    `rotation_deg`: the display rotation ffprobe reports (counter-clockwise),
    which FFmpeg undoes when extracting frames. None if the video has no
    motion track we can read, or one without gravity or gyroscope data.
    Raises MotionError if it has one but it is corrupt.
    """
    try:
        tracks = read_tracks(path)
        for track in tracks:
            if track.format == GPMF_FORMAT:
                motion = read_gpmf(path, track)
            elif track.format == CAMM_FORMAT:
                motion = read_camm(path, track)
            else:
                continue
            if motion.down or motion.turn:
                return motion.rotated(-rotation_deg)
    except (Mp4Error, MotionError, struct.error) as exc:
        raise MotionError(f"{path.name}: unreadable motion track: {exc}") from exc
    except OSError as exc:
        raise MotionError(f"{path.name}: {exc}") from exc
    return None


def frame_motion(motion: MotionTrack, time_s: float) -> dict[str, Any]:
    """What a video frame's capture.json entry records of the motion at `time_s`."""
    entry: dict[str, Any] = {}
    down = motion.down_at(time_s)
    if down is not None:
        entry["down"] = [round(c, 5) for c in down]
    turn = motion.turn_at(time_s)
    if turn is not None:
        entry["turn_deg_s"] = round(turn, 2)
    return entry


FORMAT_NAMES = {GPMF_FORMAT: "GoPro", CAMM_FORMAT: "CAMM"}


def describe(source_info: dict[str, Any]) -> str | None:
    """A video capture's motion data in a sentence; None if it had none."""
    summary = source_info.get(MOTION_KEY)
    if not isinstance(summary, dict):
        return None
    if "error" in summary:
        return f"Its motion data can't be read ({summary['error']}), so it isn't used."
    name = FORMAT_NAMES.get(str(summary.get("format")), "motion")
    device = f" ({summary['device']})" if summary.get("device") else ""
    if summary.get("gravity"):
        return f"Uses its {name} motion data{device}: gravity from its {summary['gravity']}."
    return f"Has {name} motion data{device}, but no gravity in it."


def measured_downs(bundles: Iterable[CaptureBundle]) -> dict[str, Vector]:
    """Gravity's direction in the camera's axes, by COLMAP image name, where measured."""
    found = {}
    for bundle in bundles:
        for file in bundle.used:
            entry = file.metadata.get(MOTION_KEY)
            down = entry.get("down") if isinstance(entry, dict) else None
            if (
                isinstance(down, list)
                and len(down) == 3
                and all(isinstance(c, int | float) for c in down)
            ):
                found[f"{bundle.id}/{file.name}"] = (float(down[0]), float(down[1]), float(down[2]))
    return found


def _between(
    samples: list[tuple[float, Any]], time_s: float, window: float
) -> list[tuple[float, Any]]:
    lo = bisect.bisect_left(samples, time_s - window / 2, key=lambda s: s[0])
    hi = bisect.bisect_right(samples, time_s + window / 2, key=lambda s: s[0])
    return samples[lo:hi]


def _image_turn(degrees: int) -> tuple[Vector, Vector, Vector]:
    """Vectors in a picture's axes (x right, y down) once it is turned clockwise."""
    quarter = round(degrees / 90) % 4
    c, s = ((1, 0), (0, 1), (-1, 0), (0, -1))[quarter]
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _rodrigues(axis_angle: Sequence[float]) -> tuple[Vector, Vector, Vector]:
    angle = _length((axis_angle[0], axis_angle[1], axis_angle[2]))
    if angle < 1e-12:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    x, y, z = (a / angle for a in axis_angle)
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return (
        (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
        (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
        (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
    )


def _mul(m: tuple[Vector, Vector, Vector], v: Vector) -> Vector:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _mul_transposed(m: tuple[Vector, Vector, Vector], v: Vector) -> Vector:
    return (
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    )


def _length(v: Vector) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
