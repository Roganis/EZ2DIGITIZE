# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Motion logs: the phone's motion recorded next to its photos or video.

Videos can carry their motion inside (GoPro's GPMF, Google's CAMM; see
ez2digitize.motion), but photos can't, and a phone app can't add a track
to the video its camera writes. So a capture app records the motion in a
file of its own, `<anything>.motion.json`, which travels in the capture
bundle as a file of kind "motion", untouched like the photos:

    {
      "format": "ez2digitize-motion", "version": 1,
      "device": {"make": "Google", "model": "Pixel 8"},
      "gyroscope":     [[t, x, y, z], ...],   # rad/s
      "accelerometer": [[t, x, y, z], ...],   # m/s², as the sensor reports
      "gravity":       [[t, x, y, z], ...],   # m/s², the phone's fused gravity
      "poses": [[t, r00, r01, r02, r10, r11, r12, r20, r21, r22, x, y, z], ...],
      "metric": true,
      "frames": [{"file": "IMG_0001.jpg", "t": 12.345}, ...]
    }

- `t`: seconds on one clock for everything, the one the camera stamps its
  frames with (on Android, sensors and camera frames share it on most
  phones). Any origin.
- Vectors are in the camera's axes as the image is stored (x right, y down,
  z forward, before any EXIF or display rotation), the axes COLMAP reads
  photos in. The accelerometer and gravity readings are what Android
  reports: at rest they point up, opposing gravity.
- `poses`: camera to world, the rotation by rows, then the camera centre,
  where the app tracks the camera (ARCore). The world has y down along
  gravity, as in CAMM; `metric` says its unit is the metre (ARCore's is).
- `frames`: when each photo was taken. For a video, one entry for the
  video, `t` being when its first frame was shown.

Every list is optional except `frames`; gravity from the fused readings is
preferred over the accelerometer's. A video's log is read instead of its
own motion track (`video.import_video` looks for `<video stem>.motion.json`
next to it); a photo set's (one per folder or upload) gives each photo
listed in `frames` its motion entry when the bundle is assembled, as video
frames get theirs (motion.frame_motion).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ez2digitize.core.capture import CaptureFile
from ez2digitize.motion import MOTION_KEY, Matrix, MotionTrack, Vector, frame_motion

SUFFIX = ".motion.json"
FORMAT = "ez2digitize-motion"
VERSION = 1
LOG_FORMAT = "log"  # MotionTrack.format of a track read from a log


class MotionLogError(ValueError):
    pass


@dataclass
class MotionLog:
    track: MotionTrack  # on the log's clock
    frames: dict[str, float]  # file name -> time on the log's clock


def is_log(path: Path | str) -> bool:
    return str(path).lower().endswith(SUFFIX)


def read_log(path: Path) -> MotionLog:
    """Read a motion log; raises MotionLogError if it isn't a valid one."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MotionLogError(f"{path.name}: {exc}") from exc
    return parse_log(data, path.name)


def parse_log(data: Any, name: str = "motion log") -> MotionLog:
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise MotionLogError(f"{name}: not an EZ2DIGITIZE motion log")
    if data.get("version") != VERSION:
        raise MotionLogError(f"{name}: unsupported version {data.get('version')!r}")
    device = data.get("device")
    model = ""
    if isinstance(device, dict):
        model = " ".join(str(device[k]) for k in ("make", "model") if device.get(k))
    track = MotionTrack(LOG_FORMAT, device=model)
    track.gyro = [(t, v) for t, v in _vectors(data, "gyroscope", name)]
    gravity = _vectors(data, "gravity", name)
    accel = _vectors(data, "accelerometer", name)
    readings, track.gravity = (gravity, "gravity sensor") if gravity else (accel, "accelerometer")
    if not readings:
        track.gravity = ""
    track.down = [(t, (-v[0], -v[1], -v[2])) for t, v in readings]
    for row in _rows(data, "poses", 13, name):
        t, r, centre = row[0], row[1:10], row[10:13]
        rotation: Matrix = ((r[0], r[1], r[2]), (r[3], r[4], r[5]), (r[6], r[7], r[8]))
        track.orientations.append((t, rotation))
        track.positions.append((t, (centre[0], centre[1], centre[2])))
    track.metric = data.get("metric") is True
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise MotionLogError(f"{name}: 'frames' must list the files and when they were taken")
    times = {}
    for entry in frames:
        file, when = (
            (entry.get("file"), entry.get("t")) if isinstance(entry, dict) else (None, None)
        )
        if not isinstance(file, str) or not isinstance(when, int | float) or not _number(when):
            raise MotionLogError(f"{name}: each frame needs a 'file' and a time 't'")
        times[file] = float(when)
    return MotionLog(track, times)


def log_for(video: Path) -> Path | None:
    """The motion log recorded with `video`: `<video stem>.motion.json` next to it."""
    for name in (video.stem + SUFFIX, video.stem + SUFFIX.upper()):
        if (candidate := video.with_name(name)).is_file():
            return candidate
    return None


def video_track(path: Path, video_name: str, rotation_deg: int = 0) -> MotionTrack | None:
    """A video's motion from its log, on the video's clock and in its frames' axes.

    `rotation_deg` as for motion.read_motion. None if the log holds no
    gravity or gyroscope readings; raises MotionLogError if it is invalid or
    doesn't say when the video started.
    """
    log = read_log(path)
    start = log.frames.get(video_name)
    if start is None:
        raise MotionLogError(f"{path.name}: its 'frames' don't say when {video_name} started")
    if not (log.track.down or log.track.gyro):
        return None
    return log.track.shifted(-start).rotated(-rotation_deg)


def annotate(folder: Path, entries: Sequence[CaptureFile]) -> str | None:
    """Give the photos among `entries` their motion from the bundle's log, if it has one.

    Photos are found by the name they arrived with. Returns a problem to
    record (an unreadable log, or more than one), else None.
    """
    logs = [e for e in entries if e.kind == "motion"]
    if not logs:
        return None
    if len(logs) > 1:
        return f"more than one motion log ({', '.join(e.name for e in logs)}); none is used"
    try:
        log = read_log(folder / logs[0].name)
    except MotionLogError as exc:
        return str(exc)
    for entry in entries:
        t = log.frames.get(entry.original_name)
        if entry.kind == "image" and t is not None and (motion := frame_motion(log.track, t)):
            entry.metadata[MOTION_KEY] = motion
    return None


def _vectors(data: dict[str, Any], key: str, name: str) -> list[tuple[float, Vector]]:
    return [(row[0], (row[1], row[2], row[3])) for row in _rows(data, key, 4, name)]


def _rows(data: dict[str, Any], key: str, width: int, name: str) -> list[list[float]]:
    rows = data.get(key, [])
    if not isinstance(rows, list) or not all(
        isinstance(row, list) and len(row) == width and all(_number(c) for c in row) for row in rows
    ):
        raise MotionLogError(f"{name}: '{key}' must be rows of {width} numbers")
    return sorted(([float(c) for c in row] for row in rows), key=lambda row: row[0])


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
