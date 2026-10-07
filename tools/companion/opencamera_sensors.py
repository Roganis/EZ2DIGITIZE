#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Turn an OpenCamera Sensors recording into an EZ2DIGITIZE motion log.

OpenCamera Sensors (https://github.com/MobileRoboticsSkoltech/OpenCamera-Sensors,
GPL-3.0, on F-Droid) records video with the gyroscope and accelerometer on
the camera frames' clock: until the companion app exists, it gives a phone
video the same motion data a GoPro's has (gravity for the upright, frames
picked by angle, blurred ones dropped), though no camera poses.

    python tools/companion/opencamera_sensors.py VID_20261007_143000.mp4

writes `VID_20261007_143000.motion.json` next to the video, where Import
video finds it (ez2digitize.motion_log). Standard library only.

What the app writes (read from its source, which differs from its README
on the file names), in DCIM/OpenCamera:

- the video, `VID_<date>_<time>.mp4`;
- a folder `<date>_<time>/` beside it with
  - `VID_<date>_<time>gyro.csv`, `...accel.csv` (and `...magnetic.csv`):
    lines `x,y,z,timestamp_ns`, in Android's device axes;
  - `VID_<date>_<time>_imu_timestamps.csv`: one line per video frame, its
    camera timestamp in ns, on the sensors' clock (when the phone supports
    it: the app's settings say whether timestamps are synchronized).

The video's pixels are stored in the camera sensor's orientation, which the
CSVs don't record: `--sensor-orientation` (the back camera's
SENSOR_ORIENTATION, 90 on nearly every phone) turns the device axes into
the stored image's. The first frame timestamp is taken as the video's start.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

Vector = tuple[float, float, float]
Row = tuple[float, Vector]  # seconds, vector

NAME = re.compile(r"^(?P<prefix>.*?)(?P<stamp>\d{8}_\d{6}(?:Z)?)(?:_\d+)?$")


class ConvertError(Exception):
    pass


def device_to_image(v: Vector, sensor_orientation: int) -> Vector:
    """Android device axes to the back camera's stored-image axes (x right, y down, z forward).

    The same conversion as the companion app's Axes.deviceToImage: at
    orientation 0 the image is the screen (x, -y); each quarter turn maps
    (a, b) to (b, -a); the back camera looks out of the back (z = -z).
    """
    a, b = v[0], -v[1]
    for _ in range((sensor_orientation % 360) // 90):
        a, b = b, -a
    return (a, b, -v[2])


def read_vectors(path: Path, sensor_orientation: int) -> list[Row]:
    """Lines `x,y,z,timestamp_ns`, in seconds and image axes, sorted by time."""
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            x, y, z, t = (float(c) for c in line.split(","))
        except ValueError as exc:
            raise ConvertError(f"{path.name}, line {number}: expected x,y,z,timestamp") from exc
        rows.append((t / 1e9, device_to_image((x, y, z), sensor_orientation)))
    return sorted(rows, key=lambda row: row[0])


def read_frame_times(path: Path) -> list[float]:
    times = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                times.append(int(line.strip()) / 1e9)
            except ValueError as exc:
                raise ConvertError(f"{path.name}, line {number}: expected a timestamp") from exc
    return sorted(times)


def find_files(video: Path, folder: Path | None = None) -> dict[str, Path]:
    """The recording's CSVs: "gyro", "accel" and "frames" (the frame timestamps)."""
    match = NAME.match(video.stem)
    candidates = []
    if folder is not None:
        candidates.append(folder)
    elif match:
        candidates.append(video.parent / match["stamp"])
    candidates.append(video.parent)
    for place in candidates:
        if not place.is_dir():
            continue
        found = {}
        for path in place.iterdir():
            name = path.name
            if not name.startswith(video.stem) or not name.lower().endswith(".csv"):
                continue
            rest = name[len(video.stem) : -len(".csv")].lstrip("_").lower()
            if rest in ("gyro", "accel"):
                found[rest] = path
            elif rest in ("imu_timestamps", "timestamps"):
                found["frames"] = path
        if "frames" in found and ("gyro" in found or "accel" in found):
            return found
    where = " or ".join(str(p) for p in candidates)
    raise ConvertError(
        f"no OpenCamera Sensors data for {video.name} in {where} (looked for "
        f"{video.stem}gyro.csv, ...accel.csv and ..._imu_timestamps.csv)"
    )


def motion_log(
    video_name: str,
    gyro: Sequence[Row],
    accel: Sequence[Row],
    frame_times: Sequence[float],
    device: dict[str, str] | None = None,
) -> dict[str, object]:
    """The motion log (ez2digitize.motion_log's format) for a video."""
    if not frame_times:
        raise ConvertError("the frame timestamps file is empty")

    def rows(data: Sequence[Row]) -> list[list[float]]:
        return [[round(t, 6), *(round(c, 6) for c in v)] for t, v in data]

    log: dict[str, object] = {"format": "ez2digitize-motion", "version": 1}
    if device:
        log["device"] = device
    log["gyroscope"] = rows(gyro)
    log["accelerometer"] = rows(accel)
    log["frames"] = [{"file": video_name, "t": round(frame_times[0], 6)}]
    return log


def overlap_problem(frame_times: Sequence[float], *streams: Sequence[Row]) -> str | None:
    """A warning when the sensors don't cover the video: likely another clock."""
    start, end = frame_times[0], frame_times[-1]
    for data in streams:
        if data and (data[0][0] > end or data[-1][0] < start):
            return (
                "the sensor readings and the video frames don't overlap in time: this phone "
                "probably doesn't stamp them on one clock (see the app's IMU settings)"
            )
    return None


def convert(
    video: Path, *, folder: Path | None = None, sensor_orientation: int = 90
) -> tuple[Path, str | None]:
    """Write `<video stem>.motion.json` next to `video`; return it and any warning."""
    files = find_files(video, folder)
    gyro = read_vectors(files["gyro"], sensor_orientation) if "gyro" in files else []
    accel = read_vectors(files["accel"], sensor_orientation) if "accel" in files else []
    frames = read_frame_times(files["frames"])
    log = motion_log(video.name, gyro, accel, frames)
    out = video.with_name(f"{video.stem}.motion.json")
    out.write_text(json.dumps(log, separators=(",", ":")) + "\n", encoding="utf-8")
    return out, overlap_problem(frames, gyro, accel)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("videos", type=Path, nargs="+", help="OpenCamera Sensors videos")
    parser.add_argument(
        "--sensors", type=Path, help="the folder with the CSVs (default: found next to the video)"
    )
    parser.add_argument(
        "--sensor-orientation",
        type=int,
        default=90,
        choices=(0, 90, 180, 270),
        help="the back camera's SENSOR_ORIENTATION (default 90, nearly every phone)",
    )
    args = parser.parse_args(argv)
    failed = 0
    for video in args.videos:
        try:
            out, warning = convert(
                video, folder=args.sensors, sensor_orientation=args.sensor_orientation
            )
        except (ConvertError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            failed += 1
            continue
        print(f"{video.name} -> {out.name}")
        if warning:
            print(f"warning: {warning}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
