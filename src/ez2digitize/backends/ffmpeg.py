# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""FFmpeg: read a video's properties and extract frames from it.

Only decoding and JPEG encoding are used, so any FFmpeg from 5.0 on is
accepted (`SUPPORTED_MAJOR`) rather than one pinned version: it is the
system's FFmpeg until the app bundles one (Phase 6). The version used is
recorded in every video capture.

FFmpeg applies the rotation stored in phone videos, so frames come out
upright. 10-bit HDR video is converted to 8-bit JPEG without tone mapping
(flat colours, but fine for reconstruction).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from ez2digitize.backends.common import BackendError, BackendMissing, find_tool
from ez2digitize.core.runner import ProcessStartError, Progress, run_quick

NAME = "ffmpeg"
ENV_VAR = "EZ2D_FFMPEG"
SUPPORTED_MAJOR = 5
# Folders where FFmpeg lives outside PATH on macOS (Homebrew).
SEARCH_DIRS = (Path("/opt/homebrew/bin"), Path("/usr/local/bin"))
# JPEG quality of extracted frames: FFmpeg's scale, 2 (best) to 31.
JPEG_QUALITY = 2


@dataclass(frozen=True)
class FFmpeg:
    path: Path
    probe: Path
    version: str

    @property
    def supported(self) -> bool:
        major = re.match(r"(\d+)", self.version)
        return major is not None and int(major.group(1)) >= SUPPORTED_MAJOR


@dataclass(frozen=True)
class VideoInfo:
    duration_s: float
    frame_rate: float
    width: int
    height: int
    codec: str
    # The display rotation ffprobe reports (counter-clockwise degrees), which
    # FFmpeg applies when extracting frames.
    rotation: int = 0


def parse_version(text: str) -> str | None:
    """`ffmpeg version 6.1.1-3ubuntu5 ...` -> "6.1.1-3ubuntu5"; also n7.1, 2024-... builds."""
    match = re.search(r"ffmpeg version (\S+)", text)
    if match is None:
        return None
    return match.group(1).removeprefix("n")


def locate(explicit: Path | None = None) -> FFmpeg:
    """Find ffmpeg (and ffprobe next to it, or on PATH) and read the version."""
    if explicit is None and (env := os.environ.get(ENV_VAR)):
        explicit = Path(env).expanduser()
    path = find_tool("ffmpeg", explicit=explicit, extra_dirs=SEARCH_DIRS)
    if path is None:
        raise BackendMissing(f"FFmpeg not found (looked in ${ENV_VAR}, the bundle, PATH)")
    probe = find_tool("ffprobe", explicit=path.with_name("ffprobe"))
    probe = probe or find_tool("ffprobe", extra_dirs=SEARCH_DIRS)
    if probe is None:
        raise BackendMissing(f"ffprobe not found next to {path} or on PATH")
    try:
        text = run_quick([path, "-version"])
    except ProcessStartError as exc:
        raise BackendMissing(f"FFmpeg at {path} can't be run: {exc}") from exc
    version = parse_version(text)
    if version is None:
        said = "\n".join(text.strip().splitlines()[-5:]) or "nothing"
        raise BackendMissing(
            f"no FFmpeg version in the output of {path} -version; it said:\n{said}"
        )
    return FFmpeg(path=path, probe=probe, version=version)


def probe_video(ffmpeg: FFmpeg, video: Path) -> VideoInfo:
    """Duration, frame rate and size of the first video stream (as displayed)."""
    try:
        text = run_quick(
            [
                ffmpeg.probe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,codec_name,duration"
                ":stream_side_data=rotation:format=duration",
                "-of",
                "json",
                video,
            ],  # fmt: skip
            timeout_s=60,
        )
    except ProcessStartError as exc:
        raise BackendError(f"can't run ffprobe: {exc}") from exc
    return parse_probe(text, video)


def parse_probe(text: str, video: Path) -> VideoInfo:
    try:
        data = json.loads(text)
        stream = data["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise BackendError(f"{video.name}: no video stream found") from exc
    rate = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate"))
    duration = _float(stream.get("duration")) or _float(data.get("format", {}).get("duration"))
    if not rate or not duration:
        raise BackendError(f"{video.name}: can't read the frame rate or duration")
    rotation = 0
    for side in stream.get("side_data_list", []):
        try:
            rotation = round(float(side.get("rotation", 0)))
        except (TypeError, ValueError):
            continue
    # Phones store -90 or 90 (or 270); FFmpeg rotates the frames accordingly.
    if rotation % 180:
        width, height = height, width
    codec = str(stream.get("codec_name", "?"))
    return VideoInfo(duration, rate, width, height, codec, rotation)


def _rate(value: object) -> float | None:
    try:
        rate = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def _float(value: object) -> float | None:
    try:
        number = float(str(value))
    except ValueError:
        return None
    return number if number > 0 else None


def extract_frames_argv(ffmpeg: FFmpeg, video: Path, folder: Path, rate: float) -> list[str]:
    """Command writing `rate` frames a second of `video` as folder/c_000001.jpg, ...

    Frame k is the one shown at time (k - 1) / rate: the fps filter picks
    the frame nearest each tick.
    """
    return [
        str(ffmpeg.path), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-nostats", "-progress", "pipe:1",
        "-i", str(video),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", f"fps={rate:.6g}",
        "-q:v", str(JPEG_QUALITY),
        str(folder / "c_%06d.jpg"),
    ]  # fmt: skip


class FFmpegProgress:
    """Progress from `-progress` output (`out_time_us=...`) against the duration."""

    def __init__(self, duration_s: float, message: str = "Extracting frames") -> None:
        self.duration_us = duration_s * 1e6
        self.message = message

    def __call__(self, line: str) -> Progress | None:
        key, _, value = line.partition("=")
        if key == "out_time_us" and value.strip().lstrip("-").isdigit():
            done = max(0, int(value)) / self.duration_us
            return Progress(self.message, round(min(done, 1.0), 4))
        if key == "progress" and value.strip() == "end":
            return Progress(self.message, 1.0)
        return None
