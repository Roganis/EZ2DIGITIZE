# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Video import: a capture bundle of the sharpest frame in each time window.

The bundle keeps the original video, untouched, and adds the chosen frames
as its images (`frame_0001.jpg`, ...), so the pipeline reads it like any
photo capture. Each frame's capture.json entry records the time it was
taken from (`metadata["video"]`); the bundle's `source_info` records how
(FFmpeg version, frame rates).

Frames are not sampled uniformly: video from a moving phone has motion
blur that comes and goes. The video is cut into as many windows as frames
wanted; FFmpeg extracts CANDIDATES_PER_FRAME frames per window, and the
sharpest of each (by the photo checks' score) is kept.

Videos that carry a motion track (GoPro's GPMF, Google's CAMM; see
ez2digitize.motion) also give each frame the direction of gravity and how
fast the camera was turning (`metadata["motion"]`); `source_info["motion"]`
says where it came from.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ez2digitize.backends import ffmpeg as ffmpeg_backend
from ez2digitize.backends.common import BackendError
from ez2digitize.backends.ffmpeg import FFmpeg, FFmpegProgress, VideoInfo
from ez2digitize.core.capture import (
    CaptureBundle,
    CaptureError,
    CaptureFile,
    add_file,
    assemble_bundle,
    classify,
    copy_into,
)
from ez2digitize.core.photos import INSPECT_THREADS, measure_sharpness
from ez2digitize.core.project import Project
from ez2digitize.core.resources import cpu_threads
from ez2digitize.core.runner import CancelToken, EventHandler, Progress, run_process
from ez2digitize.motion import MOTION_KEY, MotionError, MotionTrack, frame_motion, read_motion

# About right for an object filmed all around in 30 s to 2 min.
DEFAULT_FRAMES = 100
CANDIDATES_PER_FRAME = 4
# Frames kept per second at most: closer frames add little but time.
MAX_FRAMES_PER_S = 5.0
VIDEO_KEY = "video"
CANDIDATES_DIR = ".candidates"


class VideoImportCancelled(Exception):
    pass


@dataclass(frozen=True)
class FramePlan:
    rate: float  # frames kept per second
    candidate_rate: float  # frames extracted per second, to choose from


def plan_frames(info: VideoInfo, frames: int) -> FramePlan:
    """Rates that give about `frames` frames from the whole video."""
    if frames < 1:
        raise ValueError("at least one frame is needed")
    rate = min(frames / info.duration_s, MAX_FRAMES_PER_S, info.frame_rate)
    return FramePlan(rate, min(rate * CANDIDATES_PER_FRAME, info.frame_rate))


def select_frames(scores: Sequence[float | None], plan: FramePlan) -> list[int]:
    """Index of the sharpest candidate in each window, in time order.

    Candidate i was taken at i / candidate_rate seconds; windows are
    1 / rate seconds long. Unreadable candidates (None) are skipped.
    """
    best: dict[int, int] = {}
    for index, score in enumerate(scores):
        if score is None:
            continue
        window = math.floor(index * plan.rate / plan.candidate_rate + 1e-9)
        current = best.get(window)
        if current is None or score > (scores[current] or 0.0):
            best[window] = index
    return [best[window] for window in sorted(best)]


def import_video(
    project: Project,
    video: Path,
    ffmpeg: FFmpeg,
    *,
    frames: int = DEFAULT_FRAMES,
    on_event: EventHandler | None = None,
    cancel: CancelToken | None = None,
    now: datetime | None = None,
) -> CaptureBundle:
    """Import `video` as a capture of about `frames` frames.

    Raises CaptureError if the video can't be read or FFmpeg fails,
    VideoImportCancelled if `cancel` was used; nothing is left behind then.
    """
    if classify(video) != "video":
        raise CaptureError(f"{video}: not a supported video file")
    if not video.is_file():
        raise CaptureError(f"{video}: not a file")
    try:
        info = ffmpeg_backend.probe_video(ffmpeg, video)
    except BackendError as exc:
        raise CaptureError(str(exc)) from exc
    plan = plan_frames(info, frames)
    emit = on_event or (lambda _event: None)
    motion: MotionTrack | None = None
    motion_error = None
    try:
        motion = read_motion(video, info.rotation)
    except MotionError as exc:
        motion_error = str(exc)  # recorded; the frames are still worth having
    source_info: dict[str, Any] = {
        "video": video.name,
        "ffmpeg": ffmpeg.version,
        "frames_wanted": frames,
        "frame_rate": round(plan.rate, 4),
        "candidate_rate": round(plan.candidate_rate, 4),
    }
    if motion is not None:
        source_info[MOTION_KEY] = motion.summary()
    elif motion_error is not None:
        source_info[MOTION_KEY] = {"error": motion_error}

    def fill(staging: Path) -> list[CaptureFile]:
        original = copy_into(staging, video, set())
        original.metadata[VIDEO_KEY] = asdict(info)
        work = staging / CANDIDATES_DIR
        work.mkdir()
        argv = ffmpeg_backend.extract_frames_argv(
            ffmpeg, staging / original.name, work, plan.candidate_rate
        )
        result = run_process(
            argv,
            log_path=work / "ffmpeg.log",
            on_event=emit,
            parse_line=FFmpegProgress(info.duration_s),
            cancel=cancel,
        )
        if result.cancelled:
            raise VideoImportCancelled()
        if not result.ok:
            last = result.tail[-1] if result.tail else "no output"
            raise CaptureError(f"FFmpeg failed on {video.name} (exit {result.exit_code}): {last}")
        candidates = sorted(work.glob("c_*.jpg"))
        if not candidates:
            raise CaptureError(f"FFmpeg extracted no frames from {video.name}")

        scores = _score(candidates, emit, cancel)
        chosen = select_frames(scores, plan)
        if not chosen:
            raise CaptureError(f"none of the frames extracted from {video.name} can be read")
        files = [original]
        for number, index in enumerate(chosen, start=1):
            name = f"frame_{number:04d}.jpg"
            time_s = round(index / plan.candidate_rate, 3)
            candidates[index].rename(staging / name)
            frame = add_file(
                staging, name, original_name=f"{video.name} at {time_s:.2f} s", kind="image"
            )
            frame.metadata[VIDEO_KEY] = {
                "video": original.name,
                "time_s": time_s,
                "sharpness": scores[index],
            }
            if motion is not None and (entry := frame_motion(motion, time_s)):
                frame.metadata[MOTION_KEY] = entry
            files.append(frame)
        shutil.rmtree(work)
        source_info["candidates"] = len(candidates)
        source_info["frames"] = len(chosen)
        return files

    return assemble_bundle(project, fill, source="video", source_info=source_info, now=now)


def _score(
    candidates: Sequence[Path], emit: EventHandler, cancel: CancelToken | None
) -> list[float | None]:
    scores: list[float | None] = []
    workers = min(INSPECT_THREADS, cpu_threads())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for score in pool.map(measure_sharpness, candidates):
            if cancel is not None and cancel.cancelled:
                pool.shutdown(cancel_futures=True)
                raise VideoImportCancelled()
            scores.append(score)
            if len(scores) % 10 == 0 or len(scores) == len(candidates):
                emit(Progress("Choosing the sharpest frames", len(scores) / len(candidates)))
    return scores
