# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract the sharpest frames from a video.

Uniform sampling picks whatever frame lands on the timestamp, often a
motion-blurred one. Instead, extract `window` times more candidates than
needed and keep the sharpest of each consecutive group.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from ez2d_bench.runner import capture


def sharpness(path: Path, width: int = 800) -> float:
    """Variance of the Laplacian on a downscaled grayscale copy (higher = sharper)."""
    with Image.open(path) as img:
        gray = img.convert("L")
        if gray.width > width:
            gray = gray.resize((width, round(gray.height * width / gray.width)))
        a = np.asarray(gray, dtype=np.float64)
    lap = a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1]
    return float(lap.var())


def pick_sharpest(scores: list[float], window: int) -> list[int]:
    """Index of the best score in each consecutive group of `window`."""
    picked = []
    for start in range(0, len(scores), window):
        group = scores[start : start + window]
        picked.append(start + int(np.argmax(group)))
    return picked


def video_duration_s(ffprobe: str, video: Path) -> float:
    out = capture([ffprobe, "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", video])  # fmt: skip
    try:
        return float(out.strip().splitlines()[0])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"ffprobe could not read the duration of {video}: {out!r}") from exc


def extract(video: Path, out_dir: Path, count: int, window: int = 3) -> dict[str, object]:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe must be on PATH")
    duration = video_duration_s(ffprobe, video)
    fps = count * window / duration
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ez2d-frames-") as tmp:
        tmp_dir = Path(tmp)
        log = capture([ffmpeg, "-v", "error", "-i", video, "-vf", f"fps={fps:.4f}",
                       "-q:v", "2", tmp_dir / "%06d.jpg"], timeout_s=3600)  # fmt: skip
        candidates = sorted(tmp_dir.glob("*.jpg"))
        if not candidates:
            raise RuntimeError(f"ffmpeg produced no frames: {log}")
        scores = [sharpness(p) for p in candidates]
        picked = pick_sharpest(scores, window)
        for n, index in enumerate(picked):
            shutil.move(candidates[index], out_dir / f"frame_{n:05d}.jpg")

    picked_scores = [scores[i] for i in picked]
    info: dict[str, object] = {
        "video": str(video),
        "duration_s": round(duration, 2),
        "candidates": len(candidates),
        "frames": len(picked),
        "window": window,
        "sharpness_median_all": round(float(np.median(scores)), 1),
        "sharpness_median_kept": round(float(np.median(picked_scores)), 1),
        "sharpness_min_kept": round(min(picked_scores), 1),
    }
    (out_dir.parent / f"{out_dir.name}_frames_info.json").write_text(json.dumps(info, indent=2))
    return info
