# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""System-wide memory readings that per-process rusage can't see.

- AMD GPUs on Linux: VRAM and GTT (system memory mapped for the GPU) from the
  amdgpu sysfs files of the card with the most VRAM. GTT growing during
  training means the GPU ran out of VRAM and spilled.
- Swap on Linux and macOS. On an 8 GB M1, swap growth is the first sign the
  workload doesn't fit.

All readings are system-wide, so close other GPU-heavy apps while measuring.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DRM_ROOT = Path("/sys/class/drm")


@dataclass(frozen=True)
class Usage:
    vram_mb: float | None = None
    gtt_mb: float | None = None
    swap_mb: float | None = None

    def max(self, other: Usage) -> Usage:
        return Usage(
            _opt_max(self.vram_mb, other.vram_mb),
            _opt_max(self.gtt_mb, other.gtt_mb),
            _opt_max(self.swap_mb, other.swap_mb),
        )

    def minus(self, base: Usage) -> Usage:
        return Usage(
            _opt_sub(self.vram_mb, base.vram_mb),
            _opt_sub(self.gtt_mb, base.gtt_mb),
            _opt_sub(self.swap_mb, base.swap_mb),
        )


def _opt_max(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _opt_sub(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(max(a - b, 0.0), 1)


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def amdgpu_device_dir(drm_root: Path = DRM_ROOT) -> Path | None:
    """The amdgpu device with the largest VRAM (skips small iGPU carve-outs)."""
    best: tuple[int, Path] | None = None
    for total_file in drm_root.glob("card*/device/mem_info_vram_total"):
        total = _read_int(total_file)
        if total is not None and (best is None or total > best[0]):
            best = (total, total_file.parent)
    return best[1] if best else None


_DEVICE = amdgpu_device_dir() if sys.platform.startswith("linux") else None


def _linux_swap_mb() -> float | None:
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return None
    values = dict(re.findall(r"^(SwapTotal|SwapFree):\s+(\d+) kB", text, re.MULTILINE))
    if len(values) != 2:
        return None
    return (int(values["SwapTotal"]) - int(values["SwapFree"])) / 1024


def parse_macos_swapusage(text: str) -> float | None:
    """Parse `sysctl -n vm.swapusage`, e.g. 'total = 2048.00M  used = 512.25M ...'."""
    match = re.search(r"used\s*=\s*([\d.]+)([KMG])", text)
    if not match:
        return None
    value = float(match.group(1))
    return value * {"K": 1 / 1024, "M": 1.0, "G": 1024.0}[match.group(2)]


def _macos_swap_mb() -> float | None:
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_macos_swapusage(out)


def sample() -> Usage:
    vram = gtt = None
    if _DEVICE is not None:
        v = _read_int(_DEVICE / "mem_info_vram_used")
        g = _read_int(_DEVICE / "mem_info_gtt_used")
        vram = v / 2**20 if v is not None else None
        gtt = g / 2**20 if g is not None else None
    if sys.platform == "darwin":
        swap = _macos_swap_mb()
    elif sys.platform.startswith("linux"):
        swap = _linux_swap_mb()
    else:
        swap = None
    return Usage(vram, gtt, swap)
