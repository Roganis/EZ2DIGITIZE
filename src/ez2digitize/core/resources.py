# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""How much memory and CPU this machine can give a reconstruction step.

Backends default to one thread per CPU core, and some need about the same
memory per thread whatever the machine, so on a many-core desktop they can
ask for more memory than there is. Callers use these numbers to cap threads.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

GIB = 1024**3


def cpu_threads() -> int:
    """CPU threads this process may use."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def available_memory() -> int:
    """Bytes of memory a new process can use without pushing the system into swap.

    Linux: MemAvailable, further capped by the cgroup's memory limit (systemd
    user slices, containers). Elsewhere: half the physical memory, a
    conservative stand-in (macOS doesn't report "available" simply).
    """
    if sys.platform.startswith("linux"):
        available = _meminfo_available()
        limit = _cgroup_headroom()
        candidates = [v for v in (available, limit) if v is not None]
        if candidates:
            return min(candidates)
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 4 * GIB
    return total // 2


def _meminfo_available() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cgroup_headroom() -> int | None:
    """Memory left under this process's cgroup v2 limit, if it has one."""
    try:
        group = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        folder = Path("/sys/fs/cgroup") / group.lstrip("/")
        limit = (folder / "memory.max").read_text().strip()
        if limit == "max":
            return None
        return max(0, int(limit) - int((folder / "memory.current").read_text()))
    except (OSError, ValueError, IndexError):
        return None
