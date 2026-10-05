# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Describe the machine a benchmark ran on."""

from __future__ import annotations

import contextlib
import os
import platform
import re
import shutil
import socket
import sys
from pathlib import Path
from typing import Any

from ez2d_bench import memprobe
from ez2d_bench.runner import capture
from ez2d_bench.tools import Toolbox


def collect(toolbox: Toolbox) -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "os": _os_name(),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "cpu_threads": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpus": _gpus(),
        "tools": toolbox.versions(),
    }
    device = memprobe.amdgpu_device_dir()
    if device is not None:
        total = device / "mem_info_vram_total"
        with contextlib.suppress(OSError, ValueError):
            info["amdgpu_vram_gb"] = round(int(total.read_text()) / 2**30, 1)
    return info


def _os_name() -> str:
    if sys.platform == "darwin":
        return "macOS " + platform.mac_ver()[0]
    os_release = Path("/etc/os-release")
    if os_release.is_file():
        match = re.search(r'^PRETTY_NAME="?([^"\n]+)', os_release.read_text(), re.MULTILINE)
        if match:
            return match.group(1)
    return platform.platform()


def _cpu_model() -> str:
    if sys.platform == "darwin":
        return capture(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return platform.processor()
    match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else platform.processor()


def _ram_gb() -> float | None:
    if sys.platform == "darwin":
        out = capture(["/usr/sbin/sysctl", "-n", "hw.memsize"]).strip()
        return round(int(out) / 2**30, 1) if out.isdigit() else None
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return None
    match = re.search(r"^MemTotal:\s+(\d+) kB", text, re.MULTILINE)
    return round(int(match.group(1)) / 2**20, 1) if match else None


def _gpus() -> list[dict[str, str]]:
    if sys.platform == "darwin":
        text = capture(["/usr/sbin/system_profiler", "SPDisplaysDataType"])
        return [{"name": m} for m in re.findall(r"Chipset Model:\s*(.+)", text)]
    vulkaninfo = shutil.which("vulkaninfo")
    if vulkaninfo:
        return parse_vulkaninfo_summary(capture([vulkaninfo, "--summary"]))
    return []


def parse_vulkaninfo_summary(text: str) -> list[dict[str, str]]:
    """Extract device name, driver and driver info (Mesa version) per GPU."""
    gpus: list[dict[str, str]] = []
    for block in re.split(r"^GPU\d+:", text, flags=re.MULTILINE)[1:]:
        gpu: dict[str, str] = {}
        for key in ("deviceName", "deviceType", "driverName", "driverInfo"):
            match = re.search(rf"^\s*{key}\s*=\s*(.+)$", block, re.MULTILINE)
            if match:
                gpu[key] = match.group(1).strip()
        if gpu:
            gpus.append(gpu)
    return gpus
