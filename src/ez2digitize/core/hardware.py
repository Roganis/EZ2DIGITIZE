# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which GPUs this machine has: vendor, name, memory, driver.

The splat backend (Brush) runs on the GPU through Vulkan or Metal; the GPU
and its driver (the Mesa version on Linux) decide whether it works and how
large a scene fits. Sources, all optional and read-only:

- `vulkaninfo --summary` (Linux, Windows): every Vulkan device with its
  type and driver; CPU devices (llvmpipe) are reported as such.
- Linux sysfs (`/sys/class/drm/card*/device`): PCI vendor and device ids,
  and VRAM for amdgpu (`mem_info_vram_total`).
- `system_profiler SPDisplaysDataType -json` (macOS); Apple GPUs share the
  system memory.

Missing tools or files only mean less information, never an error.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ez2digitize.core.runner import ProcessStartError, run_quick

Vendor = Literal["amd", "nvidia", "intel", "apple", "cpu", "other"]
PCI_VENDORS: dict[int, Vendor] = {0x1002: "amd", 0x10DE: "nvidia", 0x8086: "intel"}
DRM = Path("/sys/class/drm")


@dataclass(frozen=True)
class Gpu:
    vendor: Vendor
    name: str
    # Dedicated video memory; None if unknown or shared with the system.
    vram_bytes: int | None = None
    driver: str | None = None
    # Integrated GPU or Apple silicon, sharing the system memory.
    shared_memory: bool = False

    @property
    def is_cpu(self) -> bool:
        """A software renderer (llvmpipe): works, but far too slow for training."""
        return self.vendor == "cpu"


def detect_gpus() -> list[Gpu]:
    """Every GPU found, discrete ones first."""
    if sys.platform == "darwin":
        gpus = _macos()
    else:
        gpus = _merge(_vulkan(), _sysfs())
    return sorted(gpus, key=lambda g: (g.is_cpu, g.shared_memory, -(g.vram_bytes or 0)))


# --- Vulkan ------------------------------------------------------------------------


def _vulkan() -> list[Gpu]:
    try:
        text = run_quick(["vulkaninfo", "--summary"], timeout_s=20)
    except ProcessStartError:
        return []
    return parse_vulkan_summary(text)


def parse_vulkan_summary(text: str) -> list[Gpu]:
    devices: list[dict[str, str]] = []
    for line in text.splitlines():
        if re.match(r"^GPU\d+:\s*$", line.strip()):
            devices.append({})
        elif devices and (m := re.match(r"^\s+(\w+)\s*=\s*(.*)$", line)):
            devices[-1][m.group(1)] = m.group(2).strip()
    gpus = []
    for device in devices:
        kind = device.get("deviceType", "")
        vendor_id = _int(device.get("vendorID"))
        vendor: Vendor = "cpu" if kind.endswith("CPU") else _pci_vendor(vendor_id)
        gpus.append(
            Gpu(
                vendor=vendor,
                name=device.get("deviceName", "unknown"),
                driver=device.get("driverInfo") or device.get("driverName"),
                shared_memory=kind.endswith("INTEGRATED_GPU"),
            )
        )
    return gpus


# --- Linux sysfs ---------------------------------------------------------------------


def _sysfs(drm: Path = DRM) -> list[Gpu]:
    gpus = []
    for card in sorted(drm.glob("card[0-9]*")):
        if "-" in card.name:  # connectors like card0-DP-1
            continue
        device = card / "device"
        vendor_id = _int(_read(device / "vendor"))
        if vendor_id is None:
            continue
        vram = _int(_read(device / "mem_info_vram_total"))
        driver_link = device / "driver"
        driver = driver_link.resolve().name if driver_link.exists() else None
        gpus.append(
            Gpu(
                vendor=_pci_vendor(vendor_id),
                name=f"PCI {vendor_id:04x}:{_int(_read(device / 'device')) or 0:04x}",
                vram_bytes=vram if vram else None,
                driver=driver,
            )
        )
    return gpus


def _merge(vulkan: list[Gpu], sysfs: list[Gpu]) -> list[Gpu]:
    """Vulkan's names and drivers, with VRAM from sysfs for the same vendor."""
    if not vulkan:
        return sysfs
    vram = {g.vendor: g.vram_bytes for g in sysfs if g.vram_bytes}
    merged = []
    for gpu in vulkan:
        size = vram.get(gpu.vendor) if not gpu.shared_memory else None
        merged.append(Gpu(gpu.vendor, gpu.name, size, gpu.driver, gpu.shared_memory))
    return merged


# --- macOS ---------------------------------------------------------------------------


def _macos() -> list[Gpu]:
    try:
        text = run_quick(["system_profiler", "SPDisplaysDataType", "-json"], timeout_s=30)
    except ProcessStartError:
        return []
    return parse_system_profiler(text)


def parse_system_profiler(text: str) -> list[Gpu]:
    try:
        entries = json.loads(text).get("SPDisplaysDataType", [])
    except (ValueError, AttributeError):
        return []
    gpus = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("sppci_model") or entry.get("_name") or "unknown")
        vendor_text = str(entry.get("spdisplays_vendor", "")).lower()
        apple = "apple" in vendor_text or name.startswith("Apple")
        vendor: Vendor = "apple" if apple else _vendor_from_text(vendor_text)
        gpus.append(
            Gpu(
                vendor=vendor,
                name=name,
                vram_bytes=None if apple else _size(entry.get("spdisplays_vram")),
                driver=str(entry["spdisplays_mtlgpufamilysupport"])
                if "spdisplays_mtlgpufamilysupport" in entry
                else None,
                shared_memory=apple,
            )
        )
    return gpus


def _vendor_from_text(text: str) -> Vendor:
    for key, vendor in (("amd", "amd"), ("ati", "amd"), ("nvidia", "nvidia"), ("intel", "intel")):
        if key in text:
            return vendor  # type: ignore[return-value]
    return "other"


# --- helpers ---------------------------------------------------------------------------


def _pci_vendor(vendor_id: int | None) -> Vendor:
    return PCI_VENDORS.get(vendor_id, "other") if vendor_id is not None else "other"


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii", errors="replace").strip()
    except OSError:
        return None


def _int(text: str | None) -> int | None:
    if not text:
        return None
    try:
        return int(text, 0)
    except ValueError:
        return None


def _size(text: Any) -> int | None:
    """ "8 GB" / "1536 MB" -> bytes."""
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([GM])B", str(text or ""), re.IGNORECASE)
    if match is None:
        return None
    factor = 1024**3 if match.group(2).upper() == "G" else 1024**2
    return int(float(match.group(1)) * factor)


def describe(gpu: Gpu) -> str:
    parts = [gpu.name]
    if gpu.vram_bytes:
        parts.append(f"{gpu.vram_bytes / 1024**3:.0f} GB")
    elif gpu.shared_memory:
        parts.append("shared memory")
    if gpu.driver:
        parts.append(gpu.driver)
    return ", ".join(parts)
