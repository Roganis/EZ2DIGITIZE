# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import json
from pathlib import Path

from ez2digitize.core import hardware
from ez2digitize.core.hardware import Gpu

VULKAN_SUMMARY = """
Devices:
========
GPU0:
	apiVersion         = 1.4.305
	driverVersion      = 25.0.2
	vendorID           = 0x1002
	deviceID           = 0x744c
	deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	deviceName         = AMD Radeon RX 7900 GRE (RADV NAVI31)
	driverID           = DRIVER_ID_MESA_RADV
	driverName         = radv
	driverInfo         = Mesa 25.0.2-arch1.1
GPU1:
	vendorID           = 0x10005
	deviceType         = PHYSICAL_DEVICE_TYPE_CPU
	deviceName         = llvmpipe (LLVM 19.1.7, 256 bits)
	driverName         = llvmpipe
	driverInfo         = Mesa 25.0.2-arch1.1 (LLVM 19.1.7)
"""


def test_parse_vulkan_summary() -> None:
    gre, cpu = hardware.parse_vulkan_summary(VULKAN_SUMMARY)
    assert gre == Gpu("amd", "AMD Radeon RX 7900 GRE (RADV NAVI31)", None, "Mesa 25.0.2-arch1.1")
    assert cpu.is_cpu and not gre.is_cpu
    assert hardware.parse_vulkan_summary("no devices") == []


def _card(drm: Path, name: str, vendor: str, vram: int | None = None) -> None:
    device = drm / name / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text(f"{vendor}\n")
    (device / "device").write_text("0x744c\n")
    if vram is not None:
        (device / "mem_info_vram_total").write_text(f"{vram}\n")


def test_sysfs_and_merge(tmp_path: Path) -> None:
    _card(tmp_path, "card1", "0x1002", vram=16 * 1024**3)
    _card(tmp_path, "card0", "0x8086")
    (tmp_path / "card1-DP-1").mkdir()
    found = hardware._sysfs(tmp_path)
    assert [(g.vendor, g.vram_bytes) for g in found] == [("intel", None), ("amd", 16 * 1024**3)]
    merged = hardware._merge(hardware.parse_vulkan_summary(VULKAN_SUMMARY), found)
    assert merged[0].vram_bytes == 16 * 1024**3 and merged[0].name.startswith("AMD Radeon")
    assert hardware.describe(merged[0]) == (
        "AMD Radeon RX 7900 GRE (RADV NAVI31), 16 GB, Mesa 25.0.2-arch1.1"
    )


def test_parse_system_profiler() -> None:
    m1 = {"_name": "Apple M1", "sppci_model": "Apple M1", "spdisplays_vendor": "sppci_vendor_Apple",
          "spdisplays_mtlgpufamilysupport": "spdisplays_metal3"}  # fmt: skip
    radeon = {"sppci_model": "AMD Radeon Pro 5500M", "spdisplays_vendor": "sppci_vendor_amd",
              "spdisplays_vram": "8 GB"}  # fmt: skip
    gpus = hardware.parse_system_profiler(json.dumps({"SPDisplaysDataType": [m1, radeon]}))
    assert gpus[0] == Gpu("apple", "Apple M1", None, "spdisplays_metal3", shared_memory=True)
    assert (gpus[1].vendor, gpus[1].vram_bytes) == ("amd", 8 * 1024**3)
    assert hardware.parse_system_profiler("garbage") == []


def test_detect_never_fails() -> None:
    for gpu in hardware.detect_gpus():
        assert gpu.name
