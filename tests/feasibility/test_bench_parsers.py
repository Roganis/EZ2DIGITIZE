# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

from ez2d_bench.colmap import parse_model_analyzer
from ez2d_bench.memprobe import Usage, amdgpu_device_dir, parse_macos_swapusage
from ez2d_bench.meshinfo import obj_counts, ply_counts
from ez2d_bench.sysinfo import parse_vulkaninfo_summary
from ez2d_bench.tools import option_in_help

ANALYZER_LOG = """\
I20261005 14:12:03.659842  9790 model.cc:429] Cameras: 1
I20261005 14:12:03.660020  9790 model.cc:430] Images: 32
I20261005 14:12:03.660037  9790 model.cc:431] Registered images: 30
I20261005 14:12:03.660043  9790 model.cc:433] Points: 5148
I20261005 14:12:03.660051  9790 model.cc:434] Observations: 25362
I20261005 14:12:03.660059  9790 model.cc:436] Mean track length: 4.926573
I20261005 14:12:03.660068  9790 model.cc:438] Mean observations per image: 792.562500
I20261005 14:12:03.660079  9790 model.cc:441] Mean reprojection error: 0.288496px
"""


def test_parse_model_analyzer() -> None:
    stats = parse_model_analyzer(ANALYZER_LOG)
    assert stats["registered_images"] == 30
    assert stats["points"] == 5148
    assert stats["mean_reprojection_error_px"] == 0.288496
    assert stats["mean_track_length"] == 4.926573


def test_option_in_help_matches_whole_names_only() -> None:
    text = "  --SiftExtraction.use_gpu arg (=1)\n  --SiftExtraction.max_image_size_x arg\n"
    assert option_in_help(text, "--SiftExtraction.use_gpu")
    assert not option_in_help(text, "--SiftExtraction.max_image_size")
    assert not option_in_help(text, "--FeatureExtraction.use_gpu")
    assert option_in_help("  -p [ --pointcloud-file ] arg", "--pointcloud-file")


def test_ply_and_obj_counts(tmp_path: Path) -> None:
    ply = tmp_path / "m.ply"
    ply.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 12\nproperty float x\n"
        b"element face 20\nproperty list uchar int vertex_indices\nend_header\n\x00\x01"
    )
    assert ply_counts(ply) == {"vertex": 12, "face": 20}
    obj = tmp_path / "m.obj"
    obj.write_text("mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nf 1/1 2/1 3/1\n")
    assert obj_counts(obj) == {"vertex": 3, "face": 1}


def test_macos_swapusage() -> None:
    text = "total = 2048.00M  used = 1536.50M  free = 511.50M  (encrypted)"
    assert parse_macos_swapusage(text) == 1536.5
    assert parse_macos_swapusage("total = 1.00G  used = 1.50G") == 1536.0
    assert parse_macos_swapusage("garbage") is None


def test_usage_peak_and_delta() -> None:
    base = Usage(vram_mb=1000.0, gtt_mb=None, swap_mb=50.0)
    peak = base.max(Usage(vram_mb=5000.0, gtt_mb=10.0, swap_mb=40.0))
    assert peak == Usage(vram_mb=5000.0, gtt_mb=10.0, swap_mb=50.0)
    assert peak.minus(base) == Usage(vram_mb=4000.0, gtt_mb=None, swap_mb=0.0)


def test_amdgpu_picks_card_with_most_vram(tmp_path: Path) -> None:
    for card, total in (("card0", 512 * 2**20), ("card1", 16 * 2**30)):
        device = tmp_path / card / "device"
        device.mkdir(parents=True)
        (device / "mem_info_vram_total").write_text(f"{total}\n")
    assert amdgpu_device_dir(tmp_path) == tmp_path / "card1" / "device"
    assert amdgpu_device_dir(tmp_path / "missing") is None


def test_parse_vulkaninfo_summary() -> None:
    text = """\
Devices:
========
GPU0:
        apiVersion         = 1.4.318
        deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
        deviceName         = AMD Radeon RX 7900 GRE (RADV NAVI31)
        driverName         = radv
        driverInfo         = Mesa 25.2.4-arch1.1
GPU1:
        deviceType         = PHYSICAL_DEVICE_TYPE_CPU
        deviceName         = llvmpipe (LLVM 20.1.8, 256 bits)
        driverName         = llvmpipe
"""
    gpus = parse_vulkaninfo_summary(text)
    assert gpus[0]["deviceName"] == "AMD Radeon RX 7900 GRE (RADV NAVI31)"
    assert gpus[0]["driverInfo"] == "Mesa 25.2.4-arch1.1"
    assert gpus[1]["driverName"] == "llvmpipe"
