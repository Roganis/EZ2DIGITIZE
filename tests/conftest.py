# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import os
import sys
from pathlib import Path

import pytest
from scripts import python_script

from ez2digitize.backends.brush import Brush
from ez2digitize.backends.colmap import Colmap
from ez2digitize.backends.ffmpeg import FFmpeg
from ez2digitize.backends.openmvs import TOOLS, OpenMVS
from ez2digitize.pipeline import Tools

# The Phase 1 benchmark harness (tools/feasibility) is POSIX only.
collect_ignore_glob = ["feasibility/*"] if sys.platform == "win32" else []

# Run Qt without a display (CI, SSH sessions). Must be set before Qt loads.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Chromium refuses WebGL on software renderers (CI has no GPU); only for tests.
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--ignore-gpu-blocklist")


def pytest_configure(config: pytest.Config) -> None:
    # The viewer's scheme must be registered before pytest-qt makes the QApplication.
    from ez2digitize.ui import viewer

    viewer.prepare()


# Behaviour is steered through environment variables the fakes read:
# FAKE_FAIL=<command> makes that command exit 1, FAKE_MODELS="30,2" sets the
# registered images per model, FAKE_SLEEP=<command> makes it hang.
FAKE_COLMAP = """
import os, struct, sys, time
from pathlib import Path
cmd, args = sys.argv[1], sys.argv[2:]
opt = lambda name: args[args.index(name) + 1]
print(f"fake colmap {cmd}", flush=True)
if os.environ.get("FAKE_SLEEP") == cmd:
    time.sleep(60)
if os.environ.get("FAKE_FAIL") == cmd:
    print("something went wrong", flush=True)
    sys.exit(1)
if cmd == "feature_extractor":
    # The "database" is the image list, so the mapper knows the names.
    Path(opt("--database_path")).write_text(Path(opt("--image_list_path")).read_text())
    print("Processed file [1/1]")
elif cmd in ("mapper", "global_mapper"):
    names = Path(opt("--database_path")).read_text().split()
    for i, n in enumerate(os.environ.get("FAKE_MODELS", "3").split(",")):
        if not n:
            continue
        model = Path(opt("--output_path")) / str(i)
        model.mkdir(parents=True)
        # One SIMPLE_RADIAL camera, 8x6 pixels; images registered in order.
        cameras = struct.pack("<QIiQQ4d", 1, 1, 2, 8, 6, 7.0, 4.0, 3.0, 0.01)
        (model / "cameras.bin").write_bytes(cameras)
        images = struct.pack("<Q", int(n))
        for image_id, name in enumerate(names[: int(n)], 1):
            images += struct.pack("<I7dI", image_id, 1, 0, 0, 0, 0, 0, 0, 1)
            images += name.encode() + b"\\0" + struct.pack("<Q", 0)
        (model / "images.bin").write_bytes(images)
elif cmd == "image_undistorter":
    out = Path(opt("--output_path"))
    (out / "images").mkdir()
    (out / "sparse").mkdir()
elif cmd == "image_undistorter_standalone":
    src, out = Path(opt("--image_path")), Path(opt("--output_path"))
    for line in Path(opt("--input_file")).read_text().splitlines():
        name = line.split()[0]
        (out / name).write_bytes((src / name).read_bytes())
"""

FAKE_OPENMVS = """
import os, sys
from pathlib import Path
tool = Path(sys.argv[0]).stem  # "DensifyPointCloud.cmd" on Windows
args = sys.argv[1:]
print(f"fake {tool}", flush=True)
if os.environ.get("FAKE_FAIL") == tool:
    sys.exit(1)
out = Path(args[args.index("-o") + 1])
out.write_text("mvs")
out.with_suffix(".ply").write_text("ply")
if tool in ("ReconstructMesh", "RefineMesh"):
    # A mesh header saying 1000 faces (what the texture step reads to simplify).
    out.with_suffix(".ply").write_text(
        "ply\\nformat binary_little_endian 1.0\\nelement vertex 500\\n"
        "property float x\\nelement face 1000\\n"
        "property list uchar uint vertex_indices\\nend_header\\n"
    )
if tool == "TextureMesh":
    # A one-triangle textured PLY in OpenMVS's layout, and its texture.
    import struct
    header = (
        "ply\\nformat binary_little_endian 1.0\\ncomment TextureFile scene_textured0.png\\n"
        "element vertex 3\\nproperty float x\\nproperty float y\\nproperty float z\\n"
        "element face 1\\nproperty list uchar uint vertex_indices\\n"
        "property list uchar float texcoord\\nend_header\\n"
    )
    body = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    body += struct.pack("<B3IB6f", 3, 0, 1, 2, 6, 0, 0, 1, 0, 0, 1)
    out.with_suffix(".ply").write_bytes(header.encode() + body)
    (out.parent / "scene_textured0.png").write_bytes(b"\\x89PNG fake")
"""


# A 10 s, 30 fps, 640x480 video. The extracted frames are noise; in each run
# of four, the third is sharp and the others blurred (like motion blur).
FAKE_FFPROBE = """
import json
print(json.dumps({"streams": [{"codec_name": "h264", "width": 640, "height": 480,
    "avg_frame_rate": "30/1", "duration": "10.0"}], "format": {"duration": "10.0"}}))
"""
FAKE_FFMPEG = """
import os, sys, time
from pathlib import Path
from PIL import Image, ImageFilter
args = sys.argv[1:]
if args == ["-version"]:
    print("ffmpeg version 7.1 Copyright (c) 2000-2024 the FFmpeg developers")
    sys.exit(0)
if os.environ.get("FAKE_SLEEP") == "ffmpeg":
    time.sleep(60)
if os.environ.get("FAKE_FAIL") == "ffmpeg":
    print("Invalid data found when processing input", flush=True)
    sys.exit(1)
rate = float(args[args.index("-vf") + 1].removeprefix("fps="))
pattern = Path(args[-1])
noise = Image.effect_noise((640, 480), 80).convert("RGB")
for i in range(round(10.0 * rate)):
    frame = noise if i % 4 == 2 else noise.filter(ImageFilter.GaussianBlur(3))
    frame.save(pattern.parent / (pattern.name % (i + 1)), quality=90)
    print(f"out_time_us={int(i / rate * 1e6)}", flush=True)
print("progress=end", flush=True)
"""


# Brush: checks the dataset layout, prints a progress bar like the real one
# (with colour codes) and writes the splat file.
FAKE_BRUSH = """
import os, sys
from pathlib import Path
args = sys.argv[1:]
if args == ["--version"]:
    print("brush-cli 0.3.0")
    sys.exit(0)
if os.environ.get("FAKE_FAIL") == "brush":
    sys.exit(1)
opt = lambda name: args[args.index(name) + 1]
dataset = Path(args[0])
assert (dataset / "sparse" / "0").is_dir() and (dataset / "images").is_dir(), "bad dataset"
steps = int(opt("--total-steps"))
masks = sorted((dataset / "images" / "masks").glob("*"))
assert all(m.resolve().is_file() for m in masks), "dangling mask link"
print(f"masks: {len(masks)}", flush=True)
print("\\x1b[34mi\\x1b[0m Completed loading", flush=True)
for done in (steps // 2, steps):
    print(f"[1s] \\x1b[36m###\\x1b[0m   {done}/{steps}   Steps (9/s, 0s remaining)", flush=True)
(Path(opt("--export-path")) / opt("--export-name")).write_text("ply splats")
"""


# Stands in for mask_worker: writes the masks and report.json the real one
# would. FAKE_COVERAGE="a.jpg=0" makes a photo's mask empty; FAKE_FAIL=1
# makes it exit 1.
FAKE_MASK_WORKER = """
import json, os, sys
from pathlib import Path
from PIL import Image
args = sys.argv[1:]
jobs_file = Path(args[args.index("--jobs") + 1])
jobs = json.loads(jobs_file.read_text())["jobs"]
coverage = dict(c.split("=") for c in os.environ.get("FAKE_COVERAGE", "").split(",") if c)
print("model fake loaded, %d images" % len(jobs), flush=True)
if os.environ.get("FAKE_FAIL"):
    sys.exit(1)
report = {}
for n, job in enumerate(jobs, 1):
    print("mask %d/%d %s" % (n, len(jobs), Path(job["image"]).name), flush=True)
    share = float(coverage.get(job["key"], "0.25"))
    out = Path(job["mask"])
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (4, 4), 255 if share > 0 else 0).save(out)
    report[job["key"]] = {"coverage": share, "uncertain": 0.01}
(jobs_file.parent / "report.json").write_text(json.dumps(report))
"""


def _script(path: Path, body: str) -> Path:
    return python_script(path, body)


@pytest.fixture
def fake_tools(tmp_path: Path) -> Tools:
    """COLMAP and OpenMVS stand-ins that write the files the real tools do."""
    exe = _script(tmp_path / "fake" / "colmap", FAKE_COLMAP)
    for tool in TOOLS:
        _script(tmp_path / "fake" / "mvs" / tool, FAKE_OPENMVS)
    return Tools(Colmap(exe, "4.2.1"), OpenMVS(tmp_path / "fake" / "mvs", "2.4.0"))


@pytest.fixture
def fake_ffmpeg(tmp_path: Path) -> FFmpeg:
    exe = _script(tmp_path / "fake" / "ffmpeg" / "ffmpeg", FAKE_FFMPEG)
    probe = _script(exe.with_name("ffprobe"), FAKE_FFPROBE)
    return FFmpeg(exe, probe, "7.1")


@pytest.fixture
def fake_brush(tmp_path: Path) -> Brush:
    return Brush(_script(tmp_path / "fake" / "brush" / "brush_app", FAKE_BRUSH), "0.3.0")


@pytest.fixture
def fake_mask_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The masking worker's stand-in, and a model file where the app looks for one.

    Returns the model's path (a sparse file of the real model's size).
    """
    import ez2digitize.masks

    # What to run (on Windows a .cmd file, not one Python can run).
    script = _script(tmp_path / "fake" / "mask_worker", FAKE_MASK_WORKER)
    monkeypatch.setattr(ez2digitize.masks, "worker_argv", lambda: [str(script)])
    model = ez2digitize.masks.model_file()
    model.parent.mkdir(parents=True, exist_ok=True)
    with model.open("wb") as f:
        f.truncate(ez2digitize.masks.MODEL.size)
    return model


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("FAKE_FAIL", "FAKE_MODELS", "FAKE_SLEEP", "FAKE_COVERAGE"):
        monkeypatch.delenv(var, raising=False)
    # Never the user's model cache.
    monkeypatch.setenv("EZ2D_MODELS_DIR", str(tmp_path / "models"))


@pytest.fixture(autouse=True)
def _ample_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same machine everywhere: 4 threads and 64 GiB free, so no step is capped.

    Tests about the caps set their own values.
    """
    import ez2digitize.pipeline

    monkeypatch.setattr(ez2digitize.pipeline, "cpu_threads", lambda: 4)
    monkeypatch.setattr(ez2digitize.pipeline, "available_memory", lambda: 64 * 1024**3)
