# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import http.server
import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from scripts import FAKE_BRUSH, python_script

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
    # Without pytest-qt (`-p no:pytest-qt`: the Backends workflow, on runners
    # without Qt's system libraries) nothing makes one, and Qt mustn't load.
    if not config.pluginmanager.has_plugin("pytest-qt"):
        return
    from ez2digitize.ui import viewer

    viewer.prepare()


# Behaviour is steered through environment variables the fakes read:
# FAKE_FAIL=<command> makes that command exit 1, FAKE_MODELS="30,2" sets the
# registered images per model ("none": no model; Windows drops empty variables),
# FAKE_SLEEP=<command> makes it hang.
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
def write_model(model, names):
    # One SIMPLE_RADIAL camera, 8x6 pixels; images registered in order.
    cameras = struct.pack("<QIiQQ4d", 1, 1, 2, 8, 6, 7.0, 4.0, 3.0, 0.01)
    (model / "cameras.bin").write_bytes(cameras)
    images = struct.pack("<Q", len(names))
    for image_id, name in enumerate(names, 1):
        images += struct.pack("<I7dI", image_id, 1, 0, 0, 0, 0, 0, 0, 1)
        images += name.encode() + b"\\0" + struct.pack("<Q", 0)
    (model / "images.bin").write_bytes(images)

def database_names():
    import sqlite3
    with sqlite3.connect(opt("--database_path")) as db:
        return [n for (n,) in db.execute("SELECT name FROM images ORDER BY image_id")]

if cmd == "feature_extractor":
    # A database with COLMAP 4.2.1's tables: a SIMPLE_RADIAL camera (8x6) and
    # its rig per capture folder, a frame per image.
    import sqlite3
    names = Path(opt("--image_list_path")).read_text().split()
    with sqlite3.connect(opt("--database_path")) as db:
        db.executescript(
            "CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER,"
            " height INTEGER, params BLOB, prior_focal_length INTEGER);"
            "CREATE TABLE rigs (rig_id INTEGER PRIMARY KEY, ref_sensor_id INTEGER,"
            " ref_sensor_type INTEGER);"
            "CREATE TABLE rig_sensors (rig_id INTEGER, sensor_id INTEGER, sensor_type INTEGER,"
            " sensor_from_rig BLOB);"
            "CREATE TABLE frames (frame_id INTEGER PRIMARY KEY, rig_id INTEGER);"
            "CREATE TABLE frame_data (frame_id INTEGER, data_id INTEGER, sensor_id INTEGER,"
            " sensor_type INTEGER);"
            "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER);"
        )
        cameras = {}
        for image_id, name in enumerate(names, 1):
            folder = name.split("/")[0]
            if folder not in cameras:
                cameras[folder] = len(cameras) + 1
                params = struct.pack("<4d", 9.6, 4.0, 3.0, 0.0)
                row = (cameras[folder], params)
                db.execute("INSERT INTO cameras VALUES (?, 2, 8, 6, ?, 0)", row)
                db.execute("INSERT INTO rigs VALUES (?, ?, 0)", (cameras[folder], cameras[folder]))
            camera = cameras[folder]
            db.execute("INSERT INTO images VALUES (?, ?, ?)", (image_id, name, camera))
            db.execute("INSERT INTO frames VALUES (?, ?)", (image_id, camera))
            db.execute("INSERT INTO frame_data VALUES (?, ?, ?, 0)", (image_id, image_id, camera))
    print("Processed file [1/1]")
elif cmd in ("mapper", "global_mapper") and "--input_path" in args:
    # Continuing from a model: every image of the database placed.
    write_model(Path(opt("--output_path")), database_names())
elif cmd in ("mapper", "global_mapper"):
    names = database_names()
    for i, n in enumerate(os.environ.get("FAKE_MODELS", "3").replace("none", "").split(",")):
        if not n:
            continue
        model = Path(opt("--output_path")) / str(i)
        model.mkdir(parents=True)
        write_model(model, names[: int(n)])
elif cmd in ("point_triangulator", "image_filterer"):
    for model_file in Path(opt("--input_path")).glob("*.bin"):
        (Path(opt("--output_path")) / model_file.name).write_bytes(model_file.read_bytes())
elif cmd == "matches_importer":
    pairs = Path(opt("--match_list_path")).read_text().splitlines()
    print(f"matching {len(pairs)} pairs", flush=True)
elif cmd == "image_undistorter":
    out = Path(opt("--output_path"))
    (out / "images").mkdir()
    (out / "sparse").mkdir()
    for model_file in Path(opt("--input_path")).glob("*.bin"):  # the model, as is
        (out / "sparse" / model_file.name).write_bytes(model_file.read_bytes())
elif cmd == "poisson_mesher":
    # A coloured tetrahedron, as PoissonRecon writes it (with its density value).
    header = ("ply\\nformat binary_little_endian 1.0\\nelement vertex 4\\n"
              + "".join(f"property float {n}\\n" for n in ("x", "y", "z", "value"))
              + "".join(f"property uchar {n}\\n" for n in ("red", "green", "blue"))
              + "element face 4\\nproperty list uchar int vertex_indices\\nend_header\\n")
    body = b"".join(struct.pack("<4f3B", *v, 1.0, 200, 100, 50)
                    for v in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)))
    faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    body += b"".join(struct.pack("<B3i", 3, *f) for f in faces)
    Path(opt("--output_path")).write_bytes(header.encode() + body)
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
if tool == "DensifyPointCloud":
    # Three coloured points, binary like OpenMVS's dense cloud.
    import struct
    header = ("ply\\nformat binary_little_endian 1.0\\nelement vertex 3\\n"
              "property float x\\nproperty float y\\nproperty float z\\n"
              "property uchar red\\nproperty uchar green\\nproperty uchar blue\\n"
              "end_header\\n")
    body = b"".join(struct.pack("<3f3B", *v, 200, 100, 50)
                    for v in ((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    out.with_suffix(".ply").write_bytes(header.encode() + body)
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
    # Never the user's model cache or plugins.
    monkeypatch.setenv("EZ2D_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("EZ2D_PLUGINS_DIR", str(tmp_path / "plugins"))


@pytest.fixture
def server(tmp_path: Path) -> Iterator[str]:
    """A local web server for tmp_path/www (for download tests)."""
    root = tmp_path / "www"
    root.mkdir()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=str(root), **kwargs)  # type: ignore[arg-type]

        def log_message(self, *args: object) -> None:
            pass

    class Server(http.server.ThreadingHTTPServer):
        def handle_error(self, request: object, client_address: object) -> None:
            pass  # a cancelled download hangs up mid-transfer

    httpd = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture(autouse=True)
def _no_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never the network: COLMAP's files (vocabulary trees, learned feature models)
    can't be fetched unless a test says so."""
    from ez2digitize.backends import colmap
    from ez2digitize.core.download import DownloadError

    def offline(*_args: object, **_kwargs: object) -> Path:
        raise DownloadError("no network in the tests")

    monkeypatch.setattr(colmap, "fetch_pinned", offline)


@pytest.fixture(autouse=True)
def _ample_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same machine everywhere: 4 threads and 64 GiB free, so no step is capped.

    Tests about the caps set their own values.
    """
    import ez2digitize.pipeline

    monkeypatch.setattr(ez2digitize.pipeline, "cpu_threads", lambda: 4)
    monkeypatch.setattr(ez2digitize.pipeline, "available_memory", lambda: 64 * 1024**3)
