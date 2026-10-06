# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the EZ2DIGITIZE AppImage (Linux x86_64) with the backends inside.

    uv run --group packaging python tools/packaging/build_appimage.py \\
        --backends build/backends/ez2d-backends-linux-x86_64.tar.gz

Steps: PyInstaller onedir of the app (no QtWebEngine yet), the backend
archive unpacked into `_internal/backends/` (where the app looks for
bundled tools), the Fortran/OpenMP runtimes the backends link bundled next
to them with an `$ORIGIN` RPATH, host libraries removed (see
HOST_LIBRARIES), then appimagetool. Build on the oldest distro to support:
the bundle needs that distro's glibc or newer (Ubuntu 22.04: 2.35).

`--smoke DIR` then checks the AppImage: the GUI starts, `check` finds the
bundled tools at the pinned versions, and a full CLI run on the photos in
DIR (with `DIR/../masks` if present) exports a GLB.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from ez2digitize import __version__  # noqa: E402

APPIMAGETOOL = (
    "https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage"
)
APPIMAGE_RUNTIME = (
    "https://github.com/AppImage/type2-runtime/releases/download/continuous/runtime-x86_64"
)
NAME = "ez2digitize"

# Libraries the host must provide; bundling older copies breaks fonts, input
# and GPU drivers on newer distros. See tools/spikes/packaging/README.md.
HOST_LIBRARIES = (
    "libfontconfig.so.1", "libfreetype.so.6", "libharfbuzz.so.0",
    "libstdc++.so.6", "libgcc_s.so.1",
    "libgbm.so.1", "libdrm.so.2", "libdrm_amdgpu.so.1", "libEGL.so.1", "libGL.so.1",
    "libGLX.so.0", "libGLdispatch.so.0", "libOpenGL.so.0", "libglapi.so.0", "libvulkan.so.1",
    "libX11.so.6", "libX11-xcb.so.1", "libxkbcommon.so.0", "libxkbcommon-x11.so.0",
    "libdbus-1.so.3", "libsystemd.so.0", "libselinux.so.1",
    "libgtk-3.so.0", "libgdk-3.so.0", "libgdk_pixbuf-2.0.so.0", "libatk-1.0.so.0",
    "libatk-bridge-2.0.so.0", "libatspi.so.0", "libcairo.so.2", "libcairo-gobject.so.2",
    "libpango-1.0.so.0", "libpangocairo-1.0.so.0", "libpangoft2-1.0.so.0",
)  # fmt: skip
HOST_PLUGINS = ("PySide6/Qt/plugins/platformthemes/libqgtk3.so",)
# Qt modules the app doesn't use (the viewer, with QtWebEngine, comes later).
EXCLUDED_MODULES = (
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebChannel",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtPdf", "PySide6.Qt3DCore",
    "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization",
)  # fmt: skip
# Runtimes the backends link that a desktop may lack; bundled next to them.
# The C/C++ runtime itself comes from the host (newer is compatible).
BACKEND_RUNTIMES = re.compile(r"^lib(gfortran|quadmath|gomp)\.so\.\d+$")


def run(cmd: list[str | Path], env: dict[str, str] | None = None) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=env)  # noqa: S603 - fixed argv


def output(cmd: list[str | Path], env: dict[str, str] | None = None) -> str:
    done = subprocess.run(  # noqa: S603 - fixed argv
        [str(c) for c in cmd], capture_output=True, text=True, env=env, check=False
    )
    return done.stdout + done.stderr


def download(url: str, dest: Path) -> None:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310 - fixed URL
            dest.write_bytes(response.read())
        dest.chmod(0o755)


def pyinstaller(out: Path) -> Path:
    cmd: list[str | Path] = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
        "--name", NAME,
        "--distpath", out / "dist", "--workpath", out / "work", "--specpath", out,
        "--paths", REPO / "src",
    ]  # fmt: skip
    for module in EXCLUDED_MODULES:
        cmd += ["--exclude-module", module]
    for text in ("LICENSE", "THIRD_PARTY_LICENSES"):  # shown under Help -> Licenses
        cmd += ["--add-data", f"{REPO / text}{os.pathsep}."]
    cmd.append(HERE / "entry.py")
    run(cmd)
    return out / "dist" / NAME


def add_backends(onedir: Path, archive: Path) -> dict[str, object]:
    """Unpack the backend archive into _internal/backends and make it self-contained."""
    target = onedir / "_internal" / "backends"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    binaries = sorted(p for p in (target / "bin").iterdir() if p.is_file())
    lib = target / "lib"
    lib.mkdir(exist_ok=True)
    bundled = set()
    for binary in binaries:
        for name, path in _resolved_libraries(binary):
            if path is not None and BACKEND_RUNTIMES.match(name) and name not in bundled:
                shutil.copy2(path, lib / name)
                bundled.add(name)
    # libgfortran needs libquadmath, which ldd lists as resolved for the binary.
    for library in sorted(lib.iterdir()):
        run(["patchelf", "--set-rpath", "$ORIGIN", library])
    for binary in binaries:
        run(["patchelf", "--set-rpath", "$ORIGIN/../lib", binary])
    missing = [
        f"{binary.name}: {name}"
        for binary in binaries
        for name, path in _resolved_libraries(binary)
        if path is None
    ]
    if missing:
        raise SystemExit(f"backend libraries not found: {', '.join(missing)}")
    info = json.loads((target / "BUILDINFO.json").read_text())
    return {"versions": {k: info.get(k) for k in ("colmap", "openmvs")}, "bundled": sorted(bundled)}


def add_brush(onedir: Path, release: Path) -> str:
    """Copy Brush (an unpacked release: brush_app, LICENSE) next to the other backends.

    It is one static binary needing only libc; it loads the Vulkan loader
    itself at run time, from the host like any Vulkan program.
    """
    target = onedir / "_internal" / "backends"
    shutil.copy2(release / "brush_app", target / "bin" / "brush_app")
    licenses = target / "licenses" / "brush"
    licenses.mkdir(parents=True, exist_ok=True)
    shutil.copy2(release / "LICENSE", licenses / "LICENSE")
    return output([target / "bin" / "brush_app", "--version"]).strip()


def _resolved_libraries(binary: Path) -> list[tuple[str, Path | None]]:
    libraries = []
    for line in output(["ldd", binary]).splitlines():
        match = re.match(r"\s*(\S+) => (\S+|not found)", line)
        if match:
            name, where = match.groups()
            libraries.append((name, None if where == "not found" else Path(where)))
    return libraries


def prune_host_libraries(onedir: Path) -> list[str]:
    internal = onedir / "_internal"
    removed = []
    for path in sorted(internal.rglob("*")):
        if path.is_relative_to(internal / "backends"):
            continue
        rel = path.relative_to(internal).as_posix()
        if path.name in HOST_LIBRARIES or rel in HOST_PLUGINS:
            path.unlink()
            removed.append(rel)
    return removed


def appimage(onedir: Path, out: Path) -> Path:
    appdir = out / f"{NAME}.AppDir"
    if appdir.exists():
        shutil.rmtree(appdir)
    shutil.copytree(onedir, appdir / "usr" / "lib" / NAME, symlinks=True)
    apprun = appdir / "AppRun"
    apprun.write_text(
        '#!/bin/sh\nHERE="$(dirname "$(readlink -f "$0")")"\n'
        f'exec "$HERE/usr/lib/{NAME}/{NAME}" "$@"\n'
    )
    apprun.chmod(0o755)
    (appdir / f"{NAME}.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=EZ2DIGITIZE\n"
        "Comment=Photos of small objects in, textured meshes out\n"
        f"Exec={NAME}\nIcon={NAME}\nCategories=Graphics;3DGraphics;\nTerminal=false\n"
    )
    shutil.copy2(HERE / f"{NAME}.svg", appdir / f"{NAME}.svg")
    tools = out / "appimage-tools"
    download(APPIMAGETOOL, tools / "appimagetool")
    download(APPIMAGE_RUNTIME, tools / "runtime-x86_64")
    target = out / f"EZ2DIGITIZE-{__version__}-x86_64.AppImage"
    env = dict(os.environ, ARCH="x86_64", APPIMAGE_EXTRACT_AND_RUN="1")
    run(
        [tools / "appimagetool", "--runtime-file", tools / "runtime-x86_64", appdir, target],
        env=env,
    )
    return target


def smoke(image: Path, photos: Path, *, brush: bool = False) -> dict[str, object]:
    """Run the AppImage like a user would; raise SystemExit on any failure."""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    for var in ("EZ2D_COLMAP", "EZ2D_OPENMVS_DIR"):
        env.pop(var, None)  # must find the bundled tools, not these
    if not Path("/dev/fuse").exists():
        env["APPIMAGE_EXTRACT_AND_RUN"] = "1"
    results: dict[str, object] = {}

    gui = output([image, "--self-test"], env=env)
    if '"self-test": "gui"' not in gui or '"heif": true' not in gui:
        raise SystemExit(f"GUI self-test failed (or no HEIC decoding):\n{gui}")
    check = output([image, "check"], env=env)
    print(check)
    expected = 3 if brush else 2
    if check.count("(bundled)") != expected or "not the tested" in check:
        raise SystemExit(f"check did not find the {expected} bundled tools at pinned versions")
    results["check"] = check.strip().splitlines()

    with tempfile.TemporaryDirectory(prefix="ez2d-smoke-") as scratch:
        project = Path(scratch) / "smoke"
        imp: list[str | Path] = [image, "import", project, photos]
        masks = photos.parent / "masks"
        if masks.is_dir():
            imp += ["--masks", masks]
        new: list[str | Path] = [image, "new", project]
        for cmd in (new, imp):
            print(output(cmd, env=env))
        # Automatic masks: downloads the model through the app (HTTPS from the
        # frozen Python), then runs ONNX Runtime in the masking worker. The
        # imported masks stay; the run below uses those.
        start = time.monotonic()
        masked = output([image, "masks", project], env=dict(env, EZ2D_MODELS_DIR=scratch))
        print(masked)
        if "new masks" not in masked or not list((project / "stages").glob("masks-*/report.json")):
            raise SystemExit("automatic masks were not made")
        results["masks_s"] = round(time.monotonic() - start, 1)
        start = time.monotonic()
        log = output([image, "run", project, "--level", "2", "--export", "glb"], env=env)
        print(log)
        glbs = list((project / "exports").rglob("*.glb"))
        if not glbs:
            raise SystemExit("the pipeline run produced no GLB")
        results["pipeline_s"] = round(time.monotonic() - start, 1)
        results["glb_bytes"] = glbs[0].stat().st_size
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backends", type=Path, required=True, help="ez2d-backends-*.tar.gz")
    parser.add_argument(
        "--brush", type=Path, metavar="DIR", help="unpacked Brush release (brush_app, LICENSE)"
    )
    parser.add_argument("--out", type=Path, default=REPO / "build" / "appimage")
    parser.add_argument("--smoke", type=Path, metavar="PHOTOS", help="test with these photos")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        raise SystemExit("the AppImage is built on Linux")
    if shutil.which("patchelf") is None:
        raise SystemExit("patchelf is needed (apt install patchelf / pacman -S patchelf)")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    onedir = pyinstaller(out)
    backends = add_backends(onedir, args.backends.resolve())
    if args.brush:
        backends["brush"] = add_brush(onedir, args.brush.resolve())
    removed = prune_host_libraries(onedir)
    print(f"removed {len(removed)} host libraries")
    image = appimage(onedir, out)
    report: dict[str, object] = {
        "appimage": image.name,
        "size_mb": round(image.stat().st_size / 2**20, 1),
        "backends": backends,
        "host_libraries_removed": len(removed),
    }
    if args.smoke:
        report["smoke"] = smoke(image, args.smoke.resolve(), brush=args.brush is not None)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
