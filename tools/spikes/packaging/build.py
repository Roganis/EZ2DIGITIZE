# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Packaging spike: build both variants with PyInstaller and measure them.

    uv run python tools/spikes/viewer/fetch_vendor.py      # once, for the viewer variant
    uv run --group packaging python tools/spikes/packaging/build.py --brush /path/to/brush_app

Variants:
  core    the app window only (no QtWebEngine)
  viewer  the web viewer spike (QtWebEngine + three.js + Spark)

On Linux each variant is also wrapped into an AppImage (appimagetool 1.9.1,
downloaded on first use); on macOS PyInstaller produces a .app, which is
zipped. Each result is started three times with --self-test to measure
startup and check that QtWebEngine and the bundled backend work.
Output and report.md/report.json go to build/packaging-spike/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VIEWER = REPO / "tools" / "spikes" / "viewer"
APPIMAGETOOL = (
    "https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage"
)
APPIMAGE_RUNTIME = (
    "https://github.com/AppImage/type2-runtime/releases/download/continuous/runtime-x86_64"
)
WEBENGINE_MODULES = (
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebChannel",
    "PySide6.QtQml",
    "PySide6.QtQuick",
)


def run(cmd: list[str | Path], env: dict[str, str] | None = None) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=env)  # noqa: S603 - fixed argv


def tree_size(path: Path) -> tuple[int, int]:
    """Total bytes and file count, not following symlinks."""
    if path.is_file():
        return path.stat().st_size, 1
    total = count = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            p = Path(root) / name
            if not p.is_symlink():
                total += p.stat().st_size
                count += 1
    return total, count


def mb(n: int) -> float:
    return round(n / 2**20, 1)


def download(url: str, dest: Path) -> str:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310 - fixed URL
            dest.write_bytes(r.read())
        dest.chmod(0o755)
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def pyinstaller(variant: str, out: Path, brush: Path | None) -> Path:
    name = f"ez2digitize-{variant}"
    cmd: list[str | Path] = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
        "--name", name,
        "--distpath", out / "dist", "--workpath", out / "work", "--specpath", out,
        "--paths", REPO / "src", "--paths", HERE, "--paths", VIEWER,
    ]  # fmt: skip
    if variant == "core":
        for module in WEBENGINE_MODULES:
            cmd += ["--exclude-module", module]
    else:
        if not (VIEWER / "web" / "vendor").is_dir():
            raise SystemExit("run tools/spikes/viewer/fetch_vendor.py first")
        cmd += ["--add-data", f"{VIEWER / 'web'}{os.pathsep}viewer_web"]
    if brush:
        cmd += ["--add-binary", f"{brush}{os.pathsep}backends"]
    if sys.platform == "darwin":
        cmd.append("--windowed")  # produces name.app
    cmd.append(HERE / f"{variant}_main.py")
    run(cmd)
    dist = out / "dist"
    return dist / f"{name}.app" if sys.platform == "darwin" else dist / name


def executable(bundle: Path) -> Path:
    if bundle.suffix == ".app":
        return bundle / "Contents" / "MacOS" / bundle.stem
    if bundle.suffix == ".AppImage":
        return bundle
    return bundle / bundle.name


def appimage(onedir: Path, out: Path) -> Path:
    name = onedir.name
    appdir = out / f"{name}.AppDir"
    if appdir.exists():
        shutil.rmtree(appdir)
    shutil.copytree(onedir, appdir / "usr" / "lib" / name, symlinks=True)
    apprun = appdir / "AppRun"
    apprun.write_text(
        '#!/bin/sh\nHERE="$(dirname "$(readlink -f "$0")")"\n'
        f'exec "$HERE/usr/lib/{name}/{name}" "$@"\n'
    )
    apprun.chmod(0o755)
    (appdir / f"{name}.desktop").write_text(
        f"[Desktop Entry]\nType=Application\nName=EZ2DIGITIZE ({name})\n"
        f"Exec={name}\nIcon={name}\nCategories=Graphics;\n"
    )
    from PySide6.QtGui import QColor, QImage

    icon = QImage(256, 256, QImage.Format.Format_RGB32)
    icon.fill(QColor("#e8711a"))
    icon.save(str(appdir / f"{name}.png"))

    tools = out / "appimage-tools"
    download(APPIMAGETOOL, tools / "appimagetool")
    runtime_sha = download(APPIMAGE_RUNTIME, tools / "runtime-x86_64")
    print(f"AppImage runtime sha256 {runtime_sha}")
    target = out / f"{name}-x86_64.AppImage"
    env = dict(os.environ, ARCH="x86_64", APPIMAGE_EXTRACT_AND_RUN="1")
    run([tools / "appimagetool", "--runtime-file", tools / "runtime-x86_64", appdir, target],
        env=env)  # fmt: skip
    return target


def self_test(exe: Path, headless: bool, runs: int = 3) -> dict[str, Any]:
    env = dict(os.environ)
    if headless:
        env.update(QT_QPA_PLATFORM="offscreen", QTWEBENGINE_DISABLE_SANDBOX="1")
    if exe.suffix == ".AppImage" and not Path("/dev/fuse").exists():
        env["APPIMAGE_EXTRACT_AND_RUN"] = "1"  # no FUSE (containers): extract each start
    times: list[float] = []
    report: dict[str, Any] = {}
    for _ in range(runs):
        start = time.monotonic()
        done = subprocess.run(  # noqa: S603 - our own build output
            [str(exe), "--self-test"], capture_output=True, text=True, env=env, timeout=180,
            check=False,
        )  # fmt: skip
        times.append(time.monotonic() - start)
        lines = [ln for ln in done.stdout.splitlines() if '"self-test"' in ln]
        if done.returncode != 0 or not lines:
            return {"ok": False, "exit_code": done.returncode,
                    "stderr_tail": done.stderr.strip().splitlines()[-15:]}  # fmt: skip
        report = json.loads(lines[-1])
    return {"ok": True, "startup_s_median": round(statistics.median(times), 2),
            "startup_s_runs": [round(t, 2) for t in times], "report": report}  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", choices=["core", "viewer", "both"], default="both")
    parser.add_argument("--brush", type=Path, help="Brush binary to bundle as a backend")
    parser.add_argument("--out", type=Path, default=REPO / "build" / "packaging-spike")
    parser.add_argument(
        "--headless", action="store_true", help="self-test without a display (CI, containers)"
    )
    parser.add_argument("--no-appimage", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    brush = args.brush.expanduser().resolve() if args.brush else None

    results: list[dict[str, Any]] = []
    variants = ["core", "viewer"] if args.variant == "both" else [args.variant]
    for variant in variants:
        bundle = pyinstaller(variant, out, brush)
        size, files = tree_size(bundle)
        entry: dict[str, Any] = {
            "variant": variant, "format": "app" if bundle.suffix == ".app" else "onedir",
            "size_mb": mb(size), "files": files,
            "self_test": self_test(executable(bundle), args.headless),
        }  # fmt: skip
        results.append(entry)
        if sys.platform == "darwin":
            archive = out / f"{bundle.stem}-macos.zip"
            run(["ditto", "-c", "-k", "--keepParent", bundle, archive])
            results.append({"variant": variant, "format": "app.zip",
                            "size_mb": mb(archive.stat().st_size), "files": 1})  # fmt: skip
        elif sys.platform.startswith("linux") and not args.no_appimage:
            image = appimage(bundle, out)
            results.append({"variant": variant, "format": "AppImage",
                            "size_mb": mb(image.stat().st_size), "files": 1,
                            "self_test": self_test(image, args.headless)})  # fmt: skip

    meta = {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "pyside6": __import__("PySide6").__version__,
        "pyinstaller": __import__("PyInstaller").__version__,
        "brush_bundled": str(brush) if brush else None,
    }
    (out / "report.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    lines = [
        f"Packaging spike on {meta['platform']}: Python {meta['python']}, PySide6 "
        f"{meta['pyside6']}, PyInstaller {meta['pyinstaller']}",
        "",
        "| Variant | Format | Size | Files | Startup (median of 3) | Self-test |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        st = r.get("self_test")
        status = "–" if st is None else ("ok" if st["ok"] else f"FAILED (exit {st['exit_code']})")
        startup = f"{st['startup_s_median']} s" if st and st["ok"] else "–"
        lines.append(f"| {r['variant']} | {r['format']} | {r['size_mb']} MB | {r['files']} "
                     f"| {startup} | {status} |")  # fmt: skip
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    failed = [r for r in results if r.get("self_test") and not r["self_test"]["ok"]]
    for r in failed:
        print(f"\n{r['variant']} {r['format']} self-test failed:", *r["self_test"]["stderr_tail"],
              sep="\n  ")  # fmt: skip
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
