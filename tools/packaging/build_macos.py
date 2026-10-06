# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the EZ2DIGITIZE macOS app (Apple Silicon) with the backends inside.

    uv run --group packaging python tools/packaging/build_macos.py \\
        --backends build/backends/ez2d-backends-macos-arm64.tar.gz \\
        --brush build/brush --smoke build/synthetic/images

Steps: PyInstaller `.app` (windowed, the same entry point as the AppImage,
so `EZ2DIGITIZE.app/Contents/MacOS/EZ2DIGITIZE run ...` is the CLI), the
backend archive unpacked into `Contents/Resources/backends` (its binaries
are already ad-hoc signed and find their libomp through
`@executable_path/../lib`), Brush next to them, the whole app signed
ad hoc, and zipped with `ditto`. The app is not notarized (Phase 6): a
downloaded copy must be un-quarantined (`xattr -dr com.apple.quarantine`).

`--smoke DIR` runs the same checks as the AppImage build.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import shutil
import sys
import tarfile
from pathlib import Path

from build_appimage import EXCLUDED_MODULES, REPO, VIEWER_DATA, output, run, smoke
from release import Version

from ez2digitize import __version__

APP_NAME = "EZ2DIGITIZE"
BUNDLE_ID = "org.ez2digitize.app"


def pyinstaller_app(out: Path) -> Path:
    cmd: list[str | Path] = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
        "--windowed", "--name", APP_NAME, "--osx-bundle-identifier", BUNDLE_ID,
        "--distpath", out / "dist", "--workpath", out / "work", "--specpath", out,
        "--paths", REPO / "src",
    ]  # fmt: skip
    for module in EXCLUDED_MODULES:
        cmd += ["--exclude-module", module]
    for text in ("LICENSE", "THIRD_PARTY_LICENSES"):  # shown under Help -> Licenses
        cmd += ["--add-data", f"{REPO / text}{os.pathsep}."]
    cmd += ["--add-data", VIEWER_DATA]
    cmd.append(Path(__file__).resolve().parent / "entry.py")
    run(cmd)
    # Named EZ2DIGITIZE directly: macOS file systems ignore case, so a rename
    # from ez2digitize.app would be a no-op (or worse).
    return out / "dist" / f"{APP_NAME}.app"


def set_version(app: Path, version: Version) -> None:
    """The app's version in Info.plist (PyInstaller writes 0.0.0): Finder and About show it."""
    plist = app / "Contents" / "Info.plist"
    info = plistlib.loads(plist.read_bytes())
    # Bundle versions are numbers only: a pre-release (1.2.0rc1) shows as 1.2.0.
    info["CFBundleShortVersionString"] = version.numeric
    info["CFBundleVersion"] = version.numeric
    plist.write_bytes(plistlib.dumps(info))


def add_backends(app: Path, archive: Path) -> dict[str, object]:
    target = app / "Contents" / "Resources" / "backends"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    links = output(["otool", "-L", *sorted((target / "bin").iterdir())])
    if "/opt/homebrew" in links or "/usr/local/opt" in links:
        raise SystemExit(f"backends still depend on Homebrew:\n{links}")
    info = json.loads((target / "BUILDINFO.json").read_text())
    return {k: info.get(k) for k in ("colmap", "openmvs", "openmvs_patches")}


def add_brush(app: Path, release: Path) -> str:
    target = app / "Contents" / "Resources" / "backends"
    shutil.copy2(release / "brush_app", target / "bin" / "brush_app")
    licenses = target / "licenses" / "brush"
    licenses.mkdir(parents=True, exist_ok=True)
    shutil.copy2(release / "LICENSE", licenses / "LICENSE")
    version: str = output([target / "bin" / "brush_app", "--version"]).strip()
    return version


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backends", type=Path, required=True, help="ez2d-backends-*.tar.gz")
    parser.add_argument("--brush", type=Path, metavar="DIR", help="unpacked Brush release")
    parser.add_argument("--out", type=Path, default=REPO / "build" / "macos")
    parser.add_argument("--smoke", type=Path, metavar="PHOTOS", help="test with these photos")
    args = parser.parse_args()
    if sys.platform != "darwin":
        raise SystemExit("the .app is built on macOS")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    app = pyinstaller_app(out)
    set_version(app, Version.parse(__version__))
    report: dict[str, object] = {"backends": add_backends(app, args.backends.resolve())}
    if args.brush:
        report["brush"] = add_brush(app, args.brush.resolve())
    run(["codesign", "--force", "--deep", "--sign", "-", app])
    run(["codesign", "--verify", "--deep", "--strict", app])
    archive = out / f"{APP_NAME}-{__version__}-macos-arm64.zip"
    archive.unlink(missing_ok=True)
    run(["ditto", "-c", "-k", "--keepParent", app, archive])
    report["zip"] = archive.name
    report["size_mb"] = round(archive.stat().st_size / 2**20, 1)
    if args.smoke:
        exe = app / "Contents" / "MacOS" / APP_NAME
        report["smoke"] = smoke(exe, args.smoke.resolve(), brush=args.brush is not None)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
