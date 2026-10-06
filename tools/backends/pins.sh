# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# The pinned backend sources, shared by build.sh (which builds them) and
# collect_sources.sh (which packs their exact source for a release).
# Sourced, not run.

COLMAP_VERSION=4.2.1
OPENMVS_VERSION=v2.4.0
VCPKG_VERSION=2026.07.29   # used for OpenMVS; COLMAP pins its own vcpkg baseline
COLMAP_URL=https://github.com/colmap/colmap.git
OPENMVS_URL=https://github.com/cdcseacave/openMVS.git
VCPKG_URL=https://github.com/microsoft/vcpkg.git

platform() {  # sets TRIPLET (vcpkg), OS, ARCH and EXE (".exe" on Windows)
  EXE=
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  TRIPLET=x64-linux-release;  OS=linux; ARCH=x86_64 ;;
    Darwin-arm64)  TRIPLET=arm64-osx-release;  OS=macos; ARCH=arm64 ;;
    # Everything static, the C runtime included (/MT): the binaries then need
    # only Windows, plus MSVC's OpenMP runtime, which exists only as a DLL.
    MINGW*-x86_64 | MSYS*-x86_64)
      TRIPLET=x64-windows-static-release; OS=windows; ARCH=x86_64; EXE=.exe ;;
    *) echo "unsupported platform: $(uname -s) $(uname -m)" >&2; exit 1 ;;
  esac
}

fetch() {  # fetch <dir> <git url> <tag>
  if [ ! -d "$1" ]; then
    git clone --quiet --depth 1 --branch "$3" --recurse-submodules --shallow-submodules \
      -c advice.detachedHead=false "$2" "$1"
  fi
}

fetch_vcpkg() {  # fetch_vcpkg <dir>: the vcpkg release, with history for baselines
  if [ ! -d "$1" ]; then
    # Full history (without blobs): manifest baselines need older commits.
    git clone --filter=blob:none "$VCPKG_URL" "$1"
  fi
  git -C "$1" fetch --tags --quiet
  git -C "$1" -c advice.detachedHead=false checkout --quiet "$VCPKG_VERSION"
  if [ -x "$1/vcpkg" ] || [ -x "$1/vcpkg.exe" ]; then
    return
  fi
  case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) cmd //c "$(cygpath -w "$1/bootstrap-vcpkg.bat")" -disableMetrics ;;
    *) "$1/bootstrap-vcpkg.sh" -disableMetrics ;;
  esac
}

# Python for the manifest edits below (Windows has no python3 by that name).
PYTHON=${PYTHON:-$(command -v python3 || command -v python)}

prepare_colmap() {  # prepare_colmap <checkout>: Windows only
  # COLMAP 4.2.1 looks for GLEW at configure time even for a headless build
  # (Linux and macOS satisfy it with a system package). On Windows it comes
  # from vcpkg; it isn't linked with the GUI and GPU features off.
  "$PYTHON" - "$1/vcpkg.json" <<'PYEOF'
import json, sys
path = sys.argv[1]
manifest = json.load(open(path))
names = [d if isinstance(d, str) else d["name"] for d in manifest["dependencies"]]
if "glew" not in names:
    manifest["dependencies"].append("glew")
json.dump(manifest, open(path, "w"), indent=2)
PYEOF
}

prepare_openmvs() {  # prepare_openmvs <checkout> <repo root>: patches and manifest
  # Upstream fixes released after the pinned version (see each patch's header).
  # A checkout kept from an earlier run may already carry them.
  for patch in "$2"/tools/backends/patches/openmvs-*.patch; do
    if git -C "$1" apply --reverse --check "$patch" 2>/dev/null; then
      echo "already applied: $(basename "$patch")"
    else
      git -C "$1" apply "$patch"
    fi
  done
  # OpenMVS asks for vcpkg's "opencv" with its default features, which on Linux
  # include the GTK GUI backend: a large GTK/X11 build (it failed on at-spi2-core)
  # for windows OpenMVS only opens in debug builds. Ask for OpenCV without
  # default features, keeping what OpenMVS uses: calib3d (stereo matching,
  # speckle filter, rectification) and the image formats it reads and writes.
  "$PYTHON" - "$1/vcpkg.json" <<'PYEOF'
import json, sys
path = sys.argv[1]
manifest = json.load(open(path))
deps = [
    d for d in manifest["dependencies"]
    if not (isinstance(d, dict) and d["name"] in ("opencv", "opencv4"))
]
deps.append({
    "name": "opencv4",
    "default-features": False,
    "features": ["calib3d", "eigen", "jpeg", "jpegxl", "openexr", "png", "tiff"],
})
manifest["dependencies"] = deps
json.dump(manifest, open(path, "w"), indent=2)
PYEOF
}
