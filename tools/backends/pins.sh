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
# LLVM's OpenMP runtime for the clang-cl build on Windows: the release of the
# clang-cl in Visual Studio. LLVM 22 publishes only the whole source tree.
LLVM_VERSION=22.1.3
LLVM_URL=https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/llvm-project-$LLVM_VERSION.src.tar.xz
LLVM_SHA256=2488c33a959eafba1c44f253e5bbe7ac958eb53fa626298a3a5f4b87373767cd

platform() {  # sets TRIPLET (vcpkg), OS, ARCH and EXE (".exe" on Windows)
  EXE=
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  TRIPLET=x64-linux-release;  OS=linux; ARCH=x86_64 ;;
    Darwin-arm64)  TRIPLET=arm64-osx-release;  OS=macos; ARCH=arm64 ;;
    # Everything static, the C runtime included (/MT): the binaries then need
    # only Windows, plus the OpenMP runtime, which exists only as a DLL.
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

# Sources whose upstream servers are unreliable from CI runners: GNU's
# ftpmirror/ftp.gnu.org timed out from GitHub's macOS runners (October 2026),
# and gmplib.org with them. vcpkg uses a file already in its downloads folder
# when its SHA512 matches the port's, so these are fetched from another
# mirror first. Name, mirror URL, the pinned vcpkg port's SHA512.
MIRRORED_SOURCES=(
  "gmp-6.3.0.tar.xz https://mirrors.kernel.org/gnu/gmp/gmp-6.3.0.tar.xz e85a0dab5195889948a3462189f0e0598d331d3457612e2d3350799dba2e244316d256f8161df5219538eb003e4b5343f989aaa00f96321559063ed8c8f29fd2"
  "mpfr-4.2.2.tar.xz https://mirrors.kernel.org/gnu/mpfr/mpfr-4.2.2.tar.xz eb9e7f51b5385fb349cc4fba3a45ffdf0dd53be6dfc74932dc01258158a10514667960c530c47dd9dfc5aa18be2bd94859d80499844c5713710581e6ac6259a9"
  "automake-1.17.tar.gz https://mirrors.kernel.org/gnu/automake/automake-1.17.tar.gz 11357dfab8cbf4b5d94d9d06e475732ca01df82bef1284888a34bd558afc37b1a239bed1b5eb18a9dbcc326344fb7b1b301f77bb8385131eb8e1e118b677883a"
)

prefetch_sources() {  # prefetch_sources <vcpkg downloads dir>
  mkdir -p "$1"
  local entry name url sha got
  for entry in "${MIRRORED_SOURCES[@]}"; do
    read -r name url sha <<<"$entry"
    [ -f "$1/$name" ] && continue
    if curl -sSLf --retry 3 -m 300 -o "$1/$name.part" "$url"; then
      got=$( (sha512sum "$1/$name.part" 2>/dev/null || shasum -a 512 "$1/$name.part") | cut -d' ' -f1)
      if [ "$got" = "$sha" ]; then
        mv "$1/$name.part" "$1/$name"
        echo "prefetched $name"
        continue
      fi
      echo "warning: $name from $url has another hash; vcpkg downloads it itself" >&2
    else
      echo "warning: could not prefetch $name; vcpkg downloads it itself" >&2
    fi
    rm -f "$1/$name.part"
  done
}

prepare_vcpkg_overlays() {  # prepare_vcpkg_overlays <vcpkg root> <repo root> <overlay dir>
  # Fixes to ports of the pinned vcpkg release (see each patch's header): the
  # ports they touch are copied, patched and used as overlay ports, which win
  # over the release's version and any manifest baseline's. Sets
  # VCPKG_OVERLAY_PORTS. The fixes so far concern Windows only.
  rm -rf "$3"
  mkdir -p "$3/ports"
  local patch port
  for patch in "$2"/tools/backends/patches/vcpkg-*.patch; do
    [ -e "$patch" ] || continue
    for port in $(sed -n 's|^+++ b/ports/\([^/]*\)/.*|\1|p' "$patch" | sort -u); do
      [ -d "$3/ports/$port" ] || cp -R "$1/ports/$port" "$3/ports/$port"
    done
    patch -p1 --batch --forward -d "$3" <"$patch"
  done
  # vcpkg.exe wants a Windows path; cygpath exists only in Git Bash.
  VCPKG_OVERLAY_PORTS=$(cygpath -m "$3/ports" 2>/dev/null || echo "$3/ports")
  export VCPKG_OVERLAY_PORTS
  echo "overlay ports: $(ls "$3/ports" | tr '\n' ' ')"
}

# Python for the manifest edits below (Windows has no python3 by that name).
PYTHON=${PYTHON:-$(command -v python3 || command -v python)}

fetch_llvm_openmp() {  # fetch_llvm_openmp <dir>: LLVM's openmp/ and cmake/ (all it needs)
  [ -d "$1/openmp" ] && return
  local archive
  archive=$(basename "$LLVM_URL")
  [ -e "$archive" ] || curl -sSfL -o "$archive" "$LLVM_URL"
  if ! echo "$LLVM_SHA256  $archive" | sha256sum -c --status -; then
    echo "error: $archive doesn't match its pinned checksum" >&2
    rm -f "$archive"
    return 1
  fi
  mkdir -p "$1"
  tar -xJf "$archive" -C "$1" --strip-components=1 \
    "llvm-project-$LLVM_VERSION.src/openmp" "llvm-project-$LLVM_VERSION.src/cmake"
}

apply_patches() {  # apply_patches <checkout> <patch>...
  # A checkout kept from an earlier run may already carry them.
  local checkout=$1 patch
  shift
  for patch in "$@"; do
    if git -C "$checkout" apply --reverse --check "$patch" 2>/dev/null; then
      echo "already applied: $(basename "$patch")"
    else
      git -C "$checkout" apply "$patch"
    fi
  done
}

patch_colmap() {  # patch_colmap <checkout> <repo root>: every platform
  # Fixes to COLMAP's code (see each patch's header).
  apply_patches "$1" "$2"/tools/backends/patches/colmap-*.patch
}

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
  apply_patches "$1" "$2"/tools/backends/patches/openmvs-*.patch
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
