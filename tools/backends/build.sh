#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Build pinned, CPU-only COLMAP and OpenMVS with vcpkg and package them.
#
#   tools/backends/build.sh            # Linux x86_64 or macOS arm64
#
# All dependencies are built from source by vcpkg and linked statically, so
# the binaries only need the system C/C++ runtime. Output:
#   build/backends/ez2d-backends-<os>-<arch>.tar.gz
#     bin/          colmap, InterfaceCOLMAP, DensifyPointCloud, ...
#     licenses/     copyright files of every library linked in
#     BUILDINFO.json
#
# The first build takes hours (OpenCV, Boost, Ceres, OpenImageIO...). vcpkg
# keeps built packages in $VCPKG_BINARY_CACHE, so later builds are fast.
#
# Environment overrides:
#   EZ2D_BACKENDS_WORK   work directory          (default: build/backends)
#   VCPKG_BINARY_CACHE   vcpkg binary cache       (default: $WORK/vcpkg-cache)
#   JOBS                 parallel build jobs      (default: all cores)
set -euo pipefail

COLMAP_VERSION=4.2.1
OPENMVS_VERSION=v2.4.0
VCPKG_VERSION=2026.07.29   # used for OpenMVS; COLMAP pins its own vcpkg baseline

REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${EZ2D_BACKENDS_WORK:-$REPO/build/backends}
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)}
export VCPKG_BINARY_CACHE=${VCPKG_BINARY_CACHE:-$WORK/vcpkg-cache}

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)  TRIPLET=x64-linux-release;  OS=linux; ARCH=x86_64 ;;
  Darwin-arm64)  TRIPLET=arm64-osx-release;  OS=macos; ARCH=arm64 ;;
  *) echo "unsupported platform: $(uname -s) $(uname -m)" >&2; exit 1 ;;
esac

log() { printf '\n=== %s\n' "$*"; }

mkdir -p "$WORK" "$VCPKG_BINARY_CACHE"
cd "$WORK"
export VCPKG_ROOT=$WORK/vcpkg
export VCPKG_BINARY_SOURCES="clear;files,$VCPKG_BINARY_CACHE,readwrite"
export VCPKG_DISABLE_METRICS=1
# Remove buildtrees after each port: keeps disk use to a few GB.
export VCPKG_INSTALL_OPTIONS="--clean-after-build"
# Release-only triplets halve the build time.
export VCPKG_DEFAULT_TRIPLET=$TRIPLET VCPKG_DEFAULT_HOST_TRIPLET=$TRIPLET

log "vcpkg $VCPKG_VERSION"
if [ ! -d vcpkg ]; then
  # Full history (without blobs): manifest baselines need older commits.
  git clone --filter=blob:none https://github.com/microsoft/vcpkg.git vcpkg
fi
git -C vcpkg fetch --tags --quiet
git -C vcpkg -c advice.detachedHead=false checkout --quiet "$VCPKG_VERSION"
[ -x vcpkg/vcpkg ] || vcpkg/bootstrap-vcpkg.sh -disableMetrics

fetch() {  # fetch <dir> <git url> <tag>
  if [ ! -d "$1" ]; then
    git clone --quiet --depth 1 --branch "$3" --recurse-submodules --shallow-submodules \
      -c advice.detachedHead=false "$2" "$1"
  fi
}

# Apple's compiler has no OpenMP; use Homebrew's libomp and ship it in lib/
# (rewritten below), so the archive doesn't depend on Homebrew.
OPENMP_ARGS=()
if [ "$OS" = macos ]; then
  LIBOMP=$(brew --prefix libomp)
  # CMake can't detect OpenMP for AppleClang on its own; spell it out.
  OMP_FLAGS="-Xpreprocessor -fopenmp -I$LIBOMP/include"
  OPENMP_ARGS=(
    -DOpenMP_ROOT="$LIBOMP"
    -DOpenMP_C_FLAGS="$OMP_FLAGS" -DOpenMP_CXX_FLAGS="$OMP_FLAGS"
    -DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp
    -DOpenMP_omp_LIBRARY="$LIBOMP/lib/libomp.dylib"
  )
fi

PREFIX=$WORK/prefix
rm -rf "$PREFIX"
mkdir -p "$PREFIX/bin" "$PREFIX/licenses"
TOOLCHAIN=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# COLMAP 4.2.1 requires OpenGL and GLEW at configure time even for a headless
# build; they are only linked with the GUI or GPU features, which are off. The
# dev packages are installed just to satisfy the lookup (checked below).
log "COLMAP $COLMAP_VERSION"
fetch colmap https://github.com/colmap/colmap.git "$COLMAP_VERSION"
cmake -S colmap -B colmap-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DVCPKG_TARGET_TRIPLET="$TRIPLET" \
  -DVCPKG_MANIFEST_NO_DEFAULT_FEATURES=ON \
  -DVCPKG_INSTALLED_DIR="$WORK/colmap-vcpkg_installed" \
  -DCUDA_ENABLED=OFF -DHIP_ENABLED=OFF -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF \
  -DONNX_ENABLED=OFF -DCGAL_ENABLED=OFF -DDOWNLOAD_ENABLED=OFF -DTESTS_ENABLED=OFF \
  -DCCACHE_ENABLED=OFF \
  ${OPENMP_ARGS[@]+"${OPENMP_ARGS[@]}"} \
  -DCMAKE_INSTALL_PREFIX="$WORK/colmap-install"
cmake --build colmap-build --parallel "$JOBS"
cmake --install colmap-build
cp "$WORK/colmap-install/bin/colmap" "$PREFIX/bin/"

log "OpenMVS $OPENMVS_VERSION"
fetch openmvs https://github.com/cdcseacave/openMVS.git "$OPENMVS_VERSION"
# OpenMVS asks for vcpkg's "opencv" with its default features, which on Linux
# include the GTK GUI backend: a large GTK/X11 build (it failed on at-spi2-core)
# for windows OpenMVS only opens in debug builds. Ask for OpenCV without
# default features, keeping what OpenMVS uses: calib3d (stereo matching,
# speckle filter, rectification) and the image formats it reads and writes.
python3 - openmvs/vcpkg.json <<'EOF'
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
EOF
cmake -S openmvs -B openmvs-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DVCPKG_TARGET_TRIPLET="$TRIPLET" \
  -DVCPKG_INSTALLED_DIR="$WORK/openmvs-vcpkg_installed" \
  -DOpenMVS_USE_CUDA=OFF -DOpenMVS_USE_PYTHON=OFF -DOpenMVS_BUILD_VIEWER=OFF \
  -DOpenMVS_USE_OPENMP=ON \
  ${OPENMP_ARGS[@]+"${OPENMP_ARGS[@]}"} \
  -DCMAKE_INSTALL_PREFIX="$WORK/openmvs-install"
cmake --build openmvs-build --parallel "$JOBS"
for tool in InterfaceCOLMAP DensifyPointCloud ReconstructMesh RefineMesh TextureMesh; do
  cp "$(find openmvs-build/bin -type f -name "$tool" | head -1)" "$PREFIX/bin/"
done

if [ "$OS" = macos ]; then
  log "relocating libomp"
  mkdir -p "$PREFIX/lib"
  cp "$LIBOMP/lib/libomp.dylib" "$PREFIX/lib/libomp.dylib"
  chmod u+w "$PREFIX/lib/libomp.dylib"
  install_name_tool -id "@rpath/libomp.dylib" "$PREFIX/lib/libomp.dylib"
  codesign --force --sign - "$PREFIX/lib/libomp.dylib"
  for exe in "$PREFIX"/bin/*; do
    for dep in $(otool -L "$exe" | awk 'NR > 1 {print $1}' | grep 'libomp\.dylib$' || true); do
      install_name_tool -change "$dep" "@executable_path/../lib/libomp.dylib" "$exe"
    done
    codesign --force --sign - "$exe"   # editing the binary invalidated its signature
  done
  # Anything still pointing into Homebrew would break on machines without it.
  if otool -L "$PREFIX"/bin/* "$PREFIX"/lib/* | grep -E '/opt/homebrew|/usr/local/(opt|Cellar)'; then
    echo "error: binaries still depend on Homebrew paths (above)" >&2
    exit 1
  fi
  cp "$LIBOMP/LICENSE"* "$PREFIX/licenses/llvm-openmp.txt" 2>/dev/null || true
fi

log "licenses"
cp colmap/LICENSE.txt "$PREFIX/licenses/colmap.txt" 2>/dev/null || cp colmap/COPYING.txt "$PREFIX/licenses/colmap.txt"
cp openmvs/LICENSE "$PREFIX/licenses/openmvs.txt"
for installed in colmap-vcpkg_installed openmvs-vcpkg_installed; do
  for copyright in "$WORK/$installed/$TRIPLET"/share/*/copyright; do
    port=$(basename "$(dirname "$copyright")")
    cp "$copyright" "$PREFIX/licenses/$port.txt"
  done
done

log "smoke test"
"$PREFIX/bin/colmap" help | head -2
"$PREFIX/bin/InterfaceCOLMAP" --help | grep -m1 OpenMVS || true

if [ "$OS" = linux ]; then
  GLIBC=$(objdump -T "$PREFIX"/bin/* 2>/dev/null | grep -o 'GLIBC_2\.[0-9]*' | sort -t. -k2 -n -u | tail -1)
  DYNAMIC=$(for f in "$PREFIX"/bin/*; do ldd "$f" | awk '{print $1}'; done | sort -u | tr '\n' ' ')
  if echo "$DYNAMIC" | grep -Eq 'libGL|libGLEW|libX11'; then
    echo "error: headless binaries link graphics libraries: $DYNAMIC" >&2
    exit 1
  fi
else
  GLIBC=none
  DYNAMIC=$(for f in "$PREFIX"/bin/*; do otool -L "$f" | tail -n +2 | awk '{print $1}'; done | sort -u | tr '\n' ' ')
fi
cat > "$PREFIX/BUILDINFO.json" <<EOF
{
  "colmap": "$COLMAP_VERSION",
  "openmvs": "$OPENMVS_VERSION",
  "vcpkg": "$VCPKG_VERSION",
  "triplet": "$TRIPLET",
  "built": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "compiler": "$(c++ --version | head -1)",
  "requires_glibc": "$GLIBC",
  "dynamic_libraries": "$DYNAMIC"
}
EOF
cat "$PREFIX/BUILDINFO.json"

ARCHIVE=$WORK/ez2d-backends-$OS-$ARCH.tar.gz
tar -czf "$ARCHIVE" -C "$PREFIX" .
log "wrote $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
