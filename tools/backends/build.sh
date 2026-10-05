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

PREFIX=$WORK/prefix
rm -rf "$PREFIX"
mkdir -p "$PREFIX/bin" "$PREFIX/licenses"
TOOLCHAIN=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

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
  -DCMAKE_INSTALL_PREFIX="$WORK/colmap-install"
cmake --build colmap-build --parallel "$JOBS"
cmake --install colmap-build
cp "$WORK/colmap-install/bin/colmap" "$PREFIX/bin/"

log "OpenMVS $OPENMVS_VERSION"
fetch openmvs https://github.com/cdcseacave/openMVS.git "$OPENMVS_VERSION"
cmake -S openmvs -B openmvs-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DVCPKG_TARGET_TRIPLET="$TRIPLET" \
  -DVCPKG_INSTALLED_DIR="$WORK/openmvs-vcpkg_installed" \
  -DOpenMVS_USE_CUDA=OFF -DOpenMVS_USE_PYTHON=OFF -DOpenMVS_BUILD_VIEWER=OFF \
  -DCMAKE_INSTALL_PREFIX="$WORK/openmvs-install"
cmake --build openmvs-build --parallel "$JOBS"
for tool in InterfaceCOLMAP DensifyPointCloud ReconstructMesh RefineMesh TextureMesh; do
  cp "$(find openmvs-build/bin -type f -name "$tool" | head -1)" "$PREFIX/bin/"
done

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
