#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Build pinned, CPU-only COLMAP and OpenMVS with vcpkg and package them.
#
#   tools/backends/build.sh            # Linux x86_64, macOS arm64, Windows x64
#
# On Windows it runs in Git Bash, with the MSVC environment set up (vcvars64,
# or the msvc-dev-cmd action in CI). Keep EZ2D_BACKENDS_WORK short there
# (C:/ez2d): vcpkg's build trees easily pass the 260-character path limit.
#
# All dependencies are built from source by vcpkg and linked statically, so
# the binaries only need the system C/C++ runtime, except ONNX Runtime:
# COLMAP's learned features (ALIKED, LightGlue) run on it, and COLMAP's build
# fetches Microsoft's release library (pinned by hash in COLMAP's CMake),
# which is shipped next to it. Output:
#   build/backends/ez2d-backends-<os>-<arch>.tar.gz
#     bin/          colmap, InterfaceCOLMAP, DensifyPointCloud, ... (.exe on
#                   Windows, with vcomp140.dll, MSVC's OpenMP runtime, and
#                   any other DLL they need: see "runtime DLLs" below)
#     lib/          ONNX Runtime (Linux, macOS; on Windows its DLLs are in
#                   bin/), and libomp on macOS
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
#   WINDOWS_COMPILER     clang-cl or msvc         (default: clang-cl)
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
# shellcheck source=pins.sh
. "$REPO/tools/backends/pins.sh"  # versions, platform, fetch, prepare_*
WORK=${EZ2D_BACKENDS_WORK:-$REPO/build/backends}
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || nproc)}
export VCPKG_BINARY_CACHE=${VCPKG_BINARY_CACHE:-$WORK/vcpkg-cache}

platform  # TRIPLET, OS, ARCH, EXE

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
fetch_vcpkg vcpkg
prefetch_sources "${VCPKG_DOWNLOADS:-$VCPKG_ROOT/downloads}"
[ "$OS" = windows ] && prepare_vcpkg_overlays vcpkg "$REPO" "$WORK/vcpkg-overlays"

# Apple's compiler has no OpenMP; use Homebrew's libomp and ship it in lib/
# (rewritten below), so the archive doesn't depend on Homebrew.
PLATFORM_ARGS=()
if [ "$OS" = macos ]; then
  LIBOMP=$(brew --prefix libomp)
  # CMake can't detect OpenMP for AppleClang on its own; spell it out.
  OMP_FLAGS="-Xpreprocessor -fopenmp -I$LIBOMP/include"
  PLATFORM_ARGS=(
    -DOpenMP_ROOT="$LIBOMP"
    -DOpenMP_C_FLAGS="$OMP_FLAGS" -DOpenMP_CXX_FLAGS="$OMP_FLAGS"
    -DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp
    -DOpenMP_omp_LIBRARY="$LIBOMP/lib/libomp.dylib"
  )
fi

if [ "$OS" = windows ]; then
  # The static C runtime, as in the vcpkg libraries (the toolchain leaves the
  # project's own code on the DLL runtime otherwise).
  PLATFORM_ARGS=(-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DCMAKE_POLICY_DEFAULT_CMP0091=NEW)
  WINDOWS_COMPILER=${WINDOWS_COMPILER:-clang-cl}
  if [ "$WINDOWS_COMPILER" = clang-cl ]; then
    # COLMAP's and OpenMVS's own code with clang-cl and lld-link, the vcpkg
    # libraries staying MSVC's (the same ABI, and the cache still applies).
    # With MSVC, OpenMVS always turns on link-time code generation, and its
    # five tools took 55 to 72 minutes to link on the CI runner; turned off,
    # compiling took 2 h 46. LLVM optimises while compiling, as on Linux and
    # macOS, where OpenMVS builds in about 2 minutes; clang-cl ignores /GL and
    # lld-link /LTCG. The clang-cl that comes with Visual Studio matches its
    # C++ library; a separate LLVM install is the fallback.
    CLANG_CL="$(cygpath -u "${VCINSTALLDIR:-}")/Tools/Llvm/x64/bin/clang-cl.exe"
    [ -x "$CLANG_CL" ] || CLANG_CL=$(command -v clang-cl || true)
    [ -n "$CLANG_CL" ] || { echo "error: clang-cl not found (Visual Studio's C++ Clang tools)" >&2; exit 1; }
    LLD_LINK=$(dirname "$CLANG_CL")/lld-link.exe
    # OpenMP: LLVM's own runtime (libomp.dll, shipped below), built from the
    # pinned LLVM release with the same clang-cl in a few seconds. Visual
    # Studio's build of it for /openmp:llvm (libomp140.x86_64.dll) is only in
    # its debug_nonredist folder: not licensed for shipping.
    log "LLVM OpenMP runtime $LLVM_VERSION"
    fetch_llvm_openmp llvm
    LLVM_OPENMP=$WORK/llvm-openmp
    cmake -S llvm/openmp -B llvm-openmp-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$(cygpath -m "$CLANG_CL")" \
      -DCMAKE_CXX_COMPILER="$(cygpath -m "$CLANG_CL")" \
      -DCMAKE_LINKER="$(cygpath -m "$LLD_LINK")" \
      -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DCMAKE_POLICY_DEFAULT_CMP0091=NEW \
      -DCMAKE_INSTALL_PREFIX="$(cygpath -m "$LLVM_OPENMP")" \
      -DOPENMP_STANDALONE_BUILD=ON -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
      -DLIBOMP_OMPT_SUPPORT=OFF -DLIBOMP_OMPD_SUPPORT=OFF -DOPENMP_ENABLE_OMPT_TOOLS=OFF \
      -DLIBOMP_INSTALL_ALIASES=OFF
    cmake --build llvm-openmp-build --target omp
    rm -rf "$LLVM_OPENMP"
    cmake --install llvm-openmp-build
    LIBOMP_LIB=$(cygpath -m "$LLVM_OPENMP/lib/libomp.lib")
    [ -e "$LIBOMP_LIB" ] || { echo "error: $LIBOMP_LIB not built" >&2; exit 1; }
    OMP_FLAGS="-Xclang -fopenmp -I$(cygpath -m "$LLVM_OPENMP/include")"
    # CMake's defaults for clang-cl (written with "-": Git Bash rewrites
    # arguments starting with "/" as paths), and -w: clang-cl's warnings
    # on these sources ran to 35,000, 350,000 lines of log.
    PLATFORM_ARGS+=(
      -DCMAKE_C_FLAGS="-DWIN32 -D_WINDOWS -w"
      -DCMAKE_CXX_FLAGS="-DWIN32 -D_WINDOWS -GR -EHsc -w"
      -DCMAKE_C_COMPILER="$(cygpath -m "$CLANG_CL")"
      -DCMAKE_CXX_COMPILER="$(cygpath -m "$CLANG_CL")"
      -DCMAKE_LINKER="$(cygpath -m "$LLD_LINK")"
      -DOpenMP_C_FLAGS="$OMP_FLAGS" -DOpenMP_CXX_FLAGS="$OMP_FLAGS"
      -DOpenMP_C_LIB_NAMES=libomp -DOpenMP_CXX_LIB_NAMES=libomp
      -DOpenMP_libomp_LIBRARY="$LIBOMP_LIB"
    )
    "$CLANG_CL" --version | head -1
  fi
fi

# COLMAP picks its Windows settings by compiler ID ("MSVC"), which clang-cl
# isn't ("Clang"): without them, Windows' min/max macros break its code and
# faiss is built in a SIMD mode COLMAP says MSVC can't do. COLMAP only ever
# sets IS_MSVC to true, so setting it here gives back all its MSVC settings
# (those the MSVC build uses). Its link-time optimisation, on for every
# compiler but GCC, stays off as with GCC on Linux.
COLMAP_ARGS=()
if [ "$OS" = windows ] && [ "$WINDOWS_COMPILER" = clang-cl ]; then
  COLMAP_ARGS=(-DIS_MSVC=TRUE -DIPO_ENABLED=OFF)
fi

PREFIX=$WORK/prefix
rm -rf "$PREFIX"
mkdir -p "$PREFIX/bin" "$PREFIX/licenses"
TOOLCHAIN=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# COLMAP 4.2.1 requires OpenGL and GLEW at configure time even for a headless
# build; they are only linked with the GUI or GPU features, which are off. The
# dev packages are installed just to satisfy the lookup (checked below).
log "COLMAP $COLMAP_VERSION"
fetch colmap "$COLMAP_URL" "$COLMAP_VERSION"
patch_colmap colmap "$REPO"
[ "$OS" = windows ] && prepare_colmap colmap
cmake -S colmap -B colmap-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DVCPKG_TARGET_TRIPLET="$TRIPLET" \
  -DVCPKG_MANIFEST_NO_DEFAULT_FEATURES=ON \
  -DVCPKG_INSTALLED_DIR="$WORK/colmap-vcpkg_installed" \
  -DCUDA_ENABLED=OFF -DHIP_ENABLED=OFF -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF \
  -DONNX_ENABLED=ON -DCGAL_ENABLED=OFF -DDOWNLOAD_ENABLED=OFF -DTESTS_ENABLED=OFF \
  -DCCACHE_ENABLED=OFF \
  ${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"} \
  ${COLMAP_ARGS[@]+"${COLMAP_ARGS[@]}"} \
  -DCMAKE_INSTALL_PREFIX="$WORK/colmap-install"
cmake --build colmap-build --parallel "$JOBS"
cmake --install colmap-build
cp "$WORK/colmap-install/bin/colmap$EXE" "$PREFIX/bin/"
# ONNX Runtime, as COLMAP names it (its rpath is ../lib): the one file, not
# the symlinks around it. On Windows the DLL loop below finds its DLLs.
ONNX_VERSION=$(sed -n 's/.*set(ONNX_VERSION "\([0-9.]*\)").*/\1/p' colmap/cmake/FindDependencies.cmake)
if [ "$OS" = linux ]; then
  ONNX_LIB=$(objdump -p "$PREFIX/bin/colmap" | awk '$1 == "NEEDED" && $2 ~ /^libonnxruntime/ {print $2}')
elif [ "$OS" = macos ]; then
  ONNX_LIB=$(otool -L "$PREFIX/bin/colmap" | awk '$1 ~ /libonnxruntime/ {print $1}' | xargs basename)
fi
if [ "$OS" != windows ]; then
  [ -n "$ONNX_LIB" ] || { echo "error: colmap isn't linked to ONNX Runtime" >&2; exit 1; }
  mkdir -p "$PREFIX/lib"
  cp -L "$(ls -d "$WORK"/colmap-install/lib*/"$ONNX_LIB" | head -1)" "$PREFIX/lib/$ONNX_LIB"
fi

log "OpenMVS $OPENMVS_VERSION"
fetch openmvs "$OPENMVS_URL" "$OPENMVS_VERSION"
prepare_openmvs openmvs "$REPO"
# Only the tools the app runs, not Tests, TransformScene or the other
# Interface* importers: on Windows each of those took minutes to link.
# (MSVC's link-time code generation, /GL and /LTCG, which OpenMVS always
# turns on, is left alone: without it the OpenMVS build took 2 h 47 min
# on the CI runner instead of 58 min, single source files taking over an
# hour to optimise where the linker spreads the same work over threads.)
OPENMVS_TOOLS=(InterfaceCOLMAP DensifyPointCloud ReconstructMesh RefineMesh TextureMesh)
cmake -S openmvs -B openmvs-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DVCPKG_TARGET_TRIPLET="$TRIPLET" \
  -DVCPKG_INSTALLED_DIR="$WORK/openmvs-vcpkg_installed" \
  -DOpenMVS_USE_CUDA=OFF -DOpenMVS_USE_PYTHON=OFF -DOpenMVS_BUILD_VIEWER=OFF \
  -DOpenMVS_USE_OPENMP=ON \
  ${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"} \
  -DCMAKE_INSTALL_PREFIX="$WORK/openmvs-install"
cmake --build openmvs-build --parallel "$JOBS" --target "${OPENMVS_TOOLS[@]}"
for tool in "${OPENMVS_TOOLS[@]}"; do
  cp "$(find openmvs-build/bin -type f -name "$tool$EXE" | head -1)" "$PREFIX/bin/"
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

if [ "$OS" = windows ]; then
  log "runtime DLLs"
  # Microsoft's redistributable runtime, from the compiler that built them.
  VCOMP=$(find "$(cygpath -u "$VCToolsRedistDir")/x64" -name vcomp140.dll -path '*OpenMP*' | head -1)
  [ -n "$VCOMP" ] || { echo "error: vcomp140.dll not found under \$VCToolsRedistDir" >&2; exit 1; }
  cp "$VCOMP" "$PREFIX/bin/"
  # Copy the DLLs the binaries need, until none is missing: vcomp140 may
  # need the C++ runtime (redistributable too), and a few vcpkg ports build
  # DLLs even with a static triplet (LAPACK, compiled with MinGW's gfortran,
  # with the GCC runtime). Windows' own DLLs are found nowhere here and skipped.
  CRT=$(dirname "$(find "$(cygpath -u "$VCToolsRedistDir")/x64" -name vcruntime140.dll -path '*.CRT*' | head -1)")
  # LLVM's OpenMP runtime, for code built with clang-cl (built above).
  OMP_LLVM=${LLVM_OPENMP:-/nonexistent}/bin
  while true; do
    added=0
    for dll in $(for f in "$PREFIX"/bin/*.exe "$PREFIX"/bin/*.dll; do
                   dumpbin //nologo //dependents "$(cygpath -w "$f")"
                 done | grep -io '[a-z0-9_.+-]*\.dll' | sort -fu); do
      [ -e "$PREFIX/bin/$dll" ] && continue
      for dir in "$CRT" "$OMP_LLVM" "$WORK"/colmap-install/bin "$WORK"/colmap-vcpkg_installed/"$TRIPLET"/bin "$WORK"/openmvs-vcpkg_installed/"$TRIPLET"/bin; do
        if [ -e "$dir/$dll" ]; then
          echo "shipping $dll (from $dir)"
          cp "$dir/$dll" "$PREFIX/bin/"
          added=1
          break
        fi
      done
    done
    [ "$added" = 1 ] || break
  done
fi

log "licenses"
cp colmap/LICENSE.txt "$PREFIX/licenses/colmap.txt" 2>/dev/null || cp colmap/COPYING.txt "$PREFIX/licenses/colmap.txt"
cp openmvs/LICENSE "$PREFIX/licenses/openmvs.txt"
[ -z "${LLVM_OPENMP:-}" ] || cp llvm/openmp/LICENSE.TXT "$PREFIX/licenses/llvm-openmp.txt"
# Code COLMAP compiles in that isn't a vcpkg port, so has no copyright file
# below: from its own tree (LSD is AGPL-3.0; SiftGPU is only built with a GPU)
# and fetched while configuring (FetchContent: PoseLib, faiss).
for part in LSD PoissonRecon VLFeat; do
  cp "colmap/src/thirdparty/$part/LICENSE" "$PREFIX/licenses/colmap-$part.txt"
done
for dep in poselib faiss; do
  cp colmap-build/_deps/$dep-src/LICENSE "$PREFIX/licenses/colmap-$dep.txt"
done
cp colmap-build/_deps/onnxruntime-src/LICENSE "$PREFIX/licenses/onnxruntime.txt"
cp colmap-build/_deps/onnxruntime-src/ThirdPartyNotices.txt \
  "$PREFIX/licenses/onnxruntime-ThirdPartyNotices.txt"
for installed in colmap-vcpkg_installed openmvs-vcpkg_installed; do
  for copyright in "$WORK/$installed/$TRIPLET"/share/*/copyright; do
    port=$(basename "$(dirname "$copyright")")
    cp "$copyright" "$PREFIX/licenses/$port.txt"
  done
done

log "smoke test"
"$PREFIX/bin/colmap$EXE" help | head -2
"$PREFIX/bin/InterfaceCOLMAP$EXE" --help | grep -m1 OpenMVS || true

if [ "$OS" = linux ]; then
  GLIBC=$(objdump -T "$PREFIX"/bin/* "$PREFIX"/lib/* 2>/dev/null | grep -o 'GLIBC_2\.[0-9]*' | sort -t. -k2 -n -u | tail -1)
  DYNAMIC=$(for f in "$PREFIX"/bin/* "$PREFIX"/lib/*; do ldd "$f" | awk '{print $1}'; done | sort -u | tr '\n' ' ')
  if echo "$DYNAMIC" | grep -Eq 'libGL|libGLEW|libX11'; then
    echo "error: headless binaries link graphics libraries: $DYNAMIC" >&2
    exit 1
  fi
elif [ "$OS" = windows ]; then
  GLIBC=none
  DYNAMIC=$(for f in "$PREFIX"/bin/*.exe "$PREFIX"/bin/*.dll; do dumpbin //nologo //dependents "$(cygpath -w "$f")"; done \
    | grep -io '[a-z0-9_.-]*\.dll' | tr 'A-Z' 'a-z' | sort -u | tr '\n' ' ')
  # Only Windows' own DLLs, and the runtime DLLs shipped next to them (dxgi:
  # ONNX Runtime looks up GPUs with it; it is part of every Windows).
  SHIPPED=$(cd "$PREFIX/bin" && ls ./*.dll | sed 's|^\./||' | tr 'A-Z' 'a-z' | tr '\n' '|')
  UNEXPECTED=$(echo "$DYNAMIC" | tr ' ' '\n' | grep -v -E "^(${SHIPPED})\$" | grep -v -E '^$|^(kernel32|user32|gdi32|advapi32|shell32|ole32|oleaut32|ws2_32|bcrypt|crypt32|shlwapi|dbghelp|psapi|comdlg32|winmm|version|secur32|ncrypt|iphlpapi|opengl32|glu32|dxgi|setupapi|cfgmgr32|userenv|rpcrt4|vcomp140|api-ms-win-[a-z0-9-]*)\.dll$' || true)
  if [ -n "$UNEXPECTED" ]; then
    echo "error: the binaries need DLLs that won't be on users' machines: $UNEXPECTED" >&2
    exit 1
  fi
else
  GLIBC=none
  DYNAMIC=$(for f in "$PREFIX"/bin/*; do otool -L "$f" | tail -n +2 | awk '{print $1}'; done | sort -u | tr '\n' ' ')
fi
if [ "$OS" = windows ] && [ "$WINDOWS_COMPILER" = clang-cl ]; then
  COMPILER="$("$CLANG_CL" --version | head -1 | tr -d '\r') (vcpkg libraries: $(cl 2>&1 | head -1 | tr -d '\r'))"
elif [ "$OS" = windows ]; then
  COMPILER=$(cl 2>&1 | head -1 | tr -d '\r')
else
  COMPILER=$(c++ --version | head -1)
fi
cat > "$PREFIX/BUILDINFO.json" <<EOF
{
  "colmap": "$COLMAP_VERSION",
  "onnxruntime": "$ONNX_VERSION",
  "openmvs": "$OPENMVS_VERSION",
  "colmap_patches": "$(cd "$REPO/tools/backends/patches" && ls colmap-*.patch | tr '\n' ' ' | sed 's/ $//')",
  "openmvs_patches": "$(cd "$REPO/tools/backends/patches" && ls openmvs-*.patch | tr '\n' ' ' | sed 's/ $//')",
  "vcpkg_overlay_ports": "$(ls "${VCPKG_OVERLAY_PORTS:-/nonexistent}" 2>/dev/null | tr '\n' ' ' | sed 's/ $//')",
  "llvm_openmp": "${LLVM_OPENMP:+$LLVM_VERSION}",
  "vcpkg": "$VCPKG_VERSION",
  "triplet": "$TRIPLET",
  "built": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "compiler": "$COMPILER",
  "requires_glibc": "$GLIBC",
  "dynamic_libraries": "$DYNAMIC"
}
EOF
cat "$PREFIX/BUILDINFO.json"

# Next to the repo's build output, wherever the work folder is.
mkdir -p "$REPO/build/backends"
ARCHIVE=$REPO/build/backends/ez2d-backends-$OS-$ARCH.tar.gz
TAR_OPTS=()
# GNU tar would read "C:/..." as a remote host.
[ "$OS" = windows ] && TAR_OPTS=(--force-local)
tar ${TAR_OPTS[@]+"${TAR_OPTS[@]}"} -czf "$ARCHIVE" -C "$PREFIX" .
cp "$PREFIX/BUILDINFO.json" "$REPO/build/backends/BUILDINFO-$OS-$ARCH.json"
log "wrote $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
