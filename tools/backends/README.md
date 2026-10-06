# Backend builds

`build.sh` builds pinned, CPU-only versions of the reconstruction backends
the app drives, and packages them in one archive per platform:

| Backend | Version | License |
|---|---|---|
| COLMAP | 4.2.1 | BSD-3-Clause |
| OpenMVS | v2.4.0 | AGPL-3.0 |

Dependencies are built from source by [vcpkg](https://vcpkg.io)
and linked statically, so the binaries only need the system C/C++ runtime.
COLMAP uses the vcpkg baseline and port patches pinned in its own
repository; OpenMVS has none, so it uses vcpkg release 2026.07.29.

```sh
tools/backends/build.sh     # Linux x86_64, macOS arm64 or Windows x64
```

Output: `build/backends/ez2d-backends-<os>-<arch>.tar.gz` with `bin/`,
`licenses/` (the copyright file of every library linked in) and
`BUILDINFO.json` (versions, triplet, compiler, required glibc, remaining
dynamic libraries). To use it with the benchmark harness:

```sh
mkdir -p ~/ez2d-backends && tar -xzf ez2d-backends-linux-x86_64.tar.gz -C ~/ez2d-backends
export EZ2D_COLMAP=~/ez2d-backends/bin/colmap EZ2D_OPENMVS_DIR=~/ez2d-backends/bin
```

The first build takes hours; vcpkg caches built packages in
`build/backends/vcpkg-cache` (override with `VCPKG_BINARY_CACHE`), so
rebuilds are much faster.

Prerequisites:

- **Linux (Ubuntu/Debian):** `build-essential cmake ninja-build gfortran nasm
  autoconf autoconf-archive automake libtool bison flex pkg-config zip unzip
  curl libgl-dev libglew-dev`. On Arch: `base-devel cmake ninja gcc-fortran
  nasm autoconf-archive bison flex zip unzip libglvnd glew`. COLMAP 4.2.1
  insists on finding OpenGL and GLEW even for a headless build, but doesn't
  link them then; the script fails if the binaries end up depending on them.
- **macOS:** Xcode command line tools, then `brew install cmake ninja nasm
  autoconf autoconf-archive automake libtool pkg-config libomp glew`. Apple's
  compiler has no OpenMP; OpenMVS uses it for many parallel loops and
  silently builds without it, which would make the M1 timings not
  comparable. The script builds against Homebrew's `libomp`, then
  copies it into the archive's `lib/` and points the binaries at it
  (`@executable_path/../lib`), so the archive doesn't need Homebrew; the
  build fails if any Homebrew path is left.
- **Windows:** Visual Studio 2022 (or its Build Tools) with the C++
  workload, and Git for Windows. Run the script from Git Bash with the MSVC
  environment loaded (for example `vcvars64.bat`, then `bash`), and a short
  work folder: `EZ2D_BACKENDS_WORK=C:/ez2d tools/backends/build.sh`, since
  vcpkg's build trees easily pass Windows' 260-character path limit.
  Everything is linked statically, the C runtime included (triplet
  `x64-windows-static-release`). The exception is MSVC's OpenMP runtime,
  which exists only as a DLL: `vcomp140.dll` is copied next to the binaries
  from Visual Studio's redistributable folder. Any other DLL the binaries
  need is copied too, from there or from vcpkg's output (the C++ runtime, if
  `vcomp140.dll` needs it; LAPACK, which vcpkg builds with MinGW's gfortran
  as DLLs even for a static triplet, with the GCC runtime). The build fails
  if a binary still needs a DLL that isn't part of Windows. On Windows COLMAP's GLEW lookup is satisfied by adding
  vcpkg's `glew` to its manifest; GLEW isn't linked.

## CI

The [Backends workflow](../../.github/workflows/backends.yml) runs the build
on Ubuntu 22.04 (for an old glibc baseline), Apple Silicon macOS and
Windows. It then runs the benchmark harness (POSIX only, so not on Windows) on the synthetic scene with the fresh
binaries ([`backends-smoke.toml`](../feasibility/plans/backends-smoke.toml))
and the app's own backend modules on the same scene
(`tests/backends/test_real_pipeline.py`), and uploads the archives as
artifacts for 14 days.

## Results of the first CI builds

First successful build: Backends run 12 (COLMAP 4.2.1, OpenMVS v2.4.0,
vcpkg 2026.07.29).

| | Linux x86_64 (Ubuntu 22.04) | macOS arm64 |
|---|---|---|
| Archive (artifact zip) | 90 MB | 47 MB |
| Needs | glibc ≥ 2.35 | macOS on Apple Silicon |
| Dynamic libraries | libc, libm, libstdc++, libgcc_s, libgomp, libgfortran, libquadmath | system libraries; libomp shipped in `lib/` |
| Harness smoke test | 4/4 | 4/4 |
| App pipeline test | passes (run locally with the archive) | see the workflow |

- `libgfortran` and `libquadmath` come from the BLAS/LAPACK that COLMAP's
  dependencies pull in. Arch ships them in `gcc-libs`, Ubuntu desktop
  installs usually have them; the AppImage (Phase 2) should bundle them.
- COLMAP 4.2.1's `global_mapper` works on the synthetic scene (32/32
  images, like the incremental mapper); OpenMVS 2.4.0 exports OBJ (2.3.0
  crashed).
- The first build takes about 2 hours per platform; with the vcpkg cache,
  about 25 minutes.

## Patches

`patches/openmvs-*.patch` are upstream OpenMVS fixes released after the
pinned version; `build.sh` applies them to the checkout and lists them in
`BUILDINFO.json` (`openmvs_patches`). Each patch's header says what it fixes
and which upstream commit it comes from. Drop a patch when the pin moves
past that commit.

- `openmvs-2.4.0-sample-type.patch`: v2.4.0 samples 8-bit images as if
  they held float colours, so TextureMesh's local seam leveling fills
  the atlas with black blobs and saturated red/green/blue specks (seen on
  the skull turntable set). Fixed upstream in `eeedab7`.

`patches/vcpkg-*.patch` fix ports of the pinned vcpkg release. On Windows,
`build.sh` and `collect_sources.sh` copy the ports they touch, apply them
and use the copies as overlay ports (`prepare_vcpkg_overlays` in
`pins.sh`; `BUILDINFO.json` lists them as `vcpkg_overlay_ports`). Drop a
patch when the vcpkg pin moves past the upstream fix.

- `vcpkg-2026.07.29-gmp-autoconf.patch`: the gmp port builds on Windows
  with MSYS2's autoconf 2.71, pinned as package 2.71-3, which MSYS2 has
  since replaced with 2.71-4 and deleted from every mirror. Upstream vcpkg
  made the same change after the release.

Some sources are also fetched from mirrors.kernel.org before vcpkg asks
for them (`MIRRORED_SOURCES` in `pins.sh`: GMP, MPFR, automake), checked
against the ports' SHA512: GNU's own servers often time out from CI.

## What is turned off, and why

- **CUDA, HIP, GUI, OpenGL** (COLMAP) and **CUDA, viewer, Python**
  (OpenMVS): the app runs them headless on any GPU vendor.
- **OpenCV's default features** (OpenMVS): `build.sh` edits OpenMVS's vcpkg
  manifest to ask for OpenCV without them, keeping only what OpenMVS uses:
  `calib3d` (stereo matching, speckle filter, rectification), Eigen and the
  image formats (JPEG, PNG, TIFF, OpenEXR, JPEG XL). The defaults include the GTK
  GUI backend on Linux, a large GTK/X11 build (which failed in CI on
  at-spi2-core) for windows OpenMVS only opens in debug builds.
- **ONNX** (COLMAP's learned features, ALIKED/LightGlue): would download
  ONNX Runtime at configure time; worth evaluating later (roadmap Phase 7).
- **CGAL** (COLMAP): only used by COLMAP's own meshing, which needs its
  CUDA-only dense stereo anyway. OpenMVS still uses CGAL.
- **Downloads** (COLMAP): fetching vocabulary trees at runtime; not needed.

### To try on the GRE: COLMAP with HIP

COLMAP 4.2.1 has a `HIP_ENABLED` option for AMD GPUs through ROCm. On the
7900 GRE under Linux, it might speed up feature extraction and matching, and
possibly make COLMAP's dense stereo usable, which used to be CUDA-only. That
needs ROCm installed and a separate build with `-DHIP_ENABLED=ON`; it is not
part of these portable builds.

## Licensing

Statically linking means the binaries contain all those libraries.
Distributing them requires shipping `licenses/` (the apps do) and the exact
source of every GPL/LGPL/AGPL component. The pins live in `pins.sh`, shared
by `build.sh` and `collect_sources.sh`, which packs that source without
compiling anything:

```sh
tools/backends/collect_sources.sh   # -> build/backend-sources/ez2d-backends-source-<os>-<arch>.tar
```

The archive holds COLMAP and OpenMVS as tagged, `build.sh`, `pins.sh` and
the patches, vcpkg as pinned, the port scripts of the versions COLMAP's
own baseline selects, and every dependency's source archive as `vcpkg
install --only-downloads` fetches it for the same manifests, features and
triplet; `SOURCES.txt` says what is where. The
[Backend sources workflow](../../.github/workflows/backend-sources.yml)
builds it for Linux and macOS; releases publish it next to the binaries.
