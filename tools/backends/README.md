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
tools/backends/build.sh     # Linux x86_64 or macOS arm64
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

## CI

The [Backends workflow](../../.github/workflows/backends.yml) runs the build
on Ubuntu 22.04 (for an old glibc baseline) and Apple Silicon macOS. It
then runs the benchmark harness on the synthetic scene with the fresh
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
Distributing them (in Phase 6) requires shipping `licenses/` and offering
the exact source of every GPL/LGPL/AGPL component: COLMAP and OpenMVS
tags, the vcpkg commit, and the ports' sources, all pinned here.
