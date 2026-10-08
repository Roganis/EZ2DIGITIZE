# Backend builds

`build.sh` builds pinned, CPU-only versions of the reconstruction backends
the app drives, and packages them in one archive per platform:

| Backend | Version | License |
|---|---|---|
| COLMAP | 4.2.1 | BSD-3-Clause |
| OpenMVS | v2.4.0 | AGPL-3.0 |

Dependencies are built from source by [vcpkg](https://vcpkg.io)
and linked statically, so the binaries only need the system C/C++ runtime,
except ONNX Runtime (MIT): COLMAP runs its learned features (ALIKED,
LightGlue) with it, and its build fetches Microsoft's release library,
pinned by hash in COLMAP's CMake. It ships in `lib/` (Linux, macOS) or
`bin/` (Windows), with its license and third-party notices; on Windows that
library is the release built with CUDA support, which only loads CUDA when
asked to, and runs on the CPU here.
COLMAP uses the vcpkg baseline and port patches pinned in its own
repository; OpenMVS has none, so it uses vcpkg release 2026.07.29.

```sh
tools/backends/build.sh     # Linux x86_64, macOS arm64 or Windows x64
```

Output: `build/backends/ez2d-backends-<os>-<arch>.tar.gz` with `bin/`,
`lib/`, `licenses/` (the copyright file of every library linked in) and
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
- **Windows:** Visual Studio 2022 or newer (or its Build Tools) with the
  C++ workload and its "C++ Clang tools for Windows" (clang-cl; not needed
  with `WINDOWS_COMPILER=msvc`), and Git for Windows. Run the script from Git Bash with the MSVC
  environment loaded (for example `vcvars64.bat`, then `bash`), and a short
  work folder: `EZ2D_BACKENDS_WORK=C:/ez2d tools/backends/build.sh`, since
  vcpkg's build trees easily pass Windows' 260-character path limit.
  Everything is linked statically, the C runtime included (triplet
  `x64-windows-static-release`). The exception is the OpenMP runtime,
  which exists only as a DLL: LLVM's `libomp.dll`, built with the rest (see
  "Windows: clang-cl"), or with `WINDOWS_COMPILER=msvc`, MSVC's
  `vcomp140.dll` from Visual Studio's redistributable folder. Any other DLL
  the binaries need is copied too, from there or from vcpkg's output (the
  C++ runtime; LAPACK, which vcpkg builds with MinGW's gfortran as DLLs
  even for a static triplet, with the GCC runtime). The build fails
  if a binary still needs a DLL that isn't part of Windows. On Windows COLMAP's GLEW lookup is satisfied by adding
  vcpkg's `glew` to its manifest; GLEW isn't linked.

## CI

The [Backends workflow](../../.github/workflows/backends.yml) runs the build
on Ubuntu 22.04 (for an old glibc baseline), Apple Silicon macOS and
Windows. It then runs the benchmark harness (POSIX only, so not on Windows) on the synthetic scene with the fresh
binaries ([`backends-smoke.toml`](../feasibility/plans/backends-smoke.toml))
and the app's own backend modules on the same scene
(`tests/backends/test_real_pipeline.py`), and uploads the archives as
artifacts for 90 days. It runs only when `tools/backends` or the workflow
changes.

The [Backend tests workflow](../../.github/workflows/backend-tests.yml) runs
`tests/backends` on all three platforms without building: it takes the
archives of the newest Backends run (the branch's, then main's), so a change
to the app's side (`src/`) is checked against the real tools in about ten
minutes.

Off main, the Backends workflow's `compare` job then checks that a change
to the Windows build doesn't make reconstructions slower:
[`compare.py`](compare.py) runs the app's pipeline (`ez2d run`) on a
40-photo synthetic scene with this run's build and with main's newest, on
one runner, two rounds each with the builds taking turns, and puts the wall
time of every stage side by side in the run's summary. It also runs
locally, for any builds:

```sh
uv run python tools/backends/compare.py PHOTOS --backend main=PREFIX_A --backend new=PREFIX_B
```

Build time on the CI runners, with the vcpkg cache: Linux and macOS about
25 minutes, Windows about 16 with clang-cl (below). With MSVC, Windows took
1 hour 22 to 1 hour 57, most of it linking OpenMVS. Without the cache,
Windows took over 3 and a half hours. The cache is saved only when the build
added packages to it, since every saved copy takes about 800 MB of the
repository's 10 GB; caches belong to their branch, and a pull request also
reads main's.

## Windows: clang-cl

COLMAP's and OpenMVS's own code is compiled with clang-cl and linked with
lld-link, the LLVM tools that come with Visual Studio; the vcpkg libraries
stay MSVC-built (the same ABI, and the cache still applies).
`WINDOWS_COMPILER=msvc tools/backends/build.sh` builds everything with MSVC
as before.

Why: OpenMVS always turns on MSVC's link-time code generation (`/GL`,
`/LTCG`), and linking its five tools took 55 to 72 minutes on the CI
runner; turning it off made compiling take 2 hours 46 instead (one source
file alone over an hour). LLVM optimises while compiling, as on Linux and
macOS, and ignores `/GL`. On the same runner (Backends run 37669421809):

| Windows build step | MSVC | clang-cl |
|---|---|---|
| COLMAP | 41 min | 13 min |
| OpenMVS | 62 min (57 of them linking) | 3.5 min |
| Whole build | 1 h 44 | 16 min |

Without link-time optimisation the binaries are no slower: the `compare`
job ran the app's pipeline on 40 photos, two rounds per build, and the
clang-cl build took 0.97× MSVC's time overall (run 37669421809, with
Visual Studio's OpenMP runtime) and 1.01× (run 37688019224, with LLVM's
`libomp.dll` as shipped; a slower runner, 9.7 minutes for MSVC's build
against 6.3). Densify, about three quarters of the run, came out at 0.95×
and 1.02×: within the runner's noise. Linux has never had link-time optimisation either (COLMAP
turns it off for GCC; OpenMVS only uses it with MSVC).

What it takes:

- COLMAP picks its Windows settings by compiler ID, which for clang-cl is
  "Clang", not "MSVC": `build.sh` sets `IS_MSVC=TRUE` to give them back
  (and `IPO_ENABLED=OFF`). Flags are written with `-`, since Git Bash turns
  arguments starting with `/` into paths.
- Two patches (see Patches): `openmvs-2.4.0-clang-cl.patch`, without which
  clang-cl doesn't compile OpenMVS, and
  `colmap-4.2.1-poisson-centre-vertex.patch`, for a bug that MSVC's luck and
  the other platforms' `-ffast-math` had hidden.
- OpenMP: LLVM's own runtime, `libomp.dll`, built from the LLVM release that
  matches Visual Studio's clang-cl (`LLVM_VERSION` in `pins.sh`, the source
  archive pinned by hash) in about two minutes, and shipped with its
  license. Visual Studio's build of it for `/openmp:llvm`
  (`libomp140.x86_64.dll`) is only in its `debug_nonredist` folder, not
  ours to ship.

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

`patches/colmap-*.patch` and `patches/openmvs-*.patch` are fixes to the
pinned versions; `build.sh` applies them to the checkouts on every platform
and lists them in `BUILDINFO.json` (`colmap_patches`, `openmvs_patches`).
Each patch's header says what it fixes and, for upstream fixes, which commit
it comes from. Drop a patch when the pin moves past that commit.

- `colmap-4.2.1-poisson-centre-vertex.patch` (ours; the bug is also in
  PoissonRecon master): the Poisson mesher's polygon centre vertices read an
  uninitialised density (`Vertex c; c *= 0;`). `-ffast-math`, which COLMAP
  uses for every compiler but MSVC, folds `x * 0` to 0 and hid it; built
  with clang-cl, the densities came out NaN and the surface trimmer turned
  every vertex cut from them into NaN.
- `openmvs-2.4.0-clang-cl.patch` (ours): lets clang-cl, the Windows compiler,
  compile OpenMVS (SSE sums, resource-compiler defines); MSVC, GCC and
  Apple Clang compile the same code as before.
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
