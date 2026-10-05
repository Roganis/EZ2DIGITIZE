# Phase 1 feasibility harness

Scripts that run COLMAP, OpenMVS and Brush on your photo sets and record
time, memory and quality for each variant. They are throwaway spike tooling,
not part of the app, and run on Linux and macOS only. What to capture and how to judge the results is in
[`docs/feasibility/README.md`](../../docs/feasibility/README.md).

Everything runs through one script:

```sh
uv run --group feasibility tools/feasibility/bench.py <command> --help
```

| Command | What it does |
|---|---|
| `sysinfo` | Show the machine and the tool versions the harness finds |
| `synth OUT` | Render a small synthetic dataset (box on a mat, with masks) to check your setup |
| `frames VIDEO OUT` | Extract the sharpest frames from a video (`--count 120`) |
| `masks IMAGES OUT` | Background-removal masks in COLMAP naming, plus a preview sheet to review |
| `run PLAN` | Run every variant in a plan file; reruns skip what already succeeded |
| `report` | Write `docs/feasibility/results/<machine>.md` and `.json` from all runs |

## Installing the tools

The harness looks for `colmap`, the OpenMVS binaries and `brush_app` on
`PATH`. Override with `--colmap`, `--openmvs-dir`, `--brush`, or the
environment variables `EZ2D_COLMAP`, `EZ2D_OPENMVS_DIR`, `EZ2D_BRUSH`.
`bench.py sysinfo` shows what it found.

Masks use [rembg](https://github.com/danielgatis/rembg) (MIT), which
downloads its model on first use into `~/.rembg`. The default model,
`birefnet-general`, is close to 1 GB and the most accurate;
`birefnet-general-lite` and `isnet-general-use` are smaller and faster.

### Arch Linux (RX 7900 GRE)

```sh
sudo pacman -S --needed ffmpeg vulkan-tools base-devel cmake git \
    boost eigen opencv cgal libjxl
```

OpenMVS also needs `nanoflann`: try `pacman -S nanoflann`, then the AUR, or
build it from source (see "Building OpenMVS").

- **COLMAP:** install the `colmap` package (`pacman -Si colmap`, otherwise
  the AUR). A build with CUDA support is fine; the harness forces CPU mode.
- **GLOMAP** (optional, for `mapper = "global"`): only needed if your COLMAP
  has no `global_mapper` command (`colmap help | grep global`).
- **OpenMVS:** build v2.4.0 from source, as below. Don't use v2.3.0 or older:
  it aborts with "buffer overflow detected" when compiled with
  `_FORTIFY_SOURCE=3`, which Arch's `makepkg` uses.
- **Brush:** download `brush-app-x86_64-unknown-linux-gnu.tar.xz` from the
  [v0.3.0 release](https://github.com/ArthurBrussee/brush/releases/tag/v0.3.0)
  and put `brush_app` on `PATH`. It runs on Vulkan (RADV); no ROCm needed.

### macOS (M1)

```sh
brew install colmap ffmpeg cmake boost eigen opencv cgal jpeg-xl libomp
```

OpenMVS also needs `nanoflann`: try `brew install nanoflann`, or build it
from source (see "Building OpenMVS").

- **OpenMVS:** build from source as below (if a Homebrew formula is missing,
  that library must be built from source too).
- **Brush:** download `brush-app-aarch64-apple-darwin.tar.xz` from the same
  release. It runs on Metal.

### Building OpenMVS (both machines)

```sh
git clone --depth 1 https://github.com/cdcseacave/VCG.git vcglib
git clone --depth 1 --branch v2.4.0 --recurse-submodules \
    https://github.com/cdcseacave/openMVS.git
cmake -S openMVS -B openMVS/build -DCMAKE_BUILD_TYPE=Release \
    -DVCG_ROOT="$PWD/vcglib" -DOpenMVS_USE_CUDA=OFF \
    -DOpenMVS_USE_PYTHON=OFF -DOpenMVS_BUILD_VIEWER=OFF
cmake --build openMVS/build -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
export EZ2D_OPENMVS_DIR="$PWD/openMVS/build/bin"
```

If nanoflann isn't packaged, it is header-only and quick to install locally
before running the OpenMVS `cmake` lines (then add
`-DCMAKE_PREFIX_PATH="$HOME/.local"` to the OpenMVS configure line):

```sh
git clone --depth 1 https://github.com/jlblancoc/nanoflann.git
cmake -S nanoflann -B nanoflann/build -DNANOFLANN_BUILD_EXAMPLES=OFF -DNANOFLANN_BUILD_TESTS=OFF
cmake --install nanoflann/build --prefix "$HOME/.local"
```

v2.4.0 needs an OpenCV with JPEG XL support (newer than Ubuntu 24.04's 4.6);
current Arch and Homebrew OpenCV should be fine. The `master` branch has
moved to vcpkg with many more dependencies; stick to the release tag.

## Checking your setup

```sh
B="uv run --group feasibility tools/feasibility/bench.py"
$B sysinfo
$B synth ~/ez2d-data/synthetic
$B run tools/feasibility/plans/synthetic.toml
$B report
```

All five synthetic runs should succeed. If one fails, the report lists the
failing step with the last lines of its log.

## Plan files

One TOML file per dataset, in `plans/`. Each `[[colmap]]` entry is a
variant; `[[openmvs]]` and `[[brush]]` entries build on a COLMAP variant
through `source`. Unknown keys are rejected, so typos fail early. All keys
with their defaults are in [`ez2d_bench/plan.py`](ez2d_bench/plan.py); the
most useful ones:

| Section | Key | Default | Meaning |
|---|---|---|---|
| colmap | `max_image_size` | 3200 | Longest side for SIFT extraction |
| colmap | `matcher` | `exhaustive` | or `sequential` (video) |
| colmap | `mapper` | `incremental` | or `global` (COLMAP `global_mapper` / GLOMAP) |
| colmap | `masks` | false | Use the dataset's masks for feature extraction |
| colmap | `camera_model` | COLMAP's | e.g. `OPENCV` |
| colmap | `undistort_max_image_size` | full size | Size of the images OpenMVS and Brush get |
| openmvs | `resolution_level` | 1 | Each level halves the image size for densification |
| openmvs | `masks` | false | Undistort the masks and pass them to DensifyPointCloud |
| openmvs | `refine` | false | Run RefineMesh (slow) |
| openmvs | `texture_export` | `ply` | `obj` crashed in OpenMVS 2.3.0; worth retrying on 2.4.0 |
| brush | `args` | `[]` | Passed to `brush_app` as is |
| all | `extra` | `{}` | Extra arguments per command, e.g. `{ DensifyPointCloud = ["--number-views", "8"] }` |

## What gets measured

Each run writes `<results>/<machine>/<dataset>/<kind>/<label>/run.json` and a
log per step. Results default to `~/ez2d-feasibility` because the outputs are
large; only the reports are meant for the repo.

- **Wall and CPU time** per step.
- **Peak RAM:** the step's own peak resident memory. Commands are started
  through a tiny launcher (`ez2d_bench/_launch.py`) because on Linux a child
  inherits its parent's peak-memory counter at fork.
- **Peak VRAM / GTT** (AMD on Linux, from amdgpu sysfs): increase over the
  level before the step. GTT growing means the GPU ran out of VRAM and
  spilled into system memory.
- **Swap:** increase during the step. On the 8 GB M1 this is the first sign
  that a workload doesn't fit.
- **COLMAP:** registered images, number of models, points, reprojection error.
- **OpenMVS:** dense points, mesh faces, texture size.
- **Brush:** splat count and PSNR on held-out photos (every 8th image),
  computed from the renders Brush saves; Brush prints nothing when its output
  isn't a terminal.

VRAM and swap are system-wide, so close other GPU-heavy apps (browsers,
games) while measuring.

Ctrl+C stops the running step and its child processes. Rerunning the same
command continues where it stopped; `--force` reruns everything, and `--only
LABEL ...` limits a run to some variants.

## Measuring accuracy against ground truth

`bench.py eval` scores an EZ2DIGITIZE project the way MVS benchmarks (DTU,
Tanks and Temples) do: it aligns the reconstruction to a ground-truth mesh
(similarity transform from the camera centres, then trimmed ICP), samples
both surfaces, and reports accuracy, completeness, Chamfer distance and
precision / recall / F-score at 0.5, 1 and 2% of the object's size, plus
per-camera position and rotation errors.

```sh
uv run --group feasibility tools/feasibility/bench.py synth ~/ez2d-data/synthetic
# ... make a project from ~/ez2d-data/synthetic/images and run it ...
uv run --group feasibility tools/feasibility/bench.py eval ~/scans/synthetic \
    --gt-mesh ~/ez2d-data/synthetic/ground_truth.ply \
    --gt-cameras ~/ez2d-data/synthetic/ground_truth.json --out eval.json
```

`ground_truth.json` holds `units` and, per image file name, the camera's
`center` and world-to-camera `rotation` (COLMAP convention). The synthetic
scene writes both files; other datasets (DTU scans, rendered Google Scanned
Objects) need a small converter each. Reconstruction far outside the
ground truth's bounding box (background, turntable) is reported but left
out of accuracy.
