# Phase 1 findings

Fill this in from the reports in [`results/`](results/) and from looking at
the outputs. The four exit decisions at the end are what Phase 1 delivers.

## Machines

| Machine | Hardware | OS / driver | Report |
|---|---|---|---|
| GRE | RX 7900 GRE 16 GB | Arch Linux, Mesa ... | [results/....md](results/) |
| M1 | Apple M1, 8 GB | macOS ... | [results/....md](results/) |

## Observations per dataset

### 1. Walkaround

- COLMAP: ...
- OpenMVS (which level is good enough? how long on each machine?): ...
- Masks (time saved, quality difference): ...
- Brush (PSNR, how it looks): ...

### 2. Turntable

- Without masks: ...
- With masks: ...

### 3. Flipped (two-sided)

- One model or two? ...
- Mesh of the bottom side: ...

### 4. Video

- Sequential vs exhaustive matching: ...
- Mesh quality compared with dataset 1: ...

### 5. Hard set

- Where it failed and whether the failure was detectable from the numbers: ...

### Viewer (`tools/spikes/viewer`)

- Splat from Brush: load time, fps, smooth to rotate? (GRE / M1): ...
- Textured mesh and dense cloud from OpenMVS: GRE, packaged AppImage: 1.97M-face
  textured mesh loads in 1.5 s and renders at 60 fps (likely vsync-capped) on
  radeonsi through ANGLE. M1: ...
- 3M-splat stress file: ...
- Memory of the QtWebEngine processes: ...

### Masking

- Model used, time per image, images where masks were wrong: ...

## Exit decisions

| Question | Decision | Evidence |
|---|---|---|
| (a) OpenMVS on CPU fast enough for small objects? Default resolution level? | | |
| (b) Masking model and runtime | | |
| (c) Python packaging acceptable? | Likely yes: AppImage 144 MB without viewer, 285 MB with, 0.3-0.4 s to a window ([spike](../../tools/spikes/packaging/README.md)); size is dominated by Brush and QtWebEngine, not Python. Still to check: the macOS `.app` on the M1 | |
| (d) Viewer approach | Candidate: three.js + Spark in QWebEngineView ([spike](../../tools/spikes/viewer/README.md)); on the GRE a 2M-face textured mesh runs at 60 fps on the GPU from the AppImage. Still to measure: splats, the M1, memory | |

## Notes from validating the harness

Found while testing the scripts in a CPU-only Ubuntu 24.04 container with
COLMAP 3.9.1, OpenMVS v2.3.0 built from source, and Brush 0.3.0 on Mesa's
software Vulkan driver, using the synthetic dataset. Timings from that
container mean nothing for real hardware.

- **OpenMVS ≤ v2.3.0 aborts in DensifyPointCloud** with "buffer overflow
  detected" when compiled with `_FORTIFY_SOURCE=3` (Ubuntu's compiler
  default, and Arch's `makepkg` flags). Cause: `Util::formatTime` passes the
  full buffer size to `snprintf` at an offset. Fixed in v2.4.0.
- **OpenMVS v2.3.0 TextureMesh crashes in `Mesh::SaveOBJ`** with
  `--export-type obj`; PLY export works. The plans default to PLY; retry OBJ
  on v2.4.0.
- **OpenMVS v2.4.0 needs OpenCV with JPEG XL** (it failed to compile against
  Ubuntu 24.04's OpenCV 4.6). The `master` branch now builds through vcpkg
  with extra dependencies (PoseLib, TinyEXIF, ...).
- **Masks cut OpenMVS densification time** on the synthetic scene from 271 s
  to 47 s at level 2, because the textured mat no longer gets reconstructed.
  Expect a similar effect on real captures with a patterned mat.
- **Brush 0.3.0 prints no progress** when its output isn't a terminal, so
  quality comes from its saved eval renders (`--eval-split-every`,
  `--eval-save-to-disk`).
- On Linux a process inherits its parent's peak-memory counter, so the
  harness starts every command through a small launcher to measure the
  command's own peak RAM.
