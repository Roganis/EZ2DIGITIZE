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

- Splats: GRE, packaged AppImage: a 3.49M-splat scene loads in 4.7 s and
  renders at 60 fps with a still camera; to re-measure while orbiting. M1: ...
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
| (d) Viewer approach | Candidate: three.js + Spark in QWebEngineView ([spike](../../tools/spikes/viewer/README.md)); on the GRE a 2M-face textured mesh and a 3.5M-splat scene both run at 60 fps on the GPU from the AppImage (still camera). Still to measure: orbiting, the M1, memory | |

## Pinned backend builds (CI)

From the first successful Backends run (COLMAP 4.2.1, OpenMVS v2.4.0, CPU
only, synthetic scene; CI timings, not the reference machines):

- COLMAP 4.2.1 `global_mapper`: 32/32 images, 10 s total, same as the
  incremental mapper (8 s). Worth comparing on real captures.
- OpenMVS 2.4.0: masked densify at level 2 in 37 s, OBJ export works.
- The app's own pipeline (`tests/backends/test_real_pipeline.py`) runs the
  whole mesh path on these binaries, masks and export included.
- Linux binaries need glibc ≥ 2.35 and libgfortran/libquadmath/libgomp
  next to the C/C++ runtime (see `tools/backends/README.md`).

## First real capture through the app (GRE, CI AppImage)

Skull, turntable, strong lights, no background, dotted (62 photos, Canon
EOS 1100D/T3, 39 mm, 4272x2848; [test set](https://gitlab.com/photogrammetry-test-sets/skull-turntable-strong-lights-no-background-dotted),
CC BY 4.0).

- **Crash:** the run took the desktop session down (swap full, terminal
  killed). Reproduced in a 4-thread container: COLMAP feature extraction
  peaked at 7.8 GB, about 2 GB per thread at 3200 px, and COLMAP uses one
  thread per core. Fixed by capping feature threads to the free memory.
- **Mapping:** the incremental mapper split the set into 9 partial models
  (largest 25 images) with focal lengths of 8,500-11,500 px against the
  EXIF-based 7,248 px and large distortion. COLMAP 4.2.1's global mapper
  placed all 62 images in one model, focal 7,101 px, 14,849 points, 0.50 px
  mean reprojection error, in 69 s instead of 346 s. Now the default.
  The incremental mapper with distortion refinement off
  (`--Mapper.ba_refine_extra_params 0`) also placed all 62 images: refining
  the distortion from a narrow-angle lens is what broke it apart.
- **The GRE run** (31 GiB RAM, fixed AppImage, Medium detail = level 1,
  refine on, 63 images including `Preview.jpg`), against the 4-thread
  container (global mapper, no refine):

  | Step | GRE | Container | |
  |---|---|---|---|
  | Features | 55 s, 13.6 GB | 339 s, 7.8 GB | thread cap kept it in RAM |
  | Matching | 17 s | 71 s | |
  | Mapping (global) | 10 s | 25 s | 62/62 placed |
  | Densify | 498 s, 2.7 GB | 2042 s, 2.7 GB | the bottleneck |
  | Mesh | 39 s, 1.9 GB | 96 s, 1.9 GB | |
  | Refine | 248 s, 4.8 GB | – | |
  | Texture | 36 s, 3.5 GB | | |

  About 15 minutes end to end on the GRE with refine, 11 without: CPU-only
  OpenMVS at level 1 is usable for small objects (decision (a): leaning
  yes, level 1 default). Densify is where a GPU path would pay off.
- Other steps (container, 4 threads): matching 73 s / 210 MB, mapping
  (incremental) 346 s / 162 MB, undistortion 5 s / 270 MB.
- The folder also holds `Preview.jpg`, a collage of the set, which folder
  import took as a 63rd photo: photo checks (Phase 2) should flag images
  whose size or camera differ from the rest.

## Accuracy on the synthetic scene (`bench.py eval`)

Masked run, level 2, local COLMAP 3.9.1 / OpenMVS 2.3.0, against the exact
box (diagonal 1.22 scene units):

- Cameras: 32/32, median rotation error 0.10°, median position error 0.3%
  of the object's size.
- Surface: median accuracy 0.2% of the diagonal; completeness about 100%
  at 1%. Precision is lower (83-86%): about 15% of the reconstructed area
  lies more than 2% away from the box, so the mean accuracy is 8x the
  median. To look into (remains of the mat at the base?).

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
