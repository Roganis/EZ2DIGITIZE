# Architecture

This describes the intended design. Sections marked *(planned)* are not
implemented yet; update this file when they are.

## Layers

```
ez2digitize/
  app.py        GUI entry point (creates QApplication, main window)
  ui/           PySide6 widgets, viewers, Qt adapters for core objects
  core/         headless logic: project model, capture bundles, stages
                and manifests, process runner
  backends/     one module per external tool: finds it, checks its version,
                builds stage command lines, parses progress
  pipeline.py   chains the backend stages over a project (sparse, dense)
  cli.py        headless command line (`ez2d`) over the same pipeline
```

Dependencies point one way: `core` <- `backends` <- `pipeline` <- `cli`/`ui`.
Nothing but `ui` and `app` imports Qt. The UI observes the pipeline through
Qt adapters (a `QObject` that forwards pipeline events as signals). The CLI
uses the same code, which is how regression datasets run in CI and on the
reference machines.

## Project folder (`core/project.py`, `core/capture.py`)

```
my-scan/
  project.json        schema_version, name, preset, settings
  captures/
    20261005-203200/  one capture bundle per import
      capture.json    source, device, and name/size/SHA-256 of every file,
                      what the photo checks learned about it, whether it is
                      left out
      IMG_0001.jpg    original files, copied byte for byte
  masks/              one mask per image (same file stem), optional
  stages/
    features/
      stage.json      manifest (see below)
      log.txt
      ...outputs
    matching/
    ...
  exports/            user-facing outputs (OBJ, GLB, STL, PLY)
```

- `project.json` changes bump `SCHEMA_VERSION` and add a step to
  `MIGRATIONS`. Opening an older project migrates it and keeps the old file
  as `project.json.v<N>.bak`; a project from a newer app version is refused.
- A bundle is assembled in `captures/.importing-<id>/` and renamed when
  complete, so an interrupted import never appears as a bundle. Files with
  the same name get a numeric suffix; `original_name` keeps the name they
  arrived with. `CaptureBundle.verify()` re-hashes the files.
- A file can be left out of the reconstruction (`excluded` in capture.json,
  `CaptureBundle.set_excluded`) without touching it, and brought back.
  `bundle.images` and `bundle.videos` are the files in use, and a bundle's
  cache fingerprint covers only those, so leaving a photo out re-runs the
  pipeline from feature extraction and bringing it back reuses the old run.
- JSON files are written atomically (temporary file, fsync, rename).

## Video import (`video.py`, `backends/ffmpeg.py`)

A video becomes a capture bundle (`source: "video"`) holding the original
video, untouched, and the chosen frames as its images (`frame_0001.jpg`,
...), so the pipeline reads it like any photo capture. Folder imports skip
videos; each video is imported on its own (`ez2d import P clip.mp4`, or
Import video in the GUI).

- The user picks how many frames to keep (default 100); that sets the kept
  rate (at most 5 a second). The video is cut into that many windows and
  FFmpeg extracts 4 candidates per window (`-vf fps=`, JPEG quality 2,
  rotation applied); the sharpest of each window, by the photo checks'
  score, is kept and the others deleted. On a test clip where two of every
  three frames are motion-blurred, every window kept its sharp frame.
- Each frame's capture.json entry records the time it came from and its
  score (`metadata["video"]`); `source_info` records the FFmpeg version and
  rates. The bundle is assembled like a folder import (hidden staging
  folder, renamed when complete), so a failed or cancelled import leaves
  nothing.
- FFmpeg runs through the process runner (log, progress from
  `-progress pipe:1`, cancel). It is the system's FFmpeg, 5.0 or newer,
  found through the settings, `EZ2D_FFMPEG` or `PATH`; it isn't bundled or
  pinned yet (Phase 6).
- The photo checks don't ask video frames for an EXIF focal length.

## Diagnostics and hardware (`diagnostics.py`, `core/hardware.py`)

- `write_diagnostics` zips what a bug report needs: project.json, every
  capture.json, stage.json and export.json, each stage's logs (the last
  2 MB of each), the photo checks' findings, the tools found and system
  information. Never photos, videos or models. Logs hold file paths, which
  the GUI and CLI point out.
- `detect_gpus` lists GPUs from `vulkaninfo --summary` (name, type,
  driver: the Mesa version on Linux), with VRAM from sysfs for amdgpu, or
  from `system_profiler` on macOS (Apple GPUs share system memory).
  Software renderers (llvmpipe) are flagged. `ez2d check` prints them.

## Photo checks (`core/photos.py`)

Each photo is inspected once, on import or when a project from an older
version is shown: pixel size, EXIF make, model, lens and focal length, and a
sharpness score (variance of the Laplacian at 1024 px, decoded at a reduced
JPEG scale; about 25 ms a photo on 4 threads). The result is stored in the
file's capture.json entry (`metadata["photo"]`, with a version so a changed
inspection runs again). The checks only compare these numbers:

| Finding | When |
|---|---|
| can't be read (error) | Pillow can't decode it |
| different size | not the size of most photos in its capture (screenshots, collages, another camera) |
| low resolution | under 1000 px on the short side |
| no focal length | none in EXIF, for the whole capture or only some photos |
| several cameras | more than one body, lens or zoom setting (focal lengths within 5 % count as one) in a capture, which COLMAP calibrates as one camera |
| blurry | sharpness under 0.35 of the capture's median (5 photos or more) |
| imported twice | identical files |
| few photos | under 20 in the project |

Findings are advice; the user leaves photos out from the GUI's photo checks
tab or with `ez2d photos --exclude`.

## Stage manifest (`core/stage.py`)

Each stage folder has a `stage.json`:

- stage name, a run id (new on every run) and status (succeeded, failed,
  cancelled)
- backend name, exact version and build: the sha256 of the executable
  that ran, since a patched or rebuilt binary of the same version can give
  different results (hashed once per process; content, not path, so the
  AppImage's changing mount point doesn't matter)
- full command line (argument list)
- parameters after preset resolution
- input fingerprints: `sha256:` for a file, `capture:` for a bundle (from
  the hashes already in its `capture.json`), `tree:` for a folder such as
  the masks, `run:<run id>` for an earlier stage
- start/end time, wall and CPU time, exit code, peak memory where available
- host: OS, release, architecture, app version. *(planned)* GPU driver
  version (Mesa version on Linux), with the first GPU backend.

The cache key hashes the stage name, backend, parameters and inputs, but
not the command line, which holds absolute paths (moving a project must
not invalidate it). A stage is skipped when its manifest says it succeeded
with the same key; otherwise its folder is emptied and it runs again in
that folder. Because later stages reference earlier ones by run id,
re-running a stage invalidates every stage after it without hashing its
outputs.

## Process runner (`core/runner.py`)

- Built on `subprocess.Popen` with an argument list; no shell.
- Each backend process starts in its own process group
  (`start_new_session=True`) so cancel kills the whole tree: `SIGTERM`,
  then `SIGKILL` after a grace period. Anything a finished process leaves
  behind holding its output open is killed too. POSIX only for now;
  Windows will need a job object.
- stdout and stderr are merged and split into lines (also at the lone `\r`
  of redrawn progress lines), written to the stage's `log.txt`, and passed
  line by line to the backend module's progress parser. OpenMVS buffers its
  console output when it goes to a pipe, so its stages run under a
  pseudo-terminal (`use_pty`).
- The runner emits plain Python events (`Started`, `Output`, `Progress`)
  on the calling thread, in order. The UI runs it on a worker thread and
  wraps the events in Qt signals; the CLI prints them.
- Peak memory: `ru_maxrss` from `wait4`. On Linux a child inherits its
  parent's high-water mark, so when the value isn't above the app's own
  peak the runner uses `VmHWM` samples taken every 0.5 s instead.
- *(planned)* Only one heavy stage runs at a time: the 8 GB M1 cannot fit
  two. That is the pipeline scheduler's job, not the runner's.

## Backend modules (`backends/`)

- `brush.py` trains Gaussian splats (Brush 0.3.0, GPU via Vulkan/Metal) on
  the undistort stage's output, through a `dataset/` of relative symlinks
  in the splat stage folder; it runs under a pty (progress only on a
  terminal). GPU stages (`StageSpec.gpu`) record the GPU and driver in
  their manifest. `pipeline.run_splat` reuses the camera stages and
  refuses software renderers unless allowed.

- `common.find_tool` looks for an executable in this order: an explicit
  path from settings, an `EZ2D_*` environment variable, the bundle's
  `backends/bin`, then `PATH`. A configured path that is wrong is reported,
  never replaced by another copy.
- `locate()` reads the version (`colmap help`, the OpenMVS banner) and
  compares it with the pinned one; other versions are flagged
  `supported=False`. Option names follow the pinned version only.
- Stage builders return a `StageSpec`; the pipeline runs it with
  `run_stage`. Parameters are the option dataclasses, inputs are
  fingerprints of capture bundles, masks and earlier stages.
- COLMAP reads images straight from `captures/` through an image list, named
  `<capture id>/<file>`; masks are `masks/<capture id>/<file>.png`, COLMAP's
  own convention. Intrinsics are grouped per capture bundle by default.
- Matching works on a copy of the features database, so no stage modifies
  another stage's output.
- OpenMVS tools run with their stage folder as working folder. Scene files
  store image paths relative to that folder, and all stage folders are
  siblings, so those paths resolve the same from every stage.
- `tests/backends/test_real_pipeline.py` runs the whole mesh path on the
  synthetic scene; the Backends workflow runs it against the fresh builds.

## Export (`export.py`, `core/meshio.py`)

Formats: `obj`, `glb` (textured), `ply` (OpenMVS's own), `stl` and `3mf`
(geometry for printing, with a watertightness check recorded in
export.json and reported as a notice), `points` (the dense point cloud).
A Mesh size setting (`TextureOptions.target_faces`) has TextureMesh
simplify the mesh before texturing, so the texture keeps its detail.

Exports are stood upright by default (`orientation.py`): the photos' image
"down" directions, turned to world coordinates with each registered pose
of the undistorted model and corrected for EXIF rotation (photo checks
record it), average to gravity. The mesh and point cloud are rotated so
up is +Y, centred on the vertical axis and put on the ground (Z-up for STL
and 3MF); export.json records the transform. If the directions disagree
(mean shorter than 0.5) the model's own frame is kept.

- The texture step always writes OpenMVS's textured PLY. Exporting
  converts it in-process (architecture rule 1 allows mesh export there) into
  `exports/<timestamp>/`: `obj/<name>.obj` + `.mtl` + texture images,
  `<name>.glb` with the textures embedded, and/or a copy of the PLY, plus
  `export.json` (formats, the texture run it came from, counts).
- The pipeline exports after texturing (OBJ and GLB by default). An export
  of the same texture run in the same formats is reused, not repeated.
- `core/meshio.py` is standard library only: `struct` reads the PLY in one
  pass, and glTF vertices are split only on texture seams (a per-vertex
  fast path, a dict for seam corners). A 2-million-face mesh converts in
  about 15 s per format in a CI-class container, at 300-400 MB peak.
- GLB materials are unlit (`KHR_materials_unlit`): photogrammetry
  textures already contain the lighting. GLBs pass the Khronos glTF
  validator without errors or warnings.
- *(planned, Phase 4)* Scale, orientation and cleanup before export; STL
  and 3MF for printing.

## GUI (`ui/`)

- `PipelineRunner` (`ui/pipeline_runner.py`) is the Qt adapter: it runs a
  pipeline function on a `QThread` and re-emits its events as signals
  (`stage_started`, `progress`, `output`, `notice`, `stage_finished`, then
  one of `succeeded`/`failed`/`cancelled`, then `running_changed(False)`).
  Output lines are batched and progress coalesced to at most every 0.1 s,
  so a chatty backend can't flood the event loop.
- `ProjectPage` shows one project: captures, a few settings (detail level,
  mesh format, refine, masks), Run/Cancel, a list of steps with their
  state, overall progress, the tools' output, and on failure the end of the
  failed step's log with a button to open the full log.
- `VideoImporter` (`ui/video_import.py`) runs a video import on a thread;
  the page shows its progress and its Cancel button stops it.
- Settings: a Quality preset (`presets.py`: fast, balanced, high as full
  `MeshSettings`), saved in `project.json`'s `preset`; an Advanced box
  overrides the dense detail level and refinement and lists the values a
  run will use (`presets.describe`). The CLI has `--quality` plus the same
  single-value overrides.
- `PhotoChecks` (`ui/photo_checks.py`), in a tab beside the log: inspects
  new photos on a thread, then lists findings with their photos; unchecking
  a photo leaves it out, a "Left out" group brings photos back.
- `MainWindow` switches between a welcome page and the project page, and
  asks before closing or quitting during a run (which cancels it).
  `BackendsDialog` stores the COLMAP and OpenMVS locations in `QSettings`;
  empty fields fall back to the usual search.
- Tests drive the real pipeline through the GUI with fake backends
  (`tests/conftest.py`), offscreen.

## Pipeline (`pipeline.py`)

Mesh path (Phase 2); the stages in brackets are done:

```
import -> checks -> masks -> [features -> matching -> mapping -> undistort
       -> mask-undistort]
       -> crop box (user) -> [mvs-import -> densify -> mesh -> (refine) -> texture
       -> export]
```

- `run_sparse` and `run_dense` are the two halves around the crop box;
  `run_mesh` runs both. Each stage goes through `run_stage`, so an unchanged
  stage is reused and `force_from` re-runs a stage and everything after it.
- Matching is exhaustive, except for more than 200 images that all come
  from videos, which are matched sequentially (each frame with the next 10).
  Exhaustive matching closes the loop of an orbit, which sequential matching
  can't without a vocabulary tree; 200 images take about 3 minutes.
  The best model is the one with the most registered images; a `Notice`
  event reports split models and low registration.
- Only one pipeline runs per process (`PipelineBusy` otherwise): the 8 GB
  M1 can't fit two reconstructions.
- Errors: `StageFailed` carries the manifest, log path and last lines of the
  log; `PipelineCancelled` after a cancel; `PipelineError` for anything that
  stops the run before or between stages (no captures, no model).
  `diagnosis.explain` matches a failed stage's exit status and log tail
  against known signatures (out of memory, including the OOM killer's
  SIGKILL; disk full; no initial pair or empty pose graph; unreadable
  images; no dense points; empty mesh; crashes and illegal instructions)
  and `StageFailed`'s message leads with the explanation and what to try.
- Masks reach OpenMVS through the `mask-undistort` stage (after
  `undistort`, only when the project has masks). COLMAP's
  `image_undistorter_standalone` warps them with each image's camera and the
  same size limit as `image_undistorter`, so they line up with the
  undistorted photos (on the synthetic scene: IoU 0.98 with the originals,
  the difference being a one-pixel interpolated border that keeps a little
  more). The stage names them `<stem>.mask.png` as OpenMVS's `--mask-path`
  wants and writes a white mask for images without one, since OpenMVS
  needs one per image. Two images with the same file name in different
  captures can't be told apart there; the pipeline then densifies without
  masks and says so. On the synthetic scene, masked densify takes 61 s
  instead of 236 s.

Splat path (Phase 3) branches after `undistort`:

```
undistort -> Brush training -> PLY export
```
