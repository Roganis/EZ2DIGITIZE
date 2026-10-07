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
                      left out; `flipped` for the turned-over side of a
                      two-sided scan
      IMG_0001.jpg    original files, copied byte for byte
      x.motion.json   optional: the motion recorded with them
                      (ez2digitize.motion_log)
  masks/              optional; masks/<capture id>/<file>.png in use,
                      .../dropped/ the ones dropped in review, auto.json
                      which are automatic (ez2digitize.masks)
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
- Two-sided scans (`sides.py`): captures marked `flipped` are the
  object turned over. They join the others through masks only (the object
  moved, the table didn't), so the pipeline warns when photos have no mask
  and reports after mapping how many photos of each side the model holds.
  The up direction for export and the coverage check comes from the first
  side's photos only (`estimate_up(only=...)`).
- A file can be left out of the reconstruction (`excluded` in capture.json,
  `CaptureBundle.set_excluded`) without touching it, and brought back.
  `bundle.images` and `bundle.videos` are the files in use, and a bundle's
  cache fingerprint covers only those, so leaving a photo out re-runs the
  pipeline from feature extraction and bringing it back reuses the old run.
- capture.json has its own `schema_version` (2 since motion logs, a file
  kind version 1 readers would reject) and `MIGRATIONS` in `core.capture`;
  older bundles load migrated and are written back in the new version.
- Motion logs (`motion_log.py`): a capture app records the phone's motion
  in `<name>.motion.json` (gyroscope, accelerometer, gravity and tracked
  poses on the camera's clock, and when each photo was taken). It is kept
  in the bundle as a file of kind "motion", and when the bundle is
  assembled each photo it lists gets its `metadata["motion"]` entry, as
  video frames get theirs from GPMF or CAMM. A video's log
  (`<video stem>.motion.json` next to it) replaces its own track. Poses
  from a log that says `metric` are marked so, for plugins and the scale.
- JSON files are written atomically (temporary file, fsync, rename).

## Phone upload (`upload.py`, `ui/phone_upload.py`)

`UploadSession` serves a one-page upload site (stdlib `http.server`) on
the LAN address while the dialog is open. The URL carries a random token
(compared in constant time; every other path is a 404), clients outside
private ranges get a 403. The page sends each file in 4 MB PUTs that must
start where the stored part ends (a 409 tells it where to resume), so a
Wi-Fi drop costs at most one chunk. File names are reduced to their last
component and must be photos, videos or motion logs. `finish` moves the
complete photos into a capture bundle (`source: "upload"`, the phone's user
agent as device) through `assemble_bundle`, with a motion log if one came;
each video, with the log named after it, goes to a hidden
`captures/.received-videos-*` folder and is handed back (`Received`) for
the caller to import as frames (`video.import_video`, one after another in
the GUI), then deleted. `close` deletes what wasn't imported.

A capture app talks to the same server: `GET api` says which API version
(`API_VERSION`), limits and file names it accepts, and `POST capture`
records the source (e.g. "android"), device, app and whether the object
was turned over, used for the bundles instead of the user agent.

## Watch folder (`watch.py`, `ui/watch_dialog.py`, `ez2d watch`)

For a phone that syncs its photos to a folder. `FolderWatch` polls it (no
file system notifications: plain polling also works on synced and network
folders) down to 4 levels, skipping hidden folders such as Syncthing's
`.stversions`. New photos are those not there when watching started (sync
tools keep the capture time, so modification times can't tell), or with
"since", also those modified after a given time. A photo has arrived when
its size and modification time are unchanged for 5 s; a sync tool's
temporary file for a photo (`.syncthing.IMG_1.jpg.tmp`, `.IMG_1.jpg.icloud`,
`IMG_1.jpg.part`...) counts as arriving until renamed, or until unchanged
for the settle time (abandoned). The set has settled when nothing new or
changing was seen for 30 s: the dialog says so and leaves Import to the
user; `ez2d watch` imports then. The bundle has `source: "watch"` and the
folder in `source_info`; photos already in the project (same size and
SHA-256) are left out. Videos are counted and left to Import video.

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

## Coverage (`coverage.py`)

After camera placement, `analyse` takes the model's cameras: the object is
the point closest to all viewing axes (least squares reweighted by each
axis's miss, so a few cameras looking elsewhere don't pull it away),
cameras over 5 median distances from it are counted as misplaced and left
out, and with "up" from `orientation` each camera gets an angle around the
object and a height angle. Findings, emitted as pipeline notices: a gap
over 90° around the object, every photo within 15° of height, misplaced
photos, and a camera that didn't move (all views within 10° of their mean;
scale-free, unlike positions). On the skull: 23° largest gap, heights from
-28° to 44°; two video frames misplaced.

`rings` lays the same out for the 3D view, in the upright frame of
`upright.rotation` (so a corrected orientation moves the rings with the
model): the cameras sorted by height angle and split where it jumps by more
than 12° (fewer than 4 cameras at a height join the nearest ring: strays,
not a ring), each ring at its cameras' median height and distance, and its
gaps from 35° (two or three photos missing at 10-15° steps). `views` adds
it to the camera placement's JSON with a flag per camera, "far" (misplaced)
or "weak" (`weak_photos`, from the matching database); the page draws each
ring as a circle with its gaps shaded and labelled, orange, red over 90°,
and colours the flagged cameras. `describe` is the summary above the view.

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

## Mesh from splats (`splat_mesh.py`)

A second mesh path after splat training (`MeshSettings.splat_mesh`, the
`splat-mesh` stage). The prepare step turns the splats into oriented
points (in-process, numpy): opacity at least 0.5, the largest 1% left out,
the crop box applied, each normal along the splat's shortest axis, turned
towards the nearest camera. The stage runs COLMAP's `poisson_mesher`
(screened Poisson, with colours; depth by quality, trim 5, gentler than
COLMAP's 10, since splats are sparser than dense MVS points). The result
is a vertex-coloured PLY, exported as `<name>_splat_mesh.glb` (placed like
the other exports) and `.ply`, and shown in the 3D view as "Mesh from
splats". The 2DGS-style methods that train surface-aligned splats are
non-commercial (Inria) or CUDA-only, so they are left to plugins.

## Plugins (`plugins.py`, `ui/plugins_dialog.py`, `ez2d plugins`)

Backends the user installs: tools that can't be bundled (non-commercial
code or weights) or that the user prefers. See [PLUGINS.md](PLUGINS.md)
for the format.

- A plugin is a folder with `ez2d-plugin.toml`: the slot it fills, a
  command with placeholders, and a `[[license]]` per part (code and weights
  separately). It lives in the user's data folder (`plugins_dir()`). It
  runs like the bundled backends, through `run_stage` (manifest, cache,
  cancellation, log); it is never imported.
- Slots are contracts on stage folders. `poses` is the `mapping` stage: the
  photos as COLMAP sees them in, a binary COLMAP model in `sparse/0` out.
  Undistortion and everything after it are unchanged. `splats` is the
  `splat` stage: Brush's dataset in, `splat.ply` out.
- A `poses` plugin's placement can be refined by COLMAP
  (`MeshSettings.refine_poses`, `backends.colmap_refine`): the plugin then
  runs as the `poses` stage, followed by COLMAP's `features`, `matching`
  of only the pairs its poses suggest (`matches_importer`), `triangulation`
  with its poses held (`point_triangulator`), `pose-check` (photos with
  almost no points leave the model, `image_filterer`) and `mapping`
  (COLMAP's mapper continuing from that model: it places the left-out
  photos and refines everything). `point_triangulator` checks the model's
  cameras, rigs and frames against the database's, so the model it starts
  from is rebuilt from the database with the plugin's poses.
- The backend in the manifest is `plugin:<id>` with the plugin's version,
  and its build is the program's hash. The manifest file's hash is a
  parameter, since a plugin run as `python3 run.py` has the interpreter as
  its program.
- `plugins.json` records the plugin chosen for each slot, and the hash of
  the license texts the user accepted. A plugin can only be chosen once
  accepted. If the texts change, it must be accepted again. A chosen
  plugin that can't be used stops `Tools.locate` with a `PluginError`
  rather than falling back, which would quietly change the result.
- `Tools` carries the chosen plugins (`poses`, `splats`); the pipeline
  skips features and matching for a camera placement plugin, and announces
  each plugin run with its licenses as a Notice. The matching database
  feeds coverage only if the current mapping came from it
  (`colmap.matched_database`).
- From a packaged app, a plugin's process gets the user's library path back
  (PyInstaller puts the bundle's first), so the plugin's own Python and
  libraries load.

## Export (`export.py`, `core/meshio.py`)

Formats: `obj`, `glb` (textured), `ply` (OpenMVS's own), `stl` and `3mf`
(geometry for printing, with a watertightness check recorded in
export.json and reported as a notice), `points` (the dense point cloud).
Splats (`export_splat`, `core/splats.py`): Brush's PLY as it is, and SPZ
version 2 (positions as 24-bit fixed point, the rest quantised to bytes,
gzipped: about a tenth of the PLY). The SPZ is stood upright: positions and
rotations turned, and the view-dependent colour (SH degrees 1 to 3) turned
by a per-degree matrix fitted by least squares on sample directions; it is
centred and grounded by the camera placement's sparse points (2nd-98th
percentiles), since trained splats can have floaters far out. Written from
the format's description (Niantic's spz, MIT), and decoded identically by
its reference reader.
A Mesh size setting (`TextureOptions.target_faces`) has TextureMesh
simplify the mesh before texturing, so the texture keeps its detail.

Exports are stood upright by default (`orientation.py`): the photos' image
"down" directions, turned to world coordinates with each registered pose
of the undistorted model and corrected for EXIF rotation (photo checks
record it), average to gravity. The mesh and point cloud are rotated so
up is +Y, centred on the vertical axis and put on the ground (Z-up for STL
and 3MF); export.json records the transform. If the directions disagree
(mean shorter than 0.5) the model's own frame is kept.

The user can correct it (`upright.py`): a `base` rotation (levelled from
three picked points, the plane's normal on the cameras' side, or tipped by
quarter turns about the horizontal axes as seen) and a `turn` about the
vertical, stored in project.json `settings["orientation"]` with the mapping
run id, stale after a new camera placement like the crop box.
`upright.rotation` (the correction, else the estimate) is the one rotation
the viewer, the export and the mesh view use; export.json records it as
`upright`, and it is part of what decides whether an earlier export (or
its GLB, for the mesh view) can be reused. Changing it re-fits a crop box
level around the old one (`crop.relevelled`); the scale is in
reconstruction coordinates and stays.

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
- Scale (`scale.py`): with a scale set, the export's `Placement` gets a
  factor after its rotation and offset: millimetres for STL and 3MF,
  metres for OBJ, GLB and the point cloud (glTF's unit). `export.json`
  records `scale_mm_per_unit` and the units, and the factor is part of
  what decides whether an earlier export can be reused.

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
- Matching (`_matching`) is exhaustive up to 200 images, which take about
  3 minutes; pairs grow with the square. Beyond that, if every image has a
  recorded pose from one recording (`motion.known_poses`, CAMM's 6DoF
  samples), the pairs come from the poses (`_pose_guided_pairs`:
  `colmap.nearby_pairs`, the 20 nearest looking within 60°, plus the next 5
  frames; run by `colmap.match_listed`, `matches_importer`, like the
  refinement of a plugin's poses). Otherwise: video frames
  sequentially, with loop detection by COLMAP's vocabulary tree; photos by
  their EXIF GPS position if 90% have one (`spatial`); else by image
  retrieval with the vocabulary tree (`vocab_tree`); else sequentially in
  name order, with a notice to take them in order. The tree is COLMAP's
  own pinned file (`colmap.VOCAB_TREE`, the SIFT tree COLMAP 4.2.1 would
  download itself; our builds have downloads off), fetched once into the
  user's cache by `core.download`, like the masking model. The matching
  stage records the tree by its hash, not its path.
- Features: SIFT, or ALIKED matched with LightGlue (`FeatureOptions.kind`,
  `MatchOptions.features`), which COLMAP runs with ONNX Runtime; the
  backend builds now enable ONNX and ship its library. The models are
  `colmap.Pinned` files like the vocabulary trees (COLMAP's own URLs and
  hashes, fetched once by `pipeline._fetch`), recorded in the manifests by
  hash. Each kind of feature has its own vocabulary tree.
- The subject (`ez2digitize.subject`, in the project's settings): an
  object, or a room or outdoor scene. A scene's preset uses no masks and
  meshes with OpenMVS's free-space support (`--free-space-support`, for
  weakly supported surfaces such as plain walls). The camera rings and
  their advice are only for objects (`View.rings`, `_coverage_notes`).
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
- Automatic masks (`masks.py`): a stage per capture, `masks-<capture id>`,
  runs the masking worker (`mask_worker.py`, ISNet on ONNX Runtime) as a
  separate process through the process runner, like a backend: the app
  starts itself with `-m ez2digitize.mask_worker` (packaged: the `mask-worker`
  command of the launcher), so ONNX Runtime's 1 GB never sits in the GUI
  process and a crash in it doesn't take the app down. The stage's inputs
  are the capture's photos (left-out ones included, so leaving one out
  re-masks nothing) and the model's sha256; `apply_auto` copies its masks
  into `masks/`, never over imported ones and keeping review decisions.
- Feature extraction gets a mask for every image (a white one where there is
  none): COLMAP skips an image whose mask file is missing.
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

## 3D viewer (`views.py`, `ui/viewer.py`)

A web page (`ui/viewer_web/`: three.js and Spark, vendored and pinned by
`tools/viewer/fetch_vendor.py`) in a `QWebEngineView`, created only when
the 3D view tab is first shown.

- `views.available(project)` lists what can be shown from the stage
  manifests (camera placement, dense cloud, textured mesh, splats), each
  with the rotation that stands it upright; `views.files` produces what the
  page loads: stage outputs as they are, the sparse model as a PLY plus a
  JSON of cameras, the mesh as the export's GLB (or OpenMVS's PLY converted
  into a cached GLB).
- The 3D view tab (`ui/view_panel.py`) puts tools next to the view's
  name on the camera placement and the dense cloud, one at a time with one
  row of controls: coverage (camera placement only), crop box, scale and
  upright. What a tool draws shows while it is chosen; a crop box that is
  set always shows, and only the crop tool drags it.
- The page and files are served through an `ez2d://` scheme registered
  before the QApplication exists (`viewer.prepare()`); every other request
  is blocked. Python calls `ez2d.show(spec)`; the page answers with console
  lines (`EZ2D {json}`).
- Chromium's sandbox stays on except where it can't start (root; an
  AppImage where unprivileged user namespaces are restricted).
- Without WebGL the page still loads and says so; the packaged apps'
  self-test checks that the page loads (QtWebEngine works in the bundle).

## Crop box (`crop.py`)

An oriented box in the reconstruction's coordinates (OpenMVS's region of
interest: rotation rows, centre, half sizes), stored in project.json
`settings["crop_box"]` with the run id of the mapping stage it was drawn
on; a new camera placement makes it stale (ignored, with a notice). The
densify stage writes it to `crop_box.txt` and passes `--import-roi-file`,
replacing OpenMVS's own estimate; its text is part of the stage's
parameters, so a changed box re-runs densification and what follows.
The viewer edits it in the upright frame, level and turned about the
vertical (`UprightBox`); `from_upright`/`to_upright` convert.

## Scale (`scale.py`)

Two points in the reconstruction's coordinates and the real distance
between them (millimetres), stored in project.json `settings["scale"]`
with the mapping run id, stale after a new camera placement like the crop
box. The 3D view picks the points in the upright frame (a click casts a
ray into the cloud: of the points near the front-most hit, the one closest
to the ray) and converts them back; `ez2d scale` sets the points and the
distance, or just a corrected distance for the same points.

Markers (`markers.py`): AprilTag tag36h11 squares printed at a known size
(the black square; the sheet carries a 100 mm line to check the print).
After undistort, `auto_scale` looks for them in the undistorted photos
(pinhole cameras: a corner projects as K [R | t] X) with AprilTag (in
process; quads found at half resolution, edges refined at full, about
0.2 s for 12 MP), first in 8 photos spread over the set, stopping there if
none has a marker. Each corner seen in two or more photos is triangulated
(DLT, then again without views that reproject it more than 2 px off); the
printed size over each marker edge's length is an estimate, and the
median of them all is the scale, their median deviation the check (over
2 % gives a warning). It is stored as a Scale with `source: "markers"`:
the edge closest to the median, stretched to it. A scale picked by hand
on the same camera placement is kept (the markers are compared with it in
a notice). The sheet is drawn from the library's own code table and bit
layout (its `apriltag_to_image` draws a shifted border in this version).

Splat path (Phase 3) branches after `undistort`:

```
undistort -> [mask-undistort] -> Brush training -> PLY export
```

With masks, Brush trains on the masks warped for OpenMVS: the splat stage
links them as `dataset/images/masks/<stem>.png`, where Brush 0.3.0 looks
for an image's mask, and Brush leaves black pixels out of the loss. Brush
matches stems ignoring case across all captures, so if two photos' names
differ only in case the splats are trained without masks (with a notice).
