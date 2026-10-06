# EZ2DIGITIZE Roadmap

Photos or video of a small object in, a usable textured mesh out (with a
Gaussian splat as a secondary output), on non-NVIDIA hardware, using
open-source backends driven as external processes.

Phases are ordered by dependency. No durations are given on purpose.

## Decisions

| Topic | Decision | Consequences |
|---|---|---|
| Primary platform | AMD GPU on Linux | AppImage is the first installer. ROCm is available for optional PyTorch/ONNX extras, but the core pipeline must not need it (Brush uses Vulkan via wgpu; COLMAP and OpenMVS run on CPU). |
| Reference machine (high end) | Radeon RX 7900 GRE (RDNA3, 16 GB VRAM), Arch Linux, current Mesa | Defines the "high" preset. Because Arch is rolling, it only tests the newest libraries: the AppImage must be built on an older base (e.g. Ubuntu 22.04 container) and smoke-tested there in CI. |
| Secondary platform | macOS on Apple Silicon | Tested from Phase 1 so portability problems surface early. Brush runs on Metal. |
| Reference machine (low end) | M1, 8 GB unified memory | Defines the "low memory" preset: CPU and GPU share 8 GB, so stages must never run concurrently, images are downscaled for densification, and splat count is capped. If the pipeline works here, it works on most machines. |
| Windows, NVIDIA, Intel | No test hardware yet | Build in CI, but label them community-tested until someone with the hardware validates a release. |
| Use case | Small objects first | Masking, crop box, real-world scale, and two-sided ("flip") scans become core features. Rooms and outdoor scenes come later. |
| Output priority | Mesh is the real goal, splat is secondary | The MVP's success criterion is a mesh. Splat training reuses the same poses and is added once the mesh path works. |
| Phone transfer | QR-code upload page served by the desktop app (Phase 3); native Android app later (Phase 7) | No phone app needed for v1; works with any phone, including iPhones. Every intake path produces the same capture bundle, so the Android app can reuse it. |
| Schedule | Not fixed | No durations in this document. |

## Review notes on the original draft

| Change | Why |
|---|---|
| Phase 1 tests several capture types, not one easy object set | One easy dataset will pass and hide the failures users actually hit (turntables, video, low texture). |
| Phase 1 also tests the viewer, packaging and backend builds | All three decide whether Python/PySide6 is viable; finding out in the packaging phase is too late. |
| Minimal caching and a sparse-cloud viewer are in the MVP | You will rerun SfM dozens of times while developing later stages; seeing camera poses is the best diagnostic for a failed scan. |
| The MVP ships an installer for Linux | "A non-expert can install it" contradicts doing all packaging at the end. |
| Mesh export includes scale, orientation, crop and cleanup | Without them, OBJ/STL exports are not usable for 3D printing or CAD. |
| The license section was rewritten | For a GPL-3.0 desktop app, the obligations come from *bundling binaries* (source offers) and from non-commercial research licenses, not from linking vs. subprocess. |
| Regression testing uses tolerances | SfM and training are non-deterministic; byte-for-byte reference outputs will not work. |

## Architecture rules (decided in Phase 0)

1. **Heavy compute runs out of process.** COLMAP, OpenMVS, Brush and any
   future reconstruction backend are invoked as CLI subprocesses, never linked
   or imported. Lightweight, permissively licensed Python libraries (image
   checks, EXIF, export conversion, reading COLMAP models) may run in-process.
2. **Every stage writes a manifest** (`stage.json`): backend name and version,
   exact command line, parameters, input hashes, start/end time, exit code,
   peak memory if available. This gives caching, resume, reproducibility, and
   useful bug reports.
3. **Projects are versioned folders.** `project.json` carries a schema
   version from day one so later releases can migrate old projects.
4. **Masks are first-class project data.** Each image may have a mask; every
   stage that can use masks (feature extraction, densification, texturing,
   splat training) receives them.
5. **All captures arrive as capture bundles.** Folder import, video frame
   extraction, phone upload and the later Android app all produce the same
   thing: original files, untouched, plus a `capture.json` (schema version,
   source, device model if known, per-photo metadata where available). The
   pipeline only ever reads bundles.
6. **Undistortion is a shared stage.** Run `colmap image_undistorter` once;
   OpenMVS and the splat trainer both consume undistorted pinhole images.
7. **Never bundle non-commercially licensed components.** If a backend or
   model is NC-licensed (common for research code and weights), the user
   installs it through the plugin mechanism, with the license shown. Bundling
   it would make the release something others can't freely redistribute.

## Phase 0: Foundations

- Repo with GPL-3.0 (done), `CLAUDE.md` (architecture rules above, license
  rules, coding conventions), `THIRD_PARTY_LICENSES`, SPDX headers.
- Stack: Python 3.12+ with PySide6 (LGPL-3.0, compatible with GPL-3.0),
  managed with `uv` and a lockfile. Ruff, pyright or mypy, pytest, pytest-qt.
- CI (GitHub Actions): lint, type check, unit tests on Linux and macOS
  (Windows added later, as build-only: since October 2026 the suite runs
  there too).
- Process runner design: `QProcess` or asyncio subprocesses off the GUI
  thread; cancel kills the whole process group.

## Phase 1: Feasibility spike

No GUI. Run the tools on the AMD Linux machine (primary) and the M1
(secondary), and record results in `docs/feasibility/`. The benchmark
harness is in `tools/feasibility/`; the step-by-step protocol is in
`docs/feasibility/README.md`.

**Datasets** (own captures, redistributable later under CC-BY):

1. Small textured object on a patterned mat, photos walked around it
   (~40-60 images, two height rings).
2. Same object on a turntable with a static camera. Plain COLMAP is expected
   to lock onto the static background; then rerun with masks.
3. Same object flipped upside down, second capture, to test merging both
   sides into one reconstruction (masks required).
4. Phone video orbit of an object, frames extracted with ffmpeg.
5. A deliberately hard set: shallow depth of field, a shiny surface, a
   low-texture object.

**Per run, record:** wall time, peak RAM, peak VRAM, registered
images / total, mean reprojection error, and screenshots. On the M1, also
record whether macOS started swapping; on 8 GB that is the first limit
you will hit.

**Questions to answer:**

- **COLMAP on CPU:** SIFT extraction/matching speed at full resolution
  (small objects need detail, so check how far you can downscale before
  quality drops); feature extraction with masks; incremental mapper vs. the
  global mapper (GLOMAP, integrated into recent COLMAP releases; check the
  pinned version). COLMAP's own dense MVS requires CUDA, so it is not an
  option here.
- **OpenMVS on CPU (the critical path):** `InterfaceCOLMAP` →
  `DensifyPointCloud` (resolution levels 0, 1, 2, with and without masks)
  → `ReconstructMesh` → `RefineMesh` (optional) → `TextureMesh`. For a
  60-image object, is the result ready in minutes or in hours, and how much
  detail does each resolution level keep?
- **Masking:** background removal quality on dataset 2 and 3 with a
  permissively licensed model (e.g. BiRefNet, MIT, or SAM 2, Apache-2.0) via
  ONNX Runtime on CPU; then check whether the ROCm/MIGraphX or CoreML
  execution providers are worth the setup.
- **Brush:** training on the AMD GPU (Vulkan/RADV) and the M1 (Metal) from
  the same undistorted COLMAP output; VRAM ceiling; CLI stability. Pin a
  version (it is pre-1.0). Fallback: OpenSplat (AGPL-3.0; HIP, Metal and CPU
  paths).
- **Viewer spike:** a web splat/mesh viewer (e.g. Spark or the PlayCanvas
  engine, both MIT) inside `QWebEngineView` on Linux with Mesa and on macOS,
  versus a native option (pygfx/wgpu-py, or VTK) for meshes and point clouds.
  Note the package size QtWebEngine adds.
- **Packaging spike:** a hello-world PySide6 app plus one bundled backend
  binary as an AppImage (PyInstaller or Nuitka inside), then the same as a
  macOS `.app`.
- **Backend builds:** CI jobs that build pinned COLMAP and OpenMVS for Linux
  x86_64 and macOS arm64. Distro packages lag and differ in versions, so
  don't rely on them. Done in `tools/backends/` (COLMAP 4.2.1, OpenMVS
  v2.4.0, static vcpkg builds). COLMAP 4.2.1 also has a `HIP_ENABLED`
  option for AMD GPUs through ROCm, worth a test build on the GRE.

**Exit decision:** written yes/no on (a) OpenMVS on CPU is fast enough for
small objects, and at which resolution level by default, (b) which masking
model and runtime, (c) Python packaging acceptable, (d) viewer approach.
If (a) is "no", the fallback plan is a splat-first MVP while the mesh
path is investigated separately.

## Phase 2: MVP vertical slice (mesh)

Photos or video of a small object in, a textured mesh out, inside a GUI,
installable on Linux as an AppImage.

- Project model: folder with `project.json`, `captures/` (capture bundles),
  `masks/`, per-stage output folders with manifests. Done in
  `ez2digitize.core` (`project`, `capture`, `stage`).
- Import photos or video. Video: ffmpeg frame extraction, keeping the
  sharpest frame per window rather than uniform sampling. Done
  (`ez2digitize.video`): 4 candidates per window, scored like the photo
  checks; FFmpeg is the system's for now (bundling it is Phase 6).
- Basic checks: resolution, EXIF focal length (missing EXIF is a warning,
  not an error), blur score relative to the rest of the set, mixed cameras.
  Done (`ez2digitize.core.photos`, the GUI's photo checks tab, `ez2d
  photos`), plus odd-sized files (a collage in the skull set), duplicates
  and too few photos; flagged photos can be left out and brought back.
- Automatic masking with the model chosen in Phase 1, with a quick review
  grid where the user can drop bad masks. Done (`ez2digitize.masks`,
  `mask_worker`): ISNet on ONNX Runtime in a worker process, one cached
  stage per capture; the Masks tab shows every photo with what its mask
  removes tinted red, flags empty, near-total, unsure and odd masks, and
  unchecking drops a mask (the photo is then used whole); imported masks
  (Add masks…) win over automatic ones; `ez2d masks`. Photos without a
  mask get a white one for COLMAP too, which otherwise skipped them.
- Camera grouping: one intrinsics set per camera/lens. Done: one per
  capture as before, and when a capture mixes cameras, lenses, zoom
  settings (beyond 5 %) or sizes, features are extracted with a camera per
  photo and merged per group in the matching stage's copy of the database
  (`colmap.merge_cameras`; checked against COLMAP: the merged database is
  identical to single-camera extraction and maps).
- Pipeline runner: subprocess stages with live logs, progress parsing,
  cancel, and **minimal caching** (skip a stage if its inputs and parameters
  are unchanged). Runner, manifests and caching done in `ez2digitize.core`
  (`runner`, `stage`); COLMAP and OpenMVS modules with progress parsers in
  `ez2digitize.backends`; the pipeline that chains them in
  `ez2digitize.pipeline`, with a headless CLI (`ez2d`). Masks are warped to
  the undistorted images and used by OpenMVS densification.
- Stages: features (masked), matching (sequential for video, exhaustive
  for photo sets), mapping, undistortion, OpenMVS densify/mesh/texture.
  Video frames are matched exhaustively up to 200, which closes the loop
  of an orbit; sequentially beyond.
- GUI: project page with import, settings, Run/Cancel, per-step progress,
  log and failure details, driving the pipeline on a worker thread (done,
  `ez2digitize.ui`).
- **Sparse viewer** with camera frustums after SfM, and a **crop box** the
  user adjusts before densification. The automatic part exists: OpenMVS
  estimates a region of interest from the sparse points and crops to it
  (`--estimate-roi`, `--crop-to-roi`, on by default). Done (`ez2digitize.crop`):
  Place cameras stops after camera placement and opens the 3D view; there
  "Crop box" starts from a box around most of the sparse points, its faces
  are dragged by their handles and it turns about the vertical; the box is
  saved in project.json with the camera placement it belongs to, and
  densification keeps only what is inside (`DensifyPointCloud
  --import-roi-file`; checked on the skull set: the stand cut off level).
  `ez2d crop` sets it headless.
- Export OBJ (+MTL + textures) and GLB. Done (`ez2digitize.export`).
- AppImage with pinned backend binaries. Done (`tools/packaging`, AppImage
  workflow): GUI and CLI in one file, backends bundled.

**Done when:** a non-expert on a clean Linux machine with an AMD GPU
installs the AppImage, drops in photos or a video of a small object, presses
one button, adjusts the crop box, and gets a textured mesh; and when it
fails, they can see which stage failed and why.

## Phase 3: Robust backend layer, splats and phone upload

- Formal backend interface: inputs, outputs, parameters, capabilities
  (GPU vendor, needs CUDA, supports masks), version detection, license.
- Hardware detection: GPU vendor and VRAM (enumerate adapters via
  Vulkan/wgpu rather than vendor tools), RAM, CPU cores; pick image
  downscale, OpenMVS resolution level and splat count cap from them.
  Detection done (`ez2digitize.core.hardware`: `vulkaninfo`, sysfs VRAM for
  amdgpu, `system_profiler` on macOS; RAM and cores in
  `core.resources`), and feature threads are already capped by free memory.
  Picking the default preset from it waits for the M1 numbers (Phase 1).
- Full resume and invalidation of downstream stages when parameters change.
- Quality presets (fast, balanced, high) mapped to concrete parameters, with
  an "advanced" panel showing the actual values. Done (`ez2digitize.presets`;
  GUI Quality plus an Advanced box, `ez2d run --quality`; the choice is
  saved as the project's `preset`).
- Error translation for common failures: too few registered images, several
  disconnected models (often "the two sides didn't connect"), out-of-memory,
  missing backend. Done (`ez2digitize.diagnosis`, plus the pipeline's own
  notices for low registration and split models).
- Splat output: Brush training on the same poses and masks, `.ply` export,
  viewable in the embedded viewer. Done (`backends/brush.py`,
  `pipeline.run_splat`, Build splats in the GUI, `ez2d run --splat`; Brush
  0.3.0 bundled in the AppImage). It refuses software
  renderers. With masks, Brush gets the masks warped for OpenMVS where it
  looks for them (`images/masks/<stem>.png`) and leaves the background out
  of its loss. The 3D view shows them, stood upright. Rotating splats
  upright on export is left out (needs the SH coefficients rotated too).
- "Export diagnostics" button: logs, manifests and system info zipped for
  bug reports (images only if the user opts in). Done
  (`ez2digitize.diagnostics`, Help → Export Diagnostics and a button after
  a failure, `ez2d diagnostics`); images are never included for now.
- macOS `.app` build in CI, tested on the M1. Built and smoke-tested in CI
  (`tools/packaging/build_macos.py`, macOS app workflow); testing on the M1
  is yours.
- **Phone upload over Wi-Fi.** An "Add photos from phone" dialog shows a QR
  code; the phone opens it in its browser and gets a small upload page served
  by the desktop app. The user picks the photos taken with the normal camera
  app.
  - Pairing: the QR carries a random one-time token; the server listens
    only while the dialog is open and only on the local network.
  - Originals are stored bit for bit (no re-encoding, EXIF kept); a photo
    set of 300 MB or more must survive Wi-Fi drops, so uploads are chunked
    and resumable.
  - HEIC from iPhones is decoded on the desktop (e.g. pillow-heif with
    libheif; check and record their licenses).
  - Uploads land in a new capture bundle, with live progress on both sides.
  - Also a "watch folder" option for people who already sync their phone
    (Syncthing and similar).
  - Done (`ez2digitize.upload`, "From phone…" in the GUI, `ez2d upload`):
    token URL in a QR code (segno), LAN address only, private clients only,
    4 MB chunks resumed after drops (tested in Chromium with a dropped
    chunk), bit-for-bit originals in a new bundle. HEIC: on import (folder
    or phone) a JPEG copy is made next to the original, which is kept and
    left out (`core.heic`, pillow-heif). The watch folder too
    (`ez2digitize.watch`, From phone → From a synced folder…, `ez2d
    watch`): new photos are those that appear (sync tools keep capture
    times), one has arrived when unchanged for 5 s, sync tools' temporary
    files count as arriving, and the set has settled after 30 s without
    change; the user imports then (the CLI does it itself).

## Phase 4: Mesh quality for real-world use

- **Scale:** set real-world size from a known distance between two picked
  points; then automatic scale from printed ArUco markers on the capture mat.
  Picked points done (`ez2digitize.scale`; in the 3D view on the camera
  placement or dense cloud: Pick two points, the real distance, Set scale;
  `ez2d scale`): STL and 3MF export in millimetres, OBJ, GLB and the point
  cloud in metres. Checking against a caliper-measured object (see Testing)
  waits for a capture. Printed markers done (`ez2digitize.markers`; AprilTag
  tag36h11 instead of ArUco: OpenCV's wheels bundle OpenSSL 1.1.1, not
  GPL-compatible, while AprilTag is BSD and 4.5 MB): a printable A4 sheet
  (`ez2d markers`, Marker sheet… in the Scale tool), and after camera
  placement the scale is set from the markers found, unless one was set by
  hand (then they are compared). On a rendered sheet the scale comes out
  within 0.1 %; a real capture is still to come.
- **Orientation:** up-axis alignment (`colmap model_orientation_aligner` or
  fit to the mat plane) with manual adjust. Automatic part done
  (`ez2digitize.orientation`): up from the photos' down directions,
  corrected for EXIF rotation (COLMAP reads pixels unrotated, and its
  aligner maps gravity to +Y, upside down for glTF); exports stand upright,
  centred, on the ground, Y-up for OBJ/GLB and Z-up for STL/3MF. Manual
  adjust done (`ez2digitize.upright`; in the 3D view on the camera
  placement or dense cloud: Level from three points on the mat or base, Tip
  forward/sideways by quarter turns, Turn about the vertical, Automatic;
  `ez2d orient`): the view, the exports and the mesh view all use it, and a
  crop box is re-fitted level when it changes.
- **Two-sided scans:** guided workflow for capturing the object, flipping
  it, capturing again, and reconstructing both sets together through masks.
  Done (`ez2digitize.sides`): a capture can be marked as turned over
  (`flipped` in capture.json; Other side… and the Both sides tab, `ez2d
  import --flipped`, `ez2d flip`); the tab explains the capture and keeps a
  checklist (photos of both sides, a mask for every photo), and Build asks
  before running with masks missing. The pipeline warns about the same,
  then says after camera placement whether the sides joined (photos of each
  side placed together); the upright estimate uses the first side's photos
  only. Still to test on a real two-sided capture (Phase 1 dataset 3).
- Mesh cleanup: keep largest component, remove floaters, decimate to a
  target face count, hole filling and watertightness check for printing
  (Open3D is MIT; PyMeshLab is GPL-3.0, both fine). Done without a new
  dependency: OpenMVS's ReconstructMesh already removes spurious components
  and spikes, closes small holes and smooths (its defaults); a Mesh size
  setting simplifies to a target face count in TextureMesh, before
  texturing (`--faces`); the export checks watertightness (edges not shared
  by exactly two faces). The skull's meshes come out closed.
- Export STL and 3MF for printing (untextured; warn if not watertight),
  PLY point cloud. Done (`stl`, `3mf`, `points` export formats), in real
  units once the scale is set.
- License notice for OpenMVS (AGPL-3.0) and its dependencies (some CGAL
  components are GPL) in `THIRD_PARTY_LICENSES`, with a source offer for the
  exact bundled versions. Done: `THIRD_PARTY_LICENSES` lists what the apps
  bundle (backends, GCC runtime, libomp) and says where their source is;
  both packages carry it and LICENSE, shown under Help → Licenses (with the
  bundled backends' versions and license folder) and by `ez2d licenses`.
  Publishing the source archive with each release is Phase 6.

## Phase 5: UX and capture guidance

- Unified embedded viewer for sparse cloud, splat and mesh. Done
  (decision (d): three.js + Spark in QtWebEngine; `ez2digitize.views`,
  `ui/viewer.py`, the 3D view tab): camera placement (sparse points and a
  frustum per photo), dense cloud, textured mesh (the upright GLB export,
  or OpenMVS's PLY converted) and splats, all stood upright; a finished
  build opens in it. The crop box and the scale are set in it (Phases 2
  and 4), and so is the orientation, and the camera placement shows the
  coverage rings (below).
- Capture guide for small objects: diffuse lighting, a patterned mat,
  two or three height rings, enough depth of field, the flip workflow, and
  what to do with shiny objects (matte spray, cross-polarization). Written
  (docs/CAPTURE.md); to check against the Phase 1 captures.
- Photo set health check: overlap estimate, coverage gaps shown on the
  camera rings, warnings for blur and shiny/transparent subjects. Partly
  done: blur is in the photo checks; after camera placement
  `ez2digitize.coverage` reports gaps around the object (over 90°), photos
  all from one height, photos placed far from the rest, photos not placed,
  and the overlap estimate: photos with verified matches to fewer than two
  others (notices in the log). The 3D view shows them on the camera
  placement (`coverage.rings`): a ring per height, its gaps from 35° shaded
  and labelled (red over 90°), cameras with few matches orange and
  misplaced ones red, with a summary above the view. Shiny/transparent
  subjects aren't detected.
- Turntable mode tuned for a static camera (masking is already in place).
  The coverage check spots a camera that didn't move (all views within
  10°) and says to mask the background. With automatic masks the skull
  turntable set reconstructs cleanly (see FINDINGS); a dedicated mode
  (masks made automatically, the still-camera notice turned into a
  suggestion to make them) waits for more real turntable captures.
- Compressed splat export (e.g. SPZ, MIT) and, once adopted, the Khronos glTF
  Gaussian splatting extension. SPZ done (`core/splats.py`): every splat
  export writes `<name>.spz` next to Brush's PLY, about a tenth of its size
  (1.4 MB to 0.08 MB on the skull video test; 248 MB to 9 MB for a million
  splats), stood upright, centred and on the ground like the mesh, each
  splat's rotation and colour coefficients turned with it, in metres once
  the scale is set. Checked against Niantic's reference reader. The glTF
  extension waits for its adoption.

## Phase 6: Packaging and release

- Release builds: Linux AppImage (primary), macOS `.app` (secondary),
  Windows installer (community-tested). A portable Windows zip is built in
  CI (Windows app workflow); the installer comes later.
- Backend binaries bundled or auto-downloaded with SHA-256 checksums and
  pinned versions; source tarballs for every GPL/AGPL binary published with
  each release. Bundled (AppImage, macOS app, Windows zip); the source
  archive is built by `tools/backends/collect_sources.sh`, and each release
  attaches it.
- Release process. Done (docs/RELEASING.md): the version in
  `ez2digitize.__version__`, CHANGELOG.md, and a tag `vX.Y.Z` on main
  that the Release workflow turns into a draft GitHub release: CI, the
  backends built from the tagged commit, the three packages built and
  tested with them, the backends' source packed from the same pins, the
  app's source and SHA256SUMS. `tools/packaging/release.py` sets the
  version and dates the changelog. The notices of the Rust crates inside
  Brush ship with it (`tools/packaging/brush_notices.py`).
- macOS code signing and notarization (paid Apple developer account);
  Windows signing can wait until Windows is officially supported.
- Docs: quick-start, capture guide, troubleshooting, contribution guide,
  sample datasets with explicit licenses. Written (docs/QUICKSTART.md,
  CAPTURE.md, TROUBLESHOOTING.md, CONTRIBUTING.md); to revise against
  release builds. Sample datasets wait for your own captures.
- Release testing: AMD Linux and M1 by the maintainer; NVIDIA, Intel and
  Windows through a call for community testers before calling them
  supported.

## Phase 7: Beyond v1

- Plugin system for third-party backends (user-installed, license shown).
  Done (`ez2digitize.plugins`, docs/PLUGINS.md): plugins for camera
  placement and for splats, installed from a folder or `.zip`, licenses
  shown and accepted before use (Settings → Plugins, `ez2d plugins`), run
  as stages like the bundled tools. An example plugin is in
  `tools/plugins/example-poses`. More slots (features and matching, mesh
  from splats) can follow once a real plugin needs them.
- Larger scenes (rooms, outdoor) with appropriate matchers and presets.
  Done: a project's subject (object, or room/outdoor scene: no masks, no
  camera-ring advice, OpenMVS free-space support for plain walls), and
  matching for more than 200 photos by GPS, by COLMAP's vocabulary tree
  (downloaded once, pinned like COLMAP pins it) or in order, with loop
  detection for video (docs/CAPTURE.md). Waits for a real room and an
  outdoor capture to tune the presets (memory, dense detail).
- Better features/matching inside COLMAP: learned local features and
  matchers (e.g. ALIKED or DISK with LightGlue, permissively licensed;
  SuperPoint weights are not). A nearer-term gain than replacing SfM.
  Done: COLMAP's own ALIKED + LightGlue (ONNX Runtime, now in the backend
  builds), as an option next to SIFT; the models (BSD-3-Clause,
  Apache-2.0) and the ALIKED vocabulary tree are downloaded once, pinned as
  COLMAP pins them. Making it the default waits for a comparison on the
  Phase 1 datasets (placed photos, time on the CPU, the M1).
- Feed-forward pose estimation (VGGT, MASt3R-SfM and similar) as optional
  plugins. Most are PyTorch/CUDA-first (ROCm on Linux may work) and several
  carry non-commercial licenses, so they must not be bundled.
- Surface reconstruction from splats (2DGS-style methods) as a second mesh
  path; check licenses, many derive from Inria's non-commercial code.
- Android capture companion, sending capture bundles through the Phase 3
  upload endpoint. Its value over the upload page is control of the camera,
  which a browser can't do:
  - focus, exposure and white balance locked for the whole capture, and one
    lens only (phones otherwise switch lenses and readjust between shots);
  - live guidance: coverage ring of angles already shot, blur check after
    each photo, optional automatic shutter once the phone has moved enough;
  - optionally ARCore poses saved in `capture.json` as priors for COLMAP.

  Before starting it, check the Phase 1 photo sets (EXIF focal lengths,
  exposure differences) to see how much the phone's automatic adjustments
  actually hurt reconstruction.

## Testing and validation

- **CI (no GPU):** unit tests, GUI smoke tests with pytest-qt, and a tiny
  (~15 image) COLMAP + low-resolution OpenMVS run on CPU checked against
  tolerances: registered images ≥ N, mean reprojection error ≤ X px, mesh
  face count and bounding box within a range.
- **GPU regression (manual on the AMD Linux machine and the M1, or a
  self-hosted runner):** the Phase 1 datasets, checked for completion, mesh
  metrics against reference (bounding box, Chamfer distance to a reference
  mesh), splat PSNR on held-out views within tolerance. Run before each
  release and each backend version bump.
- **Scale accuracy:** measure a known object with calipers and check the
  scaled mesh against it.
- **Human review:** reference screenshots per dataset, compared by a person
  before release.

## Main risks

- **Reconstruction failures are the main UX problem.** Shiny, transparent and
  textureless objects, and poor overlap, fail regardless of the code.
  Mitigation: capture guidance, masks, sparse viewer, honest error messages.
- **OpenMVS CPU speed** at the detail level small objects need. Phase 1
  measures it; crop box, masks and resolution presets keep it manageable.
- **Masking quality** decides whether turntable and flip scans work. A
  bad mask silently degrades both poses and mesh, so masks must be
  reviewable.
- **Dependency drift:** pin backend versions, build them in CI, bump
  deliberately with the regression set. Brush is pre-1.0.
- **GPU driver quirks:** wgpu/Vulkan behaviour on AMD varies by Mesa version;
  record the driver version in every manifest.
- **Untested platforms:** Windows, NVIDIA and Intel have no maintainer
  hardware; keep them clearly labelled until community testing covers them.
- **License contamination:** one NC-licensed dependency bundled by accident
  makes the release non-redistributable. Review `THIRD_PARTY_LICENSES` at
  every release.
- **Local upload server:** a network listener in a desktop app is attack
  surface. Keep it off by default, token-protected, LAN-only, with size and
  file-type limits, and never derive file paths from uploaded names.
- **Scope creep:** nothing from Phase 5+ until the Phase 2 mesh path is
  polished.

## Open questions

None currently.
