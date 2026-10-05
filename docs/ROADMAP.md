# EZ2DIGITIZE Roadmap

Photos or video in, Gaussian splat or textured mesh out, on any GPU vendor
(AMD, Intel, Apple, NVIDIA), using open-source backends driven as external
processes.

Durations assume one developer working full time. Treat them as relative
sizes, not commitments.

## What changed from the first draft

| Change | Why |
|---|---|
| Phase 1 tests several capture types, not one 20-50 photo object set | A single easy dataset will pass and hide the failure modes users actually hit (turntables, video, low texture). |
| Phase 1 also spikes the viewer and packaging | Both decide whether Python/PySide6 is viable; finding out in Phase 6 is too late. |
| Minimal stage abstraction, caching and a sparse-cloud viewer moved into Phase 2 | You will rerun SfM dozens of times while developing the training stage; and seeing the camera poses is the single best diagnostic for "why did my scan fail". |
| Phase 2 ships an installer for one OS | "A non-expert can install it" contradicts packaging only in Phase 6. |
| Video input moved to Phase 2/3 | Video is the easiest capture method for non-experts and is cheap to add (ffmpeg + sharpness-based frame selection). |
| Phase 4 adds scale, orientation, cropping and mesh cleanup | Without them, OBJ/STL exports are not usable for 3D printing or CAD. |
| The license section was rewritten | The AGPL concern is not about linking vs. subprocess for a GPL-3.0 desktop app; the real obligations come from *bundling binaries* (source offers) and from non-commercial research licenses. |
| Backend build CI starts in Phase 1, not Phase 6 | Linux/macOS binaries of COLMAP and OpenMVS must be built by us; this is weeks of work, not a packaging afterthought. |
| Regression testing is defined with tolerances | SfM and training are non-deterministic; byte-for-byte reference outputs will not work. |

## Architecture rules (decided in Phase 0)

1. **Heavy compute runs out of process.** COLMAP, Brush, OpenMVS and any
   future reconstruction backend are invoked as CLI subprocesses, never linked
   or imported. Lightweight, permissively licensed Python libraries (image
   checks, EXIF, export conversion, reading COLMAP models) may run in-process.
2. **Every stage writes a manifest** (`stage.json`): backend name and version,
   exact command line, parameters, input hashes, start/end time, exit code,
   peak memory if available. This one file gives you caching, resume,
   reproducibility, and useful bug reports.
3. **Projects are versioned folders.** `project.json` carries a schema
   version from day one so later releases can migrate old projects.
4. **Undistortion is a shared stage.** Run `colmap image_undistorter` once;
   both the splat path and the OpenMVS path consume undistorted pinhole
   images.
5. **Never bundle non-commercially licensed components.** If a backend is
   NC-licensed (common for research code and model weights), the user
   installs it themselves through the plugin mechanism, with the license
   shown to them. Bundling it would make the distributed app something
   others can't freely redistribute, which defeats the point of GPL-3.0.

## Phase 0: Foundations (~1 week)

- Repo with GPL-3.0 (done), `CLAUDE.md` (architecture rules above, license
  rules, coding conventions), `THIRD_PARTY_LICENSES`, SPDX headers on source
  files.
- Stack: Python 3.12+ with PySide6 (LGPL-3.0, compatible with GPL-3.0),
  managed with `uv` and a lockfile. Ruff, pyright or mypy, pytest,
  pytest-qt.
- CI skeleton (GitHub Actions): lint, type check, unit tests on Linux and
  Windows.
- Process runner design note: use `QProcess` or asyncio subprocesses off the
  GUI thread; cancel must kill the whole process tree (Windows job objects,
  POSIX process groups).

## Phase 1: Feasibility spike (~2-3 weeks)

No GUI. Run the candidate tools by hand and record results in
`docs/feasibility/`.

**Datasets** (own captures, so they can be redistributed later under
CC-BY):

1. Small textured object, photos walked around it (~40 images).
2. Same object on a turntable with a static camera. Expect plain COLMAP to
   fail or reconstruct the background; this tells you whether masking is
   a v1 requirement.
3. Phone video of a larger object or room (frames extracted with ffmpeg).
4. A deliberately bad set: blur, low overlap, a shiny or low-texture object.
5. A larger set (~300 images) to measure scaling.

**Per run, record:** wall time, peak RAM, peak VRAM, registered images /
total, mean reprojection error, and a screenshot of the result.

**Questions to answer:**

- COLMAP without CUDA: CPU SIFT extraction and matching speed; exhaustive
  vs. sequential vs. vocabulary-tree matching; incremental mapper vs. the
  global mapper (GLOMAP, which recent COLMAP releases integrate; check the
  version you pin). Note that COLMAP's own dense MVS requires CUDA, so it is
  not an option on AMD.
- Brush on the AMD GPU: training time and quality at 1-2 resolutions,
  VRAM ceiling, and whether its CLI is stable enough to drive. Pin a version;
  it is pre-1.0. If it fails, try OpenSplat (AGPL-3.0, has HIP/ROCm, Metal
  and CPU paths) as a fallback.
- OpenMVS on CPU: `DensifyPointCloud` at resolution levels 1 and 2,
  `ReconstructMesh`, `TextureMesh`; skip `RefineMesh` unless time allows.
  Is a 40-image object done in minutes or hours?
- **Viewer spike:** can a web splat viewer (e.g. Spark or the PlayCanvas
  engine, both MIT) run inside `QWebEngineView` on the AMD machine, and how
  much does QtWebEngine add to the package size? Compare with launching
  Brush's own viewer.
- **Packaging spike:** a hello-world PySide6 app packaged with PyInstaller or
  Nuitka for the main target OS, plus one bundled backend binary.
- **Backend builds:** start CI jobs that build pinned COLMAP and OpenMVS for
  Linux (Windows has upstream binaries; verify they include the non-CUDA
  paths you need).

**Exit decision:** written yes/no on (a) mesh output viable without NVIDIA,
(b) masking needed for v1, (c) Python packaging acceptable, (d) viewer
approach.

## Phase 2: MVP vertical slice (~4-5 weeks)

Photos or video in, splat out, inside a GUI, installable on one OS.

- Project model: folder with `project.json`, `images/`, per-stage output
  folders with manifests.
- Import photos or video. Video: ffmpeg frame extraction, keep the
  sharpest frame per window instead of uniform sampling.
- Basic checks: resolution, EXIF focal length (missing EXIF is a warning,
  not an error; it only removes a calibration prior), blur score relative to
  the rest of the set rather than an absolute threshold, mixed cameras.
- Camera grouping: one intrinsics set per camera/lens, which improves SfM
  for phone captures noticeably.
- Pipeline runner: subprocess stages with live logs, progress parsing,
  cancel, and **minimal caching** (skip a stage if its manifest's inputs and
  parameters are unchanged).
- Stages: feature extraction, matching (sequential for video, exhaustive
  for small photo sets), mapping, undistortion, Brush training, `.ply` export.
- **Sparse viewer:** point cloud and camera frustums after SfM, so users can
  see whether poses are plausible before spending GPU time.
- Splat viewing: whatever the Phase 1 viewer spike chose; fall back to
  launching Brush's viewer.
- Installer for the main target OS, with pinned backend binaries.

**Done when:** a non-expert on a clean machine with an AMD GPU installs the
app, drops in photos or a video, presses one button, and gets a splat; and
when it fails, they can see which stage failed and why.

## Phase 3: Robust backend layer (~2-3 weeks)

- Formal backend interface: inputs, outputs, parameters, capabilities
  (GPU vendor, needs CUDA, supports masks), version detection, license.
- Hardware detection: GPU vendor and VRAM (enumerate adapters through
  Vulkan/wgpu rather than vendor tools), RAM, CPU cores. Use it to pick image
  downscale factor, splat count cap and matcher.
- Full resume and invalidation of downstream stages when parameters change.
- Quality presets (fast, balanced, high) mapped to concrete backend
  parameters, with an "advanced" panel that shows the actual values.
- Error translation for common failures: too few registered images,
  multiple disconnected models, out-of-memory on GPU, missing backend.
- "Export diagnostics" button: zips logs, manifests and system info for bug
  reports (no images unless the user opts in).

## Phase 4: Mesh pipeline (~4-5 weeks)

Only if Phase 1 showed CPU OpenMVS is usable.

- OpenMVS: `InterfaceCOLMAP` → densify → mesh → texture, with resolution
  presets.
- **Bounding box / crop** in the viewer before densification. This saves a
  lot of time and removes most background.
- **Orientation:** up-axis alignment (`colmap model_orientation_aligner` or
  ground-plane fit) with manual adjust.
- **Scale:** set real-world scale from a known distance between two picked
  points, later optionally from printed ArUco markers. Required for STL to be
  useful.
- Mesh cleanup: keep largest component, remove floaters, decimate to a
  target face count, optional hole filling for printing (Open3D is MIT;
  PyMeshLab is GPL-3.0, both fine).
- Export OBJ (+MTL + textures), GLB (trimesh or pygltflib), STL (untextured;
  warn if not watertight), PLY point cloud.
- License notice for OpenMVS (AGPL-3.0) and its dependencies (CGAL parts
  are GPL) in `THIRD_PARTY_LICENSES`, with a source offer for the exact
  bundled versions.

## Phase 5: UX and capture guidance (~3-4 weeks)

- Unified embedded viewer for sparse cloud, splats and meshes, built on the
  Phase 2 viewer.
- Capture tips per scenario (object, turntable, room) shown before import.
- Photo set health check: overlap estimate, coverage gaps visualised on the
  camera ring, warnings for shiny/transparent subjects.
- Masking / object isolation: background removal (e.g. SAM 2, Apache-2.0,
  or BiRefNet via ONNX Runtime; DirectML on Windows, CPU elsewhere) feeding
  COLMAP masks and training. Promote to Phase 2-3 if Phase 1 showed
  turntable support is needed.
- Compressed splat export (e.g. SPZ, MIT) and, once ratified and adopted,
  the Khronos glTF Gaussian splatting extension.

## Phase 6: Packaging and release (~4-6 weeks)

- Installers for the remaining OSes: Windows installer, Linux AppImage
  (simpler than Flatpak for bundling subprocess binaries), macOS app.
- Backend binaries bundled or auto-downloaded, with SHA-256 checksums and
  pinned versions; source tarballs for every GPL/AGPL binary published
  alongside each release.
- Code signing: Windows (otherwise SmartScreen warnings scare the target
  audience) and macOS (Developer ID + notarization; requires a paid Apple
  developer account). Budget for both.
- Docs: quick-start, capture guide, troubleshooting, contribution guide,
  sample datasets with explicit licenses.
- Test matrix on real hardware: AMD, NVIDIA, Intel Arc, Apple Silicon. If
  you don't own a Mac, mark macOS as community-supported for v1.

## Phase 7: Beyond v1

- Plugin system for third-party backends (user-installed, license shown).
- Better features/matching inside COLMAP: learned local features and
  matchers (e.g. ALIKED or DISK with LightGlue, which have permissive
  licenses; SuperPoint weights do not). This is a nearer-term gain than
  replacing SfM.
- Feed-forward pose estimation (VGGT, MASt3R-SfM and similar) as optional
  plugins. Most are PyTorch/CUDA-first and several carry non-commercial
  licenses on code or weights, so they reintroduce the NVIDIA dependency and
  must not be bundled. Re-evaluate licenses at that time.
- Surface reconstruction from splats (2DGS-style methods) as a second mesh
  path; check licenses, many derive from the Inria non-commercial code.
- Android capture companion: guided shooting, optionally recording ARCore
  poses as priors, uploads to the desktop app.

## Testing and validation

- **CI (no GPU):** unit tests, GUI smoke tests with pytest-qt, and a tiny
  (~15 image) COLMAP run on CPU checked against tolerances: registered
  images ≥ N, mean reprojection error ≤ X px.
- **GPU regression (manual or self-hosted runner):** the Phase 1 datasets,
  checked for training completion, PSNR on held-out views within a tolerance
  of the reference, mesh face count and bounding box within tolerance.
  Run before every release and every backend version bump.
- **Human review:** keep reference screenshots per dataset; a person compares
  them before a release, since metrics miss visual regressions.

## Main risks

- **Reconstruction failures are the main UX problem.** Shiny, transparent and
  textureless subjects, and poor overlap, will fail regardless of the code.
  Mitigation: capture guidance, sparse viewer, honest error messages.
- **Dependency drift:** pin backend versions, build them in CI, bump
  deliberately with the regression set. Brush in particular is pre-1.0 and
  its CLI may change.
- **GPU driver quirks:** wgpu/Vulkan behaviour on AMD and Intel varies by
  driver and OS; test on real hardware from Phase 1.
- **VRAM limits** on consumer AMD cards: cap splat count and training
  resolution per detected VRAM.
- **License contamination:** one NC-licensed dependency bundled by accident
  makes the release non-redistributable. Keep `THIRD_PARTY_LICENSES`
  current and review it at every release.
- **Packaging effort** for three OSes plus native backends is routinely
  underestimated; that is why it is spiked early.
- **Scope creep:** nothing from Phase 5+ until Phase 2 is polished.

## Open questions

1. Which OS is the primary target, and which exact AMD GPU (model, VRAM) is
   the reference machine? AMD on Windows vs. Linux changes the backend
   options considerably.
2. What is the main use case: small objects (3D printing, product shots,
   museum items), or rooms and outdoor scenes? This decides whether mesh,
   scale and masking are v1 or later.
3. Is a mesh the real goal and splats a stepping stone, or are splats
   enough for most users?
4. Is this full time, part time, or a team? The durations above assume
   one person full time.
5. Which machines do you have access to for testing (NVIDIA, Intel, Mac)?
