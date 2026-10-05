# Viewer spike

Phase 1 question (d): can one embedded viewer show splats, textured meshes
and point clouds inside the Qt app, fast enough on the reference machines?

The candidate is a web viewer inside `QWebEngineView`: three.js for meshes
and point clouds, [Spark](https://sparkjs.dev) for Gaussian splats (both
MIT). The page and the model are served through a custom `ez2d://` URL
scheme from Python, the same way the app would do it.

```sh
uv run python tools/spikes/viewer/fetch_vendor.py          # once: pinned three.js + Spark
uv run python tools/spikes/viewer/viewer_spike.py MODEL.ply  # opens a window
```

`MODEL.ply` can be any PLY the pipeline produces; the kind is detected from
its header:

| File | Shown as |
|---|---|
| Brush export (`export_*.ply`) | Gaussian splat |
| OpenMVS `scene_textured.ply` (+ `scene_textured0.png`) | textured mesh |
| OpenMVS `scene_dense.ply`, COLMAP `sparse.ply` | point cloud |

Mouse: left drag rotates, right drag pans, wheel zooms.

## What to measure on the GRE and the M1

Use the outputs of the walkaround dataset (paths are in each `run.json`):

```sh
V="uv run python tools/spikes/viewer/viewer_spike.py"
$V ~/ez2d-feasibility/<machine>/walkaround/brush/brush-default/export/export_30000.ply --measure
$V ~/ez2d-feasibility/<machine>/walkaround/openmvs/lvl1/scene_textured.ply --measure
$V ~/ez2d-feasibility/<machine>/walkaround/openmvs/lvl1/scene_dense.ply --measure
```

`--measure` prints JSON lines: the GPU the page got, load time, and the
average frame rate over 3 s, then exits. Also open each file without
`--measure` and rotate it for a while: note stutter, how it looks, and the
memory of the `QtWebEngineProcess` processes (`htop` / Activity Monitor).
For a size stress test, `make_test_splat.py out.ply --count 3000000` writes
a 3-million-splat file. Record results in `docs/feasibility/FINDINGS.md`.

## Results so far (CPU-only container, software rendering)

Mesa llvmpipe through ANGLE, so frame rates say nothing about real GPUs:

| Model | Size | Load | fps (software) |
|---|---|---|---|
| Textured mesh (synthetic) | 26k faces | 0.1-0.5 s | 25-37 |
| Sparse point cloud | 5k points | 0.05 s | 60 |
| Splat (Brush, 300 steps) | 5k splats | 0.14 s | 35 |
| Synthetic splat | 1M splats, 237 MB | 3.3 s | 2.4 |

## Notes for the real implementation

- The custom scheme needs the `SecureScheme`, `LocalAccessAllowed`,
  `CorsEnabled` and `FetchApiAllowed` flags (the last one since Qt 6.6;
  without it `fetch()`, used by both loaders, fails). It must be registered
  before `QApplication` is created.
- Chromium refuses WebGL on software renderers (its GPU blocklist). That only
  matters for headless tests, which need
  `QTWEBENGINE_CHROMIUM_FLAGS=--ignore-gpu-blocklist`; don't ship that flag.
- OpenMVS textured PLYs carry per-face UVs with a bottom-left origin, which
  three.js's PLYLoader and default `flipY` handle as is. Faces no camera saw
  get OpenMVS's "empty" colour (orange by default); on the synthetic box
  that's the bottom face.
- Framing uses the 2nd-98th percentile of the points: SfM/MVS clouds always
  have outliers that otherwise shrink the object to a dot.
- Package size: QtWebEngine is the big cost. In the PySide6 6.11 wheel,
  `libQt6WebEngineCore.so` alone is about 195 MB (uncompressed). The
  packaging spike measures what an installer actually grows by.
- The alternative considered was a native viewer (pygfx/wgpu-py or VTK).
  Both handle meshes and point clouds well, but as far as I know neither
  renders 3DGS splats (anisotropic Gaussians with spherical harmonics) out
  of the box; VTK's point-Gaussian mapper is not the same thing. Check
  before deciding: if that's still true, splats would need a custom
  renderer. The web route gets all three from maintained libraries; the cost
  is QtWebEngine's size.
