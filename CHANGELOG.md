# Changelog

What changed in each release of EZ2DIGITIZE. New entries go under
Unreleased as the work lands; `tools/packaging/release.py prepare` turns
that section into a release's (see docs/RELEASING.md). Versions follow
[semantic versioning](https://semver.org); until 1.0, a minor version may
change project files or settings (old projects are migrated).

## [Unreleased]

The first release.

### Making a model

- Photos or video of a small object in, a textured mesh out (COLMAP for the
  camera placement, OpenMVS for the dense cloud, mesh and texture), on the
  CPU or any GPU: no CUDA needed.
- Quality presets (fast, balanced, high) and a mesh size setting; each step
  is cached and resumes where it stopped.
- Gaussian splats with Brush, from the same camera placement, and
  optionally a mesh made from them (vertex colours, on the CPU), to
  compare with the textured mesh.
- Automatic masks with a review grid, or imported masks, used by every step
  that can.
- Two-sided scans: turn the object over, mark the second side, and the app
  checks and reports how the sides joined.
- Photo checks on import (blur, size, camera, exposure changes); leave
  photos out. `ez2d photos --exposure` shows how the camera's automatic
  exposure varied and whether those photos were placed less often.
- Videos import as their sharpest frames; iPhone HEIC photos as JPEG copies.
- Phone upload over Wi-Fi from a QR code, or from the folder a phone syncs
  its photos to (Syncthing, iCloud Drive...), imported once they have all
  arrived.
- Each camera and zoom setting calibrated separately.
- Learned features as an option (Advanced → Features, `ez2d run --features
  aliked`): ALIKED matched with LightGlue, which hold up better than SIFT on
  weak texture and large changes of viewpoint; slower on the CPU, models
  downloaded once.
- Rooms and outdoor scenes as well as small objects (**Subject**), and
  large photo sets: beyond 200 photos, they are paired by GPS position, by
  how alike they look (with COLMAP's vocabulary tree, downloaded once) or
  in the order they were taken, instead of every pair.

### The 3D view

- Camera placement, dense cloud, textured mesh and splats in the app, with
  four tools on the camera placement and the dense cloud, one at a time:
  - Coverage on the camera rings: where photos are missing, and the photos
    that matched few others.
  - A crop box: the dense cloud and mesh keep only what is inside.
  - Real-world scale from two picked points and their measured distance, or
    by itself from a printed sheet of markers in the photos.
  - Upright: which way is up comes from how the photos were held, or is
    set by hand (level on three points, tip, turn).

### Exports

- OBJ and GLB (textured), STL and 3MF for printing (in millimetres once the
  scale is set), the dense point cloud, and the splats; stood upright,
  centred and on the ground.
- Splats also as SPZ, about a tenth of the PLY's size, stood upright like
  the mesh.
- Common backend failures explained in plain words, and a diagnostics zip
  for bug reports.

### Plugins

- Plugins for tools that can't come with the app (such as research code or
  model weights for non-commercial use only): camera placement or splat
  training by a program you install, with its licenses shown and accepted
  first (Settings → Plugins, `ez2d plugins`). See docs/PLUGINS.md.
- A VGGT plugin (tools/plugins/vggt-poses): cameras placed by Meta's VGGT
  network in one pass, on a GPU, under VGGT's own licenses.

### Packages

- Linux AppImage, macOS app (Apple Silicon) and a portable Windows zip, each
  with COLMAP 4.2.1, OpenMVS v2.4.0 and Brush v0.3.0 inside. The macOS app
  isn't notarized and the Windows app isn't signed yet.
- `ez2d`, a command line for everything the app does.
- Programs from the computer rather than the app (a system FFmpeg, a
  COLMAP chosen in the settings, plugins) run with the computer's own
  libraries, so the packaged app doesn't break them.
