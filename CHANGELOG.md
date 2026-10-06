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
- Gaussian splats with Brush, from the same camera placement.
- Automatic masks with a review grid, or imported masks, used by every step
  that can.
- Two-sided scans: turn the object over, mark the second side, and the app
  checks and reports how the sides joined.
- Photo checks on import (blur, size, camera); leave photos out.
- Videos import as their sharpest frames; iPhone HEIC photos as JPEG copies.
- Phone upload over Wi-Fi from a QR code.
- Each camera and zoom setting calibrated separately.

### The 3D view

- Camera placement, dense cloud, textured mesh and splats in the app, with
  four tools on the camera placement and the dense cloud, one at a time:
  - Coverage on the camera rings: where photos are missing, and the photos
    that matched few others.
  - A crop box: the dense cloud and mesh keep only what is inside.
  - Real-world scale from two picked points and their measured distance.
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

### Packages

- Linux AppImage, macOS app (Apple Silicon) and a portable Windows zip, each
  with COLMAP 4.2.1, OpenMVS v2.4.0 and Brush v0.3.0 inside. The macOS app
  isn't notarized and the Windows app isn't signed yet.
- `ez2d`, a command line for everything the app does.
