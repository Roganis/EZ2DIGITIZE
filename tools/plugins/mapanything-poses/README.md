# MapAnything camera placement plugin

Places the cameras with [MapAnything](https://github.com/facebookresearch/map-anything)
(Meta, 2025) instead of COLMAP's features, matching and mapping. Like
[VGGT](../vggt-poses), it predicts every camera and a depth map per photo
in one pass of a neural network, with no feature matching. That helps where
matching struggles (little texture, few photos, wide gaps between them).
The dense cloud, mesh and splats are then made as usual.

How it differs from the VGGT plugin:

- **It uses what the photos already say.** MapAnything can take known
  camera parameters as well as the photos. The plugin passes each photo's
  focal length from EXIF (the 35 mm equivalent phones and cameras write)
  when every photo has one. Video frames have none, so then MapAnything
  estimates it, as VGGT always does.
- **It can start from known camera poses.** A video recorded by an app that
  tracks the phone (ARCore) and writes the poses into the video (Google's
  CAMM track, with orientation and position) gives every frame a pose. The
  plugin passes them to MapAnything when every photo has one from the same
  video, leaving the scale to MapAnything since the format doesn't promise
  metres.
- **Weights for commercial use with no access request.** The code is under
  Apache-2.0, and so are the `map-anything-apache` weights, trained on data
  that allows commercial use.
- **More photos per run.** It processes the per-photo part of the network
  in small batches, so memory grows more slowly with the number of photos.

It is not part of EZ2DIGITIZE: it needs PyTorch and a GPU, and its other
weights, `map-anything`, are under CC-BY-NC-4.0. The app shows these
licenses before you can use the plugin. This folder holds only the glue
(GPL-3.0-or-later): it doesn't contain MapAnything or any weights.

## Install

1. Install the plugin: Settings → Plugins → Install from folder… (this
   folder), or `ez2d plugins install tools/plugins/mapanything-poses`. Read
   and accept the licenses.
2. In the installed plugin's folder (Settings → Plugins → Open plugins
   folder, or `ez2d plugins`), make its Python environment with PyTorch and
   MapAnything:

   ```sh
   python3 install_mapanything.py
   # AMD GPU on Linux: PyTorch's ROCm build
   python3 install_mapanything.py --torch-index https://download.pytorch.org/whl/rocm6.4
   # non-commercial use only; Meta reports it does better:
   python3 install_mapanything.py --weights research
   # let MapAnything estimate the focal length even when EXIF has one:
   python3 install_mapanything.py --no-exif-focal
   ```

   The first run downloads the weights (about 5 GB) from Hugging Face, and
   the code of DINOv2 (Apache-2.0), the image encoder inside MapAnything,
   from GitHub.
3. Choose it for camera placement: Settings → Plugins, or
   `ez2d plugins use poses mapanything-poses`.

## What to expect

- **A GPU is close to required.** MapAnything runs on CUDA (NVIDIA), ROCm
  (AMD on Linux, with the ROCm PyTorch build) or Metal (Apple). On the CPU
  it works but takes very long, and the model alone needs about 5 GB of
  memory.
- **Masks are not used** by MapAnything. The dense cloud and the texture
  still use them.
- **The scale is MapAnything's estimate in metres,** and only an estimate.
  The log shows it, but the app still counts the model as unscaled: set
  the scale in the 3D view, or use the printed markers.
- **No refinement.** Like VGGT, the cameras come straight from the network,
  at about 518 px. They are less precise than COLMAP's refined ones, which
  shows in the finest mesh detail.

## How it works

`mapanything_inputs.py` reads each photo as stored, without turning it by
its EXIF orientation, because COLMAP reads it that way and the model must
match. It then scales and crops each photo to the size MapAnything's own
loader would choose: 518 px on the long side, at the supported aspect
ratio nearest the photos' average. `mapanything_poses.py` passes the
photos, and the focal lengths at that size, to MapAnything's `infer` and
keeps the depth it doesn't flag as ambiguous or an edge.

`feedforward_colmap.py` (numpy only, shared with the VGGT plugin, which
has an identical copy, and tested in `tests/test_vggt_plugin.py`) writes
the COLMAP model:

- one PINHOLE camera per photo, with MapAnything's intrinsics mapped back
  to the photo's own pixels;
- the poses;
- sparse points from the confident depth pixels. Each point is kept only
  where other photos see it and their depth agrees, because OpenMVS picks
  each photo's neighbours from the points they share.

Checked so far: the whole script, run on the CPU against MapAnything's real
code (the pinned commit) with random weights, writes a model that COLMAP
reads. A run with the real weights on a GPU, and a comparison with COLMAP
and VGGT on the Phase 1 datasets, wait for the reference machines.
