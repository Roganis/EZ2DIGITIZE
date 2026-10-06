# VGGT camera placement plugin

Places the cameras with [VGGT](https://github.com/facebookresearch/vggt)
(Meta, CVPR 2025) instead of COLMAP's features, matching and mapping. VGGT
predicts every camera, and a depth map per photo, in one pass of a neural
network, with no feature matching. That helps where matching struggles
(little texture, few photos, wide gaps between them), and it is fast on a
GPU. The dense cloud, mesh and splats are then made as usual.

It is not part of EZ2DIGITIZE and can't be: VGGT's code is under the VGGT
License, and its weights are either VGGT-1B-Commercial (VGGT License,
commercial use allowed except military, access approved by Meta) or VGGT-1B
(CC-BY-NC-4.0, non-commercial use only). The app shows these licenses
before you can use the plugin. This folder holds only the glue
(GPL-3.0-or-later): it doesn't contain VGGT or any weights.

## Install

1. Install the plugin: Settings → Plugins → Install from folder… (this
   folder), or `ez2d plugins install tools/plugins/vggt-poses`. Read and
   accept the licenses.
2. In the installed plugin's folder (Settings → Plugins → Open plugins
   folder, or `ez2d plugins`), make its Python environment with PyTorch and
   VGGT, choosing the weights:

   ```sh
   python3 install_vggt.py --weights commercial
   # AMD GPU on Linux: PyTorch's ROCm build
   python3 install_vggt.py --weights commercial --torch-index https://download.pytorch.org/whl/rocm6.4
   # non-commercial use only, no access request needed:
   python3 install_vggt.py --weights research
   ```

   For the commercial weights, request access on
   [their Hugging Face page](https://huggingface.co/facebook/VGGT-1B-Commercial)
   and set `HF_TOKEN` to a token of that account before starting
   EZ2DIGITIZE. The weights (about 5 GB) are downloaded on the first run.
3. Choose it for camera placement: Settings → Plugins, or
   `ez2d plugins use poses vggt-poses`.

## What to expect

- **A GPU is close to required.** VGGT runs on CUDA (NVIDIA), ROCm (AMD on
  Linux, with the ROCm PyTorch build) or Metal (Apple). On the CPU it works
  but takes very long.
- **Memory grows with the number of photos,** since all of them go through
  at once: roughly 100 to 200 photos on a 16 to 24 GB GPU. Larger sets
  need COLMAP.
- **Masks are not used** by VGGT. The dense cloud and the texture still use
  them.
- **The scale is arbitrary,** as with COLMAP: set it in the 3D view, or use
  the printed markers.

## How it works

`vggt_poses.py` pads each photo to a square and scales it to 518 px, as
VGGT's own COLMAP export does, then runs VGGT's aggregator, camera head and
depth head. `feedforward_colmap.py` (numpy only, tested in
`tests/test_vggt_plugin.py`) writes the COLMAP model:

- one PINHOLE camera per photo, with its intrinsics mapped back to the
  photo's own pixels;
- the poses;
- sparse points from the confident depth pixels. Each point is kept only
  where other photos see it and their depth agrees, because OpenMVS picks
  each photo's neighbours from the points they share.
