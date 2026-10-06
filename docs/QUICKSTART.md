# Quick start

EZ2DIGITIZE turns photos or a video of a small object into a textured 3D
model (OBJ, GLB, STL, 3MF) and, with a GPU, Gaussian splats. Everything
runs on your computer; nothing is uploaded anywhere.

## Install

- **Linux (x86_64):** download `EZ2DIGITIZE-<version>-x86_64.AppImage`,
  make it executable (`chmod +x`) and run it. It needs glibc 2.35 or newer
  (Ubuntu 22.04, Fedora 36, any current Arch).
- **macOS (Apple Silicon):** unzip `EZ2DIGITIZE-<version>-macos-arm64.zip`.
  The app isn't notarized yet, so remove the quarantine once:
  `xattr -dr com.apple.quarantine EZ2DIGITIZE.app`.

COLMAP, OpenMVS and Brush come inside the app. Video import uses the FFmpeg
installed on your computer (`pacman -S ffmpeg`, `apt install ffmpeg`,
`brew install ffmpeg`). Splats need a GPU with a Vulkan driver (Mesa's
RADV, NVIDIA's, Intel's) or Apple silicon.

## Your first model

1. **Take the photos** (see the [capture guide](CAPTURE.md)): 40 to 80
   photos all around the object, from two heights, everything in focus.
2. **New project** (File → New Project), then **Import photos…** and pick
   the folder. Or **Import video…**, or **From phone…** to send photos over
   Wi-Fi by scanning a QR code.
3. Read the **Photo checks** tab: it flags photos that are blurry, a
   different size, from another camera, or that can't be read. Uncheck a
   photo to leave it out.
4. Optional, needed for turntables: in the **Masks** tab, **Make masks**
   keeps only the object in every photo (cleaner, faster). Uncheck any mask
   that cuts off part of the object. To scan the underside too, see
   **Other side…** and the Both sides tab ([capture guide](CAPTURE.md)).
5. Choose a **Quality**: Fast for a preview (a few minutes), Balanced (the
   default), High for the finest surface (much slower). **Mesh size**
   simplifies the result for the web or a slicer.
6. Press **Build mesh**. The steps list shows progress; the Log tab shows
   the tools' output. On a recent desktop, 60 photos take about 15 minutes
   at Balanced.
7. **Open folder** shows the result in `exports/`: OBJ (with its texture),
   GLB, and STL/3MF for printing if you chose them. The model stands
   upright, centred, on the ground.

**Build splats** trains Gaussian splats from the same camera positions
(PLY, for splat viewers).

## From the command line

The same app runs headless (the AppImage or
`EZ2DIGITIZE.app/Contents/MacOS/EZ2DIGITIZE` with a command):

```sh
ez2d new ~/scans/skull
ez2d import ~/scans/skull ~/Pictures/skull          # or a video, --frames 100
ez2d photos ~/scans/skull                           # photo checks
ez2d masks ~/scans/skull                            # automatic masks
ez2d run ~/scans/skull --quality balanced --export obj,glb,stl
ez2d run ~/scans/skull --splat                      # Gaussian splats
ez2d status ~/scans/skull
```

`ez2d check` lists the tools and GPUs found; `ez2d --help` the rest.

If something goes wrong, see [troubleshooting](TROUBLESHOOTING.md).
