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
- **Windows (x64, community-tested):** unzip
  `EZ2DIGITIZE-<version>-windows-x86_64.zip` and run `EZ2DIGITIZE.exe`
  (`ez2d.exe` is the command line). The app isn't signed yet, so Windows
  SmartScreen warns once: More info → Run anyway.

Releases are on the project's
[GitHub releases page](https://github.com/Roganis/EZ2DIGITIZE/releases).

COLMAP, OpenMVS and Brush come inside the app. Video import uses the FFmpeg
installed on your computer (`pacman -S ffmpeg`, `apt install ffmpeg`,
`brew install ffmpeg`). Splats need a GPU with a Vulkan driver (Mesa's
RADV, NVIDIA's, Intel's) or Apple silicon.

## Your first model

1. **Take the photos** (see the [capture guide](CAPTURE.md)): 40 to 80
   photos all around the object, from two heights, everything in focus.
2. **New project** (File → New Project), then **Import photos…** and pick
   the folder. Or **Import video…**, or **From phone…** to send photos over
   Wi-Fi by scanning a QR code, or, if the phone already syncs its photos to
   this computer (Syncthing, iCloud Drive…), to take them from that folder
   as they arrive.
3. Read the **Photo checks** tab: it flags photos that are blurry, a
   different size, from another camera, or that can't be read. Uncheck a
   photo to leave it out.
4. Optional, needed for turntables: in the **Masks** tab, **Make masks**
   keeps only the object in every photo (cleaner, faster). Uncheck any mask
   that cuts off part of the object. To scan the underside too, see
   **Other side…** and the Both sides tab ([capture guide](CAPTURE.md)).
5. For a room or an outdoor scene instead of an object, set **Subject** to
   **Room or outdoor scene** (see the [capture guide](CAPTURE.md)).
   Choose a **Quality**: Fast for a preview (a few minutes), Balanced (the
   default), High for the finest surface (much slower). **Mesh size**
   simplifies the result for the web or a slicer.
6. Press **Build mesh**. The steps list shows progress; the Log tab shows
   the tools' output. On a recent desktop, 60 photos take about 15 minutes
   at Balanced. Better: press **Place cameras** first. The 3D view opens on
   the camera placement, with four tools next to the view's name:
   - **Coverage** shows where photos are missing: a ring per height, gaps
     shaded orange (red when a whole side is missing), and cameras that
     matched few others in orange. Take more photos there.
   - **Crop box** keeps the table out of the model: **Use a crop box**
     puts a box around the object; drag its yellow handles to fit.
   - **Scale** gives the model its real size. If you photographed it on
     the printed marker sheet (**Marker sheet…** here; see the capture
     guide), that is done already. Otherwise measure two points on the
     object (its height, say), press **Pick two points**, click them, type
     the distance and press **Set scale**.
   - **Upright**: if the model lies tilted, **Level: pick 3 points** on the
     mat stands it up.

   Then Build mesh.
7. The finished model opens in the **3D view** tab (turn it with the
   mouse); it also shows the dense cloud, the camera placement and splats.
8. **Open folder** shows the result in `exports/`: OBJ (with its texture),
   GLB, and the other formats you chose under **Save as** (USDZ for AR on
   an iPhone or iPad, glTF, PLY and OFF with vertex colours, STL/3MF for
   printing). The model stands
   upright, centred, on the ground; with the scale set, STL and 3MF are in
   millimetres and OBJ and GLB in metres.

**Build splats** trains Gaussian splats from the same camera positions.
They are exported twice: as Brush's PLY, and as a ten times smaller SPZ
stood upright like the mesh (most splat viewers open either). With **and a mesh
from them** ticked, it also makes a surface through the splats (on the
CPU, with vertex colours instead of a texture): a second mesh to compare
with the textured one, sometimes better on thin or shiny parts.

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
ez2d run ~/scans/skull --splat --splat-mesh         # and a mesh made from them
ez2d status ~/scans/skull
```

`ez2d check` lists the tools and GPUs found; `ez2d --help` the rest.

## Mesh and splat files from elsewhere

**File → Open Model File…** shows any mesh or splat file in the 3D view,
not only a project's: PLY, OBJ, STL, OFF, glTF/GLB and 3MF meshes, PLY,
SPZ, .splat, .ksplat and SOG splats. If it shows lying down or upside
down, change **Up**. **Convert…** saves it in another format: a mesh as
GLB, glTF, OBJ, PLY, STL, 3MF, OFF or USDZ (for AR Quick Look on an iPhone
or iPad), splats as PLY or SPZ. From the command line:

```sh
ez2d view statue.ply                   # opens a window
ez2d convert statue.obj statue.usdz    # textures go along
ez2d convert part.stl part.glb --scale 0.001   # millimetres to metres
ez2d convert scene.ply scene.spz       # splats, about ten times smaller
```

If something goes wrong, see [troubleshooting](TROUBLESHOOTING.md).
