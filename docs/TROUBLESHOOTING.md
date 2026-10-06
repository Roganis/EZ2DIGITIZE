# Troubleshooting

When a step fails, the status line says what probably happened and what to
try, and the Log tab ends with the tool's last lines (**Open full log** for
all of it). If that doesn't help, use **Help → Export Diagnostics** (or
`ez2d diagnostics PROJECT`) and attach the zip to a bug report: it holds the
logs and settings, never your photos. The logs do contain file paths.

## The app doesn't start

- **AppImage: "fuse: device not found" / "Cannot mount AppImage".** Install
  FUSE 2 (`libfuse2` on Ubuntu, `fuse2` on Arch), or run it with
  `APPIMAGE_EXTRACT_AND_RUN=1 ./EZ2DIGITIZE-*.AppImage`.
- **AppImage: "GLIBC_2.35 not found".** The distribution is older than
  Ubuntu 22.04.
- **macOS: "can't be opened because Apple cannot check it".** The app isn't
  notarized yet: `xattr -dr com.apple.quarantine EZ2DIGITIZE.app`.
- **"Reconstruction tools not found".** Only when running from source:
  build or download the backends (see `tools/backends`) and set them in
  Settings → Reconstruction tools, or `EZ2D_COLMAP` / `EZ2D_OPENMVS_DIR`.
  `ez2d check` shows what was found.

## A step fails

| Message | What to do |
|---|---|
| The step ran out of memory | Choose a lower Quality (or Advanced → Detail), close other programs. On 8 GB machines start with Fast. |
| The disk is full | A run needs a few GB in the project folder. Delete old exports or projects. |
| The cameras could not be placed | Too little overlap or texture: more photos, closer together, a patterned mat; for a turntable, masks (see the [capture guide](CAPTURE.md)). |
| Some photos could not be read | Leave them out in the Photo checks tab. |
| No dense points / No mesh could be built | Usually wrong camera positions: check how many photos were placed (log), try a lower detail level, check the masks don't hide the object. |
| The tool crashed / needs a newer processor | A bug: please report it with Export Diagnostics. |

## The result looks wrong

- **"The two sides did not join".** Only one side made it into the model.
  In the Masks tab, look for masks that kept the table, the stand or
  something next to the object, and drop or fix them; check that every
  photo has a mask (Both sides tab). If they are clean, the sets overlap
  too little: take a low ring of photos for each side, so the object's
  sides show in both.
- **Only part of the object, or a lump of background.** Look at the notes
  in the log after camera placement: "only N of M images were placed", "the
  photos split into separate groups", "no photos from about 120° of the way
  around", "the camera hardly moved". Each points to the capture. The 3D
  view's camera placement shows the same on the rings of cameras: shaded
  gaps where photos are missing, and orange or red cameras that matched
  few others or were placed far off.
- **The table or mat is part of the model.** Expected without masks; the
  export keeps everything the photos saw. Make masks (Masks tab), or set a
  crop box (3D view, camera placement or dense cloud: the Crop box tool),
  and build again.
- **"The crop box was drawn on an earlier camera placement".** The cameras
  were placed again (new photos, other settings), which changes the
  coordinates; set the box again in the 3D view.
- **Part of the object is missing after using masks.** A mask cut it off.
  In the Masks tab, look at "To look at" first, then the rest: uncheck the
  photos whose red tint covers part of the object, and build again.
- **Blotchy or striped texture.** Uneven light between photos, or blurry
  photos (see the photo checks). Development builds before the OpenMVS fix
  of October 2026 gave black blobs and coloured specks: update, and run the
  project again (the texture step re-runs by itself).
- **The model lies on its side, or is tilted.** The upright estimate comes
  from how the photos were held; if the photos were taken at all angles it
  can't tell and keeps the reconstruction's own frame. Correct it in the 3D
  view (camera placement or dense cloud, the Upright tool): **Level: pick 3
  points** and click three points far apart on the mat or the surface the
  object stands on; or **Tip forward** / **Tip sideways** by quarter turns.
  **Turn** sets which way it faces, **Automatic** goes back to the estimate.
  Export again afterwards. (The "Stand the model upright" setting turns all
  of this off.)
- **Wrong size.** Photos alone don't give the size: set the scale in the 3D
  view (camera placement or dense cloud, the Scale tool: Pick two points,
  then their real distance) and export again. STL and 3MF are then in
  millimetres, OBJ and GLB in metres (some programs assume other units on
  import: a model 1000 times too small or large is that). Measure a long
  distance: an error of half a millimetre over 10 mm is 5 % of the size.
- **"The scale was set on an earlier camera placement".** As with the crop
  box: the cameras were placed again, so pick the two points again.

## The 3D view

- **"The 3D view needs WebGL".** The graphics driver isn't one Chromium
  (inside the app) accepts for WebGL. Update the driver (Mesa on Linux);
  in virtual machines and remote desktops, enable 3D acceleration. The
  results are still saved in exports/ and open in any 3D program.
- **The 3D view stays empty on Linux.** Run the app from a terminal and look
  for "sandbox" errors; `QTWEBENGINE_DISABLE_SANDBOX=1` works around a
  system that blocks Chromium's sandbox (the AppImage does this itself on
  Ubuntu 24.04 and later).

## Video, phone and splats

- **"Importing a video needs FFmpeg".** Install FFmpeg 5 or newer, or set
  its location in Settings → Reconstruction tools.
- **The phone can't open the page.** Phone and computer must be on the
  same network (not a guest network that isolates devices), and the
  computer's firewall must allow incoming connections to the app while the
  dialog is open. The address under the QR code can be typed in by hand.
- **"Training splats needs a GPU".** Brush needs a Vulkan driver (Linux:
  Mesa's RADV/ANV, or NVIDIA's) or Apple silicon; `ez2d check` lists the
  GPUs found. A software renderer (llvmpipe) is refused because it would
  take days.

## Masks

- **"Could not download the masking model".** The first Make masks
  downloads it (179 MB) from GitHub. Without internet access, download
  `isnet-general-use.onnx` from the address in the message on another
  computer and save it where the message says (Linux:
  `~/.cache/ez2digitize/models/`, macOS: `~/Library/Caches/ez2digitize/models/`).
- **Masks keep the stand or turntable too.** The model keeps whatever looks
  like part of the object. It is usually harmless; a plain cloth over the
  turntable helps.

## Starting a step again

Steps whose inputs and settings haven't changed are reused. To run them
again anyway: `ez2d run PROJECT --force-from <step>` (for example
`texture`), which also re-runs every step after it. A new version of a
reconstruction tool (a different binary) re-runs the steps it ran by
itself.
