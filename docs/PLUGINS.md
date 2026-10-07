# Plugins

A plugin adds a reconstruction tool that can't come with EZ2DIGITIZE. Often
it is research code or model weights under a license that forbids
commercial use (VGGT, MASt3R-SfM, many Gaussian splat trainers), which
EZ2DIGITIZE never bundles (see the license rules in
[ROADMAP.md](ROADMAP.md)). A plugin replaces one step of the pipeline, and
the rest of the pipeline works as before.

## Using plugins

In the app: **Settings → Plugins…**. Install a plugin from its folder or a
`.zip`, read its licenses, accept them, and choose it for its step. In a
terminal:

```sh
ez2d plugins install ~/Downloads/vggt-plugin.zip
ez2d plugins license vggt          # read them
ez2d plugins accept vggt
ez2d plugins use poses vggt        # camera placement by the plugin from now on
ez2d run ~/scans/skull
ez2d plugins use poses built-in    # back to COLMAP
ez2d plugins                       # what is installed and used
```

Before you accept, check:

- **The licenses.** A plugin lists one for each part, and model weights
  often have a different license from the code. "Not known to be a free
  license" usually means non-commercial or research-only use, which can
  also cover what you make with it. The app shows the texts, but it can't
  tell you what they allow.
- **That you trust it.** A plugin is a program that runs on your computer
  with your rights, like anything else you install.

The choice holds for every project until you change it. If a chosen plugin
is removed, breaks, or comes back from an update with different license
texts, runs stop with a message and don't fall back to the built-in tool,
because that would quietly give a different result. Each stage's
`stage.json` records the plugin and its version (`plugin:<id>`), and
diagnostics zips list the installed plugins.

Plugins are installed in the user's data folder, one folder each:

| System  | Folder                                                  |
| ------- | ------------------------------------------------------- |
| Linux   | `~/.local/share/ez2digitize/plugins` (`$XDG_DATA_HOME`) |
| macOS   | `~/Library/Application Support/ez2digitize/plugins`     |
| Windows | `%APPDATA%\ez2digitize\plugins`                         |

`EZ2D_PLUGINS_DIR` overrides the folder. `plugins.json` in it records
the choices, and which license texts were accepted (their hashes).

## Writing a plugin

A plugin is a folder holding `ez2d-plugin.toml`, its license files, and
whatever it runs. EZ2DIGITIZE runs the command in the manifest as a
subprocess, like its built-in tools. It never imports the plugin, so the
plugin can be in any language and bring its own Python, PyTorch (ROCm,
Metal, CUDA) and weights. The working folder is the stage folder; the
output and the log go there. Cancelling stops the whole process group.

[`tools/plugins/example-poses`](../tools/plugins/example-poses) is a
working plugin to start from.
[`tools/plugins/vggt-poses`](../tools/plugins/vggt-poses) is a real one: it
places the cameras with VGGT, a neural network under its own licenses,
in a Python environment of its own (see its README).
[`tools/plugins/mapanything-poses`](../tools/plugins/mapanything-poses)
does the same with MapAnything, which also takes the focal length from the
photos' EXIF and has Apache-2.0 weights.

```toml
api = 1                          # this format; required
id = "vggt"                      # lower case, digits, . _ -; also its folder name
name = "VGGT camera placement"
version = "1.0.0"                # bump it when the plugin changes: results are cached
provides = "poses"               # or "splats"
description = "Feed-forward camera poses from VGGT-1B."
homepage = "https://example.org/vggt-plugin"
command = ["bin/run", "--images", "{captures}", "--list", "{image_list}",
           "--out", "{output}"]
with_masks = ["--masks", "{masks}"]  # added when the project has masks (poses only)
gpu = true                       # record the GPU in the manifest; check there is one
platforms = ["linux", "macos"]   # leave out for all (linux, macos, windows)
pty = false                      # true: run under a terminal (Unix), for tools that
                                 # only print progress to one

[progress]                       # optional: progress bar from the output
pattern = 'step (?P<done>\d+)/(?P<total>\d+)'  # or the first two groups
message = "Placing cameras (VGGT)"

[[license]]                      # at least one; one per part
covers = "code"
spdx = "Apache-2.0"              # SPDX identifier or expression; LicenseRef-... else
file = "LICENSE"

[[license]]
covers = "model weights"
spdx = "CC-BY-NC-4.0"
file = "WEIGHTS-LICENSE.txt"
```

The command's first item is the program. A path such as `bin/run` is
relative to the plugin's folder (on Windows, `bin/run` finds `run.exe`,
`run.cmd` or `run.bat`). A bare name such as `python3` is looked up on
`PATH`. Placeholders in braces are filled in for every run (write `{{` for
a brace):

| Placeholder        | Value                                                        |
| ------------------ | ------------------------------------------------------------ |
| `{plugin}`         | the plugin's folder                                          |
| `{output}`         | the stage folder: write the results here                     |
| `{threads}`        | the number of CPU threads                                    |
| `{colmap}`         | the COLMAP the app uses (bundled, or set in its settings)    |
| poses: `{captures}`   | the project's `captures/` folder                          |
| poses: `{image_list}` | a file naming one photo per line, relative to `{captures}` |
| poses: `{masks}`      | (in `with_masks`) a mask per photo                         |
| poses: `{priors}`     | a JSON file of what is known of the cameras (see below)    |
| splats: `{dataset}`   | the COLMAP dataset to train on                             |
| splats: `{steps}`, `{max_resolution}`, `{max_splats}`, `{sh_degree}` | the quality preset's values (Brush's meaning) |

### Camera placement (`provides = "poses"`)

This step replaces COLMAP's features, matching and mapping. It runs as the
`mapping` stage.

- **In:** the photos are `{captures}/<name>` for each name in
  `{image_list}`, which looks like `<capture id>/IMG_0001.jpg`. The files
  are the originals: rotate by EXIF yourself if your model needs to.
  With `with_masks` and a masked project, `{masks}/<name>.png` is a mask
  of the photo's size for every photo. White is the object, black is to
  be ignored.
- **Known poses:** `{priors}` is a JSON file of the camera poses already
  known, from a video whose motion track records them (CAMM's 6DoF samples,
  written by ARCore-style tracking apps) or a motion log recorded with the
  photos or video (`ez2digitize.motion_log`):

  ```json
  {"version": 1, "images": {"<capture id>/frame_0001.jpg": {
    "camera_to_world": [[r, r, r, x], [r, r, r, y], [r, r, r, z]],
    "frame": "<capture id>", "metric": false}}}
  ```

  OpenCV camera axes (x right, y down, z forward), camera to world, the
  last column the camera centre. Poses with different `frame`s are in
  different worlds (each recording has its own) and can't be mixed;
  `metric` true means metres (a motion log that says so), false that the
  unit isn't guaranteed (CAMM leaves it to the app). Photos without a known
  pose are absent, and the file is written (with no images) for every run.
- **Out:** a binary COLMAP model in `{output}/sparse/0`: `cameras.bin`,
  `images.bin` and `points3D.bin` (the points may be few, but the file must
  exist). Name each image exactly as in the list. If the photos fall into
  separate groups, write `sparse/1` and so on; the largest is used. For a
  text model, `colmap model_converter --output_type BIN` converts it.
  Cameras may be any COLMAP camera model; the next step undistorts them.
- **Then:** COLMAP undistorts the photos with this model, and the dense
  cloud, mesh, splats, coverage notes, crop box and scale all work on it.
  Photos the model leaves out count as not placed.
- **Refined, if asked:** with "Refine the plugin's camera placement with
  COLMAP" (`ez2d run --refine-poses`), the plugin runs as the `poses`
  stage instead, and COLMAP takes its model as the start: features, then
  matching of only the photos its cameras say overlap (each with its 20
  nearest looking within 60° of the same way), triangulation with its
  poses, and COLMAP's own refinement of cameras, focal length and points.
  Photos it placed too far off to get points are placed again by COLMAP.
  A rough but complete placement is all a plugin then needs to give.

### Splats (`provides = "splats"`)

This step replaces Brush. It runs as the `splat` stage, after the camera
placement (COLMAP's or a plugin's).

- **In:** `{dataset}` is the dataset Brush gets: `images/<capture
  id>/<file>` (undistorted, pinhole) and `sparse/0/` (their COLMAP model).
  On a masked project it also has `images/masks/<file stem>.png`, the
  masks warped to the undistorted photos.
- **Out:** `{output}/splat.ply` in the usual 3DGS PLY layout (positions,
  `f_dc_*`, optional `f_rest_*`, `opacity`, `scale_*`, `rot_*`). It is
  exported as PLY and SPZ and shown in the 3D view like Brush's.

### Packaging

- Ship the plugin as a folder, or as a `.zip` of one (the manifest at the
  top, or in a single folder inside). The `.zip` keeps executable bits.
  Paths that leave the folder, and symbolic links, are refused.
- If the plugin needs downloads (weights, a Python environment), let its
  program fetch them on the first run, or give it a setup script to run
  once. Nothing runs at install time.
- From the packaged app, a plugin's programs start with the computer's own
  library path, not the app's, so a system Python or PyTorch finds its own
  libraries.
- Exit with a non-zero code on failure, and print why: the last lines of
  the log are shown to the user.
