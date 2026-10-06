# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Set up the MapAnything plugin: a Python environment with PyTorch and MapAnything.

Run it once, in the installed plugin's folder, with Python 3.10 or newer
(not EZ2DIGITIZE's own; the plugin brings its own environment):

    python3 install_mapanything.py                     # Apache-2.0 weights
    python3 install_mapanything.py --torch-index https://download.pytorch.org/whl/rocm6.4

`ez2d plugins` shows where plugins are installed. The PyTorch build decides
the GPU: the default one uses CUDA (NVIDIA) on Linux and Windows and Metal
on a Mac; for an AMD GPU on Linux pass PyTorch's ROCm index. The weights
(about 5 GB) are downloaded on the first run, with no account needed:

- apache (the default): map-anything-apache, under Apache-2.0, trained on
  data that allows commercial use.
- research: map-anything, under CC-BY-NC-4.0: non-commercial use only. Meta
  says it does somewhat better.

Run it again to change the choice; --no-exif-focal makes MapAnything
estimate the focal length itself instead of taking it from the photos.
"""

import argparse
import json
import subprocess
import sys
import venv
from pathlib import Path

# MapAnything as this plugin was written against (Apache-2.0).
MAPANYTHING = (
    "mapanything @ git+https://github.com/facebookresearch/map-anything"
    "@3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", choices=["apache", "research"], default="apache")
    parser.add_argument("--torch-index", help="PyTorch package index (e.g. its ROCm one)")
    parser.add_argument(
        "--no-exif-focal",
        action="store_true",
        help="don't give MapAnything the focal length from the photos' EXIF",
    )
    args = parser.parse_args()
    if sys.version_info < (3, 10):  # noqa: UP036 - run by any Python the user has
        print("MapAnything needs Python 3.10 or newer")
        return 1
    here = Path(__file__).resolve().parent
    env = here / ".venv"
    if not env.exists():
        print(f"creating {env}")
        venv.create(env, with_pip=True)
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    torch = [str(python), "-m", "pip", "install", "torch", "torchvision"]
    if args.torch_index:
        torch += ["--index-url", args.torch_index]
    for command in (torch, [str(python), "-m", "pip", "install", MAPANYTHING]):
        print("$", " ".join(command), flush=True)
        if subprocess.run(command, check=False).returncode != 0:  # noqa: S603 - fixed commands
            return 1
    settings = {"weights": args.weights, "exif_focal": not args.no_exif_focal}
    (here / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"done: the plugin uses the {args.weights} weights", end="")
    print(", and the focal length from EXIF" if settings["exif_focal"] else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
