# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Set up the VGGT plugin: a Python environment with PyTorch and VGGT, and the weights.

Run it once, in the installed plugin's folder, with Python 3.10 or newer
(not EZ2DIGITIZE's own; the plugin brings its own environment):

    python3 install_vggt.py --weights commercial      # or research
    python3 install_vggt.py --weights commercial --torch-index https://download.pytorch.org/whl/rocm6.4

`ez2d plugins` shows where plugins are installed. The PyTorch build decides
the GPU: the default one uses CUDA (NVIDIA) on Linux and Windows and Metal
on a Mac; for an AMD GPU on Linux pass PyTorch's ROCm index. The weights
(about 5 GB) are downloaded on the first run:

- commercial: VGGT-1B-Commercial, under the VGGT License (commercial use
  allowed, not military). Ask for access on its Hugging Face page and set
  HF_TOKEN to a token of that account before running EZ2DIGITIZE.
- research: VGGT-1B, under CC-BY-NC-4.0: non-commercial use only.
"""

import argparse
import subprocess
import sys
import venv
from pathlib import Path

# VGGT as tested with this plugin (its code is under the VGGT License).
VGGT = (
    "vggt @ git+https://github.com/facebookresearch/vggt@a288dd0f14786c93483e45524328726ab7b1b4ce"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", choices=["commercial", "research"], required=True)
    parser.add_argument("--torch-index", help="PyTorch package index (e.g. its ROCm one)")
    args = parser.parse_args()
    if sys.version_info < (3, 10):  # noqa: UP036 - run by any Python the user has
        print("VGGT needs Python 3.10 or newer")
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
    for command in (torch, [str(python), "-m", "pip", "install", VGGT]):
        print("$", " ".join(command), flush=True)
        if subprocess.run(command, check=False).returncode != 0:  # noqa: S603 - fixed commands
            return 1
    (here / "weights.txt").write_text(args.weights + "\n")
    print(f"done: the plugin uses the {args.weights} weights")
    if args.weights == "commercial":
        print("set HF_TOKEN to a Hugging Face token with access to VGGT-1B-Commercial")
    return 0


if __name__ == "__main__":
    sys.exit(main())
