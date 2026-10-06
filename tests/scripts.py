# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stand-in executables for the tests, on every platform.

POSIX runs a script with a `#!` line directly. Windows doesn't: there the
script is a `name.cmd` (the app finds it for `name`) whose first line runs
the file itself with this Python, `-x` skipping that line. Like a POSIX
script, the file holds the whole program, so its hash changes with the
body and not with its folder (the stage tests rely on both).
"""

import sys
from pathlib import Path

# Brush's stand-in: checks the dataset, prints progress, writes two splats.
# Splat plugins in the tests run it too, with Brush's arguments.
FAKE_BRUSH = """
import os, struct, sys
from pathlib import Path
args = sys.argv[1:]
if args == ["--version"]:
    print("brush-cli 0.3.0")
    sys.exit(0)
if os.environ.get("FAKE_FAIL") == "brush":
    sys.exit(1)
opt = lambda name: args[args.index(name) + 1]
dataset = Path(args[0])
assert (dataset / "sparse" / "0").is_dir() and (dataset / "images").is_dir(), "bad dataset"
steps = int(opt("--total-steps"))
masks = sorted((dataset / "images" / "masks").glob("*"))
assert all(m.resolve().is_file() for m in masks), "dangling mask link"
print(f"masks: {len(masks)}", flush=True)
print("\\x1b[34mi\\x1b[0m Completed loading", flush=True)
for done in (steps // 2, steps):
    print(f"[1s] \\x1b[36m###\\x1b[0m   {done}/{steps}   Steps (9/s, 0s remaining)", flush=True)
# Two splats, SH degree 0, in Brush's property order.
names = ["f_dc_0", "f_dc_1", "f_dc_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3",
         "scale_0", "scale_1", "scale_2", "x", "y", "z"]
header = "ply\\nformat binary_little_endian 1.0\\ncomment Exported from Brush\\n"
header += "element vertex 2\\n" + "".join(f"property float {n}\\n" for n in names)
header += "end_header\\n"
rows = [[0.1, 0.2, 0.3, 2.0, 1, 0, 0, 0, -3, -3, -3, 0, 0, 0],
        [-0.1, 0.0, 0.1, 2.0, 1, 0, 0, 0, -3, -3, -3, 1, 1, 1]]
body = struct.pack("<28f", *rows[0], *rows[1])
(Path(opt("--export-path")) / opt("--export-name")).write_bytes(header.encode() + body)
"""

# A camera placement plugin's stand-in: places every listed photo in a
# COLMAP model (one SIMPLE_RADIAL camera, 8x6 pixels) and says which masks
# it got. FAKE_FAIL=plugin makes it fail.
FAKE_POSES = """
import os, struct, sys
from pathlib import Path
args = sys.argv[1:]
opt = lambda name: args[args.index(name) + 1]
if os.environ.get("FAKE_FAIL") == "plugin":
    print("out of memory", flush=True)
    sys.exit(1)
names = Path(opt("--list")).read_text().split()
assert all((Path(opt("--images")) / n).is_file() for n in names), "missing photo"
if "--masks" in args:
    folder = Path(opt("--masks"))
    masks = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.png"))
    print("masks: " + " ".join(masks), flush=True)
model = Path(opt("--output")) / "sparse" / "0"
model.mkdir(parents=True)
(model / "cameras.bin").write_bytes(struct.pack("<QIiQQ4d", 1, 1, 2, 8, 6, 7.0, 4.0, 3.0, 0.01))
images = struct.pack("<Q", len(names))
for image_id, name in enumerate(names, 1):
    images += struct.pack("<I7dI", image_id, 1, 0, 0, 0, 0, 0, 0, 1)
    images += name.encode() + b"\\0" + struct.pack("<Q", 0)
(model / "images.bin").write_bytes(images)
(model / "points3D.bin").write_bytes(struct.pack("<Q", 0))
for n in range(1, 3):
    print(f"placing {n} of 2", flush=True)
"""


def python_script(path: Path, body: str) -> Path:
    """An executable at `path` running `body` with this Python; returns what to run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        launcher = path if path.suffix == ".cmd" else path.with_name(path.name + ".cmd")
        # cmd must stop before the Python lines, passing Python's exit code on:
        # "exit /b" alone would end with 0, and %errorlevel% is read when the
        # line is parsed, so "call" reads it again after Python ran.
        launcher.write_text(
            f'@"{sys.executable}" -x "%~f0" %* & call exit /b %%errorlevel%%\n{body}'
        )
        return launcher
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)
    return path


def printing_script(path: Path, text: str) -> Path:
    """An executable that prints `text`, whatever its arguments."""
    return python_script(path, f"print({text!r})\n")


def make_plugin(
    folder: Path,
    *,
    provides: str,
    script: str,
    command: list[str],
    plugin_id: str | None = None,
    licenses: tuple[tuple[str, str], ...] = (("code", "MIT"),),
    extra: str = "",
) -> Path:
    """A plugin folder: `bin/run` running `script`, the manifest, a file per license."""
    folder.mkdir(parents=True, exist_ok=True)
    python_script(folder / "bin" / "run", script)
    toml = [
        "api = 1",
        f'id = "{plugin_id or folder.name}"',
        f'name = "Test {provides}"',
        'version = "1.0"',
        f'provides = "{provides}"',
        "command = [" + ", ".join(f"'{arg}'" for arg in command) + "]",
        extra,
    ]
    for n, (covers, spdx) in enumerate(licenses):
        (folder / f"LICENSE-{n}.txt").write_text(f"{spdx} terms for the {covers}\n")
        toml += ["[[license]]", f'covers = "{covers}"', f'spdx = "{spdx}"']
        toml.append(f'file = "LICENSE-{n}.txt"')
    (folder / "ez2d-plugin.toml").write_text("\n".join(toml) + "\n")
    return folder
