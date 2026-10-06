# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Example camera placement plugin: COLMAP features, matching and mapper.

What a camera placement plugin gets and must write (docs/PLUGINS.md):

- the photos: `--images` is the project's captures folder, and the image
  list names each photo relative to it (`<capture id>/IMG_0001.jpg`);
- with masks, `--masks`: a PNG for every photo at `<masks>/<name>.png`,
  black where to ignore;
- it writes a binary COLMAP model to `<output>/sparse/0` (or several,
  `sparse/1`... if the photos fall apart; the largest is used), with the
  images named exactly as in the list.

Progress is printed as "step N of 3" for the [progress] pattern.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", default="-1")
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--colmap", default="colmap")
    args = parser.parse_args()
    colmap = args.colmap
    database = args.output / "example.db"
    extract = [
        colmap, "feature_extractor",
        "--database_path", database,
        "--image_path", args.images,
        "--image_list_path", args.image_list,
        "--ImageReader.single_camera_per_folder", "1",
        "--FeatureExtraction.use_gpu", "0",
        "--FeatureExtraction.num_threads", args.threads,
    ]  # fmt: skip
    if args.masks is not None:
        extract += ["--ImageReader.mask_path", args.masks]
    steps = [
        extract,
        [colmap, "exhaustive_matcher", "--database_path", database,
         "--FeatureMatching.use_gpu", "0", "--FeatureMatching.num_threads", args.threads],
        [colmap, "mapper", "--database_path", database, "--image_path", args.images,
         "--output_path", args.output / "sparse"],
    ]  # fmt: skip
    for n, command in enumerate(steps, 1):
        print(f"step {n} of {len(steps)}: {command[1]}", flush=True)
        result = subprocess.run([str(part) for part in command], check=False)  # noqa: S603
        if result.returncode != 0:
            print(f"{command[1]} failed with exit code {result.returncode}", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
