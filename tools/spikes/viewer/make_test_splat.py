# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Write a synthetic Gaussian splat PLY (standard 3DGS layout) for load tests.

    uv run --group feasibility python tools/spikes/viewer/make_test_splat.py out.ply --count 1000000

The splats form a noisy sphere shell; it tests file size and parsing speed,
not visual quality. 3 SH degrees means 62 floats (248 bytes) per splat, the
same layout Brush and the reference 3DGS code export.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FIELDS = (
    ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    + [f"f_rest_{i}" for i in range(45)]
    + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
)


def make(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = np.zeros((count, len(FIELDS)), dtype=np.float32)
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    data[:, 0:3] = direction * rng.normal(1.0, 0.02, size=(count, 1))
    data[:, 6:9] = rng.normal(0, 1, size=(count, 3))  # DC colour (SH space)
    data[:, 54] = 2.0  # opacity logit
    data[:, 55:58] = np.log(0.004)  # scale (log)
    data[:, 58] = 1.0  # identity rotation quaternion
    return data


def write(path: Path, data: np.ndarray) -> None:
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
    header += [f"property float {name}" for name in FIELDS]
    header.append("end_header")
    with path.open("wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(data.astype("<f4").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--count", type=int, default=1_000_000)
    args = parser.parse_args()
    write(args.out, make(args.count))
    print(f"wrote {args.count:,} splats to {args.out} ({args.out.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
