# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small meshes and splats for the model file tests."""

import io

import numpy as np
from PIL import Image as PILImage

from ez2digitize.core import meshfiles as mf
from ez2digitize.core.splats import Splats

# A unit square in z = 0 as one quad (or two triangles), corners counter-clockwise.
SQUARE = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]


def png(color: tuple[int, int, int], size: tuple[int, int] = (4, 4)) -> bytes:
    out = io.BytesIO()
    PILImage.new("RGB", size, color).save(out, "PNG")
    return out.getvalue()


def textured_square(colors: tuple[tuple[int, int, int], ...] = ((200, 10, 10),)) -> mf.Mesh:
    """The square as two triangles, each textured with one of `colors` (one or two)."""
    uvs = [[(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 1), (0, 1)]]
    materials = [mf.Material(f"m{i}", image=mf.Image(f"t{i}.png", png(c))) for i, c in
                 enumerate(colors)]  # fmt: skip
    return mf.mesh(
        SQUARE,
        [(0, 1, 2), (0, 2, 3)],
        uvs=uvs,
        materials=materials,
        face_materials=[0, len(colors) - 1],
    )


def random_splats(n: int, k: int = 15, seed: int = 1) -> Splats:
    rng = np.random.default_rng(seed)
    return Splats(
        positions=rng.uniform(-3, 3, (n, 3)),
        scales=rng.uniform(-6, -1, (n, 3)),
        rotations=rng.normal(size=(n, 4)),
        opacities=rng.normal(size=n),
        sh_dc=rng.uniform(-1, 1, (n, 3)),
        sh_rest=rng.uniform(-0.6, 0.6, (n, k, 3)),
    )
