# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The mesh from the splats with COLMAP's real Poisson mesher (skipped without COLMAP).

The Backends workflow runs it against the fresh builds (EZ2D_COLMAP).
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_splat_mesh import CAMERAS, sphere_splats

from ez2digitize import splat_mesh
from ez2digitize.backends.common import find_tool


def test_poisson_onsphere_splats(tmp_path: Path) -> None:
    """With the real Poisson mesher: a surface close to the sphere, in its colour."""
    program = find_tool("colmap", env_var="EZ2D_COLMAP")
    if program is None:
        pytest.skip("needs COLMAP (EZ2D_COLMAP or PATH)")
    positions, normals, rgb = splat_mesh.oriented_points(sphere_splats(20000), CAMERAS)
    splat_mesh.write_points(tmp_path / "points.ply", positions, normals, rgb)
    options = splat_mesh.SplatMeshOptions(depth=8)
    subprocess.run(  # noqa: S603 - fixed command
        [str(program), "poisson_mesher", "--input_path", str(tmp_path / "points.ply"),
         "--output_path", str(tmp_path / "mesh.ply"),
         "--PoissonMeshing.depth", str(options.depth),
         "--PoissonMeshing.trim", str(options.trim)],
        check=True, capture_output=True,
    )  # fmt: skip
    mesh = splat_mesh.read_mesh(tmp_path / "mesh.ply")
    radius = np.linalg.norm(mesh.positions, axis=1)
    assert len(mesh.faces) > 10000
    assert np.percentile(radius, 1) > 0.95 and np.percentile(radius, 99) < 1.05
    assert np.allclose(mesh.colors.mean(axis=0), rgb[0], atol=2)
