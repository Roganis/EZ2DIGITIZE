# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import numpy as np
import pytest
from model_files import SQUARE, random_splats, textured_square

from ez2digitize import models, views
from ez2digitize.core import gltf
from ez2digitize.core import meshfiles as mf
from ez2digitize.core import splats as sp


def test_identify_by_suffix_and_ply_header(tmp_path: Path) -> None:
    mesh_ply = mf.write_ply(mf.mesh(SQUARE, [(0, 1, 2)]), tmp_path / "m.ply")[0]
    splat_ply = sp.write_ply(random_splats(3, k=0), tmp_path / "s.ply")
    points = tmp_path / "p.ply"
    points.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n"
                      "property float y\nproperty float z\nend_header\n0 0 0\n")  # fmt: skip
    assert models.identify(mesh_ply).kind == "mesh"
    assert models.identify(splat_ply).kind == "splat"
    assert models.identify(points).kind == "points"
    for name, kind in (("a.STL", "mesh"), ("b.spz", "splat"), ("c.ksplat", "splat")):
        (tmp_path / name).write_bytes(b"x")
        assert models.identify(tmp_path / name).kind == kind
    (tmp_path / "d.fbx").write_bytes(b"x")
    with pytest.raises(models.ModelError, match="not a model file"):
        models.identify(tmp_path / "d.fbx")
    with pytest.raises(models.ModelError, match="no such file"):
        models.identify(tmp_path / "missing.obj")


def test_up_axis_and_outputs_by_format(tmp_path: Path) -> None:
    def model(name: str, kind: models.Kind) -> models.ModelFile:
        return models.ModelFile(tmp_path / name, kind)

    assert model("a.glb", "mesh").up == "y" and model("a.stl", "mesh").up == "z"
    assert model("a.ply", "splat").up == "-y" and model("a.spz", "splat").up == "y"
    assert model("a.spz", "splat").outputs == models.SPLAT_WRITE
    assert model("a.sog", "splat").outputs == ()
    assert model("a.ply", "points").outputs == ()
    assert ".usdz" in model("a.obj", "mesh").outputs


def test_convert_meshes_between_y_up_and_z_up(tmp_path: Path) -> None:
    source = gltf.write_glb(textured_square(), tmp_path / "s.glb")[0]
    # The square stands in the XY plane: Y is its height, so in a Z-up STL, Z is.
    stl = models.convert(source, tmp_path / "s.stl")[0]
    low, high = mf.finite_bounds(models.read_mesh(stl))
    assert high.tolist() == [1.0, 0.0, 1.0] and low.tolist() == [0.0, 0.0, 0.0]
    back = models.convert(stl, tmp_path / "back.glb", scale=2.0)[0]
    low, high = mf.finite_bounds(models.read_mesh(back))
    assert high.tolist() == [2.0, 2.0, 0.0]
    usdz = models.convert(source, tmp_path / "s.usdz")
    assert usdz == [tmp_path / "s.usdz"]
    textured = models.read_mesh(models.convert(source, tmp_path / "s.obj")[0])
    assert textured.textured


def test_convert_splats_turns_between_frames(tmp_path: Path) -> None:
    splats = random_splats(10, k=3)
    ply = sp.write_ply(splats, tmp_path / "s.ply")
    spz = models.convert(ply, tmp_path / "s.spz")[0]
    flipped = sp.read_spz(spz)
    assert np.abs(flipped.positions - splats.positions * [1, -1, -1]).max() < 1e-3
    again = sp.read_ply(models.convert(spz, tmp_path / "again.ply", scale=10)[0])
    assert np.abs(again.positions - splats.positions * 10).max() < 1e-2
    assert np.allclose(again.scales, flipped.scales + np.log(10))
    # Between two files in the same frame nothing turns.
    same = sp.read_ply(models.convert(ply, tmp_path / "copy.ply")[0])
    assert np.allclose(same.positions, splats.positions)


def test_convert_errors(tmp_path: Path) -> None:
    glb = gltf.write_glb(mf.mesh(SQUARE, [(0, 1, 2)]), tmp_path / "m.glb")[0]
    spz = sp.write_spz(random_splats(3, k=0), tmp_path / "s.spz")
    (tmp_path / "x.sog").write_bytes(b"PK")
    cases = [
        (glb, tmp_path / "m.spz", "can't become splats"),
        (spz, tmp_path / "s.glb", "can't become a mesh"),
        (tmp_path / "x.sog", tmp_path / "x.spz", "can be converted to nothing"),
        (glb, tmp_path / "m.fbx", "can be converted to"),
        (glb, glb, "replace the original"),
    ]
    for source, target, message in cases:
        with pytest.raises(models.ModelError, match=message):
            models.convert(source, target)
    with pytest.raises(models.ModelError, match="more than 0"):
        models.convert(glb, tmp_path / "n.obj", scale=0)
    (tmp_path / "broken.obj").write_text("v 0 0 0\n")
    with pytest.raises(models.ModelError, match="no faces"):
        models.convert(tmp_path / "broken.obj", tmp_path / "b.glb")


def test_summary(tmp_path: Path) -> None:
    glb = gltf.write_glb(textured_square(), tmp_path / "m.glb")[0]
    assert (
        models.summary(models.identify(glb)) == "2 triangles, 4 vertices, textured; size 1 x 1 x 0"
    )
    spz = sp.write_spz(random_splats(3, k=3), tmp_path / "s.spz")
    assert models.summary(models.identify(spz)) == "3 splats, colour degree 1"


def test_file_views(tmp_path: Path) -> None:
    obj = mf.write_obj(textured_square(), tmp_path / "m.obj")[0]
    view = views.file_view(models.identify(obj))
    assert (view.key, view.kind, view.upright, view.label) == ("file", "glb", None, "m.obj")
    cache = tmp_path / "cache"
    made = views.files(view, cache)["model"]
    assert isinstance(made, Path) and made.parent == cache and made.suffix == ".glb"
    assert views.files(view, cache)["model"] == made  # cached
    assert gltf.read_gltf(made).textured

    glb = gltf.write_glb(textured_square(), tmp_path / "m.glb")[0]
    assert views.files(views.file_view(models.identify(glb)), cache)["model"] == glb
    stl = mf.write_stl(mf.mesh(SQUARE, [(0, 1, 2)]), tmp_path / "m.stl")[0]
    assert views.file_view(models.identify(stl)).upright == models.UP_MATRICES["z"]
    splat = sp.write_ply(random_splats(3, k=0), tmp_path / "s.ply")
    view = views.file_view(models.identify(splat), "y")
    assert view.kind == "splat" and view.upright is None
    assert views.files(view, cache) == {"model": splat}
    spec = views.spec(views.file_view(models.identify(splat)), {"model": "ez2d://data/x"})
    assert spec["kind"] == "splat" and spec["upright"] == [[1, 0, 0], [0, -1, 0], [0, 0, -1]]

    (tmp_path / "bad.obj").write_text("nothing\n")
    with pytest.raises(views.ViewError, match="bad.obj can't be shown"):
        views.files(views.file_view(models.identify(tmp_path / "bad.obj")), cache)


def test_up_matrices_turn_the_named_axis_to_y() -> None:
    axes: dict[models.Up, list[int]] = {"y": [0, 1, 0], "-y": [0, -1, 0], "z": [0, 0, 1],
                                         "-z": [0, 0, -1]}  # fmt: skip
    for up, axis in axes.items():
        matrix = np.array(models.UP_MATRICES[up], np.float64)
        assert (matrix @ axis).tolist() == [0, 1, 0]
        assert np.isclose(np.linalg.det(matrix), 1)
