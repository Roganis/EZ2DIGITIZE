# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import struct
import zipfile
from pathlib import Path

from model_files import SQUARE, textured_square

from ez2digitize.core import meshfiles as mf
from ez2digitize.core.usdz import ALIGN, LAYER, write_usdz


def data_offsets(path: Path) -> dict[str, int]:
    """Where each file's data starts in the zip."""
    raw = path.read_bytes()
    with zipfile.ZipFile(path) as package:
        offsets = {}
        for info in package.infolist():
            name_length, extra_length = struct.unpack_from("<HH", raw, info.header_offset + 26)
            offsets[info.filename] = info.header_offset + 30 + name_length + extra_length
        return offsets


def test_package_layout(tmp_path: Path) -> None:
    path = write_usdz(textured_square(((200, 10, 10), (10, 200, 10))), tmp_path / "s.usdz")[0]
    with zipfile.ZipFile(path) as package:
        infos = package.infolist()
        assert [i.filename for i in infos] == [LAYER, "textures/texture0.png",
                                               "textures/texture1.png"]  # fmt: skip
        assert all(i.compress_type == zipfile.ZIP_STORED for i in infos)
        assert package.testzip() is None
        layer = package.read(LAYER).decode()
    assert all(offset % ALIGN == 0 for offset in data_offsets(path).values())
    assert layer.startswith("#usda 1.0") and 'upAxis = "Y"' in layer
    assert "faceVertexIndices = [0, 1, 2, 0, 2, 3]" in layer
    assert 'interpolation = "faceVarying"' in layer
    # Two materials: a subset of faces bound to each.
    assert layer.count('def GeomSubset "Part') == 2
    assert "asset inputs:file = @textures/texture1.png@" in layer
    assert "rel material:binding = </Root/Materials/Material1>" in layer


def test_colours_and_plain_material(tmp_path: Path) -> None:
    mesh = mf.mesh(SQUARE, [(0, 1, 2)], colors=[(255, 0, 0)] * 4)
    mesh.materials = [mf.Material("paint", (0.5, 0.5, 0.5, 1.0))]
    path = write_usdz(mesh, tmp_path / "c.usdz")[0]
    with zipfile.ZipFile(path) as package:
        assert package.namelist() == [LAYER]
        layer = package.read(LAYER).decode()
    assert "primvars:displayColor = [(1, 0, 0)" in layer
    assert "color3f inputs:diffuseColor = (0.5, 0.5, 0.5)" in layer
    assert "GeomSubset" not in layer and "</Root/Materials/Material0>" in layer
