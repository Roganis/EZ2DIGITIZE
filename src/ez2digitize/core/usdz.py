# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""USDZ: a mesh for Apple's AR Quick Look (iPhone, iPad, Mac) and other USD tools.

A USDZ file is an uncompressed zip whose first file is the USD layer and
whose every file starts at a multiple of 64 bytes, so it can be read in
place. The layer here is text (USDA): one Mesh, Y up, one metre per unit,
with its texture coordinates (`primvars:st`, per face corner) and a
UsdPreviewSurface material per texture or colour, bound to the faces that
use it (GeomSubsets when there are several). Vertex colours go into
`primvars:displayColor`; USD tools show them, Quick Look shows only
textures and material colours.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np

from ez2digitize.core.meshfiles import Material, Mesh, web_image

ALIGN = 64
LAYER = "model.usda"
# Zip's local file header: 30 bytes plus the name and the extra field.
LOCAL_HEADER = 30
PADDING_ID = 0x1986  # an extra field nobody else uses, to pad with


def write_usdz(source: Mesh, path: Path) -> list[Path]:
    """`path` (.usdz): the mesh, its materials and texture images."""
    materials = source.materials or [Material("default")]
    face_material = (
        source.face_materials.astype(np.int64)
        if source.face_materials is not None and source.materials
        else np.zeros(source.face_count, np.int64)
    )
    files: list[tuple[str, bytes]] = []
    shaders = []
    for index, material in enumerate(materials):
        texture = None
        if material.image is not None and source.uvs is not None:
            image = web_image(material.image)
            texture = f"textures/texture{index}{image.suffix}"
            files.append((texture, image.data))
        shaders.append(_material(index, material, texture))
    layer = _layer(source, face_material, shaders, len(materials))
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as package:
        for name, data in [(LAYER, layer.encode("utf-8")), *files]:
            _add_aligned(package, name, data)
    return [path]


def _add_aligned(package: zipfile.ZipFile, name: str, data: bytes) -> None:
    """Store `data` so it starts at a multiple of 64 bytes in the zip."""
    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    assert package.fp is not None
    start = package.fp.tell()
    zip64 = 20 if len(data) * 1.05 > zipfile.ZIP64_LIMIT else 0
    used = start + LOCAL_HEADER + len(name.encode("utf-8")) + zip64 + 4
    pad = -used % ALIGN
    info.extra = struct.pack("<HH", PADDING_ID, pad) + b"\0" * pad
    package.writestr(info, data)


def _layer(source: Mesh, face_material: np.ndarray, shaders: list[str], count: int) -> str:
    def points(rows: np.ndarray, digits: str = ".7g") -> str:
        return ", ".join(
            "(" + ", ".join(format(v, digits) for v in row) + ")" for row in rows.tolist()
        )

    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Root"',
        "    metersPerUnit = 1",
        '    upAxis = "Y"',
        '    customLayerData = {string creator = "EZ2DIGITIZE"}',
        ")",
        "",
        'def Xform "Root" (',
        '    kind = "component"',
        ")",
        "{",
        '    def Mesh "Model" (',
        '        prepend apiSchemas = ["MaterialBindingAPI"]',
        "    )",
        "    {",
        f"        int[] faceVertexCounts = [{', '.join(['3'] * source.face_count)}]",
        f"        int[] faceVertexIndices = [{_ints(source.faces.reshape(-1))}]",
        f"        point3f[] points = [{points(source.positions)}]",
        '        uniform token subdivisionScheme = "none"',
        '        uniform token orientation = "rightHanded"',
    ]
    low, high = source.positions.min(axis=0), source.positions.max(axis=0)
    lines.append(f"        float3[] extent = [{points(np.stack([low, high]))}]")
    if source.uvs is not None:
        lines += [
            f"        texCoord2f[] primvars:st = [{points(source.uvs.reshape(-1, 2), '.6g')}] (",
            '            interpolation = "faceVarying"',
            "        )",
        ]
    if source.colors is not None:
        colors = source.colors.astype(np.float64) / 255
        lines += [
            f"        color3f[] primvars:displayColor = [{points(colors, '.4g')}] (",
            '            interpolation = "vertex"',
            "        )",
        ]
    if count == 1:
        lines.append("        rel material:binding = </Root/Materials/Material0>")
    else:
        lines.append('        uniform token subsetFamily:materialBind:familyType = "partition"')
        for index in range(count):
            faces = np.flatnonzero(face_material == index)
            if not len(faces):
                continue
            lines += [
                f'        def GeomSubset "Part{index}" (',
                '            prepend apiSchemas = ["MaterialBindingAPI"]',
                "        )",
                "        {",
                '            uniform token elementType = "face"',
                '            uniform token familyName = "materialBind"',
                f"            int[] indices = [{_ints(faces)}]",
                f"            rel material:binding = </Root/Materials/Material{index}>",
                "        }",
            ]
    lines += ["    }", "", '    def Scope "Materials"', "    {", *shaders, "    }", "}", ""]
    return "\n".join(lines)


def _ints(values: np.ndarray) -> str:
    return ", ".join(map(str, values.tolist()))


def _material(index: int, material: Material, texture: str | None) -> str:
    base = f"/Root/Materials/Material{index}"
    r, g, b, a = material.color
    lines = [
        f'        def Material "Material{index}"',
        "        {",
        f"            token outputs:surface.connect = <{base}/Surface.outputs:surface>",
        '            def Shader "Surface"',
        "            {",
        '                uniform token info:id = "UsdPreviewSurface"',
        "                float inputs:metallic = 0",
        "                float inputs:roughness = 1",
        f"                float inputs:opacity = {a:.4g}",
    ]
    if texture is None:
        lines.append(f"                color3f inputs:diffuseColor = ({r:.4g}, {g:.4g}, {b:.4g})")
    else:
        lines.append(
            f"                color3f inputs:diffuseColor.connect = <{base}/Texture.outputs:rgb>"
        )
    lines += ["                token outputs:surface", "            }"]
    if texture is not None:
        lines += [
            '            def Shader "Coordinates"',
            "            {",
            '                uniform token info:id = "UsdPrimvarReader_float2"',
            '                string inputs:varname = "st"',
            "                float2 outputs:result",
            "            }",
            '            def Shader "Texture"',
            "            {",
            '                uniform token info:id = "UsdUVTexture"',
            f"                asset inputs:file = @{texture}@",
            f"                float2 inputs:st.connect = <{base}/Coordinates.outputs:result>",
            '                token inputs:sourceColorSpace = "sRGB"',
            '                token inputs:wrapS = "repeat"',
            '                token inputs:wrapT = "repeat"',
            f"                float4 inputs:scale = ({r:.4g}, {g:.4g}, {b:.4g}, 1)",
            "                float3 outputs:rgb",
            "            }",
        ]
    lines.append("        }")
    return "\n".join(lines)
