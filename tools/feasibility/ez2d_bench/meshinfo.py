# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cheap statistics for PLY and OBJ files, without loading the geometry."""

from __future__ import annotations

import re
from pathlib import Path


def ply_counts(path: Path) -> dict[str, int]:
    """Element counts from a PLY header, e.g. {'vertex': 120000, 'face': 240000}."""
    counts: dict[str, int] = {}
    with path.open("rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        for raw in fh:
            line = raw.decode("ascii", errors="replace").strip()
            if line == "end_header":
                return counts
            match = re.match(r"element\s+(\S+)\s+(\d+)", line)
            if match:
                counts[match.group(1)] = int(match.group(2))
    raise ValueError(f"{path}: PLY header has no end_header")


def obj_counts(path: Path) -> dict[str, int]:
    vertices = faces = 0
    with path.open("rb") as fh:
        for line in fh:
            if line.startswith(b"v "):
                vertices += 1
            elif line.startswith(b"f "):
                faces += 1
    return {"vertex": vertices, "face": faces}


def mesh_counts(path: Path) -> dict[str, int]:
    return obj_counts(path) if path.suffix.lower() == ".obj" else ply_counts(path)
