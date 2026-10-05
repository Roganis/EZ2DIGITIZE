# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Quality presets: fast, balanced and high, as concrete pipeline settings.

Each preset is a full MeshSettings; the GUI's advanced panel and the CLI's
options override single values on top (`mesh_settings(..., level=0)`).
`describe` lists the values that differ between presets in words, for the
advanced panel and the CLI.

Reference timings, 63 photos of a skull on the RX 7900 GRE desktop (CPU
only): balanced about 15 minutes, of which densify 8 and refine (high only)
4. Fast reads a quarter of the pixels in every step.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, get_args

from ez2digitize.backends import brush, colmap, openmvs
from ez2digitize.pipeline import MeshSettings

Quality = Literal["fast", "balanced", "high"]
QUALITIES: tuple[Quality, ...] = get_args(Quality)
DEFAULT_QUALITY: Quality = "balanced"

LABELS: dict[Quality, str] = {"fast": "Fast", "balanced": "Balanced", "high": "High"}
HINTS: dict[Quality, str] = {
    "fast": "A quick preview: half-size photos everywhere, coarse surface.",
    "balanced": "Good detail in reasonable time; the default.",
    "high": "Full-size photos for the surface, then refined against the photos. "
    "Several times slower and needs much more memory.",
}

_PRESETS: dict[Quality, MeshSettings] = {
    "fast": MeshSettings(
        features=colmap.FeatureOptions(max_image_size=1600, max_num_features=4096),
        densify=openmvs.DensifyOptions(resolution_level=2),
        texture=openmvs.TextureOptions(resolution_level=1),
        splat=brush.SplatOptions(total_steps=7_000, max_resolution=1280),
    ),
    "balanced": MeshSettings(
        features=colmap.FeatureOptions(max_image_size=3200),
        densify=openmvs.DensifyOptions(resolution_level=1),
        texture=openmvs.TextureOptions(resolution_level=0),
    ),
    "high": MeshSettings(
        features=colmap.FeatureOptions(max_image_size=3200),
        densify=openmvs.DensifyOptions(resolution_level=0),
        refine=openmvs.RefineOptions(resolution_level=1),
        texture=openmvs.TextureOptions(resolution_level=0),
    ),
}


def mesh_settings(
    quality: Quality = DEFAULT_QUALITY,
    *,
    level: int | None = None,
    refine: bool | None = None,
    max_image_size: int | None = None,
    faces: int | None = None,
    steps: int | None = None,
) -> MeshSettings:
    """The preset's settings with single values overridden.

    `level` is the dense detail (OpenMVS resolution level, 0 = full size);
    refining uses the same level as the dense step, or 1 if that is 0 (full
    size refinement needs far more memory for little gain). `faces`
    simplifies the mesh to about that many faces before texturing.
    """
    settings = _PRESETS[quality]
    if faces is not None:
        settings = replace(settings, texture=replace(settings.texture, target_faces=faces))
    if steps is not None:
        settings = replace(settings, splat=replace(settings.splat, total_steps=steps))
    if max_image_size is not None:
        settings = replace(
            settings, features=replace(settings.features, max_image_size=max_image_size)
        )
    if level is not None:
        settings = replace(settings, densify=replace(settings.densify, resolution_level=level))
    if refine is None:
        refine = settings.refine is not None
    if refine:
        refine_level = max(1, settings.densify.resolution_level)
        settings = replace(settings, refine=openmvs.RefineOptions(resolution_level=refine_level))
    else:
        settings = replace(settings, refine=None)
    return settings


def describe(settings: MeshSettings) -> list[tuple[str, str]]:
    """The values that set speed and detail, as (label, value) rows."""

    def size(level: int) -> str:
        return "full size" if level == 0 else f"1/{2**level} size"

    refine = settings.refine
    return [
        ("Photo size for features", f"up to {settings.features.max_image_size} px"),
        ("Features per photo", f"up to {settings.features.max_num_features}"),
        (
            "Dense point cloud",
            f"{size(settings.densify.resolution_level)} photos, "
            f"up to {settings.densify.max_resolution} px",
        ),
        ("Refine mesh", "no" if refine is None else f"yes, {size(refine.resolution_level)}"),
        ("Texture", f"{size(settings.texture.resolution_level)} photos"),
        (
            "Mesh size",
            "full detail"
            if settings.texture.target_faces is None
            else f"about {settings.texture.target_faces:,} faces",
        ),
        ("Camera placement", f"{settings.mapper.kind} mapper"),
        (
            "Splats",
            f"{settings.splat.total_steps:,} steps, photos up to "
            f"{settings.splat.max_resolution} px",
        ),
    ]


def parse_quality(value: str | None) -> Quality:
    """A stored or typed preset name; unknown ones fall back to the default."""
    for quality in QUALITIES:
        if value == quality:
            return quality
    return DEFAULT_QUALITY
