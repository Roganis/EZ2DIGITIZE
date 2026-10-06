# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The license texts the app shows (Help -> Licenses, `ez2d licenses`).

LICENSE and THIRD_PARTY_LICENSES are looked up in the packaged app (the
build scripts put them next to the code), the source tree, or the installed
package's metadata. The bundled backends bring their own license files and
build information (backends/licenses, BUILDINFO.json).
"""

from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from ez2digitize import plugins
from ez2digitize.backends.common import bundled_bin_dir

LICENSE = "LICENSE"
THIRD_PARTY = "THIRD_PARTY_LICENSES"


def license_text(name: str) -> str | None:
    """LICENSE or THIRD_PARTY_LICENSES, wherever this copy of the app has it."""
    candidates = []
    if (base := getattr(sys, "_MEIPASS", None)) is not None:
        candidates += [Path(base) / name, Path(base).parent / "Resources" / name]
    candidates.append(Path(__file__).resolve().parents[2] / name)  # source tree
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    try:
        distribution = metadata.distribution("ez2digitize")
    except metadata.PackageNotFoundError:
        return None
    for file in distribution.files or []:
        if file.name == name:
            return file.read_text(encoding="utf-8")
    return None


def backends_dir() -> Path | None:
    """The bundled backends' folder (with licenses/ and BUILDINFO.json), if any."""
    bin_dir = bundled_bin_dir()
    return bin_dir.parent if bin_dir is not None else None


def backend_build_info() -> dict[str, Any] | None:
    folder = backends_dir()
    if folder is None:
        return None
    try:
        info = json.loads((folder / "BUILDINFO.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) else None


def summary() -> str:
    """What the user is running and where its licenses are, as plain text."""
    lines = []
    info = backend_build_info()
    folder = backends_dir()
    if info is not None and folder is not None:
        lines.append(
            f"Bundled backends: COLMAP {info.get('colmap')}, OpenMVS {info.get('openmvs')}"
            + (f" (patches: {info['openmvs_patches']})" if info.get("openmvs_patches") else "")
            + f", built with vcpkg {info.get('vcpkg')}."
        )
        lines.append(f"Their license files: {folder / 'licenses'}")
    else:
        lines.append("No bundled backends: the tools installed on this computer are used.")
    found = plugins.installed().plugins
    if found:
        lines.append("Plugins you installed (not part of EZ2DIGITIZE, under their own licenses):")
        lines += [f"  {p.label()}: {p.license_summary()}, in {p.folder}" for p in found]
    return "\n".join(lines)
