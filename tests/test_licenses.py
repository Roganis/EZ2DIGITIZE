# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import json
import sys
from pathlib import Path

import pytest

from ez2digitize import licenses


def test_texts_from_the_source_tree() -> None:
    assert "GNU GENERAL PUBLIC LICENSE" in (licenses.license_text(licenses.LICENSE) or "")
    assert "COLMAP 4.2.1" in (licenses.license_text(licenses.THIRD_PARTY) or "")


def test_bundled_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The AppImage layout: _MEIPASS/backends/bin, texts next to the code.
    (tmp_path / "backends" / "bin").mkdir(parents=True)
    (tmp_path / "backends" / "BUILDINFO.json").write_text(
        json.dumps(
            {
                "colmap": "4.2.1",
                "openmvs": "v2.4.0",
                "vcpkg": "2026.07.29",
                "openmvs_patches": "openmvs-2.4.0-sample-type.patch",
            }
        )  # fmt: skip
    )
    (tmp_path / "THIRD_PARTY_LICENSES").write_text("bundled copy")
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert licenses.license_text(licenses.THIRD_PARTY) == "bundled copy"
    summary = licenses.summary()
    assert "COLMAP 4.2.1, OpenMVS v2.4.0 (patches: openmvs-2.4.0-sample-type.patch)" in summary
    assert str(tmp_path / "backends" / "licenses") in summary


def test_macos_app_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contents = tmp_path / "EZ2DIGITIZE.app" / "Contents"
    (contents / "Frameworks").mkdir(parents=True)
    (contents / "Resources" / "backends" / "bin").mkdir(parents=True)
    monkeypatch.setattr(sys, "_MEIPASS", str(contents / "Frameworks"), raising=False)
    assert licenses.backends_dir() == contents / "Resources" / "backends"
