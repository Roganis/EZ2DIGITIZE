# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The notices of the crates inside Brush (tools/packaging/brush_notices.py)."""

from pathlib import Path

import brush_notices
import pytest
from brush_notices import Crate

from ez2digitize.backends.brush import PINNED_VERSION


def test_crates_from_cargo_tree() -> None:
    lines = [
        "brush-app v0.3.0 (/src/crates/brush-app)",
        "serde v1.0.219",
        "serde v1.0.219 (*)",
        "egui v0.32.3 (https://github.com/emilk/egui?rev=abc#abc)",
        "",
    ]
    assert brush_notices.tree_crates(lines) == {
        ("brush-app", "0.3.0"), ("serde", "1.0.219"), ("egui", "0.32.3"),
    }  # fmt: skip


def test_license_ids() -> None:
    assert brush_notices.license_ids("MIT OR Apache-2.0") == ["MIT", "Apache-2.0"]
    assert brush_notices.license_ids("MIT/Apache-2.0") == ["MIT", "Apache-2.0"]
    assert brush_notices.license_ids("(MIT OR Apache-2.0) AND Unicode-3.0") == [
        "MIT", "Apache-2.0", "Unicode-3.0",
    ]  # fmt: skip
    assert brush_notices.license_ids("Apache-2.0 WITH LLVM-exception OR MIT") == [
        "Apache-2.0", "LLVM-exception", "MIT",
    ]  # fmt: skip


def test_notice_files(tmp_path: Path) -> None:
    for name in ("LICENSE-MIT", "NOTICE.txt", "COPYING", "README.md", "src.rs"):
        (tmp_path / name).write_text(name)
    (tmp_path / "LICENSES").mkdir()
    (tmp_path / "LICENSES" / "Apache-2.0.txt").write_text("apache")
    names = [p.name for p in brush_notices.notice_files(tmp_path)]
    assert names == ["COPYING", "LICENSE-MIT", "Apache-2.0.txt", "NOTICE.txt"]


def test_render() -> None:
    crates = [
        Crate("a", "1.0", "MIT", ["Ann"], "https://a", [0]),
        Crate("b", "2.0", "MIT OR Apache-2.0", ["Bo <b@x>"], ""),
    ]
    text = brush_notices.render("0.3.0", crates, ["Copyright Ann\nMIT text", "MIT std"], {"MIT": 1})
    assert text.startswith("Third-party notices for Brush v0.3.0\n")
    assert "a 1.0  MIT  https://a\n    Texts: [1]\n" in text
    # Without its own files: the authors and the standard text.
    assert "b 2.0  MIT OR Apache-2.0\n    Copyright: Bo <b@x>\n    Texts: [2]\n" in text
    assert "[2] MIT (standard text)\n\nMIT std\n" in text


def test_version_check(tmp_path: Path) -> None:
    notices = tmp_path / "notices.txt"
    notices.write_text(f"Third-party notices for Brush v{PINNED_VERSION}\n===\n")
    brush_notices.check(notices, f"brush-cli {PINNED_VERSION}")
    with pytest.raises(SystemExit, match="run tools/packaging/brush_notices.py"):
        brush_notices.check(notices, "brush-cli 9.9.9")  # another Brush in the package
    notices.write_text("Third-party notices for Brush v0.0.1\n")
    with pytest.raises(SystemExit):
        brush_notices.check(notices, f"brush-cli {PINNED_VERSION}")  # stale notices


def test_the_committed_notices_are_for_the_pinned_brush(tmp_path: Path) -> None:
    from build_appimage import add_brush_notices

    text = brush_notices.OUT.read_text(encoding="utf-8")
    assert text.startswith(f"Third-party notices for Brush v{PINNED_VERSION}\n")
    for crate in ("wgpu", "burn", "egui", "ring"):
        assert f"\n{crate} " in text
    add_brush_notices(tmp_path, f"brush-cli {PINNED_VERSION}")
    assert (tmp_path / "THIRD-PARTY-NOTICES.txt").read_text(encoding="utf-8") == text
