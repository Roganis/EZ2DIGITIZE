# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from ez2digitize.backends.common import bundled_bin_dir, find_tool

FakeTool = Callable[[Path, str], Path]


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_tool: FakeTool) -> dict[str, Path]:
    found = {
        name: fake_tool(tmp_path / name / "tool", name)
        for name in ("explicit", "env", "bundle/backends/bin", "extra", "path")
    }
    monkeypatch.setenv("PATH", str(found["path"].parent))
    monkeypatch.delenv("EZ2D_TEST_TOOL", raising=False)
    return found


def test_search_order(dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    extra = [dirs["extra"].parent]
    assert find_tool("tool") == dirs["path"]
    assert find_tool("tool", extra_dirs=extra) == dirs["extra"]
    monkeypatch.setattr(sys, "_MEIPASS", str(dirs["bundle/backends/bin"].parents[2]), raising=False)
    assert bundled_bin_dir() == dirs["bundle/backends/bin"].parent
    assert find_tool("tool", extra_dirs=extra) == dirs["bundle/backends/bin"]
    monkeypatch.setenv("EZ2D_TEST_TOOL", str(dirs["env"]))
    assert find_tool("tool", env_var="EZ2D_TEST_TOOL") == dirs["env"]
    assert (
        find_tool("tool", explicit=dirs["explicit"], env_var="EZ2D_TEST_TOOL") == dirs["explicit"]
    )


def test_configured_paths_do_not_fall_back(
    dirs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A wrong setting must surface, not be papered over by another copy.
    assert find_tool("tool", explicit=tmp_path / "nope") is None
    monkeypatch.setenv("EZ2D_TEST_TOOL", str(tmp_path / "nope"))
    assert find_tool("tool", env_var="EZ2D_TEST_TOOL") is None


def test_not_executable_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plain = tmp_path / "tool"
    plain.write_text("x")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert find_tool("tool", explicit=plain) is None
    assert find_tool("tool", extra_dirs=[tmp_path]) is None


def test_no_bundle_outside_pyinstaller() -> None:
    assert not hasattr(sys, "_MEIPASS")
    assert bundled_bin_dir() is None


def test_relative_paths_become_absolute(
    dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(dirs["env"].parent.parent)
    monkeypatch.setenv("EZ2D_TEST_TOOL", "env/tool")
    found = find_tool("tool", env_var="EZ2D_TEST_TOOL")
    assert found == dirs["env"] and found.is_absolute()
