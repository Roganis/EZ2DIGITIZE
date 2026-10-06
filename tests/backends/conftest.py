# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from collections.abc import Callable
from pathlib import Path

import pytest

from ez2digitize.core.project import Project

FakeTool = Callable[[Path, str], Path]


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "project")


def _fake_tool(path: Path, output: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\ncat <<'EOF'\n{output}\nEOF\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def fake_tool() -> FakeTool:
    """Makes an executable that prints `output` whatever its arguments."""
    return _fake_tool
