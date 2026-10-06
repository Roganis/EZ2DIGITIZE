# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from collections.abc import Callable
from pathlib import Path

import pytest
from scripts import printing_script

from ez2digitize.core.project import Project

FakeTool = Callable[[Path, str], Path]


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "project")


def _fake_tool(path: Path, output: str) -> Path:
    return printing_script(path, output)


@pytest.fixture
def fake_tool() -> FakeTool:
    """Makes an executable that prints `output` whatever its arguments."""
    return _fake_tool
