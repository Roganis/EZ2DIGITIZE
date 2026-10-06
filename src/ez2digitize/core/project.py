# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Project folders and their versioned `project.json`.

Layout of a project folder:

    my-scan/
      project.json    schema_version, name, preset, settings
      captures/       one capture bundle per import (see capture.py)
      masks/          masks, by capture id and image stem
      stages/         one folder per pipeline stage, each with stage.json
      exports/        user-facing outputs

Any change to the `project.json` format bumps SCHEMA_VERSION and adds a
function to MIGRATIONS that upgrades data from the previous version.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ez2digitize
from ez2digitize.core.files import FormatError, read_json_object, utc_now, write_json_atomic

PROJECT_FILE = "project.json"
SCHEMA_VERSION = 1

# MIGRATIONS[n] upgrades project.json data from version n to n + 1.
Migration = Callable[[dict[str, Any]], dict[str, Any]]
MIGRATIONS: dict[int, Migration] = {}

SUBDIRS = ("captures", "masks", "stages", "exports")
_STAGE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ProjectError(Exception):
    """A project can't be created, opened or saved."""


def migrate(data: dict[str, Any], target: int | None = None) -> dict[str, Any]:
    """Upgrade project.json data to `target` (default: current), one version at a time."""
    target = SCHEMA_VERSION if target is None else target
    version = data.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ProjectError(f"invalid schema_version: {version!r}")
    if version > target:
        raise ProjectError(
            f"this project was saved by a newer version of EZ2DIGITIZE "
            f"(format {version}, this version reads up to {target})"
        )
    while version < target:
        step = MIGRATIONS.get(version)
        if step is None:
            raise ProjectError(f"no migration from format {version} to {version + 1}")
        data = step(dict(data))
        version += 1
        data["schema_version"] = version
    return data


@dataclass
class Project:
    root: Path
    name: str
    created: str
    preset: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    # Version of the app that last saved the project; for diagnostics only.
    app_version: str = ez2digitize.__version__

    @classmethod
    def create(cls, root: Path, name: str | None = None) -> Project:
        """Create a new project in `root`, which must not exist or be empty."""
        root = root.absolute()
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise ProjectError(f"{root} already exists and is not an empty folder")
        root.mkdir(parents=True, exist_ok=True)
        for sub in SUBDIRS:
            (root / sub).mkdir()
        project = cls(root=root, name=name or root.name, created=utc_now())
        project.save()
        return project

    @classmethod
    def open(cls, root: Path) -> Project:
        """Open an existing project, upgrading its project.json if it is older.

        The file of an older format is kept as `project.json.v<N>.bak`.
        """
        root = root.absolute()
        path = root / PROJECT_FILE
        try:
            data = read_json_object(path)
        except FormatError as exc:
            raise ProjectError(str(exc)) from exc
        original_version = data.get("schema_version")
        data = migrate(data)
        project = cls._from_dict(root, data)
        if original_version != SCHEMA_VERSION:
            shutil.copy2(path, root / f"{PROJECT_FILE}.v{original_version}.bak")
            project.save()
        for sub in SUBDIRS:
            (root / sub).mkdir(exist_ok=True)
        return project

    @classmethod
    def _from_dict(cls, root: Path, data: dict[str, Any]) -> Project:
        name, created = data.get("name"), data.get("created")
        preset, settings = data.get("preset"), data.get("settings", {})
        if not isinstance(name, str) or not isinstance(created, str):
            raise ProjectError(f"{root / PROJECT_FILE}: 'name' and 'created' must be strings")
        if preset is not None and not isinstance(preset, str):
            raise ProjectError(f"{root / PROJECT_FILE}: 'preset' must be a string or null")
        if not isinstance(settings, dict):
            raise ProjectError(f"{root / PROJECT_FILE}: 'settings' must be an object")
        app_version = data.get("app_version")
        return cls(
            root=root,
            name=name,
            created=created,
            preset=preset,
            settings=settings,
            app_version=app_version if isinstance(app_version, str) else "unknown",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "created": self.created,
            "app_version": self.app_version,
            "preset": self.preset,
            "settings": self.settings,
        }

    def save(self) -> None:
        self.app_version = ez2digitize.__version__
        write_json_atomic(self.root / PROJECT_FILE, self.to_dict())

    @property
    def captures_dir(self) -> Path:
        return self.root / "captures"

    @property
    def masks_dir(self) -> Path:
        return self.root / "masks"

    @property
    def stages_dir(self) -> Path:
        return self.root / "stages"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    def stage_dir(self, stage: str) -> Path:
        """Folder of one pipeline stage. Names are short slugs like `01-features`."""
        if not _STAGE_NAME.fullmatch(stage):
            raise ProjectError(f"invalid stage name: {stage!r}")
        return self.stages_dir / stage
