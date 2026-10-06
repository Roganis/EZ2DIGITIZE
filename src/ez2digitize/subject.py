# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""What a project scans: a small object (the default), or a room or scene.

An object is photographed all round, and can be masked out of its
background; a room, a building or a landscape is photographed from inside
it or walking through it. The subject adjusts the presets (see presets)
and turns off the advice that assumes photos all round an object (the
camera rings). A project keeps it in its settings.
"""

from __future__ import annotations

from typing import Literal, get_args

from ez2digitize.core.project import Project

Subject = Literal["object", "scene"]
SUBJECTS: tuple[Subject, ...] = get_args(Subject)
DEFAULT: Subject = "object"
SETTING = "subject"
LABELS: dict[Subject, str] = {"object": "Small object", "scene": "Room or outdoor scene"}
HINTS: dict[Subject, str] = {
    "object": "Photos all round one object, which can be masked out of its background.",
    "scene": "Photos of a room, a building or a landscape, from inside it or walking "
    "through it. No masks; plain walls and floors are kept when meshing.",
}


def parse(value: object) -> Subject:
    """A stored or typed subject; unknown ones fall back to the default."""
    return next((s for s in SUBJECTS if value == s), DEFAULT)


def of(project: Project) -> Subject:
    return parse(project.settings.get(SETTING))


def store(project: Project, subject: Subject) -> None:
    """Store the subject in the project (saved only if it changed)."""
    if project.settings.get(SETTING) != subject:
        project.settings[SETTING] = subject
        project.save()
