# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The project page driving the real pipeline with fake backends."""

import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox
from pytestqt.qtbot import QtBot

from ez2digitize.backends.common import BackendMissing
from ez2digitize.core.project import Project
from ez2digitize.pipeline import Tools
from ez2digitize.ui.project_page import ProjectPage

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX fake backends")

TIMEOUT_MS = 20_000


@pytest.fixture
def photos(tmp_path: Path) -> Path:
    folder = tmp_path / "photos"
    folder.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (folder / name).write_bytes(name.encode())
    return folder


@pytest.fixture
def page(qtbot: QtBot, tmp_path: Path, fake_tools: Tools) -> ProjectPage:
    page = ProjectPage(Project.create(tmp_path / "project"), lambda: fake_tools)
    qtbot.addWidget(page)
    return page


def _states(page: ProjectPage) -> dict[str, str]:
    return {stage: item.text(1) for stage, item in page._stage_items.items()}


def _wait_idle(qtbot: QtBot, page: ProjectPage) -> None:
    qtbot.waitUntil(lambda: not page.runner.running, timeout=TIMEOUT_MS)


def test_import_enables_run(page: ProjectPage, photos: Path) -> None:
    assert not page.run_button.isEnabled()
    assert "No photos yet" in page.captures_label.text()
    page.import_folder(photos)
    assert page.captures_label.text() == "3 photos in 1 import"
    assert page.run_button.isEnabled()
    assert not page.use_masks.isEnabled()  # the project has no masks


def test_run_to_textured_mesh(qtbot: QtBot, page: ProjectPage, photos: Path) -> None:
    page.import_folder(photos)
    page.detail.setCurrentIndex(2)  # Low -> resolution level 2
    page.start_run()
    assert page.runner.running and not page.run_button.isEnabled()
    assert page.cancel_button.isEnabled()
    _wait_idle(qtbot, page)

    assert page.last_result is not None and page.last_failure is None
    assert set(_states(page).values()) == {"Done"}
    assert list(_states(page)) == [
        "features", "matching", "mapping", "undistort",
        "mvs-import", "densify", "mesh", "texture",
    ]  # fmt: skip
    assert page.status.text() == "Finished"
    assert page.overall.value() == 1000
    assert page.result_label.text().startswith(f"Textured mesh saved in {page.project.exports_dir}")
    assert page.last_result.exports and page._result_folder() is not None
    assert "fake colmap feature_extractor" in page.log.toPlainText()
    assert "Note: 3 of 3 images registered" in page.log.toPlainText()
    assert "--resolution-level 2" in (page.project.stage_dir("densify") / "log.txt").read_text()
    assert page.run_button.isEnabled() and not page.cancel_button.isEnabled()

    # Running again reuses every stage.
    page.start_run()
    _wait_idle(qtbot, page)
    assert set(_states(page).values()) == {"Unchanged"}


def test_failure_shows_log_tail(
    qtbot: QtBot, page: ProjectPage, photos: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_FAIL", "global_mapper")
    page.import_folder(photos)
    page.start_run()
    _wait_idle(qtbot, page)
    assert page.last_failure is not None and page.last_result is None
    assert page.last_failure.log == page.project.stage_dir("mapping") / "log.txt"
    assert _states(page)["mapping"] == "Failed"
    assert page.status.text().startswith("Stopped: stage mapping failed")
    assert "something went wrong" in page.log.toPlainText()
    assert not page.open_log_button.isHidden()


def test_cancel(
    qtbot: QtBot, page: ProjectPage, photos: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SLEEP", "global_mapper")
    page.import_folder(photos)
    page.start_run()
    qtbot.waitUntil(lambda: _states(page).get("mapping") == "Running", timeout=TIMEOUT_MS)
    page.cancel_run()
    _wait_idle(qtbot, page)
    assert page.status.text() == "Cancelled"
    assert _states(page)["mapping"] == "Cancelled"
    assert page.last_result is None


def test_missing_backends(
    qtbot: QtBot, tmp_path: Path, photos: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing() -> Tools:
        raise BackendMissing("COLMAP not found")

    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _p, title, text: warnings.append(text))
    page = ProjectPage(Project.create(tmp_path / "p"), missing)
    qtbot.addWidget(page)
    page.import_folder(photos)
    page.start_run()
    assert not page.runner.running
    assert warnings and "COLMAP not found" in warnings[0]


def test_reopened_project_shows_earlier_results(
    qtbot: QtBot, page: ProjectPage, photos: Path, fake_tools: Tools
) -> None:
    page.import_folder(photos)
    page.start_run()
    _wait_idle(qtbot, page)
    reopened = ProjectPage(Project.open(page.project.root), lambda: fake_tools)
    qtbot.addWidget(reopened)
    assert _states(reopened)["texture"] == "Done"
    assert reopened.stages.topLevelItemCount() == 8
