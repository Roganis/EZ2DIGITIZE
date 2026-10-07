# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The project page driving the real pipeline with fake backends."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox
from pytestqt.qtbot import QtBot

from ez2digitize.backends.brush import Brush
from ez2digitize.backends.common import BackendMissing
from ez2digitize.backends.ffmpeg import FFmpeg
from ez2digitize.core.project import Project
from ez2digitize.pipeline import MeshResult, Tools
from ez2digitize.ui.project_page import ProjectPage

TIMEOUT_MS = 20_000


@pytest.fixture
def photos(tmp_path: Path) -> Path:
    folder = tmp_path / "photos"
    folder.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (folder / name).write_bytes(name.encode())
    return folder


@pytest.fixture
def page(qtbot: QtBot, tmp_path: Path, fake_tools: Tools) -> Iterator[ProjectPage]:
    page = ProjectPage(Project.create(tmp_path / "project"), lambda: fake_tools)
    qtbot.addWidget(page)
    yield page
    assert page.photo_checks.wait()
    assert page.masks_panel.wait()


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
    page.quality.setCurrentIndex(0)  # Fast -> resolution level 2
    page.run_button.click()  # the button, not start_run(): its signal passes an argument
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
    # The result opens in the 3D view (the viewer itself starts when shown).
    assert page.tabs.currentWidget() is page.view_panel
    choices = [page.view_panel.choice.itemText(i) for i in range(page.view_panel.choice.count())]
    assert choices == ["Textured mesh", "Dense point cloud"]  # the fakes write no sparse model
    assert page.view_panel.choice.currentText() == "Textured mesh"

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


@pytest.fixture
def video_page(
    qtbot: QtBot, tmp_path: Path, fake_tools: Tools, fake_ffmpeg: FFmpeg
) -> Iterator[ProjectPage]:
    page = ProjectPage(
        Project.create(tmp_path / "vproject"),
        lambda: fake_tools,
        ffmpeg_factory=lambda: fake_ffmpeg,
    )
    qtbot.addWidget(page)
    yield page
    page.video_importer.cancel()
    assert page.video_importer.wait() and page.photo_checks.wait()


def _clip(tmp_path: Path) -> Path:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    return clip


def test_videos_from_the_phone(qtbot: QtBot, video_page: ProjectPage) -> None:
    """Videos sent from the phone are imported one after the other, then cleared away."""
    from ez2digitize.core.capture import list_bundles
    from ez2digitize.upload import Received

    page = video_page
    page.video_frames.setValue(20)
    folder = page.project.captures_dir / ".received-videos-test"
    folder.mkdir()
    for name in ("VID_1.mp4", "VID_2.mp4"):
        (folder / name).write_bytes(b"video")
    page._phone_videos = Received(None, sorted(folder.iterdir()), folder, flipped=True)
    page._import_next_phone_video()
    qtbot.waitUntil(lambda: len(list_bundles(page.project)) == 2, timeout=TIMEOUT_MS)
    qtbot.waitUntil(lambda: page._phone_videos is None, timeout=TIMEOUT_MS)
    assert not folder.exists()
    bundles = list_bundles(page.project)
    assert sorted(b.source_info["video"] for b in bundles) == ["VID_1.mp4", "VID_2.mp4"]
    assert all(b.flipped for b in bundles)
    assert page.video_importer.wait() and page.photo_checks.wait()


def test_video_import(qtbot: QtBot, video_page: ProjectPage, tmp_path: Path) -> None:
    page = video_page
    page.video_frames.setValue(20)
    with qtbot.waitSignal(page.video_importer.succeeded, timeout=TIMEOUT_MS):
        page.import_video(_clip(tmp_path))
        assert not page.run_button.isEnabled() and page.cancel_button.isEnabled()
        assert not page.import_button.isEnabled()
    qtbot.waitUntil(lambda: not page.video_importer.running)
    assert page.status.text() == "Imported 20 frames from clip.mp4, the sharpest of 80."
    assert page.captures_label.text() == "20 photos in 1 import"
    assert page.run_button.isEnabled()
    assert page.tabs.currentWidget() is page.photo_checks


def test_video_import_cancel(
    qtbot: QtBot, video_page: ProjectPage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SLEEP", "ffmpeg")
    page = video_page
    with qtbot.waitSignal(page.video_importer.cancelled, timeout=TIMEOUT_MS):
        page.import_video(_clip(tmp_path))
        qtbot.waitUntil(lambda: page.cancel_button.isEnabled())
        page.cancel_run()
    assert page.status.text() == "Video import cancelled"
    assert "No photos yet" in page.captures_label.text()


def test_video_import_without_ffmpeg(
    qtbot: QtBot, tmp_path: Path, fake_tools: Tools, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing() -> FFmpeg:
        raise BackendMissing("FFmpeg not found")

    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _p, title, text: warnings.append(text))
    page = ProjectPage(Project.create(tmp_path / "p"), lambda: fake_tools, ffmpeg_factory=missing)
    qtbot.addWidget(page)
    page.import_video(_clip(tmp_path))
    assert not page.video_importer.running
    assert warnings and "Importing a video needs FFmpeg" in warnings[0]


def test_quality_presets_and_advanced_values(page: ProjectPage) -> None:
    assert page.chosen_quality == "balanced"
    assert page.settings().densify.resolution_level == 1 and page.settings().refine is None
    page.quality.setCurrentIndex(2)  # High
    assert page.detail.currentText() == "High" and page.refine.isChecked()
    assert "Refine mesh: yes, 1/2 size" in page.values.text()
    # The advanced panel overrides the preset only while it is switched on.
    page.advanced.setChecked(True)
    page.refine.setChecked(False)
    assert page.settings().refine is None and page.settings().densify.resolution_level == 0
    page.advanced.setChecked(False)
    assert page.settings().refine is not None and page.refine.isChecked()


def test_scene_subject(qtbot: QtBot, page: ProjectPage, fake_tools: Tools) -> None:
    from ez2digitize import subject

    assert page.chosen_subject == "object" and page.use_masks.isChecked()
    page.subject.setCurrentIndex(1)  # Room or outdoor scene
    assert subject.of(Project.open(page.project.root)) == "scene"  # stored at once
    assert not page.use_masks.isChecked()
    settings = page.settings()
    assert settings.subject == "scene" and settings.mesh.free_space_support
    assert "Subject: Room or outdoor scene" in page.values.text()
    reopened = ProjectPage(Project.open(page.project.root), lambda: fake_tools)
    qtbot.addWidget(reopened)
    assert reopened.chosen_subject == "scene" and not reopened.use_masks.isChecked()


def test_quality_is_saved_with_the_project(
    qtbot: QtBot, page: ProjectPage, photos: Path, fake_tools: Tools
) -> None:
    page.import_folder(photos)
    page.quality.setCurrentIndex(0)
    page.start_run()
    _wait_idle(qtbot, page)
    assert Project.open(page.project.root).preset == "fast"
    reopened = ProjectPage(Project.open(page.project.root), lambda: fake_tools)
    qtbot.addWidget(reopened)
    assert reopened.chosen_quality == "fast"
    assert reopened.photo_checks.wait()


def test_export_diagnostics(page: ProjectPage, photos: Path, tmp_path: Path) -> None:
    import zipfile

    page.import_folder(photos)
    path = page.export_diagnostics(tmp_path / "d.zip")
    assert path is not None and zipfile.is_zipfile(path)
    assert "Diagnostics saved" in page.status.text()


def test_build_splats(
    qtbot: QtBot,
    tmp_path: Path,
    photos: Path,
    fake_tools: Tools,
    fake_brush: Brush,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from ez2digitize import pipeline
    from ez2digitize.core.hardware import Gpu

    monkeypatch.setattr(pipeline, "detect_gpus", lambda: [Gpu("amd", "RX 7900 GRE")])
    tools = replace(fake_tools, brush=fake_brush)
    page = ProjectPage(Project.create(tmp_path / "sp"), lambda: tools)
    qtbot.addWidget(page)
    page.import_folder(photos)
    page.start_splats()
    _wait_idle(qtbot, page)
    assert page.last_failure is None, page.last_failure
    assert "splat" in page._stage_items
    assert page.result_label.text().startswith("Splats saved in")
    assert page.photo_checks.wait()


def test_build_splats_without_brush(
    qtbot: QtBot, page: ProjectPage, photos: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _p, title, text: warnings.append(text))
    page.import_folder(photos)
    page.start_splats()
    assert not page.runner.running and "Splats need Brush" in warnings[0]


def test_making_masks_blocks_runs_then_enables_use_masks(
    qtbot: QtBot, page: ProjectPage, photos: Path, fake_mask_worker: Path
) -> None:
    page.import_folder(photos)
    assert not page.use_masks.isEnabled()
    panel = page.masks_panel
    with qtbot.waitSignal(panel.busy_changed, timeout=TIMEOUT_MS):
        panel.make_masks()
    assert not page.run_button.isEnabled() and not page.import_button.isEnabled()
    qtbot.waitUntil(lambda: not panel.maker.running, timeout=TIMEOUT_MS)
    assert page.run_button.isEnabled()
    assert page.use_masks.isEnabled() and page.use_masks.isChecked()
    assert page.settings().use_masks


def test_place_cameras_then_look(qtbot: QtBot, page: ProjectPage, photos: Path) -> None:
    page.import_folder(photos)
    page.place_button.click()
    _wait_idle(qtbot, page)
    assert list(_states(page)) == ["features", "matching", "mapping", "undistort"]
    assert page.status.text().startswith("Cameras placed: 3 of 3 photos.")
    assert page.tabs.currentWidget() is page.view_panel
    assert page.last_result is None  # no mesh yet: nothing to open
    assert page.run_button.isEnabled() and page.place_button.isEnabled()


def test_a_bug_shows_where_it_happened(qtbot: QtBot, page: ProjectPage, photos: Path) -> None:
    def broken(*_args: object, **_kwargs: object) -> MeshResult:
        raise TypeError("'bool' object is not callable")

    page.import_folder(photos)
    page.start_run(broken)
    _wait_idle(qtbot, page)
    assert page.status.text().startswith("Stopped: unexpected error: TypeError")
    assert page.last_failure is not None and not page.last_failure.tail
    log = page.log.toPlainText()
    assert "where it happened" in log and "in broken" in log
