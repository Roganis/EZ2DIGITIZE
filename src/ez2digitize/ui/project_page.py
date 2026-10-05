# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The page of an open project: captures, settings, run, progress and log."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ez2digitize import diagnostics, presets, video
from ez2digitize.backends import colmap, ffmpeg, openmvs
from ez2digitize.backends.common import BackendError
from ez2digitize.backends.ffmpeg import FFmpeg
from ez2digitize.core.capture import (
    VIDEO_SUFFIXES,
    CaptureBundle,
    CaptureError,
    import_folder,
    list_bundles,
)
from ez2digitize.core.project import Project
from ez2digitize.core.stage import load_manifest
from ez2digitize.pipeline import STAGES, MeshResult, MeshSettings, Tools
from ez2digitize.ui.photo_checks import PhotoChecks
from ez2digitize.ui.pipeline_runner import Failure, PipelineRunner
from ez2digitize.ui.video_import import VideoImporter

STAGE_LABELS = {
    "features": "Find features",
    "matching": "Match photos",
    "mapping": "Place cameras",
    "undistort": "Undistort photos",
    "mask-undistort": "Prepare masks",
    "mvs-import": "Prepare dense step",
    "densify": "Dense point cloud",
    "mesh": "Build mesh",
    "refine": "Refine mesh",
    "texture": "Texture mesh",
}

# Detail -> OpenMVS resolution level (each level halves the image size).
DETAIL_LEVELS = (("High", 0), ("Medium", 1), ("Low", 2))
# What the finished mesh is exported as, into the project's exports/ folder.
EXPORT_CHOICES = (
    ("OBJ and GLB", ("obj", "glb")),
    ("GLB", ("glb",)),
    ("OBJ", ("obj",)),
    ("PLY (as OpenMVS writes it)", ("ply",)),
)
LOG_MAX_LINES = 5000

ToolsFactory = Callable[[], Tools]
FFmpegFactory = Callable[[], FFmpeg]


class ProjectPage(QWidget):
    """Shows one project and runs its pipeline.

    `tools_factory` finds the backends when a run starts, `ffmpeg_factory`
    FFmpeg when a video is imported (both may raise BackendError); tests
    pass fakes.
    """

    project_changed = Signal()

    def __init__(
        self,
        project: Project,
        tools_factory: ToolsFactory,
        parent: QWidget | None = None,
        *,
        ffmpeg_factory: FFmpegFactory = ffmpeg.locate,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.tools_factory = tools_factory
        self.ffmpeg_factory = ffmpeg_factory
        self.runner = PipelineRunner(self)
        self.video_importer = VideoImporter(self)
        self.last_result: MeshResult | None = None
        self.last_failure: Failure | None = None
        self._stage_items: dict[str, QTreeWidgetItem] = {}
        self._stage_index, self._stage_count = 1, 1

        self.title = QLabel()
        title_font = QFont()
        title_font.setPointSizeF(title_font.pointSizeF() * 1.4)
        title_font.setBold(True)
        self.title.setFont(title_font)
        self.captures_label = QLabel()
        self.import_button = QPushButton("Import photos…")
        self.import_button.clicked.connect(self.choose_folder_to_import)
        self.import_video_button = QPushButton("Import video…")
        self.import_video_button.clicked.connect(self.choose_video_to_import)

        header = QHBoxLayout()
        header_text = QVBoxLayout()
        header_text.addWidget(self.title)
        header_text.addWidget(self.captures_label)
        header.addLayout(header_text, 1)
        header.addWidget(self.import_button, 0, Qt.AlignmentFlag.AlignTop)
        header.addWidget(self.import_video_button, 0, Qt.AlignmentFlag.AlignTop)

        self.quality = QComboBox()
        for quality in presets.QUALITIES:
            self.quality.addItem(presets.LABELS[quality], quality)
            self.quality.setItemData(
                self.quality.count() - 1, presets.HINTS[quality], Qt.ItemDataRole.ToolTipRole
            )
        self.quality.setCurrentIndex(presets.QUALITIES.index(presets.parse_quality(project.preset)))
        self.detail = QComboBox()
        for label, level in DETAIL_LEVELS:
            self.detail.addItem(label, level)
        self.detail.setToolTip("Image size used for the dense point cloud; High is slowest")
        self.export_formats = QComboBox()
        for label, formats in EXPORT_CHOICES:
            self.export_formats.addItem(label, formats)
        self.export_formats.setToolTip(
            "OBJ (with MTL and texture images) for most 3D programs; GLB is a single "
            "file for viewers and the web"
        )
        self.refine = QCheckBox("Refine the mesh (slow, sharper detail)")
        self.use_masks = QCheckBox("Use masks")
        self.use_masks.setChecked(True)
        self.video_frames = QSpinBox()
        self.video_frames.setRange(10, 1000)
        self.video_frames.setSingleStep(10)
        self.video_frames.setValue(video.DEFAULT_FRAMES)
        self.video_frames.setToolTip(
            "How many frames to keep from a video: the sharpest of each stretch of "
            "the video. About 100 suits an object filmed all around in a minute."
        )
        settings_box = QGroupBox("Settings")
        form = QFormLayout(settings_box)
        form.addRow("Quality:", self.quality)
        form.addRow("Save as:", self.export_formats)
        form.addRow(self.use_masks)
        form.addRow("Video frames:", self.video_frames)
        # Advanced: override the preset's dense detail and refinement, and see
        # the values a run will use.
        self.advanced = QGroupBox("Advanced: change the preset")
        self.advanced.setCheckable(True)
        self.advanced.setChecked(False)
        advanced_form = QFormLayout(self.advanced)
        advanced_form.addRow("Detail:", self.detail)
        advanced_form.addRow(self.refine)
        self.values = QLabel()
        self.values.setWordWrap(True)
        self.values.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        advanced_form.addRow(self.values)
        form.addRow(self.advanced)
        self.quality.currentIndexChanged.connect(self._show_values)
        self.advanced.toggled.connect(self._show_values)
        self.detail.currentIndexChanged.connect(self._show_values)
        self.refine.toggled.connect(self._show_values)
        self._show_values()

        self.run_button = QPushButton("Build mesh")
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self.start_run)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_run)
        self.cancel_button.setEnabled(False)
        self.overall = QProgressBar()
        self.overall.setRange(0, 1000)
        self.overall.setTextVisible(False)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)

        self.stages = QTreeWidget()
        self.stages.setColumnCount(3)
        self.stages.setHeaderLabels(["Step", "State", "Details"])
        self.stages.setRootIsDecorated(False)
        self.stages.setUniformRowHeights(True)
        header_view = self.stages.header()
        header_view.setStretchLastSection(True)
        for column in (0, 1):
            header_view.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.open_result_button = QPushButton("Open folder")
        self.open_result_button.clicked.connect(self.open_result_folder)
        self.open_log_button = QPushButton("Open full log")
        self.open_log_button.clicked.connect(self.open_failure_log)
        self.diagnostics_button = QPushButton("Export diagnostics…")
        self.diagnostics_button.setToolTip(
            "Save the logs and settings (not the photos) to attach to a bug report"
        )
        self.diagnostics_button.clicked.connect(self.export_diagnostics)
        result_row = QHBoxLayout()
        result_row.addWidget(self.result_label, 1)
        result_row.addWidget(self.open_result_button)
        result_row.addWidget(self.open_log_button)
        result_row.addWidget(self.diagnostics_button)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addLayout(header)
        left_layout.addWidget(settings_box)
        left_layout.addLayout(buttons)
        left_layout.addWidget(self.overall)
        left_layout.addWidget(self.status)
        left_layout.addWidget(self.stages, 1)
        left_layout.addLayout(result_row)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(LOG_MAX_LINES)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log.setFont(QFont("monospace"))
        self.log.setPlaceholderText("Output of the reconstruction tools appears here.")

        self.photo_checks = PhotoChecks(project)
        self.photo_checks.exclusions_changed.connect(self._show_counts)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.photo_checks, "Photo checks")
        self.tabs.addTab(self.log, "Log")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        r = self.runner
        r.running_changed.connect(self._on_running_changed)
        r.stage_started.connect(self._on_stage_started)
        r.progress.connect(self._on_progress)
        r.output.connect(self._on_output)
        r.notice.connect(self._on_notice)
        r.stage_finished.connect(self._on_stage_finished)
        r.succeeded.connect(self._on_succeeded)
        r.failed.connect(self._on_failed)
        r.cancelled.connect(self._on_cancelled)

        v = self.video_importer
        v.running_changed.connect(lambda _running: self._update_buttons())
        v.progress.connect(self._on_video_progress)
        v.succeeded.connect(self._on_video_imported)
        v.failed.connect(self._on_video_failed)
        v.cancelled.connect(self._on_video_cancelled)

        self.refresh()

    # --- state shown --------------------------------------------------------

    def refresh(self) -> None:
        """Re-read the project folder: captures, masks, earlier stage results."""
        self._show_counts()
        self.photo_checks.refresh()
        if not self.runner.running:
            self._show_previous_stages()

    def _show_counts(self) -> None:
        bundles = list_bundles(self.project)
        images = sum(len(b.images) for b in bundles)
        self.title.setText(self.project.name)
        if bundles:
            plural = "s" if len(bundles) != 1 else ""
            self.captures_label.setText(f"{images} photos in {len(bundles)} import{plural}")
        else:
            self.captures_label.setText("No photos yet: import a folder of photos to start.")
        has_masks = any(self.project.masks_dir.rglob("*.png"))
        self.use_masks.setEnabled(has_masks)
        self.use_masks.setToolTip(
            "" if has_masks else "This project has no masks; import them with the photos."
        )
        self._update_buttons()

    def _show_previous_stages(self) -> None:
        self.stages.clear()
        self._stage_items.clear()
        for stage in STAGES:
            manifest = load_manifest(self.project.stage_dir(stage))
            if manifest is None:
                continue
            item = self._item(stage)
            state = {"succeeded": "Done", "failed": "Failed", "cancelled": "Cancelled"}
            item.setText(1, state[manifest.status])
            item.setText(2, f"{manifest.wall_s:.1f} s")

    def _update_buttons(self) -> None:
        importing = self.video_importer.running
        running = self.runner.running
        busy = running or importing
        has_photos = any(b.images for b in list_bundles(self.project))
        self.run_button.setEnabled(not busy and has_photos)
        self.cancel_button.setEnabled(busy)
        self.import_button.setEnabled(not busy)
        self.import_video_button.setEnabled(not busy)
        for widget in (self.quality, self.advanced, self.export_formats, self.video_frames):
            widget.setEnabled(not busy)
        self.photo_checks.set_locked(busy)
        self.use_masks.setEnabled(not busy and any(self.project.masks_dir.rglob("*.png")))
        self.open_result_button.setVisible(self.last_result is not None)
        self.open_log_button.setVisible(
            self.last_failure is not None and self.last_failure.log is not None
        )
        self.diagnostics_button.setVisible(self.last_failure is not None)

    def _item(self, stage: str) -> QTreeWidgetItem:
        item = self._stage_items.get(stage)
        if item is None:
            item = QTreeWidgetItem([STAGE_LABELS.get(stage, stage), "", ""])
            self.stages.addTopLevelItem(item)
            self._stage_items[stage] = item
        return item

    # --- actions ------------------------------------------------------------

    @property
    def chosen_quality(self) -> presets.Quality:
        return presets.parse_quality(str(self.quality.currentData()))

    def settings(self) -> MeshSettings:
        """The preset, with the advanced panel's values if it is switched on."""
        if self.advanced.isChecked():
            settings = presets.mesh_settings(
                self.chosen_quality,
                level=int(self.detail.currentData()),
                refine=self.refine.isChecked(),
            )
        else:
            settings = presets.mesh_settings(self.chosen_quality)
        return replace(
            settings,
            export_formats=tuple(self.export_formats.currentData()),
            use_masks=self.use_masks.isChecked(),
        )

    def _show_values(self) -> None:
        if not self.advanced.isChecked():
            # Show what the preset uses, as the starting point for changing it.
            preset = presets.mesh_settings(self.chosen_quality)
            for widget in (self.detail, self.refine):
                widget.blockSignals(True)
            levels = [level for _, level in DETAIL_LEVELS]
            self.detail.setCurrentIndex(levels.index(preset.densify.resolution_level))
            self.refine.setChecked(preset.refine is not None)
            for widget in (self.detail, self.refine):
                widget.blockSignals(False)
        rows = presets.describe(self.settings())
        self.values.setText("\n".join(f"{label}: {value}" for label, value in rows))
        self.quality.setToolTip(presets.HINTS[self.chosen_quality])

    def choose_folder_to_import(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Import a folder of photos")
        if folder:
            self.import_folder(Path(folder))

    def import_folder(self, folder: Path) -> None:
        try:
            bundle, skipped = import_folder(self.project, folder)
        except CaptureError as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
            return
        message = f"Imported {len(bundle.files)} photos from {folder.name}."
        videos = [p for p in skipped if p.suffix.lower() in VIDEO_SUFFIXES]
        if videos:
            message += f" Import its {len(videos)} video(s) with Import video."
        if len(skipped) > len(videos):
            message += f" Skipped {len(skipped) - len(videos)} files that are not photos."
        self.status.setText(message)
        self.refresh()
        self.tabs.setCurrentWidget(self.photo_checks)
        self.project_changed.emit()

    def choose_video_to_import(self) -> None:
        patterns = " ".join(f"*{suffix}" for suffix in sorted(VIDEO_SUFFIXES))
        path, _ = QFileDialog.getOpenFileName(
            self, "Import a video", "", f"Videos ({patterns});;All files (*)"
        )
        if path:
            self.import_video(Path(path))

    def import_video(self, path: Path) -> None:
        """Start importing `path` in the background (see the video_importer signals)."""
        try:
            tool = self.ffmpeg_factory()
        except BackendError as exc:
            QMessageBox.warning(
                self,
                "FFmpeg not found",
                f"Importing a video needs FFmpeg. {exc}\n\nInstall it (e.g. from your "
                "distribution's packages) or set its location in Settings → "
                "Reconstruction tools.",
            )
            return
        self.overall.setValue(0)
        self.status.setText(f"Importing {path.name}…")
        self.video_importer.start(self.project, path, tool, self.video_frames.value())

    def start_run(self) -> None:
        try:
            tools = self.tools_factory()
        except BackendError as exc:
            QMessageBox.warning(
                self,
                "Reconstruction tools not found",
                f"{exc}\n\nSet their location in Settings → Reconstruction tools.",
            )
            return
        self.last_result = None
        self.last_failure = None
        self.result_label.clear()
        self.log.clear()
        self.tabs.setCurrentWidget(self.log)
        self.stages.clear()
        self._stage_items.clear()
        self.overall.setValue(0)
        self.status.setText("Starting…")
        for name, tool, pinned in (
            ("COLMAP", tools.colmap, colmap.PINNED_VERSION),
            ("OpenMVS", tools.openmvs, openmvs.PINNED_VERSION),
        ):
            if not tool.supported:
                self._on_notice(
                    f"{name} {tool.version} found; {pinned} is the tested version, steps may fail"
                )
        if self.project.preset != self.chosen_quality:
            self.project.preset = self.chosen_quality
            self.project.save()
        self.runner.start(self.project, tools, self.settings())

    def cancel_run(self) -> None:
        self.status.setText("Cancelling…")
        self.cancel_button.setEnabled(False)
        self.runner.cancel()
        self.video_importer.cancel()

    def open_result_folder(self) -> None:
        folder = self._result_folder()
        if folder is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _result_folder(self) -> Path | None:
        """The export folder (holding obj/ and the GLB), else the texture stage's."""
        result = self.last_result
        if result is None:
            return None
        if result.exports:
            first = result.exports[0]
            return first.parent.parent if first.parent.name in ("obj", "ply") else first.parent
        return result.files[0].parent if result.files else None

    def export_diagnostics(self, target: Path | None = None) -> Path | None:
        """Save the diagnostics zip (asking where, unless `target` is given)."""
        if target is None:
            default = Path.home() / diagnostics.default_name(self.project)
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Export diagnostics", str(default), "Zip files (*.zip)"
            )
            if not chosen:
                return None
            target = Path(chosen)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            path = diagnostics.write_diagnostics(self.project, target, tools=self._tool_report)
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return None
        finally:
            QApplication.restoreOverrideCursor()
        self.status.setText(
            f"Diagnostics saved to {path}. It holds logs and settings, not photos; the "
            "logs contain file paths."
        )
        return path

    def _tool_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {}
        try:
            tools = self.tools_factory()
        except BackendError as exc:
            report["reconstruction"] = {"error": str(exc)}
        else:
            report["colmap"] = {"version": tools.colmap.version, "path": str(tools.colmap.path)}
            report["openmvs"] = {
                "version": tools.openmvs.version,
                "path": str(tools.openmvs.bin_dir),
            }
        try:
            tool = self.ffmpeg_factory()
        except BackendError as exc:
            report["ffmpeg"] = {"error": str(exc)}
        else:
            report["ffmpeg"] = {"version": tool.version, "path": str(tool.path)}
        return report

    def open_failure_log(self) -> None:
        if self.last_failure is not None and self.last_failure.log is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_failure.log)))

    # --- video import signals -------------------------------------------------

    def _on_video_progress(self, message: str, fraction: float) -> None:
        self.status.setText(f"{message}… {fraction:.0%}")
        # Extracting is most of the time; choosing frames the rest.
        share = 0.85 if message.startswith("Extracting") else 0.15
        start = 0.0 if message.startswith("Extracting") else 0.85
        self.overall.setValue(int((start + share * fraction) * 1000))

    def _on_video_imported(self, bundle: CaptureBundle) -> None:
        info = bundle.source_info
        self.overall.setValue(1000)
        self.status.setText(
            f"Imported {info.get('frames')} frames from {info.get('video')}, the sharpest of "
            f"{info.get('candidates')}."
        )
        self.refresh()
        self.tabs.setCurrentWidget(self.photo_checks)
        self.project_changed.emit()

    def _on_video_failed(self, message: str) -> None:
        self.overall.setValue(0)
        self.status.setText(f"Video import failed: {message}")
        QMessageBox.warning(self, "Video import failed", message)

    def _on_video_cancelled(self) -> None:
        self.overall.setValue(0)
        self.status.setText("Video import cancelled")

    # --- runner signals -----------------------------------------------------

    def _on_running_changed(self, running: bool) -> None:
        self._update_buttons()

    def _on_stage_started(self, stage: str, index: int, count: int) -> None:
        self._stage_index, self._stage_count = index, count
        item = self._item(stage)
        item.setText(1, "Running")
        item.setText(2, "")
        self.stages.scrollToItem(item)
        self.status.setText(f"Step {index} of {count}: {STAGE_LABELS.get(stage, stage)}")
        self.overall.setValue(int((index - 1) / count * 1000))
        self.log.appendPlainText(f"── {STAGE_LABELS.get(stage, stage)} ──")

    def _on_progress(self, stage: str, message: str, fraction: float) -> None:
        item = self._item(stage)
        if fraction >= 0:
            item.setText(1, f"Running {fraction:.0%}")
            done = self._stage_index - 1 + min(fraction, 1.0)
            self.overall.setValue(int(done / self._stage_count * 1000))
        item.setText(2, message)

    def _on_output(self, lines: list[str]) -> None:
        self.log.appendPlainText("\n".join(lines))

    def _on_notice(self, message: str) -> None:
        self.log.appendPlainText(f"Note: {message}")
        self.status.setText(message)

    def _on_stage_finished(self, stage: str, status: str, reused: bool, seconds: float) -> None:
        item = self._item(stage)
        if reused:
            item.setText(1, "Unchanged")
            item.setText(2, "reused from the last run")
        else:
            item.setText(1, {"succeeded": "Done", "failed": "Failed"}.get(status, "Cancelled"))
            item.setText(2, f"{seconds:.1f} s")

    def _on_succeeded(self, result: MeshResult) -> None:
        self.last_result = result
        self.overall.setValue(1000)
        self.status.setText("Finished")
        folder = self._result_folder()
        self.result_label.setText(f"Textured mesh saved in {folder}" if folder else "Finished.")
        self._update_buttons()

    def _on_failed(self, failure: Failure) -> None:
        self.last_failure = failure
        self.status.setText(f"Stopped: {failure.message}")
        if failure.tail:
            self.log.appendPlainText("── last lines of the log ──")
            self.log.appendPlainText("\n".join(failure.tail))
        self.result_label.setText(
            "Something went wrong. The last lines of the tool's output are in the log."
            if failure.tail
            else ""
        )
        self._update_buttons()

    def _on_cancelled(self) -> None:
        self.status.setText("Cancelled")
        self._update_buttons()
