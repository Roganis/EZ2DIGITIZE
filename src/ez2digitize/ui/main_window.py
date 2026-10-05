# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ez2digitize import __version__
from ez2digitize.core.project import Project, ProjectError
from ez2digitize.ui.backends_dialog import BackendsDialog, locate_ffmpeg, locate_tools
from ez2digitize.ui.project_page import ProjectPage, ToolsFactory

ABOUT_TEXT = f"""<h3>EZ2DIGITIZE {__version__}</h3>
<p>Turn photos or video of small objects into textured meshes and
Gaussian splats.</p>
<p>This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version. It comes with ABSOLUTELY NO WARRANTY.</p>
<p>Third-party components and their licenses are listed in
THIRD_PARTY_LICENSES.</p>"""

LAST_PROJECT_KEY = "projects/last"


class MainWindow(QMainWindow):
    """Welcome page until a project is open, then that project's page.

    `settings` and `tools_factory` are injectable for tests; by default the
    app's QSettings and the tool paths stored in them are used.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        settings: QSettings | None = None,
        tools_factory: ToolsFactory | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("EZ2DIGITIZE")
        self.resize(1200, 760)
        self.settings = settings or QSettings()
        self.tools_factory = tools_factory or (lambda: locate_tools(self.settings))
        self.page: ProjectPage | None = None

        self.stack = QStackedWidget()
        self.welcome = self._welcome_page()
        self.stack.addWidget(self.welcome)
        self.setCentralWidget(self.stack)

        file_menu = self.menuBar().addMenu("&File")
        new_action = file_menu.addAction("&New Project…")
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.choose_new_project)
        open_action = file_menu.addAction("&Open Project…")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.choose_project_to_open)
        self.import_action = file_menu.addAction("&Import Photos…")
        self.import_action.setShortcut("Ctrl+I")
        self.import_action.triggered.connect(self._import_photos)
        self.close_action = file_menu.addAction("&Close Project")
        self.close_action.triggered.connect(self.close_project)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("&Quit")
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        settings_menu = self.menuBar().addMenu("&Settings")
        tools_action = settings_menu.addAction("&Reconstruction tools…")
        tools_action.triggered.connect(self.edit_backends)

        help_menu = self.menuBar().addMenu("&Help")
        self.diagnostics_action = help_menu.addAction("Export &Diagnostics…")
        self.diagnostics_action.setToolTip(
            "Save the project's logs and settings (not the photos) for a bug report"
        )
        self.diagnostics_action.triggered.connect(self._export_diagnostics)
        about_action = help_menu.addAction("&About EZ2DIGITIZE")
        about_action.triggered.connect(self.show_about)

        self._update_actions()

    def _welcome_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch(1)
        intro = QLabel(
            "<h2>EZ2DIGITIZE</h2><p>Photos of a small object in, a textured mesh out.</p>"
        )
        intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(intro)
        for text, slot in (
            ("New project…", self.choose_new_project),
            ("Open project…", self.choose_project_to_open),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            layout.addWidget(button, 0, Qt.AlignmentFlag.AlignHCenter)
        last = str(self.settings.value(LAST_PROJECT_KEY, "") or "")
        if last and (Path(last) / "project.json").is_file():
            reopen = QPushButton(f"Reopen {Path(last).name}")
            reopen.clicked.connect(lambda: self.open_project(Path(last)))
            layout.addWidget(reopen, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(2)
        return page

    # --- projects -----------------------------------------------------------

    def choose_new_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "New project: choose a name and location for its folder"
        )
        if path:
            self.new_project(Path(path))

    def choose_project_to_open(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open a project folder")
        if path:
            self.open_project(Path(path))

    def new_project(self, path: Path) -> bool:
        try:
            project = Project.create(path)
        except (ProjectError, OSError) as exc:
            QMessageBox.warning(self, "Cannot create project", str(exc))
            return False
        return self._show_project(project)

    def open_project(self, path: Path) -> bool:
        try:
            project = Project.open(path)
        except ProjectError as exc:
            QMessageBox.warning(self, "Cannot open project", str(exc))
            return False
        return self._show_project(project)

    def close_project(self) -> bool:
        """Back to the welcome page; False if a running reconstruction was kept."""
        if self.page is None:
            return True
        if not self._stop_running("Close the project?"):
            return False
        self.stack.removeWidget(self.page)
        self.page.deleteLater()
        self.page = None
        self.setWindowTitle("EZ2DIGITIZE")
        self._update_actions()
        return True

    def _show_project(self, project: Project) -> bool:
        if not self.close_project():
            return False
        self.page = ProjectPage(
            project, self.tools_factory, ffmpeg_factory=lambda: locate_ffmpeg(self.settings)
        )
        self.page.runner.running_changed.connect(lambda _running: self._update_actions())
        self.stack.addWidget(self.page)
        self.stack.setCurrentWidget(self.page)
        self.setWindowTitle(f"{project.name} – EZ2DIGITIZE")
        self.settings.setValue(LAST_PROJECT_KEY, str(project.root))
        self._update_actions()
        return True

    def _import_photos(self) -> None:
        if self.page is not None:
            self.page.choose_folder_to_import()

    def _export_diagnostics(self) -> None:
        if self.page is not None:
            self.page.export_diagnostics()

    def _update_actions(self) -> None:
        has_page = self.page is not None
        running = has_page and self.page is not None and self.page.runner.running
        self.import_action.setEnabled(has_page and not running)
        self.close_action.setEnabled(has_page)
        self.diagnostics_action.setEnabled(has_page)

    def _stop_running(self, question: str) -> bool:
        """Ask before abandoning a running reconstruction; cancel it if confirmed."""
        if self.page is not None:
            # Seconds at most; the inspection thread must not outlive its page.
            self.page.photo_checks.wait()
            if self.page.video_importer.running:
                answer = QMessageBox.question(
                    self, "Video import running", f"A video is being imported. {question}"
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return False
                self.page.video_importer.cancel()
                self.page.video_importer.wait()
        if self.page is None or not self.page.runner.running:
            return True
        answer = QMessageBox.question(
            self,
            "Reconstruction running",
            f"A reconstruction is running. {question} It will be cancelled.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        self.page.runner.cancel()
        self.page.runner.wait()
        return True

    # --- other --------------------------------------------------------------

    def edit_backends(self) -> None:
        BackendsDialog(self.settings, self).exec()

    def show_about(self) -> None:
        QMessageBox.about(self, "About EZ2DIGITIZE", ABOUT_TEXT)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        if self._stop_running("Quit anyway?"):
            event.accept()
        else:
            event.ignore()
