# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The "3D view" tab: pick a result of the project and look at it.

The viewer (ui.viewer, a QtWebEngine page) starts only when the tab is
first shown: it costs a Chromium process. Meshes converted for it are
cached in a temporary folder that goes away with the panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTemporaryDir
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ez2digitize import views
from ez2digitize.core.project import Project
from ez2digitize.ui.viewer import ViewerWidget


class ViewPanel(QWidget):
    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.available: list[views.View] = []
        self.viewer: ViewerWidget | None = None
        self._cache = QTemporaryDir()

        self.choice = QComboBox()
        self.choice.currentIndexChanged.connect(lambda _i: self._show_chosen())
        self.status = QLabel()
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.placeholder = QLabel("Build a mesh or splats to see them here.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row = QHBoxLayout()
        row.addWidget(QLabel("Show:"))
        row.addWidget(self.choice)
        row.addWidget(self.status, 1)
        self.body = QVBoxLayout()
        self.body.addWidget(self.placeholder, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addLayout(self.body, 1)

    def refresh(self, prefer: views.ViewKey | None = None) -> None:
        """Re-read what the project has; show `prefer` (or keep the current one)."""
        current = self.choice.currentData()
        self.available = views.available(self.project)
        keys = [v.key for v in self.available]
        self.choice.blockSignals(True)
        self.choice.clear()
        for view in self.available:
            self.choice.addItem(view.label, view.key)
        wanted = prefer if prefer in keys else current if current in keys else None
        self.choice.setCurrentIndex(keys.index(wanted) if wanted else 0 if keys else -1)
        self.choice.blockSignals(False)
        self.choice.setEnabled(bool(keys))
        self.placeholder.setVisible(not keys and self.viewer is None)
        if self.isVisible():
            self._show_chosen()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self._show_chosen()

    def _chosen(self) -> views.View | None:
        key = self.choice.currentData()
        return next((v for v in self.available if v.key == key), None)

    def _show_chosen(self) -> None:
        view = self._chosen()
        if view is None:
            if self.viewer is not None:
                self.viewer.clear()
            return
        viewer = self._viewer()
        shown = viewer.shown
        if shown is not None and (shown.key, shown.run_id) == (view.key, view.run_id):
            return  # already on screen
        self.status.setText("Loading…")
        viewer.show_view(view)

    def _viewer(self) -> ViewerWidget:
        if self.viewer is None:
            self.placeholder.hide()
            cache = Path(self._cache.path()) if self._cache.isValid() else self.project.root
            self.viewer = ViewerWidget(cache)
            self.viewer.loaded.connect(self._on_loaded)
            self.viewer.failed.connect(lambda message: self.status.setText(message))
            self.body.addWidget(self.viewer, 1)
        return self.viewer

    def _on_loaded(self, event: dict[str, Any]) -> None:
        count = event.get("count")
        unit = event.get("unit", "")
        seconds = float(event.get("load_ms", 0)) / 1000
        shown = f"{count:,} {unit}" if isinstance(count, int) else ""
        self.status.setText(f"{shown}, loaded in {seconds:.1f} s" if shown else "")
