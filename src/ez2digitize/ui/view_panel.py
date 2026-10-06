# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The "3D view" tab: pick a result of the project and look at it.

The viewer (ui.viewer, a QtWebEngine page) starts only when the tab is
first shown: it costs a Chromium process. Meshes converted for it are
cached in a temporary folder that goes away with the panel.

With the camera placement or the dense cloud on screen, the crop box can
be set (see ez2digitize.crop): ticking "Crop box" starts from a box around
most of the sparse points; dragging its handles in the view, or turning
it, saves it to the project, and the next build keeps only what is inside.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTemporaryDir
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ez2digitize import crop, views
from ez2digitize.core.project import Project
from ez2digitize.ui.viewer import ViewerWidget

# The views drawn in the reconstruction's frame, where the box belongs (the
# exported mesh has been moved onto the ground).
CROP_VIEWS = ("cameras", "dense")
CROP_HINT = "Drag the yellow handles to move the box's faces. The next build keeps what is inside."


class ViewPanel(QWidget):
    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.available: list[views.View] = []
        self.viewer: ViewerWidget | None = None
        self._cache = QTemporaryDir()
        self._locked = False
        self._syncing = False

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

        self.use_crop = QCheckBox("Crop box")
        self.use_crop.setToolTip(
            "Keep only what is inside a box in the dense reconstruction (and so the mesh). "
            "Without one, OpenMVS picks the region itself, often with some of the table."
        )
        self.use_crop.toggled.connect(self._on_crop_toggled)
        self.yaw = QDoubleSpinBox()
        self.yaw.setRange(-180.0, 180.0)
        self.yaw.setSingleStep(5.0)
        self.yaw.setSuffix("°")
        self.yaw.setWrapping(True)
        self.yaw.setToolTip("Turn the box about the vertical axis")
        self.yaw.valueChanged.connect(self._on_yaw_changed)
        self.fit = QPushButton("Fit to the points")
        self.fit.setToolTip("Start again from a box around most of the sparse points")
        self.fit.clicked.connect(self._fit_crop_box)
        self.crop_hint = QLabel()
        self.crop_hint.setWordWrap(True)
        self.crop_row = QWidget()
        crop_layout = QHBoxLayout(self.crop_row)
        crop_layout.setContentsMargins(0, 0, 0, 0)
        crop_layout.addWidget(self.use_crop)
        crop_layout.addWidget(QLabel("Turn:"))
        crop_layout.addWidget(self.yaw)
        crop_layout.addWidget(self.fit)
        crop_layout.addWidget(self.crop_hint, 1)
        self.crop_row.hide()

        self.body = QVBoxLayout()
        self.body.addWidget(self.placeholder, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self.crop_row)
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
        else:
            self._sync_crop()

    def set_locked(self, locked: bool) -> None:
        """No crop box changes while a reconstruction runs."""
        self._locked = locked
        self._sync_crop()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self._show_chosen()

    def _chosen(self) -> views.View | None:
        key = self.choice.currentData()
        return next((v for v in self.available if v.key == key), None)

    def _show_chosen(self) -> None:
        view = self._chosen()
        self._sync_crop()
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
            self.viewer.crop_changed.connect(self._on_crop_dragged)
            self.body.addWidget(self.viewer, 1)
            self._sync_crop()
        return self.viewer

    def _on_loaded(self, event: dict[str, Any]) -> None:
        count = event.get("count")
        unit = event.get("unit", "")
        seconds = float(event.get("load_ms", 0)) / 1000
        shown = f"{count:,} {unit}" if isinstance(count, int) else ""
        self.status.setText(f"{shown}, loaded in {seconds:.1f} s" if shown else "")

    # --- the crop box -----------------------------------------------------------

    def _sync_crop(self) -> None:
        """Show the crop controls and the box for the view on screen."""
        view = self._chosen()
        here = view is not None and view.key in CROP_VIEWS
        self.crop_row.setVisible(here)
        box = crop.current(self.project)
        upright_box = crop.to_upright(box, views.upright_rotation(self.project)) if box else None
        self._syncing = True
        self.use_crop.setChecked(box is not None)
        self.yaw.setValue(upright_box.yaw if upright_box else 0.0)
        self._syncing = False
        editable = here and not self._locked
        self.use_crop.setEnabled(editable)
        self.yaw.setEnabled(editable and box is not None)
        self.fit.setEnabled(editable and box is not None)
        if box is not None:
            self.crop_hint.setText(CROP_HINT)
        elif crop.stored(self.project) is not None:
            self.crop_hint.setText(
                "The crop box was set on an earlier camera placement; tick it to set it again."
            )
        else:
            self.crop_hint.setText("")
        if self.viewer is not None:
            shown = upright_box.to_dict() if (here and upright_box) else None
            self.viewer.set_crop_box(shown, editable=editable)

    def _save(self, upright_box: crop.UprightBox | None) -> None:
        run = crop.camera_run(self.project)
        if upright_box is None or run is None:
            crop.save(self.project, None)
        else:
            upright = views.upright_rotation(self.project)
            crop.save(self.project, crop.from_upright(upright_box, upright, run))
        self._sync_crop()

    def _on_crop_toggled(self, checked: bool) -> None:
        if self._syncing:
            return
        if checked:
            self._fit_crop_box()
        else:
            self._save(None)

    def _fit_crop_box(self) -> None:
        start = crop.automatic_for(self.project)
        if start is None:
            self.crop_hint.setText("Place the cameras first: the box starts from their points.")
            self._syncing = True
            self.use_crop.setChecked(False)
            self._syncing = False
            return
        self._save(start)
        if self.viewer is not None:
            self.viewer.frame_crop_box()

    def _on_yaw_changed(self, value: float) -> None:
        box = crop.current(self.project)
        if self._syncing or box is None:
            return
        upright_box = crop.to_upright(box, views.upright_rotation(self.project))
        self._save(replace(upright_box, yaw=value))

    def _on_crop_dragged(self, event: dict[str, Any]) -> None:
        if self._locked or crop.current(self.project) is None:
            self._sync_crop()  # put the box back
            return
        upright_box = crop.UprightBox.from_dict(event)
        if upright_box is not None:
            self._save(upright_box)
            self.status.setText("Crop box saved: the next build keeps what is inside")
