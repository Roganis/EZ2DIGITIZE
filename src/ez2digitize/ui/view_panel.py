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

On the same views the scale is set (see ez2digitize.scale): "Pick two
points", click them, type their real distance, "Set scale"; exports then
come out in millimetres and metres.

And the orientation (see ez2digitize.upright): which way is up comes from
the photos; "Level" (three points on the surface the object stands on),
the tip buttons (quarter turns) and "Turn" correct it, "Automatic" goes
back to the estimate. The view reloads stood up the new way.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTemporaryDir, QTimer
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

from ez2digitize import crop, scale, upright, views
from ez2digitize.core.project import Project
from ez2digitize.orientation import IDENTITY, Vector
from ez2digitize.ui.viewer import ViewerWidget

# The views drawn in the reconstruction's frame, where the box belongs (the
# exported mesh has been moved onto the ground).
CROP_VIEWS = ("cameras", "dense")
CROP_HINT = "Drag the yellow handles to move the box's faces. The next build keeps what is inside."
PICK_HINT = "Click two points whose real distance you know (the ends of the object, say)."
LEVEL_HINT = "Click three points, far apart, on the surface the object stands on."


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

        self._picked: tuple[Vector, Vector] | None = None  # upright frame, not saved yet
        self.pick = QPushButton("Pick two points")
        self.pick.setCheckable(True)
        self.pick.setToolTip("Then type their real distance: exports come out in real units")
        self.pick.toggled.connect(self._on_pick_toggled)
        self.distance = QDoubleSpinBox()
        self.distance.setRange(0.1, 100_000.0)
        self.distance.setDecimals(1)
        self.distance.setSuffix(" mm")
        self.distance.setValue(100.0)
        self.distance.setToolTip("The real distance between the two points, measured on the object")
        self.set_scale = QPushButton("Set scale")
        self.set_scale.clicked.connect(self._on_set_scale)
        self.clear_scale = QPushButton("Clear")
        self.clear_scale.setToolTip("No scale: exports in the reconstruction's arbitrary units")
        self.clear_scale.clicked.connect(self._on_clear_scale)
        self.scale_hint = QLabel()
        self.scale_hint.setWordWrap(True)
        self.scale_row = QWidget()
        scale_layout = QHBoxLayout(self.scale_row)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.addWidget(QLabel("Scale:"))
        scale_layout.addWidget(self.pick)
        scale_layout.addWidget(QLabel("Real distance:"))
        scale_layout.addWidget(self.distance)
        scale_layout.addWidget(self.set_scale)
        scale_layout.addWidget(self.clear_scale)
        scale_layout.addWidget(self.scale_hint, 1)
        self.scale_row.hide()

        self.level = QPushButton("Level: pick 3 points")
        self.level.setCheckable(True)
        self.level.setToolTip("Make the surface the object stands on the ground")
        self.level.toggled.connect(self._on_level_toggled)
        self.tip_forward = QPushButton("Tip forward")
        self.tip_forward.setToolTip("A quarter turn about the left-right axis")
        self.tip_forward.clicked.connect(lambda: self._tilt("x"))
        self.tip_sideways = QPushButton("Tip sideways")
        self.tip_sideways.setToolTip("A quarter turn about the front-back axis")
        self.tip_sideways.clicked.connect(lambda: self._tilt("z"))
        self.turn = QDoubleSpinBox()
        self.turn.setRange(-180.0, 180.0)
        self.turn.setSingleStep(15.0)
        self.turn.setSuffix("°")
        self.turn.setWrapping(True)
        self.turn.setKeyboardTracking(False)
        self.turn.setToolTip("Turn about the vertical: which way the model faces in exports")
        # Applied once the value rests: each change reloads the view.
        self._turn_timer = QTimer(self)
        self._turn_timer.setSingleShot(True)
        self._turn_timer.setInterval(400)
        self._turn_timer.timeout.connect(self._apply_turn)
        self.turn.valueChanged.connect(self._on_turn_changed)
        self.automatic_up = QPushButton("Automatic")
        self.automatic_up.setToolTip("Back to the estimate from how the photos were held")
        self.automatic_up.clicked.connect(lambda: self._orient(None))
        self.orient_hint = QLabel()
        self.orient_hint.setWordWrap(True)
        self.orient_row = QWidget()
        orient_layout = QHBoxLayout(self.orient_row)
        orient_layout.setContentsMargins(0, 0, 0, 0)
        orient_layout.addWidget(QLabel("Upright:"))
        orient_layout.addWidget(self.level)
        orient_layout.addWidget(self.tip_forward)
        orient_layout.addWidget(self.tip_sideways)
        orient_layout.addWidget(QLabel("Turn:"))
        orient_layout.addWidget(self.turn)
        orient_layout.addWidget(self.automatic_up)
        orient_layout.addWidget(self.orient_hint, 1)
        self.orient_row.hide()

        self.body = QVBoxLayout()
        self.body.addWidget(self.placeholder, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self.crop_row)
        layout.addWidget(self.scale_row)
        layout.addWidget(self.orient_row)
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
        if (current := scale.current(self.project)) is not None:
            self.distance.setValue(current.distance_mm)
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
            self.viewer.measured.connect(self._on_measured)
            self.viewer.level_picked.connect(self._on_level_picked)
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
        self._sync_scale()
        self._sync_orientation()

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

    # --- the scale --------------------------------------------------------------

    def _sync_scale(self) -> None:
        """Show the scale controls, and the scale's points, for the view on screen."""
        view = self._chosen()
        here = view is not None and view.key in CROP_VIEWS
        self.scale_row.setVisible(here)
        editable = here and not self._locked and crop.camera_run(self.project) is not None
        current = scale.current(self.project)
        self.pick.setEnabled(editable)
        self.distance.setEnabled(editable)
        self.set_scale.setEnabled(editable and (self._picked is not None or current is not None))
        self.clear_scale.setEnabled(editable and scale.stored(self.project) is not None)
        if not editable and self.pick.isChecked():
            self.pick.setChecked(False)
        upright = views.upright_rotation(self.project)
        points: tuple[Vector, Vector] | None = None
        label = ""
        if self._picked is not None:
            points = self._picked
            label = "? mm"
            self.scale_hint.setText("Type their real distance, then Set scale.")
        elif current is not None:
            points = scale.to_upright(current.points, upright)
            label = f"{current.distance_mm:g} mm"
            self.scale_hint.setText(f"Set: {scale.describe(current)}. Exports are in real units.")
        elif scale.stored(self.project) is not None:
            self.scale_hint.setText(
                "The scale was set on an earlier camera placement; pick the points again."
            )
        elif not self.pick.isChecked():
            self.scale_hint.setText("Not set: exports are in arbitrary units.")
        if self.viewer is not None:
            shown = [list(p) for p in points] if (here and points) else None
            self.viewer.set_measure(shown, label)

    def _on_pick_toggled(self, on: bool) -> None:
        if on and self.level.isChecked():  # one picking at a time
            self.level.blockSignals(True)
            self.level.setChecked(False)
            self.level.blockSignals(False)
            self._sync_orientation()
        if self.viewer is None:
            return
        self.viewer.set_measuring(on)
        if on:
            self._picked = None
            self.scale_hint.setText(PICK_HINT)
            self.viewer.set_measure(None)
        else:
            self._sync_scale()

    def _on_measured(self, event: dict[str, Any]) -> None:
        try:
            a, b = (tuple(float(v) for v in p) for p in event["points"])
        except (KeyError, TypeError, ValueError):
            return
        if len(a) != 3 or len(b) != 3:
            return
        self._picked = (a, b)
        self.pick.blockSignals(True)
        self.pick.setChecked(False)
        self.pick.blockSignals(False)
        self._sync_scale()
        self.distance.setFocus()
        self.distance.selectAll()

    def _on_set_scale(self) -> None:
        run = crop.camera_run(self.project)
        if run is None:
            return
        if self._picked is not None:
            points = scale.from_upright(self._picked, views.upright_rotation(self.project))
        elif (current := scale.current(self.project)) is not None:
            points = current.points  # a corrected distance for the same points
        else:
            return
        try:
            scale.save(self.project, scale.make(points, self.distance.value(), run))
        except scale.ScaleError as exc:
            self.scale_hint.setText(f"Not set: {exc}.")
            return
        self._picked = None
        self._sync_scale()
        self.status.setText("Scale saved: exports are in millimetres (STL, 3MF) and metres")

    def _on_clear_scale(self) -> None:
        self._picked = None
        scale.save(self.project, None)
        self._sync_scale()

    # --- the orientation --------------------------------------------------------

    def _sync_orientation(self) -> None:
        view = self._chosen()
        here = view is not None and view.key in CROP_VIEWS
        self.orient_row.setVisible(here)
        editable = here and not self._locked and crop.camera_run(self.project) is not None
        for widget in (self.level, self.tip_forward, self.tip_sideways, self.turn):
            widget.setEnabled(editable)
        manual = upright.current(self.project)
        self.automatic_up.setEnabled(editable and upright.stored(self.project) is not None)
        if not editable and self.level.isChecked():
            self.level.setChecked(False)
        if not self._turn_timer.isActive():
            self.turn.blockSignals(True)
            self.turn.setValue(manual.turn if manual is not None else 0.0)
            self.turn.blockSignals(False)
        if self.level.isChecked():
            self.orient_hint.setText(LEVEL_HINT)
        elif manual is not None:
            self.orient_hint.setText("Corrected by hand; exports stand this way up.")
        elif upright.stored(self.project) is not None:
            self.orient_hint.setText(
                "Corrected on an earlier camera placement; from the photos until set again."
            )
        elif upright.automatic(self.project) is not None:
            self.orient_hint.setText("From how the photos were held.")
        else:
            self.orient_hint.setText("The photos don't say which way is up: level it.")

    def _orient(self, orientation: upright.Orientation | None) -> None:
        """Save a correction (None: automatic) and show the view stood up the new way."""
        upright.change(self.project, orientation)
        self.refresh()
        view = self._chosen()
        if self.viewer is not None and view is not None and self.isVisible():
            self.viewer.show_view(view)  # same run, new upright rotation
        self.status.setText("Orientation saved: exports stand this way up")

    def _start(self) -> upright.Orientation | None:
        try:
            return upright.starting_point(self.project)
        except upright.OrientationError:
            return None

    def _tilt(self, axis: str) -> None:
        start = self._start()
        if start is not None:
            self._orient(upright.tilted(start, axis))

    def _on_turn_changed(self, _value: float) -> None:
        self._turn_timer.start()

    def _apply_turn(self) -> None:
        start = self._start()
        if start is not None and abs(start.turn - self.turn.value()) > 1e-9:
            self._orient(upright.turned(start, self.turn.value()))

    def _on_level_toggled(self, on: bool) -> None:
        if on and self.pick.isChecked():  # one picking at a time
            self.pick.blockSignals(True)
            self.pick.setChecked(False)
            self.pick.blockSignals(False)
        if self.viewer is not None:
            self.viewer.set_picking(3 if on else 0, "level")
        self._sync_orientation()

    def _on_level_picked(self, event: dict[str, Any]) -> None:
        self.level.blockSignals(True)
        self.level.setChecked(False)
        self.level.blockSignals(False)
        try:
            shown = [tuple(float(v) for v in p) for p in event["points"]]
        except (KeyError, TypeError, ValueError):
            shown = []
        start = self._start()
        centre = upright.camera_centre(self.project)
        if len(shown) != 3 or any(len(p) != 3 for p in shown) or start is None or centre is None:
            self._sync_orientation()
            return
        rotation = views.upright_rotation(self.project)
        points = tuple(upright.mul_transposed(rotation or IDENTITY, p) for p in shown)  # type: ignore[arg-type]
        try:
            self._orient(upright.levelled(start, points, centre))  # type: ignore[arg-type]
        except upright.OrientationError as exc:
            self.orient_hint.setText(f"Not levelled: {exc}.")
