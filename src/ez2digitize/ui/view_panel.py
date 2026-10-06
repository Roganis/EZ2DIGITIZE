# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The "3D view" tab: pick a result of the project and look at it.

The viewer (ui.viewer, a QtWebEngine page) starts only when the tab is
first shown: it costs a Chromium process. Meshes converted for it are
cached in a temporary folder that goes away with the panel.

On the camera placement and the dense cloud, tools next to the view's
name work on it, one at a time, each with one row of controls:

- Coverage (camera placement only, see coverage.rings): the cameras' rings
  around the object with their gaps, the cameras that matched few photos
  or were placed far off, and a summary.
- Crop box (see ez2digitize.crop): "Use a crop box" starts from a box
  around most of the sparse points; dragging its handles in the view, or
  turning it, saves it to the project, and the next build keeps only what
  is inside. A box that is set shows with every tool; only this one drags it.
- Scale (see ez2digitize.scale): "Pick two points", click them, type their
  real distance, "Set scale"; exports then come out in millimetres and
  metres.
- Upright (see ez2digitize.upright): which way is up comes from the
  photos; "Level" (three points on the surface the object stands on), the
  tip buttons (quarter turns) and "Turn" correct it, "Automatic" goes back
  to the estimate. The view reloads stood up the new way.

What a tool draws (rings, the scale's points) shows only while it is
chosen, and leaving a tool stops its point picking.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTemporaryDir, QTimer
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ez2digitize import coverage, crop, markers, scale, upright, views
from ez2digitize.core.project import Project
from ez2digitize.orientation import IDENTITY, Vector
from ez2digitize.ui.viewer import ViewerWidget

# The views drawn in the reconstruction's frame, where the box belongs (the
# exported mesh has been moved onto the ground).
CROP_VIEWS = ("cameras", "dense")
# The tools: their button and the views they work on, in order.
TOOLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "coverage": ("Coverage", ("cameras",)),
    "crop": ("Crop box", CROP_VIEWS),
    "scale": ("Scale", CROP_VIEWS),
    "upright": ("Upright", CROP_VIEWS),
}
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
        self._coverage_of: tuple[views.View, str] | None = None  # (view, summary), cached
        self._wanted_tool = "coverage"  # the user's choice; another if the view lacks it

        self.choice = QComboBox()
        self.choice.currentIndexChanged.connect(lambda _i: self._show_chosen())
        self.status = QLabel()
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.placeholder = QLabel("Build a mesh or splats to see them here.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tools: dict[str, QPushButton] = {}
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        row = QHBoxLayout()
        row.addWidget(QLabel("Show:"))
        row.addWidget(self.choice)
        row.addSpacing(12)
        for name, (label, _views) in TOOLS.items():
            button = QPushButton(label)
            button.setCheckable(True)
            button.toggled.connect(partial(self._on_tool_toggled, name))
            self._tool_group.addButton(button)
            self.tools[name] = button
            row.addWidget(button)
        row.addWidget(self.status, 1)

        self.use_crop = QCheckBox("Use a crop box")
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
        self.from_markers = QPushButton("From markers")
        self.from_markers.setToolTip(
            "Measure the printed scale markers in the photos (found by itself after camera "
            "placement, unless a scale was set by hand)"
        )
        self.from_markers.clicked.connect(self._on_from_markers)
        self.marker_size = QDoubleSpinBox()
        self.marker_size.setRange(5.0, 200.0)
        self.marker_size.setDecimals(1)
        self.marker_size.setSuffix(" mm")
        self.marker_size.setValue(markers.project_size(project))
        self.marker_size.setToolTip("The markers' black squares as printed: measure one")
        self.marker_size.valueChanged.connect(self._on_marker_size)
        self.sheet = QPushButton("Marker sheet…")
        self.sheet.setToolTip("Save the sheet of markers to print (A4, SVG)")
        self.sheet.clicked.connect(self.save_marker_sheet)
        self.scale_hint = QLabel()
        self.scale_hint.setWordWrap(True)
        self.scale_row = QWidget()
        scale_layout = QHBoxLayout(self.scale_row)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.addWidget(self.pick)
        scale_layout.addWidget(QLabel("Real distance:"))
        scale_layout.addWidget(self.distance)
        scale_layout.addWidget(self.set_scale)
        scale_layout.addWidget(self.clear_scale)
        scale_layout.addWidget(QLabel("or"))
        scale_layout.addWidget(self.from_markers)
        scale_layout.addWidget(self.marker_size)
        scale_layout.addWidget(self.sheet)
        scale_layout.addWidget(self.scale_hint, 1)

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
        orient_layout.addWidget(self.level)
        orient_layout.addWidget(self.tip_forward)
        orient_layout.addWidget(self.tip_sideways)
        orient_layout.addWidget(QLabel("Turn:"))
        orient_layout.addWidget(self.turn)
        orient_layout.addWidget(self.automatic_up)
        orient_layout.addWidget(self.orient_hint, 1)

        self.coverage_summary = QLabel()
        self.coverage_summary.setWordWrap(True)
        self.coverage_summary.setToolTip(
            "Rings of cameras around the object: orange and red where photos are missing"
        )
        self.coverage_row = QWidget()
        coverage_layout = QHBoxLayout(self.coverage_row)
        coverage_layout.setContentsMargins(0, 0, 0, 0)
        coverage_layout.addWidget(self.coverage_summary, 1)

        # One row of controls: the chosen tool's.
        self.tool_rows = {
            "coverage": self.coverage_row,
            "crop": self.crop_row,
            "scale": self.scale_row,
            "upright": self.orient_row,
        }
        self.tool_pages = QStackedWidget()
        self.tool_pages.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        for page in self.tool_rows.values():
            self.tool_pages.addWidget(page)
        self.tool_pages.hide()

        self.body = QVBoxLayout()
        self.body.addWidget(self.placeholder, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self.tool_pages)
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
        if (current := scale.current(self.project)) is not None and current.source == "points":
            self.distance.setValue(current.distance_mm)
        self.marker_size.blockSignals(True)
        self.marker_size.setValue(markers.project_size(self.project))
        self.marker_size.blockSignals(False)
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

    @property
    def tool(self) -> str | None:
        """The tool in use: the one chosen if the view has it, else its first; None if none."""
        view = self._chosen()
        usable = [t for t, (_label, keys) in TOOLS.items() if view and _offers(view, t, keys)]
        if not usable:
            return None
        return self._wanted_tool if self._wanted_tool in usable else usable[0]

    def choose_tool(self, name: str) -> None:
        self.tools[name].setChecked(True)

    def _on_tool_toggled(self, name: str, on: bool) -> None:
        if on and name != self._wanted_tool:
            self._wanted_tool = name
            self._sync_crop()

    def _sync_tools(self) -> None:
        """The buttons of the view's tools, and the chosen one's row."""
        view = self._chosen()
        tool = self.tool
        for name, button in self.tools.items():
            button.setVisible(view is not None and _offers(view, name, TOOLS[name][1]))
            if name == tool and not button.isChecked():
                button.blockSignals(True)
                button.setChecked(True)
                button.blockSignals(False)
        self.tool_pages.setVisible(tool is not None)
        if tool is not None:
            self.tool_pages.setCurrentWidget(self.tool_rows[tool])

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
        """Show the tools, the crop controls and the box for the view on screen."""
        self._sync_tools()
        view = self._chosen()
        here = view is not None and view.key in CROP_VIEWS
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
            # Shown with every tool (it changes what is built); dragged with its own.
            shown = upright_box.to_dict() if (here and upright_box) else None
            self.viewer.set_crop_box(shown, editable=editable and self.tool == "crop")
        self._sync_scale()
        self._sync_orientation()
        self._sync_coverage()

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
        editable = here and not self._locked and crop.camera_run(self.project) is not None
        current = scale.current(self.project)
        self.pick.setEnabled(editable)
        self.distance.setEnabled(editable)
        self.from_markers.setEnabled(editable)
        by_hand = current is not None and current.source == "points"
        self.set_scale.setEnabled(editable and (self._picked is not None or by_hand))
        self.clear_scale.setEnabled(editable and scale.stored(self.project) is not None)
        if (not editable or self.tool != "scale") and self.pick.isChecked():
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
            shown = [list(p) for p in points] if (points and self.tool == "scale") else None
            self.viewer.set_measure(shown, label)

    def _on_pick_toggled(self, on: bool) -> None:
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
        elif (current := scale.current(self.project)) is not None and current.source == "points":
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

    def _on_from_markers(self) -> None:
        run = crop.camera_run(self.project)
        if run is None:
            return
        size = self.marker_size.value()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            found = markers.measure_project(self.project, size)
        except markers.MarkerError as exc:
            found = None
            self.scale_hint.setText(f"Not set: {exc}.")
        finally:
            QApplication.restoreOverrideCursor()
        if found is None:
            if not self.scale_hint.text().startswith("Not set:"):
                self.scale_hint.setText("No markers found in the photos.")
            return
        self._picked = None
        scale.save(self.project, markers.to_scale(found, size, run))
        self._sync_scale()
        self.status.setText("Scale saved from the markers: exports are in real units")

    def _on_marker_size(self, value: float) -> None:
        self.project.settings[markers.SIZE_SETTING] = value
        self.project.save()

    def save_marker_sheet(self, target: Path | None = None) -> Path | None:
        if target is None:
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Save the marker sheet", str(Path.home() / "scale-markers.svg"), "SVG (*.svg)"
            )
            if not chosen:
                return None
            target = Path(chosen)
        target.write_text(markers.sheet_svg(self.marker_size.value()), encoding="utf-8")
        self.status.setText(f"Saved {target.name}: print it at 100 % (no fit to page)")
        return target

    def _on_clear_scale(self) -> None:
        self._picked = None
        scale.save(self.project, None)
        self._sync_scale()

    # --- coverage ---------------------------------------------------------------

    def _sync_coverage(self) -> None:
        view = self._chosen()
        if self.viewer is not None:
            self.viewer.set_coverage(self.tool == "coverage")
        if view is None or view.key != "cameras":
            return
        if self._coverage_of is None or self._coverage_of[0] != view:
            rings, weak = views.camera_coverage(view)
            if rings is not None:
                summary = coverage.describe(rings, len(weak))
            elif view.upright is None:
                summary = "Which way is up is unknown, so no rings: level the model (Upright)."
            else:
                summary = "Too few cameras, or a camera that stood still: no rings."
            self._coverage_of = (view, summary)
        self.coverage_summary.setText(self._coverage_of[1])

    # --- the orientation --------------------------------------------------------

    def _sync_orientation(self) -> None:
        view = self._chosen()
        here = view is not None and view.key in CROP_VIEWS
        editable = here and not self._locked and crop.camera_run(self.project) is not None
        for widget in (self.level, self.tip_forward, self.tip_sideways, self.turn):
            widget.setEnabled(editable)
        manual = upright.current(self.project)
        self.automatic_up.setEnabled(editable and upright.stored(self.project) is not None)
        if (not editable or self.tool != "upright") and self.level.isChecked():
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


def _offers(view: views.View, tool: str, keys: tuple[str, ...]) -> bool:
    """Whether the view has the tool: coverage only for rings round an object."""
    return view.key in keys and (tool != "coverage" or view.rings)
