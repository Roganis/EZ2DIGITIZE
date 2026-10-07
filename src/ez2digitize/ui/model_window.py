# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""A model file from anywhere (a mesh, splats or a point cloud) in its own
window: the 3D view of it, which way is up, and Convert… to another format.

What the file holds and which formats it converts to are in
ez2digitize.models; the 3D view is the same page as a project's (see
ui.viewer), showing a `views.file_view`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTemporaryDir
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ez2digitize import models, views
from ez2digitize.ui.viewer import ViewerWidget

UP_CHOICES: tuple[tuple[str, models.Up], ...] = (
    ("Y up", "y"),
    ("Y down (upside down)", "-y"),
    ("Z up (3D printing)", "z"),
    ("Z down", "-z"),
)
KIND_TEXT = {"mesh": "Mesh", "splat": "Gaussian splats", "points": "Point cloud"}


def open_dialog_filter() -> str:
    """The file dialog filter for every model file the app opens."""
    patterns = " ".join(f"*{s}" for s in models.OPENABLE)
    return f"Model files ({patterns});;All files (*)"


class ModelWindow(QMainWindow):
    """Shows one model file; `convert` writes it in another format."""

    def __init__(self, model: models.ModelFile, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"{model.path.name} – EZ2DIGITIZE")
        self.resize(1000, 720)
        self._cache = QTemporaryDir()
        cache = Path(self._cache.path()) if self._cache.isValid() else model.path.parent

        self.kind = QLabel(f"<b>{KIND_TEXT[model.kind]}</b> · {model.path.name}")
        self.kind.setToolTip(str(model.path))
        self.up = QComboBox()
        for label, key in UP_CHOICES:
            self.up.addItem(label, key)
        self.up.setCurrentIndex([key for _label, key in UP_CHOICES].index(model.up))
        self.up.setToolTip(
            "Which way is up in the file. The usual way for its format is chosen; "
            "change it if the model shows lying down or upside down."
        )
        self.up.currentIndexChanged.connect(lambda _index: self.show_model())
        self.convert_button = QPushButton("Convert…")
        self.convert_button.setEnabled(bool(model.outputs))
        self.convert_button.setToolTip(
            "Save the model in another format"
            if model.outputs
            else "This file can be looked at, not converted"
        )
        self.convert_button.clicked.connect(lambda: self.convert())
        self.status = QLabel()
        self.status.setWordWrap(True)

        row = QHBoxLayout()
        row.addWidget(self.kind, 1)
        row.addWidget(QLabel("Up:"))
        row.addWidget(self.up)
        row.addWidget(self.convert_button)
        self.viewer = ViewerWidget(cache)
        self.viewer.loaded.connect(self._on_loaded)
        self.viewer.failed.connect(self.status.setText)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(row)
        layout.addWidget(self.viewer, 1)
        layout.addWidget(self.status)
        self.setCentralWidget(central)
        self.show_model()

    @property
    def chosen_up(self) -> models.Up:
        up: models.Up = self.up.currentData()
        return up

    def show_model(self) -> None:
        try:
            view = views.file_view(self.model, self.chosen_up)
        except views.ViewError as exc:
            self.status.setText(str(exc))
            return
        self.status.setText("Loading…")
        self.viewer.show_view(view)

    def _on_loaded(self, event: dict[str, Any]) -> None:
        count = event.get("count")
        unit = event.get("unit", "")
        seconds = float(event.get("load_ms", 0)) / 1000
        if isinstance(count, int):
            self.status.setText(f"{count:,} {unit}, loaded in {seconds:.1f} s")
        else:
            self.status.setText("")

    def convert(self, target: Path | None = None) -> list[Path] | None:
        """Ask where (unless `target` is given), convert, and say what was written."""
        outputs = self.model.outputs
        if not outputs:
            return None
        if target is None:
            first = next((s for s in outputs if s != self.model.suffix), outputs[0])
            filters = [f"{models.DESCRIPTIONS.get(s, s)} (*{s})" for s in outputs]
            first_filter = filters[outputs.index(first)]
            name, chosen = QFileDialog.getSaveFileName(
                self,
                "Convert to",
                str(self.model.path.with_name(f"{self.model.path.stem}_converted{first}")),
                ";;".join(filters),
                first_filter,
            )
            if not name:
                return None
            target = Path(name)
            if target.suffix.lower() not in outputs:  # no suffix typed: the filter's
                target = target.with_name(target.name + outputs[filters.index(chosen)])
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            files = models.convert(self.model.path, target)
        except models.ModelError as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Conversion failed", str(exc))
            return None
        QApplication.restoreOverrideCursor()
        self.status.setText("Saved " + ", ".join(f.name for f in files) + f" in {target.parent}")
        return files


def open_model(path: Path, parent: QWidget | None = None) -> ModelWindow | None:
    """A window showing the model file at `path`; None (after saying why) if it can't be read."""
    try:
        model = models.identify(path)
    except models.ModelError as exc:
        QMessageBox.warning(parent, "Cannot open the file", str(exc))
        return None
    window = ModelWindow(model, parent)
    window.setWindowFlag(Qt.WindowType.Window, True)
    window.show()
    return window
