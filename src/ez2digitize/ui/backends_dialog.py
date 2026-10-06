# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where the reconstruction tools are: settings dialog and the lookup it feeds."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ez2digitize import plugins
from ez2digitize.backends import brush, colmap, ffmpeg, openmvs
from ez2digitize.backends.common import BackendError, BackendMissing
from ez2digitize.backends.ffmpeg import FFmpeg
from ez2digitize.pipeline import Tools

COLMAP_KEY = "backends/colmap"
OPENMVS_KEY = "backends/openmvs_dir"
FFMPEG_KEY = "backends/ffmpeg"
BRUSH_KEY = "backends/brush"


def locate_tools(settings: QSettings) -> Tools:
    """Find both tools: paths from the settings if set, else the usual search;
    and the plugins chosen in Settings -> Plugins.

    Raises BackendError (BackendMissing, or PluginError) with a message for the user.
    """
    colmap_path = str(settings.value(COLMAP_KEY, "") or "")
    openmvs_dir = str(settings.value(OPENMVS_KEY, "") or "")
    brush_path = str(settings.value(BRUSH_KEY, "") or "")
    try:
        splats: brush.Brush | None = brush.locate(Path(brush_path) if brush_path else None)
    except BackendMissing:
        splats = None  # only needed for splats
    return Tools(
        colmap=colmap.locate(Path(colmap_path) if colmap_path else None),
        openmvs=openmvs.locate(Path(openmvs_dir) if openmvs_dir else None),
        brush=splats,
        poses=plugins.chosen("poses"),
        splats=plugins.chosen("splats"),
    )


def locate_ffmpeg(settings: QSettings) -> FFmpeg:
    """FFmpeg (for video import), from the settings if set, else the usual search."""
    path = str(settings.value(FFMPEG_KEY, "") or "")
    return ffmpeg.locate(Path(path) if path else None)


class BackendsDialog(QDialog):
    """Lets the user point at COLMAP and the OpenMVS folder, and checks them."""

    def __init__(self, settings: QSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reconstruction tools")
        self.settings = settings
        self.colmap_edit = QLineEdit(str(settings.value(COLMAP_KEY, "") or ""))
        self.colmap_edit.setPlaceholderText("found automatically (EZ2D_COLMAP, bundle, PATH)")
        self.openmvs_edit = QLineEdit(str(settings.value(OPENMVS_KEY, "") or ""))
        self.openmvs_edit.setPlaceholderText("found automatically (EZ2D_OPENMVS_DIR, bundle, PATH)")
        self.ffmpeg_edit = QLineEdit(str(settings.value(FFMPEG_KEY, "") or ""))
        self.ffmpeg_edit.setPlaceholderText("found automatically (EZ2D_FFMPEG, PATH)")
        self.brush_edit = QLineEdit(str(settings.value(BRUSH_KEY, "") or ""))
        self.brush_edit.setPlaceholderText("found automatically (EZ2D_BRUSH, bundle, PATH)")
        self.check_label = QLabel()
        self.check_label.setWordWrap(True)

        form = QFormLayout()
        form.addRow("COLMAP program:", self._with_browse(self.colmap_edit, folder=False))
        form.addRow("OpenMVS folder:", self._with_browse(self.openmvs_edit, folder=True))
        form.addRow("FFmpeg program:", self._with_browse(self.ffmpeg_edit, folder=False))
        form.addRow("Brush program:", self._with_browse(self.brush_edit, folder=False))
        check = QPushButton("Check")
        check.clicked.connect(self.check)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"EZ2DIGITIZE is tested with COLMAP {colmap.PINNED_VERSION} and "
                f"OpenMVS {openmvs.PINNED_VERSION} and Brush {brush.PINNED_VERSION}; FFmpeg is "
                "only needed to import videos, Brush for splats. Leave a field empty to "
                "search for it."
            )
        )
        layout.addLayout(form)
        layout.addWidget(check)
        layout.addWidget(self.check_label)
        layout.addWidget(buttons)
        self.resize(560, self.sizeHint().height())

    def _with_browse(self, edit: QLineEdit, *, folder: bool) -> QWidget:
        browse = QPushButton("Browse…")

        def choose() -> None:
            if folder:
                path = QFileDialog.getExistingDirectory(self, "OpenMVS folder", edit.text())
            else:
                path, _ = QFileDialog.getOpenFileName(self, "Program", edit.text())
            if path:
                edit.setText(path)

        browse.clicked.connect(choose)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        layout.addWidget(browse)
        return row

    def _store(self) -> None:
        self.settings.setValue(COLMAP_KEY, self.colmap_edit.text().strip())
        self.settings.setValue(OPENMVS_KEY, self.openmvs_edit.text().strip())
        self.settings.setValue(FFMPEG_KEY, self.ffmpeg_edit.text().strip())
        self.settings.setValue(BRUSH_KEY, self.brush_edit.text().strip())

    def check(self) -> None:
        """Try the entered paths without saving them."""
        keys = (COLMAP_KEY, OPENMVS_KEY, FFMPEG_KEY, BRUSH_KEY)
        previous = [self.settings.value(key) for key in keys]
        self._store()
        lines = []
        try:
            tools = locate_tools(self.settings)
        except BackendError as exc:
            lines.append(f"Problem: {exc}")
        else:
            sfm, mvs = tools.colmap, tools.openmvs
            lines += [
                f"COLMAP {sfm.version}{_untested(sfm.supported, colmap.PINNED_VERSION)}: "
                f"{sfm.path}",
                f"OpenMVS {mvs.version}{_untested(mvs.supported, openmvs.PINNED_VERSION)}: "
                f"{mvs.bin_dir}",
            ]
            if tools.brush is None:
                lines.append("Brush (only for splats): not found")
            else:
                lines.append(
                    f"Brush {tools.brush.version}"
                    f"{_untested(tools.brush.supported, brush.PINNED_VERSION)}: {tools.brush.path}"
                )
            for plugin in tools.used_plugins():
                lines.append(f"{plugins.SLOT_LABELS[plugin.slot]}: {plugins.describe(plugin)}")
        try:
            video = locate_ffmpeg(self.settings)
        except BackendError as exc:
            lines.append(f"FFmpeg (only for videos): {exc}")
        else:
            old = "" if video.supported else f" (older than {ffmpeg.SUPPORTED_MAJOR}.0)"
            lines.append(f"FFmpeg {video.version}{old}: {video.path}")
        self.check_label.setText("\n".join(lines))
        for key, value in zip(keys, previous, strict=True):
            self.settings.setValue(key, value or "")

    def save(self) -> None:
        self._store()
        self.accept()


def _untested(supported: bool, pinned: str) -> str:
    return "" if supported else f" (not the tested {pinned})"
