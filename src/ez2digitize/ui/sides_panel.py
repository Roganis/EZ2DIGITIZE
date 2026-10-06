# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The "Both sides" panel: guides a two-sided scan (see ez2digitize.sides).

How to turn the object over and photograph it again, which import is which
side (each import has a First side / Turned over choice), and a checklist:
photos of both sides, a mask for every photo. Importing the second side
and making masks are done by the page (`import_other_side`, `show_masks`).
"""

from __future__ import annotations

from html import escape

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ez2digitize import sides
from ez2digitize.core.capture import CaptureError, list_bundles
from ez2digitize.core.project import Project

SIDE_CHOICES = (("First side", False), ("Turned over", True))
SOURCE_LABELS = {"folder": "Photos", "video": "Video", "upload": "Phone"}

HOW_TO = (
    "<b>Scanning both sides</b><br>"
    "1. Photograph the object all around as usual, including a low ring that shows "
    "its sides.<br>"
    "2. Turn it over (upside down, or onto a side) and photograph it again the same "
    "way, with the same light. The sides of the object must show in both sets: that "
    "is where they are joined.<br>"
    "3. Import the second set with <b>Import the turned-over side…</b>, make masks for "
    "every photo, and build. Masks are what join the sides: the object moved between "
    "the sets, the table didn't."
)


class SidesPanel(QWidget):
    """Which import is which side, and what is left to do."""

    # The user changed an import's side.
    sides_changed = Signal()
    # Buttons the page acts on: import a folder as the turned-over side; show
    # the masks tab.
    import_other_side = Signal()
    show_masks = Signal()

    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self._filling = False

        self.how_to = QLabel(HOW_TO)
        self.how_to.setWordWrap(True)
        self.how_to.setTextFormat(Qt.TextFormat.RichText)
        self.captures = QTreeWidget()
        self.captures.setColumnCount(3)
        self.captures.setHeaderLabels(["Import", "Photos", "Side"])
        self.captures.setRootIsDecorated(False)
        header = self.captures.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.checklist = QLabel()
        self.checklist.setWordWrap(True)
        self.checklist.setTextFormat(Qt.TextFormat.RichText)
        self.import_button = QPushButton("Import the turned-over side…")
        self.import_button.clicked.connect(self.import_other_side)
        self.masks_button = QPushButton("Masks…")
        self.masks_button.setToolTip("Make or review the masks (Masks tab)")
        self.masks_button.clicked.connect(self.show_masks)

        buttons = QHBoxLayout()
        buttons.addWidget(self.import_button)
        buttons.addWidget(self.masks_button)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.how_to)
        layout.addWidget(self.checklist)
        layout.addLayout(buttons)
        layout.addWidget(self.captures, 1)

    def set_locked(self, locked: bool) -> None:
        """No changes while a reconstruction or an import runs."""
        self.import_button.setEnabled(not locked)
        self.captures.setEnabled(not locked)

    def refresh(self) -> None:
        bundles = list_bundles(self.project)
        self._filling = True
        self.captures.clear()
        for bundle in bundles:
            source = SOURCE_LABELS.get(bundle.source, bundle.source)
            item = QTreeWidgetItem([f"{source}, {bundle.id}", str(len(bundle.images)), ""])
            self.captures.addTopLevelItem(item)
            combo = QComboBox()
            for label, flipped in SIDE_CHOICES:
                combo.addItem(label, flipped)
            combo.setCurrentIndex(1 if bundle.flipped else 0)
            combo.currentIndexChanged.connect(
                lambda _i, capture=bundle.id, box=combo: self._set_side(capture, box)
            )
            self.captures.setItemWidget(item, 2, combo)
        self._filling = False
        self._show_checklist()

    def _set_side(self, capture: str, box: QComboBox) -> None:
        if self._filling:
            return
        bundle = next((b for b in list_bundles(self.project) if b.id == capture), None)
        if bundle is None:
            return
        try:
            bundle.set_flipped(bool(box.currentData()))
        except (CaptureError, OSError) as exc:
            self.checklist.setText(escape(f"Could not change the side: {exc}"))
            return
        self._show_checklist()
        self.sides_changed.emit()

    def _show_checklist(self) -> None:
        bundles = list_bundles(self.project)
        found = sides.sides(bundles)
        if found is None:
            self.checklist.setText(
                "This project is one-sided: no import is marked as turned over." if bundles else ""
            )
            return
        missing = sides.without_mask(self.project, found.first + found.turned)
        total = len(found.first) + len(found.turned)
        rows = [
            (bool(found.first), f"First side: {len(found.first)} photos"),
            (bool(found.turned), f"Turned over: {len(found.turned)} photos"),
            (
                total > 0 and not missing,
                f"Masks: {total - len(missing)} of {total} photos have one",
            ),
        ]
        lines = [f"{'✔' if done else '✘'} {escape(text)}" for done, text in rows]
        if all(done for done, _ in rows):
            lines.append("Ready to build: both sides are joined through their masks.")
        self.checklist.setText("<br>".join(lines))

    def problems(self, *, use_masks: bool) -> list[str]:
        """What a run should be warned about (empty when one-sided or ready)."""
        return sides.check(self.project, list_bundles(self.project), use_masks=use_masks)
