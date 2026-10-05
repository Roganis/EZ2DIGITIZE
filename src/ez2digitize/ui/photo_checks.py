# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The photo checks panel: findings, and photos left out of the reconstruction.

Photos not inspected yet (just imported, or from an older version) are
inspected on a worker thread first; it takes a few seconds per hundred
photos. Each finding lists its photos with a check box: unchecking one
leaves it out of the reconstruction (`CaptureBundle.set_excluded`); the
"Left out" group at the end brings photos back.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QStyle,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ez2digitize.core import photos
from ez2digitize.core.capture import CaptureBundle, CaptureError, list_bundles
from ez2digitize.core.photos import Finding
from ez2digitize.core.project import Project

# Item data: (capture id, file name) of a photo row; the finding of a finding row.
PHOTO_ROLE = Qt.ItemDataRole.UserRole
FINDING_ROLE = Qt.ItemDataRole.UserRole + 1

TITLES = {
    "unreadable": "Can't be read",
    "odd-size": "Different size",
    "low-resolution": "Low resolution",
    "no-focal-length": "No focal length",
    "mixed-cameras": "Several cameras",
    "blurry": "Blurry",
    "duplicate": "Imported twice",
    "few-photos": "Few photos",
}

# Inspections outlive the panel that started them: Qt aborts if a running
# thread is destroyed, and closing a project shouldn't wait for one. They
# end on their own within seconds; finished ones are dropped on the next start.
_inspectors: list[_Inspector] = []


class _Inspector(QThread):
    progress = Signal(int, int)  # photos done, photos to inspect
    failed = Signal(str)

    def __init__(self, project: Project) -> None:
        super().__init__()
        self.project = project

    def run(self) -> None:
        try:
            bundles = [b for b in list_bundles(self.project) if photos.needs_inspection(b)]
            total = sum(
                1 for b in bundles for info in photos.photo_infos(b).values() if info is None
            )
            done = 0
            for bundle in bundles:

                def report(count: int, _total: int, base: int = done) -> None:
                    self.progress.emit(min(base + count, total), total)

                done += photos.inspect_bundle(bundle, on_progress=report)
        except (OSError, CaptureError) as exc:
            self.failed.emit(str(exc))


class PhotoChecks(QWidget):
    """Shows the project's photo findings; lets the user leave photos out."""

    # Photos were left out or brought back (the page updates its counts).
    exclusions_changed = Signal()

    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.findings: list[Finding] = []
        self._inspector: _Inspector | None = None
        self._locked = False
        self._filling = False

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.currentItemChanged.connect(self._show_details)
        # Tree rows can't wrap; the selected finding is explained here.
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.summary)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.details)

    @property
    def inspecting(self) -> bool:
        return self._inspector is not None

    def set_locked(self, locked: bool) -> None:
        """No changes while a reconstruction reads the captures."""
        self._locked = locked
        self.tree.setEnabled(not locked and not self.inspecting)

    def refresh(self) -> None:
        """Re-read the captures; inspect new photos first if there are any."""
        if self.inspecting:
            return
        bundles = list_bundles(self.project)
        if any(photos.needs_inspection(b) for b in bundles):
            self._start_inspection()
            return
        self.findings = photos.check_project(bundles)
        self._fill(bundles)

    def wait(self, timeout_ms: int = 30_000) -> bool:
        """Block until a running inspection ends (for shutdown); True if it did."""
        inspector = self._inspector
        return inspector is None or inspector.wait(timeout_ms)

    # --- inspection -----------------------------------------------------------

    def _start_inspection(self) -> None:
        _inspectors[:] = [i for i in _inspectors if i.isRunning()]
        self._inspector = _Inspector(self.project)
        _inspectors.append(self._inspector)
        self._inspector.progress.connect(self._on_progress)
        self._inspector.failed.connect(self._on_failed)
        self._inspector.finished.connect(self._on_finished)
        self.summary.setText("Checking the photos…")
        self.tree.setEnabled(False)
        self._inspector.start()

    def _on_progress(self, done: int, total: int) -> None:
        self.summary.setText(f"Checking the photos… {done} of {total}")

    def _on_failed(self, message: str) -> None:
        self.summary.setText(f"The photos could not be checked: {message}")

    def _on_finished(self) -> None:
        self._inspector = None
        self.tree.setEnabled(not self._locked)
        if not self.summary.text().startswith("The photos could not be checked"):
            self.refresh()

    # --- display --------------------------------------------------------------

    def _fill(self, bundles: list[CaptureBundle]) -> None:
        self._filling = True
        self.tree.clear()
        style = self.style()
        icons = {
            "error": style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxCritical),
            "warning": style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning),
        }
        for finding in self.findings:
            title = TITLES.get(finding.code, finding.code)
            if finding.files:
                count = len(finding.files)
                title += f" ({count} photo{'s' if count != 1 else ''})"
            item = QTreeWidgetItem([title])
            item.setIcon(0, icons[finding.level])
            item.setToolTip(0, finding.message)
            item.setData(0, FINDING_ROLE, finding.message)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            for name in finding.files:
                capture, file_name = _split(finding.capture, name)
                item.addChild(_photo_item(capture, file_name, used=True))
            self.tree.addTopLevelItem(item)
            item.setExpanded(len(finding.files) <= 10)
        left_out = [(b.id, f.name) for b in bundles for f in b.excluded]
        if left_out:
            group = QTreeWidgetItem([f"Left out of the reconstruction ({len(left_out)})"])
            group.setFlags(Qt.ItemFlag.ItemIsEnabled)
            for capture, name in left_out:
                group.addChild(_photo_item(capture, name, used=False))
            self.tree.addTopLevelItem(group)
            group.setExpanded(True)
        self._filling = False
        first = self.tree.topLevelItem(0) if self.findings else None
        if first is not None:
            self.tree.setCurrentItem(first)
        self._show_details(first)

        used = sum(len(b.images) for b in bundles)
        if not bundles:
            self.summary.setText("Import photos to check them.")
        elif not self.findings:
            self.summary.setText(f"No problems found in {used} photos.")
        else:
            errors = sum(1 for f in self.findings if f.level == "error")
            warnings = len(self.findings) - errors
            counts = [
                f"{n} {word}{'s' if n != 1 else ''}"
                for n, word in ((errors, "error"), (warnings, "warning"))
                if n
            ]
            self.summary.setText(
                f"Checked {used} photos: {' and '.join(counts)}. Uncheck a photo to leave it "
                "out of the reconstruction."
            )

    def _show_details(self, item: QTreeWidgetItem | None, _previous: object = None) -> None:
        while item is not None and item.data(0, FINDING_ROLE) is None:
            item = item.parent()
        message = item.data(0, FINDING_ROLE) if item is not None else None
        self.details.setText(message or "")
        self.details.setVisible(bool(message))

    def _on_item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        data = item.data(0, PHOTO_ROLE)
        if self._filling or not data:
            return
        capture, name = data
        used = item.checkState(0) == Qt.CheckState.Checked
        bundle = next((b for b in list_bundles(self.project) if b.id == capture), None)
        if bundle is None:
            return
        bundle.set_excluded([name], excluded=not used)
        self.exclusions_changed.emit()
        # Not now: refreshing rebuilds the tree, deleting the item being changed.
        QTimer.singleShot(0, self.refresh)


def _split(capture: str | None, name: str) -> tuple[str, str]:
    """Findings about several captures name photos as `<capture>/<file>`."""
    if capture is not None:
        return capture, name
    prefix, _, file_name = name.rpartition("/")
    return prefix, file_name


def _photo_item(capture: str, name: str, *, used: bool) -> QTreeWidgetItem:
    item = QTreeWidgetItem([name])
    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
    item.setCheckState(0, Qt.CheckState.Checked if used else Qt.CheckState.Unchecked)
    item.setData(0, PHOTO_ROLE, (capture, name))
    item.setToolTip(0, f"{capture}/{name}")
    return item
