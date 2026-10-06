# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Settings -> Plugins: install plugins, read and accept their licenses, choose them.

See ez2digitize.plugins. A plugin can only be chosen once its licenses were
shown here (or by `ez2d plugins`) and accepted.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ez2digitize import plugins
from ez2digitize.plugins import Plugin, PluginError, Slot

INTRO = (
    "Plugins add reconstruction tools that can't come with EZ2DIGITIZE, often research "
    "code or model weights under licenses that forbid commercial use. A plugin is a "
    "program that runs on this computer with your rights: install only plugins you trust. "
    "Its licenses are shown before it can be used; they may limit what you can do with "
    "the results."
)


class PluginLicenseDialog(QDialog):
    """A plugin's licenses, one tab each; Accept records the acceptance."""

    def __init__(self, plugin: Plugin, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{plugin.name}: licenses")
        self.plugin = plugin
        lines = [f"<b>{plugin.label()}</b>: {plugin.description or plugin.slot}"]
        if not plugin.free:
            lines.append(
                "At least one of these is not known to be a free license: it may allow "
                "only non-commercial or research use. Read it before accepting."
            )
        header = QLabel("<br>".join(lines))
        header.setWordWrap(True)
        header.setTextFormat(Qt.TextFormat.RichText)
        self.tabs = QTabWidget()
        for lic in plugin.licenses:
            try:
                body = lic.text()
            except OSError as exc:
                body = f"Can't read {lic.file}: {exc}"
            text = QPlainTextEdit(body)
            text.setReadOnly(True)
            text.setFont(QFont("monospace"))
            free = "" if lic.free else " ⚠"
            self.tabs.addTab(text, f"{lic.covers}: {lic.spdx}{free}")
        buttons = QDialogButtonBox()
        self.accept_button = buttons.addButton("Accept", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        if plugin.homepage:
            home = QPushButton("Plugin's website")
            home.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(plugin.homepage)))
            buttons.addButton(home, QDialogButtonBox.ButtonRole.HelpRole)
        layout = QVBoxLayout(self)
        layout.addWidget(header)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(buttons)
        self.resize(720, 560)


class PluginsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plugins")
        self.found = plugins.Installed(())
        intro = QLabel(INTRO)
        intro.setWordWrap(True)

        self.choices: dict[Slot, QComboBox] = {}
        form = QFormLayout()
        for slot in plugins.SLOTS:
            combo = QComboBox()
            combo.activated.connect(lambda _i, s=slot: self._on_choice(s))
            self.choices[slot] = combo
            form.addRow(f"{plugins.SLOT_LABELS[slot]}:", combo)

        self.table = QTreeWidget()
        self.table.setHeaderLabels(["Plugin", "Provides", "Licenses", "State"])
        self.table.setRootIsDecorated(False)
        self.table.itemSelectionChanged.connect(self._sync_buttons)
        self.problems = QLabel()
        self.problems.setWordWrap(True)

        install_folder = QPushButton("Install from folder…")
        install_folder.clicked.connect(self.choose_folder)
        install_zip = QPushButton("Install from .zip…")
        install_zip.clicked.connect(self.choose_zip)
        self.licenses_button = QPushButton("Licenses…")
        self.licenses_button.clicked.connect(self.show_selected_licenses)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_selected)
        open_folder = QPushButton("Open plugins folder")
        open_folder.clicked.connect(self._open_folder)
        row = QHBoxLayout()
        for button in (install_folder, install_zip, self.licenses_button, self.remove_button):
            row.addWidget(button)
        row.addStretch(1)
        row.addWidget(open_folder)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(self.table, 1)
        layout.addLayout(row)
        layout.addWidget(self.problems)
        layout.addWidget(buttons)
        self.resize(760, 480)
        self.refresh()

    # --- state ----------------------------------------------------------------

    def refresh(self) -> None:
        found = plugins.installed()
        self.found = found
        selected = self.selected()
        self.table.clear()
        for plugin in found.plugins:
            state = "Licenses accepted" if plugins.accepted(plugin) else "Licenses not accepted"
            if not plugin.runs_here:
                state += "; not for this system"
            item = QTreeWidgetItem(
                [
                    plugin.label(),
                    plugins.SLOT_LABELS[plugin.slot],
                    plugin.license_summary(),
                    state,
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, plugin.id)
            item.setToolTip(0, plugin.description or str(plugin.folder))
            self.table.addTopLevelItem(item)
            if selected is not None and plugin.id == selected.id:
                item.setSelected(True)
        for column in range(self.table.columnCount()):
            self.table.resizeColumnToContents(column)
        for slot, combo in self.choices.items():
            combo.clear()
            combo.addItem(f"{plugins.BUILT_IN[slot]} (built in)", None)
            for plugin in found.for_slot(slot):
                if plugin.runs_here:
                    combo.addItem(f"{plugin.label()} (plugin)", plugin.id)
            chosen = plugins.chosen_id(slot)
            index = combo.findData(chosen) if chosen is not None else 0
            if index < 0:  # chosen but gone or broken: say so, don't pretend
                combo.addItem(f"{chosen} (missing)", chosen)
                index = combo.count() - 1
            combo.setCurrentIndex(index)
        self.problems.setText(
            "\n".join(f"Not usable: {p}" for p in found.problems) if found.problems else ""
        )
        self.problems.setVisible(bool(found.problems))
        self._sync_buttons()

    def selected(self) -> Plugin | None:
        items = self.table.selectedItems()
        if not items:
            return None
        return self.found.get(str(items[0].data(0, Qt.ItemDataRole.UserRole)))

    def _sync_buttons(self) -> None:
        has = self.selected() is not None
        self.licenses_button.setEnabled(has)
        self.remove_button.setEnabled(has)

    # --- actions --------------------------------------------------------------

    def ask_accept(self, plugin: Plugin) -> bool:
        """Show the licenses; True (and recorded) if the user accepts them."""
        if PluginLicenseDialog(plugin, self).exec() != QDialog.DialogCode.Accepted:
            return False
        plugins.accept(plugin)
        return True

    def _on_choice(self, slot: Slot) -> None:
        combo = self.choices[slot]
        plugin_id = combo.currentData()
        plugin = self.found.get(str(plugin_id)) if plugin_id is not None else None
        if plugin_id is not None and plugin is None:
            self.refresh()
            return
        if plugin is not None and not plugins.accepted(plugin) and not self.ask_accept(plugin):
            self.refresh()  # back to the previous choice
            return
        try:
            plugins.choose(slot, plugin)
        except PluginError as exc:
            QMessageBox.warning(self, "Plugin not used", str(exc))
        self.refresh()

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Plugin folder")
        if folder:
            self.install(Path(folder))

    def choose_zip(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Plugin archive", "", "Zip archives (*.zip)")
        if path:
            self.install(Path(path))

    def install(self, source: Path) -> Plugin | None:
        try:
            plugin = plugins.install(source)
        except PluginError as exc:
            if "already installed" not in str(exc):
                QMessageBox.warning(self, "Not installed", str(exc))
                return None
            answer = QMessageBox.question(
                self, "Already installed", f"{exc}.\n\nReplace it with this version?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                return None
            try:
                plugin = plugins.install(source, replace_existing=True)
            except PluginError as again:
                QMessageBox.warning(self, "Not installed", str(again))
                return None
        if not plugins.accepted(plugin):
            self.ask_accept(plugin)
        self.refresh()
        return plugin

    def show_selected_licenses(self) -> None:
        plugin = self.selected()
        if plugin is not None:
            self.ask_accept(plugin)
            self.refresh()

    def remove_selected(self) -> None:
        plugin = self.selected()
        if plugin is None:
            return
        answer = QMessageBox.question(
            self, "Remove plugin", f"Remove {plugin.label()} and its files from {plugin.folder}?"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            plugins.remove(plugin.id)
        except (PluginError, OSError) as exc:
            QMessageBox.warning(self, "Not removed", str(exc))
        self.refresh()

    def _open_folder(self) -> None:
        folder = plugins.plugins_dir()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
