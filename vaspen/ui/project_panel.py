"""Project panel — structures/runs tree for a calculation project.

Left dock showing the open project (§7.15): a Structures group and a
Runs group with status labels. Double-click opens things into the main
window; a context menu on a run changes its status. Run CREATION from
the GUI lands in P2 (Generate All "write into project") — this panel
manages and consumes; the MCP ``create_run`` tool fills the tree in
the meantime.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.project import RUN_STATUSES, Project

#: Column indices of the tree.
_COL_NAME = 0
_COL_STATUS = 1


class ProjectPanel(QWidget):
    """Project tree: structures + runs with statuses."""

    structure_open_requested = Signal(str)   # structure file path
    run_open_requested = Signal(str)         # run POSCAR path
    run_status_change_requested = Signal(str, str)  # run id, status
    add_current_structure_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: Project | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._header = QLabel(self.tr("No project open"))
        layout.addWidget(self._header)

        self._tree = QTreeWidget(self)
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(
            [self.tr("Item"), self.tr("Status / Type")])
        self._tree.setRootIsDecorated(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.header().setStretchLastSection(True)
        self._tree.itemDoubleClicked.connect(self._on_double_clicked)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._tree, 1)

        self._hint = QLabel(self.tr(
            "Open a project (File → Open Project) or let the AI create "
            "runs — new runs appear here in real time."))
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        self.refresh()

    # -- public ----------------------------------------------------------

    def set_project(self, project: Project | None) -> None:
        self._project = project
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the tree from the current project manifest."""
        self._tree.clear()
        if self._project is None:
            self._header.setText(self.tr("No project open"))
            self._hint.setVisible(True)
            return
        self._hint.setVisible(False)
        self._header.setText(self._project.name)

        structures_root = QTreeWidgetItem(
            [self.tr("Structures"), ""])
        for entry in self._project.structures:
            item = QTreeWidgetItem(
                [entry["id"], entry.get("formula", "")])
            item.setData(0, Qt.ItemDataRole.UserRole, "structure")
            item.setData(1, Qt.ItemDataRole.UserRole, entry["id"])
            structures_root.addChild(item)
        self._tree.addTopLevelItem(structures_root)
        structures_root.setExpanded(True)

        runs_root = QTreeWidgetItem([self.tr("Runs"), ""])
        for entry in self._project.runs:
            item = QTreeWidgetItem(
                [entry["id"], _status_text(entry.get("status", ""))])
            item.setData(0, Qt.ItemDataRole.UserRole, "run")
            item.setData(1, Qt.ItemDataRole.UserRole, entry["id"])
            runs_root.addChild(item)
        self._tree.addTopLevelItem(runs_root)
        runs_root.setExpanded(True)

    # -- interactions ----------------------------------------------------

    def _selected(self) -> QTreeWidgetItem | None:
        items = self._tree.selectedItems()
        return items[0] if items else None

    def _on_double_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        if self._project is None:
            return
        kind = item.data(0, Qt.ItemDataRole.UserRole)
        ident = item.data(1, Qt.ItemDataRole.UserRole)
        if kind == "structure":
            try:
                self.structure_open_requested.emit(
                    str(self._project.structure_path(ident)))
            except ValueError:
                pass  # missing file — status message via main window
        elif kind == "run":
            poscar = self._project.run_dir(ident) / "POSCAR"
            if poscar.exists():
                self.run_open_requested.emit(str(poscar))

    def _on_context_menu(self, pos) -> None:
        if self._project is None:
            return
        item = self._tree.itemAt(pos)
        if item is None:
            return
        kind = item.data(0, Qt.ItemDataRole.UserRole)
        ident = item.data(1, Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        if kind == "run":
            entry = self._project.run_entry(ident)
            status_menu = menu.addMenu(self.tr("Set Status"))
            for status in RUN_STATUSES:
                act: QAction = status_menu.addAction(
                    _status_text(status))
                act.setCheckable(True)
                act.setChecked(entry.get("status") == status)
                act.triggered.connect(
                    lambda checked=False, s=status, i=ident:
                    self.run_status_change_requested.emit(i, s))
        elif kind == "structure":
            act_open = menu.addAction(self.tr("Open"))
            act_open.triggered.connect(
                lambda: self._on_double_clicked(item, 0))
            menu.addSeparator()
            act_add = menu.addAction(
                self.tr("Add Current Structure Here"))
            act_add.triggered.connect(
                self.add_current_structure_requested.emit)
        else:
            # group headers
            if kind is None and item.text(0) == self.tr("Structures"):
                act_add = menu.addAction(
                    self.tr("Add Current Structure"))
                act_add.triggered.connect(
                    self.add_current_structure_requested.emit)
        if not menu.isEmpty():
            menu.exec(self._tree.viewport().mapToGlobal(pos))

    # -- i18n --------------------------------------------------------------

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.LanguageChange:
            self._retranslate()
        super().changeEvent(event)

    def _retranslate(self) -> None:
        """Re-apply every tr() string (persistent panel, §11.2)."""
        self._header.setText(
            self._project.name if self._project is not None
            else self.tr("No project open"))
        self._tree.setHeaderLabels(
            [self.tr("Item"), self.tr("Status / Type")])
        self._hint.setText(self.tr(
            "Open a project (File → Open Project) or let the AI create "
            "runs — new runs appear here in real time."))
        self.refresh()


def _status_text(status: str) -> str:
    """Status labels stay in their canonical English form — they are
    manifest values (AI-facing identifiers), not UI prose."""
    return status
