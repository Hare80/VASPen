"""Project panel — structures/runs tree for a calculation project.

Left dock showing the open project (§7.15): a Structures group and a
Runs group with status labels. Double-click opens things into the main
window; a context menu on a run changes its status or deletes the run
(manifest entry + directory). Runs are CREATED by the Generate All
dialog ("Current project (new run)" destination) and by the MCP
``create_run``/``neb_setup`` tools — this panel manages and consumes.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QUrl, Qt, Signal
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
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
    run_parameters_requested = Signal(str)           # run id
    run_deleted = Signal(str)                        # run id
    file_open_requested = Signal(str)                # text file path
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
            item.setToolTip(0, self._run_tooltip(entry["id"]))
            for rel in self._project.run_files(entry["id"]):
                child = QTreeWidgetItem(
                    [str(rel), _file_kind_text(rel)])
                child.setData(0, Qt.ItemDataRole.UserRole, "run-file")
                child.setData(1, Qt.ItemDataRole.UserRole, str(rel))
                child.setToolTip(0, str(self._project.run_dir(entry["id"])
                                        / rel))
                item.addChild(child)
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
            # the run ROW is the calculation object — its editor opens
            # here (MS convention); files open via the file children
            self.run_parameters_requested.emit(ident)
        elif kind == "run-file":
            rel = item.data(1, Qt.ItemDataRole.UserRole)
            run_id = item.parent().data(1, Qt.ItemDataRole.UserRole)
            path = self._project.run_dir(run_id) / rel
            # text first: run.json's .json suffix would otherwise
            # classify as a (non-ASE) structure and hang the opener
            if _is_text_file(path):
                self.file_open_requested.emit(str(path))
            elif _is_structure_file(path):
                self.structure_open_requested.emit(str(path))
            else:
                QDesktopServices.openUrl(
                    QUrl.fromLocalFile(str(self._project.run_dir(run_id))))

    def _on_context_menu(self, pos) -> None:
        if self._project is None:
            return
        item = self._tree.itemAt(pos)
        if item is None:
            return
        kind = item.data(0, Qt.ItemDataRole.UserRole)
        ident = item.data(1, Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        if kind == "run-file":
            rel = item.data(1, Qt.ItemDataRole.UserRole)
            run_id = item.parent().data(1, Qt.ItemDataRole.UserRole)
            path = self._project.run_dir(run_id) / rel
            act_open = menu.addAction(self.tr("Open"))
            act_open.triggered.connect(
                lambda: self._on_double_clicked(item, 0))
            act_reveal = menu.addAction(self.tr("Show in Explorer"))
            act_reveal.triggered.connect(
                lambda: QDesktopServices.openUrl(
                    QUrl.fromLocalFile(str(path.parent))))
        elif kind == "run":
            act_params = menu.addAction(self.tr("Edit Inputs…"))
            act_params.triggered.connect(
                lambda: self.run_parameters_requested.emit(ident))
            menu.addSeparator()
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
            act_del = menu.addAction(self.tr("Delete Run..."))
            act_del.triggered.connect(
                lambda checked=False, i=ident: self._on_delete_run(i))
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

    def _on_delete_run(self, run_id: str) -> None:
        """Confirm, then remove the run (manifest entry + directory)."""
        if self._project is None:
            return
        entry = self._project.run_entry(run_id)
        run_dir = self._project.run_dir(run_id)
        if entry.get("status") in ("submitted", "running"):
            QMessageBox.warning(
                self, self.tr("Delete Run"),
                self.tr("Run {} is {} — refusing to delete it. Set its "
                        "status to inputs-ready/done/failed first.")
                .format(run_id, entry["status"]))
            return
        message = self.tr("Delete run {} and its whole directory?\n\n{}"
                          ).format(run_id, run_dir)
        if ((run_dir / "CONTCAR").exists()
                or (run_dir / "OSZICAR").exists()):
            message += "\n\n" + self.tr(
                "The run already contains calculation outputs — they "
                "will be deleted with it.")
        reply = QMessageBox.question(self, self.tr("Delete Run"), message)
        if reply != QMessageBox.Yes:
            return
        try:
            self._project.remove_run(run_id)
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, self.tr("Delete Run"), str(e))
            return
        self.refresh()
        self.run_deleted.emit(run_id)

    def _run_tooltip(self, run_id: str) -> str:
        """One-line parameter summary from the run's provenance card."""
        if self._project is None:
            return ""
        params = self._project.run_parameters(run_id)
        parts = [f"{key}: {value}" for key, value in params.items()
                 if key in ("preset", "functional", "kpoints_mode")]
        return " · ".join(parts)

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


from vaspen.core.file_io import resolve_format  # noqa: E402


def _is_structure_file(path: Path) -> bool:
    """Structure-like files open in the 3D viewport."""
    return resolve_format(str(path)) is not None


_TEXT_NAMES = ("INCAR", "KPOINTS", "OSZICAR", "OUTCAR", "run.json",
               "EIGENVAL", "DOSCAR", "IBZKPT", "XDATCAR", "PROCAR",
               "POTCAR")


def _is_text_file(path: Path) -> bool:
    """Known VASP text files (and .json/.txt) open in the viewer."""
    if path.suffix.lower() in (".json", ".txt"):
        return True
    return path.name.upper() in _TEXT_NAMES


def _file_kind_text(rel: Path) -> str:
    """Second-column hint for a run file row (text check FIRST — see
    the double-click branch comment)."""
    if _is_text_file(rel):
        return "text"
    if _is_structure_file(rel):
        return "structure"
    return "binary"
