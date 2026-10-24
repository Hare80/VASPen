"""Project structure picker — modal chooser over a project's
registered structures (§7.15 P2).

Used by the NEB endpoint rows in the Generate All dialog: the user
picks one of the project's structures/ entries and the caller loads it
from disk (File → Open Project semantics do not apply — the file is
read, never adopted). Lists id, formula metadata (when present) and
the registration date; double-click or Enter accepts, Esc cancels.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)


class StructurePickerDialog(QDialog):
    """Pick one of the open project's registered structures."""

    def __init__(self, project, parent=None) -> None:
        super().__init__(parent)
        self._project = project
        self.setWindowTitle(self.tr("Choose Project Structure"))
        self.resize(440, 380)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(self.tr("Registered structures:")))
        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(lambda _item: self.accept())
        layout.addWidget(self._list, 1)
        self._empty_label = QLabel(self.tr(
            "(No structures registered — add some to the project first.)"))
        layout.addWidget(self._empty_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._buttons = buttons

        self._fill()

    # ------------------------------------------------------------------

    def _fill(self) -> None:
        structures = self._project.structures
        for entry in structures:
            formula = entry.get("formula", "")
            text = entry["id"] + (f"  ({formula})" if formula else "")
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, entry["id"])
            item.setToolTip(str(self._project.root / entry["file"]))
            self._list.addItem(item)
        has_structures = bool(structures)
        self._list.setVisible(has_structures)
        self._empty_label.setVisible(not has_structures)
        if has_structures:
            self._list.setCurrentRow(0)
        self._buttons.button(
            QDialogButtonBox.StandardButton.Ok).setEnabled(has_structures)

    def selected_structure_id(self) -> str | None:
        """Id of the picked structure (None when nothing selected)."""
        item = self._list.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def selected_structure_path(self):
        """Absolute path of the picked structure (None when nothing is
        selected — the caller then treats the dialog as cancelled)."""
        structure_id = self.selected_structure_id()
        if structure_id is None:
            return None
        return self._project.structure_path(structure_id)
