"""Structure tree — atom list + cell parameters panel (left dock).

Shows cell parameters and an editable atom table. Selection is synced
both ways with the StructureModel / 3D viewport: clicking a row selects
the atom (and highlights it in 3D), and selecting in 3D highlights the
row here. Positions are editable by double-click; atoms can be added or
deleted via the context menu.
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.builder import StructureBuilder
from vaspen.core.structure import StructureModel
from vaspen.ui.viewport3d import element_color
from vaspen.utils.logger import logger


class StructureTreePanel(QWidget):
    """Left-dock panel: cell parameters + atom table."""

    def __init__(self, model: StructureModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._refreshing = False  # guards feedback loops during refresh

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # ── Cell parameters ──
        self._cell_label = QLabel(self.tr("No structure loaded."))
        self._cell_label.setWordWrap(True)
        layout.addWidget(self._cell_label)

        # ── Atom table ──
        self._table = QTableWidget(0, 5)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_context_menu)
        self._table.cellChanged.connect(self._on_cell_changed)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table, 1)

        self.set_model(model)

    # ------------------------------------------------------------------
    # Model wiring
    # ------------------------------------------------------------------

    def set_model(self, model: StructureModel) -> None:
        """Connect to a new StructureModel (models are replaced on New)."""
        if self._model is not None and self._model is not model:
            try:
                self._model.structure_loaded.disconnect(self.refresh)
                self._model.structure_modified.disconnect(self.refresh)
                self._model.atom_selected.disconnect(self._on_model_selection)
                self._model.selection_cleared.disconnect(self._on_selection_cleared)
            except (TypeError, RuntimeError):
                pass  # not connected to this model
        self._model = model
        model.structure_loaded.connect(self.refresh)
        model.structure_modified.connect(self.refresh)
        model.atom_selected.connect(self._on_model_selection)
        model.selection_cleared.connect(self._on_selection_cleared)
        self.refresh()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def changeEvent(self, event: QEvent) -> None:
        """Re-apply translatable texts when the language changes."""
        if event.type() == QEvent.Type.LanguageChange:
            self.refresh()
        super().changeEvent(event)

    def refresh(self) -> None:
        """Rebuild the cell label and atom table from the model."""
        model = self._model
        self._refreshing = True
        try:
            self._table.setHorizontalHeaderLabels(
                [self.tr("Index"), self.tr("Element"), "x (Å)", "y (Å)", "z (Å)"]
            )

            # Cell parameters
            if model.n_atoms == 0:
                self._cell_label.setText(self.tr("No structure loaded."))
            else:
                a, b, c = model.cell_lengths
                alpha, beta, gamma = model.cell_angles
                self._cell_label.setText(
                    self.tr("Formula: {}\na = {:.3f} Å   b = {:.3f} Å   c = {:.3f} Å\n"
                            "α = {:.2f}°   β = {:.2f}°   γ = {:.2f}°").format(
                        model.chemical_formula, a, b, c, alpha, beta, gamma
                    )
                )

            # Atom table
            symbols = model.symbols
            positions = model.positions
            self._table.setRowCount(0)
            self._table.setRowCount(len(symbols))
            for i, sym in enumerate(symbols):
                idx_item = QTableWidgetItem(str(i + 1))
                idx_item.setFlags(idx_item.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(i, 0, idx_item)

                el_item = QTableWidgetItem(sym)
                el_item.setFlags(el_item.flags() & ~Qt.ItemIsEditable)
                base = element_color(sym)
                el_item.setForeground(QColor(
                    int(base[0] * 255), int(base[1] * 255), int(base[2] * 255)
                ))
                self._table.setItem(i, 1, el_item)

                for j in range(3):
                    val_item = QTableWidgetItem(f"{positions[i, j]:.6f}")
                    self._table.setItem(i, 2 + j, val_item)

            # Restore selection highlight
            if model.selected_index is not None:
                self._table.selectRow(model.selected_index)
        finally:
            self._refreshing = False

    # ------------------------------------------------------------------
    # Selection sync (table → model)
    # ------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        if self._refreshing:
            return
        row = self._table.currentRow()
        if 0 <= row < self._model.n_atoms and row != self._model.selected_index:
            self._model.select_atom(row)

    def _on_model_selection(self, index: int) -> None:
        if not self._refreshing and 0 <= index < self._table.rowCount():
            if self._table.currentRow() != index:
                self._refreshing = True
                try:
                    self._table.selectRow(index)
                finally:
                    self._refreshing = False

    def _on_selection_cleared(self) -> None:
        if not self._refreshing:
            self._refreshing = True
            try:
                self._table.clearSelection()
            finally:
                self._refreshing = False

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def _on_cell_changed(self, row: int, col: int) -> None:
        """Apply an edited x/y/z value (col 2..4) to the model."""
        if self._refreshing or col < 2:
            return
        item = self._table.item(row, col)
        if item is None:
            return
        try:
            new_value = float(item.text())
        except ValueError:
            QMessageBox.warning(
                self,
                self.tr("Invalid Position"),
                self.tr("Position must be a number."),
            )
            self.refresh()  # revert the bad text
            return
        pos = self._model.positions[row].copy()
        pos[col - 2] = new_value
        self._model.set_atom_position(row, pos)
        logger.debug("Atom %d moved to %s", row, pos)

    def _on_context_menu(self, pos) -> None:
        row = self._table.rowAt(pos.y())
        menu = QMenu(self)

        add_action = menu.addAction(self.tr("Add Atom..."))
        delete_action = menu.addAction(self.tr("Delete Atom"))
        delete_action.setEnabled(0 <= row < self._model.n_atoms)

        chosen = menu.exec(self._table.mapToGlobal(pos))
        if chosen is add_action:
            self._add_atom()
        elif chosen is delete_action:
            self._model.delete_atom(row)

    def _add_atom(self) -> None:
        """Prompt for 'Element x y z' and append the atom."""
        text, ok = QInputDialog.getText(
            self,
            self.tr("Add Atom"),
            self.tr("Element and Cartesian position (x y z), e.g. Fe 1.0 2.0 3.0:"),
        )
        if not ok or not text.strip():
            return
        parts = text.split()
        if len(parts) != 4:
            QMessageBox.warning(
                self,
                self.tr("Invalid Input"),
                self.tr("Expected format: Element x y z"),
            )
            return
        try:
            symbol = parts[0].capitalize()
            position = np.array([float(x) for x in parts[1:4]])
            StructureBuilder.add_atom(self._model, symbol, position)
        except (ValueError, TypeError) as e:
            QMessageBox.warning(self, self.tr("Invalid Input"), str(e))
