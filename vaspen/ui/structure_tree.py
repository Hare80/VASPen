"""Structure tree — atom list + cell parameters panel (left dock).

Shows cell parameters and an editable atom table. Selection is synced
both ways with the StructureModel / 3D viewport: clicking a row selects
the atom (and highlights it in 3D), and selecting in 3D highlights the
row here. Positions are editable by double-click; atoms can be added or
deleted via the context menu.
"""

from __future__ import annotations

import numpy as np
from ase.data import chemical_symbols

from PySide6.QtCore import QEvent, QItemSelection, QItemSelectionModel, Qt
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
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.builder import StructureBuilder
from vaspen.core.structure import StructureModel
from vaspen.ui.viewport3d import element_text_color
from vaspen.utils.logger import logger


class StructureTreePanel(QWidget):
    """Left-dock panel: cell parameters + atom table."""

    def __init__(self, model: StructureModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._refreshing = False  # guards feedback loops during refresh
        self._frac_mode = False   # False = Cartesian, True = fractional

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # ── Cell parameters ──
        self._cell_label = QLabel(self.tr("No structure loaded."))
        self._cell_label.setWordWrap(True)
        layout.addWidget(self._cell_label)

        # ── Cartesian / fractional toggle (periodic structures only).
        # The label shows the CURRENT mode; clicking switches — no
        # checked/highlighted state (user preference). ──
        self._frac_btn = QToolButton(self)
        self._frac_btn.setText(self.tr("Cartesian"))
        self._frac_btn.setToolTip(
            self.tr("Click to switch between Cartesian and fractional coordinates"))
        self._frac_btn.clicked.connect(self._on_frac_clicked)
        layout.addWidget(self._frac_btn)

        # ── Atom table ──
        self._table = QTableWidget(0, 5)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
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
                self._model.selection_changed.disconnect(self._on_model_selection_changed)
            except (TypeError, RuntimeError):
                pass  # not connected to this model
        self._model = model
        model.structure_loaded.connect(self.refresh)
        model.structure_modified.connect(self.refresh)
        model.atom_selected.connect(self._on_model_selection)
        model.selection_cleared.connect(self._on_selection_cleared)
        model.selection_changed.connect(self._on_model_selection_changed)
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
            periodic = model.is_periodic
            self._frac_btn.setEnabled(periodic and model.n_atoms > 0)
            frac_mode = self._frac_mode and periodic
            if self._frac_mode and not periodic:
                self._frac_mode = False  # model became a molecule
            # The label shows the CURRENT coordinate mode.
            self._frac_btn.setText(
                self.tr("Fractional") if self._frac_mode else self.tr("Cartesian"))

            if frac_mode:
                self._table.setHorizontalHeaderLabels(
                    [self.tr("Index"), self.tr("Element"), "fx", "fy", "fz"]
                )
            else:
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
            values = model.scaled_positions if frac_mode else model.positions
            self._table.setRowCount(0)
            self._table.setRowCount(len(symbols))
            for i, sym in enumerate(symbols):
                idx_item = QTableWidgetItem(str(i + 1))
                idx_item.setFlags(idx_item.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(i, 0, idx_item)

                el_item = QTableWidgetItem(sym)
                # editable: double-click to change the element (validated
                # in _on_cell_changed); text color is darkened for
                # contrast on the light panel (white H would be invisible)
                base = element_text_color(sym)
                el_item.setForeground(QColor(
                    int(base[0] * 255), int(base[1] * 255), int(base[2] * 255)
                ))
                self._table.setItem(i, 1, el_item)

                for j in range(3):
                    val_item = QTableWidgetItem(f"{values[i, j]:.6f}")
                    self._table.setItem(i, 2 + j, val_item)

            # Restore selection highlight (multi-select aware)
            self._select_rows([
                r for r in model.selected_indices if 0 <= r < self._table.rowCount()
            ])
        finally:
            self._refreshing = False

    def _on_frac_clicked(self) -> None:
        """Cartesian ↔ fractional coordinate display/edit mode."""
        self._frac_mode = not self._frac_mode
        self.refresh()

    # ------------------------------------------------------------------
    # Selection sync (table → model)
    # ------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        if self._refreshing:
            return
        rows = {i.row() for i in self._table.selectedIndexes()}
        rows = {r for r in rows if 0 <= r < self._model.n_atoms}
        if rows != self._model.selected_indices:
            self._model.set_selection(rows)

    def _on_model_selection(self, index: int) -> None:
        if not self._refreshing and 0 <= index < self._table.rowCount():
            if self._table.currentRow() != index:
                self._refreshing = True
                try:
                    self._table.selectRow(index)
                finally:
                    self._refreshing = False

    def _select_rows(self, rows) -> None:
        """Replace the table selection with the given rows (one operation —
        calling selectRow() repeatedly would replace the selection each
        time, keeping only the last row in ExtendedSelection mode).

        The selection spans every column: selectedRows() only reports
        rows whose cells are ALL selected, and a programmatic select
        does not get the SelectRows expansion that mouse clicks do.
        """
        n_cols = self._table.columnCount()
        selection = QItemSelection()
        for row in rows:
            selection.select(
                self._table.model().index(row, 0),
                self._table.model().index(row, n_cols - 1),
            )
        self._table.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect)

    def _on_model_selection_changed(self) -> None:
        """Multi-select sync: mirror the model's selection set in the table."""
        if self._refreshing:
            return
        sel = self._model.selected_indices
        rows = {i.row() for i in self._table.selectedIndexes()}
        if rows == sel:
            return
        self._refreshing = True
        try:
            self._select_rows(
                [r for r in sorted(sel) if 0 <= r < self._table.rowCount()]
            )
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
        """Apply an edited Element (col 1) or x/y/z value (cols 2..4)."""
        if self._refreshing:
            return
        item = self._table.item(row, col)
        if item is None:
            return
        if col == 1:
            self._change_element(row, item)
            return
        if col < 2:
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
        if self._frac_mode and self._model.is_periodic:
            frac = self._model.scaled_positions[row].copy()
            frac[col - 2] = new_value
            self._model.set_atom_scaled_position(row, frac)
            logger.debug("Atom %d frac position set to %s", row, frac)
        else:
            pos = self._model.positions[row].copy()
            pos[col - 2] = new_value
            self._model.set_atom_position(row, pos)
            logger.debug("Atom %d moved to %s", row, pos)

    def _change_element(self, row: int, item) -> None:
        """Validate an edited element symbol and replace the atom."""
        symbol = item.text().strip().capitalize()
        if symbol not in chemical_symbols:
            QMessageBox.warning(
                self,
                self.tr("Invalid Element"),
                self.tr("Unknown element: {}").format(item.text()),
            )
            self.refresh()  # revert the bad text
            return
        try:
            StructureBuilder.replace_element(self._model, row, symbol)
        except Exception as e:  # noqa: BLE001 — mirror the xyz path
            QMessageBox.warning(self, self.tr("Invalid Element"), str(e))
            self.refresh()
        logger.debug("Atom %d replaced with %s", row, symbol)

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
