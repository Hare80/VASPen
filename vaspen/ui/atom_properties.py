"""Atom properties panel (right dock) — edit the selected atom.

Shows element (editable), stable ID, Cartesian x/y/z and fractional
coordinates (editable, periodic structures only), plus charge/force/
velocity when the file carries them (read-only).

Non-modal/persistent → implements changeEvent + refresh so it
re-translates live (CLAUDE.md §11.2 pattern).
"""

from __future__ import annotations

import numpy as np
from ase.data import chemical_symbols

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QWidget,
)

from vaspen.core.builder import StructureBuilder
from vaspen.core.structure import StructureModel


class AtomPropertiesPanel(QWidget):
    """Right-dock panel editing the currently selected atom."""

    def __init__(self, model: StructureModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model: StructureModel | None = None
        self._refreshing = False

        layout = QFormLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self._status_label = QLabel(self.tr("No atom selected"))
        self._status_label.setWordWrap(True)
        layout.addRow(self._status_label)

        self._element_combo = QComboBox(self)
        self._element_combo.addItems(chemical_symbols[1:])  # skip "X"
        self._element_combo.currentTextChanged.connect(self._on_element_changed)
        self._element_label = QLabel(self.tr("Element:"))
        layout.addRow(self._element_label, self._element_combo)

        self._id_label = QLabel("—")
        self._id_caption = QLabel(self.tr("ID:"))
        layout.addRow(self._id_caption, self._id_label)

        self._x_spin = self._make_spin()
        self._y_spin = self._make_spin()
        self._z_spin = self._make_spin()
        layout.addRow(QLabel("x (Å):"), self._x_spin)
        layout.addRow(QLabel("y (Å):"), self._y_spin)
        layout.addRow(QLabel("z (Å):"), self._z_spin)

        self._frac_caption = QLabel(self.tr("Fractional (periodic only)"))
        layout.addRow(self._frac_caption)
        self._fx_spin = self._make_spin()
        self._fy_spin = self._make_spin()
        self._fz_spin = self._make_spin()
        layout.addRow(QLabel("fx:"), self._fx_spin)
        layout.addRow(QLabel("fy:"), self._fy_spin)
        layout.addRow(QLabel("fz:"), self._fz_spin)

        self._charge_label = QLabel("")
        self._charge_caption = QLabel(self.tr("Charge:"))
        layout.addRow(self._charge_caption, self._charge_label)
        self._force_label = QLabel("")
        self._force_caption = QLabel(self.tr("Force:"))
        layout.addRow(self._force_caption, self._force_label)
        self._velocity_label = QLabel("")
        self._velocity_caption = QLabel(self.tr("Velocity:"))
        layout.addRow(self._velocity_caption, self._velocity_label)

        # Connect edit commits ONCE — programmatic setValue never fires
        # editingFinished, so no connect/disconnect churn is needed
        # (repeated disconnect+reconnect leaked duplicate connections).
        for spin in (self._x_spin, self._y_spin, self._z_spin):
            spin.editingFinished.connect(self._on_cartesian_edited)
        for spin in (self._fx_spin, self._fy_spin, self._fz_spin):
            spin.editingFinished.connect(self._on_fractional_edited)

        self.set_model(model)

    # ------------------------------------------------------------------
    # Model wiring
    # ------------------------------------------------------------------

    def set_model(self, model: StructureModel) -> None:
        """Re-bind to a new model (models are replaced on New)."""
        if self._model is not None and self._model is not model:
            for sig in (self._model.selection_changed, self._model.atom_selected,
                        self._model.selection_cleared, self._model.structure_modified,
                        self._model.structure_loaded):
                try:
                    sig.disconnect(self.refresh)
                except (TypeError, RuntimeError):
                    pass  # not connected to this model
        self._model = model
        for sig in (model.selection_changed, model.atom_selected,
                    model.selection_cleared, model.structure_modified,
                    model.structure_loaded):
            sig.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------
    # Widgets
    # ------------------------------------------------------------------

    @staticmethod
    def _make_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1e6, 1e6)
        spin.setDecimals(6)
        return spin

    @staticmethod
    def _property_value(atoms, key: str, index: int) -> np.ndarray | None:
        """Per-atom value from atoms.arrays or a SinglePointCalculator
        (ASE's extxyz reader stores forces in a calculator)."""
        if key in atoms.arrays:
            return np.asarray(atoms.arrays[key][index])
        calc = getattr(atoms, "calc", None)
        results = getattr(calc, "results", None)
        if results and key in results:
            return np.asarray(results[key][index])
        return None

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.LanguageChange:
            self._retranslate()
        super().changeEvent(event)

    def _retranslate(self) -> None:
        self._element_label.setText(self.tr("Element:"))
        self._id_caption.setText(self.tr("ID:"))
        self._frac_caption.setText(self.tr("Fractional (periodic only)"))
        self._charge_caption.setText(self.tr("Charge:"))
        self._force_caption.setText(self.tr("Force:"))
        self._velocity_caption.setText(self.tr("Velocity:"))
        self.refresh()

    def refresh(self) -> None:
        """Populate the panel from the current selection."""
        model = self._model
        if model is None:
            return
        sel = model.selected_indices
        self._refreshing = True
        try:
            single = len(sel) == 1
            if not sel:
                self._status_label.setText(self.tr("No atom selected"))
            elif not single:
                self._status_label.setText(self.tr("{} atoms selected").format(len(sel)))
            else:
                index = next(iter(sel))
                symbol = model.symbols[index]
                self._status_label.setText(self.tr("Atom {} ({})").format(index, symbol))

            for w in (self._element_combo, self._id_label,
                      self._x_spin, self._y_spin, self._z_spin,
                      self._fx_spin, self._fy_spin, self._fz_spin,
                      self._charge_label, self._force_label, self._velocity_label):
                w.setEnabled(single)

            self._charge_label.setText("")
            self._force_label.setText("")
            self._velocity_label.setText("")
            if single:
                index = next(iter(sel))
                self._element_combo.setCurrentText(model.symbols[index])
                self._id_label.setText(str(model.atom_id(index)))
                pos = model.positions[index]
                self._x_spin.setValue(float(pos[0]))
                self._y_spin.setValue(float(pos[1]))
                self._z_spin.setValue(float(pos[2]))
                periodic = model.is_periodic
                for spin in (self._fx_spin, self._fy_spin, self._fz_spin):
                    spin.setEnabled(single and periodic)
                if periodic:
                    frac = model.scaled_positions[index]
                    self._fx_spin.setValue(float(frac[0]))
                    self._fy_spin.setValue(float(frac[1]))
                    self._fz_spin.setValue(float(frac[2]))
                for key, label in (("initial_charges", self._charge_label),
                                   ("forces", self._force_label),
                                   ("velocities", self._velocity_label)):
                    value = self._property_value(model.atoms, key, index)
                    if value is not None:
                        label.setText(np.array2string(value, precision=4))
        finally:
            self._refreshing = False

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def _single_index(self) -> int | None:
        if self._model is None:
            return None
        sel = self._model.selected_indices
        return next(iter(sel)) if len(sel) == 1 else None

    def _on_element_changed(self, symbol: str) -> None:
        if self._refreshing or not symbol or self._model is None:
            return
        index = self._single_index()
        if index is None:
            return
        if symbol != self._model.symbols[index]:
            StructureBuilder.replace_element(self._model, index, symbol)

    def _on_cartesian_edited(self) -> None:
        if self._refreshing or self._model is None:
            return
        index = self._single_index()
        if index is None:
            return
        pos = [self._x_spin.value(), self._y_spin.value(), self._z_spin.value()]
        self._model.set_atom_position(index, np.asarray(pos, dtype=float))

    def _on_fractional_edited(self) -> None:
        if self._refreshing or self._model is None:
            return
        index = self._single_index()
        if index is None or not self._model.is_periodic:
            return
        frac = [self._fx_spin.value(), self._fy_spin.value(), self._fz_spin.value()]
        self._model.set_atom_scaled_position(index, np.asarray(frac, dtype=float))
