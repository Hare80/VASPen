"""Atom properties panel (right dock) — edit the selected atom.

Shows element (editable text with autocomplete + a "…" periodic-table
picker), stable ID, Cartesian x/y/z and fractional coordinates (editable
text fields, periodic structures only), plus charge/force/velocity when
the file carries them (read-only). Coordinate and element fields are
plain text inputs — no spin-box arrows; invalid input reverts to the
model value.

Also shows the fixed (frozen) state — VASP selective dynamics: a master
"Fixed" checkbox (all three directions) plus per-axis X/Y/Z boxes
(periodic structures only). A frozen atom's coordinate fields are
read-only; the element field stays editable.

Non-modal/persistent → implements changeEvent + refresh so it
re-translates live (CLAUDE.md §11.2 pattern).
"""

from __future__ import annotations

import numpy as np
from ase.data import chemical_symbols

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QCompleter,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QToolButton,
    QWidget,
)

from vaspen.core.builder import StructureBuilder
from vaspen.core.structure import StructureModel
from vaspen.ui.periodic_table_dialog import PeriodicTableDialog
from vaspen.ui.structure_tree import composition_text

# Element symbols for the autocompleter (skip placeholder "X").
_ELEMENT_SYMBOLS = chemical_symbols[1:]


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

        # Element: editable text + autocomplete + "…" periodic table.
        self._element_edit = QLineEdit(self)
        completer = QCompleter(_ELEMENT_SYMBOLS, self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self._element_edit.setCompleter(completer)
        self._element_edit.editingFinished.connect(self._on_element_edited)
        self._element_pick_btn = QToolButton(self)
        self._element_pick_btn.setText("…")
        self._element_pick_btn.setToolTip(self.tr("Open periodic table…"))
        self._element_pick_btn.clicked.connect(self._open_periodic_table)
        element_row = QWidget(self)
        element_layout = QHBoxLayout(element_row)
        element_layout.setContentsMargins(0, 0, 0, 0)
        element_layout.setSpacing(2)
        element_layout.addWidget(self._element_edit, 1)
        element_layout.addWidget(self._element_pick_btn)
        self._element_label = QLabel(self.tr("Element:"))
        layout.addRow(self._element_label, element_row)

        self._id_label = QLabel("—")
        self._id_caption = QLabel(self.tr("ID:"))
        layout.addRow(self._id_caption, self._id_label)

        self._x_edit = self._make_coord_edit()
        self._y_edit = self._make_coord_edit()
        self._z_edit = self._make_coord_edit()
        layout.addRow(QLabel("x (Å):"), self._x_edit)
        layout.addRow(QLabel("y (Å):"), self._y_edit)
        layout.addRow(QLabel("z (Å):"), self._z_edit)

        self._frac_caption = QLabel(self.tr("Fractional (periodic only)"))
        layout.addRow(self._frac_caption)
        self._fx_edit = self._make_coord_edit()
        self._fy_edit = self._make_coord_edit()
        self._fz_edit = self._make_coord_edit()
        layout.addRow(QLabel("fx:"), self._fx_edit)
        layout.addRow(QLabel("fy:"), self._fy_edit)
        layout.addRow(QLabel("fz:"), self._fz_edit)

        # Fixed (frozen) state — VASP selective dynamics. The master
        # checkbox freezes/unfreezes all three directions; the per-axis
        # boxes adjust individual directions (periodic structures only —
        # the axes are the cell-vector directions).
        self._fixed_check = QCheckBox(self.tr("Fixed"))
        self._fixed_check.toggled.connect(self._on_fixed_toggled)
        self._fixed_caption = QLabel(self.tr("Fixed (selective dynamics)"))
        layout.addRow(self._fixed_caption, self._fixed_check)
        self._fx_fixed = QCheckBox("X")
        self._fy_fixed = QCheckBox("Y")
        self._fz_fixed = QCheckBox("Z")
        for axis, chk in enumerate((self._fx_fixed, self._fy_fixed, self._fz_fixed)):
            chk.toggled.connect(
                lambda checked, ax=axis: self._on_direction_toggled(ax, checked))
        dir_row = QWidget(self)
        dir_layout = QHBoxLayout(dir_row)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        dir_layout.setSpacing(4)
        for chk in (self._fx_fixed, self._fy_fixed, self._fz_fixed):
            dir_layout.addWidget(chk)
        dir_layout.addStretch()
        layout.addRow("", dir_row)

        # Initial magnetic moments — VASP MAGMOM (collinear, one value
        # per atom). Checking prefills 1.0 (VASP's default when the tag
        # is absent); unchecking unsets. Works for multi-selections
        # (mixed state shows as partially checked).
        self._magmom_check = QCheckBox(self.tr("MAGMOM"))
        self._magmom_check.toggled.connect(self._on_magmom_toggled)
        self._magmom_caption = QLabel(self.tr("MAGMOM (initial magnetic moments)"))
        self._magmom_caption.setToolTip(self.tr(
            "Set initial magnetic moments per atom (VASP MAGMOM); "
            "unset atoms are written as 0.0"))
        layout.addRow(self._magmom_caption, self._magmom_check)
        self._magmom_edit = QLineEdit()
        self._magmom_edit.setToolTip(self.tr(
            "Initial magnetic moment in μB (e.g. 5 or -5)"))
        self._magmom_edit.editingFinished.connect(self._on_magmom_edited)
        magmom_row = QWidget(self)
        magmom_layout = QHBoxLayout(magmom_row)
        magmom_layout.setContentsMargins(0, 0, 0, 0)
        magmom_layout.setSpacing(2)
        magmom_layout.addWidget(self._magmom_edit, 1)
        magmom_layout.addWidget(QLabel("μB"))
        layout.addRow("", magmom_row)

        # Site composition (partially-occupied structures only; hidden
        # otherwise — see refresh()).
        self._composition_label = QLabel("")
        self._composition_label.setWordWrap(True)
        self._composition_caption = QLabel(self.tr("Composition"))
        layout.addRow(self._composition_caption, self._composition_label)
        self._composition_caption.setVisible(False)
        self._composition_label.setVisible(False)

        self._charge_label = QLabel("")
        self._charge_caption = QLabel(self.tr("Charge:"))
        layout.addRow(self._charge_caption, self._charge_label)
        self._force_label = QLabel("")
        self._force_caption = QLabel(self.tr("Force:"))
        layout.addRow(self._force_caption, self._force_label)
        self._velocity_label = QLabel("")
        self._velocity_caption = QLabel(self.tr("Velocity:"))
        layout.addRow(self._velocity_caption, self._velocity_label)

        # Connect edit commits ONCE — programmatic setText never fires
        # editingFinished, so no connect/disconnect churn is needed
        # (repeated disconnect+reconnect leaked duplicate connections).
        for edit in (self._x_edit, self._y_edit, self._z_edit):
            edit.editingFinished.connect(self._on_cartesian_edited)
        for edit in (self._fx_edit, self._fy_edit, self._fz_edit):
            edit.editingFinished.connect(self._on_fractional_edited)

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
    def _make_coord_edit() -> QLineEdit:
        """Plain text coordinate field — no spin-box arrows (user
        decision 2026-08-14: values are typed, not stepped)."""
        return QLineEdit()

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
        self._element_pick_btn.setToolTip(self.tr("Open periodic table…"))
        self._id_caption.setText(self.tr("ID:"))
        self._frac_caption.setText(self.tr("Fractional (periodic only)"))
        self._fixed_check.setText(self.tr("Fixed"))
        self._fixed_caption.setText(self.tr("Fixed (selective dynamics)"))
        self._magmom_check.setText(self.tr("MAGMOM"))
        self._magmom_caption.setText(self.tr("MAGMOM (initial magnetic moments)"))
        self._magmom_caption.setToolTip(self.tr(
            "Set initial magnetic moments per atom (VASP MAGMOM); "
            "unset atoms are written as 0.0"))
        self._magmom_edit.setToolTip(self.tr(
            "Initial magnetic moment in μB (e.g. 5 or -5)"))
        self._composition_caption.setText(self.tr("Composition"))
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
                # 1-based display to match the structure tree's Index
                # column (internal indices stay 0-based — see model).
                self._status_label.setText(
                    self.tr("Atom {} ({})").format(index + 1, symbol))

            for w in (self._element_edit, self._element_pick_btn, self._id_label,
                      self._x_edit, self._y_edit, self._z_edit,
                      self._fx_edit, self._fy_edit, self._fz_edit,
                      self._fixed_check, self._fx_fixed, self._fy_fixed,
                      self._fz_fixed,
                      self._charge_label, self._force_label, self._velocity_label):
                w.setEnabled(single)
            # MAGMOM works for multi-selections too (batch assignment)
            for w in (self._magmom_check, self._magmom_edit):
                w.setEnabled(bool(sel))

            # MAGMOM state: checked = all selected set, unchecked = none,
            # partially checked = mixed. The value box shows the common
            # value (empty when mixed or differing).
            magmoms = [model.magmom(i) for i in sel]
            set_all = bool(magmoms) and all(m is not None for m in magmoms)
            # Mixed selections show as unchecked; checking then fills
            # only the unset atoms (set values survive).
            self._magmom_check.setChecked(bool(sel) and set_all)
            values = {m for m in magmoms if m is not None}
            if set_all and len(values) == 1:
                self._magmom_edit.setText(f"{next(iter(values)):g}")
            else:
                self._magmom_edit.setText("")

            self._charge_label.setText("")
            self._force_label.setText("")
            self._velocity_label.setText("")

            # Site composition — shown only for a single selected atom
            # whose site carries partial occupancy (mixed species or a
            # vacancy); hidden otherwise.
            comp_text = ""
            if single:
                index = next(iter(sel))
                occ = model.occupancy
                if occ is not None and index < len(occ):
                    site = occ[index]
                    if len(site) > 1 or (
                            site and next(iter(site.values())) < 1.0):
                        pairs = sorted(
                            site.items(), key=lambda kv: (-kv[1], kv[0]))
                        comp_text = composition_text(pairs, 1, self.tr)
            self._composition_label.setText(comp_text)
            self._composition_caption.setVisible(bool(comp_text))
            self._composition_label.setVisible(bool(comp_text))

            if single:
                index = next(iter(sel))
                self._element_edit.setText(model.symbols[index])
                # 1-based display (stable ID + 1); the 0-based ID itself
                # is internal (measurements reference it).
                self._id_label.setText(str(model.atom_id(index) + 1))
                pos = model.positions[index]
                self._x_edit.setText(f"{pos[0]:.6f}")
                self._y_edit.setText(f"{pos[1]:.6f}")
                self._z_edit.setText(f"{pos[2]:.6f}")

                # Fixed state: master checkbox + per-axis direction boxes.
                # Direction boxes are periodic-only (they are cell-vector
                # directions); a frozen atom's coordinate fields are
                # read-only (element replacement stays available).
                flags = model.fixed_flags[index]
                self._fixed_check.setChecked(bool(flags.any()))
                for chk, v in zip((self._fx_fixed, self._fy_fixed, self._fz_fixed),
                                  flags):
                    chk.setChecked(bool(v))
                rank3 = bool(model.atoms.get_cell().rank == 3)
                for chk in (self._fx_fixed, self._fy_fixed, self._fz_fixed):
                    chk.setEnabled(single and rank3)
                frozen = model.is_fixed(index)
                for edit in (self._x_edit, self._y_edit, self._z_edit):
                    edit.setEnabled(single and not frozen)

                periodic = model.is_periodic
                for edit in (self._fx_edit, self._fy_edit, self._fz_edit):
                    edit.setEnabled(single and periodic and not frozen)
                if periodic:
                    frac = model.scaled_positions[index]
                    self._fx_edit.setText(f"{frac[0]:.6f}")
                    self._fy_edit.setText(f"{frac[1]:.6f}")
                    self._fz_edit.setText(f"{frac[2]:.6f}")
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

    def _revert_edits(self) -> None:
        """Restore field texts from the model (invalid input guard)."""
        self.refresh()

    def _open_periodic_table(self) -> None:
        """Open the periodic table; clicking an element applies it."""
        index = self._single_index()
        if index is None:
            return
        dlg = PeriodicTableDialog(self)
        if dlg.exec() != PeriodicTableDialog.DialogCode.Accepted:
            return
        if dlg.selected_symbol is None:
            return
        self._element_edit.setText(dlg.selected_symbol)
        self._on_element_edited()

    def _on_element_edited(self) -> None:
        if self._refreshing or self._model is None:
            return
        index = self._single_index()
        if index is None:
            return
        symbol = self._element_edit.text().strip().capitalize()
        if symbol not in _ELEMENT_SYMBOLS:
            self._revert_edits()
            return
        if symbol != self._model.symbols[index]:
            StructureBuilder.replace_element(self._model, index, symbol)

    def _on_fixed_toggled(self, checked: bool) -> None:
        if self._refreshing or self._model is None:
            return
        index = self._single_index()
        if index is None:
            return
        try:
            self._model.set_fixed(index, bool(checked))
        except ValueError:
            self._revert_edits()

    def _on_direction_toggled(self, axis: int, checked: bool) -> None:
        if self._refreshing or self._model is None:
            return
        index = self._single_index()
        if index is None:
            return
        mask = self._model.fixed_flags[index]
        mask[axis] = bool(checked)
        try:
            self._model.set_fixed(index, mask)
        except ValueError:
            self._revert_edits()

    def _on_magmom_toggled(self, checked: bool) -> None:
        if self._refreshing or self._model is None:
            return
        sel = self._model.selected_indices
        if not sel:
            return
        try:
            if checked:
                # Prefill 1.0 only for currently unset atoms — values of
                # already-set atoms (mixed selection) survive.
                values = [
                    self._model.magmom(i) if self._model.magmom(i) is not None
                    else 1.0 for i in sel
                ]
                self._model.set_magmom(list(sel), values)
                self._magmom_edit.setFocus()
                self._magmom_edit.selectAll()  # typing replaces the prefill
            else:
                self._model.set_magmom(list(sel), None)
        except ValueError:
            self._revert_edits()

    def _on_magmom_edited(self) -> None:
        if self._refreshing or self._model is None:
            return
        sel = self._model.selected_indices
        if not sel:
            return
        try:
            value = float(self._magmom_edit.text().strip())
        except ValueError:
            self._revert_edits()
            return
        try:
            self._model.set_magmom(list(sel), value)
        except ValueError:
            self._revert_edits()

    def _on_cartesian_edited(self) -> None:
        if self._refreshing or self._model is None:
            return
        index = self._single_index()
        if index is None:
            return
        try:
            pos = [float(self._x_edit.text()), float(self._y_edit.text()),
                   float(self._z_edit.text())]
        except ValueError:
            self._revert_edits()
            return
        try:
            self._model.set_atom_position(index, np.asarray(pos, dtype=float))
        except ValueError:  # movement guard (frozen atom) — backstop
            self._revert_edits()

    def _on_fractional_edited(self) -> None:
        if self._refreshing or self._model is None:
            return
        index = self._single_index()
        if index is None or not self._model.is_periodic:
            return
        try:
            frac = [float(self._fx_edit.text()), float(self._fy_edit.text()),
                    float(self._fz_edit.text())]
        except ValueError:
            self._revert_edits()
            return
        try:
            self._model.set_atom_scaled_position(index, np.asarray(frac, dtype=float))
        except ValueError:  # movement guard (frozen atom) — backstop
            self._revert_edits()
