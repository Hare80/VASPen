"""Numeric transform dialog: translate / rotate / align."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QMessageBox,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.structure import StructureModel
from vaspen.core.transform import (
    rotate,
    rotation_matrix,
    translate,
    wrap_periodic,
)

_DIRECTIONS = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
    "-X": (-1.0, 0.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "-Z": (0.0, 0.0, -1.0),
}


class TransformDialog(QDialog):
    """Apply translate / rotate numerically, in that order.

    Modal and created fresh per invocation (picks up the current
    language). The transforms act on a working copy; on Accept the
    result is exposed as ``result_atoms`` and the caller applies it to
    the model as ONE undo step.

    Whole-structure transforms of a PERIODIC structure rotate the unit
    cell together with the atoms (the crystal stays intact — bonds do
    not tear); transforms of a selection only move those atoms inside
    the fixed cell.
    """

    def __init__(self, model: StructureModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self.result_atoms: Atoms | None = None
        self.setWindowTitle(self.tr("Transform"))
        self.setModal(True)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Scope ──
        scope_box = QGroupBox(self.tr("Apply to"))
        scope_layout = QVBoxLayout(scope_box)
        n_total = len(self._model.atoms)
        n_sel = len(self._model.selected_indices)
        self._scope_all = QRadioButton(self.tr("Whole structure ({})").format(n_total))
        self._scope_sel = QRadioButton(self.tr("Selected atoms ({})").format(n_sel))
        self._scope_all.setChecked(n_sel == 0)
        self._scope_sel.setChecked(n_sel > 0)
        self._scope_sel.setEnabled(n_sel > 0)
        scope_layout.addWidget(self._scope_all)
        scope_layout.addWidget(self._scope_sel)
        layout.addWidget(scope_box)

        # ── Translate ──
        trans_box = QGroupBox(self.tr("Translate"))
        form = QFormLayout(trans_box)
        self._dx, self._dy, self._dz = (self._make_spin(-100.0, 100.0, " Å")
                                        for _ in range(3))
        for name, spin in zip("xyz", (self._dx, self._dy, self._dz)):
            form.addRow(name.upper() + ":", spin)
        layout.addWidget(trans_box)

        # ── Rotate ──
        rot_box = QGroupBox(self.tr("Rotate"))
        form = QFormLayout(rot_box)
        self._axis_combo = QComboBox()
        self._axis_combo.addItems(["X", "Y", "Z"])
        form.addRow(self.tr("Axis:"), self._axis_combo)
        self._angle = self._make_spin(-180.0, 180.0, "°")
        self._angle.setSingleStep(5.0)
        form.addRow(self.tr("Angle:"), self._angle)
        self._center_check = QCheckBox(
            self.tr("Rotate about the center of the moved atoms"))
        self._center_check.setChecked(True)
        form.addRow("", self._center_check)
        layout.addWidget(rot_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _make_spin(low: float, high: float, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(3)
        spin.setSingleStep(0.1)
        spin.setSuffix(suffix)
        return spin

    def _scope_indices(self) -> list[int] | None:
        if self._scope_sel.isChecked():
            indices = sorted(self._model.selected_indices)
            if not indices:
                QMessageBox.information(
                    self, self.tr("Transform"), self.tr("No atoms selected."))
                return None
            return indices
        return None

    def _on_accept(self) -> None:
        indices = self._scope_indices()
        if indices is None and self._scope_sel.isChecked():
            return  # message already shown
        working = self._model.atoms.copy()
        # Whole-structure transform of a periodic structure: the unit
        # cell rotates together with the atoms. Wrapping is postponed
        # until AFTER the cell rotation — wrapping into the old cell
        # first would scramble fractional positions whenever the
        # rotation is not a lattice symmetry.
        whole_periodic = indices is None and self._model.is_periodic
        wrap = not whole_periodic
        # Accumulated rotation (rotate + align contributions).
        r_total = np.eye(3)
        try:
            delta = np.array([self._dx.value(), self._dy.value(),
                              self._dz.value()], dtype=float)
            if np.any(delta):
                working = translate(working, delta, indices, wrap=wrap)
            angle = self._angle.value()
            if angle:
                axis = _DIRECTIONS[self._axis_combo.currentText()]
                center = None if self._center_check.isChecked() else (0.0, 0.0, 0.0)
                working = rotate(working, (0.0, 0.0, 0.0), axis, angle,
                                 indices, center, wrap=wrap)
                r_total = rotation_matrix(axis, angle) @ r_total
        except ValueError as e:
            QMessageBox.warning(self, self.tr("Transform"), str(e))
            return
        if whole_periodic and not np.allclose(r_total, np.eye(3), atol=1e-12):
            # whole periodic structure: the cell rotates with the atoms
            working.set_cell(working.get_cell().array @ r_total.T)
        if whole_periodic:
            # now wrap into the FINAL cell (old if only translated,
            # rotated otherwise) — fractional positions stay intact
            working = wrap_periodic(working)
        self.result_atoms = working
        self.accept()
