"""Unit-cell parameter editor with live 3D preview."""

from __future__ import annotations

import numpy as np
from ase.geometry import cellpar_to_cell
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.structure import StructureModel


class LatticeDialog(QDialog):
    """Edit a/b/c/α/β/γ with a live preview in the 3D viewport.

    Modal and created fresh per invocation, so it picks up the current
    UI language automatically (no retranslate needed). Every value
    change renders a WORKING COPY of the structure into the viewport
    (the model is untouched); Accept applies the parameters to the
    model as ONE undo step, Cancel restores the viewport to the model
    state.
    """

    def __init__(
        self,
        model: StructureModel,
        viewport,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._viewport = viewport
        self.setWindowTitle(self.tr("Edit Lattice"))
        self.setModal(True)
        self._build_ui()
        self._refresh_preview()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._len_spins: list[QDoubleSpinBox] = []
        self._ang_spins: list[QDoubleSpinBox] = []
        params = [
            *[(name, " Å", 0.1, 1000.0, float(self._model.cell_lengths[i]))
              for i, name in enumerate("abc")],
            *[(name, "°", 5.0, 175.0, float(self._model.cell_angles[i]))
              for i, name in enumerate("αβγ")],
        ]
        for name, suffix, low, high, current in params:
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
            spin.setSuffix(suffix)
            spin.setValue(current)
            spin.valueChanged.connect(self._on_value_changed)
            form.addRow(name + ":", spin)
            (self._len_spins if suffix == " Å" else self._ang_spins).append(spin)
        layout.addLayout(form)

        self._scale_check = QCheckBox(
            self.tr("Scale atom positions (keep fractional coordinates)"))
        self._scale_check.setChecked(True)
        self._scale_check.toggled.connect(self._on_value_changed)
        if self._model.any_fixed:
            # Scaling atoms moves frozen atoms (fractional coords kept) —
            # force the Cartesian-preserving mode and lock the option.
            self._scale_check.setChecked(False)
            self._scale_check.setEnabled(False)
        layout.addWidget(self._scale_check)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        # Theme-aware via QLabel[hintKind="error"] in the app QSS.
        self._hint.setProperty("hintKind", "error")
        layout.addWidget(self._hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._ok_button = buttons.button(QDialogButtonBox.Ok)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _parameters(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        lengths = tuple(float(s.value()) for s in self._len_spins)
        angles = tuple(float(s.value()) for s in self._ang_spins)
        return lengths, angles

    def _on_value_changed(self, *_args) -> None:
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        """Render a working copy with the current parameters.

        The copy keeps the model's bond list so the preview matches the
        edited structure's styling; the highlight is replayed because
        set_structure resets it.
        """
        lengths, angles = self._parameters()
        try:
            cell = cellpar_to_cell([*lengths, *angles])
            # relative criterion: a volume below 1 ppm of the axis-aligned
            # box means the parameters are (numerically) degenerate
            valid = float(np.linalg.det(cell)) > 1e-6 * float(np.prod(lengths))
        except (ValueError, RuntimeError):
            valid = False
        if not valid:
            self._hint.setText(
                self.tr("These parameters do not form a valid unit cell."))
            self._ok_button.setEnabled(False)
            return
        self._hint.setText("")
        self._ok_button.setEnabled(True)
        working = self._model.atoms.copy()
        working.set_cell(cell, scale_atoms=self._scale_check.isChecked())
        self._viewport.set_structure(working, reset_view=False,
                                     bonds=self._model.bonds,
                                     fixed=self._model.fixed_flags)
        self._viewport.set_highlight(self._model.selected_indices)

    # ------------------------------------------------------------------
    # Accept / reject
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        if not self._ok_button.isEnabled():
            return
        lengths, angles = self._parameters()
        try:
            self._model.set_cell_parameters(
                lengths, angles, scale_atoms=self._scale_check.isChecked())
        except ValueError:
            # Movement guard (frozen atoms) — the scale option is already
            # locked for such models; this is the backstop.
            QMessageBox.warning(
                self,
                self.tr("Edit Lattice"),
                self.tr("Cannot scale atoms: some atoms are frozen."),
            )
            return
        self.accept()

    def reject(self) -> None:
        """Restore the viewport to the real model state before closing."""
        self._viewport.set_structure(self._model.atoms, reset_view=False,
                                     bonds=self._model.bonds,
                                     fixed=self._model.fixed_flags)
        self._viewport.set_highlight(self._model.selected_indices)
        super().reject()
