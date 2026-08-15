"""Re-box Slab dialog: re-apply vacuum to a structure that carries it.

The standalone counterpart of Cleave Surface: for periodic structures
that already have vacuum (e.g. cut surfaces), it unwraps layers split
across the periodic boundary and re-applies the requested vacuum along
c with the slab centered. The in-plane cell is untouched and the atom
order is unchanged — fixed flags and magnetic moments map 1:1.

Modal, created fresh per invocation (current language, no retranslate
needed).
"""

from __future__ import annotations

from ase import Atoms
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from vaspen.core.structure import StructureModel
from vaspen.core.surface import rebox_slab
from vaspen.ui.tools import confirm_disorder_loss
from vaspen.utils.config import AppConfig


class ReBoxDialog(QDialog):
    """Choose the vacuum for re-boxing the current slab-like structure."""

    def __init__(self, model: StructureModel,
                 parent: QDialog | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._config = AppConfig()
        self.result_atoms: Atoms | None = None
        self.setWindowTitle(self.tr("Re-box Slab"))
        self.setModal(True)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        note = QLabel(self.tr(
            "For periodic structures that already carry vacuum "
            "(e.g. cut surfaces): layers split across the periodic "
            "boundary are unwrapped, the vacuum along c is re-applied "
            "and the slab is centered. The in-plane cell stays "
            "unchanged.\n\n"
            "For BULK structures, use Calculate → Cleave Surface."))
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        self._vacuum_spin = QDoubleSpinBox()
        self._vacuum_spin.setRange(0.5, 100.0)
        # same remember semantics as the periodic-wrap dialog: the value
        # is always saved; the checkbox controls whether it becomes the
        # default next time
        default = self._config.rebox_vacuum if self._config.remember_rebox else 15.0
        self._vacuum_spin.setValue(min(max(default, 0.5), 100.0))
        self._vacuum_spin.setDecimals(1)
        self._vacuum_spin.setSuffix(" Å")
        form.addRow(self.tr("Vacuum:"), self._vacuum_spin)

        self._remember_check = QCheckBox(self.tr("Remember this value"))
        self._remember_check.setChecked(self._config.remember_rebox)
        form.addRow("", self._remember_check)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        vacuum = float(self._vacuum_spin.value())
        self._config.rebox_vacuum = vacuum
        self._config.remember_rebox = self._remember_check.isChecked()
        # Re-boxing rebuilds the box from scratch — fractional occupancy
        # (disorder) cannot survive it (same policy as cleave/symmetrize).
        if self._model.has_disorder and not confirm_disorder_loss(
                self, self.tr("Re-boxing")):
            return
        self.result_atoms = rebox_slab(self._model.atoms, vacuum)
        self.accept()
