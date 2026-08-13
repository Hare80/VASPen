"""Symmetry analysis dialog: space-group info + optional symmetrization."""

from __future__ import annotations

from ase import Atoms
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.structure import StructureModel
from vaspen.core.symmetry import analyze, symmetrize


class SymmetryDialog(QDialog):
    """Show the space group of the current structure.

    For periodic structures a "Symmetrize" button standardizes the cell
    and positions (conventional cell, spglib); the result is exposed as
    ``result_atoms`` after a successful exec. Non-periodic structures
    only show an explanatory message. Modal, created fresh per
    invocation (current language, no retranslate needed).
    """

    def __init__(self, model: StructureModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self.result_atoms: Atoms | None = None
        self.setWindowTitle(self.tr("Symmetry"))
        self.setModal(True)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info = analyze(self._model.atoms)
        if info is None:
            note = QLabel(self.tr(
                "Symmetry analysis requires a periodic structure "
                "(a full-rank cell with periodic boundary conditions)."))
            note.setWordWrap(True)
            layout.addWidget(note)
        elif info.kind == "space":
            form = QFormLayout()
            form.addRow(self.tr("Space group number:"),
                        QLabel(str(info.number)))
            form.addRow(self.tr("International symbol:"),
                        QLabel(info.international))
            form.addRow(self.tr("Hall symbol:"), QLabel(info.hall))
            form.addRow(self.tr("Point group:"), QLabel(info.pointgroup))
            if info.choice:
                form.addRow(self.tr("Setting:"), QLabel(info.choice))
            layout.addLayout(form)
        else:  # point group of an isolated system
            form = QFormLayout()
            form.addRow(self.tr("Point group (Schoenflies):"),
                        QLabel(info.international))
            form.addRow(self.tr("Rotational symmetry number:"),
                        QLabel(str(info.number)))
            layout.addLayout(form)

        if info is not None:
            self._symmetrize_btn = QPushButton(self.tr("&Symmetrize"))
            self._symmetrize_btn.setToolTip(self.tr(
                "Standardize the cell and positions to the conventional cell"))
            self._symmetrize_btn.clicked.connect(self._on_symmetrize)
            layout.addWidget(self._symmetrize_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_symmetrize(self) -> None:
        result = symmetrize(self._model.atoms)
        if result is None:
            QMessageBox.warning(
                self, self.tr("Symmetry"),
                self.tr("Could not symmetrize the structure."))
            return
        self.result_atoms = result
        self.accept()
