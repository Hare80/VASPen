"""Supercell repeat-factor dialog."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class SupercellDialog(QDialog):
    """Ask for n_a/n_b/n_c repeat factors before building a supercell.

    Modal and created fresh per invocation, so it picks up the current
    UI language automatically. The chosen factors are exposed as
    ``factors`` after a successful exec.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.factors: tuple[int, int, int] | None = None
        self.setWindowTitle(self.tr("Supercell"))
        self.setModal(True)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._spins: list[QSpinBox] = []
        for i, name in enumerate(("a", "b", "c")):
            spin = QSpinBox()
            spin.setRange(1, 20)
            spin.setValue(1)
            spin.setSuffix("×")
            form.addRow(self.tr("Repeat along {}:").format(name), spin)
            self._spins.append(spin)
        layout.addLayout(form)

        note = QLabel(self.tr(
            "Bonds are re-detected automatically after the supercell is created."))
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        self.factors = tuple(s.value() for s in self._spins)
        self.accept()
