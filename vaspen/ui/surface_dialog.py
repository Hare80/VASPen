"""Surface / slab cutting dialog.

Allows the user to specify Miller indices, number of layers,
and vacuum thickness, then previews the resulting slab.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.structure import StructureModel
from vaspen.core.surface import SurfaceCutter


class SurfaceDialog(QDialog):
    """Dialog for cutting a surface/slab from a bulk structure."""

    def __init__(self, structure_model: StructureModel, parent=None) -> None:
        super().__init__(parent)
        self._model = structure_model
        self.result_structure: StructureModel | None = None

        self.setWindowTitle(self.tr("Cut Surface / Slab"))
        self.resize(450, 300)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Miller indices ──
        miller_group = QGroupBox(self.tr("Miller Indices"))
        miller_layout = QHBoxLayout()

        miller_layout.addWidget(QLabel("h:"))
        self._h_spin = QSpinBox()
        self._h_spin.setRange(-5, 5)
        self._h_spin.setValue(1)
        miller_layout.addWidget(self._h_spin)

        miller_layout.addWidget(QLabel("k:"))
        self._k_spin = QSpinBox()
        self._k_spin.setRange(-5, 5)
        self._k_spin.setValue(1)
        miller_layout.addWidget(self._k_spin)

        miller_layout.addWidget(QLabel("l:"))
        self._l_spin = QSpinBox()
        self._l_spin.setRange(-5, 5)
        self._l_spin.setValue(1)
        miller_layout.addWidget(self._l_spin)

        miller_group.setLayout(miller_layout)
        layout.addWidget(miller_group)

        # ── Parameters ──
        params_layout = QFormLayout()

        self._layers_spin = QSpinBox()
        self._layers_spin.setRange(1, 100)
        self._layers_spin.setValue(4)
        self._layers_spin.setToolTip(self.tr(
            "Number of atomic layers in the slab.\n"
            "More layers = thicker slab, more computational cost."
        ))
        params_layout.addRow(self.tr("Layers:"), self._layers_spin)

        self._vacuum_spin = QDoubleSpinBox()
        self._vacuum_spin.setRange(0.0, 100.0)
        self._vacuum_spin.setValue(15.0)
        self._vacuum_spin.setSuffix(" Å")
        self._vacuum_spin.setToolTip(self.tr(
            "Vacuum thickness added above the slab.\n"
            "Standard: 10–15 Å for surface calculations."
        ))
        params_layout.addRow(self.tr("Vacuum:"), self._vacuum_spin)

        layout.addLayout(params_layout)

        # ── Info ──
        info_text = QLabel(self.tr(
            "The slab will be generated from the current bulk structure.\n"
            "Ensure your structure is a bulk crystal (periodic in all directions)."
        ))
        info_text.setWordWrap(True)
        layout.addWidget(info_text)

        # ── Preview info (placeholder) ──
        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        # ── Buttons ──
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        if self._model.n_atoms == 0:
            QMessageBox.warning(
                self,
                self.tr("No Structure"),
                self.tr("Load a bulk structure before cutting a surface."),
            )
            return

        h = self._h_spin.value()
        k = self._k_spin.value()
        l = self._l_spin.value()
        layers = self._layers_spin.value()
        vacuum = self._vacuum_spin.value()

        if h == 0 and k == 0 and l == 0:
            QMessageBox.warning(
                self,
                self.tr("Invalid Miller Indices"),
                self.tr("At least one Miller index must be non-zero."),
            )
            return

        try:
            cutter = SurfaceCutter(self._model)
            self.result_structure = cutter.cut(
                miller=(h, k, l),
                layers=layers,
                vacuum=vacuum,
            )
            QMessageBox.information(
                self,
                self.tr("Slab Created"),
                self.tr("Surface slab ({hkl}) created:\n"
                        "  {n} atoms, {layers} layers, {vacuum} Å vacuum").format(
                    hkl=f"({h}{k}{l})",
                    n=self.result_structure.n_atoms,
                    layers=layers,
                    vacuum=vacuum,
                ),
            )
            self.accept()
        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("Surface Cut Failed"),
                self.tr("Could not create slab:\n{}").format(str(e)),
            )
