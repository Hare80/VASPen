"""Vacuum-padding dialog shown before wrapping a molecule into a periodic cell."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from vaspen.utils.config import AppConfig


class PeriodicWrapDialog(QDialog):
    """Ask for vacuum padding before converting a molecule to a periodic cell.

    Modal and created fresh per invocation, so it picks up the current UI
    language automatically (no retranslate needed). The chosen padding is
    exposed as ``padding`` after a successful exec; whether it becomes the
    default next time is controlled by the "remember" checkbox.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = AppConfig()
        self.padding: float = 5.0
        self.setWindowTitle(self.tr("Wrap in Periodic Cell"))
        self.setModal(True)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info = QLabel(self.tr(
            "The structure is not periodic. VASPen will create a periodic cell "
            "around it with vacuum padding on each side."
        ))
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()
        self._padding_spin = QDoubleSpinBox()
        self._padding_spin.setRange(0.5, 50.0)
        self._padding_spin.setDecimals(2)
        self._padding_spin.setSingleStep(0.5)
        self._padding_spin.setSuffix(" Å")
        default = self._config.wrap_padding if self._config.remember_wrap_padding else 5.0
        self._padding_spin.setValue(min(max(default, 0.5), 50.0))
        form.addRow(self.tr("Vacuum padding:"), self._padding_spin)
        layout.addLayout(form)

        self._remember_check = QCheckBox(self.tr("Remember this value"))
        self._remember_check.setChecked(self._config.remember_wrap_padding)
        layout.addWidget(self._remember_check)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        self.padding = float(self._padding_spin.value())
        self._config.wrap_padding = self.padding
        self._config.remember_wrap_padding = self._remember_check.isChecked()
        self.accept()
