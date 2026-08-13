"""Periodic table dialog — click an element to select it.

Materials-Studio-style element picker: the standard 18-column periodic
table with the lanthanides/actinides on two extra rows. Clicking an
element accepts the dialog with ``selected_symbol`` set.

Modal and created fresh per invocation, so it picks up the current UI
language automatically (no retranslate needed).
"""

from __future__ import annotations

from ase.data import atomic_names, chemical_symbols

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.render_settings import PALETTES

# Standard periodic table arrangement: (Z_first, Z_last, row, column_first)
# of each contiguous run. Lanthanides/actinides live on rows 8/9 below
# the main table (columns 3–17).
_LAYOUT_SEGMENTS = (
    (1, 1, 1, 1),       # H
    (2, 2, 1, 18),      # He
    (3, 4, 2, 1),       # Li–Be
    (5, 10, 2, 13),     # B–Ne
    (11, 12, 3, 1),     # Na–Mg
    (13, 18, 3, 13),    # Al–Ar
    (19, 20, 4, 1),     # K–Ca
    (21, 30, 4, 3),     # Sc–Zn
    (31, 36, 4, 13),    # Ga–Kr
    (37, 38, 5, 1),     # Rb–Sr
    (39, 48, 5, 3),     # Y–Cd
    (49, 54, 5, 13),    # In–Xe
    (55, 56, 6, 1),     # Cs–Ba
    (57, 71, 8, 3),     # La–Lu
    (72, 86, 6, 4),     # Hf–Rn
    (87, 88, 7, 1),     # Fr–Ra
    (89, 103, 9, 3),    # Ac–Lr
    (104, 118, 7, 4),   # Rf–Og
)

# (symbol → (row, column)) resolved once at import.
_ELEMENT_POSITIONS: dict[str, tuple[int, int]] = {}
for _z0, _z1, _row, _col0 in _LAYOUT_SEGMENTS:
    for _z in range(_z0, _z1 + 1):
        _ELEMENT_POSITIONS[chemical_symbols[_z]] = (_row, _col0 + (_z - _z0))


def element_position(symbol: str) -> tuple[int, int] | None:
    """Grid position (row, column) of an element, or None."""
    return _ELEMENT_POSITIONS.get(symbol)


class PeriodicTableDialog(QDialog):
    """Modal element picker — click an element, then exec() returns Accepted."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.selected_symbol: str | None = None
        self.setWindowTitle(self.tr("Periodic Table"))
        self.setModal(True)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        grid_widget = QWidget(self)
        grid = QGridLayout(grid_widget)
        grid.setSpacing(2)
        block_colors = PALETTES["block"]
        self._element_buttons: dict[str, QToolButton] = {}
        for z in range(1, 119):
            sym = chemical_symbols[z]
            btn = QToolButton(grid_widget)
            btn.setText(sym)
            btn.setToolTip(f"{z} — {atomic_names[z]}")
            btn.setFixedSize(44, 32)
            r, g, b = block_colors[sym]
            btn.setStyleSheet(
                f"QToolButton {{ background-color: "
                f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)}); "
                f"border: 1px solid #888; }}")
            btn.clicked.connect(lambda checked=False, s=sym: self._on_element(s))
            row, col = _ELEMENT_POSITIONS[sym]
            grid.addWidget(btn, row, col)
            self._element_buttons[sym] = btn
        layout.addWidget(grid_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_element(self, symbol: str) -> None:
        self.selected_symbol = symbol
        self.accept()
