"""Measurement manager + panel for the 3D viewport (distances, angles,
dihedrals).

Measurements reference atoms by their STABLE model IDs (not indices),
so they survive atom deletion, undo/redo and any index shifting:
``sync_with_model`` re-resolves the IDs on every structure change and
drops orphans (atoms that were deleted).
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vaspen.core import measure
from vaspen.core.structure import StructureModel


@dataclass(frozen=True)
class Measurement:
    """One measurement: kind + the stable IDs of its atoms."""

    kind: str  # "distance" | "angle" | "dihedral"
    atom_ids: tuple[int, ...]


class MeasurementManager(QObject):
    """Owns the measurement list; resolves IDs against a StructureModel.

    changed is emitted whenever the list or its resolved state changes.
    """

    changed = Signal()

    def __init__(self, model: StructureModel, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._items: list[Measurement] = []
        self._resolved: list[tuple[Measurement, list[int]]] = []

    def add(self, kind: str, indices) -> None:
        """Add a measurement for atom indices (resolved to IDs now)."""
        ids = tuple(self._model.atom_id(i) for i in indices)
        if any(i is None for i in ids):
            return
        self._items.append(Measurement(kind, ids))
        self.sync_with_model()

    def set_model(self, model: StructureModel) -> None:
        """Re-bind to a new model (models are replaced on New)."""
        self._model = model
        self.sync_with_model()

    def remove(self, index: int) -> None:
        """Remove the measurement at list position ``index``."""
        if 0 <= index < len(self._items):
            del self._items[index]
            self.sync_with_model()

    def clear(self) -> None:
        """Remove all measurements."""
        self._items.clear()
        self.sync_with_model()

    def sync_with_model(self) -> None:
        """Re-resolve IDs against the model; drop orphans; recompute."""
        resolved: list[tuple[Measurement, list[int]]] = []
        kept: list[Measurement] = []
        for item in self._items:
            idx = [self._model.index_of_id(i) for i in item.atom_ids]
            if any(i is None for i in idx):
                continue  # atom deleted — drop the measurement
            kept.append(item)
            resolved.append((item, [i for i in idx if i is not None]))
        self._items = kept
        self._resolved = resolved
        self.changed.emit()

    def payload(self) -> list[tuple[str, list[int], str]]:
        """Display payload for the viewport: (kind, indices, value text)."""
        atoms = self._model.atoms
        out = []
        for item, idx in self._resolved:
            if item.kind == "distance":
                value = measure.distance(atoms, idx[0], idx[1])
                text = f"{value:.3f} Å"
            elif item.kind == "angle":
                value = measure.angle(atoms, idx[0], idx[1], idx[2])
                text = f"{value:.2f}°"
            else:
                value = measure.dihedral(atoms, idx[0], idx[1], idx[2], idx[3])
                text = f"{value:.2f}°"
            out.append((item.kind, idx, text))
        return out

    def __len__(self) -> int:
        return len(self._items)


class MeasurementPanel(QWidget):
    """Right-dock panel listing measurements with a Clear All button.

    Non-modal/persistent → implements changeEvent + refresh so it
    re-translates live (CLAUDE.md §11.2 pattern).
    """

    def __init__(self, manager: MeasurementManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._manager = manager

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self._label = QLabel(self.tr("Measurements"))
        layout.addWidget(self._label)

        self._list = QListWidget(self)
        layout.addWidget(self._list, 1)

        button_row = QHBoxLayout()
        self._clear_btn = QPushButton(self.tr("Clear All"))
        self._clear_btn.clicked.connect(self._manager.clear)
        button_row.addStretch(1)
        button_row.addWidget(self._clear_btn)
        layout.addLayout(button_row)

        self._manager.changed.connect(self.refresh)
        self.refresh()

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.LanguageChange:
            self._retranslate()
        super().changeEvent(event)

    def _retranslate(self) -> None:
        self._label.setText(self.tr("Measurements"))
        self._clear_btn.setText(self.tr("Clear All"))
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the list from the manager's current payload."""
        kind_labels = {
            "distance": self.tr("Distance"),
            "angle": self.tr("Angle"),
            "dihedral": self.tr("Dihedral"),
        }
        self._list.clear()
        for kind, _idx, text in self._manager.payload():
            label = kind_labels.get(kind, kind)
            self._list.addItem(f"{label}: {text}")
