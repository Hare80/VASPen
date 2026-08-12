"""StructureModel — wraps ASE Atoms with Qt signal support.

This is the central data model for the application. All structural
information flows through this class, and changes are broadcast via
Qt signals so that every observer (3D viewport, structure tree,
status bar) stays in sync.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from PySide6.QtCore import QObject, Signal


class StructureModel(QObject):
    """Observable wrapper around an ASE Atoms object.

    Signals:
        structure_loaded: Emitted after a structure is loaded from disk.
        structure_modified: Emitted after any structural change (add/remove/
            move atoms, change cell, etc.).
        atom_selected(int): Emitted when the user selects an atom by index.
        selection_cleared: Emitted when the selection is cleared.
    """

    structure_loaded = Signal()
    structure_modified = Signal()
    atom_selected = Signal(int)
    selection_cleared = Signal()

    def __init__(self, atoms: Atoms | None = None, parent: QObject | None = None) -> None:
        """Initialize the model.

        Args:
            atoms: Optional ASE Atoms object. If None, creates an empty cell.
            parent: Qt parent object.
        """
        super().__init__(parent)
        self._atoms: Atoms = atoms if atoms is not None else Atoms()
        self._filepath: str | None = None
        self._selected_index: int | None = None
        self._dirty: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def atoms(self) -> Atoms:
        """Return the underlying ASE Atoms object (read-only)."""
        return self._atoms

    @property
    def filepath(self) -> str | None:
        """Path to the currently loaded file, or None if unsaved."""
        return self._filepath

    @property
    def is_dirty(self) -> bool:
        """True if the structure has unsaved changes."""
        return self._dirty

    @property
    def selected_index(self) -> int | None:
        """Index of the currently selected atom, or None."""
        return self._selected_index

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def load(self, filepath: str | Path) -> None:
        """Load a structure from file (cif, xyz, POSCAR, CONTCAR, etc.).

        ASE auto-detects the format from the file extension and content.
        """
        path = Path(filepath)
        atoms = ase_read(str(path))
        self._atoms = atoms
        self._filepath = str(path)
        self._dirty = False
        self._selected_index = None
        self.structure_loaded.emit()

    def save(self, filepath: str | Path | None = None, fmt: str | None = None) -> None:
        """Save the current structure to a file.

        Args:
            filepath: Destination path. Uses the loaded path if None.
            fmt: ASE format string (e.g. 'vasp', 'cif', 'xyz').
                 Auto-detected from extension if None.
        """
        path = Path(filepath) if filepath else Path(self._filepath) if self._filepath else None
        if path is None:
            raise ValueError("No filepath specified and no file loaded.")

        ase_write(str(path), self._atoms, format=fmt)
        self._filepath = str(path)
        self._dirty = False

    def to_ase_atoms(self) -> Atoms:
        """Return a copy of the ASE Atoms object."""
        return self._atoms.copy()

    # ------------------------------------------------------------------
    # Structure queries
    # ------------------------------------------------------------------

    @property
    def n_atoms(self) -> int:
        return len(self._atoms)

    @property
    def chemical_formula(self) -> str:
        return self._atoms.get_chemical_formula()

    @property
    def cell(self) -> np.ndarray:
        """3x3 cell matrix."""
        return self._atoms.get_cell().array

    @property
    def cell_lengths(self) -> np.ndarray:
        """a, b, c lengths in Angstrom."""
        return self._atoms.get_cell().lengths()

    @property
    def cell_angles(self) -> np.ndarray:
        """α, β, γ angles in degrees."""
        return self._atoms.get_cell().angles()

    @property
    def positions(self) -> np.ndarray:
        """Cartesian positions (N×3)."""
        return self._atoms.get_positions()

    @property
    def scaled_positions(self) -> np.ndarray:
        """Fractional / direct coordinates (N×3)."""
        return self._atoms.get_scaled_positions()

    @property
    def atomic_numbers(self) -> np.ndarray:
        return self._atoms.get_atomic_numbers()

    @property
    def symbols(self) -> list[str]:
        return list(self._atoms.get_chemical_symbols())

    @property
    def unique_symbols(self) -> list[str]:
        """Unique element symbols, sorted."""
        return sorted(set(self.symbols), key=lambda s: self.symbols.index(s))

    @property
    def pbc(self) -> tuple[bool, bool, bool]:
        """Periodic boundary conditions."""
        return tuple(self._atoms.get_pbc())

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_atom(self, index: int) -> None:
        """Select an atom by index."""
        if 0 <= index < len(self._atoms):
            self._selected_index = index
            self.atom_selected.emit(index)

    def clear_selection(self) -> None:
        """Clear the current atom selection."""
        self._selected_index = None
        self.selection_cleared.emit()

    # ------------------------------------------------------------------
    # Structure mutation
    # ------------------------------------------------------------------

    def replace_atoms(self, new_atoms: Atoms) -> None:
        """Replace the entire structure with a new Atoms object."""
        self._atoms = new_atoms
        self._dirty = True
        self._selected_index = None
        self.structure_modified.emit()

    def set_cell(self, cell: np.ndarray) -> None:
        """Set the unit cell (3x3 matrix)."""
        self._atoms.set_cell(cell)
        self._dirty = True
        self.structure_modified.emit()

    def set_positions(self, positions: np.ndarray) -> None:
        """Set Cartesian positions (N×3)."""
        self._atoms.set_positions(positions)
        self._dirty = True
        self.structure_modified.emit()

    def set_scaled_positions(self, scaled: np.ndarray) -> None:
        """Set fractional coordinates (N×3)."""
        self._atoms.set_scaled_positions(scaled)
        self._dirty = True
        self.structure_modified.emit()

    def extend_atoms(self, other: Atoms | StructureModel) -> None:
        """Append atoms from another Atoms or StructureModel."""
        if isinstance(other, StructureModel):
            other = other._atoms
        self._atoms.extend(other)
        self._dirty = True
        self.structure_modified.emit()

    def delete_atom(self, index: int) -> None:
        """Delete the atom at the given index."""
        if 0 <= index < len(self._atoms):
            del self._atoms[index]
            self._dirty = True
            if self._selected_index == index:
                self._selected_index = None
            self.structure_modified.emit()

    def translate_atom(self, index: int, vector: np.ndarray) -> None:
        """Translate a single atom by a Cartesian vector."""
        if 0 <= index < len(self._atoms):
            self._atoms[index].position += np.asarray(vector)
            self._dirty = True
            self.structure_modified.emit()

    def __repr__(self) -> str:
        return (
            f"StructureModel(formula={self.chemical_formula}, "
            f"n_atoms={self.n_atoms}, file={self._filepath})"
        )
