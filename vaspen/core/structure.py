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
from PySide6.QtCore import QObject, Signal

from vaspen.core.file_io import FileIO


def wrap_in_padded_cell(atoms: Atoms, padding: float) -> Atoms:
    """Return a copy of ``atoms`` inside an orthogonal padded periodic cell.

    The new cell is diagonal with each side length = bounding-box extent
    along that axis + 2*padding; atoms are shifted so the bounding-box
    center coincides with the cell center (guaranteeing exactly ``padding``
    of vacuum on every face); pbc is set to (True, True, True). The input
    Atoms object is not modified.

    Args:
        atoms: Molecule-like structure (any cell/pbc; only positions are used).
        padding: Vacuum padding on each side of the bounding box, in Angstrom.

    Returns:
        New Atoms with the padded cell, centered, pbc=(True, True, True).

    Raises:
        ValueError: If atoms is empty or padding is not a positive number.
    """
    if len(atoms) == 0:
        raise ValueError("Cannot wrap an empty structure.")
    if not np.isfinite(padding) or padding <= 0:
        raise ValueError(f"Padding must be a positive number, got {padding}.")

    positions = np.asarray(atoms.get_positions(), dtype=float)
    low = positions.min(axis=0)
    high = positions.max(axis=0)
    side_lengths = (high - low) + 2.0 * float(padding)

    wrapped = atoms.copy()
    wrapped.set_cell(np.diag(side_lengths))
    wrapped.set_pbc(True)
    # Shift so the bounding-box center coincides with the cell center.
    wrapped.translate(side_lengths / 2.0 - (low + high) / 2.0)
    return wrapped


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
        # Edit history (snapshot-based undo/redo)
        self._undo_stack: list[Atoms] = []
        self._redo_stack: list[Atoms] = []
        self._max_history = 50

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
        self._clear_history()
        self.structure_loaded.emit()

    def load_atoms(self, atoms: Atoms, filepath: str | Path | None = None) -> None:
        """Replace contents with the given Atoms and mark as loaded.

        Emits structure_loaded (not structure_modified) so observers
        rebuild the scene exactly once. Used by the UI when loading
        through FileIO.
        """
        self._atoms = atoms
        if filepath is not None:
            self._filepath = str(Path(filepath))
        self._dirty = False
        self._selected_index = None
        self._clear_history()
        self.structure_loaded.emit()

    def save(self, filepath: str | Path | None = None, fmt: str | None = None) -> None:
        """Save the current structure to a file.

        Args:
            filepath: Destination path. Uses the loaded path if None.
            fmt: ASE format string (e.g. 'vasp', 'cif', 'xyz').
                 Auto-detected from extension if None.

        Raises:
            ValueError: If no filepath is available, the extension is not
                supported, or a non-periodic structure is saved to a
                periodic format (vasp/cif).
        """
        path = Path(filepath) if filepath else Path(self._filepath) if self._filepath else None
        if path is None:
            raise ValueError("No filepath specified and no file loaded.")

        FileIO.write(str(path), self._atoms, fmt=fmt)
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

    @property
    def is_periodic(self) -> bool:
        """True if the structure has a full-rank 3D cell and at least one periodic axis.

        A surface slab (pbc partially True) counts as periodic; a molecule
        in a vacuum box (pbc all False) does not. Wrapped in bool() because
        Cell.rank is a numpy integer and ``==`` yields a numpy bool.
        """
        return bool(self._atoms.get_cell().rank == 3 and self._atoms.pbc.any())

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
        self._push_undo()
        self._atoms = new_atoms
        self._dirty = True
        self._selected_index = None
        self.structure_modified.emit()

    def reset_filepath(self) -> None:
        """Detach from the loaded file so the next Save forces Save As.

        Call after deriving a new structure (surface cut, supercell) so
        the original file is never silently overwritten.
        """
        self._filepath = None

    # ------------------------------------------------------------------
    # Edit history (snapshot-based undo/redo)
    # ------------------------------------------------------------------

    def _push_undo(self) -> None:
        """Snapshot the current atoms before a mutation."""
        self._undo_stack.append(self._atoms.copy())
        self._redo_stack.clear()
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)

    def _clear_history(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> None:
        """Revert the most recent structure change."""
        if not self._undo_stack:
            return
        self._redo_stack.append(self._atoms.copy())
        self._restore(self._undo_stack.pop())

    def redo(self) -> None:
        """Re-apply the most recently undone change."""
        if not self._redo_stack:
            return
        self._undo_stack.append(self._atoms.copy())
        self._restore(self._redo_stack.pop())

    def _restore(self, atoms: Atoms) -> None:
        """Restore a history snapshot and notify observers."""
        self._atoms = atoms
        self._dirty = True
        self._selected_index = None
        self.structure_modified.emit()

    def set_cell(self, cell: np.ndarray) -> None:
        """Set the unit cell (3x3 matrix)."""
        self._push_undo()
        self._atoms.set_cell(cell)
        self._dirty = True
        self.structure_modified.emit()

    def make_periodic(self, padding: float) -> None:
        """Wrap the structure into a padded periodic cell (vacuum box).

        Undoable: pushes the current state onto the undo stack and emits
        structure_modified. Atom indices (and thus the selection) are
        preserved; the filepath is NOT changed. Intended for molecule-like
        structures; do not call on slabs (pbc partially True) — it rebuilds
        the cell from the atom bounding box.

        Args:
            padding: Vacuum padding on each side of the bounding box, in Angstrom.

        Raises:
            ValueError: If the structure is empty or padding is not positive.
        """
        if len(self._atoms) == 0:
            raise ValueError("Cannot wrap an empty structure.")
        if not np.isfinite(padding) or padding <= 0:
            raise ValueError(f"Padding must be a positive number, got {padding}.")
        self._push_undo()
        self._atoms = wrap_in_padded_cell(self._atoms, float(padding))
        self._dirty = True
        self.structure_modified.emit()

    def set_positions(self, positions: np.ndarray) -> None:
        """Set Cartesian positions (N×3)."""
        self._push_undo()
        self._atoms.set_positions(positions)
        self._dirty = True
        self.structure_modified.emit()

    def set_scaled_positions(self, scaled: np.ndarray) -> None:
        """Set fractional coordinates (N×3)."""
        self._push_undo()
        self._atoms.set_scaled_positions(scaled)
        self._dirty = True
        self.structure_modified.emit()

    def extend_atoms(self, other: Atoms | StructureModel) -> None:
        """Append atoms from another Atoms or StructureModel."""
        if isinstance(other, StructureModel):
            other = other._atoms
        self._push_undo()
        self._atoms.extend(other)
        self._dirty = True
        self.structure_modified.emit()

    def set_atom_position(self, index: int, position: np.ndarray) -> None:
        """Set one atom's Cartesian position (absolute)."""
        if 0 <= index < len(self._atoms):
            self._push_undo()
            self._atoms.positions[index] = np.asarray(position, dtype=float)
            self._dirty = True
            self.structure_modified.emit()

    def delete_atom(self, index: int) -> None:
        """Delete the atom at the given index.

        Shifts the selection when an atom before it is deleted, and
        emits selection_cleared when the selected atom itself is removed.
        """
        if 0 <= index < len(self._atoms):
            self._push_undo()
            del self._atoms[index]
            self._dirty = True
            if self._selected_index is not None:
                if self._selected_index == index:
                    self._selected_index = None
                    self.selection_cleared.emit()
                elif self._selected_index > index:
                    self._selected_index -= 1
            self.structure_modified.emit()

    def translate_atom(self, index: int, vector: np.ndarray) -> None:
        """Translate a single atom by a Cartesian vector."""
        if 0 <= index < len(self._atoms):
            self._push_undo()
            self._atoms[index].position += np.asarray(vector)
            self._dirty = True
            self.structure_modified.emit()

    def __repr__(self) -> str:
        return (
            f"StructureModel(formula={self.chemical_formula}, "
            f"n_atoms={self.n_atoms}, file={self._filepath})"
        )
