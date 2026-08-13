"""StructureModel — wraps ASE Atoms with Qt signal support.

This is the central data model for the application. All structural
information flows through this class, and changes are broadcast via
Qt signals so that every observer (3D viewport, structure tree,
status bar) stays in sync.

Beyond the ASE Atoms payload the model owns:
- a persistent bond list (detected once at load; manual by default —
  the Auto Detect Bonds toggle re-enables per-edit recomputation),
- stable atom IDs (survive deletion/undo — measurements reference
  them instead of indices),
- multi-atom selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell
from ase.io import read as ase_read
from PySide6.QtCore import QObject, Signal

from vaspen.core.bonds import Bond, find_bonds
from vaspen.core.file_io import FileIO, extract_occupancy


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


def _compute_bonds(atoms: Atoms) -> list[Bond]:
    """Auto-connectivity for an Atoms object (empty-safe)."""
    if len(atoms) == 0:
        return []
    cell = atoms.get_cell().array if atoms.get_cell().rank == 3 else np.eye(3)
    pbc = tuple(atoms.get_pbc()) if atoms.get_pbc().any() else (False, False, False)
    return find_bonds(
        np.asarray(atoms.get_positions(), dtype=float),
        list(atoms.get_chemical_symbols()),
        cell,
        pbc,
    )


@dataclass(frozen=True)
class _Snapshot:
    """One undo history entry: the full model state before a mutation.

    Bonds and atom IDs cannot live on the ASE Atoms object (e.g.
    ase.build.make_supercell drops ``info``, ``extend`` fills custom
    arrays with zeros), so the undo stack stores them beside the atoms.
    The selection is stored too — undo/redo restores it, so the user
    can step back to the state where they had just selected atoms.
    """

    atoms: Atoms
    bonds: list[Bond]
    bond_mode: str
    atom_ids: list[int]
    next_id: int
    selected_indices: frozenset[int]
    selected_bonds: frozenset[int]


class StructureModel(QObject):
    """Observable wrapper around an ASE Atoms object.

    Signals:
        structure_loaded: Emitted after a structure is loaded from disk.
        structure_modified: Emitted after any structural change (add/remove/
            move atoms, change cell, bond edits, etc.).
        atom_selected(int): Emitted when the user selects an atom by index.
        selection_cleared: Emitted when the selection is cleared.
        selection_changed: Emitted whenever the selected-atom set changes
            (multi-select; also fires for the legacy single selection).
    """

    structure_loaded = Signal()
    structure_modified = Signal()
    atom_selected = Signal(int)
    selection_cleared = Signal()
    selection_changed = Signal()
    bond_selection_changed = Signal()

    def __init__(self, atoms: Atoms | None = None, parent: QObject | None = None) -> None:
        """Initialize the model.

        Args:
            atoms: Optional ASE Atoms object. If None, creates an empty cell.
            parent: Qt parent object.
        """
        super().__init__(parent)
        self._atoms: Atoms = atoms if atoms is not None else Atoms()
        self._filepath: str | None = None
        self._selected_indices: set[int] = set()
        self._selected_bonds: set[int] = set()  # bond list indices
        self._dirty: bool = False
        # Edit history (snapshot-based undo/redo)
        self._undo_stack: list[_Snapshot] = []
        self._redo_stack: list[_Snapshot] = []
        self._max_history = 50
        # Persistent bond list + stable atom IDs
        self._bonds: list[Bond] = []
        # Settled policy (2026-08-13): bonds are detected ONCE at load;
        # manual mode is the default — the Auto Detect Bonds toggle
        # re-enables per-edit recomputation.
        self._bond_mode: str = "manual"  # "auto" | "manual"
        self._atom_ids: list[int] = []
        self._next_id: int = 0
        # Per-atom site compositions for partially-occupied (disordered)
        # structures; None when the structure has no occupancy data.
        # Derived from atoms.info['occupancy'] at load time.
        self._occupancy: list[dict[str, float]] | None = None

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
        """Lowest index of the selected atoms, or None (legacy single-select)."""
        return min(self._selected_indices) if self._selected_indices else None

    @property
    def selected_indices(self) -> set[int]:
        """Copy of the selected atom index set."""
        return set(self._selected_indices)

    @property
    def bonds(self) -> list[Bond]:
        """Copy of the persistent bond list."""
        return list(self._bonds)

    @property
    def bond_mode(self) -> str:
        """Bond mode: "auto" (recomputed per mutation) or "manual"."""
        return self._bond_mode

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
        self._occupancy = extract_occupancy(atoms)
        self._filepath = str(path)
        self._dirty = False
        self._selected_indices = set()
        self._reset_derived_state()
        self._clear_history()
        self.structure_loaded.emit()

    def load_atoms(self, atoms: Atoms, filepath: str | Path | None = None) -> None:
        """Replace contents with the given Atoms and mark as loaded.

        Emits structure_loaded (not structure_modified) so observers
        rebuild the scene exactly once. Used by the UI when loading
        through FileIO.
        """
        self._atoms = atoms
        self._occupancy = extract_occupancy(atoms)
        if filepath is not None:
            self._filepath = str(Path(filepath))
        self._dirty = False
        self._selected_indices = set()
        self._reset_derived_state()
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
        if self._occupancy is None:
            return self._atoms.get_chemical_formula()
        # Disordered structure: fractional coefficients from the site
        # composition (only occupied species; e.g. "Fe1.3Ni0.5Co0.2").
        parts = []
        for sym, total in self.composition():
            coeff = f"{total:.2f}".rstrip("0").rstrip(".")
            parts.append(sym + coeff)
        return "".join(parts)

    @property
    def occupancy(self) -> list[dict[str, float]] | None:
        """Per-atom site compositions ({symbol: occupancy}), or None.

        Only populated for partially-occupied (disordered) structures;
        uniform single-species occupancy reads as None.
        """
        return (list(self._occupancy)
                if self._occupancy is not None else None)

    @property
    def has_disorder(self) -> bool:
        """True when the structure has partial (fractional) occupancy."""
        return self._occupancy is not None

    def composition(self) -> list[tuple[str, float]]:
        """Overall composition as (symbol, total occupancy) pairs.

        Each merged atom represents one expanded lattice site, so the
        per-atom site compositions sum to the full-cell composition
        (symmetry multiplicities included). Sorted by descending total,
        ties by symbol. The deficit against ``n_atoms`` is vacancy.

        Raises:
            RuntimeError: If called on a structure without occupancy
                data (``has_disorder`` is False).
        """
        if self._occupancy is None:
            raise RuntimeError("Structure has no occupancy data.")
        totals: dict[str, float] = {}
        for comp in self._occupancy:
            for sym, occ in comp.items():
                totals[sym] = totals.get(sym, 0.0) + occ
        return sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))

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

    def atom_id(self, index: int) -> int | None:
        """Stable ID of the atom at ``index`` (None if out of bounds).

        IDs survive deletion and undo — measurements reference atoms by
        ID so their endpoints never silently shift.
        """
        if 0 <= index < len(self._atom_ids):
            return self._atom_ids[index]
        return None

    def index_of_id(self, atom_id: int) -> int | None:
        """Current index of the atom with ``atom_id`` (None if absent)."""
        try:
            return self._atom_ids.index(atom_id)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_atom(self, index: int) -> None:
        """Select a single atom by index (legacy single-select entry)."""
        if 0 <= index < len(self._atoms):
            self._selected_indices = {index}
            self.atom_selected.emit(index)
            self.selection_changed.emit()

    def clear_selection(self) -> None:
        """Clear the current atom selection."""
        self._selected_indices = set()
        self.selection_cleared.emit()
        self.selection_changed.emit()

    def set_selection(self, indices: Iterable[int]) -> None:
        """Replace the selection with the given atom indices."""
        new = {i for i in indices if 0 <= i < len(self._atoms)}
        if new != self._selected_indices:
            self._selected_indices = new
            self.selection_changed.emit()

    def add_to_selection(self, indices: Iterable[int]) -> None:
        """Add atoms to the current selection (union)."""
        new = self._selected_indices | {i for i in indices if 0 <= i < len(self._atoms)}
        if new != self._selected_indices:
            self._selected_indices = new
            self.selection_changed.emit()

    def toggle_selection(self, indices: Iterable[int]) -> None:
        """Toggle atoms in/out of the selection (symmetric difference)."""
        new = self._selected_indices ^ {i for i in indices if 0 <= i < len(self._atoms)}
        if new != self._selected_indices:
            self._selected_indices = new
            self.selection_changed.emit()

    def select_all(self) -> None:
        """Select every atom."""
        self.set_selection(range(len(self._atoms)))

    def select_none(self) -> None:
        """Clear the selection (menu alias for clear_selection)."""
        self.clear_selection()

    def select_invert(self) -> None:
        """Invert the selection within the structure."""
        self.set_selection(
            {i for i in range(len(self._atoms)) if i not in self._selected_indices}
        )

    def select_neighbors(self) -> None:
        """Select the atoms bonded to the currently selected atoms."""
        if not self._selected_indices:
            return
        neighbors: set[int] = set()
        for b in self._bonds:
            if b.i in self._selected_indices:
                neighbors.add(b.j)
            if b.j in self._selected_indices:
                neighbors.add(b.i)
        self.set_selection(neighbors)

    def select_connected(self) -> None:
        """Select the connected component(s) containing the selection (BFS)."""
        if not self._selected_indices:
            return
        adjacency: dict[int, set[int]] = {}
        for b in self._bonds:
            adjacency.setdefault(b.i, set()).add(b.j)
            adjacency.setdefault(b.j, set()).add(b.i)
        seen: set[int] = set()
        frontier = list(self._selected_indices)
        while frontier:
            atom = frontier.pop()
            if atom in seen:
                continue
            seen.add(atom)
            frontier.extend(adjacency.get(atom, ()))
        self.set_selection(seen)

    # ------------------------------------------------------------------
    # Bond selection (bond list indices)
    # ------------------------------------------------------------------

    @property
    def selected_bonds(self) -> set[int]:
        """Copy of the selected bond indices (into the bond list)."""
        return set(self._selected_bonds)

    def select_bond(self, index: int) -> None:
        """Select a single bond by its list index."""
        if 0 <= index < len(self._bonds):
            self._selected_bonds = {index}
            self.bond_selection_changed.emit()

    def clear_bond_selection(self) -> None:
        """Clear the bond selection."""
        self._selected_bonds = set()
        self.bond_selection_changed.emit()

    def _clear_stale_bond_selection(self) -> None:
        """Drop the bond selection when the bond list changes."""
        if self._selected_bonds:
            self._selected_bonds = set()
            self.bond_selection_changed.emit()

    # ------------------------------------------------------------------
    # Structure mutation
    # ------------------------------------------------------------------

    def replace_atoms(self, new_atoms: Atoms) -> None:
        """Replace the entire structure with a new Atoms object.

        Bonds reset to auto-connectivity (indices of the derived
        structure are unrelated to the previous ones).
        """
        self._push_undo()
        self._atoms = new_atoms
        self._clear_occupancy()
        self._dirty = True
        self._selected_indices = set()
        self._reset_derived_state()
        self.structure_modified.emit()

    def _clear_occupancy(self) -> None:
        """Drop occupancy data after a structure mutation.

        Disordered (partially-occupied) structures are display-only: any
        structural edit makes the site composition invalid, and ASE's own
        mutation routines corrupt the info (``Atoms.extend`` zero-fills
        ``spacegroup_kinds``, ``make_supercell`` drops ``info``) — a
        desynced occupancy dict would later crash or corrupt the CIF
        writer. Clearing keeps the atoms in a consistent state.
        """
        if "occupancy" in self._atoms.info:
            del self._atoms.info["occupancy"]
        if "spacegroup_kinds" in self._atoms.arrays:
            del self._atoms.arrays["spacegroup_kinds"]
        self._occupancy = None

    def reset_filepath(self) -> None:
        """Detach from the loaded file so the next Save forces Save As.

        Call after deriving a new structure (surface cleave, supercell) so
        the original file is never silently overwritten.
        """
        self._filepath = None

    # ------------------------------------------------------------------
    # Edit history (snapshot-based undo/redo)
    # ------------------------------------------------------------------

    def _reset_derived_state(self) -> None:
        """Reset bonds/IDs to match the current atoms (load / replace).

        Bonds are detected ONCE here; the mode stays manual (settled
        policy) so later edits do not silently recompute them.
        """
        self._bonds = _compute_bonds(self._atoms)
        self._bond_mode = "manual"
        self._selected_bonds = set()
        n = len(self._atoms)
        self._atom_ids = list(range(n))
        self._next_id = n

    def _recompute_if_auto(self) -> None:
        """Refresh the bond list after a mutation while in auto mode."""
        if self._bond_mode == "auto":
            self._bonds = _compute_bonds(self._atoms)
            self._clear_stale_bond_selection()

    def _snapshot(self) -> _Snapshot:
        """Capture the full model state for the undo stack."""
        return _Snapshot(
            self._atoms.copy(),
            list(self._bonds),
            self._bond_mode,
            list(self._atom_ids),
            self._next_id,
            frozenset(self._selected_indices),
            frozenset(self._selected_bonds),
        )

    def _push_undo(self) -> None:
        """Snapshot the current state before a mutation."""
        self._undo_stack.append(self._snapshot())
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
        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())

    def redo(self) -> None:
        """Re-apply the most recently undone change."""
        if not self._redo_stack:
            return
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())

    def _restore(self, snapshot: _Snapshot) -> None:
        """Restore a history snapshot and notify observers.

        The selection is restored too (indices that were deleted in the
        restored state are dropped silently).
        """
        self._atoms = snapshot.atoms
        # The snapshot's atoms carry the pre-edit occupancy info (ASE
        # copy preserves info + arrays) — recompute the cache so undoing
        # an edit restores the composition display too.
        self._occupancy = extract_occupancy(snapshot.atoms)
        self._bonds = list(snapshot.bonds)
        self._bond_mode = snapshot.bond_mode
        self._atom_ids = list(snapshot.atom_ids)
        self._next_id = snapshot.next_id
        self._dirty = True
        self._selected_indices = {
            i for i in snapshot.selected_indices if 0 <= i < len(self._atoms)
        }
        self._selected_bonds = {
            k for k in snapshot.selected_bonds if 0 <= k < len(self._bonds)
        }
        self.structure_modified.emit()
        self.selection_changed.emit()
        self.bond_selection_changed.emit()

    def set_cell(self, cell: np.ndarray) -> None:
        """Set the unit cell (3x3 matrix)."""
        self._push_undo()
        self._atoms.set_cell(cell)
        self._dirty = True
        self._recompute_if_auto()
        self.structure_modified.emit()

    def set_cell_parameters(
        self,
        lengths: tuple[float, float, float],
        angles: tuple[float, float, float],
        scale_atoms: bool = True,
    ) -> None:
        """Set the unit cell from lengths (a, b, c) and angles (α, β, γ).

        Undoable: one push, emits structure_modified. With
        ``scale_atoms`` the fractional coordinates are kept (atoms follow
        the deforming cell); otherwise Cartesian coordinates are kept.

        Args:
            lengths: (a, b, c) cell lengths in Angstrom.
            angles: (α, β, γ) cell angles in degrees.
            scale_atoms: Keep fractional coordinates (default True).

        Raises:
            ValueError: For non-positive lengths, angles outside (0, 180),
                or a degenerate cell (zero/negative volume).
        """
        if not all(np.isfinite(lengths)) or not all(l > 0 for l in lengths):
            raise ValueError(f"Cell lengths must be positive, got {lengths}.")
        if not all(np.isfinite(angles)) or not all(0 < a < 180 for a in angles):
            raise ValueError(f"Cell angles must be in (0, 180), got {angles}.")
        cell = cellpar_to_cell([*lengths, *angles])
        # relative criterion: degenerate if the volume is below 1 ppm of
        # the axis-aligned box (also catches floating-point near-zero)
        if float(np.linalg.det(cell)) <= 1e-6 * float(np.prod(lengths)):
            raise ValueError("Degenerate cell parameters (zero or negative volume).")
        self._push_undo()
        self._atoms.set_cell(cell, scale_atoms=scale_atoms)
        self._dirty = True
        self._recompute_if_auto()
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
        self._recompute_if_auto()
        self.structure_modified.emit()

    def set_positions(self, positions: np.ndarray) -> None:
        """Set Cartesian positions (N×3)."""
        self._push_undo()
        self._atoms.set_positions(positions)
        self._dirty = True
        self._recompute_if_auto()
        self.structure_modified.emit()

    def set_geometry(
        self,
        positions: np.ndarray,
        cell: np.ndarray | None = None,
    ) -> None:
        """Set Cartesian positions and optionally the unit cell in ONE
        undo step.

        Used by the Transform dialog: a whole-structure transform of a
        periodic structure rotates the cell together with the atoms, and
        both must land in a single undo entry. Bonds survive (manual
        mode) or re-detect (auto mode).
        """
        self._push_undo()
        self._atoms.set_positions(positions)
        if cell is not None:
            self._atoms.set_cell(cell)
        self._dirty = True
        self._recompute_if_auto()
        self.structure_modified.emit()

    def set_scaled_positions(self, scaled: np.ndarray) -> None:
        """Set fractional coordinates (N×3)."""
        self._push_undo()
        self._atoms.set_scaled_positions(scaled)
        self._dirty = True
        self._recompute_if_auto()
        self.structure_modified.emit()

    def set_atom_scaled_position(self, index: int, scaled: np.ndarray) -> None:
        """Set one atom's fractional coordinates (absolute).

        Requires a full-rank cell (raises ASE's error otherwise).
        """
        if not 0 <= index < len(self._atoms):
            return
        self._push_undo()
        scaled_positions = self._atoms.get_scaled_positions()
        scaled_positions[index] = np.asarray(scaled, dtype=float)
        self._atoms.set_scaled_positions(scaled_positions)
        self._dirty = True
        self._recompute_if_auto()
        self.structure_modified.emit()

    def extend_atoms(self, other: Atoms | StructureModel) -> None:
        """Append atoms from another Atoms or StructureModel."""
        if isinstance(other, StructureModel):
            other = other._atoms
        self._push_undo()
        self._atoms.extend(other)
        self._clear_occupancy()
        self._atom_ids.extend(range(self._next_id, self._next_id + len(other)))
        self._next_id += len(other)
        self._dirty = True
        self._recompute_if_auto()
        self.structure_modified.emit()

    def set_atom_position(self, index: int, position: np.ndarray) -> None:
        """Set one atom's Cartesian position (absolute)."""
        if 0 <= index < len(self._atoms):
            self._push_undo()
            self._atoms.positions[index] = np.asarray(position, dtype=float)
            self._dirty = True
            self._recompute_if_auto()
            self.structure_modified.emit()

    def set_atom_positions(self, indices: Iterable[int], positions: np.ndarray) -> None:
        """Set Cartesian positions of several atoms in ONE undo step.

        Args:
            indices: Atom indices (len K).
            positions: New positions (K×3).

        Raises:
            ValueError: If the arrays have different lengths.
        """
        idx = list(indices)
        pos = np.asarray(positions, dtype=float)
        if len(idx) != len(pos):
            raise ValueError(
                f"Number of indices ({len(idx)}) does not match "
                f"number of positions ({len(pos)})."
            )
        if not idx:
            return
        if not all(0 <= i < len(self._atoms) for i in idx):
            raise ValueError("Atom index out of bounds.")
        self._push_undo()
        for i, p in zip(idx, pos):
            self._atoms.positions[i] = p
        self._dirty = True
        self._recompute_if_auto()
        self.structure_modified.emit()

    def delete_atom(self, index: int) -> None:
        """Delete the atom at the given index.

        Shifts the selection when an atom before it is deleted, and
        emits selection_cleared when the selected atom itself is removed.
        """
        self.delete_atoms([index])

    def delete_atoms(self, indices: Iterable[int]) -> None:
        """Delete several atoms in one undo step.

        Bonds touching a deleted atom are removed; remaining bonds are
        remapped to the shifted indices. Selection follows the legacy
        rule: deleted members are dropped (selection_cleared is emitted
        when the selection becomes empty), members after a deleted atom
        shift down by one.
        """
        deleted = sorted({i for i in indices if 0 <= i < len(self._atoms)},
                         reverse=True)
        if not deleted:
            return
        self._push_undo()

        removed = set(deleted)
        # Selection remap (legacy single-delete semantics generalized)
        new_sel: set[int] = set()
        became_empty = False
        for s in self._selected_indices:
            if s in removed:
                became_empty = True
            else:
                new_sel.add(s - sum(1 for d in deleted if d < s))
        self._selected_indices = new_sel

        # Bond remap: drop bonds touching deleted atoms, shift the rest
        remapped: list[Bond] = []
        for b in self._bonds:
            if b.i in removed or b.j in removed:
                continue
            remapped.append(Bond(
                b.i - sum(1 for d in deleted if d < b.i),
                b.j - sum(1 for d in deleted if d < b.j),
                b.order,
            ))
        self._bonds = remapped
        self._clear_stale_bond_selection()

        for d in deleted:
            del self._atoms[d]
            del self._atom_ids[d]
        self._clear_occupancy()

        self._dirty = True
        if became_empty and not self._selected_indices:
            self.selection_cleared.emit()
        self.structure_modified.emit()

    def translate_atom(self, index: int, vector: np.ndarray) -> None:
        """Translate a single atom by a Cartesian vector."""
        if 0 <= index < len(self._atoms):
            self._push_undo()
            self._atoms[index].position += np.asarray(vector)
            self._dirty = True
            self._recompute_if_auto()
            self.structure_modified.emit()

    # ------------------------------------------------------------------
    # Bond editing (first manual edit switches the structure to manual
    # mode — the current auto list is frozen as the base)
    # ------------------------------------------------------------------

    def _validate_atom_pair(self, i: int, j: int) -> None:
        if i == j:
            raise ValueError("A bond needs two distinct atoms.")
        if not (0 <= i < len(self._atoms) and 0 <= j < len(self._atoms)):
            raise ValueError("Atom index out of bounds.")

    def _switch_to_manual(self) -> None:
        """Freeze the current auto bond list (called before manual edits)."""
        if self._bond_mode == "auto":
            self._bond_mode = "manual"

    def add_bond(self, i: int, j: int, order: int = 1) -> None:
        """Create (or update) a bond between atoms i and j.

        Args:
            i, j: Atom indices (order-insensitive).
            order: Bond order 1..4 (4 = aromatic).

        Raises:
            ValueError: For out-of-range indices, i == j, or bad order.
        """
        bond = Bond(i, j, order)
        self._validate_atom_pair(bond.i, bond.j)
        if not 1 <= order <= 4:
            raise ValueError(f"Bond order must be 1..4, got {order}.")
        self._push_undo()
        self._switch_to_manual()
        self._clear_stale_bond_selection()
        for k, existing in enumerate(self._bonds):
            if (existing.i, existing.j) == (bond.i, bond.j):
                self._bonds[k] = bond
                break
        else:
            self._bonds.append(bond)
        self._dirty = True
        self.structure_modified.emit()

    def remove_bond(self, i: int, j: int) -> None:
        """Delete the bond between atoms i and j (no-op if absent)."""
        bond = Bond(i, j)
        self._validate_atom_pair(bond.i, bond.j)
        self._push_undo()
        self._switch_to_manual()
        self._bonds = [
            b for b in self._bonds if (b.i, b.j) != (bond.i, bond.j)
        ]
        self._clear_stale_bond_selection()
        self._dirty = True
        self.structure_modified.emit()

    def set_bond_order(self, i: int, j: int, order: int) -> None:
        """Change the order of the bond between atoms i and j.

        Creates the bond if it does not exist yet.
        """
        self.add_bond(i, j, order)

    def set_bond_mode(self, mode: str) -> None:
        """Switch bond mode ("auto" | "manual") — undoable.

        Switching back to auto re-derives the connectivity from the
        current geometry (manual edits are discarded); switching to
        manual freezes the current list as the base for further edits.
        """
        if mode not in ("auto", "manual"):
            raise ValueError(f"Unknown bond mode: {mode}")
        if mode == self._bond_mode:
            return
        self._push_undo()
        self._bond_mode = mode
        self._clear_stale_bond_selection()
        if mode == "auto":
            self._bonds = _compute_bonds(self._atoms)
        self._dirty = True
        self.structure_modified.emit()

    def detect_bonds(self) -> None:
        """Re-detect connectivity from the current geometry (one-shot).

        Undoable; does NOT change the bond mode — call this to refresh
        bonds manually while Auto Detect is off.
        """
        self._push_undo()
        self._bonds = _compute_bonds(self._atoms)
        self._clear_stale_bond_selection()
        self._dirty = True
        self.structure_modified.emit()

    def __repr__(self) -> str:
        return (
            f"StructureModel(formula={self.chemical_formula}, "
            f"n_atoms={self.n_atoms}, file={self._filepath})"
        )
