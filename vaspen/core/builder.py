"""Structure builder — add/remove atoms, supercell, symmetry operations.

Operates on a StructureModel and emits signals so the UI stays in sync.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import cut, make_supercell, sort as ase_sort
from ase.build import molecule as ase_molecule
from ase.data import atomic_numbers as _atomic_numbers
from ase.data import chemical_symbols as _chemical_symbols

from vaspen.core.structure import StructureModel


class StructureBuilder:
    """Stateless structure manipulation methods.

    Each method takes a StructureModel and optional parameters, performs
    the operation, and updates the model (which emits signals).
    """

    @staticmethod
    def add_atom(
        model: StructureModel,
        symbol: str,
        position: np.ndarray | tuple[float, float, float],
    ) -> None:
        """Add a single atom at a Cartesian position.

        Args:
            model: Target structure model.
            symbol: Element symbol (e.g. "Fe").
            position: Cartesian (x, y, z) in Angstrom.
        """
        new_atom = Atoms(symbol, positions=[position], cell=model.cell, pbc=model.pbc)
        model.extend_atoms(new_atom)

    @staticmethod
    def remove_atom(model: StructureModel, index: int) -> None:
        """Remove the atom at the given index."""
        model.delete_atom(index)

    @staticmethod
    def replace_element(model: StructureModel, index: int, new_symbol: str) -> None:
        """Replace the element of a single atom.

        The selection is preserved (the atom keeps its index), so the
        properties panel / tree can chain element + position edits.

        Args:
            model: Target structure model.
            index: Atom index.
            new_symbol: New element symbol.
        """
        atoms = model.atoms.copy()  # copy first so undo snapshots stay valid
        if 0 <= index < len(atoms):
            atoms[index].symbol = new_symbol
            selection = model.selected_indices
            model.replace_atoms(atoms)  # triggers signal (and clears selection)
            if selection:
                model.set_selection(selection)

    @staticmethod
    def make_supercell(
        model: StructureModel,
        scaling: tuple[int, int, int],
    ) -> None:
        """Create a supercell by repeating the unit cell.

        Args:
            model: Target structure model.
            scaling: (n_a, n_b, n_c) repeat factors.
        """
        P = np.diag(scaling)
        new_atoms = make_supercell(model.atoms, P)
        model.replace_atoms(new_atoms)

    @staticmethod
    def translate_atoms(
        model: StructureModel,
        vector: np.ndarray,
        indices: list[int] | None = None,
    ) -> None:
        """Translate atoms by a Cartesian vector.

        Args:
            model: Target structure model.
            vector: (dx, dy, dz) in Angstrom.
            indices: Atom indices to move. If None, moves all atoms.
        """
        atoms = model.atoms.copy()  # copy first so undo snapshots stay valid
        vec = np.asarray(vector)
        if indices is None:
            atoms.translate(vec)
        else:
            for i in indices:
                if 0 <= i < len(atoms):
                    atoms[i].position += vec
        model.replace_atoms(atoms)

    @staticmethod
    def sort_atoms(model: StructureModel) -> None:
        """Sort atoms by atomic number (default ASE sort)."""
        new_atoms = ase_sort(model.atoms)
        model.replace_atoms(new_atoms)

    @staticmethod
    def wrap_atoms(model: StructureModel) -> None:
        """Wrap atoms back into the unit cell."""
        atoms = model.atoms.copy()
        atoms.wrap()
        model.replace_atoms(atoms)

    @staticmethod
    def center_atoms(model: StructureModel, axis: tuple[bool, bool, bool] = (True, True, True)) -> None:
        """Center atoms in the unit cell along the given axes."""
        atoms = model.atoms.copy()
        atoms.center()
        model.replace_atoms(atoms)

    @staticmethod
    def build_molecule(symbol: str) -> StructureModel:
        """Create a StructureModel pre-populated with a molecule.

        Uses ASE's G2 molecule database for common molecules.

        Args:
            symbol: Molecule identifier (e.g. "H2O", "CH4", "C6H6").

        Returns:
            New StructureModel with the molecule, in a large vacuum box.
        """
        try:
            mol = ase_molecule(symbol)
        except Exception:
            raise ValueError(f"Unknown molecule: {symbol}")

        # Place in a box with generous vacuum
        mol.center(vacuum=10.0)
        return StructureModel(mol)
