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
            # Element replacement keeps the fixed flags and moments
            # (the atom stays).
            model.replace_atoms(atoms, fixed_flags=model.fixed_flags,
                                magmoms=model.magmoms)
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
        # Diagonal-P supercells are block-ordered (new index = block*N + i),
        # so tiling the per-atom flags/moments replicates them per image.
        n_blocks = int(np.prod(scaling))
        flags = np.tile(model.fixed_flags, (n_blocks, 1))
        model.replace_atoms(new_atoms, fixed_flags=flags,
                            magmoms=np.tile(model.magmoms, n_blocks))

    @staticmethod
    def translate_atoms(
        model: StructureModel,
        vector: np.ndarray,
        indices: list[int] | None = None,
    ) -> None:
        """Translate atoms by a Cartesian vector.

        NOTE: not wired into the UI — goes through replace_atoms, which
        clears fixed flags; callers must pass the model's flags (or
        reject frozen atoms) themselves.

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
        model.replace_atoms(atoms, fixed_flags=model.fixed_flags,
                            magmoms=model.magmoms)

    @staticmethod
    def sort_atoms(model: StructureModel) -> None:
        """Sort atoms by atomic number (default ASE sort).

        A pure re-ordering: the fixed flags follow the atoms through the
        same stable-sort permutation ASE uses, but the mapping between
        old and new indices is lost for bonds/IDs (replace_atoms rebuilds
        them) — not wired into the UI.
        """
        new_atoms = ase_sort(model.atoms)
        # ASE sorts by chemical symbol (stable) — the permutation must
        # match it exactly so per-atom flags/moments follow their atoms.
        order = np.argsort(model.atoms.get_chemical_symbols(), kind="stable")
        model.replace_atoms(new_atoms, fixed_flags=model.fixed_flags[order],
                            magmoms=model.magmoms[order])

    @staticmethod
    def wrap_atoms(model: StructureModel) -> None:
        """Wrap atoms back into the unit cell.

        NOTE: not wired into the UI — see translate_atoms. A wrap is a
        periodic identity, so flags carry over unchanged.
        """
        atoms = model.atoms.copy()
        atoms.wrap()
        model.replace_atoms(atoms, fixed_flags=model.fixed_flags,
                            magmoms=model.magmoms)

    @staticmethod
    def center_atoms(model: StructureModel, axis: tuple[bool, bool, bool] = (True, True, True)) -> None:
        """Center atoms in the unit cell along the given axes.

        NOTE: not wired into the UI — see translate_atoms.
        """
        atoms = model.atoms.copy()
        atoms.center()
        model.replace_atoms(atoms, fixed_flags=model.fixed_flags,
                            magmoms=model.magmoms)

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
