"""Symmetry analysis and symmetrization (thin wrappers).

Periodic structures go through spglib (space groups); isolated systems
(molecules — no pbc) go through pymatgen's PointGroupAnalyzer
(Schoenflies point groups). Both return the same SymmetryInfo shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib
from ase import Atoms
from ase.geometry import cell_to_cellpar, cellpar_to_cell

# Point-group tolerance floor in Angstrom: ASE/pymatgen reference
# geometries are essentially exact, but files from the wild are not —
# 0.01 Å keeps e.g. H2O as C2v without over-symmetrizing noise.
_POINT_TOL_FLOOR = 0.01


@dataclass(frozen=True)
class SymmetryInfo:
    """Symmetry data of a structure.

    kind == "space": number/international/hall/pointgroup/choice from
        spglib (space group of a periodic structure).
    kind == "point": number = rotational symmetry number σ;
        international = Schoenflies symbol (e.g. C2v); hall/choice empty.
    """

    kind: str
    number: int
    international: str
    hall: str = ""
    pointgroup: str = ""
    choice: str = ""


def _as_spglib_cell(atoms: Atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lattice, scaled_positions, numbers) tuple expected by spglib."""
    return (
        np.asarray(atoms.get_cell().array, dtype=float),
        np.asarray(atoms.get_scaled_positions(), dtype=float),
        np.asarray(atoms.get_atomic_numbers(), dtype=int),
    )


def _as_pymatgen_molecule(atoms: Atoms):
    """pymatgen Molecule built from the atoms (isolated systems only)."""
    from pymatgen.core import Element, Molecule

    return Molecule(
        species=[Element(s) for s in atoms.get_chemical_symbols()],
        coords=np.asarray(atoms.get_positions(), dtype=float),
    )


def analyze(atoms: Atoms, symprec: float = 1e-5) -> SymmetryInfo | None:
    """Symmetry of a structure, or None when analysis fails.

    Periodic structures (full-rank cell + at least one pbc axis) return
    the spglib space group; isolated systems (molecules) return the
    pymatgen Schoenflies point group.

    Args:
        atoms: Structure to analyze (not modified).
        symprec: Position tolerance in Angstrom (spglib; for point
            groups the tolerance is floored at 0.01 Å).

    Returns:
        SymmetryInfo, or None if the analyzer fails.
    """
    if atoms.get_cell().rank == 3 and atoms.get_pbc().any():
        dataset = spglib.get_symmetry_dataset(_as_spglib_cell(atoms),
                                              symprec=symprec)
        if dataset is None:
            return None
        return SymmetryInfo(
            kind="space",
            number=int(dataset.number),
            international=dataset.international,
            hall=dataset.hall,
            pointgroup=dataset.pointgroup,
            choice=dataset.choice or "",
        )
    if len(atoms) == 0:
        return None
    try:
        from pymatgen.symmetry.analyzer import PointGroupAnalyzer

        analyzer = PointGroupAnalyzer(
            _as_pymatgen_molecule(atoms),
            tolerance=max(float(symprec), _POINT_TOL_FLOOR),
        )
        sch = analyzer.sch_symbol or ""
        if not sch:
            return None
        return SymmetryInfo(
            kind="point",
            number=int(analyzer.get_rotational_symmetry_number()),
            international=sch,
            pointgroup=sch,
        )
    except (ValueError, TypeError, KeyError):
        return None


def symmetrize(atoms: Atoms, symprec: float = 1e-5,
               cell_type: str = "conventional") -> Atoms | None:
    """Symmetrized copy, or None.

    Periodic: spglib standardize_cell —
    ``cell_type="conventional"`` returns the standardized conventional
    cell; ``cell_type="primitive"`` returns the primitive cell in the
    standard orientation (a ∥ x, b in the xy-plane, c along +z — the
    VESTA-style presentation; a pure rotation of the spglib primitive,
    atom distances unchanged).
    Isolated system: pymatgen symmetrize_molecule (idealized point-group
    symmetric coordinates; the cell/pbc of the input are preserved, the
    atom ORDER may follow pymatgen's species sorting).

    Note: custom arrays (charges, forces, velocities) are NOT carried
    over. The input Atoms object is not modified.

    Args:
        atoms: Structure to symmetrize (not modified).
        symprec: Position tolerance in Angstrom (same semantics as analyze).
        cell_type: "conventional" or "primitive" (periodic structures).

    Returns:
        Symmetrized Atoms, or None if the symmetrizer fails.
    """
    if atoms.get_cell().rank == 3 and atoms.get_pbc().any():
        result = spglib.standardize_cell(
            _as_spglib_cell(atoms),
            to_primitive=(cell_type == "primitive"),
            no_idealize=False,
            symprec=symprec,
        )
        if result is None:
            return None
        lattice, scaled, numbers = result
        if cell_type == "primitive":
            # Standard orientation from the cell parameters (pure
            # rotation — spglib's own presentation is expressed in
            # the input basis and reads as a skewed box).
            lattice = cellpar_to_cell(cell_to_cellpar(lattice))
        return Atoms(numbers=numbers, scaled_positions=scaled,
                     cell=lattice, pbc=True)
    if len(atoms) == 0:
        return None
    try:
        from pymatgen.symmetry.analyzer import PointGroupAnalyzer

        analyzer = PointGroupAnalyzer(
            _as_pymatgen_molecule(atoms),
            tolerance=max(float(symprec), _POINT_TOL_FLOOR),
        )
        result = analyzer.symmetrize_molecule()
        # pymatgen returns {"sym_mol": Molecule, ...} in recent versions
        sym_mol = result["sym_mol"] if isinstance(result, dict) else result
        if sym_mol is None or len(sym_mol) != len(atoms):
            return None
        # Molecule iteration yields Sites; str(site) is "coords species",
        # so the element symbol must come from site.specie
        symbols: list[str] = []
        for site in sym_mol:
            specie = getattr(site, "specie", None)
            if specie is None:
                specie = site
            symbols.append(specie.symbol if hasattr(specie, "symbol")
                           else str(specie))
        out = Atoms(
            symbols=symbols,
            positions=np.asarray(sym_mol.cart_coords, dtype=float),
        )
        # preserve the input's cell/pbc so molecules in a box stay put
        out.set_cell(atoms.get_cell().array)
        out.set_pbc(atoms.get_pbc())
        return out
    except (ValueError, TypeError, KeyError):
        return None
