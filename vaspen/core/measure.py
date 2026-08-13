"""Geometric measurements on structures (pure functions, no Qt).

Distance, bond angle and dihedral are computed from minimum-image
vectors, so periodic structures measure the physically shortest value
across cell boundaries.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.geometry import get_angles, get_dihedrals, get_distances


def _cell_pbc(atoms: Atoms) -> tuple[np.ndarray, tuple[bool, bool, bool]]:
    cell = atoms.get_cell().array if atoms.get_cell().rank == 3 else np.eye(3)
    pbc = tuple(atoms.get_pbc()) if atoms.get_pbc().any() else (False, False, False)
    return np.asarray(cell, dtype=float), pbc


def _mic_displacement(atoms: Atoms, i: int, j: int) -> np.ndarray:
    """Minimum-image vector from atom i to atom j."""
    pos = np.asarray(atoms.get_positions(), dtype=float)
    cell, pbc = _cell_pbc(atoms)
    vecs, _dists = get_distances(pos[[i]], pos[[j]], cell=cell, pbc=pbc)
    return np.asarray(vecs[0, 0], dtype=float)


def distance(atoms: Atoms, i: int, j: int) -> float:
    """Minimum-image distance between atoms i and j (Å)."""
    pos = np.asarray(atoms.get_positions(), dtype=float)
    cell, pbc = _cell_pbc(atoms)
    _vecs, dists = get_distances(pos[[i]], pos[[j]], cell=cell, pbc=pbc)
    return float(dists[0, 0])


def angle(atoms: Atoms, i: int, j: int, k: int) -> float:
    """Bond angle i–j–k in degrees (0..180), minimum-image vectors."""
    v1 = _mic_displacement(atoms, j, i)
    v2 = _mic_displacement(atoms, j, k)
    return float(get_angles(np.array([v1]), np.array([v2]))[0])


def dihedral(atoms: Atoms, i: int, j: int, k: int, l: int) -> float:
    """Dihedral angle i–j–k–l in degrees, minimum-image vectors.

    ASE returns the angle in [0, 360); this normalizes to (-180, 180]
    (the common chemistry convention).
    """
    v0 = _mic_displacement(atoms, i, j)
    v1 = _mic_displacement(atoms, j, k)
    v2 = _mic_displacement(atoms, k, l)
    value = float(get_dihedrals(
        np.array([v0]), np.array([v1]), np.array([v2]))[0])
    if value > 180.0:
        value -= 360.0
    return value
