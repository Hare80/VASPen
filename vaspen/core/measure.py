"""Geometric measurements on structures (pure functions, no Qt).

Settled with the user (2026-08-13): measurements are between the
DISPLAYED atoms — plain Cartesian geometry, NO minimum-image vectors —
so a periodic distance is the straight line between the atoms as shown,
never crossing the cell boundary. (A future Display Style - Lattice
feature that shows periodic images outside the cell may re-introduce a
periodic option.)
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.geometry import get_angles, get_dihedrals


def _positions(atoms: Atoms) -> np.ndarray:
    return np.asarray(atoms.get_positions(), dtype=float)


def distance(atoms: Atoms, i: int, j: int) -> float:
    """Direct Cartesian distance between atoms i and j (Å)."""
    pos = _positions(atoms)
    return float(np.linalg.norm(pos[j] - pos[i]))


def angle(atoms: Atoms, i: int, j: int, k: int) -> float:
    """Bond angle i–j–k in degrees (0..180), direct vectors."""
    pos = _positions(atoms)
    v1 = pos[i] - pos[j]
    v2 = pos[k] - pos[j]
    return float(get_angles(np.array([v1]), np.array([v2]))[0])


def dihedral(atoms: Atoms, i: int, j: int, k: int, l: int) -> float:
    """Dihedral angle i–j–k–l in degrees, direct vectors.

    ASE returns the angle in [0, 360); this normalizes to (-180, 180]
    (the common chemistry convention).
    """
    pos = _positions(atoms)
    v0 = pos[j] - pos[i]
    v1 = pos[k] - pos[j]
    v2 = pos[l] - pos[k]
    value = float(get_dihedrals(
        np.array([v0]), np.array([v1]), np.array([v2]))[0])
    if value > 180.0:
        value -= 360.0
    return value
