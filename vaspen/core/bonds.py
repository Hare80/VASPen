"""Chemical bond detection for structures (pure functions, no Qt).

Bond connectivity lives here — not in the viewport — so the data model
(StructureModel) can own a persistent bond list. Rules (auto-connectivity,
materials convention):

- Covalent bond when the interatomic distance is below the sum of the
  covalent radii times a tolerance (radii capped so heavy metals like Ba
  do not bond to everything).
- Ionic contact between metal/non-metal pairs within the maximum length.
- Distances are minimum-image (periodic boundary aware).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import get_distances

# Bond detection constants
BOND_TOLERANCE = 1.07   # covalent bond if d < (r1 + r2) * tolerance
BOND_MAX_LENGTH = 3.0   # global maximum bond / contact length
MAX_BOND_RADIUS = 1.35  # cap covalent radius used for bond detection
                        # (heavy metals like Ba=2.15 would otherwise bond
                        # to everything)

# Pauling electronegativity for non-metal detection (ionic-contact rule).
# Pairs metal↔non-metal within BOND_MAX_LENGTH are drawn as contacts
# (e.g. Ba–O 2.83 Å in perovskites), while metal–metal pairs are not.
ELECTRONEGATIVITY: dict[str, float] = {
    "H": 2.20, "He": 0.0, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44,
    "F": 3.98, "Ne": 0.0, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16,
    "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3.00,
    "Te": 2.10, "I": 2.66, "Xe": 2.60, "At": 2.20,
}
EN_NONMETAL_THRESHOLD = 2.0


@dataclass(frozen=True)
class Bond:
    """A chemical bond between atom indices ``i`` and ``j`` (invariant: i < j).

    Args:
        i, j: Atom indices in the parent structure.
        order: 1 = single, 2 = double, 3 = triple, 4 = aromatic.
    """

    i: int
    j: int
    order: int = 1

    def __post_init__(self) -> None:
        if self.i > self.j:
            i, j = self.i, self.j
            object.__setattr__(self, "i", j)
            object.__setattr__(self, "j", i)


def _is_nonmetal(symbol: str) -> bool:
    """True if the element is a non-metal (Pauling EN ≥ 2.0).

    Unknown elements default to metallic behavior (EN < 2.0).
    """
    return ELECTRONEGATIVITY.get(symbol, 1.5) >= EN_NONMETAL_THRESHOLD


def mic_vector(
    i: int,
    j: int,
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: tuple[bool, bool, bool],
) -> np.ndarray:
    """Minimum-image vector from atom i to atom j for bond DRAWING.

    Enumerates all 27 lattice translations and picks the shortest.
    On exact half-cell ties (e.g. Ti–O = 2.0 Å in a 4.0 Å cell), both
    images are equally short — prefer the one whose endpoint lies
    INSIDE the home cell, so the bond connects to the displayed atom
    instead of its periodic image outside the cell (which reads as a
    bond "to the boundary").
    """
    d = positions[j] - positions[i]
    inv_cell = np.linalg.inv(cell)

    best_v: np.ndarray | None = None
    best_len = float("inf")
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            for n3 in (-1, 0, 1):
                # Only allow shifts along periodic directions
                if (n1 != 0 and not pbc[0]) or \
                   (n2 != 0 and not pbc[1]) or \
                   (n3 != 0 and not pbc[2]):
                    continue
                shift = n1 * cell[0] + n2 * cell[1] + n3 * cell[2]
                v = d + shift
                length = float(np.linalg.norm(v))
                if length < best_len - 1e-9:
                    best_v, best_len = v, length
                elif abs(length - best_len) <= 1e-9:
                    # Tie — prefer endpoint inside the home cell
                    frac = (positions[i] + v) @ inv_cell
                    if np.all((frac >= 0.0) & (frac < 1.0)):
                        best_v, best_len = v, length
    return np.asarray(best_v, dtype=float)


def find_bonds(
    positions: np.ndarray,
    symbols: list[str],
    cell: np.ndarray,
    pbc: tuple[bool, bool, bool],
) -> list[Bond]:
    """Find covalent bonds / ionic contacts between atoms (periodic-aware).

    Args:
        positions: Cartesian positions (N×3).
        symbols: Element symbols (N).
        cell: 3×3 cell matrix.
        pbc: Periodic boundary flags.

    Returns:
        List of :class:`Bond` (single order) in (i, j) pair order.
    """
    n = len(positions)
    if n == 0:
        return []

    radii = np.array([
        min(covalent_radii[atomic_numbers[s]], MAX_BOND_RADIUS)
        for s in symbols
    ])
    nonmetals = [_is_nonmetal(s) for s in symbols]

    # Pairwise distances with minimum-image convention.
    # NOTE: ASE ≥3.29 returns (vectors (N,M,3), distances (N,M)) — order
    # swapped relative to older versions.
    _vecs, dists = get_distances(positions, positions, cell=cell, pbc=pbc)

    bonds: list[Bond] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if d <= 0.3 or d > BOND_MAX_LENGTH:
                continue
            # Rule 1: covalent bond
            covalent = d < (radii[i] + radii[j]) * BOND_TOLERANCE
            # Rule 2: ionic contact between metal and non-metal
            ionic = nonmetals[i] != nonmetals[j]
            if covalent or ionic:
                bonds.append(Bond(i, j, 1))
    return bonds
