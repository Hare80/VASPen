"""Tests for core.bonds — connectivity detection and MIC vectors."""

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule

from vaspen.core.bonds import (
    BOND_MAX_LENGTH,
    Bond,
    find_bonds,
    mic_vector,
)


def _cell_pbc(atoms):
    cell = atoms.get_cell().array if atoms.get_cell().rank == 3 else np.eye(3)
    pbc = tuple(atoms.get_pbc()) if atoms.get_pbc().any() else (False, False, False)
    return cell, pbc


def _bonds_of(atoms):
    cell, pbc = _cell_pbc(atoms)
    return find_bonds(
        np.asarray(atoms.get_positions(), dtype=float),
        list(atoms.get_chemical_symbols()), cell, pbc)


def test_bond_invariant_orders_indices():
    b = Bond(3, 1, 2)
    assert (b.i, b.j, b.order) == (1, 3, 2)


def test_water_has_two_oh_bonds():
    bonds = _bonds_of(molecule("H2O"))
    assert len(bonds) == 2
    assert {b.i for b in bonds} | {b.j for b in bonds} == {0, 1, 2}
    assert all(b.order == 1 for b in bonds)


def test_h2_periodic_crosses_boundary_via_mic():
    """Direct H–H 1.5 Å is above the covalent threshold; the MIC image
    at 0.5 Å must be found instead — proof that get_distances MIC is used."""
    atoms = Atoms("H2", positions=[[0.0, 0, 0], [1.5, 0, 0]],
                  cell=[2.0, 2.0, 2.0], pbc=True)
    bonds = _bonds_of(atoms)
    assert len(bonds) == 1
    vec = mic_vector(0, 1, atoms.get_positions(),
                     np.diag([2.0, 2.0, 2.0]), (True, True, True))
    assert np.allclose(vec, [-0.5, 0.0, 0.0])


def test_silicon_bulk_bonds():
    """Primitive Si has 2 atoms bonded via minimum image."""
    si = bulk("Si", a=5.43)
    bonds = _bonds_of(si)
    assert len(bonds) == 1
    assert (bonds[0].i, bonds[0].j) == (0, 1)


def test_barium_oxygen_ionic_contact():
    """Ba–O 2.83 Å: not covalent (capped radii) but metal↔non-metal contact."""
    atoms = Atoms("BaO", positions=[[0.0, 0, 0], [2.83, 0, 0]],
                  cell=[10, 10, 10], pbc=False)
    bonds = _bonds_of(atoms)
    assert len(bonds) == 1
    assert (bonds[0].i, bonds[0].j) == (0, 1)


def test_barium_barium_no_bond_beyond_capped_radius():
    """Radius cap: Ba–Ba at 3.5 Å must NOT bond (uncapped 2.15 Å radii
    would give a 4.6 Å threshold)."""
    atoms = Atoms("Ba2", positions=[[0.0, 0, 0], [3.5, 0, 0]],
                  cell=[10, 10, 10], pbc=False)
    assert _bonds_of(atoms) == []


def test_bond_order_inference():
    """Detected bonds carry an inferred order (1/2/3) by length ratio."""
    ethylene = molecule("C2H4")
    bonds = _bonds_of(ethylene)
    c_c = next(b for b in bonds if {b.i, b.j} == {0, 1})
    assert c_c.order == 2

    co = molecule("CO")
    assert _bonds_of(co)[0].order == 3

    water = molecule("H2O")
    assert all(b.order == 1 for b in _bonds_of(water))


def test_max_bond_length_cutoff():
    atoms = Atoms("NaCl", positions=[[0.0, 0, 0], [BOND_MAX_LENGTH + 0.1, 0, 0]],
                  cell=[20, 20, 20], pbc=False)
    assert _bonds_of(atoms) == []


def test_mic_vector_half_cell_tie_prefers_home_cell():
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    cell = np.diag([2.0, 2.0, 2.0])
    vec = mic_vector(0, 1, positions, cell, (True, True, True))
    # both ±1.0 Å images equally short; prefer endpoint inside home cell
    assert np.allclose(vec, [1.0, 0.0, 0.0])


def test_mic_vector_respects_pbc_flags():
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    cell = np.diag([2.0, 2.0, 2.0])
    vec = mic_vector(0, 1, positions, cell, (False, False, False))
    assert np.allclose(vec, [1.0, 0.0, 0.0])  # no wrapping allowed
