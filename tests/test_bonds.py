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


def test_peroxide_oo_bond():
    """H2O2: the long O–O single bond (ratio 1.11) must be detected —
    regression test for the tolerance bump to 1.15."""
    atoms = molecule("H2O2")
    bonds = _bonds_of(atoms)
    oo = [b for b in bonds if {b.i, b.j} == {0, 1}]
    assert len(oo) == 1
    assert oo[0].order == 1  # long single bond
    assert len(bonds) == 3    # O–O + 2×O–H


def test_water_dimer_hydrogen_bonds_not_bonded():
    """H···O hydrogen bonds (~1.9 Å, ratio ~2.0) must stay unbonded
    even with the relaxed tolerance."""
    w1 = molecule("H2O")
    w2 = molecule("H2O").copy()
    w2.translate([0.0, 0.0, 2.5])
    dimer = w1 + w2
    dimer.set_cell([20, 20, 20])
    dimer.set_pbc(False)
    bonds = _bonds_of(dimer)
    # two intra-molecular O–H bonds per molecule, NO intermolecular contacts
    assert len(bonds) == 4
    assert all(({b.i, b.j} <= {0, 1, 2}) or ({b.i, b.j} <= {3, 4, 5})
               for b in bonds)


def test_h2_bond_detected():
    """H–H 0.74 Å: the detection-radius floor keeps the hydrogen bond
    inside the threshold (H's tabulated radius 0.31 is too small)."""
    atoms = molecule("H2")
    bonds = _bonds_of(atoms)
    assert len(bonds) == 1
    assert bonds[0].order == 1


def test_fe_bcc_only_nearest_neighbors(fe_bcc_2x2x2):
    """bcc Fe: only the 8 nearest neighbors per atom bond (64 pairs in
    the 2×2×2 cell) — the second shell at 2.87 Å must NOT bond even
    though it is inside the covalent tolerance."""
    from ase.geometry import get_distances

    atoms = fe_bcc_2x2x2
    cell, pbc = _cell_pbc(atoms)
    bonds = find_bonds(
        np.asarray(atoms.get_positions(), dtype=float),
        list(atoms.get_chemical_symbols()), cell, pbc)
    assert len(bonds) == 64
    # every bond is a nearest neighbor (2.4855 Å, MIC)
    _vecs, dists = get_distances(
        np.asarray(atoms.get_positions(), dtype=float),
        np.asarray(atoms.get_positions(), dtype=float), cell=cell, pbc=pbc)
    for b in bonds:
        assert dists[b.i, b.j] < 2.6


def test_al_fcc_bonds_via_shell_gap():
    """fcc Al: nearest neighbors at 2.86 Å bond (mid-gap cutoff ~3.45)
    even though they sit just below the bcc-Fe second shell."""
    from ase.build import bulk

    al = bulk("Al", "fcc", a=4.05, cubic=True)
    cell, pbc = _cell_pbc(al)
    bonds = find_bonds(
        np.asarray(al.get_positions(), dtype=float),
        list(al.get_chemical_symbols()), cell, pbc)
    # 4-atom conventional cell: each corner bonds its 3 face-center
    # atoms (periodic images of the same pair are not double-counted)
    assert len(bonds) == 6


def test_metal_pair_in_molecule_keeps_tolerance_rule():
    """Non-periodic metal–metal pairs skip the shell rule (molecules
    use the pure tolerance path)."""
    atoms = Atoms("Cu2", positions=[[0.0, 0, 0], [2.6, 0, 0]],
                  cell=[20, 20, 20], pbc=False)
    bonds = _bonds_of(atoms)
    assert len(bonds) == 1


# ----------------------------------------------------------------------
# Code-review regression test (2026-08-14, user-found surface hang)
# ----------------------------------------------------------------------

def test_find_bonds_caps_large_structures(monkeypatch):
    """Auto detection is O(N²) with a 27-image distance matrix — beyond
    the cap it must return [] instead of allocating gigabytes (a 10×10
    slab supercell froze the surface dialog for minutes / MemoryError)."""
    import vaspen.core.bonds as bonds_mod

    atoms = Atoms("H6", positions=[(i, 0, 0) for i in range(6)],
                  cell=[20, 20, 20], pbc=True)
    cell, pbc = _cell_pbc(atoms)
    monkeypatch.setattr(bonds_mod, "MAX_AUTO_BOND_ATOMS", 5)
    assert find_bonds(atoms.get_positions(), atoms.get_chemical_symbols(),
                      cell, pbc) == []
    monkeypatch.setattr(bonds_mod, "MAX_AUTO_BOND_ATOMS", 6)
    result = find_bonds(atoms.get_positions(), atoms.get_chemical_symbols(),
                        cell, pbc)
    assert isinstance(result, list)  # just below the cap runs normally
