"""Tests for core.measure — distance / angle / dihedral (direct vectors)."""

import numpy as np
from ase import Atoms
from ase.build import molecule

from vaspen.core.measure import angle, dihedral, distance


def test_water_geometry():
    water = molecule("H2O")
    # O–H and H–H distances (ASE H2O geometry)
    assert np.isclose(distance(water, 0, 1), 0.9686, atol=1e-3)
    assert np.isclose(distance(water, 1, 2), 1.5265, atol=1e-3)
    # H–O–H angle ≈ 104°
    assert np.isclose(angle(water, 1, 0, 2), 104.0, atol=0.5)


def test_distance_periodic_uses_direct_vectors():
    """Displayed atoms at 1.5 Å measure 1.5 Å even though the periodic
    image is 0.5 Å away (settled: measure the DISPLAYED atoms)."""
    atoms = Atoms("H2", positions=[[0.0, 0, 0], [1.5, 0, 0]],
                  cell=[2.0, 2.0, 2.0], pbc=True)
    assert np.isclose(distance(atoms, 0, 1), 1.5, atol=1e-9)


def test_angle_right_angle():
    atoms = Atoms("H3", positions=[[1, 0, 0], [0, 0, 0], [0, 1, 0]])
    assert np.isclose(angle(atoms, 0, 1, 2), 90.0, atol=1e-6)


def test_angle_periodic_uses_direct_vectors():
    """The angle uses the raw displacement to the DISPLAYED atom, not
    the minimum-image vector across the boundary."""
    # direct v1 = pos0 − pos1 = (0.6, 0.5, 0); the MIC image would be
    # (−0.4, 0.5, 0) → a different angle
    atoms = Atoms("H3", positions=[[0.6, 0.5, 0], [0, 0, 0], [0, 1, 0]],
                  cell=[1.0, 10.0, 10.0], pbc=(True, False, False))
    v1 = np.array([0.6, 0.5, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    expected = np.rad2deg(np.arccos(v1 @ v2 / np.linalg.norm(v1)))
    assert np.isclose(angle(atoms, 0, 1, 2), expected, atol=1e-6)
    # and it differs from the minimum-image value (~38.7°)
    assert not np.isclose(angle(atoms, 0, 1, 2), 38.66, atol=0.5)


def test_dihedral_perpendicular():
    i = np.array([0.0, 0, 0])
    j = np.array([1.0, 0, 0])
    k = np.array([2.0, 1.0, 0])
    atoms = Atoms("C4", positions=[i, j, k, k + [0, 0, 1.0]])
    assert np.isclose(dihedral(atoms, 0, 1, 2, 3), 90.0, atol=1e-6)


def test_dihedral_sign_normalized():
    """ASE returns [0, 360); measure.dihedral normalizes to (-180, 180]."""
    i = np.array([0.0, 0, 0])
    j = np.array([1.0, 0, 0])
    k = np.array([2.0, 1.0, 0])
    atoms = Atoms("C4", positions=[i, j, k, k - [0, 0, 1.0]])
    assert np.isclose(dihedral(atoms, 0, 1, 2, 3), -90.0, atol=1e-6)


def test_dihedral_planar_cis():
    i = np.array([0.0, 0, 0])
    j = np.array([1.0, 0, 0])
    k = np.array([2.0, 1.0, 0])
    atoms = Atoms("C4", positions=[i, j, k, k + [-1.0, 0, 0]])
    assert np.isclose(dihedral(atoms, 0, 1, 2, 3), 0.0, atol=1e-6)


def test_h2o2_dihedral_within_range():
    # ASE H2O2 atom order is O, O, H, H — the H–O–O–H dihedral uses 3-1-2-0
    per = molecule("H2O2")
    value = dihedral(per, 3, 1, 2, 0)
    assert -180.0 < value <= 180.0
    assert np.isclose(abs(value), 66.957, atol=1e-2)
