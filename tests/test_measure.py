"""Tests for core.measure — distance / angle / dihedral (MIC-aware)."""

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


def test_distance_mic_across_boundary():
    """Direct H–H 1.5 Å, but the periodic image is 0.5 Å away."""
    atoms = Atoms("H2", positions=[[0.0, 0, 0], [1.5, 0, 0]],
                  cell=[2.0, 2.0, 2.0], pbc=True)
    assert np.isclose(distance(atoms, 0, 1), 0.5, atol=1e-9)


def test_angle_right_angle():
    atoms = Atoms("H3", positions=[[1, 0, 0], [0, 0, 0], [0, 1, 0]])
    assert np.isclose(angle(atoms, 0, 1, 2), 90.0, atol=1e-6)


def test_angle_mic_across_boundary():
    """The angle vertex's neighbor across the cell boundary must use
    the minimum-image vector (not the raw displacement)."""
    atoms = Atoms("H3", positions=[[0.9, 0, 0], [0, 0, 0], [0, 1, 0]],
                  cell=[1.0, 10.0, 10.0], pbc=(True, False, False))
    # atom 0's nearest image of (0.9,0,0) is (-0.1,0,0) → angle = 90°
    assert np.isclose(angle(atoms, 0, 1, 2), 90.0, atol=1e-6)


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
