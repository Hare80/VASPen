"""Tests for core.transform — translate / rotate / align."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from vaspen.core.transform import align, rotate, translate


@pytest.fixture
def pair() -> Atoms:
    """Two atoms 1 Å apart along x; no cell."""
    return Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_translate_does_not_modify_input(pair):
    original = pair.get_positions().copy()
    out = translate(pair, [1.0, 2.0, 3.0])
    assert np.allclose(pair.get_positions(), original)
    assert np.allclose(out.get_positions()[0], [1.0, 2.0, 3.0])
    assert np.allclose(out.get_positions()[1], [2.0, 2.0, 3.0])


def test_translate_selection_only(pair):
    out = translate(pair, [0.0, 1.0, 0.0], indices=[1])
    assert np.allclose(out.get_positions()[0], [0.0, 0.0, 0.0])
    assert np.allclose(out.get_positions()[1], [1.0, 1.0, 0.0])


def test_translate_bad_index_raises(pair):
    with pytest.raises(ValueError):
        translate(pair, [1.0, 0.0, 0.0], indices=[5])


def test_rotate_90_about_z(pair):
    out = rotate(pair, [0, 0, 0], [0, 0, 1], 90.0, center=[0, 0, 0])
    assert np.allclose(out.get_positions()[0], [0.0, 0.0, 0.0])
    # (1,0,0) rotated +90° about +z → (0,1,0) (right-hand rule)
    assert np.allclose(out.get_positions()[1], [0.0, 1.0, 0.0], atol=1e-10)


def test_rotate_default_center_is_set_centroid(pair):
    # rotating about the centroid of the two atoms keeps it fixed
    out = rotate(pair, [0, 0, 0], [0, 0, 1], 180.0)
    assert np.allclose(out.get_positions().mean(axis=0), [0.5, 0.0, 0.0], atol=1e-10)


def test_rotate_selection_only(pair):
    out = rotate(pair, [0, 0, 0], [0, 0, 1], 90.0, indices=[1], center=[0, 0, 0])
    assert np.allclose(out.get_positions()[0], [0.0, 0.0, 0.0])
    assert np.allclose(out.get_positions()[1], [0.0, 1.0, 0.0], atol=1e-10)


def test_rotate_zero_axis_raises(pair):
    with pytest.raises(ValueError):
        rotate(pair, [0, 0, 0], [0, 0, 0], 90.0)


def test_align_zx_to_xy(pair):
    """a1=z, a2=x → b1=x, b2=y maps (z,x) frame onto (x,y) frame.

    The rotation acts about the set centroid (0.5, 0, 0): atom 1 at
    (1,0,0) is (0.5,0,0) from the center, which maps onto (0,0.5,0).
    """
    out = align(pair, [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0])
    assert np.allclose(out.get_positions()[1], [0.5, 0.5, 0.0], atol=1e-10)
    assert np.allclose(out.get_positions()[0], [0.5, -0.5, 0.0], atol=1e-10)


def test_align_parallel_directions_raise(pair):
    with pytest.raises(ValueError):
        align(pair, [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 1, 0])


def test_rotate_preserves_bond_length(pair):
    out = rotate(pair, [0, 0, 0], [1, 1, 0], 37.5)
    d = out.get_distance(0, 1)
    assert d == pytest.approx(1.0, abs=1e-10)


def test_periodic_translate_wraps():
    atoms = Atoms("H", positions=[[0.9, 0.0, 0.0]],
                  cell=[1.0, 1.0, 1.0], pbc=True)
    out = translate(atoms, [0.5, 0.0, 0.0])
    assert np.allclose(out.get_positions()[0], [0.4, 0.0, 0.0])


def test_periodic_rotate_wraps_only_pbc_axes():
    atoms = Atoms("H", positions=[[0.9, 0.0, 0.0]],
                  cell=[2.0, 2.0, 2.0], pbc=(True, False, False))
    # rotate about z at origin: x wraps (pbc), y/z do not
    out = rotate(atoms, [0, 0, 0], [0, 0, 1], 90.0, center=[0, 0, 0])
    assert np.allclose(out.get_positions()[0], [0.0, 0.9, 0.0], atol=1e-10)


def test_translate_whole_structure_keeps_relative_positions(pair):
    out = translate(pair, [2.0, 0.0, 0.0])
    assert out.get_distance(0, 1) == pytest.approx(1.0)


def test_align_molecule_keeps_geometry():
    h2o = molecule("H2O")
    out = align(h2o, [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0])
    for i in range(len(h2o)):
        for j in range(i + 1, len(h2o)):
            assert out.get_distance(i, j) == pytest.approx(
                h2o.get_distance(i, j), abs=1e-10)
