"""Tests for core.symmetry — spglib wrappers."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from vaspen.core.symmetry import SymmetryInfo, analyze, symmetrize


def test_silicon_space_group(si_bulk):
    info = analyze(si_bulk)
    assert isinstance(info, SymmetryInfo)
    assert info.number == 227
    assert info.international == "Fd-3m"
    assert info.pointgroup == "m-3m"


def test_molecule_point_group(water_molecule):
    info = analyze(water_molecule)
    assert isinstance(info, SymmetryInfo)
    assert info.kind == "point"
    assert info.international == "C2v"
    assert info.number == 2  # rotational symmetry number σ = 2


def test_molecule_in_box_without_pbc_is_still_a_point_group():
    # A full-rank cell but pbc all False is a molecule (app policy)
    atoms = molecule("H2O").copy()
    atoms.set_cell([20, 20, 20])
    atoms.set_pbc(False)
    info = analyze(atoms)
    assert info is not None and info.kind == "point"
    assert info.international == "C2v"


def test_methane_point_group_td():
    info = analyze(molecule("CH4"))
    assert info is not None and info.kind == "point"
    assert info.international == "Td"


def test_ammonia_point_group_c3v():
    info = analyze(molecule("NH3"))
    assert info is not None and info.kind == "point"
    assert info.international == "C3v"


def test_perturbed_silicon_keeps_group_at_low_precision():
    """Small noise below symprec must not change the space group."""
    from ase.build import bulk
    si = bulk("Si", a=5.43)
    si.positions[1] += [0.0001, 0.0, 0.0]
    info = analyze(si, symprec=1e-3)
    assert info is not None and info.number == 227


def test_symmetrize_silicon(si_bulk):
    out = symmetrize(si_bulk)
    assert out is not None
    assert out.get_cell().rank == 3
    assert all(out.get_pbc())
    # standardize_cell(to_primitive=False): the 2-atom primitive fcc
    # input expands to the 8-atom conventional cubic cell
    assert len(out) == 8
    assert np.allclose(out.get_cell().lengths(), 5.43, atol=1e-8)
    assert set(out.get_chemical_symbols()) == set(si_bulk.get_chemical_symbols())


def test_symmetrize_molecule_keeps_point_group(water_molecule):
    out = symmetrize(water_molecule)
    assert out is not None
    assert len(out) == len(water_molecule)
    assert set(out.get_chemical_symbols()) == {"O", "H"}
    # still C2v after symmetrization
    info = analyze(out)
    assert info is not None and info.kind == "point"
    assert info.international == "C2v"
    # the two O–H bonds are equalized
    d1 = out.get_distance(0, 1)
    d2 = out.get_distance(0, 2)
    assert d1 == pytest.approx(d2, abs=1e-6)


def test_symmetrize_result_is_analysis_consistent(si_bulk):
    out = symmetrize(si_bulk)
    assert out is not None
    info = analyze(out)
    assert info is not None and info.number == 227


def test_symmetrize_does_not_modify_input(si_bulk):
    cell_before = si_bulk.get_cell().array.copy()
    symmetrize(si_bulk)
    assert (si_bulk.get_cell().array == cell_before).all()


def test_analyze_cubic_rocksalt():
    # 8-atom conventional rocksalt cell (Na at fcc sites, Cl at edges/center)
    na = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    cl = [[0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
    atoms = Atoms(
        "Na4Cl4",
        positions=[[p * 5.64 for p in v] for v in na + cl],
        cell=[5.64, 5.64, 5.64],
        pbc=True,
    )
    info = analyze(atoms)
    assert info is not None
    assert info.number == 225
    assert info.international == "Fm-3m"
