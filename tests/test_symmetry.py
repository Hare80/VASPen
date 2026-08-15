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


# ----------------------------------------------------------------------
# Conventional / primitive cell conversion (2026-08-15, user request)
# ----------------------------------------------------------------------

def _cu_conventional() -> Atoms:
    """Cu fcc conventional cell (4 atoms, a=3.61 Å)."""
    a = 3.61
    return Atoms(
        "Cu4",
        cell=[a, a, a],
        pbc=True,
        positions=[[0, 0, 0], [0, a / 2, a / 2],
                   [a / 2, 0, a / 2], [a / 2, a / 2, 0]],
    )


def test_symmetrize_primitive_cu_standard_orientation():
    """The primitive of fcc Cu: 1 atom, a=b=c=3.61/√2, 60° angles, in
    the standard orientation (a ∥ x, b in the xy-plane, c along +z)."""
    from ase.geometry import cell_to_cellpar

    atoms = symmetrize(_cu_conventional(), cell_type="primitive")
    assert atoms is not None
    assert len(atoms) == 1
    cell = np.asarray(atoms.get_cell().array)
    params = cell_to_cellpar(cell)
    assert np.allclose(params[:3], 3.61 / np.sqrt(2), atol=1e-3)
    assert np.allclose(params[3:], 60.0, atol=1e-3)
    # standard presentation: a ∥ x, b in the xy-plane, c completes
    # the rhombohedral cell with a positive z-component (c itself is
    # NOT along z for a 60° rhombohedron)
    assert np.allclose(cell[0, 1:], 0.0, atol=1e-6)
    assert abs(cell[1, 2]) < 1e-6
    assert cell[0, 0] > 0 and cell[1, 1] > 0 and cell[2, 2] > 0


def test_symmetrize_primitive_matches_reference_file(cu_primitive_standard):
    """Loose reference check against the standard-orientation primitive
    (a = 3.61/√2, 60° rhombohedron, 1 atom)."""
    ref = cu_primitive_standard
    atoms = symmetrize(_cu_conventional(), cell_type="primitive")
    assert len(ref) == 1
    assert np.allclose(np.sort(np.asarray(ref.get_cell().array), axis=0),
                       np.sort(np.asarray(atoms.get_cell().array), axis=0),
                       atol=1e-3)
    assert ref.get_chemical_symbols() == atoms.get_chemical_symbols()


def test_symmetrize_conventional_unchanged_for_cu():
    """The default path still returns the conventional cell."""
    atoms = symmetrize(_cu_conventional(), cell_type="conventional")
    assert atoms is not None
    assert len(atoms) == 4
    assert np.allclose(atoms.get_cell().array, np.eye(3) * 3.61, atol=1e-6)


def test_symmetrize_bcc_supercell_primitive_vs_conventional(
        tmp_path, fe_bcc_2x2x2):
    """Fe bcc 2×2×2 (16 atoms): conventional → 2 atoms bcc,
    primitive → 1 atom with a = a_conv·√3/2."""
    from ase.geometry import cell_to_cellpar

    fe = fe_bcc_2x2x2
    conv = symmetrize(fe, cell_type="conventional")
    prim = symmetrize(fe, cell_type="primitive")
    assert conv is not None and len(conv) == 2
    assert prim is not None and len(prim) == 1
    params = cell_to_cellpar(np.asarray(prim.get_cell().array))
    # bcc primitive: a = a_conv·√3/2, angles ≈ 109.47°
    assert np.allclose(params[:3], np.linalg.norm(fe.get_cell().array[0]) * 0.5
                       * np.sqrt(3) / 2, atol=0.02)
