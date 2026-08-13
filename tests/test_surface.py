"""Tests for surface/slab generation and termination enumeration.

Termination counts below pin the REAL pymatgen behavior with NO
symmetry dedup (verified 2026-08-14, pymatgen-core 2026.7.31,
filter_out_sym_slabs=False — user decision: mirror images are
different cleavages and are listed as separate choices). The count
equals the number of clustered atomic planes per oriented unit cell:
SrTiO3 (100) -> 2 (SrO/TiO2 mirror pair), MgO rocksalt (100) -> 2,
GaAs (111) -> 2 (polar Ga/As). Do not "fix" the counts to textbook
guesses — they are empirical.
"""

from collections import Counter

import numpy as np
from ase import Atoms
from ase.build import bulk

from vaspen.core.surface import (
    SlabInfo,
    SurfaceCutter,
    _subscript_formula,
    _surface_compositions,
    supercell_in_plane,
)
from vaspen.core.structure import StructureModel


# ----------------------------------------------------------------------
# Builders (small cells — each slab generation is ~0.1-0.5 s)
# ----------------------------------------------------------------------

def _mgo() -> Atoms:
    """MgO rocksalt primitive cell (2 atoms, a=4.21 Å)."""
    return bulk("MgO", "rocksalt", a=4.21)


def _gaas() -> Atoms:
    """GaAs zincblende primitive cell (2 atoms, a=5.65 Å)."""
    return bulk("GaAs", "zincblende", a=5.65)


# ----------------------------------------------------------------------
# SurfaceCutter.cut / cut_termination
# ----------------------------------------------------------------------

def test_cut_backward_compat_signature_and_result(srtio3):
    """cut() keeps its old signature; returns a periodic slab model."""
    slab = SurfaceCutter(StructureModel(srtio3)).cut((1, 0, 0), 4, 15.0)
    assert isinstance(slab, StructureModel)
    assert slab.n_atoms > 0
    assert tuple(slab.atoms.get_pbc()) == (True, True, True)
    assert slab.atoms.get_cell().rank == 3
    assert StructureModel(slab.atoms).is_periodic


def test_cut_equals_cut_termination_zero(srtio3):
    cutter = SurfaceCutter(StructureModel(srtio3))
    a = cutter.cut((1, 0, 0), 4, 15.0)
    b = cutter.cut_termination((1, 0, 0), 4, 15.0, termination=0)
    assert np.allclose(a.atoms.get_positions(), b.atoms.get_positions())


def test_cut_termination_index_matches_slabs(gaas):
    cutter = SurfaceCutter(StructureModel(gaas))
    chosen = cutter.cut_termination((1, 1, 1), 4, 15.0, termination=1)
    assert np.allclose(chosen.atoms.get_positions(),
                       cutter.slabs((1, 1, 1), 4, 15.0)[1].atoms.get_positions())


# ----------------------------------------------------------------------
# Termination enumeration (empirical counts, see module docstring)
# ----------------------------------------------------------------------

def test_slabs_srtio3_100_two_mirror_terminations(srtio3):
    """(100) has two cleavages: SrO-top and TiO2-top (mirror pair).

    No symmetry dedup — mirror images are listed as separate choices
    (user decision 2026-08-14).
    """
    slabs = SurfaceCutter(StructureModel(srtio3)).slabs((1, 0, 0), 4, 15.0)
    assert len(slabs) == 2
    assert {(s.top_composition, s.bottom_composition) for s in slabs} == {
        ("SrO", "TiO2"),
        ("TiO2", "SrO"),
    }
    for s in slabs:
        assert isinstance(s, SlabInfo)
        counts = Counter(s.atoms.get_chemical_symbols())
        # stoichiometry per slab (atom count is NOT pinned: pymatgen's
        # float ceil makes the same inputs yield 20 or 25 atoms)
        assert counts["Sr"] == counts["Ti"]
        assert counts["O"] == 3 * counts["Sr"]


def test_slabs_mgo_100_two_terminations():
    """Rocksalt (100) primitive cell: mixed MgO/MgO and O/Mg cleavages."""
    slabs = SurfaceCutter(StructureModel(_mgo())).slabs((1, 0, 0), 4, 15.0)
    assert len(slabs) == 2
    assert {(s.top_composition, s.bottom_composition) for s in slabs} == {
        ("MgO", "MgO"),
        ("O", "Mg"),
    }


def test_slabs_gaas_111_two_polar_terminations():
    """GaAs (111) cleaves polar: Ga-terminated and As-terminated slabs."""
    slabs = SurfaceCutter(StructureModel(_gaas())).slabs((1, 1, 1), 4, 15.0)
    assert len(slabs) == 2
    assert {(s.top_composition, s.bottom_composition) for s in slabs} == {
        ("Ga", "As"),
        ("As", "Ga"),
    }


def test_slabs_magmom_preserved(srtio3):
    """Initial magnetic moments round-trip ASE -> pymatgen -> ASE."""
    srtio3.set_initial_magnetic_moments([1, -1, 2, -2, 0.5])
    slabs = SurfaceCutter(StructureModel(srtio3)).slabs((1, 0, 0), 4, 15.0)
    expected = sorted([1, -1, 2, -2, 0.5] * (slabs[0].n_atoms // 5))
    got = sorted(slabs[0].atoms.get_initial_magnetic_moments())
    assert np.allclose(got, expected)


def test_slabs_all_periodic_full_rank(srtio3):
    slabs = SurfaceCutter(StructureModel(srtio3)).slabs((1, 0, 0), 4, 15.0)
    for s in slabs:
        assert tuple(s.atoms.get_pbc()) == (True, True, True)
        assert s.atoms.get_cell().rank == 3


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def test_surface_compositions_layers():
    """Top/bottom layer composition from projection along c."""
    atoms = Atoms(
        "SrOTiO",
        cell=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 20.0]],
        pbc=True,
        positions=[
            [1.5, 1.5, 0.0],    # Sr  (bottom plane)
            [1.5, 1.5, 2.0],    # O   (layer 2)
            [1.5, 1.5, 4.0],    # Ti  (layer 3)
            [1.5, 1.5, 18.0],   # O   (top plane)
        ],
    )
    assert _surface_compositions(atoms) == ("O", "Sr")


def test_surface_compositions_coplanar_merge():
    """Co-planar atoms of the same layer merge into one formula."""
    atoms = Atoms(
        "TiO2",
        cell=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 20.0]],
        pbc=True,
        positions=[
            [1.5, 1.5, 0.0],    # Ti (bottom)
            [1.5, 1.5, 18.0],   # O  (top plane)
            [1.0, 1.0, 18.0],   # O  (same top plane)
        ],
    )
    assert _surface_compositions(atoms) == ("O2", "Ti")


def test_subscript_formula():
    assert _subscript_formula("TiO2") == "TiO₂"
    assert _subscript_formula("Ga2") == "Ga₂"
    assert _subscript_formula("SrO") == "SrO"


def test_supercell_in_plane():
    """Repeat scales a/b in the surface plane; the c axis is untouched."""
    atoms = Atoms(
        "SrO",
        cell=[[3.905, 0.0, 0.0], [0.0, 3.905, 0.0], [0.0, 0.0, 25.0]],
        pbc=True,
        positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    )
    expanded = supercell_in_plane(atoms, 2, 3)
    assert len(expanded) == 2 * 3 * len(atoms)
    lengths = expanded.get_cell().lengths()
    assert np.isclose(lengths[0], 2 * 3.905)
    assert np.isclose(lengths[1], 3 * 3.905)
    assert np.isclose(lengths[2], 25.0)  # vacuum direction untouched
    assert tuple(expanded.get_pbc()) == (True, True, True)
    # repeated fractional coordinates: 2 base sites x 6 in-plane cells
    assert len({tuple(np.round(p, 6)) for p in expanded.get_scaled_positions()}) == 12
