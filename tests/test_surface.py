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
import pytest
from ase import Atoms
from ase.build import bulk

from vaspen.core.surface import (
    SlabInfo,
    SurfaceCutter,
    _subscript_formula,
    _surface_compositions,
    rebox_slab,
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
    """Rocksalt (100): alternating pure-species planes (the physical
    (100) layers are O and Mg planes, not mixed ones) — two mirror
    cleavages with pure O / pure Mg faces."""
    slabs = SurfaceCutter(StructureModel(_mgo())).slabs((1, 0, 0), 4, 15.0)
    assert len(slabs) == 2
    assert {(s.top_composition, s.bottom_composition) for s in slabs} == {
        ("O", "Mg"),
        ("Mg", "O"),
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


# ----------------------------------------------------------------------
# Code-review regression tests (2026-08-14)
# ----------------------------------------------------------------------

def test_zero_miller_raises_value_error(si_bulk):
    """(0,0,0) must raise the documented ValueError (not ZeroDivisionError)."""
    cutter = SurfaceCutter(StructureModel(si_bulk.copy()))
    # both paths must raise the documented ValueError (not
    # ZeroDivisionError / OverflowError from 1/norm(0))
    with pytest.raises(ValueError):
        cutter.slabs((0, 0, 0), 4, 15.0)
    with pytest.raises(ValueError):
        cutter.cut_with_thickness((0, 0, 0), 10.0, 15.0)


# ----------------------------------------------------------------------
# Conventional slab presentation (2026-08-15, user request)
# ----------------------------------------------------------------------

def test_slab_vacuum_perpendicular_to_ab():
    """The vacuum direction (c) must be exactly perpendicular to the
    in-plane vectors — pymatgen's raw box was sheared for cubic (111)."""
    cu = Atoms("Cu4", cell=[3.61, 3.61, 3.61], pbc=True,
               positions=[[0, 0, 0], [0, 1.805, 1.805],
                          [1.805, 0, 1.805], [1.805, 1.805, 0]])
    for miller, gamma in (((1, 1, 1), 60.0), ((1, 0, 0), 90.0)):
        slab = SurfaceCutter(StructureModel(cu)).slabs(miller, 4, 15.0)[0]
        cell = np.asarray(slab.atoms.get_cell().array)
        a, b, c = cell
        assert abs(float(c @ a)) < 1e-6
        assert abs(float(c @ b)) < 1e-6
        assert abs(a[2]) < 1e-6 and abs(b[2]) < 1e-6  # a,b in the xy-plane
        got_gamma = np.degrees(np.arccos(float(a @ b)
                                         / (np.linalg.norm(a) * np.linalg.norm(b))))
        assert got_gamma == pytest.approx(gamma, abs=0.5)
        assert a[1] == pytest.approx(0.0, abs=1e-6)  # a ∥ x (standard orientation)


def test_slab_vacuum_exact_and_centered():
    """c_z = slab thickness + requested vacuum; slab centered along z."""
    cu = Atoms("Cu4", cell=[3.61, 3.61, 3.61], pbc=True,
               positions=[[0, 0, 0], [0, 1.805, 1.805],
                          [1.805, 0, 1.805], [1.805, 1.805, 0]])
    slab = SurfaceCutter(StructureModel(cu)).slabs((1, 1, 1), 4, 15.0)[0]
    z = slab.atoms.positions[:, 2]
    cell = np.asarray(slab.atoms.get_cell().array)
    assert (z.max() - z.min()) + 15.0 == pytest.approx(cell[2, 2], abs=1e-6)
    assert (z.min() + z.max()) / 2 == pytest.approx(cell[2, 2] / 2, abs=1e-6)


# ----------------------------------------------------------------------
# Re-box Slab (standalone feature, Tools menu; 2026-08-15)


def _cu111_slab() -> Atoms:
    """36-atom 4-layer Cu(111) slab with vacuum (examples/Cu_111_slab.vasp)."""
    from ase.io import read

    from pathlib import Path
    return read(Path(__file__).parent.parent / "examples" / "Cu_111_slab.vasp")


def test_rebox_slab_contiguous_and_vacuum():
    """rebox_slab: contiguous 4-layer slab, exact vacuum, centered,
    in-plane cell untouched."""
    out = rebox_slab(_cu111_slab(), 15.0)
    assert len(out) == 36
    z = np.sort(out.positions[:, 2])
    levels = np.unique(np.round(z, 3))
    assert len(levels) == 4  # contiguous 4-layer slab, nothing detached
    gaps = np.diff(levels)
    assert np.allclose(gaps, gaps[0], atol=1e-3)  # uniform layer spacing
    cell = np.asarray(out.get_cell().array)
    assert (z.max() - z.min()) + 15.0 == pytest.approx(cell[2, 2], abs=1e-6)
    assert (z.min() + z.max()) / 2 == pytest.approx(cell[2, 2] / 2, abs=1e-6)
    # in-plane cell untouched by the re-box
    assert np.allclose(cell[:2], np.asarray(_cu111_slab().get_cell().array)[:2],
                       atol=1e-6)


def test_rebox_slab_preserves_atom_set():
    """Re-boxing only translates atoms along c — the structure (minimum-
    image distance multiset, invariant to lattice translations and rigid
    shifts) is identical to the input."""
    from ase.geometry import get_distances

    inp = _cu111_slab()
    out = rebox_slab(inp, 15.0)
    cell = np.asarray(inp.get_cell().array)
    pbc = inp.get_pbc()
    _v, din = get_distances(inp.positions, inp.positions, cell=cell, pbc=pbc)
    _v, dout = get_distances(out.positions, out.positions, cell=cell, pbc=pbc)
    assert np.allclose(np.sort(din.ravel()), np.sort(dout.ravel()), atol=1e-4)


def test_rebox_slab_keeps_atom_order():
    """The atom order is preserved 1:1 (fixed flags / magmoms map)."""
    inp = _cu111_slab()
    out = rebox_slab(inp, 15.0)
    assert out.get_chemical_symbols() == inp.get_chemical_symbols()


# ----------------------------------------------------------------------
# Incremental termination computation (2026-08-15, performance work)
# ----------------------------------------------------------------------

def test_iter_slabs_matches_slabs_and_order_hint():
    """iter_slabs yields the same list as slabs(); the compute-order
    hint jumps the queue while slot indices stay canonical."""
    from vaspen.core.surface import (
        _ComputeOrder,
        iter_slabs,
        slab_count,
    )

    cu = Atoms("Cu4", cell=[3.61, 3.61, 3.61], pbc=True,
               positions=[[0, 0, 0], [0, 1.805, 1.805],
                          [1.805, 0, 1.805], [1.805, 1.805, 0]])
    full = [info for _i, info in iter_slabs(cu, (1, 1, 1), 4, 15.0)]
    assert slab_count(cu, (1, 1, 1), 4, 15.0) == len(full)
    ref = SurfaceCutter(StructureModel(cu)).slabs((1, 1, 1), 4, 15.0)
    assert len(full) == len(ref)
    for a, b in zip(full, ref):
        assert np.allclose(a.atoms.positions, b.atoms.positions)
        assert a.top_composition == b.top_composition

    # priority: the clicked index is computed next; every slot arrives
    # exactly once, in the canonical positions (rocksalt (100) has two
    # mirror terminations)
    full_mgo = [info for _i, info in iter_slabs(_mgo(), (1, 0, 0), 4, 15.0)]
    assert len(full_mgo) == 2
    order = _ComputeOrder()
    gen = iter_slabs(_mgo(), (1, 0, 0), 4, 15.0, order=order)
    first_idx, _first = next(gen)
    assert first_idx == 0
    order.priority = 1
    second_idx, _second = next(gen)
    assert second_idx == 1
    rest = [idx for idx, _info in gen]
    assert sorted(rest + [first_idx, second_idx]) == [0, 1]


def test_unwrap_layers_folds_wrapped_slab():
    """Atoms pushed across the periodic c boundary fold back into one
    contiguous block; the structure is unchanged (mic distances)."""
    from ase.geometry import get_distances

    from vaspen.core.surface import _unwrap_layers

    inp = _cu111_slab()
    wrapped = inp.copy()
    frac = wrapped.get_scaled_positions()
    frac[0:5, 2] += 1.0  # push five atoms of the bottom layer across c
    wrapped.set_scaled_positions(frac)
    out = _unwrap_layers(wrapped)
    cell = np.asarray(inp.get_cell().array)
    pbc = inp.get_pbc()
    _v, d1 = get_distances(inp.positions, inp.positions, cell=cell, pbc=pbc)
    _v, d2 = get_distances(out.positions, out.positions, cell=cell, pbc=pbc)
    assert np.allclose(np.sort(d1.ravel()), np.sort(d2.ravel()), atol=1e-4)
    z = np.sort(out.positions[:, 2])
    assert len(np.unique(np.round(z, 3))) == 4  # contiguous layers again
