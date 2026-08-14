"""Tests for vaspen.core.neb — NEB interpolation and distance metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from vaspen.core.neb import (
    constraints_to_fixed_flags,
    detect_order_mismatch,
    interpolate_idpp,
    interpolate_neb,
    neb_distance,
    pbc_wrap,
    suggest_n_images,
    ws_minimal_image,
)
from vaspen.core.file_io import FileIO

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples" / "neb_vacancy_hop"


def _cubic(atoms: Atoms, a: float = 10.0) -> Atoms:
    """Put atoms in a cubic periodic box of side a."""
    atoms.cell = [a, a, a]
    atoms.pbc = True
    return atoms


# ----------------------------------------------------------------------
# pbc_wrap
# ----------------------------------------------------------------------

def test_pbc_wrap_boundaries():
    assert pbc_wrap(-0.5) == pytest.approx(0.5)   # -0.5 wraps to +0.5
    assert pbc_wrap(0.5) == pytest.approx(0.5)    # +0.5 stays
    assert pbc_wrap(0.4) == pytest.approx(0.4)
    assert pbc_wrap(-0.4) == pytest.approx(-0.4)
    assert pbc_wrap(0.0) == pytest.approx(0.0)


def test_pbc_wrap_array():
    result = pbc_wrap(np.array([-0.9, -0.5, 0.0, 0.5, 0.9]))
    assert np.allclose(result, [0.1, 0.5, 0.0, 0.5, -0.1])


# ----------------------------------------------------------------------
# ws_minimal_image
# ----------------------------------------------------------------------

def test_ws_minimal_image_orthogonal():
    cell = np.eye(3) * 10.0
    v = ws_minimal_image(cell, np.array([0.6, 0.0, 0.0]))
    assert np.allclose(v, [-4.0, 0.0, 0.0])  # 0.6 → -0.4 across the wall


def test_ws_minimal_image_inside_cell():
    cell = np.eye(3) * 10.0
    v = ws_minimal_image(cell, np.array([0.1, 0.2, 0.3]))
    assert np.allclose(v, [1.0, 2.0, 3.0])


# ----------------------------------------------------------------------
# neb_distance
# ----------------------------------------------------------------------

def test_neb_distance_zero_for_identical():
    a = _cubic(Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]]))
    b = a.copy()
    assert neb_distance(a, b) == pytest.approx(0.0, abs=1e-12)


def test_neb_distance_single_atom_hop():
    a = _cubic(Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]]))
    b = a.copy()
    b.positions[1] = [4.0, 0, 0]
    assert neb_distance(a, b) == pytest.approx(1.5)


def test_neb_distance_wraps_across_boundary():
    a = _cubic(Atoms("Cu2", positions=[[0.5, 0, 0], [2.5, 0, 0]]))
    b = a.copy()
    b.positions[0] = [9.5, 0, 0]  # 0.5 → -0.5 = 1.0 Å across the wall
    assert neb_distance(a, b) == pytest.approx(1.0)


def test_neb_distance_vacancy_hop_example():
    """The committed example: fcc Cu vacancy hop, a/√2 = 2.5562 Å.

    Cross-checked against the classic reference implementation (exact
    match to ~1e-15): the metric reproduces the published value for
    the same input files.
    """
    ini = FileIO.read(str(EXAMPLES / "initial" / "POSCAR"))
    fin = FileIO.read(str(EXAMPLES / "final" / "POSCAR"))
    a = 3.615
    assert neb_distance(ini, fin) == pytest.approx(a / np.sqrt(2), rel=1e-6)
    assert suggest_n_images(neb_distance(ini, fin)) == 4


def test_neb_distance_validation_errors():
    a = _cubic(Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]]))
    with pytest.raises(ValueError, match="same number of atoms"):
        neb_distance(a, Atoms("Cu", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True))
    with pytest.raises(ValueError, match="same elements"):
        neb_distance(a, _cubic(Atoms("CuAg", positions=[[0, 0, 0], [2.5, 0, 0]])))
    with pytest.raises(ValueError, match="full-rank"):
        neb_distance(a, Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]]))  # no cell
    b = a.copy()
    b.cell = [11, 10, 10]
    with pytest.raises(ValueError, match="cells differ"):
        neb_distance(a, b)


# ----------------------------------------------------------------------
# suggest_n_images
# ----------------------------------------------------------------------

def test_suggest_n_images():
    assert suggest_n_images(2.5562) == 4      # 3.195 → 4
    assert suggest_n_images(0.8) == 1         # exactly one image
    assert suggest_n_images(0.81) == 2
    assert suggest_n_images(0.0) == 1         # never below 1
    assert suggest_n_images(10.0) == 13       # 12.5 → 13


# ----------------------------------------------------------------------
# detect_order_mismatch
# ----------------------------------------------------------------------

def test_detect_order_mismatch_swapped_atoms():
    # Two Cu atoms swapped between ini/fin: file order has a large
    # displacement, reordering gives zero.
    a = _cubic(Atoms("Cu2", positions=[[0.0, 0, 0], [2.0, 0, 0]]))
    b = _cubic(Atoms("Cu2", positions=[[2.0, 0, 0], [0.0, 0, 0]]))
    result = detect_order_mismatch(a, b)
    assert result is not None
    assert result == pytest.approx(0.0, abs=1e-9)


def test_detect_order_mismatch_aligned():
    a = _cubic(Atoms("Cu2", positions=[[0.0, 0, 0], [2.0, 0, 0]]))
    b = a.copy()
    assert detect_order_mismatch(a, b) is None


def test_detect_order_mismatch_small_displacement_not_flagged():
    a = _cubic(Atoms("Cu2", positions=[[0.0, 0, 0], [2.0, 0, 0]]))
    b = a.copy()
    b.positions[1] = [2.1, 0, 0]  # tiny, aligned hop
    assert detect_order_mismatch(a, b) is None


def test_detect_order_mismatch_invalid_pair():
    a = _cubic(Atoms("Cu2", positions=[[0.0, 0, 0], [2.0, 0, 0]]))
    b = Atoms("Ag2", positions=[[0.0, 0, 0], [2.0, 0, 0]], cell=[10, 10, 10], pbc=True)
    assert detect_order_mismatch(a, b) is None


# ----------------------------------------------------------------------
# interpolate_neb
# ----------------------------------------------------------------------

def test_interpolate_neb_frame_count():
    a = _cubic(Atoms("Cu2", positions=[[0.0, 0, 0], [2.5, 0, 0]]))
    b = a.copy()
    b.positions[1] = [4.0, 0, 0]
    frames = interpolate_neb(a, b, 4)
    assert len(frames) == 6  # 4 intermediates + endpoints


def test_interpolate_neb_endpoints_exact():
    a = _cubic(Atoms("Cu2", positions=[[0.0, 0, 0], [2.5, 0, 0]]))
    b = a.copy()
    b.positions[1] = [4.0, 0, 0]
    frames = interpolate_neb(a, b, 3)
    assert np.allclose(frames[0].positions, a.positions)
    assert np.allclose(frames[-1].positions, b.positions)
    # every frame shares the initial lattice
    for f in frames:
        assert np.allclose(f.get_cell(), a.get_cell())


def test_interpolate_neb_linear_midpoint():
    a = _cubic(Atoms("Cu2", positions=[[0.0, 0, 0], [2.5, 0, 0]]))
    b = a.copy()
    b.positions[1] = [4.0, 0, 0]
    frames = interpolate_neb(a, b, 1)
    expected = (a.positions + b.positions) / 2
    assert np.allclose(frames[1].positions, expected)


def test_interpolate_neb_wraps_across_boundary():
    a = _cubic(Atoms("Cu", positions=[[0.5, 0, 0]]))
    b = _cubic(Atoms("Cu", positions=[[9.5, 0, 0]]))
    frames = interpolate_neb(a, b, 1)
    mid = frames[1].get_scaled_positions()
    # 0.05 → 0.95 crosses the wall: midpoint is 0.0, not 0.5
    assert mid[0][0] == pytest.approx(0.0, abs=1e-9)


def test_interpolate_neb_noise_snapped_to_zero():
    """Frames never carry ~1.0 for positions meant to be at 0."""
    a = _cubic(Atoms("Cu2", positions=[[1e-14, 0, 0], [2.5, 0, 0]]))
    b = a.copy()
    frames = interpolate_neb(a, b, 1)
    for f in frames:
        frac = f.get_scaled_positions(wrap=False)
        assert np.all((frac >= 0.0) & (frac < 1.0))


def test_interpolate_neb_validation():
    a = _cubic(Atoms("Cu2", positions=[[0.0, 0, 0], [2.5, 0, 0]]))
    b = a.copy()
    with pytest.raises(ValueError, match="between 1 and 98"):
        interpolate_neb(a, b, 0)
    with pytest.raises(ValueError, match="between 1 and 98"):
        interpolate_neb(a, b, 99)
    with pytest.raises(ValueError, match="same elements"):
        interpolate_neb(a, _cubic(Atoms("CuAg", positions=[[0.0, 0, 0], [2.5, 0, 0]])), 2)
    bad_cell = b.copy()
    bad_cell.cell = [11, 10, 10]
    with pytest.raises(ValueError, match="cells differ"):
        interpolate_neb(a, bad_cell, 2)


def test_interpolate_neb_strips_constraints():
    from ase.constraints import FixAtoms

    a = _cubic(Atoms("Cu2", positions=[[0.0, 0, 0], [2.5, 0, 0]]))
    a.constraints = [FixAtoms(indices=[0])]
    b = a.copy()
    frames = interpolate_neb(a, b, 2)
    for f in frames:
        assert f.constraints == []


def test_interpolate_neb_matches_reference_implementation():
    """Reference comparison for the committed example.

    The classic reference tool (run externally during development)
    produces frames 00-05 whose fractional coordinates agree with
    interpolate_neb(init, final, 4) to ~6e-15 (float precision), and
    its distance metric prints 2.55619101398937 Å for these files —
    identical to neb_distance. This test pins the behavior.
    """
    ini = FileIO.read(str(EXAMPLES / "initial" / "POSCAR"))
    fin = FileIO.read(str(EXAMPLES / "final" / "POSCAR"))
    frames = interpolate_neb(ini, fin, 4)
    assert len(frames) == 6
    assert neb_distance(ini, fin) == pytest.approx(2.55619101398937, rel=1e-12)


# ----------------------------------------------------------------------
# IDPP interpolation
# ----------------------------------------------------------------------

ETHANE = REPO / "examples" / "neb_ethane_rotation"


def _ethane_pair():
    """Ethane in a 10 Å box; final = one methyl rotated 120° (the
    linear path collides H atoms — IDPP's classic demo case)."""
    from ase.build import molecule

    eth = molecule("C2H6")
    pos = eth.positions
    c_idx = [i for i, s in enumerate(eth.get_chemical_symbols()) if s == "C"]
    c0, c1 = c_idx
    axis = pos[c1] - pos[c0]
    axis /= np.linalg.norm(axis)
    h_idx = [i for i, s in enumerate(eth.get_chemical_symbols()) if s == "H"]
    methyl0 = [i for i in h_idx
               if np.linalg.norm(pos[i] - pos[c0]) < np.linalg.norm(pos[i] - pos[c1])]

    def rotate_about_axis(points, origin, axis, angle_deg):
        a = angle_deg * np.pi / 180
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)
        return (points - origin) @ R.T + origin

    def in_box(atoms, a=10.0):
        atoms.cell = [a, a, a]
        atoms.pbc = True
        atoms.center()
        return atoms

    ini = in_box(eth.copy())
    fin = eth.copy()
    for i in [c0] + methyl0:
        fin.positions[i] = rotate_about_axis(fin.positions[i], pos[c0], axis, 120.0)
    return ini, in_box(fin)


def _min_pair_distance(atoms: Atoms) -> float:
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def test_interpolate_idpp_frame_count_and_endpoints():
    ini, fin = _ethane_pair()
    frames = interpolate_idpp(ini, fin, 4)
    assert len(frames) == 6
    assert np.allclose(frames[0].positions, ini.positions)
    assert np.allclose(frames[-1].positions, fin.positions)
    for f in frames:
        assert np.allclose(f.get_cell(), ini.get_cell())
        assert f.constraints == []


def test_interpolate_idpp_avoids_collisions():
    """Linear interpolation collides the rotating hydrogens; IDPP
    keeps every interatomic distance physical."""
    ini, fin = _ethane_pair()
    linear = interpolate_neb(ini, fin, 5)
    idpp = interpolate_idpp(ini, fin, 5)

    assert min(_min_pair_distance(f) for f in linear) < 0.75   # collision
    assert min(_min_pair_distance(f) for f in idpp) > 0.9      # physical


def test_interpolate_idpp_uniform_distance_steps():
    """IDPP evens out the distance-matrix change between adjacent
    images (its defining property); linear does not."""
    ini, fin = _ethane_pair()
    linear = interpolate_neb(ini, fin, 5)
    idpp = interpolate_idpp(ini, fin, 5)

    def step_std(frames):
        dm = [f.get_all_distances(mic=True) for f in frames]
        steps = [np.linalg.norm(dm[i + 1] - dm[i]) for i in range(len(dm) - 1)]
        return float(np.std(steps))

    assert step_std(idpp) < step_std(linear) / 10


def test_interpolate_idpp_validation():
    a = _cubic(Atoms("Cu2", positions=[[0.0, 0, 0], [2.5, 0, 0]]))
    b = a.copy()
    with pytest.raises(ValueError, match="between 1 and 98"):
        interpolate_idpp(a, b, 0)
    with pytest.raises(ValueError, match="same elements"):
        interpolate_idpp(a, _cubic(Atoms("CuAg", positions=[[0.0, 0, 0], [2.5, 0, 0]])), 2)
    bad_cell = b.copy()
    bad_cell.cell = [11, 10, 10]
    with pytest.raises(ValueError, match="cells differ"):
        interpolate_idpp(a, bad_cell, 2)


def test_idpp_wraps_frames_into_cell():
    """IDPP relaxes in Cartesian space — frames must come back into
    the cell."""
    ini, fin = _ethane_pair()
    for f in interpolate_idpp(ini, fin, 3):
        frac = f.get_scaled_positions(wrap=False)
        assert np.all((frac >= 0.0) & (frac < 1.0))


def test_ethane_example_files_regression():
    """The committed example reproduces the collision-free IDPP path."""
    ini = FileIO.read(str(ETHANE / "initial" / "POSCAR"))
    fin = FileIO.read(str(ETHANE / "final" / "POSCAR"))
    assert len(ini) == 8
    linear = interpolate_neb(ini, fin, 5)
    idpp = interpolate_idpp(ini, fin, 5)
    assert min(_min_pair_distance(f) for f in linear) < 0.75
    assert min(_min_pair_distance(f) for f in idpp) > 0.9


# ----------------------------------------------------------------------
# Frozen atoms in interpolation
# ----------------------------------------------------------------------

FROZEN = REPO / "examples" / "neb_frozen"


def test_constraints_to_fixed_flags_conversion():
    from ase.constraints import FixAtoms, FixScaled

    atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    atoms.constraints = [FixAtoms(indices=[0]), FixScaled([1], [False, False, True])]
    flags = constraints_to_fixed_flags(atoms)
    assert flags.shape == (3, 3)
    assert flags[0].all()
    assert flags[1].tolist() == [False, False, True]
    assert not flags[2].any()


def test_interpolate_frozen_atoms_stay_put_linear():
    from ase.constraints import FixAtoms

    ini, fin = _ethane_pair()
    mask = np.zeros(len(ini), dtype=bool)
    mask[0] = True  # freeze the first carbon
    frames = interpolate_neb(ini, fin, 4, frozen_mask=mask)
    for f in frames:
        assert np.allclose(f.positions[0], ini.positions[0])
    # and the constraint is attached (written as Selective dynamics)
    assert any(isinstance(c, FixAtoms) for f in frames for c in f.constraints)


def test_interpolate_frozen_atoms_stay_put_idpp():
    ini, fin = _ethane_pair()
    mask = np.zeros(len(ini), dtype=bool)
    mask[0] = True
    frames = interpolate_idpp(ini, fin, 4, frozen_mask=mask)
    for f in frames:
        assert np.allclose(f.positions[0], ini.positions[0])


def test_interpolate_frozen_contradiction_raises():
    ini, fin = _ethane_pair()
    mask = np.zeros(len(ini), dtype=bool)
    mask[0] = True
    fin.positions[0] += [1.0, 0.0, 0.0]  # frozen atom moved
    with pytest.raises(ValueError, match="Frozen atoms"):
        interpolate_neb(ini, fin, 3, frozen_mask=mask)
    with pytest.raises(ValueError, match="1 \(1-based\)"):
        interpolate_idpp(ini, fin, 3, frozen_mask=mask)


def test_interpolate_frozen_mask_shape_check():
    a = _cubic(Atoms("Cu2", positions=[[0.0, 0, 0], [2.5, 0, 0]]))
    b = a.copy()
    with pytest.raises(ValueError, match="one entry per atom"):
        interpolate_neb(a, b, 1, frozen_mask=np.array([True]))


def test_frozen_example_pass_block():
    """The committed pass/block example pair."""
    pi = FileIO.read(str(FROZEN / "pass" / "initial" / "POSCAR"))
    pf = FileIO.read(str(FROZEN / "pass" / "final" / "POSCAR"))
    mask = constraints_to_fixed_flags(pi).all(axis=1)
    assert mask.sum() == 6
    for frames in (interpolate_neb(pi, pf, 4, frozen_mask=mask),
                   interpolate_idpp(pi, pf, 4, frozen_mask=mask)):
        for f in frames:
            assert np.allclose(f.positions[mask], pi.positions[mask])

    bi = FileIO.read(str(FROZEN / "block" / "initial" / "POSCAR"))
    bf = FileIO.read(str(FROZEN / "block" / "final" / "POSCAR"))
    with pytest.raises(ValueError, match="Frozen atoms"):
        interpolate_neb(bi, bf, 4,
                        frozen_mask=constraints_to_fixed_flags(bi).all(axis=1))
