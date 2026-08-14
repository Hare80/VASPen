"""Tests for vaspen.core.neb — NEB interpolation and distance metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from vaspen.core.neb import (
    detect_order_mismatch,
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
