"""Shared test fixtures: offscreen Qt platform + isolated QSettings."""

import io
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import numpy as np
from ase import Atoms
from PySide6.QtCore import QSettings

from vaspen.utils.config import AppConfig

# P1 disordered alloy: two mixed sites + one partially vacant site.
DISORDERED_CIF = """data_disordered
_cell_length_a 3.60
_cell_length_b 3.60
_cell_length_c 3.60
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Fe1 Fe 0.0 0.0 0.0 0.5
Ni1 Ni 0.0 0.0 0.0 0.5
Fe2 Fe 0.5 0.5 0.5 0.8
Co1 Co 0.5 0.5 0.5 0.2
Fe3 Fe 0.0 0.5 0.5 0.75
"""


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch):
    """Bind AppConfig to a throwaway QSettings scope for every test.

    Prevents tests from reading/writing the developer's real VASPen
    settings (language, recent files, window geometry).
    """
    cfg = AppConfig.__new__(AppConfig)
    cfg._settings = QSettings("VASPen", "VASPen-tests")
    cfg._settings.clear()
    monkeypatch.setattr(AppConfig, "_instance", cfg)
    yield
    cfg._settings.clear()


@pytest.fixture
def si_bulk() -> Atoms:
    """2-atom Si in a primitive fcc cell (a=5.43 Å), pbc=True."""
    from ase.build import bulk
    return bulk("Si", a=5.43)


@pytest.fixture
def srtio3() -> Atoms:
    """Cubic SrTiO3 bulk (5 atoms, a=3.905 Å), pbc=True.

    Non-polar surfaces ((100) etc.) dedup to ONE unique termination
    (mirror cleavages are translation-equivalent when the slab is
    centered) — used by the surface-dialog tests.
    """
    a = 3.905
    return Atoms(
        "SrTiO3",
        cell=[a, a, a],
        pbc=True,
        positions=[
            [0.0, 0.0, 0.0],          # Sr
            [a / 2, a / 2, a / 2],    # Ti
            [a / 2, a / 2, 0.0],      # O
            [a / 2, 0.0, a / 2],      # O
            [0.0, a / 2, a / 2],      # O
        ],
    )


@pytest.fixture
def gaas() -> Atoms:
    """GaAs zincblende primitive cell (2 atoms, a=5.65 Å), pbc=True.

    (111) is polar: two unique terminations (Ga- and As-terminated).
    """
    from ase.build import bulk
    return bulk("GaAs", "zincblende", a=5.65)


@pytest.fixture
def water_molecule() -> Atoms:
    """H2O molecule: no cell, pbc=False."""
    from ase.build import molecule
    return molecule("H2O")


@pytest.fixture
def disordered_atoms() -> Atoms:
    """Partially-occupied P1 alloy (Fe/Ni + Fe/Co sites, one vacancy).

    ASE merges co-located species into one atom per site; the full
    composition lives in atoms.info['occupancy'] + spacegroup_kinds.
    """
    from ase.io import read as ase_read
    return ase_read(io.StringIO(DISORDERED_CIF), format="cif")


# ----------------------------------------------------------------------
# Structures rebuilt in code — the test suite must never read from
# examples/ (CLAUDE.md §9).
# ----------------------------------------------------------------------

def _vacancy_hop_pair() -> tuple[Atoms, Atoms]:
    """fcc Cu vacancy hop: the 2x2x2 primitive repeat minus the
    (0, 1/2, 1/2) corner; final = the (0, 0, 1/2) neighbor hops into
    the vacancy. Path length a/sqrt(2) = 2.5562 A for a = 3.615."""
    a = 3.615
    # fcc primitive cell (a/2·(0,1,1) …) repeated 2x2x2.
    cell = np.array([[0.0, a, a],
                     [a, 0.0, a],
                     [a, a, 0.0]])
    fracs = [[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0],
             [0.5, 0, 0.5], [0.5, 0.5, 0], [0.5, 0.5, 0.5]]
    ini = Atoms("Cu7", scaled_positions=fracs, cell=cell, pbc=True)
    fin = ini.copy()
    # hop frac (0, 0, 1/2) -> (0, 1/2, 1/2): d_frac = (0, 1/2, 0)
    fin.positions[1] = ini.positions[1] + cell[1] / 2
    return ini, fin


@pytest.fixture
def vacancy_hop_pair() -> tuple[Atoms, Atoms]:
    return _vacancy_hop_pair()


@pytest.fixture
def frozen_pass_block_pair() -> tuple[Atoms, Atoms, Atoms, Atoms]:
    """Vacancy-hop pair with 6 of 7 atoms frozen (FixAtoms).

    pass: the free atom (index 1) hops — interpolation is allowed.
    block: a frozen atom (index 0) is additionally displaced —
    interpolation must be rejected.
    """
    from ase.constraints import FixAtoms

    pi, pf = _vacancy_hop_pair()
    fixed = FixAtoms(indices=[0, 2, 3, 4, 5, 6])
    pi.constraints = [fixed]
    pf.constraints = [fixed]
    bi = pi.copy()
    bf = pf.copy()
    bf.positions[0] = bf.cell.T @ np.array(
        [-0.0691562932226832, 0.0691562932226833, 0.0691562932226833])
    return pi, pf, bi, bf


@pytest.fixture
def ethane_pair() -> tuple[Atoms, Atoms]:
    """Ethane in a 10 A box; final = one methyl rotated 120 deg (the
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


@pytest.fixture
def fe_bcc_2x2x2() -> Atoms:
    """bcc Fe 2x2x2 conventional repeat (16 atoms, a=5.74 A cell)."""
    from ase.build import bulk
    return bulk("Fe", "bcc", a=2.87, cubic=True).repeat((2, 2, 2))


@pytest.fixture
def cu111_slab() -> Atoms:
    """36-atom 4-layer Cu(111) slab, 3x3 in-plane (a=7.658 A),
    24 A vacuum."""
    from ase.build import fcc111
    return fcc111("Cu", (3, 3, 4), a=3.61, vacuum=24.0)


@pytest.fixture
def benzene_molecule() -> Atoms:
    """Benzene ring: no cell, pbc=False."""
    from ase.build import molecule
    return molecule("C6H6")


@pytest.fixture
def cu_primitive_standard() -> Atoms:
    """fcc Cu primitive in the standard orientation (a || x, b in the
    xy-plane): a = 3.61/sqrt(2), 60 deg rhombohedron, 1 atom."""
    from ase.geometry import cellpar_to_cell

    a_p = 3.61 / np.sqrt(2)
    cell = cellpar_to_cell([a_p, a_p, a_p, 60.0, 60.0, 60.0])
    return Atoms("Cu", cell=cell, scaled_positions=[[0.0, 0.0, 0.0]], pbc=True)
