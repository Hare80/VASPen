"""Shared test fixtures: offscreen Qt platform + isolated QSettings."""

import io
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
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
