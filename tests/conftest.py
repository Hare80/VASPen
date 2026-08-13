"""Shared test fixtures: offscreen Qt platform + isolated QSettings."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from ase import Atoms
from PySide6.QtCore import QSettings

from vaspen.utils.config import AppConfig


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
def water_molecule() -> Atoms:
    """H2O molecule: no cell, pbc=False."""
    from ase.build import molecule
    return molecule("H2O")
