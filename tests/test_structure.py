"""Tests for StructureModel — signals, filepath handling, selection."""

import numpy as np
import pytest
from ase import Atoms

from vaspen.core.structure import StructureModel


class _SignalCounter:
    """Counts emissions of a bound signal."""

    def __init__(self):
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1


def _counter(model: StructureModel, signal_name: str) -> _SignalCounter:
    c = _SignalCounter()
    getattr(model, signal_name).connect(c)
    return c


@pytest.fixture
def model() -> StructureModel:
    return StructureModel()


def test_load_atoms_emits_loaded_not_modified(model):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    loaded = _counter(model, "structure_loaded")
    modified = _counter(model, "structure_modified")

    model.load_atoms(atoms, "test.xyz")

    assert loaded.count == 1
    assert modified.count == 0  # no double rebuild for observers
    assert model.is_dirty is False
    assert model.filepath == "test.xyz"
    assert model.n_atoms == 2
    assert model.selected_index is None


def test_replace_atoms_marks_dirty_and_emits_modified(model):
    modified = _counter(model, "structure_modified")
    model.replace_atoms(Atoms("H"))
    assert modified.count == 1
    assert model.is_dirty is True


def test_replace_atoms_keeps_filepath_until_reset(model):
    """Derived structures must call reset_filepath to force Save As."""
    model.load_atoms(Atoms("H"), "bulk.cif")
    model.replace_atoms(Atoms("He"))
    assert model.filepath == "bulk.cif"

    model.reset_filepath()
    assert model.filepath is None


def test_reset_filepath_forces_save_as(model):
    model.load_atoms(Atoms("H"), "bulk.cif")
    model.reset_filepath()
    assert model.filepath is None


def test_delete_selected_atom_clears_selection(model):
    atoms = Atoms(["H"] * 5, positions=np.arange(15).reshape(5, 3) * 0.5)
    model.load_atoms(atoms)
    model.select_atom(3)
    model.delete_atom(3)
    assert model.selected_index is None


@pytest.mark.xfail(reason="known bug: deleting an earlier atom leaves a stale index")
def test_delete_atom_before_selection_shifts_selection(model):
    atoms = Atoms(["H"] * 5, positions=np.arange(15).reshape(5, 3) * 0.5)
    model.load_atoms(atoms)
    model.select_atom(3)
    model.delete_atom(0)  # indices shift left; selection should become 2
    assert model.selected_index == 2


def test_unique_symbols_first_appearance_order(model):
    model.load_atoms(Atoms("OFe2", positions=np.eye(3) * 1.5, cell=[3, 3, 3]))
    assert model.unique_symbols == ["O", "Fe"]


def test_save_roundtrip(tmp_path, model):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10])
    model.load_atoms(atoms)
    out = tmp_path / "h2.xyz"
    model.save(out)
    assert (out).exists()
    assert model.filepath == str(out)
    assert model.is_dirty is False
