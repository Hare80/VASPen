"""Tests for StructureModel — signals, filepath handling, selection."""

import numpy as np
import pytest
from ase import Atoms

from vaspen.core.structure import StructureModel, wrap_in_padded_cell


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


def test_delete_selected_atom_clears_selection_and_emits(model):
    atoms = Atoms(["H"] * 5, positions=np.arange(15).reshape(5, 3) * 0.5)
    model.load_atoms(atoms)
    cleared = _counter(model, "selection_cleared")
    model.select_atom(3)
    model.delete_atom(3)
    assert model.selected_index is None
    assert cleared.count == 1


def test_delete_atom_before_selection_shifts_selection(model):
    atoms = Atoms(["H"] * 5, positions=np.arange(15).reshape(5, 3) * 0.5)
    model.load_atoms(atoms)
    model.select_atom(3)
    model.delete_atom(0)  # indices shift left; selection becomes 2
    assert model.selected_index == 2


def test_set_atom_position_moves_single_atom(model):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10])
    model.load_atoms(atoms)
    modified = _counter(model, "structure_modified")
    model.set_atom_position(0, [1.0, 2.0, 3.0])
    assert np.allclose(model.positions[0], [1.0, 2.0, 3.0])
    assert np.allclose(model.positions[1], [0.74, 0, 0])
    assert modified.count == 1
    assert model.is_dirty is True


def test_undo_redo_delete_atom(model):
    atoms = Atoms(["H"] * 3, positions=np.arange(9).reshape(3, 3) * 0.5)
    model.load_atoms(atoms)
    assert not model.can_undo and not model.can_redo

    model.delete_atom(1)
    assert model.n_atoms == 2
    assert model.can_undo and not model.can_redo

    model.undo()
    assert model.n_atoms == 3
    assert model.is_dirty is True
    assert not model.can_undo and model.can_redo

    model.redo()
    assert model.n_atoms == 2
    assert model.can_undo and not model.can_redo


def test_undo_replaces_derived_structure(model):
    """Surface cut / supercell use replace_atoms — must be undoable."""
    model.load_atoms(Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    model.replace_atoms(Atoms("Fe", positions=[[1, 1, 1]]))
    assert model.n_atoms == 1
    model.undo()
    assert model.n_atoms == 2
    assert model.symbols == ["H", "H"]


def test_load_clears_history(model):
    model.load_atoms(Atoms("H"))
    model.delete_atom(0)
    assert model.can_undo
    model.load_atoms(Atoms("He"))
    assert not model.can_undo and not model.can_redo


def test_history_capped(model):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    model.load_atoms(atoms)
    for _ in range(model._max_history + 5):
        model.set_atom_position(0, [0.0, 0.0, 0.0])
    assert len(model._undo_stack) <= model._max_history


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


# ----------------------------------------------------------------------
# Non-periodic → periodic wrapping
# ----------------------------------------------------------------------

def test_wrap_in_padded_cell_bbox_and_padding():
    padding = 10.0
    atoms = Atoms("H2O", positions=[[5, 2, 3], [6, 3, 3], [5.5, 2.5, 4]])
    original_positions = atoms.get_positions().copy()

    wrapped = wrap_in_padded_cell(atoms, padding)

    low, high = original_positions.min(axis=0), original_positions.max(axis=0)
    extent = high - low
    expected_side = extent + 2 * padding
    assert np.allclose(np.diag(wrapped.get_cell()[:]), expected_side)
    assert tuple(wrapped.pbc) == (True, True, True)
    # exactly `padding` vacuum on every face
    new_low = wrapped.get_positions().min(axis=0)
    new_high = wrapped.get_positions().max(axis=0)
    assert np.allclose(new_low, padding)
    assert np.allclose(new_high, extent + padding)
    # bounding-box center coincides with the cell center
    assert np.allclose((new_low + new_high) / 2, expected_side / 2)
    # input atoms untouched
    assert np.allclose(atoms.get_positions(), original_positions)
    assert atoms.get_cell().rank == 0
    assert not atoms.pbc.any()


def test_wrap_in_padded_cell_single_atom():
    wrapped = wrap_in_padded_cell(Atoms("H", positions=[[0, 0, 0]]), 10.0)
    assert np.allclose(np.diag(wrapped.get_cell()[:]), [20, 20, 20])
    assert np.allclose(wrapped.get_positions()[0], [10, 10, 10])


def test_wrap_in_padded_cell_flat_molecule():
    """Zero-extent axis must still produce a full-rank cell."""
    wrapped = wrap_in_padded_cell(
        Atoms("H2", positions=[[0, 0, 5], [0.74, 0, 5]]), 10.0)
    cell = wrapped.get_cell()[:]
    assert wrapped.get_cell().rank == 3
    assert np.allclose(np.diag(cell)[2], 20.0)  # flat axis: only padding
    assert np.allclose(wrapped.get_positions()[:, 2].min(), 10.0)


def test_wrap_in_padded_cell_validation():
    with pytest.raises(ValueError):
        wrap_in_padded_cell(Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]), 0.0)
    with pytest.raises(ValueError):
        wrap_in_padded_cell(Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]), -1.0)
    with pytest.raises(ValueError):
        wrap_in_padded_cell(Atoms(), 10.0)


def test_make_periodic_emits_modified_and_preserves_filepath(model):
    model.load_atoms(Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]), "mol.xyz")
    model.select_atom(1)
    modified = _counter(model, "structure_modified")

    model.make_periodic(10.0)

    assert modified.count == 1
    assert model.is_periodic is True
    assert model.is_dirty is True
    assert model.filepath == "mol.xyz"
    assert model.selected_index == 1  # atom indices unchanged


def test_make_periodic_undo_restores_molecule(model):
    atoms = Atoms("H2O", positions=[[5, 2, 3], [6, 3, 3], [5.5, 2.5, 4]])
    original = atoms.get_positions().copy()
    model.load_atoms(atoms)
    model.make_periodic(10.0)
    side_lengths = np.diag(model.cell).copy()
    assert model.is_periodic is True

    model.undo()
    assert model.is_periodic is False
    assert model.atoms.get_cell().rank == 0
    assert np.allclose(model.positions, original)

    model.redo()
    assert model.is_periodic is True
    assert np.allclose(np.diag(model.cell), side_lengths)
