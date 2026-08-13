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


# ----------------------------------------------------------------------
# Bonds, atom IDs, multi-select (3D structure editor)
# ----------------------------------------------------------------------

def test_bonds_auto_computed_on_load(model):
    from ase.build import molecule

    model.load_atoms(molecule("H2O"))
    assert model.bond_mode == "auto"
    assert len(model.bonds) == 2


def test_first_manual_bond_edit_switches_mode_and_freezes_auto_list(model):
    from ase.build import molecule

    model.load_atoms(molecule("H2O"))
    auto_bonds = model.bonds
    model.add_bond(1, 2)  # H–H
    assert model.bond_mode == "manual"
    assert len(model.bonds) == len(auto_bonds) + 1
    assert (1, 2) in [(b.i, b.j) for b in model.bonds]
    # manual mode: moving atoms must NOT recompute the bond list
    model.set_atom_position(0, [0.5, 0.5, 0.5])
    assert len(model.bonds) == 3


def test_bond_edit_undo_restores_mode_and_list(model):
    from ase.build import molecule

    model.load_atoms(molecule("H2O"))
    before = model.bonds
    model.add_bond(1, 2)
    assert model.bond_mode == "manual"

    model.undo()
    assert model.bond_mode == "auto"
    assert model.bonds == before

    model.redo()
    assert model.bond_mode == "manual"
    assert len(model.bonds) == 3


def test_remove_bond_switches_to_manual(model):
    from ase.build import molecule

    model.load_atoms(molecule("H2O"))
    model.remove_bond(0, 1)
    assert model.bond_mode == "manual"
    assert [(b.i, b.j) for b in model.bonds] == [(0, 2)]


def test_add_bond_updates_existing_order(model):
    from ase.build import molecule

    model.load_atoms(molecule("H2O"))
    model.add_bond(0, 1, order=2)
    assert len(model.bonds) == 2  # updated, not duplicated
    assert [b.order for b in model.bonds if (b.i, b.j) == (0, 1)] == [2]


def test_add_bond_validates(model):
    from ase.build import molecule

    model.load_atoms(molecule("H2O"))
    with pytest.raises(ValueError):
        model.add_bond(0, 0)
    with pytest.raises(ValueError):
        model.add_bond(0, 9)
    with pytest.raises(ValueError):
        model.add_bond(0, 1, order=7)


def test_delete_atom_remaps_manual_bonds(model):
    from ase.build import molecule

    model.load_atoms(molecule("H2O"))
    model.add_bond(1, 2)  # manual H–H
    model.delete_atom(0)  # remove O — the two O–H bonds vanish
    assert model.bond_mode == "manual"
    assert [(b.i, b.j) for b in model.bonds] == [(0, 1)]


def test_delete_atoms_single_undo_entry(model):
    atoms = Atoms(["H"] * 5, positions=np.arange(15).reshape(5, 3) * 0.5)
    model.load_atoms(atoms)
    model.delete_atoms([1, 3])
    assert model.n_atoms == 3
    assert len(model._undo_stack) == 1
    model.undo()
    assert model.n_atoms == 5


def test_atom_ids_survive_delete_and_undo(model):
    atoms = Atoms(["H"] * 3, positions=np.arange(9).reshape(3, 3) * 0.5)
    model.load_atoms(atoms)
    id1 = model.atom_id(1)
    model.delete_atom(1)
    assert model.index_of_id(id1) is None
    model.undo()
    assert model.index_of_id(id1) == 1


def test_extend_atoms_assigns_distinct_ids(model):
    model.load_atoms(Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    model.extend_atoms(Atoms("O", positions=[[1, 1, 1]]))
    ids = {model.atom_id(i) for i in range(3)}
    assert len(ids) == 3


def test_set_atom_positions_single_undo_entry(model):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    model.load_atoms(atoms)
    model.set_atom_positions([0, 1], [[1, 0, 0], [2, 0, 0]])
    assert np.allclose(model.positions, [[1, 0, 0], [2, 0, 0]])
    assert len(model._undo_stack) == 1
    model.undo()
    assert np.allclose(model.positions, [[0, 0, 0], [0.74, 0, 0]])


def test_multi_select_semantics(model):
    atoms = Atoms(["H"] * 5, positions=np.arange(15).reshape(5, 3) * 0.5)
    model.load_atoms(atoms)
    model.set_selection({0, 2})
    assert model.selected_indices == {0, 2}
    assert model.selected_index == 0  # min of the set
    model.add_to_selection({4})
    assert model.selected_indices == {0, 2, 4}
    model.toggle_selection({2, 3})
    assert model.selected_indices == {0, 3, 4}
    model.select_invert()
    assert model.selected_indices == {1, 2}
    model.select_all()
    assert model.selected_indices == set(range(5))
    model.clear_selection()
    assert model.selected_indices == set()


def test_selection_changed_fires_on_real_change(model):
    atoms = Atoms(["H"] * 3, positions=np.arange(9).reshape(3, 3) * 0.5)
    model.load_atoms(atoms)
    changed = _counter(model, "selection_changed")
    model.set_selection({1})
    assert changed.count == 1
    model.set_selection({1})  # no actual change → no signal
    assert changed.count == 1
    model.toggle_selection({2})
    assert changed.count == 2


def test_select_neighbors_and_connected(model):
    from ase.build import molecule

    model.load_atoms(molecule("H2O"))  # O(0)–H(1), O(0)–H(2)
    model.select_atom(0)
    model.select_neighbors()
    assert model.selected_indices == {1, 2}

    model.select_atom(1)
    model.select_connected()
    assert model.selected_indices == {0, 1, 2}


def test_multi_delete_clears_selection_once(model):
    atoms = Atoms(["H"] * 5, positions=np.arange(15).reshape(5, 3) * 0.5)
    model.load_atoms(atoms)
    cleared = _counter(model, "selection_cleared")
    model.set_selection({1, 3})
    model.delete_atoms([1, 3])
    assert model.selected_indices == set()
    assert cleared.count == 1
    assert len(model._undo_stack) == 1


def test_undo_redo_restore_selection(model):
    """Undo/redo steps back to the selection that existed at each step
    (user request: undo must be able to return to the 'only selected
    atoms' state, not clear the selection)."""
    atoms = Atoms(["H"] * 3, positions=np.arange(9).reshape(3, 3) * 0.5)
    model.load_atoms(atoms)
    model.set_selection({1})
    model.set_atom_position(2, [1.0, 1.0, 1.0])

    model.undo()
    assert model.selected_indices == {1}
    assert np.allclose(model.positions[2], [3.0, 3.5, 4.0])  # original

    model.redo()
    assert model.selected_indices == {1}


def test_set_bond_mode_roundtrip(model):
    from ase.build import molecule

    model.load_atoms(molecule("H2O"))
    model.add_bond(1, 2)  # switches to manual
    assert model.bond_mode == "manual"
    assert len(model.bonds) == 3

    model.set_atom_position(1, [1.0, -0.2, 0.0])  # manual: no recompute
    assert len(model.bonds) == 3

    model.set_bond_mode("auto")  # re-derive from geometry
    assert model.bond_mode == "auto"
    assert len(model.bonds) == 2

    model.undo()  # mode switch is undoable
    assert model.bond_mode == "manual"
    assert len(model.bonds) == 3

    with pytest.raises(ValueError):
        model.set_bond_mode("bogus")


def test_detect_bonds_one_shot(model):
    """Detect Bonds refreshes connectivity without touching the mode —
    the manual-mode on-demand path (toolbar button)."""
    from ase.build import molecule

    model.load_atoms(molecule("H2O"))
    model.set_bond_mode("manual")
    model.set_atom_position(0, [5.0, 5.0, 5.0])  # geometry broken; no recompute
    assert len(model.bonds) == 2  # stale list

    model.detect_bonds()
    assert len(model.bonds) == 0  # re-derived from geometry
    assert model.bond_mode == "manual"  # mode unchanged

    model.undo()  # one-shot detection is undoable
    assert len(model.bonds) == 2
