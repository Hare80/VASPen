"""Phase 3 dialog tests: lattice preview, supercell, transform, symmetry.

Offscreen; the LatticeDialog gets a fake viewport so no GL is touched.
"""

import numpy as np
import pytest
from ase import Atoms

from vaspen.core.structure import StructureModel
from vaspen.ui.lattice_dialog import LatticeDialog
from vaspen.ui.supercell_dialog import SupercellDialog
from vaspen.ui.symmetry_dialog import SymmetryDialog
from vaspen.ui.transform_dialog import TransformDialog


class _FakeViewport:
    """Records set_structure calls for LatticeDialog previews."""

    def __init__(self) -> None:
        self.rendered: list = []  # (atoms, reset_view, bonds)

    def set_structure(self, atoms, reset_view=False, bonds=None):
        self.rendered.append((atoms, reset_view, bonds))

    def set_highlight(self, indices):
        pass


@pytest.fixture
def periodic_model() -> StructureModel:
    m = StructureModel()
    m.load_atoms(Atoms("Si", positions=[[2.0, 2.0, 2.0]],
                       cell=[4.0, 4.0, 4.0], pbc=True))
    return m


# ----------------------------------------------------------------------
# Lattice dialog
# ----------------------------------------------------------------------


def test_lattice_dialog_preview_renders_working_copy(qtbot, periodic_model):
    view = _FakeViewport()
    dlg = LatticeDialog(periodic_model, view)
    qtbot.addWidget(dlg)
    assert len(view.rendered) == 1  # initial preview on open

    dlg._len_spins[0].setValue(8.0)  # a: 4 → 8
    assert len(view.rendered) == 2
    atoms = view.rendered[-1][0]
    assert np.allclose(atoms.get_cell().lengths(), [8.0, 4.0, 4.0])
    # model untouched until Accept
    assert np.allclose(periodic_model.cell_lengths, [4.0, 4.0, 4.0])


def test_lattice_dialog_accept_applies_single_undo(qtbot, periodic_model):
    view = _FakeViewport()
    dlg = LatticeDialog(periodic_model, view)
    qtbot.addWidget(dlg)
    dlg._len_spins[0].setValue(8.0)
    dlg._on_accept()

    assert np.allclose(periodic_model.cell_lengths, [8.0, 4.0, 4.0])
    assert len(periodic_model._undo_stack) == 1
    # scale_atoms=True: fractional coordinates kept
    assert np.allclose(periodic_model.scaled_positions[0], [0.5, 0.5, 0.5])


def test_lattice_dialog_reject_restores_viewport(qtbot, periodic_model):
    view = _FakeViewport()
    dlg = LatticeDialog(periodic_model, view)
    qtbot.addWidget(dlg)
    dlg._len_spins[0].setValue(8.0)
    dlg.reject()

    # the last render is the REAL model state again
    atoms = view.rendered[-1][0]
    assert np.allclose(atoms.get_cell().lengths(), [4.0, 4.0, 4.0])
    assert np.allclose(atoms.get_positions(), periodic_model.positions)
    assert periodic_model.can_undo is False


def test_lattice_dialog_invalid_cell_disables_ok(qtbot, periodic_model):
    view = _FakeViewport()
    dlg = LatticeDialog(periodic_model, view)
    qtbot.addWidget(dlg)
    dlg._ang_spins[0].setValue(120.0)
    dlg._ang_spins[1].setValue(120.0)
    dlg._ang_spins[2].setValue(120.0)  # α+β+γ = 360 → degenerate

    assert not dlg._ok_button.isEnabled()
    dlg._on_accept()  # no-op while invalid
    assert np.allclose(periodic_model.cell_lengths, [4.0, 4.0, 4.0])

    dlg._ang_spins[2].setValue(90.0)  # valid again
    assert dlg._ok_button.isEnabled()


def test_lattice_dialog_unscale_keeps_cartesian(qtbot, periodic_model):
    view = _FakeViewport()
    dlg = LatticeDialog(periodic_model, view)
    qtbot.addWidget(dlg)
    dlg._scale_check.setChecked(False)
    dlg._len_spins[0].setValue(8.0)
    dlg._on_accept()

    assert np.allclose(periodic_model.positions[0], [2.0, 2.0, 2.0])
    assert np.allclose(periodic_model.scaled_positions[0], [0.25, 0.5, 0.5])


# ----------------------------------------------------------------------
# Supercell dialog
# ----------------------------------------------------------------------


def test_supercell_dialog_collects_factors(qtbot):
    dlg = SupercellDialog()
    qtbot.addWidget(dlg)
    dlg._spins[0].setValue(2)
    dlg._spins[1].setValue(2)
    dlg._spins[2].setValue(3)
    dlg._on_accept()
    assert dlg.factors == (2, 2, 3)


# ----------------------------------------------------------------------
# Transform dialog
# ----------------------------------------------------------------------


@pytest.fixture
def pair_model() -> StructureModel:
    m = StructureModel()
    m.load_atoms(Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    return m


def test_transform_dialog_translate_whole_structure(qtbot, pair_model):
    dlg = TransformDialog(pair_model)
    qtbot.addWidget(dlg)
    dlg._dx.setValue(1.0)
    dlg._dy.setValue(2.0)
    dlg._on_accept()

    assert dlg.result_atoms is not None
    assert np.allclose(dlg.result_atoms.get_positions()[1], [2.0, 2.0, 0.0])
    # model untouched — the caller applies the result
    assert np.allclose(pair_model.positions[1], [1.0, 0.0, 0.0])


def test_transform_dialog_rotate_scope_selection(qtbot, pair_model):
    pair_model.select_atom(1)
    dlg = TransformDialog(pair_model)
    qtbot.addWidget(dlg)
    assert dlg._scope_sel.isChecked()  # selection preferred by default
    dlg._axis_combo.setCurrentText("Z")
    dlg._angle.setValue(90.0)
    dlg._center_check.setChecked(False)  # rotate about the origin
    dlg._on_accept()

    out = dlg.result_atoms
    assert np.allclose(out.get_positions()[0], [0.0, 0.0, 0.0])  # untouched
    assert np.allclose(out.get_positions()[1], [0.0, 1.0, 0.0], atol=1e-10)






# ----------------------------------------------------------------------
# Symmetry dialog
# ----------------------------------------------------------------------


def test_symmetry_dialog_shows_group_and_symmetrizes(qtbot, si_bulk):
    m = StructureModel()
    m.load_atoms(si_bulk)
    dlg = SymmetryDialog(m)
    qtbot.addWidget(dlg)

    assert dlg._symmetrize_btn.isEnabled()
    dlg._on_symmetrize()

    assert dlg.result_atoms is not None
    # 2-atom primitive fcc → 8-atom conventional cell
    assert len(dlg.result_atoms) == 8
    assert dlg.result_atoms.get_cell().rank == 3


def test_symmetry_dialog_molecule_point_group(qtbot, water_molecule):
    """Isolated systems show the Schoenflies point group and can be
    symmetrized too."""
    m = StructureModel()
    m.load_atoms(water_molecule)
    dlg = SymmetryDialog(m)
    qtbot.addWidget(dlg)

    assert dlg._symmetrize_btn.isEnabled()
    dlg._on_symmetrize()

    assert dlg.result_atoms is not None
    assert len(dlg.result_atoms) == len(water_molecule)


def test_transform_dialog_periodic_whole_rotate_rotates_cell(qtbot, si_bulk):
    """Whole-structure rotate of a periodic structure rotates the unit
    cell WITH the atoms — the crystal stays intact."""
    from vaspen.core.symmetry import analyze

    m = StructureModel()
    m.load_atoms(si_bulk)
    dlg = TransformDialog(m)
    qtbot.addWidget(dlg)
    assert dlg._scope_all.isChecked()
    dlg._axis_combo.setCurrentText("Z")
    # 45° is NOT a lattice symmetry — wrapping in the old cell before
    # rotating the cell would scramble the fractional positions
    dlg._angle.setValue(45.0)
    dlg._on_accept()

    out = dlg.result_atoms
    assert out is not None
    cell0 = np.asarray(si_bulk.get_cell().array)
    c, s = np.cos(np.deg2rad(45.0)), np.sin(np.deg2rad(45.0))
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(out.get_cell().array, cell0 @ r.T, atol=1e-10)
    # the rotated crystal is still diamond
    info = analyze(out, symprec=1e-4)
    assert info is not None and info.number == 227
    # atoms wrapped into the rotated cell
    frac = out.get_scaled_positions()
    assert np.all((frac >= 0.0) & (frac < 1.0))


def test_transform_dialog_periodic_scope_selection_keeps_cell(qtbot, si_bulk):
    """Selection-scope rotate moves only the selected atoms inside the
    fixed cell."""
    m = StructureModel()
    m.load_atoms(si_bulk)
    m.select_atom(1)  # atom 1 is NOT at the origin
    dlg = TransformDialog(m)
    qtbot.addWidget(dlg)
    assert dlg._scope_sel.isChecked()
    dlg._axis_combo.setCurrentText("Z")
    dlg._angle.setValue(90.0)
    dlg._center_check.setChecked(False)  # rotate about the origin
    dlg._on_accept()

    out = dlg.result_atoms
    assert np.allclose(out.get_cell().array, si_bulk.get_cell().array)
    assert np.allclose(out.get_positions()[0], si_bulk.get_positions()[0])
    assert not np.allclose(out.get_positions()[1], si_bulk.get_positions()[1])
    frac = out.get_scaled_positions()
    assert np.all((frac >= 0.0) & (frac < 1.0))
