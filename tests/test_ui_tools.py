"""UI flow tests for the 3D edit tools (Phase 1).

All tests avoid GL: picking is monkeypatched at the instance level and
camera math (screen_to_world / project_to_screen) is pure numpy, so the
full tool → signal → model flow runs offscreen.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.io import write as ase_write
from PySide6.QtCore import QPointF, Qt

from vaspen.core import file_io as fi
from vaspen.ui.main_window import MainWindow
from vaspen.ui.tools import ToolMode


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


class _FakeMouseEvent:
    def __init__(self, button=Qt.LeftButton, pos=(0.0, 0.0),
                 modifiers=Qt.NoModifier):
        self._button = button
        self._pos = QPointF(*pos)
        self._mods = modifiers

    def button(self):
        return self._button

    def position(self):
        return self._pos

    def modifiers(self):
        return self._mods

    def accept(self):
        pass


class _FakeKeyEvent:
    def __init__(self, key):
        self._key = key

    def key(self):
        return self._key

    def accept(self):
        pass


def _click(view, pos=(400.0, 300.0), mods=Qt.NoModifier):
    view.mousePressEvent(_FakeMouseEvent(Qt.LeftButton, pos, mods))
    view.mouseReleaseEvent(_FakeMouseEvent(Qt.LeftButton, pos, mods))


def _drag(view, start, end, mods=Qt.NoModifier):
    view.mousePressEvent(_FakeMouseEvent(Qt.LeftButton, start, mods))
    view._last_pos = QPointF(*start)
    view._dragged = True  # simulate exceeding the click threshold
    view.mouseMoveEvent(_FakeMouseEvent(Qt.LeftButton, end, mods))
    view.mouseReleaseEvent(_FakeMouseEvent(Qt.LeftButton, end, mods))


def _load(window, monkeypatch, atoms, name="fake.xyz"):
    monkeypatch.setattr(fi.FileIO, "read", classmethod(lambda cls, p: atoms))
    window._open_file(name)


# ----------------------------------------------------------------------
# Add atom
# ----------------------------------------------------------------------

def test_add_atom_tool_places_combo_element(window, monkeypatch, si_bulk):
    _load(window, monkeypatch, si_bulk, "bulk.vasp")
    view = window._viewport
    view.set_mode(ToolMode.ADD_ATOM)
    view._pick = lambda pos: None  # click on empty space
    window._element_combo.setCurrentText("Fe")

    n_before = window._structure.n_atoms
    _click(view)

    assert window._structure.n_atoms == n_before + 1
    assert window._structure.symbols[-1] == "Fe"
    assert len(window._structure._undo_stack) == 1
    # periodic structure: the new atom is wrapped into the home cell
    frac = window._structure.scaled_positions[-1]
    assert np.all((frac >= 0) & (frac < 1))


def test_add_atom_click_on_atom_grows_bonded_atom(window, monkeypatch):
    """Clicking an atom in add mode grows a new atom at the ideal bond
    length (sum of covalent radii) instead of being ignored."""
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.ADD_ATOM)
    window._element_combo.setCurrentText("H")
    view._pick = lambda pos: ("atom", 0)  # the O atom

    n_before = window._structure.n_atoms
    _click(view)

    assert window._structure.n_atoms == n_before + 1
    new = window._structure.positions[-1]
    anchor = window._structure.positions[0]
    # ideal O–H distance: 0.66 + 0.31 = 0.97 Å
    assert np.isclose(np.linalg.norm(new - anchor), 0.97, atol=1e-6)
    # auto mode: the grown atom is bonded to its anchor
    assert (0, n_before) in [(b.i, b.j) for b in window._structure.bonds]


# ----------------------------------------------------------------------
# Move atom
# ----------------------------------------------------------------------

def test_move_tool_drag_commits_once(window, monkeypatch):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]],
                  cell=[10, 10, 10], pbc=True)
    _load(window, monkeypatch, atoms)
    view = window._viewport
    view.set_mode(ToolMode.MOVE_ATOM)
    view._pick = lambda pos: ("atom", 0)

    modified = []
    window._structure.structure_modified.connect(lambda: modified.append(1))
    original = view._atom_pos[0].copy()

    start, end = (400.0, 300.0), (450.0, 300.0)
    view.mousePressEvent(_FakeMouseEvent(Qt.LeftButton, start))
    view._last_pos = QPointF(*start)
    view._dragged = True
    view.mouseMoveEvent(_FakeMouseEvent(Qt.LeftButton, end))

    assert len(modified) == 0  # preview only — no model call during drag
    assert not np.allclose(view._atom_pos[0], original)  # fast path moved it

    view.mouseReleaseEvent(_FakeMouseEvent(Qt.LeftButton, end))

    assert len(modified) == 1  # committed exactly once
    assert len(window._structure._undo_stack) == 1
    assert window._structure.positions[0][0] > original[0]


def test_move_tool_drag_wraps_periodic_boundary(window, monkeypatch):
    atoms = Atoms("H", positions=[[9.5, 5, 5]], cell=[10, 10, 10], pbc=True)
    _load(window, monkeypatch, atoms)
    view = window._viewport
    view.set_mode(ToolMode.MOVE_ATOM)
    view._pick = lambda pos: ("atom", 0)

    start, end = (400.0, 300.0), (450.0, 300.0)
    view.mousePressEvent(_FakeMouseEvent(Qt.LeftButton, start))
    view._last_pos = QPointF(*start)
    view._dragged = True
    view.mouseMoveEvent(_FakeMouseEvent(Qt.LeftButton, end))
    view.mouseReleaseEvent(_FakeMouseEvent(Qt.LeftButton, end))

    pos = window._structure.positions[0]
    assert 0.0 <= pos[0] < 10.0  # wrapped back into the cell
    assert np.allclose(pos[1:], [5.0, 5.0])


def test_move_tool_esc_cancels_drag(window, monkeypatch):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]],
                  cell=[10, 10, 10], pbc=True)
    _load(window, monkeypatch, atoms)
    view = window._viewport
    view.set_mode(ToolMode.MOVE_ATOM)
    view._pick = lambda pos: ("atom", 0)
    original = view._atom_pos[0].copy()

    start, end = (400.0, 300.0), (450.0, 300.0)
    view.mousePressEvent(_FakeMouseEvent(Qt.LeftButton, start))
    view._last_pos = QPointF(*start)
    view._dragged = True
    view.mouseMoveEvent(_FakeMouseEvent(Qt.LeftButton, end))
    assert not np.allclose(view._atom_pos[0], original)

    view.keyPressEvent(_FakeKeyEvent(Qt.Key_Escape))
    assert np.allclose(view._atom_pos[0], original)

    view.mouseReleaseEvent(_FakeMouseEvent(Qt.LeftButton, end))
    assert window._structure._undo_stack == []  # nothing committed


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------

def test_select_click_modifier_semantics(window, monkeypatch):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]],
                  cell=[10, 10, 10], pbc=True)
    _load(window, monkeypatch, atoms)
    view = window._viewport
    view.set_mode(ToolMode.SELECT)

    view._pick = lambda pos: ("atom", 0)
    _click(view, mods=Qt.NoModifier)          # replace
    assert window._structure.selected_indices == {0}

    view._pick = lambda pos: ("atom", 1)
    _click(view, mods=Qt.ControlModifier)     # add
    assert window._structure.selected_indices == {0, 1}

    _click(view, mods=Qt.ShiftModifier)       # toggle atom 1 off
    assert window._structure.selected_indices == {0}

    _click(view, mods=Qt.NoModifier)          # replace
    assert window._structure.selected_indices == {1}


def test_select_click_on_empty_clears(window, monkeypatch):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]],
                  cell=[10, 10, 10], pbc=True)
    _load(window, monkeypatch, atoms)
    view = window._viewport
    view.set_mode(ToolMode.SELECT)
    window._structure.set_selection({0})

    view._pick = lambda pos: None
    _click(view)

    assert window._structure.selected_indices == set()


def test_box_select_via_shift_drag(window, monkeypatch):
    atoms = Atoms("H4", positions=[[0, 0, 0], [2, 0, 0], [5, 5, 5], [7, 7, 7]],
                  cell=[10, 10, 10], pbc=True)
    _load(window, monkeypatch, atoms)
    view = window._viewport
    view.set_mode(ToolMode.SELECT)
    view._pick = lambda pos: None

    # The camera fit puts every atom inside the viewport, so a rubber
    # band over the whole widget must select all four atoms.
    w, h = view.width(), view.height()
    _drag(view, (0.0, 0.0), (float(w), float(h)), mods=Qt.ShiftModifier)

    assert window._structure.selected_indices == {0, 1, 2, 3}
    assert view._rubber_rect is None  # band cleared on release


# ----------------------------------------------------------------------
# Delete / bonds
# ----------------------------------------------------------------------

def test_delete_tool_deletes_atom(window, monkeypatch):
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.DELETE)
    view._pick = lambda pos: ("atom", 0)  # the O atom

    _click(view)

    assert window._structure.n_atoms == 2
    assert window._structure.bonds == []  # both O–H bonds vanished


def test_delete_tool_deletes_bond(window, monkeypatch):
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.DELETE)
    view._pick = lambda pos: ("bond", 0)  # O–H

    _click(view)

    assert window._structure.bond_mode == "manual"
    assert [(b.i, b.j) for b in window._structure.bonds] == [(0, 2)]


def test_create_bond_two_click_and_cancel(window, monkeypatch):
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.CREATE_BOND)

    view._pick = lambda pos: ("atom", 1)
    _click(view)
    assert view._preview_indices == {1}

    view._pick = lambda pos: ("atom", 2)
    _click(view)
    assert window._structure.bond_mode == "manual"
    assert (1, 2) in [(b.i, b.j) for b in window._structure.bonds]
    assert view._preview_indices == set()  # preview cleared

    # start again, then cancel on empty space
    view._pick = lambda pos: ("atom", 1)
    _click(view)
    assert view._preview_indices == {1}
    view._pick = lambda pos: None
    _click(view)
    assert view._preview_indices == set()


def test_create_bond_undo_restores_bond_list(window, monkeypatch):
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    window._structure.add_bond(1, 2)
    assert window._structure.bond_mode == "manual"

    window.act_undo.trigger()

    assert window._structure.bond_mode == "manual"  # settled default
    assert len(window._structure.bonds) == 2
    assert window.act_redo.isEnabled()


# ----------------------------------------------------------------------
# Modes / keyboard / sync
# ----------------------------------------------------------------------

def test_mode_actions_switch_viewport_mode_exclusively(window):
    view = window._viewport
    for mode in (ToolMode.ADD_ATOM, ToolMode.MOVE_ATOM,
                 ToolMode.DELETE, ToolMode.CREATE_BOND, ToolMode.SELECT):
        window._mode_actions[mode].setChecked(True)
        assert view.mode() == mode
    window._mode_actions[ToolMode.ADD_ATOM].setChecked(True)
    assert not window._mode_actions[ToolMode.SELECT].isChecked()


def test_delete_key_deletes_selection_in_one_undo_step(window, monkeypatch):
    atoms = Atoms(["H"] * 3, positions=np.arange(9).reshape(3, 3) * 0.5,
                  cell=[10, 10, 10], pbc=True)
    _load(window, monkeypatch, atoms)
    window._structure.set_selection({0, 2})

    window.act_delete_selection.trigger()

    assert window._structure.n_atoms == 1
    assert len(window._structure._undo_stack) == 1


def test_highlight_survives_structure_edit(window, monkeypatch):
    """set_structure clears the viewport highlight — the window must
    re-apply the model selection after every edit."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]],
                  cell=[10, 10, 10], pbc=True)
    _load(window, monkeypatch, atoms)
    window._structure.select_atom(0)

    window._structure.set_atom_position(1, [1.0, 0, 0])

    assert window._viewport._selected_indices == {0}


def test_tree_multi_select_sync(window, monkeypatch):
    atoms = Atoms(["H"] * 4, positions=np.arange(12).reshape(4, 3) * 0.5,
                  cell=[10, 10, 10], pbc=True)
    _load(window, monkeypatch, atoms)
    tree = window._structure_tree

    window._structure.set_selection({0, 2})
    rows = {i.row() for i in tree._table.selectionModel().selectedRows()}
    assert rows == {0, 2}

    tree._table.selectRow(1)
    assert window._structure.selected_indices == {1}


def test_view_direction_presets_set_camera(window):
    view = window._viewport
    window._set_view_direction(90.0, 0.0)
    assert view._cam_azimuth == 90.0
    assert view._cam_elevation == 0.0
    window._set_view_direction(0.0, 89.9, (0.0, 0.0, 1.0))
    assert np.allclose(view._cam_up, [0.0, 0.0, 1.0])


# ----------------------------------------------------------------------
# Phase 1.5 acceptance fixes
# ----------------------------------------------------------------------

def test_move_moves_connected_group(window, monkeypatch):
    """Dragging one atom of a molecule moves the whole bond-connected
    molecule as a rigid body (user decision 2026-08-13)."""
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.MOVE_ATOM)
    view._pick = lambda pos: ("atom", 0)  # the O atom

    start, end = (400.0, 300.0), (450.0, 300.0)
    view.mousePressEvent(_FakeMouseEvent(Qt.LeftButton, start))
    view._last_pos = QPointF(*start)
    view._dragged = True
    view.mouseMoveEvent(_FakeMouseEvent(Qt.LeftButton, end))
    view.mouseReleaseEvent(_FakeMouseEvent(Qt.LeftButton, end))

    original = molecule("H2O").get_positions()
    moved = window._structure.positions
    delta = moved[0] - original[0]
    assert np.linalg.norm(delta) > 0
    # whole molecule shifted rigidly
    assert np.allclose(moved[1] - original[1], delta, atol=1e-6)
    assert np.allclose(moved[2] - original[2], delta, atol=1e-6)
    assert len(window._structure._undo_stack) == 1


def test_move_prefers_selection_over_component(window, monkeypatch):
    """Selection has top priority for Move: only the selected atoms move."""
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.MOVE_ATOM)
    window._structure.set_selection({1, 2})
    view._pick = lambda pos: ("atom", 1)

    start, end = (400.0, 300.0), (450.0, 300.0)
    view.mousePressEvent(_FakeMouseEvent(Qt.LeftButton, start))
    view._last_pos = QPointF(*start)
    view._dragged = True
    view.mouseMoveEvent(_FakeMouseEvent(Qt.LeftButton, end))
    view.mouseReleaseEvent(_FakeMouseEvent(Qt.LeftButton, end))

    original = molecule("H2O").get_positions()
    moved = window._structure.positions
    assert np.allclose(moved[0], original[0])  # O untouched
    assert not np.allclose(moved[1], original[1])
    assert np.allclose(moved[2] - original[2], moved[1] - original[1], atol=1e-6)


def test_add_atom_drag_snaps_to_ideal_bond_length(window, monkeypatch):
    """Press on an atom and drag: the new atom follows the cursor at the
    ideal bond length (sum of covalent radii), with a ghost preview."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    _load(window, monkeypatch, atoms)  # non-periodic: no wrapping
    view = window._viewport
    view.set_mode(ToolMode.ADD_ATOM)
    window._element_combo.setCurrentText("H")
    view._pick = lambda pos: ("atom", 0)

    n_before = window._structure.n_atoms
    start, end = (400.0, 300.0), (400.0, 250.0)
    view.mousePressEvent(_FakeMouseEvent(Qt.LeftButton, start))
    view._last_pos = QPointF(*start)
    view._dragged = True
    view.mouseMoveEvent(_FakeMouseEvent(Qt.LeftButton, end))
    assert view._ghost_pos is not None  # ghost preview during drag
    assert view._ghost_bond_from is not None  # ghost bond to the anchor
    view.mouseReleaseEvent(_FakeMouseEvent(Qt.LeftButton, end))

    assert view._ghost_pos is None  # cleared on release
    assert view._ghost_bond_from is None
    assert window._structure.n_atoms == n_before + 1
    new = window._structure.positions[-1]
    anchor = window._structure.positions[0]
    # ideal H–H distance: 0.31 + 0.31 = 0.62 Å from the anchor
    assert np.isclose(np.linalg.norm(new - anchor), 0.62, atol=1e-6)
    # auto mode: the grown atom is bonded to its anchor
    assert (0, n_before) in [(b.i, b.j) for b in window._structure.bonds]


def test_delete_tool_deletes_whole_selection(window, monkeypatch):
    atoms = Atoms(["H"] * 4, positions=np.arange(12).reshape(4, 3) * 0.5,
                  cell=[10, 10, 10], pbc=True)
    _load(window, monkeypatch, atoms)
    view = window._viewport
    view.set_mode(ToolMode.DELETE)
    window._structure.set_selection({0, 2})
    view._pick = lambda pos: ("atom", 0)  # clicked atom is in the selection

    _click(view)

    assert window._structure.n_atoms == 2
    assert len(window._structure._undo_stack) == 1


def test_delete_tool_click_empty_with_selection_deletes_group(window, monkeypatch):
    atoms = Atoms(["H"] * 4, positions=np.arange(12).reshape(4, 3) * 0.5,
                  cell=[10, 10, 10], pbc=True)
    _load(window, monkeypatch, atoms)
    view = window._viewport
    view.set_mode(ToolMode.DELETE)
    window._structure.set_selection({1, 3})
    view._pick = lambda pos: None

    _click(view)

    assert window._structure.n_atoms == 2


def test_create_bond_from_two_selected(window, monkeypatch):
    """Two selected atoms + one click in bond mode = the bond between them."""
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.CREATE_BOND)
    window._structure.set_selection({1, 2})
    view._pick = lambda pos: ("atom", 1)

    _click(view)

    assert window._structure.bond_mode == "manual"
    assert (1, 2) in [(b.i, b.j) for b in window._structure.bonds]


def test_element_column_edit(window, monkeypatch):
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    _load(window, monkeypatch, atoms)
    tree = window._structure_tree

    tree._table.item(0, 1).setText("fe")
    tree._on_cell_changed(0, 1)

    assert window._structure.symbols[0] == "Fe"
    assert window._structure.n_atoms == 3


def test_element_column_invalid_reverts(window, monkeypatch):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    _load(window, monkeypatch, atoms)
    tree = window._structure_tree
    monkeypatch.setattr(
        "vaspen.ui.structure_tree.QMessageBox.warning",
        staticmethod(lambda *a, **k: None),
    )

    tree._table.item(0, 1).setText("Xx")
    tree._on_cell_changed(0, 1)

    assert window._structure.symbols[0] == "H"  # unchanged
    assert tree._table.item(0, 1).text() == "H"  # reverted


def test_element_text_colors_readable():
    """Light Jmol colors (white H etc.) must be darkened for text use."""
    from ase.data import chemical_symbols
    from vaspen.ui.viewport3d import element_text_color

    for sym in chemical_symbols[1:]:
        r, g, b = element_text_color(sym)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        assert luminance <= 0.63, sym


def test_redo_has_ctrl_y(window):
    shortcuts = {s.toString() for s in window.act_redo.shortcuts()}
    assert "Ctrl+Shift+Z" in shortcuts
    assert "Ctrl+Y" in shortcuts


# ----------------------------------------------------------------------
# Phase 1.5b — undo selection, auto bonds, screen-outward placement
# ----------------------------------------------------------------------

def test_undo_restores_selection_highlight(window, monkeypatch):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]],
                  cell=[10, 10, 10], pbc=True)
    _load(window, monkeypatch, atoms)
    window._structure.set_selection({0})
    window._structure.set_atom_position(1, [1.0, 0, 0])

    window.act_undo.trigger()

    assert window._structure.selected_indices == {0}
    assert window._viewport._selected_indices == {0}  # highlight restored


def test_auto_bonds_action_toggles_mode(window, monkeypatch):
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    # settled: auto-detect OFF by default; bonds detected once at load
    assert not window.act_auto_bonds.isChecked()
    assert window._structure.bond_mode == "manual"
    assert len(window._structure.bonds) == 2

    window.act_auto_bonds.setChecked(True)
    assert window._structure.bond_mode == "auto"

    # manual edits keep it manual; re-checking re-derives connectivity
    window.act_auto_bonds.setChecked(False)
    window._structure.add_bond(1, 2)
    assert len(window._structure.bonds) == 3
    assert not window.act_auto_bonds.isChecked()  # synced after edit

    window.act_auto_bonds.setChecked(True)
    assert window._structure.bond_mode == "auto"
    assert len(window._structure.bonds) == 2


def test_add_atom_drag_creates_bond_in_manual_mode(window, monkeypatch):
    """Dragging an atom out of an existing atom creates the anchor bond
    even when Auto Detect Bonds is off (manual mode)."""
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    window._structure.set_bond_mode("manual")
    view = window._viewport
    view.set_mode(ToolMode.ADD_ATOM)
    window._element_combo.setCurrentText("H")
    view._pick = lambda pos: ("atom", 1)

    n_before = window._structure.n_atoms
    start, end = (400.0, 300.0), (400.0, 250.0)
    view.mousePressEvent(_FakeMouseEvent(Qt.LeftButton, start))
    view._last_pos = QPointF(*start)
    view._dragged = True
    view.mouseMoveEvent(_FakeMouseEvent(Qt.LeftButton, end))
    view.mouseReleaseEvent(_FakeMouseEvent(Qt.LeftButton, end))

    assert window._structure.n_atoms == n_before + 1
    assert (1, n_before) in [(b.i, b.j) for b in window._structure.bonds]


def test_detect_bonds_action_refreshes_connectivity(window, monkeypatch):
    """Toolbar Detect Bonds re-derives bonds on demand in manual mode."""
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    window._structure.set_bond_mode("manual")
    window._structure.set_atom_position(0, [5.0, 5.0, 5.0])  # breaks geometry
    assert len(window._structure.bonds) == 2  # stale (no auto recompute)

    window.act_detect_bonds.trigger()

    assert len(window._structure.bonds) == 0  # re-derived
    assert window._structure.bond_mode == "manual"  # mode untouched


def test_bond_click_selects_bond(window, monkeypatch):
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.SELECT)
    view._pick = lambda pos: ("bond", 1)

    _click(view)

    assert window._structure.selected_bonds == {1}
    assert view._selected_bonds == {1}

    # empty click clears the bond selection too
    view._pick = lambda pos: None
    _click(view)
    assert window._structure.selected_bonds == set()


def test_bond_order_bake_vertex_counts(window):
    """Double/triple bonds render as parallel cylinders, aromatic (4)
    as dashed segments — ranges stay per-bond and aligned."""
    from vaspen.core.bonds import Bond

    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    view = window._viewport
    view.set_structure(atoms, bonds=[Bond(0, 1, 2)])
    assert view._bond_ranges[0][1] == 144  # 2 × 72

    view.set_structure(atoms, bonds=[Bond(0, 1, 3)])
    assert view._bond_ranges[0][1] == 216  # 3 × 72

    view.set_structure(atoms, bonds=[Bond(0, 1, 1)])
    assert view._bond_ranges[0][1] == 72

    view.set_structure(atoms, bonds=[Bond(0, 1, 4)])
    assert view._bond_ranges[0][1] == 6 * 72  # dashed


def test_measure_distance_flow(window, monkeypatch):
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.MEASURE_DISTANCE)
    picks = [("atom", 0), ("atom", 1)]
    view._pick = lambda pos: picks.pop(0)

    _click(view)  # first atom
    assert view._preview_indices == {0}

    _click(view)  # second atom → measurement complete
    assert len(window._measurement_manager) == 1
    payload = view._measurements
    assert len(payload) == 1
    kind, idx, text = payload[0]
    assert kind == "distance"
    assert idx == [0, 1]
    assert "Å" in text


def test_measure_tool_cancel_on_empty_click(window, monkeypatch):
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.MEASURE_ANGLE)
    view._pick = lambda pos: ("atom", 0)
    _click(view)
    assert view._preview_indices == {0}

    view._pick = lambda pos: None
    _click(view)
    assert view._preview_indices == set()
    assert len(window._measurement_manager) == 0


def test_measurement_survives_atom_deletion(window, monkeypatch):
    """Orphaned measurements (deleted atom) are dropped; others survive."""
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    manager = window._measurement_manager
    manager.add("distance", [0, 1])
    manager.add("angle", [0, 1, 2])
    assert len(manager) == 2

    window._structure.delete_atom(2)  # kills the angle measurement

    assert len(manager) == 1
    assert manager.payload()[0][0] == "distance"


def test_measurements_cleared_on_new_file_open(window, monkeypatch):
    """Regression: measurements belong to the previous structure — their
    IDs would silently resolve to unrelated atoms of the new file."""
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"), "first.xyz")
    window._measurement_manager.add("distance", [0, 1])
    assert len(window._measurement_manager) == 1

    _load(window, monkeypatch, molecule("C2H4"), "second.xyz")
    assert len(window._measurement_manager) == 0
    assert window._viewport._measurements == []


def test_bond_order_menu_sets_order(window, monkeypatch):
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    assert not window._bond_order_actions[2].isEnabled()

    window._structure.select_bond(0)
    assert window._bond_order_actions[2].isEnabled()

    window._bond_order_actions[2].trigger()

    assert window._structure.bonds[0].order == 2
    window._bond_order_actions[4].trigger()
    assert window._structure.bonds[0].order == 4


def test_frac_button_label_follows_mode(window, monkeypatch, si_bulk):
    """The button shows the CURRENT mode and clicks toggle — no
    checked/highlighted state."""
    _load(window, monkeypatch, si_bulk, "bulk.vasp")
    tree = window._structure_tree
    assert tree._frac_btn.text() == "Cartesian"  # current mode

    tree._frac_btn.click()
    assert tree._frac_btn.text() == "Fractional"
    assert tree._table.horizontalHeaderItem(2).text() == "fx"

    tree._frac_btn.click()
    assert tree._frac_btn.text() == "Cartesian"
    assert tree._table.horizontalHeaderItem(2).text() == "x (Å)"


def test_measurement_line_vert_counts(window):
    """Lines are dashed in SCREEN space: 10 px dash / 5 px gap, so the
    dash pattern holds at any zoom (fixed world-space dashes collapsed
    into solid lines when zoomed out)."""
    view = window._viewport
    atoms = Atoms("H4", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    view.set_structure(atoms)

    view.set_measurements([("distance", [0, 1], "1.00 Å")])
    wpp = 2.0 * view._cam_distance / max(view.height(), 1)
    verts = view._meas_verts
    assert len(verts) % 2 == 0 and len(verts) > 0
    # dashes cover the line from start to end
    assert np.allclose(verts[0], [0, 0, 0])
    assert np.allclose(verts[-1], [1, 0, 0])
    # every dash but the last is exactly 10 px long (in world units)
    for k in range(0, len(verts) - 2, 2):
        assert np.isclose(np.linalg.norm(verts[k + 1] - verts[k]),
                          10.0 * wpp, atol=1e-6)

    # angle = 2 line segments, dihedral = 3
    view.set_measurements([("angle", [0, 1, 2], "90.00°")])
    assert len(view._meas_pairs) == 2
    view.set_measurements([("dihedral", [0, 1, 2, 3], "60.00°")])
    assert len(view._meas_pairs) == 3

    # zooming out keeps the same on-screen dash length
    view._cam_distance *= 4.0
    view.set_measurements([("distance", [0, 1], "1.00 Å")])
    wpp2 = 2.0 * view._cam_distance / max(view.height(), 1)
    verts2 = view._meas_verts
    assert np.isclose(np.linalg.norm(verts2[1] - verts2[0]),
                      10.0 * wpp2, atol=1e-6)
    assert len(verts2) < len(verts)  # longer dashes → fewer of them


def test_clear_all_pushes_empty_payload_to_viewport(window, monkeypatch):
    """Regression: Clear All must remove the 3D measurement lines too."""
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    window._measurement_manager.add("distance", [0, 1])
    assert len(window._viewport._measurements) == 1
    assert len(window._viewport._meas_verts) > 0

    window._measurement_manager.clear()

    assert window._viewport._measurements == []
    assert len(window._viewport._meas_verts) == 0


def test_multi_bonds_use_thinner_radius(window, monkeypatch):
    """Double/triple bond components are thinner (0.05 Å) so the
    parallel sticks stay visually separate."""
    from vaspen.core.bonds import Bond
    from vaspen.ui import viewport3d as vp

    calls = []

    def fake_cylinder(starts, ends, radius, segments=12, lateral_offset=0.0,
                      color_i=None, color_j=None):
        calls.append((radius, lateral_offset))
        return np.zeros((0, 9), dtype=np.float32)

    monkeypatch.setattr(vp, "_cylinder_verts", fake_cylinder)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    window._viewport.set_structure(atoms, bonds=[Bond(0, 1, 2)])
    assert len(calls) == 2
    assert all(radius == 0.05 for radius, _off in calls)
    assert abs(calls[0][1]) == 0.09


# ----------------------------------------------------------------------
# Phase 2: display styles, frac toggle, properties, selection menu
# ----------------------------------------------------------------------

def test_display_styles_change_radii(window):
    from ase.build import molecule

    view = window._viewport
    view.set_structure(molecule("H2O"))
    base = view._atom_radius.copy()

    view.set_structure_style("cpk")
    assert np.allclose(view._atom_radius, view._covalent_radii)
    assert view._show_bonds is False

    view.set_structure_style("wireframe")
    assert np.allclose(view._atom_radius, view._covalent_radii * 0.25)
    assert view._show_bonds is True

    view.set_structure_style("ball_stick")
    assert np.allclose(view._atom_radius, base)
    assert view._show_bonds is True


def test_display_style_menu_actions(window):
    window._style_actions["cpk"].setChecked(True)
    assert window._viewport.structure_style() == "cpk"
    window.act_show_cell.setChecked(False)
    assert window._viewport._show_cell is False
    window.act_show_labels.setChecked(True)
    assert window._viewport._show_labels is True


def test_tree_fractional_toggle(window, monkeypatch, si_bulk):
    _load(window, monkeypatch, si_bulk, "bulk.vasp")
    tree = window._structure_tree
    assert tree._frac_btn.isEnabled()

    tree._frac_btn.click()
    headers = [tree._table.horizontalHeaderItem(j).text() for j in range(5)]
    assert headers[2] == "fx"
    assert float(tree._table.item(0, 2).text()) < 1.0  # fractional values

    tree._table.item(0, 2).setText("0.5")
    tree._on_cell_changed(0, 2)
    assert np.isclose(window._structure.scaled_positions[0][0], 0.5)

    tree._frac_btn.click()
    headers = [tree._table.horizontalHeaderItem(j).text() for j in range(5)]
    assert headers[2] == "x (Å)"


def test_properties_panel_edits_selection(window, monkeypatch):
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    panel = window._atom_props
    assert "No atom selected" in panel._status_label.text()

    window._structure.select_atom(0)
    assert panel._element_combo.currentText() == "O"
    assert panel._id_label.text() == "0"

    panel._element_combo.setCurrentText("Fe")
    assert window._structure.symbols[0] == "Fe"

    panel._x_spin.setValue(1.5)
    panel._on_cartesian_edited()
    assert np.isclose(window._structure.positions[0][0], 1.5)

    window._structure.set_selection({1, 2})
    assert "2" in panel._status_label.text()


def test_properties_panel_fractional_edit(window, monkeypatch, si_bulk):
    _load(window, monkeypatch, si_bulk, "bulk.vasp")
    panel = window._atom_props
    window._structure.select_atom(0)
    assert panel._fx_spin.isEnabled()

    panel._fx_spin.setValue(0.5)
    panel._on_fractional_edited()
    assert np.isclose(window._structure.scaled_positions[0][0], 0.5)


def test_edit_menu_selection_ops(window, monkeypatch):
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    window._structure.set_selection({1})

    window.act_select_all.trigger()
    assert window._structure.selected_indices == {0, 1, 2}

    window.act_select_invert.trigger()
    assert window._structure.selected_indices == set()

    window._structure.set_selection({0})
    window.act_select_neighbors.trigger()
    assert window._structure.selected_indices == {1, 2}

    window.act_select_connected.trigger()
    assert window._structure.selected_indices == {0, 1, 2}

    window.act_select_none.trigger()
    assert window._structure.selected_indices == set()


def test_add_atom_placement_is_screen_outward(window, monkeypatch):
    """The new atom grows toward the viewer, never into the screen —
    lateral cursor position only steers the angle."""
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.ADD_ATOM)
    window._element_combo.setCurrentText("H")
    view._pick = lambda pos: ("atom", 0)  # the O atom

    _proj, view_mat = view._camera_matrices()
    outward = np.asarray(view_mat[2, :3], dtype=float)  # toward the viewer

    # click without drag, cursor at the widget center
    _click(view, pos=(400.0, 300.0))
    new = window._structure.positions[-1]
    anchor = window._structure.positions[0]
    assert float(np.dot(new - anchor, outward)) > 0

    # drag laterally: still screen-outward
    n_before = window._structure.n_atoms
    start, end = (400.0, 300.0), (480.0, 260.0)
    view.mousePressEvent(_FakeMouseEvent(Qt.LeftButton, start))
    view._last_pos = QPointF(*start)
    view._dragged = True
    view.mouseMoveEvent(_FakeMouseEvent(Qt.LeftButton, end))
    view.mouseReleaseEvent(_FakeMouseEvent(Qt.LeftButton, end))
    assert window._structure.n_atoms == n_before + 1
    new2 = window._structure.positions[-1]
    assert float(np.dot(new2 - anchor, outward)) > 0


# ----------------------------------------------------------------------
# Move tool — axis constraint (X/Y/Z during drag, Phase 3)
# ----------------------------------------------------------------------


def _drag_with_key(view, start, end, key):
    """Press → move → press ``key`` → move → release."""
    view.mousePressEvent(_FakeMouseEvent(Qt.LeftButton, start))
    view._last_pos = QPointF(*start)
    view._dragged = True
    view.mouseMoveEvent(_FakeMouseEvent(Qt.LeftButton, end))
    view.keyPressEvent(_FakeKeyEvent(key))
    view.mouseMoveEvent(_FakeMouseEvent(Qt.LeftButton, end))
    view.mouseReleaseEvent(_FakeMouseEvent(Qt.LeftButton, end))


def test_move_axis_lock_x_non_periodic(window, monkeypatch):
    """X during drag: only the world-x coordinates change."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    _load(window, monkeypatch, atoms)
    view = window._viewport
    view.set_mode(ToolMode.MOVE_ATOM)
    view._pick = lambda pos: ("atom", 0)

    before = window._structure.positions.copy()
    _drag_with_key(view, (400.0, 300.0), (500.0, 400.0), Qt.Key_X)
    after = window._structure.positions

    assert np.allclose(after[:, 1:], before[:, 1:])  # y/z untouched
    assert not np.allclose(after[:, 0], before[:, 0])  # x moved


def test_move_axis_lock_periodic_single_fractional_component(window, monkeypatch, si_bulk):
    """Periodic + X lock: the fractional shift has a single component
    (lattice-vector direction) — group geometry preserved."""
    _load(window, monkeypatch, si_bulk, "bulk.vasp")
    view = window._viewport
    view.set_mode(ToolMode.MOVE_ATOM)
    view._pick = lambda pos: ("atom", 1)  # bonded to atom 0 → group of 2

    before = window._structure.scaled_positions.copy()
    _drag_with_key(view, (400.0, 300.0), (500.0, 400.0), Qt.Key_X)
    after = window._structure.scaled_positions

    frac_delta = after - before
    assert np.allclose(frac_delta[:, 1:], 0.0, atol=1e-12)  # only axis 0
    assert np.any(np.abs(frac_delta[:, 0]) > 1e-9)  # and it moved
    # both atoms share the same fractional delta modulo the per-atom
    # wrap (bonds never tear across the boundary)
    assert np.allclose(frac_delta[0] % 1.0, frac_delta[1] % 1.0, atol=1e-9)


def test_move_axis_lock_toggles_off(window, monkeypatch):
    """Pressing the same key again unlocks the axis."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    _load(window, monkeypatch, atoms)
    view = window._viewport
    view.set_mode(ToolMode.MOVE_ATOM)
    view._pick = lambda pos: ("atom", 0)

    before = window._structure.positions.copy()
    start, end = (400.0, 300.0), (500.0, 400.0)
    view.mousePressEvent(_FakeMouseEvent(Qt.LeftButton, start))
    view._last_pos = QPointF(*start)
    view._dragged = True
    view.mouseMoveEvent(_FakeMouseEvent(Qt.LeftButton, end))
    view.keyPressEvent(_FakeKeyEvent(Qt.Key_X))  # lock
    view.keyPressEvent(_FakeKeyEvent(Qt.Key_X))  # unlock again
    view.mouseMoveEvent(_FakeMouseEvent(Qt.LeftButton, end))
    view.mouseReleaseEvent(_FakeMouseEvent(Qt.LeftButton, end))
    after = window._structure.positions

    # unlocked: the y/z components are free to change too
    assert not np.allclose(after[:, 1:], before[:, 1:])


def test_move_preview_bonds_follow_atoms(window):
    """During a move drag the bond geometry must be re-baked from the
    preview positions (bonds follow the atoms, no lagging behind)."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    view = window._viewport
    view.set_structure(atoms)

    before = view._bond_verts.copy()
    # drag preview: atom 0 moves up by 1 Å (simulating a move drag frame)
    view.preview_atom_positions([0], [[0.0, 1.0, 0.0]])
    assert view._preview_dirty is True
    # what paintGL does each frame with the preview flag set
    view._bake_bond_verts()

    assert not np.allclose(view._bond_verts, before)
    # the re-baked bond connects the MOVED atom (ends near y=1)
    max_y = float(view._bond_verts[:, 1].max())
    assert np.isclose(max_y, 1.0, atol=0.2)
    assert len(view._bond_verts) == len(before)  # same vertex count


def test_bond_bake_uses_current_structure_cell(window):
    """Regression: loading a periodic structure AFTER a molecule must
    bake the bond geometry with the NEW cell/pbc. Stale molecule
    cell/pbc drew Fe bonds as minimum-image-less sticks across the
    whole box (only Detect Bonds — a second set_structure — fixed it)."""
    from ase.io import read

    from vaspen.core.bonds import find_bonds

    view = window._viewport
    view.set_structure(read("examples/benzene.xyz"))  # molecule first
    assert view._pbc == (False, False, False)

    fe = read("examples/Fe_bcc_2x2x2.vasp")
    bonds = find_bonds(
        np.asarray(fe.get_positions(), dtype=float),
        list(fe.get_chemical_symbols()),
        fe.get_cell().array, tuple(fe.get_pbc()))
    view.set_structure(fe, bonds=bonds)

    assert view._pbc == (True, True, True)
    assert np.allclose(view._cell, fe.get_cell().array)
    # every baked stick is a short nearest-neighbor tube, not a
    # box-diagonal spanning the cell (vertices are pos+normal+color —
    # compare positions only)
    for first, count in view._bond_ranges:
        if count == 0:
            continue
        start = view._bond_verts[first, :3]
        end = view._bond_verts[first + count - 1, :3]
        assert np.linalg.norm(end - start) < 2.7


def test_move_single_selection_moves_only_that_atom(window, monkeypatch):
    """Any selection wins (settled 2026-08-13): a SINGLE selected atom
    moves alone — the connected-component fallback does not apply."""
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.MOVE_ATOM)
    window._structure.select_atom(0)  # O only
    view._pick = lambda pos: ("atom", 0)

    before = window._structure.positions.copy()
    _drag(view, (400.0, 300.0), (450.0, 300.0))

    after = window._structure.positions
    assert not np.allclose(after[0], before[0])  # O moved
    assert np.allclose(after[1:], before[1:])    # H stayed


def test_move_selection_wins_when_grabbing_outside(window, monkeypatch):
    """Selection priority holds even when the grabbed atom is NOT in
    the selection."""
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.MOVE_ATOM)
    window._structure.set_selection({1, 2})  # the two H atoms
    view._pick = lambda pos: ("atom", 0)     # grab the O atom

    before = window._structure.positions.copy()
    _drag(view, (400.0, 300.0), (450.0, 300.0))

    after = window._structure.positions
    assert np.allclose(after[0], before[0])       # O stayed
    assert not np.allclose(after[1:], before[1:])  # the selection moved


# ----------------------------------------------------------------------
# Rotate tool (mouse rotation of a selection / molecule, Phase 3)
# ----------------------------------------------------------------------


def test_rotate_tool_keeps_centroid_and_distances(window, monkeypatch):
    """Rotation acts about the group CENTROID (settled with the user):
    the centroid stays put and intra-group distances survive."""
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.ROTATE)
    view._pick = lambda pos: ("atom", 0)  # O → connected component = all 3

    before = window._structure.positions.copy()
    centroid = before.mean(axis=0)
    _drag(view, (400.0, 300.0), (500.0, 340.0))

    after = window._structure.positions
    assert len(window._structure._undo_stack) == 1  # one commit
    assert np.allclose(after.mean(axis=0), centroid, atol=1e-8)  # centroid fixed
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.isclose(np.linalg.norm(after[i] - after[j]),
                              np.linalg.norm(before[i] - before[j]), atol=1e-6)
    assert not np.allclose(after, before)  # something actually rotated


def test_rotate_tool_esc_cancels(window, monkeypatch):
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.ROTATE)
    view._pick = lambda pos: ("atom", 0)
    before = window._structure.positions.copy()

    start, end = (400.0, 300.0), (500.0, 340.0)
    view.mousePressEvent(_FakeMouseEvent(Qt.LeftButton, start))
    view._last_pos = QPointF(*start)
    view._dragged = True
    view.mouseMoveEvent(_FakeMouseEvent(Qt.LeftButton, end))
    view.keyPressEvent(_FakeKeyEvent(Qt.Key_Escape))
    view.mouseReleaseEvent(_FakeMouseEvent(Qt.LeftButton, end))

    assert window._structure._undo_stack == []
    assert np.allclose(window._structure.positions, before)


def test_rotate_tool_prefers_selection(window, monkeypatch):
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.ROTATE)
    window._structure.set_selection({1, 2})  # the two H atoms
    view._pick = lambda pos: ("atom", 0)     # grab the O atom

    before = window._structure.positions.copy()
    _drag(view, (400.0, 300.0), (500.0, 340.0))

    after = window._structure.positions
    assert np.allclose(after[0], before[0])  # O untouched


# ----------------------------------------------------------------------
# Interaction conflicts (second button / Esc during a drag, Phase 3)
# ----------------------------------------------------------------------


def test_right_press_mid_box_select_keeps_drag(window, monkeypatch):
    """Regression: pressing the right button mid box-select must NOT
    hijack the drag state or leave a stuck rubber band."""
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.SELECT)

    start, end = (100.0, 100.0), (300.0, 250.0)
    view.mousePressEvent(_FakeMouseEvent(Qt.LeftButton, start, Qt.ShiftModifier))
    view._last_pos = QPointF(*start)
    view._dragged = True
    view.mouseMoveEvent(_FakeMouseEvent(Qt.LeftButton, end, Qt.ShiftModifier))
    assert view._rubber_rect is not None

    # right button press + release while the left drag is active
    view.mousePressEvent(_FakeMouseEvent(Qt.RightButton, (200.0, 180.0)))
    view.mouseReleaseEvent(_FakeMouseEvent(Qt.RightButton, (200.0, 180.0)))
    assert view._rubber_rect is not None  # drag still in progress
    assert view._press_button == Qt.LeftButton

    # releasing the left button finishes the box select and clears it
    view.mouseReleaseEvent(_FakeMouseEvent(Qt.LeftButton, end, Qt.ShiftModifier))
    assert view._rubber_rect is None


def test_esc_cancels_box_select(window, monkeypatch):
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    view = window._viewport
    view.set_mode(ToolMode.SELECT)

    start, end = (100.0, 100.0), (300.0, 250.0)
    view.mousePressEvent(_FakeMouseEvent(Qt.LeftButton, start, Qt.ShiftModifier))
    view._last_pos = QPointF(*start)
    view._dragged = True
    view.mouseMoveEvent(_FakeMouseEvent(Qt.LeftButton, end, Qt.ShiftModifier))

    view.keyPressEvent(_FakeKeyEvent(Qt.Key_Escape))
    assert view._rubber_rect is None  # band cleared immediately

    view.mouseReleaseEvent(_FakeMouseEvent(Qt.LeftButton, end, Qt.ShiftModifier))
    assert window._structure.selected_indices == set()  # nothing selected


def test_rotate_action_is_on_toolbar(window):
    """Regression: the Rotate mode must be registered in BOTH the mode
    actions AND the edit toolbar (it was once missing from the latter)."""
    buttons = window._edit_buttons
    rotate_btns = [b for b in buttons
                   if b.defaultAction() is window._mode_actions.get(ToolMode.ROTATE)]
    assert len(rotate_btns) == 1
