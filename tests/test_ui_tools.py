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


def test_create_bond_undo_restores_auto_mode(window, monkeypatch):
    from ase.build import molecule

    _load(window, monkeypatch, molecule("H2O"))
    window._structure.add_bond(1, 2)
    assert window._structure.bond_mode == "manual"

    window.act_undo.trigger()

    assert window._structure.bond_mode == "auto"
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
    assert window.act_auto_bonds.isChecked()
    assert window._structure.bond_mode == "auto"

    window.act_auto_bonds.setChecked(False)
    assert window._structure.bond_mode == "manual"

    # manual edits keep it manual; re-checking re-derives connectivity
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
