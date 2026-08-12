"""UI smoke tests — main window launch, live language switch, open path."""

import numpy as np
import pytest
from ase import Atoms

from vaspen.core import file_io as fi
from vaspen.ui.main_window import MainWindow


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_main_window_launch(window):
    assert window.windowTitle() == "VASPen"
    assert window._viewport is not None


def test_language_switch_is_live_and_roundtrips(window):
    window._switch_language("zh")
    assert window.act_open.text() == "打开(&O)..."
    assert window._menu_calc.title() == "计算(&C)"
    window._switch_language("en")
    assert window.act_open.text() == "&Open..."


def test_open_file_builds_scene_once(window, monkeypatch):
    """Regression: _open_file used to emit structure_modified AND
    structure_loaded, rebuilding the viewport twice."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10])
    monkeypatch.setattr(
        fi.FileIO, "read", classmethod(lambda cls, p: atoms)
    )
    loaded = []
    modified = []
    window._structure.structure_loaded.connect(lambda: loaded.append(1))
    window._structure.structure_modified.connect(lambda: modified.append(1))

    window._open_file("fake.xyz")

    assert len(loaded) == 1
    assert len(modified) == 0
    assert window._structure.filepath == "fake.xyz"
    assert window._structure.is_dirty is False


def test_open_failure_shows_error(window, monkeypatch):
    def boom(cls, p):
        raise RuntimeError("bad file")
    monkeypatch.setattr(fi.FileIO, "read", classmethod(boom))
    # _open_file shows a modal error box — stub it so the test does not block
    monkeypatch.setattr(
        "vaspen.ui.main_window.QMessageBox.critical",
        staticmethod(lambda *args, **kwargs: None),
    )
    window._open_file("fake.xyz")  # must not raise
    assert window._structure.n_atoms == 0


def test_structure_tree_panel_syncs_with_model(window, monkeypatch):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10])
    monkeypatch.setattr(
        fi.FileIO, "read", classmethod(lambda cls, p: atoms)
    )
    window._open_file("fake.xyz")

    tree = window._structure_tree
    assert tree._table.rowCount() == 2
    assert tree._table.item(1, 1).text() == "H"
    assert "H2" in tree._cell_label.text()

    # selecting in 3D highlights the row
    window._structure.select_atom(0)
    assert tree._table.currentRow() == 0

    # selecting a row selects the atom in the model
    tree._table.selectRow(1)
    assert window._structure.selected_index == 1


def test_far_plane_covers_scene_after_edit_without_refit(window, monkeypatch):
    """Regression: after in-place edits (supercell etc.) the camera is
    not re-fit, so the ortho far plane must be computed from the CURRENT
    atom positions — a stale _fit_radius clipped far atoms into a
    cross-section that zooming out could never recover."""
    from vaspen.ui import viewport3d as vp

    captured = {}

    def fake_ortho(left, right, bottom, top, near, far):
        captured["far"] = far
        return np.eye(4)

    monkeypatch.setattr(vp, "_ortho", fake_ortho)

    small = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]],
                  cell=[10, 10, 10], pbc=True)
    big = Atoms(["H"] * 9, positions=np.arange(27).reshape(9, 3) * 2.0,
                cell=[60, 60, 60], pbc=True)

    view = window._viewport
    view.resize(800, 600)
    view.set_structure(small)                    # fits camera to small scene
    view.set_structure(big, reset_view=False)    # simulate supercell: no refit
    view._camera_matrices()

    required = view._cam_distance + float(
        np.linalg.norm(view._atom_pos - view._cam_center, axis=1).max()
    ) + float(view._atom_radius.max()) + 5.0
    assert captured["far"] >= required - 1e-6


def test_undo_action_enabled_after_edit(window, monkeypatch):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10])
    monkeypatch.setattr(
        fi.FileIO, "read", classmethod(lambda cls, p: atoms)
    )
    window._open_file("fake.xyz")
    assert not window.act_undo.isEnabled()

    window._structure.delete_atom(0)
    assert window.act_undo.isEnabled()
    assert not window.act_redo.isEnabled()

    window.act_undo.trigger()
    assert window._structure.n_atoms == 2
    assert window.act_redo.isEnabled()
