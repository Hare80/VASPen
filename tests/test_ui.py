"""UI smoke tests — main window launch, live language switch, open path."""

import numpy as np
import pytest
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from PySide6.QtWidgets import QDialog, QMessageBox

from vaspen.core import file_io as fi
from vaspen.ui.main_window import MainWindow


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


class _FakeWrapDialog:
    """Stands in for PeriodicWrapDialog in UI flow tests."""

    def __init__(self, parent=None, accept=True, padding=12.0):
        self.padding = padding
        self.accept_flag = accept

    def exec(self):
        return QDialog.DialogCode.Accepted if self.accept_flag else QDialog.DialogCode.Rejected


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


def test_reset_view_action_restores_default_camera(window, monkeypatch):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]],
                  cell=[10, 10, 10], pbc=True)
    monkeypatch.setattr(fi.FileIO, "read", classmethod(lambda cls, p: atoms))
    window._open_file("fake.xyz")

    view = window._viewport
    view._cam_azimuth = 123.0
    view._cam_elevation = 88.0
    window.act_reset_view.trigger()
    # c axis is [0,0,10] → fitted default is az=0, el=0
    assert abs(view._cam_azimuth) < 1e-6
    assert abs(view._cam_elevation) < 1e-6


def test_far_plane_covers_cell_frame_for_padded_molecule(window, monkeypatch):
    """Regression: the far plane used to cover only atom positions. For a
    molecule wrapped in a padded vacuum box the cell corners extend far
    beyond the atoms, so the frame's back corners were far-plane-clipped
    ("blocked by an invisible face" when looking at a corner)."""
    from ase.build import molecule
    from vaspen.core.structure import wrap_in_padded_cell
    from vaspen.ui import viewport3d as vp

    captured = {}

    def fake_ortho(left, right, bottom, top, near, far):
        captured["far"] = far
        return np.eye(4)

    monkeypatch.setattr(vp, "_ortho", fake_ortho)

    wrapped = wrap_in_padded_cell(molecule("H2O"), 10.0)
    view = window._viewport
    view.resize(800, 600)
    view.set_structure(wrapped)
    view._camera_matrices()

    required = view._cam_distance + float(
        np.linalg.norm(view._cell_verts - view._cam_center, axis=1).max()
    ) + 5.0
    assert captured["far"] >= required - 1e-6


def test_fit_camera_anchors_on_cell_center_and_covers_frame(window):
    """Regression: _fit_camera used to anchor on the atom centroid and fit
    only atom positions — for a molecule in a padded vacuum box the view
    looked off-center and the cell frame overflowed the viewport."""
    from ase.build import molecule
    from vaspen.core.structure import wrap_in_padded_cell

    wrapped = wrap_in_padded_cell(molecule("H2O"), 10.0)
    view = window._viewport
    view.resize(800, 600)
    view.set_structure(wrapped)  # reset_view=True → _fit_camera

    cell = np.asarray(wrapped.get_cell().array, dtype=np.float64)
    center = cell.sum(axis=0) / 2.0
    assert np.allclose(view._cam_center, center)
    corner_radius = float(np.linalg.norm(
        np.asarray(view._cell_verts, dtype=np.float64) - center, axis=1).max())
    # ortho half-height = cam_distance = 1.35 × fit_radius — must cover corners
    assert view._cam_distance >= corner_radius * 1.35 - 1e-6

    # non-periodic molecules still anchor on the atom centroid
    mol = molecule("H2O")
    view.set_structure(mol)
    assert np.allclose(view._cam_center, mol.get_positions().mean(axis=0))
    assert abs(view._cam_azimuth - 45.0) < 1e-6
    assert abs(view._cam_elevation - 30.0) < 1e-6


def test_free_zoom_passes_through_cell_wall(window, monkeypatch):
    """Regression: zoom is completely free (user decision 2026-08-13) —
    the previous wall-limited floor clamped the camera at the cell wall."""
    from PySide6.QtCore import QPoint

    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]],
                  cell=[10, 10, 10], pbc=True)
    monkeypatch.setattr(fi.FileIO, "read", classmethod(lambda cls, p: atoms))
    window._open_file("fake.xyz")
    view = window._viewport

    class FakeWheel:
        def angleDelta(self):
            return QPoint(0, 120 * 60)  # 60 zoom-in ticks

        def accept(self):
            pass

    view.wheelEvent(FakeWheel())
    # 0.9^60 ≈ 0.0018 × initial ~12.7 (cell-corner fit) ≈ 0.02 — far inside
    # the cell (wall at 5 along the view axis); the old wall-limited floor
    # clamped the camera at the cell wall
    assert view._cam_distance < 0.05


def test_kpoints_dialog_spin_slider_sync_and_zno_preview(qtbot):
    """KSPACING numeric input syncs with the slider; ZnO example gives 9 9 5."""
    from vaspen.core.structure import StructureModel
    from vaspen.ui.kpoints_editor import KpointsEditorDialog

    a, c = 3.289, 5.307
    model = StructureModel()
    model.load_atoms(Atoms(
        "Zn2O2",
        positions=[[0.0, 0.0, 0.0]] * 4,  # positions don't affect the mesh
        cell=[[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]],
        pbc=True,
    ))
    dlg = KpointsEditorDialog(model)
    qtbot.addWidget(dlg)

    # default: spin 0.040, slider at position 40
    assert abs(dlg._kspacing_spin.value() - 0.040) < 1e-9
    assert dlg._kspacing_slider.value() == 40
    assert "9 9 5" in dlg._preview.toPlainText()

    # typing into the spinbox moves the slider and updates the preview
    dlg._kspacing_spin.setValue(0.030)
    assert dlg._kspacing_slider.value() == 30
    assert "12 12 7" in dlg._preview.toPlainText()  # ceil(11.703)=12, ceil(6.281)=7
    dlg.close()


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


# ----------------------------------------------------------------------
# Non-periodic → periodic wrapping (UI flow)
# ----------------------------------------------------------------------

def _patch_save_paths(monkeypatch, dest, dialog):
    monkeypatch.setattr(
        "vaspen.ui.main_window.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (str(dest), "")),
    )
    monkeypatch.setattr(
        "vaspen.ui.periodic_wrap_dialog.PeriodicWrapDialog", dialog
    )


def test_save_as_molecule_to_poscar_wraps(window, monkeypatch, water_molecule, tmp_path):
    window._structure.load_atoms(water_molecule)
    dest = tmp_path / "mol.POSCAR"
    _patch_save_paths(monkeypatch, dest,
                      lambda parent=None: _FakeWrapDialog(accept=True, padding=12.0))
    undo_before = len(window._structure._undo_stack)

    window._on_save_as()

    extent = (water_molecule.get_positions().max(axis=0)
              - water_molecule.get_positions().min(axis=0))
    assert window._structure.is_periodic is True
    assert np.allclose(np.diag(window._structure.cell), extent + 24.0)
    assert dest.exists()
    assert window._structure.filepath.endswith("mol.POSCAR")
    assert window._structure.is_dirty is False
    assert len(window._structure._undo_stack) == undo_before + 1


def test_save_as_molecule_cancel_aborts(window, monkeypatch, water_molecule, tmp_path):
    window._structure.load_atoms(water_molecule)
    dest = tmp_path / "mol.POSCAR"
    _patch_save_paths(monkeypatch, dest,
                      lambda parent=None: _FakeWrapDialog(accept=False))
    undo_before = len(window._structure._undo_stack)

    window._on_save_as()

    assert not dest.exists()
    assert window._structure.is_periodic is False
    assert window._structure.is_dirty is False
    assert len(window._structure._undo_stack) == undo_before


def test_save_molecule_to_xyz_no_dialog(window, monkeypatch, water_molecule, tmp_path):
    """xyz is non-periodic — saving a molecule must not show the wrap dialog."""

    class _ExplodingDialog:
        def __init__(self, parent=None):
            raise AssertionError("wrap dialog must not appear for xyz targets")

    window._structure.load_atoms(water_molecule)
    dest = tmp_path / "mol.xyz"
    _patch_save_paths(monkeypatch, dest, _ExplodingDialog)

    window._on_save_as()

    assert dest.exists()
    assert window._structure.is_periodic is False


def test_export_poscar_wraps_molecule(window, monkeypatch, water_molecule, tmp_path):
    window._structure.load_atoms(water_molecule)
    dest = tmp_path / "POSCAR"
    _patch_save_paths(monkeypatch, dest,
                      lambda parent=None: _FakeWrapDialog(accept=True, padding=15.0))

    window._on_export_poscar()

    extent = (water_molecule.get_positions().max(axis=0)
              - water_molecule.get_positions().min(axis=0))
    loaded = ase_read(str(dest), format="vasp")
    assert np.allclose(np.diag(loaded.get_cell()[:]), extent + 30.0, atol=1e-4)
    assert window._structure.is_periodic is True


def test_generate_all_wraps_molecule(window, monkeypatch, water_molecule, tmp_path):
    window._structure.load_atoms(water_molecule)
    monkeypatch.setattr(
        "vaspen.ui.main_window.QMessageBox.question",
        staticmethod(lambda *a, **k: QMessageBox.Yes),
    )
    monkeypatch.setattr(
        "vaspen.ui.main_window.QMessageBox.information",
        staticmethod(lambda *a, **k: None),
    )
    monkeypatch.setattr(
        "vaspen.ui.main_window.QFileDialog.getExistingDirectory",
        staticmethod(lambda *a, **k: str(tmp_path)),
    )
    monkeypatch.setattr(
        "vaspen.ui.periodic_wrap_dialog.PeriodicWrapDialog",
        lambda parent=None: _FakeWrapDialog(accept=True, padding=10.0),
    )

    window._on_generate_all()

    for name in ("INCAR", "KPOINTS", "POSCAR"):
        content = (tmp_path / name).read_text()
        assert content.strip(), name
    poscar_lines = (tmp_path / "POSCAR").read_text().splitlines()
    assert len(poscar_lines) >= 6  # comment + scaling + 3 lattice vectors
    assert window._structure.is_periodic is True


def test_wrap_dialog_uses_remembered_padding(qtbot):
    from vaspen.ui.periodic_wrap_dialog import PeriodicWrapDialog
    from vaspen.utils.config import AppConfig

    config = AppConfig()
    config.remember_wrap_padding = True
    config.wrap_padding = 7.5

    dlg = PeriodicWrapDialog()
    qtbot.addWidget(dlg)
    assert abs(dlg._padding_spin.value() - 7.5) < 1e-9
    assert dlg._remember_check.isChecked()

    dlg._padding_spin.setValue(9.0)
    dlg._remember_check.setChecked(False)
    dlg._on_accept()
    assert abs(config.wrap_padding - 9.0) < 1e-9
    assert config.remember_wrap_padding is False


def test_open_real_poscar_shows_cell(window, qtbot, si_bulk, tmp_path):
    path = tmp_path / "POSCAR"
    ase_write(str(path), si_bulk)

    window._open_file(str(path))

    assert tuple(window._structure.pbc) == (True, True, True)
    assert window._viewport._has_cell is True
    assert window._viewport._cell_verts is not None


def test_open_real_cif_shows_cell(window, qtbot, si_bulk, tmp_path):
    """Regression: ASE's CIF reader drops pbc — opened crystals must show a cell."""
    path = tmp_path / "bulk.cif"
    ase_write(str(path), si_bulk)

    window._open_file(str(path))

    assert tuple(window._structure.pbc) == (True, True, True)
    assert window._viewport._has_cell is True


def test_status_bar_dash_for_molecule(window, monkeypatch, water_molecule, si_bulk):
    monkeypatch.setattr(fi.FileIO, "read",
                        classmethod(lambda cls, p: water_molecule))
    window._open_file("fake.xyz")
    assert "—" in window._cell_label.text()

    monkeypatch.setattr(fi.FileIO, "read", classmethod(lambda cls, p: si_bulk))
    window._open_file("fake2.vasp")
    # si_bulk is the fcc primitive cell (a = 5.43/√2 ≈ 3.84, α=60°)
    assert "a=3.84" in window._cell_label.text()
    assert "—" not in window._cell_label.text()
