"""UI smoke tests — main window launch, live language switch, open path."""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from PySide6.QtCore import QObject, Qt, Signal
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


class _FakeGenerateAllDialog(QObject):
    """Stands in for GenerateAllDialog in main-window flow tests.

    Records what the main window passed in; show() immediately emits
    finished (the real dialog is non-modal and closes on Generate/
    Cancel). The real dialog's behaviour is covered by
    tests/test_generate_all.py — main-window tests only verify the
    gates (periodic wrap, task default) that run before it opens.
    """

    finished = Signal(int)
    constructed: list = []

    def __init__(self, structure, parent=None, preview_callback=None,
                 default_task="scf"):
        super().__init__()
        type(self).constructed.append({
            "structure": structure,
            "preview_callback": preview_callback,
            "default_task": default_task,
        })

    def show(self):
        self.finished.emit(0)  # triggers the main window's cleanup handler


def test_main_window_launch(window):
    assert window.windowTitle() == "VASPen"
    assert window._viewport is not None


def test_welcome_page_shown_when_no_structure(window):
    assert window._central_stack.currentWidget() is window._welcome_page


def test_open_file_switches_to_viewport_and_refreshes_recents(
        window, monkeypatch):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10])
    monkeypatch.setattr(
        fi.FileIO, "read", classmethod(lambda cls, p: atoms)
    )
    window._open_file("fake.xyz")
    assert window._central_stack.currentWidget() is window._viewport
    # The welcome list mirrors the persisted recent files.
    assert window._welcome_page._recent_list.count() == 1
    assert window._welcome_page._recent_list.item(0).text() == "fake.xyz"


def test_new_enters_empty_viewport_session(window, monkeypatch):
    """New Structure = an explicit empty session: the viewport is shown
    (its own empty-state overlay takes over), not the welcome page."""
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.Yes,
    )
    window._on_new()
    assert window._central_stack.currentWidget() is window._viewport
    assert window._structure.n_atoms == 0


def test_new_on_empty_state_skips_confirm(window, monkeypatch):
    """With nothing to discard, New must not ask for confirmation."""
    asked = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.question",
        lambda *args, **kwargs: asked.append(1) or QMessageBox.Yes,
    )
    window._on_new()
    assert asked == []
    assert window._central_stack.currentWidget() is window._viewport


def test_welcome_open_opens_selected_file_without_dialog(
        window, monkeypatch):
    """Open = the SELECTED recent file, no file dialog."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10])
    monkeypatch.setattr(
        fi.FileIO, "read", classmethod(lambda cls, p: atoms)
    )
    def _fail_dialog(*args, **kwargs):
        raise AssertionError("file dialog must not open")
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getOpenFileName", _fail_dialog)
    window._welcome_page.set_recent_files(["D:/vasp/cu111.vasp"])
    window._welcome_page._recent_list.setCurrentRow(0)
    window._on_welcome_open()
    # StructureModel normalizes the stored path (Windows separators).
    assert window._structure.filepath == str(Path("D:/vasp/cu111.vasp"))
    assert window._central_stack.currentWidget() is window._viewport


def test_welcome_browse_opens_file_dialog(window, monkeypatch):
    """Browse = the standard file dialog (same as the File menu)."""
    calls = []

    def _fake_dialog(*args, **kwargs):
        calls.append(1)
        return "", ""

    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getOpenFileName", _fake_dialog)
    window._on_open()
    assert calls == [1]


def test_language_switch_is_live_and_roundtrips(window):
    window._switch_language("zh")
    assert window.act_open.text() == "打开(&O)..."
    assert window._menu_calc.title() == "计算(&C)"
    window._switch_language("en")
    assert window.act_open.text() == "&Open..."


def test_live_theme_switch_rethemes_app_and_viewport(
        window, qapp, monkeypatch):
    """Settings Accept → theme applied app-wide, GL background follows
    the default swap, and the choice persists (mirrors the live
    language-switch pattern)."""
    from PySide6.QtWidgets import QApplication

    from vaspen.core.render_settings import BACKGROUND_DARK
    from vaspen.utils import theme as theme_mod
    from vaspen.utils.config import AppConfig

    class _FakeSettingsDialog:
        def __init__(self, parent=None):
            pass

        def exec(self):
            AppConfig().theme = "dark"
            return QDialog.DialogCode.Accepted

    # _on_preferences imports SettingsDialog inside the function, so
    # patching the module attribute catches it at call time.
    monkeypatch.setattr(
        "vaspen.ui.settings_dialog.SettingsDialog", _FakeSettingsDialog)
    try:
        theme_mod.apply_theme(qapp, "light")
        window._on_preferences()
        assert theme_mod.is_dark()
        assert "#1e1e24" in QApplication.instance().styleSheet()
        assert window._viewport.render_settings().background_color \
            == BACKGROUND_DARK
        assert AppConfig().render_settings.background_color \
            == BACKGROUND_DARK  # persisted
        assert AppConfig().theme == "dark"
    finally:
        theme_mod.apply_theme(qapp, "light")


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


def test_generate_all_wraps_molecule_before_dialog(window, monkeypatch, water_molecule):
    """The periodic-wrap gate runs before the unified dialog opens."""
    window._structure.load_atoms(water_molecule)
    _FakeGenerateAllDialog.constructed.clear()
    monkeypatch.setattr(
        "vaspen.ui.periodic_wrap_dialog.PeriodicWrapDialog",
        lambda parent=None: _FakeWrapDialog(accept=True, padding=10.0),
    )
    monkeypatch.setattr(
        "vaspen.ui.main_window.GenerateAllDialog", _FakeGenerateAllDialog)

    window._on_generate_all()

    assert window._structure.is_periodic is True  # wrapped in place
    assert len(_FakeGenerateAllDialog.constructed) == 1
    call = _FakeGenerateAllDialog.constructed[0]
    assert call["structure"] is window._structure
    # bound-method identity: same underlying function on the same window
    assert call["preview_callback"] == window._preview_image_atoms
    assert call["default_task"] == window._config.default_calc_type


def test_generate_all_wrap_cancel_blocks_dialog(window, monkeypatch, water_molecule):
    window._structure.load_atoms(water_molecule)
    _FakeGenerateAllDialog.constructed.clear()
    monkeypatch.setattr(
        "vaspen.ui.periodic_wrap_dialog.PeriodicWrapDialog",
        lambda parent=None: _FakeWrapDialog(accept=False),
    )
    monkeypatch.setattr(
        "vaspen.ui.main_window.GenerateAllDialog", _FakeGenerateAllDialog)

    window._on_generate_all()

    assert window._structure.is_periodic is False
    assert _FakeGenerateAllDialog.constructed == []  # cancelled before the dialog


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


# ----------------------------------------------------------------------
# Render settings: backward-compat setters + startup persistence
# ----------------------------------------------------------------------

def test_legacy_setters_update_render_settings(window):
    """The pre-existing granular setters keep working — they now mutate
    the internal RenderSettings object."""
    view = window._viewport
    view.set_background_color((0.1, 0.2, 0.3))
    assert view.render_settings().background_color == (0.1, 0.2, 0.3)
    view.set_show_cell(False)
    assert view.render_settings().show_cell is False
    view.set_show_labels(True)
    assert view.render_settings().show_labels is True
    view.set_structure_style("cpk")
    assert view.render_settings().style == "cpk"


def test_startup_applies_persisted_render_settings(window, qtbot):
    """A fresh window reads the persisted render settings and derives
    its View-menu check states from them (not hardcoded values)."""
    from vaspen.core.render_settings import RenderSettings
    from vaspen.utils.config import AppConfig

    cfg = AppConfig()
    rs = RenderSettings.default()
    rs.style = "wireframe"
    rs.show_cell = False
    rs.show_labels = True
    cfg.render_settings = rs
    cfg.sync()

    w2 = MainWindow()
    qtbot.addWidget(w2)
    vp = w2._viewport
    assert vp.structure_style() == "wireframe"
    assert vp.render_settings().show_cell is False
    assert vp.render_settings().show_labels is True
    assert w2._style_actions["wireframe"].isChecked()
    assert not w2.act_show_cell.isChecked()
    assert w2.act_show_labels.isChecked()


def test_display_options_menu_action_exists(window):
    actions = [a.text() for a in window._menu_view.actions()
               if not a.isSeparator()]
    assert "Display &Options..." in actions


def test_viewport_effective_color_override_and_scheme(window):
    """Element color resolution: override > scheme palette > Jmol."""
    from vaspen.core.render_settings import PALETTES, RenderSettings
    from vaspen.ui.viewport3d import element_color

    view = window._viewport
    rs = RenderSettings.default()
    view.set_render_settings(rs)

    fe_jmol = element_color("Fe")
    assert view._effective_atom_color("Fe") == (*fe_jmol, 1.0)

    rs2 = RenderSettings.default()
    rs2.color_scheme = "metal_nonmetal"
    view.set_render_settings(rs2)
    assert view._effective_atom_color("Fe")[:3] == PALETTES["metal_nonmetal"]["Fe"]

    rs3 = RenderSettings.default()
    rs3.atom_colors = {"Fe": [1.0, 0.0, 0.0, 0.4]}
    view.set_render_settings(rs3)
    assert view._effective_atom_color("Fe") == (1.0, 0.0, 0.0, 0.4)
    # other elements fall through to the palette/Jmol
    assert view._effective_atom_color("O") == (*element_color("O"), 1.0)


def test_viewport_structure_symbols(window, monkeypatch):
    from ase import Atoms

    view = window._viewport
    view.set_structure(Atoms("Fe2O3", positions=np.zeros((5, 3))))
    assert view.structure_symbols() == ["Fe", "O"]


# ----------------------------------------------------------------------
# Round-2 fixes: bond alpha sync, label colors
# ----------------------------------------------------------------------

def test_bond_alphas_none_without_overrides(window, monkeypatch):
    from ase import Atoms

    view = window._viewport
    view.set_structure(Atoms("Fe2O", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]]))
    assert view._bond_alphas() is None


def test_bond_alphas_mean_of_atom_opacities(window, monkeypatch):
    """A bond's alpha is the mean of its two atoms' override opacities."""
    from ase import Atoms

    from vaspen.core.render_settings import RenderSettings

    view = window._viewport
    view.set_structure(Atoms("Fe2O", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]]))
    rs = RenderSettings.default()
    rs.atom_colors = {"Fe": [0.878, 0.4, 0.2, 0.4]}  # Fe half-transparent
    view.set_render_settings(rs)

    alphas = view._bond_alphas()
    assert alphas is not None
    assert len(alphas) == len(view._bonds)
    assert alphas[0] == pytest.approx(0.4)   # Fe-Fe bond: (0.4+0.4)/2
    assert alphas[-1] == pytest.approx(0.7)  # Fe-O bond: (0.4+1.0)/2


def test_bond_alphas_none_when_overrides_fully_opaque(window, monkeypatch):
    from ase import Atoms

    from vaspen.core.render_settings import RenderSettings

    view = window._viewport
    view.set_structure(Atoms("Fe2O", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]]))
    rs = RenderSettings.default()
    rs.atom_colors = {"Fe": [0.878, 0.4, 0.2, 1.0]}
    view.set_render_settings(rs)
    assert view._bond_alphas() is None  # all alphas 1.0 → single draw


def test_label_color_contrast_against_background(window):
    """Labels must not blend into the background."""
    view = window._viewport
    # dark background
    view.set_background_color((0.118, 0.118, 0.141))
    dark = view._label_color("H")   # white H stays white on dark
    assert dark.redF() > 0.9 and dark.greenF() > 0.9
    n_dark = view._label_color("N")  # dark blue N gets brightened
    n_lum = 0.299 * n_dark.redF() + 0.587 * n_dark.greenF() + 0.114 * n_dark.blueF()
    assert n_lum > 0.35

    view.set_background_color((1.0, 1.0, 1.0))  # white background (default)
    h_light = view._label_color("H")  # white H must be darkened
    h_lum = 0.299 * h_light.redF() + 0.587 * h_light.greenF() + 0.114 * h_light.blueF()
    assert h_lum <= 0.71  # luminance gap 0.3 → capped at 0.7 (float fuzz)


def test_label_scale_zooms_with_camera(window):
    """Labels grow when zoomed in, shrink when zoomed out (clamped)."""
    view = window._viewport
    view._fit_distance = 10.0
    view._cam_distance = 10.0
    assert view._label_scale() == pytest.approx(1.0)
    view._cam_distance = 2.5  # zoomed in 4× → sqrt(4) = 2×
    assert view._label_scale() == pytest.approx(2.0)
    view._cam_distance = 100.0  # zoomed way out → clamped floor
    assert view._label_scale() == pytest.approx(0.6)
    view._cam_distance = 0.5  # extreme zoom-in → clamped ceiling
    assert view._label_scale() == pytest.approx(2.5)


def test_orbit_free_rotation_no_clamp(window):
    """Arcball orbit: vertical drag rotates around the camera-right
    axis even when the camera sits at the elevation pole (c-axis
    top-down view) — the old ±89.9° clamp stalled there."""
    view = window._viewport
    # near-top-down view (c vertical): elevation pinned at the pole
    view._cam_azimuth = 25.0
    view._cam_elevation = 89.9
    view._cam_up = np.array([0.0, 0.0, 1.0])
    view._cam_center = np.zeros(3)
    view._cam_distance = 10.0

    view.orbit(0.0, 8.0)  # drag up → elevation should DECREASE smoothly
    assert view._cam_elevation < 89.9
    assert abs(view._cam_elevation - 89.9) > 3.0  # no clamp stall


def test_orbit_matches_legacy_direction_small_drag(window):
    """Small drags rotate in the same sense as before the arcball
    rewrite: right → azimuth down, up → elevation up."""
    view = window._viewport
    view._cam_azimuth = 40.0
    view._cam_elevation = 20.0
    view._cam_up = np.array([0.0, 1.0, 0.0])
    view._cam_center = np.zeros(3)
    view._cam_distance = 10.0

    view.orbit(4.0, 0.0)
    assert view._cam_azimuth == pytest.approx(40.0 - 2.0, abs=1.0)
    view.orbit(0.0, 4.0)
    assert view._cam_elevation > 20.0  # drag up → elevation up


def test_orbit_pole_drag_no_spin(window):
    """Regression: repeated drags near the elevation pole must NOT flip
    the view ~180° per frame (the crazy-spin bug — the camera up used
    to be recomputed from the pole-unstable world-up projection).

    Note: the azimuth parametrization itself is nearly singular at the
    pole (a smooth 2° tilt swings the azimuth value wildly), so the
    stability criterion is the ANGLE between consecutive look
    directions — must stay ~2° per frame, never ~180°.
    """
    import math

    view = window._viewport
    view._cam_azimuth = 25.0
    view._cam_elevation = 89.9
    view._cam_up = np.array([0.0, 0.0, 1.0])
    view._cam_center = np.zeros(3)
    view._cam_distance = 10.0

    def look_dir() -> np.ndarray:
        az = math.radians(view._cam_azimuth)
        el = math.radians(view._cam_elevation)
        d = np.array([math.cos(el) * math.sin(az), math.sin(el),
                      math.cos(el) * math.cos(az)])
        return -d  # look direction = opposite of the eye direction

    prev = look_dir()
    for _ in range(12):
        view.orbit(0.0, 4.0)
        cur = look_dir()
        angle = math.degrees(math.acos(float(np.clip(prev @ cur, -1.0, 1.0))))
        assert angle < 3.5  # ~2°/frame, no 180° flips
        prev = cur
    assert view._cam_elevation < 85.0  # kept rotating past the pole


def test_cell_frame_quads_keep_constant_screen_width(window):
    """Regression: frame quads were offset along the screen-right axis
    projected onto the edge plane, so their on-screen width thinned
    toward zero as an edge turned parallel to screen-right — the 4
    parallel edges of one cell direction vanished together, and near
    that angle the sub-pixel width rendered as dashes. Every on-screen
    edge must keep its full pixel width at every orientation.
    """
    import math

    view = window._viewport
    corners = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=float)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6),
             (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
    view._cell_verts = np.array([corners[[i, j]] for i, j in pairs],
                                dtype=float)
    view._render_settings.cell_line_width = 1.0  # floor is 1.5 px
    view._cam_up = np.array([0.0, 1.0, 0.0])
    view._cam_center = np.zeros(3)
    view._cam_distance = 10.0
    view._cam_elevation = 10.0

    w, h = view.width(), view.height()
    for az in range(0, 360, 3):
        view._cam_azimuth = float(az)
        view._rebuild_cell_frame_verts()
        proj, vmat = view._camera_matrices()
        mvp = proj @ vmat
        right, up = vmat[:3, :3][0], vmat[:3, :3][1]
        quads = view._cell_frame_verts.reshape(-1, 3)
        for k, (p0, p1) in enumerate(pairs):
            e = corners[p1] - corners[p0]
            e /= np.linalg.norm(e)
            if math.hypot(float(right @ e), float(up @ e)) < 0.01:
                continue  # end-on edge → dot quad, orientation-safe
            # max distance from any quad vertex to the edge's
            # projected segment = half of the on-screen width
            verts = quads[k * 6:k * 6 + 6]
            d_max = 0.0
            for v in verts:
                clip = mvp @ np.append(v, 1.0)
                px = np.array([(clip[0] / clip[3] + 1) / 2 * w,
                               (clip[1] / clip[3] + 1) / 2 * h])
                a = np.append(corners[p0], 1.0)
                b = np.append(corners[p1], 1.0)
                ca, cb = mvp @ a, mvp @ b
                pa = np.array([(ca[0] / ca[3] + 1) / 2 * w,
                               (ca[1] / ca[3] + 1) / 2 * h])
                pb = np.array([(cb[0] / cb[3] + 1) / 2 * w,
                               (cb[1] / cb[3] + 1) / 2 * h])
                seg = pb - pa
                seg_len = float(np.linalg.norm(seg))
                if seg_len < 1e-9:
                    d = float(np.linalg.norm(px - pa))
                else:
                    t = float(np.clip((px - pa) @ seg / seg_len**2, 0.0, 1.0))
                    d = float(np.linalg.norm(px - (pa + t * seg)))
                d_max = max(d_max, d)
            # half-width = 1.5 px / 2; orientation must not thin it
            assert 0.65 <= d_max <= 0.85, (
                f"az={az} edge {k}: on-screen half-width {d_max:.3f} px "
                "(expected ~0.75)")


# ----------------------------------------------------------------------
# Partial occupancy: composition display + POSCAR save confirmation
# ----------------------------------------------------------------------

def test_structure_tree_shows_composition(window, monkeypatch, disordered_atoms):
    """The left panel shows an MS-style Composition line with element
    percentages (and vacancy) for disordered structures."""
    monkeypatch.setattr(
        fi.FileIO, "read", classmethod(lambda cls, p: disordered_atoms))
    window._open_file("fake.cif")

    tree = window._structure_tree
    label = tree._cell_label.text()
    assert "Composition:" in label
    assert "Fe 68.3%" in label
    assert "Ni 16.7%" in label
    assert "Co 6.7%" in label
    assert "Vacancy 8.3%" in label


def test_atom_properties_site_composition(window, monkeypatch, disordered_atoms):
    """The properties panel shows the selected site's composition; it is
    hidden for atoms without partial occupancy."""
    monkeypatch.setattr(
        fi.FileIO, "read", classmethod(lambda cls, p: disordered_atoms))
    window._open_file("fake.cif")
    panel = window._atom_props

    window._structure.select_atom(0)  # Fe 0.5 / Ni 0.5
    assert not panel._composition_label.isHidden()
    assert "Fe 50.0%" in panel._composition_label.text()
    assert "Ni 50.0%" in panel._composition_label.text()

    window._structure.select_atom(2)  # Fe 0.75 → vacancy
    assert not panel._composition_label.isHidden()
    assert "Fe 75.0%" in panel._composition_label.text()
    assert "Vacancy 25.0%" in panel._composition_label.text()

    # plain structure → no composition row
    from ase import Atoms as _Atoms
    monkeypatch.setattr(
        fi.FileIO, "read",
        classmethod(lambda cls, p: _Atoms("H2",
                                          positions=[[0, 0, 0], [0.74, 0, 0]],
                                          cell=[10, 10, 10])))
    window._open_file("fake.xyz")
    assert panel._composition_label.isHidden()


def test_poscar_save_confirm_disorder(window, monkeypatch, disordered_atoms,
                                      tmp_path):
    """Saving a disordered structure to POSCAR warns and requires
    confirmation; a clean structure saves without a warning."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    monkeypatch.setattr(
        fi.FileIO, "read", classmethod(lambda cls, p: disordered_atoms))
    monkeypatch.setattr(window, "_ensure_periodic_for", lambda fmt: True)
    window._open_file("fake.cif")

    target = tmp_path / "out.vasp"

    # Cancel → no file written
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "")))
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: QMessageBox.Cancel))
    window._on_export_poscar()
    assert not target.exists()

    # Confirm → file written
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: QMessageBox.Save))
    window._on_export_poscar()
    assert target.exists()

    # Clean structure → no warning at all
    from ase import Atoms as _Atoms
    monkeypatch.setattr(
        fi.FileIO, "read",
        classmethod(lambda cls, p: _Atoms("H2",
                                          positions=[[0, 0, 0], [0.74, 0, 0]],
                                          cell=[10, 10, 10])))
    window._open_file("fake.xyz")

    def _boom(*a, **k):
        raise AssertionError("warning shown for a clean structure")
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    window._on_export_poscar()
    assert target.exists()


def test_save_as_default_filename(window, monkeypatch, disordered_atoms):
    """Save As suggests a default filename: the loaded file's stem, or
    the (sanitized) chemical formula for unnamed structures."""
    from PySide6.QtWidgets import QFileDialog

    captured = {}

    def _fake_save(parent, title, default_path, filt, selected):
        captured["path"] = default_path
        captured["selected"] = selected
        return ("", "")

    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(_fake_save))

    # unnamed structure → formula-based name
    window._structure.load_atoms(
        Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10]))
    window._on_save_as()
    assert Path(captured["path"]).name == "H2.cif"
    # the pre-selected filter is the first (CIF) category
    assert captured["selected"].startswith("CIF — ")

    # loaded file → its stem
    monkeypatch.setattr(
        fi.FileIO, "read", classmethod(lambda cls, p: Atoms("H2")))
    window._open_file("water.xyz")
    window._on_save_as()
    assert Path(captured["path"]).name == "water.cif"

    # disordered formula → decimal points become underscores
    window._structure.reset_filepath()  # unnamed/derived structure
    window._structure.load_atoms(disordered_atoms)
    window._on_save_as()
    assert Path(captured["path"]).name == "Fe2_05Ni0_5Co0_2.cif"


def test_supercell_disorder_warns(window, monkeypatch, disordered_atoms):
    """Creating a supercell from a disordered structure requires
    confirmation (make_supercell drops the occupancy info silently)."""
    from PySide6.QtWidgets import QMessageBox

    from vaspen.ui.supercell_dialog import SupercellDialog as _RealSupercell

    monkeypatch.setattr(
        fi.FileIO, "read", classmethod(lambda cls, p: disordered_atoms))
    window._open_file("fake.cif")
    n0 = window._structure.n_atoms

    class _FakeDlg:
        Accepted = _RealSupercell.Accepted

        def __init__(self, parent=None):
            self.factors = (2, 2, 2)

        def exec(self):
            return _RealSupercell.Accepted

    monkeypatch.setattr("vaspen.ui.supercell_dialog.SupercellDialog",
                        _FakeDlg)

    # Cancel → structure unchanged, disorder intact
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Cancel))
    window._on_supercell()
    assert window._structure.n_atoms == n0
    assert window._structure.has_disorder

    # Confirm → supercell applied, occupancy cleared
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Yes))
    window._on_supercell()
    assert window._structure.n_atoms == n0 * 8
    assert not window._structure.has_disorder

    # Clean structure → no warning at all
    def _boom(*a, **k):
        raise AssertionError("warning shown for a clean structure")
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    from ase import Atoms as _Atoms
    monkeypatch.setattr(
        fi.FileIO, "read",
        classmethod(lambda cls, p: _Atoms("H2",
                                          positions=[[0, 0, 0], [0.74, 0, 0]],
                                          cell=[10, 10, 10])))
    window._open_file("fake.xyz")
    n_clean = window._structure.n_atoms
    window._on_supercell()
    assert window._structure.n_atoms == n_clean * 8


# ----------------------------------------------------------------------
# Surface dialog (non-modal wiring)
# ----------------------------------------------------------------------


class _FakeSurfaceDialog(QObject):
    """Stands in for the non-modal SurfaceDialog in UI flow tests."""

    accepted = Signal()
    finished = Signal(int)

    def __init__(self, structure, viewport, parent=None):
        super().__init__(parent)
        self.result_structure = None
        self.shown = False

    def show(self):
        self.shown = True


def test_surface_dialog_modeless_wiring(window, monkeypatch, si_bulk):
    from vaspen.core.structure import StructureModel
    from vaspen.ui import surface_dialog as sd
    from vaspen.ui.tools import ToolMode

    window._structure.load_atoms(si_bulk)
    monkeypatch.setattr(sd, "SurfaceDialog", _FakeSurfaceDialog)
    window._on_surface()

    dlg = window._surface_dialog
    assert isinstance(dlg, _FakeSurfaceDialog)
    assert dlg.shown is True
    # structure-editing entry points are paused while the dialog lives
    assert window.act_supercell.isEnabled() is False
    assert window._mode_actions[ToolMode.ADD_ATOM].isEnabled() is False
    assert window._element_label.isEnabled() is False
    assert window._element_btn.isEnabled() is False
    assert window._element_more_btn.isEnabled() is False
    assert window._dock_structure.isEnabled() is False
    assert window.act_reset_view.isEnabled() is True  # view-only stays live

    slab = Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                 cell=[10.0, 10.0, 10.0], pbc=True)
    dlg.result_structure = StructureModel(slab)
    dlg.accepted.emit()
    assert window._structure.n_atoms == 2

    dlg.finished.emit(0)
    assert window._surface_dialog is None
    assert window.act_supercell.isEnabled() is True
    assert window._mode_actions[ToolMode.ADD_ATOM].isEnabled() is True
    assert window._element_label.isEnabled() is True
    assert window._element_btn.isEnabled() is True
    assert window._element_more_btn.isEnabled() is True
    assert window._dock_structure.isEnabled() is True


def test_element_picker_next_to_add_atom(window):
    """The element picker sits immediately right of the Add Atom button."""
    flow = window._element_btn.parentWidget().layout()
    add_atom_btn = window._edit_buttons[1]  # SELECT=0, ADD_ATOM=1
    assert flow.indexOf(add_atom_btn) + 1 == flow.indexOf(window._element_label)
    assert flow.indexOf(add_atom_btn) + 2 == flow.indexOf(window._element_btn)
    assert flow.indexOf(add_atom_btn) + 3 == flow.indexOf(window._element_more_btn)


def test_element_picker_shortlist_and_periodic_table(window, monkeypatch):
    """Shortlist menu holds the 9 common elements (default C); the
    ellipsis button opens the periodic table for any other element."""
    from PySide6.QtWidgets import QDialog

    assert window._element_btn.text() == "C"
    assert window._current_element == "C"
    assert [a.text() for a in window._element_btn._actions] == [
        "C", "H", "O", "N", "S", "P", "F", "Cl", "Si"]
    assert window._element_btn._actions[0].isChecked()

    window._element_btn._actions[2].trigger()  # pick O from the menu
    assert window._current_element == "O"
    assert window._element_btn.text() == "O"
    assert window._viewport.current_element == "O"
    assert window._element_btn._actions[2].isChecked()

    class _FakePT(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.selected_symbol = None

        def exec(self):
            self.selected_symbol = "Au"
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        "vaspen.ui.periodic_table_dialog.PeriodicTableDialog", _FakePT)
    window._on_pick_element()
    assert window._current_element == "Au"
    assert window._element_btn.text() == "Au"
    assert window._viewport.current_element == "Au"
    # "Au" is not in the shortlist — no menu item is checked
    assert not any(a.isChecked() for a in window._element_btn._actions)


def test_surface_dialog_nonperiodic_guard(window, monkeypatch, water_molecule):
    from vaspen.ui import surface_dialog as sd

    window._structure.load_atoms(water_molecule)
    infos = []
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **k: infos.append(a)))

    def _boom(*a, **k):
        raise AssertionError("dialog constructed for a non-periodic structure")
    monkeypatch.setattr(sd, "SurfaceDialog", _boom)

    window._on_surface()  # guard shows an info box, never constructs
    assert len(infos) == 1
    assert window._surface_dialog is None


# ----------------------------------------------------------------------
# VASP coordinate mode follows the Preferences setting (no dialogs)
# ----------------------------------------------------------------------

def _poscar_keyword(path):
    """Coordinate keyword line ("Direct" or "Cartesian") of a POSCAR."""
    for line in Path(path).read_text().splitlines():
        if line.strip() in ("Direct", "Cartesian"):
            return line.strip()
    raise AssertionError(f"No coordinate keyword in {path}")


def test_save_as_vasp_follows_coords_setting(window, monkeypatch, si_bulk, tmp_path):
    """Save As writes the coordinate form selected in Preferences."""
    window._structure.load_atoms(si_bulk)
    dest = tmp_path / "bulk.POSCAR"
    monkeypatch.setattr(
        "vaspen.ui.main_window.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (str(dest), "")),
    )

    window._config.poscar_coords_direct = True  # default: fractional
    window._on_save_as()
    assert _poscar_keyword(dest) == "Direct"

    window._config.poscar_coords_direct = False  # Cartesian
    window._on_save_as()
    assert _poscar_keyword(dest) == "Cartesian"


def test_export_poscar_follows_coords_setting(window, monkeypatch, si_bulk, tmp_path):
    window._structure.load_atoms(si_bulk)
    dest = tmp_path / "POSCAR"
    monkeypatch.setattr(
        "vaspen.ui.main_window.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (str(dest), "")),
    )

    window._config.poscar_coords_direct = True
    window._on_export_poscar()
    assert _poscar_keyword(dest) == "Direct"


def test_generate_all_passes_default_task(window, monkeypatch, si_bulk):
    """The configured default calculation type reaches the dialog."""
    window._structure.load_atoms(si_bulk)
    window._config.default_calc_type = "band"
    _FakeGenerateAllDialog.constructed.clear()
    monkeypatch.setattr(
        "vaspen.ui.main_window.GenerateAllDialog", _FakeGenerateAllDialog)

    window._on_generate_all()

    assert len(_FakeGenerateAllDialog.constructed) == 1
    assert _FakeGenerateAllDialog.constructed[0]["default_task"] == "band"


def test_save_as_non_vasp_ignores_coords_setting(window, monkeypatch, si_bulk, tmp_path):
    """CIF output is untouched by the POSCAR coordinate preference."""
    window._structure.load_atoms(si_bulk)
    dest = tmp_path / "bulk.cif"
    monkeypatch.setattr(
        "vaspen.ui.main_window.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (str(dest), "")),
    )

    window._config.poscar_coords_direct = True
    window._on_save_as()
    assert "_cell_length_a" in dest.read_text()


# ----------------------------------------------------------------------
# Freeze / fixed atoms (VASP selective dynamics)
# ----------------------------------------------------------------------

def _load_h3(window, monkeypatch):
    atoms = Atoms("H3", positions=[[0, 0, 0], [0.8, 0, 0], [1.6, 0, 0]],
                  cell=[10, 10, 10], pbc=True)
    monkeypatch.setattr(
        fi.FileIO, "read", classmethod(lambda cls, p: atoms)
    )
    window._open_file("fake.xyz")
    return window._structure


def test_freeze_actions_enabled_only_with_selection(window):
    assert not window.act_freeze.isEnabled()
    assert not window.act_unfreeze.isEnabled()
    window._structure.load_atoms(Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    assert not window.act_freeze.isEnabled()  # load clears selection
    assert not window.act_unfreeze.isEnabled()
    window._structure.select_atom(0)
    assert window.act_freeze.isEnabled()
    assert window.act_unfreeze.isEnabled()
    window._structure.clear_selection()
    assert not window.act_freeze.isEnabled()
    assert not window.act_unfreeze.isEnabled()


def test_freeze_and_unfreeze_actions(window, monkeypatch):
    model = _load_h3(window, monkeypatch)
    model.select_atom(0)

    window.act_freeze.trigger()
    assert model.is_fixed(0)
    assert model.fixed_flags[0].tolist() == [True, True, True]
    assert not model.is_fixed(1)
    assert len(model._undo_stack) == 1

    window.act_unfreeze.trigger()
    assert not model.any_fixed
    assert len(model._undo_stack) == 2

    # undo/redo steps restore the freeze state
    model.undo()
    assert model.is_fixed(0)
    model.redo()
    assert not model.is_fixed(0)


def test_unfreeze_clears_partial_flags(window, monkeypatch):
    model = _load_h3(window, monkeypatch)
    model.select_atom(0)
    model.set_fixed(0, np.array([True, False, True]))  # partial freeze
    undo_len = len(model._undo_stack)

    window.act_unfreeze.trigger()

    assert model.fixed_flags[0].tolist() == [False, False, False]
    assert not model.is_fixed(0)
    assert len(model._undo_stack) == undo_len + 1


def test_freeze_mixed_selection_freezes_all(window, monkeypatch):
    model = _load_h3(window, monkeypatch)
    model.set_fixed(0, True)
    model.set_selection({0, 1})
    undo_len = len(model._undo_stack)

    window.act_freeze.trigger()  # not ALL frozen → freeze the rest

    assert model.is_fixed(0) and model.is_fixed(1)
    assert not model.is_fixed(2)
    assert len(model._undo_stack) == undo_len + 1


def test_properties_panel_fixed_group_reflects_and_edits(window, monkeypatch):
    model = _load_h3(window, monkeypatch)
    panel = window._atom_props
    model.select_atom(0)

    # initially free
    assert not panel._fixed_check.isChecked()
    assert all(chk.isEnabled() for chk in
               (panel._fx_fixed, panel._fy_fixed, panel._fz_fixed))

    # freeze via the panel → model updates in one undo step
    panel._fixed_check.setChecked(True)
    assert model.fixed_flags[0].tolist() == [True, True, True]
    assert len(model._undo_stack) == 1
    # frozen atom: coordinate fields read-only, element field editable
    assert not panel._x_edit.isEnabled()
    assert panel._element_edit.isEnabled()

    # per-direction edit
    panel._fy_fixed.setChecked(False)
    assert model.fixed_flags[0].tolist() == [True, False, True]
    assert model.is_fixed(0)  # any direction fixed counts

    # unfreeze via master checkbox
    panel._fixed_check.setChecked(False)
    assert not model.any_fixed
    assert panel._x_edit.isEnabled()


def test_properties_panel_direction_boxes_need_cell(window, monkeypatch):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])  # molecule
    monkeypatch.setattr(
        fi.FileIO, "read", classmethod(lambda cls, p: atoms)
    )
    window._open_file("fake.xyz")
    window._structure.select_atom(0)
    panel = window._atom_props

    assert panel._fixed_check.isEnabled()
    assert not panel._fx_fixed.isEnabled()
    assert not panel._fy_fixed.isEnabled()
    assert not panel._fz_fixed.isEnabled()


def test_open_poscar_with_selective_dynamics_restores_flags(window, monkeypatch):
    """A POSCAR with "Selective dynamics" loads with the flags adopted."""
    from ase.constraints import FixAtoms
    atoms = Atoms("H3", positions=[[0, 0, 0], [0.8, 0, 0], [1.6, 0, 0]],
                  cell=[10, 10, 10], pbc=True)
    atoms.set_constraint([FixAtoms(indices=[0])])
    monkeypatch.setattr(
        fi.FileIO, "read", classmethod(lambda cls, p: atoms)
    )
    window._open_file("fake.vasp")
    assert window._structure.is_fixed(0)
    assert not window._structure.is_fixed(1)
    assert window._structure.atoms.constraints == []


def test_structure_tree_frozen_rows_read_only_coords(window, monkeypatch):
    model = _load_h3(window, monkeypatch)
    model.set_fixed(0, True)
    tree = window._structure_tree
    # x/y/z columns (2..4) not editable for the frozen row
    assert not (tree._table.item(0, 2).flags() & Qt.ItemIsEditable)
    # element column stays editable
    assert tree._table.item(0, 1).flags() & Qt.ItemIsEditable
    # free atom rows keep editable coordinates
    assert tree._table.item(1, 2).flags() & Qt.ItemIsEditable


# ----------------------------------------------------------------------
# Tools → Wrap in Periodic Cell (molecule → periodic, menu action)
# ----------------------------------------------------------------------

def test_wrap_periodic_action_converts_molecule(window, monkeypatch, water_molecule):
    window._structure.load_atoms(water_molecule, "mol.xyz")
    monkeypatch.setattr(
        "vaspen.ui.periodic_wrap_dialog.PeriodicWrapDialog",
        lambda parent=None: _FakeWrapDialog(accept=True, padding=8.0),
    )
    undo_before = len(window._structure._undo_stack)

    window.act_wrap_periodic.trigger()

    extent = (water_molecule.get_positions().max(axis=0)
              - water_molecule.get_positions().min(axis=0))
    assert window._structure.is_periodic is True
    assert np.allclose(np.diag(window._structure.cell), extent + 16.0)
    assert len(window._structure._undo_stack) == undo_before + 1  # one undo step
    assert window._structure.filepath == "mol.xyz"  # conversion keeps the path
    assert "Wrapped in periodic cell" in window._status_label.text()


def test_wrap_periodic_action_refuses_periodic(window, monkeypatch, si_bulk):
    window._structure.load_atoms(si_bulk)
    opened = []
    monkeypatch.setattr(
        "vaspen.ui.periodic_wrap_dialog.PeriodicWrapDialog",
        lambda parent=None: opened.append(1),
    )
    infos = []
    monkeypatch.setattr(
        "vaspen.ui.main_window.QMessageBox.information",
        staticmethod(lambda *args, **kwargs: infos.append(args)),
    )
    before = window._structure.positions.copy()

    window.act_wrap_periodic.trigger()

    assert opened == []  # dialog never shown
    assert infos  # informational message shown
    assert np.allclose(window._structure.positions, before)


def test_wrap_periodic_action_refuses_empty(window, monkeypatch):
    opened = []
    monkeypatch.setattr(
        "vaspen.ui.periodic_wrap_dialog.PeriodicWrapDialog",
        lambda parent=None: opened.append(1),
    )
    infos = []
    monkeypatch.setattr(
        "vaspen.ui.main_window.QMessageBox.information",
        staticmethod(lambda *args, **kwargs: infos.append(args)),
    )

    window.act_wrap_periodic.trigger()

    assert opened == []
    assert infos


# ----------------------------------------------------------------------
# MAGMOM — properties panel + INCAR editor injection
# ----------------------------------------------------------------------

def test_properties_panel_magmom_group_edits(window, monkeypatch):
    model = _load_h3(window, monkeypatch)
    panel = window._atom_props
    model.select_atom(0)

    # initially unset
    assert not panel._magmom_check.isChecked()
    assert panel._magmom_edit.text() == ""
    assert model.magmom(0) is None

    # check → prefilled with 1.0, focus lands in the value field
    panel._magmom_check.setChecked(True)
    assert model.magmom(0) == 1.0
    assert panel._magmom_edit.text() == "1"
    # the prefill is selected so typing replaces it (hasFocus itself
    # depends on the test process being the foreground window — skip)
    assert panel._magmom_edit.selectedText() == "1"

    # 1-based display, matching the structure tree's Index column
    assert "Atom 1 (H)" in panel._status_label.text()
    assert panel._id_label.text() == "1"

    # edit the value
    panel._magmom_edit.setText("5")
    panel._on_magmom_edited()
    assert model.magmom(0) == 5.0

    # invalid input reverts
    panel._magmom_edit.setText("abc")
    panel._on_magmom_edited()
    assert model.magmom(0) == 5.0
    assert panel._magmom_edit.text() == "5"

    # uncheck → unset
    panel._magmom_check.setChecked(False)
    assert model.magmom(0) is None
    assert not model.any_magmom


def test_properties_panel_magmom_multi_select(window, monkeypatch):
    model = _load_h3(window, monkeypatch)
    panel = window._atom_props
    model.set_selection([0, 1])

    # batch set via checkbox → 1.0 for both
    assert panel._magmom_check.isEnabled()
    panel._magmom_check.setChecked(True)
    assert model.magmom(0) == 1.0 and model.magmom(1) == 1.0

    # batch edit the value → both updated
    panel._magmom_edit.setText("-5")
    panel._on_magmom_edited()
    assert model.magmom(0) == -5.0 and model.magmom(1) == -5.0

    # mixed state: unset atom 0 via the model → shown as unchecked
    # (no tristate — a click always checks directly)
    model.set_selection([0, 1, 2])
    model.set_magmom(0, None)
    assert not panel._magmom_check.isChecked()

    # checking a mixed selection prefills only the unset atoms
    panel._magmom_check.setChecked(True)
    assert model.magmom(0) == 1.0   # refilled
    assert model.magmom(1) == -5.0  # existing value survives
    assert model.magmom(2) == 1.0


def test_incar_editor_injects_magmoms(qtbot):
    from vaspen.core.structure import StructureModel
    from vaspen.ui.incar_editor import IncarEditorDialog

    model = StructureModel()
    model.load_atoms(Atoms(
        "Fe2O3",
        positions=np.eye(5, 3) * 1.5,
        cell=[5, 5, 5],
        pbc=True,
    ))
    model.set_magmom([0, 1], np.array([5.0, -5.0]))

    dlg = IncarEditorDialog(model)
    qtbot.addWidget(dlg)
    dlg._load_preset("scf")

    table_tags = {
        dlg._table.item(r, 0).text(): dlg._table.item(r, 1).text()
        for r in range(dlg._table.rowCount())
    }
    assert table_tags["MAGMOM"].split() == ["5", "-5", "0", "0", "0"]
    assert table_tags["ISPIN"] == "2"
    # preview shows the active MAGMOM line, not the commented suggestion
    preview = dlg._preview.toPlainText()
    assert "  MAGMOM =  5 -5 0 0 0" in preview
    assert not any(l.startswith("  # MAGMOM") for l in preview.splitlines())
    dlg.close()


def test_incar_editor_without_magmoms_keeps_ispin_1(qtbot):
    from vaspen.core.structure import StructureModel
    from vaspen.ui.incar_editor import IncarEditorDialog

    model = StructureModel()
    model.load_atoms(Atoms(
        "H3", positions=np.eye(3, 3) * 1.5, cell=[5, 5, 5], pbc=True,
    ))

    dlg = IncarEditorDialog(model)
    qtbot.addWidget(dlg)
    dlg._load_preset("scf")

    table_tags = {
        dlg._table.item(r, 0).text(): dlg._table.item(r, 1).text()
        for r in range(dlg._table.rowCount())
    }
    assert "MAGMOM" not in table_tags
    assert table_tags["ISPIN"] == "1"
    dlg.close()


# ----------------------------------------------------------------------
# Code-review regression tests (2026-08-14)
# ----------------------------------------------------------------------

class _FakePreviewDialog(QObject):
    """Fake non-modal preview dialog: close() emits finished like the
    real dialogs, so the main window's finished handlers run."""

    finished = Signal(int)

    def __init__(self):
        super().__init__()
        self.closed = False

    def close(self):
        self.closed = True
        self.finished.emit(0)

    def deleteLater(self):
        pass


def test_open_file_closes_live_preview_dialogs(window, monkeypatch):
    """dropEvent / recent-files can load a file while a preview dialog
    is open — the dialogs hold stale snapshots and must close first."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10])
    monkeypatch.setattr(fi.FileIO, "read", classmethod(lambda cls, p: atoms))
    fake_surface = _FakePreviewDialog()
    fake_generate = _FakePreviewDialog()
    window._surface_dialog = fake_surface
    window._generate_dialog = fake_generate
    fake_surface.finished.connect(
        lambda _r: window._on_surface_dialog_finished(fake_surface))
    fake_generate.finished.connect(
        lambda _r: window._on_generate_dialog_finished(fake_generate))

    window._open_file("fake.xyz")

    assert fake_surface.closed and fake_generate.closed
    assert window._surface_dialog is None
    assert window._generate_dialog is None
    assert window._structure.n_atoms == 2


def test_selection_ignored_during_non_frame_preview(window):
    """The viewport shows a TEMPORARY structure during a slab/NEB
    preview — frame indices must never become model indices."""
    window._structure.load_atoms(Atoms("H2O", positions=np.eye(3) * 1.5))
    window._surface_dialog = _FakePreviewDialog()
    window._on_atoms_selected([0, 1], "replace")
    assert window._structure.selected_indices == set()


def test_pause_resume_paired_between_preview_dialogs(window):
    """Editing stays paused until the LAST preview dialog closes."""
    window._set_preview_editing_enabled(False)
    assert not window.act_select_all.isEnabled()

    window._surface_dialog = _FakePreviewDialog()
    window._generate_dialog = _FakePreviewDialog()
    window._on_generate_dialog_finished(window._generate_dialog)
    assert not window.act_select_all.isEnabled()  # surface still open

    window._on_surface_dialog_finished(window._surface_dialog)
    assert window.act_select_all.isEnabled()


def test_structure_loaded_clears_frame_edit_state(window):
    window._frame_edit = {"panel": object(), "index": 2}
    window._frame_selection = {1}
    window._previewing_neb = True
    window._on_structure_loaded()
    assert window._frame_edit is None
    assert window._frame_selection == set()
    assert window._previewing_neb is False


def test_model_selection_change_ignored_during_frame_edit(window, monkeypatch):
    highlights = []
    monkeypatch.setattr(
        window._viewport, "set_highlight",
        lambda s: highlights.append(s))
    window._frame_edit = {"panel": object(), "index": 1}
    window._on_selection_changed()
    assert highlights == []  # the frame highlight stays untouched


# ----------------------------------------------------------------------
# Tools → Re-box Slab (2026-08-15)
# ----------------------------------------------------------------------

class _FakeReBoxDialog:
    """Stands in for ReBoxDialog in main-window flow tests."""

    Accepted = 1

    def __init__(self, model, parent=None):
        self.model = model

    def exec(self):
        from vaspen.core.surface import rebox_slab

        self.result_atoms = rebox_slab(self.model.atoms, 15.0)
        return self.Accepted


def test_rebox_applies_and_carries_flags(window, monkeypatch):
    """Re-box applies in one undo step, resets the filepath, and carries
    the frozen flags / magmoms 1:1 (same atoms, same order)."""
    import numpy as np

    # the handler imports the dialog lazily from vaspen.ui.rebox_dialog
    monkeypatch.setattr("vaspen.ui.rebox_dialog.ReBoxDialog", _FakeReBoxDialog)
    model = window._structure
    # build a slab-like model with a frozen atom and a magnetic moment
    atoms = Atoms("Cu4", cell=[2.5527, 2.5527, 21.0], pbc=True,
                  positions=[[0, 0, 5.0], [1.276, 2.21, 7.084],
                             [0, 0, 9.168], [1.276, 2.21, 11.252]])
    model.load_atoms(atoms, "slab.vasp")
    model.set_fixed(1, np.array([True, True, True]))
    model.set_magmom(2, 1.5)
    flags_before = model.fixed_flags.copy()
    magmoms_before = model.magmoms.copy()

    window._on_rebox()

    assert model.n_atoms == 4
    assert model.filepath is None  # derived structure → Save As
    assert model.can_undo
    assert np.allclose(model.fixed_flags, flags_before)
    assert np.allclose(model.magmoms, magmoms_before, equal_nan=True)
    cell = np.asarray(model.cell)
    z = model.positions[:, 2]
    assert (z.max() - z.min()) + 15.0 == pytest.approx(cell[2, 2], abs=1e-6)


def test_rebox_guards(window, monkeypatch):
    """Empty / non-periodic structures are rejected with a message."""
    import numpy as np

    from PySide6.QtWidgets import QMessageBox

    calls = []
    monkeypatch.setattr(
        "vaspen.ui.main_window.QMessageBox.information",
        staticmethod(lambda *a, **k: calls.append(a)))
    monkeypatch.setattr("vaspen.ui.rebox_dialog.ReBoxDialog", _FakeReBoxDialog)

    window._structure.load_atoms(Atoms("H2O", positions=np.eye(3) * 1.5))
    window._on_rebox()  # non-periodic → message, no apply
    assert calls and window._structure.n_atoms == 3


# ----------------------------------------------------------------------
# Code review 2026-08-15 — preview-pause completeness + click guards
# ----------------------------------------------------------------------

class _RaiseTrackingSurfaceDialog(_FakeSurfaceDialog):
    """The standard fake dialog + raise_/activateWindow tracking."""

    created = 0
    raised = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).created += 1

    def raise_(self):
        type(self).raised += 1

    def activateWindow(self):
        pass

    def close(self):
        self.finished.emit(0)

    def deleteLater(self):
        pass


def test_surface_double_open_raises_existing(window, monkeypatch, si_bulk):
    """A second Calculate → Cleave Surface while the dialog is open
    raises/activates the existing one instead of creating another."""
    window._structure.load_atoms(si_bulk)
    monkeypatch.setattr("vaspen.ui.surface_dialog.SurfaceDialog",
                        _RaiseTrackingSurfaceDialog)
    _RaiseTrackingSurfaceDialog.created = 0
    _RaiseTrackingSurfaceDialog.raised = 0
    window._on_surface()
    assert window._surface_dialog is not None
    window._on_surface()
    assert _RaiseTrackingSurfaceDialog.created == 1
    assert _RaiseTrackingSurfaceDialog.raised == 1
    window._surface_dialog.finished.emit(0)  # cleanup
    assert window._surface_dialog is None


def test_cleave_apply_clears_flags_and_magmoms(window, si_bulk):
    """Cleaving builds a NEW atom set — frozen flags / magmoms are
    cleared (settled §7.8); re-box carries them, cleave does not."""
    from vaspen.core.structure import StructureModel

    model = window._structure
    model.load_atoms(si_bulk, "bulk.vasp")
    model.set_fixed(0, np.array([True, True, True]))
    model.set_magmom(1, 2.0)

    dlg = _FakeSurfaceDialog(model, None)
    dlg.result_structure = StructureModel(si_bulk.copy())
    window._apply_surface_result(dlg)

    assert not model.any_fixed
    assert np.isnan(model.magmoms).all()
    assert model.filepath is None  # never silently overwrite the bulk


def test_preview_pauses_auto_bonds_and_bond_order(window):
    """Auto Detect Bonds + Bond Order mutate the model and rebind the
    viewport off the preview — they must be paused with the other
    editing entry points (code review 2026-08-15)."""
    window._set_preview_editing_enabled(False)
    assert not window.act_auto_bonds.isEnabled()
    assert all(not a.isEnabled()
               for a in window._bond_order_actions.values())
    window._set_preview_editing_enabled(True)
    assert window.act_auto_bonds.isEnabled()


def test_bond_click_ignored_during_preview(window):
    """Preview bond indices are temporary — a bond click during a slab/
    NEB preview must not select a model bond (code review 2026-08-15)."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]],
                  cell=[10, 10, 10])
    window._structure.load_atoms(atoms)
    window._surface_dialog = _FakePreviewDialog()
    window._on_bond_clicked(0)
    assert window._structure.selected_bonds == set()


def test_background_click_ignored_during_preview(window):
    """Clicking empty space during a preview clears nothing on the model."""
    atoms = Atoms("H2O", positions=np.eye(3) * 1.5)
    window._structure.load_atoms(atoms)
    window._structure.set_selection([0, 1])
    window._surface_dialog = _FakePreviewDialog()
    window._on_background_clicked()
    assert window._structure.selected_indices == {0, 1}


class _FakeSymmetryDialog:
    """Stands in for SymmetryDialog in main-window flow tests."""

    Accepted = 1

    def __init__(self, model, parent=None):
        self.model = model

    def exec(self):
        self.result_atoms = self.model.atoms.copy()
        return self.Accepted


def test_symmetry_resets_filepath(window, monkeypatch, si_bulk):
    """Symmetrizing derives a new cell — the filepath is reset so
    Ctrl+S prompts Save As instead of silently overwriting the source
    (same policy as cleave/supercell/rebox; code review 2026-08-15)."""
    monkeypatch.setattr("vaspen.ui.symmetry_dialog.SymmetryDialog",
                        _FakeSymmetryDialog)
    window._structure.load_atoms(si_bulk, "bulk.vasp")
    window._on_symmetry()
    assert window._structure.filepath is None
