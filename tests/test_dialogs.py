"""Phase 3 dialog tests: lattice preview, supercell, transform, symmetry.

Offscreen; the LatticeDialog gets a fake viewport so no GL is touched.
"""

import numpy as np
import pytest
from ase import Atoms

from vaspen.core.render_settings import (
    BACKGROUND_DARK,
    BACKGROUND_LIGHT,
    GRADIENT_DARK_BOTTOM,
    GRADIENT_DARK_TOP,
    GRADIENT_LIGHT_BOTTOM,
    GRADIENT_LIGHT_TOP,
)
from vaspen.core.structure import StructureModel
from vaspen.ui.display_options_dialog import DisplayOptionsDialog
from vaspen.ui.lattice_dialog import LatticeDialog
from vaspen.ui.settings_dialog import SettingsDialog
from vaspen.ui.supercell_dialog import SupercellDialog
from vaspen.ui.surface_dialog import SurfaceDialog
from vaspen.ui.symmetry_dialog import SymmetryDialog
from vaspen.ui.transform_dialog import TransformDialog
from vaspen.utils.config import AppConfig


class _FakeViewport:
    """Records set_structure/set_render_settings calls for dialog previews."""

    def __init__(self) -> None:
        self.rendered: list = []  # (atoms, reset_view, bonds)
        from vaspen.core.render_settings import RenderSettings

        self._rs = RenderSettings.default()
        self.render_settings_calls: list = []

    def set_structure(self, atoms, reset_view=False, bonds=None, fixed=None):
        self.rendered.append((atoms, reset_view, bonds))

    def set_highlight(self, indices):
        pass

    def set_bond_highlight(self, indices):
        pass

    def set_render_settings(self, rs):
        from vaspen.core.render_settings import RenderSettings

        self._rs = RenderSettings.from_dict(rs.to_dict())
        self.render_settings_calls.append(self._rs)

    def render_settings(self):
        return self._rs

    def structure_symbols(self) -> list[str]:
        return ["Fe", "O"]


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


# ----------------------------------------------------------------------
# Display Options dialog
# ----------------------------------------------------------------------


def test_display_options_live_preview(qtbot):
    view = _FakeViewport()
    dlg = DisplayOptionsDialog(view)
    qtbot.addWidget(dlg)
    # initial control population fires several applies; snapshot the count
    n = len(view.render_settings_calls)
    assert n >= 1

    dlg._amb_slider.setValue(80)  # 0.80
    assert len(view.render_settings_calls) == n + 1
    assert view.render_settings_calls[-1].ambient == pytest.approx(0.80)

    dlg._headlight_check.setChecked(True)
    assert view.render_settings_calls[-1].headlight is True
    dlg._az_slider.setValue(60)
    assert view.render_settings_calls[-1].light_azimuth == 60


def test_display_options_reject_restores(qtbot):
    view = _FakeViewport()
    dlg = DisplayOptionsDialog(view)
    qtbot.addWidget(dlg)
    dlg._gamma_slider.setValue(round(3.0 / 0.05))
    dlg.reject()

    restored = view.render_settings_calls[-1]
    assert restored.gamma == pytest.approx(2.2)  # original value back


def test_display_options_accept_persists(qtbot):
    view = _FakeViewport()
    dlg = DisplayOptionsDialog(view)
    qtbot.addWidget(dlg)
    dlg._bg_light_radio.setChecked(True)
    dlg._gamma_slider.setValue(round(1.8 / 0.05))
    dlg._on_accept()

    cfg = AppConfig()
    assert cfg.render_settings.background_color == BACKGROUND_LIGHT
    assert cfg.render_settings.gamma == pytest.approx(1.8)


def test_display_options_bg_preset_swaps_gradient(qtbot):
    """Switching light → dark → light swaps the untouched gradient
    presets along with the background."""
    view = _FakeViewport()
    dlg = DisplayOptionsDialog(view)
    qtbot.addWidget(dlg)
    # default is LIGHT now; go dark first
    dlg._bg_dark_radio.setChecked(True)
    rs = view.render_settings_calls[-1]
    assert rs.background_color == BACKGROUND_DARK
    assert rs.gradient_top == GRADIENT_DARK_TOP

    # and back to light
    dlg._bg_light_radio.setChecked(True)
    rs = view.render_settings_calls[-1]
    assert rs.background_color == BACKGROUND_LIGHT
    assert rs.gradient_top == GRADIENT_LIGHT_TOP
    assert rs.gradient_bottom == GRADIENT_LIGHT_BOTTOM


def test_display_options_element_override(qtbot):
    view = _FakeViewport()
    dlg = DisplayOptionsDialog(view)
    qtbot.addWidget(dlg)
    assert dlg._row_swatch_for("Fe") is not None  # structure element rows

    dlg._on_element_opacity("Fe", 40)
    rs = view.render_settings_calls[-1]
    assert rs.atom_colors["Fe"][3] == pytest.approx(0.4)

    dlg._on_element_reset("Fe")
    assert "Fe" not in view.render_settings_calls[-1].atom_colors


def test_display_options_scheme_changes_effective_color(qtbot):
    view = _FakeViewport()
    dlg = DisplayOptionsDialog(view)
    qtbot.addWidget(dlg)
    dlg._scheme_btn._actions["metal_nonmetal"].trigger()

    rs = view.render_settings_calls[-1]
    assert rs.color_scheme == "metal_nonmetal"


def test_display_options_reset_defaults(qtbot):
    view = _FakeViewport()
    dlg = DisplayOptionsDialog(view)
    qtbot.addWidget(dlg)
    dlg._amb_slider.setValue(90)
    dlg._style_btn._actions["wireframe"].trigger()
    dlg._on_reset_defaults()

    rs = view.render_settings_calls[-1]
    assert rs.ambient == pytest.approx(0.75)
    assert rs.style == "ball_stick"
    assert rs.gamma == pytest.approx(2.2)


def test_display_options_specular_and_cell_controls(qtbot):
    view = _FakeViewport()
    dlg = DisplayOptionsDialog(view)
    qtbot.addWidget(dlg)

    dlg._spec_check.setChecked(False)
    assert view.render_settings_calls[-1].specular_enabled is False

    dlg._cell_width_slider.setValue(3)
    assert view.render_settings_calls[-1].cell_line_width == pytest.approx(3.0)

    dlg._cell_color = (1.0, 0.0, 0.0)
    dlg._apply()
    assert view.render_settings_calls[-1].cell_color == (1.0, 0.0, 0.0)


def test_display_options_style_button_is_menu_based(qtbot):
    """The style selector is a QToolButton+QMenu, not a QComboBox
    (the combo popup reopened repeatedly inside the scroll area)."""
    from PySide6.QtWidgets import QComboBox, QToolButton

    view = _FakeViewport()
    dlg = DisplayOptionsDialog(view)
    qtbot.addWidget(dlg)
    assert isinstance(dlg._style_btn, QToolButton)
    assert not isinstance(dlg._style_btn, QComboBox)
    assert isinstance(dlg._scheme_btn, QToolButton)
    assert set(dlg._style_btn._actions) == {"ball_stick", "cpk", "wireframe"}


def test_display_options_atom_opacity_slider(qtbot):
    view = _FakeViewport()
    dlg = DisplayOptionsDialog(view)
    qtbot.addWidget(dlg)
    n = len(view.render_settings_calls)

    dlg._opacity_slider.setValue(0)  # fully transparent
    assert len(view.render_settings_calls) == n + 1
    assert view.render_settings_calls[-1].atom_opacity == pytest.approx(0.0)

    dlg._opacity_slider.setValue(65)
    assert view.render_settings_calls[-1].atom_opacity == pytest.approx(0.65)


def test_display_options_corner_labels_checkbox(qtbot):
    view = _FakeViewport()
    dlg = DisplayOptionsDialog(view)
    qtbot.addWidget(dlg)
    dlg._cell_corners_check.setChecked(True)
    assert view.render_settings_calls[-1].show_cell_corners is True
    # the feature was renamed from "Cell axis labels (a/b/c)"
    assert "O/A/B/C" in dlg._cell_corners_check.text()


def test_display_options_corner_label_size_independent(qtbot):
    """Corner labels have their own size — decoupled from element labels."""
    view = _FakeViewport()
    dlg = DisplayOptionsDialog(view)
    qtbot.addWidget(dlg)
    dlg._corner_size_spin.setValue(20)
    rs = view.render_settings_calls[-1]
    assert rs.corner_label_size == 20
    assert rs.label_size == 12  # element labels unchanged


def test_display_options_ambient_slider_goes_to_2(qtbot):
    """Ambient/diffuse can be over-brightened (max 2.0) for white bg."""
    view = _FakeViewport()
    dlg = DisplayOptionsDialog(view)
    qtbot.addWidget(dlg)
    assert dlg._amb_slider.maximum() == 200
    dlg._amb_slider.setValue(150)
    assert view.render_settings_calls[-1].ambient == pytest.approx(1.50)


# ----------------------------------------------------------------------
# Periodic table dialog
# ----------------------------------------------------------------------

def test_periodic_table_dialog_layout_and_selection(qtbot):
    """118 element buttons in the standard periodic arrangement; clicking
    one accepts the dialog with the chosen symbol."""
    from ase.data import chemical_symbols

    from vaspen.ui.periodic_table_dialog import (
        PeriodicTableDialog, element_position,
    )

    dlg = PeriodicTableDialog()
    qtbot.addWidget(dlg)
    assert len(dlg._element_buttons) == 118
    # layout spot checks (standard periodic table positions)
    assert element_position("H") == (1, 1)
    assert element_position("He") == (1, 18)
    assert element_position("Fe") == (4, 8)
    assert element_position("La") == (8, 3)
    assert element_position("Og") == (7, 18)
    # every element is placed
    for z in range(1, 119):
        assert chemical_symbols[z] in dlg._element_buttons

    dlg._on_element("Fe")
    assert dlg.selected_symbol == "Fe"
    assert dlg.result() == 1  # Accepted


def test_display_options_add_element_uses_periodic_table(qtbot, monkeypatch):
    """The Add element… button opens the periodic table; clicking an
    element there adds its override row (replaces the old 118-entry
    QMenu)."""
    from vaspen.ui.periodic_table_dialog import PeriodicTableDialog

    view = _FakeViewport()
    dlg = DisplayOptionsDialog(view)
    qtbot.addWidget(dlg)
    assert dlg._row_swatch_for("Au") is None  # not in the fake structure

    class _FakeDlg:
        DialogCode = PeriodicTableDialog.DialogCode

        def __init__(self, parent=None):
            self.selected_symbol = "Au"

        def exec(self):
            return self.DialogCode.Accepted

    monkeypatch.setattr(
        "vaspen.ui.display_options_dialog.PeriodicTableDialog", _FakeDlg)
    dlg._add_element_btn.click()
    assert dlg._row_swatch_for("Au") is not None


def test_symmetry_dialog_disorder_warns(qtbot, monkeypatch, disordered_atoms):
    """Symmetrizing a disordered structure requires confirmation — the
    operation rebuilds atoms from scratch and drops the occupancy."""
    from ase import Atoms as _Atoms
    from PySide6.QtWidgets import QMessageBox

    m = StructureModel()
    m.load_atoms(disordered_atoms)
    dlg = SymmetryDialog(m)
    qtbot.addWidget(dlg)

    monkeypatch.setattr(
        "vaspen.ui.symmetry_dialog.QMessageBox.warning",
        staticmethod(lambda *a, **k: QMessageBox.Cancel))
    dlg._on_symmetrize()
    assert dlg.result_atoms is None  # cancelled — nothing applied

    monkeypatch.setattr(
        "vaspen.ui.symmetry_dialog.QMessageBox.warning",
        staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(
        "vaspen.core.symmetry.symmetrize", lambda atoms: _Atoms("Fe"))
    dlg._on_symmetrize()
    assert dlg.result_atoms is not None


def test_symmetry_dialog_clean_no_warning(qtbot, monkeypatch, si_bulk):
    """A structure without disorder symmetrizes without any warning."""
    def _boom(*a, **k):
        raise AssertionError("warning shown for a clean structure")
    monkeypatch.setattr(
        "vaspen.ui.symmetry_dialog.QMessageBox.warning",
        staticmethod(_boom))

    m = StructureModel()
    m.load_atoms(si_bulk)
    dlg = SymmetryDialog(m)
    qtbot.addWidget(dlg)
    dlg._on_symmetrize()
    assert dlg.result_atoms is not None


def test_display_options_measurement_controls(qtbot):
    view = _FakeViewport()
    dlg = DisplayOptionsDialog(view)
    qtbot.addWidget(dlg)
    assert dlg._measurement_size_spin.minimum() == 8
    assert dlg._measurement_size_spin.maximum() == 28

    dlg._measurement_color = (0.1, 0.2, 0.3)
    dlg._apply()
    rs = view.render_settings_calls[-1]
    assert rs.measurement_color == pytest.approx((0.1, 0.2, 0.3))

    dlg._measurement_size_spin.setValue(20)
    rs = view.render_settings_calls[-1]
    assert rs.measurement_label_size == 20
    assert rs.label_size == 12  # element labels untouched


def test_display_options_measurement_accept_persists(qtbot):
    view = _FakeViewport()
    dlg = DisplayOptionsDialog(view)
    qtbot.addWidget(dlg)
    dlg._measurement_color = (0.0, 0.5, 1.0)
    dlg._measurement_size_spin.setValue(18)
    dlg._on_accept()

    cfg = AppConfig()
    assert cfg.render_settings.measurement_color == pytest.approx(
        (0.0, 0.5, 1.0))
    assert cfg.render_settings.measurement_label_size == 18


def test_display_options_measurement_reset_defaults(qtbot):
    view = _FakeViewport()
    dlg = DisplayOptionsDialog(view)
    qtbot.addWidget(dlg)
    dlg._measurement_color = (1.0, 0.0, 0.0)
    dlg._measurement_size_spin.setValue(24)
    dlg._on_reset_defaults()

    rs = view.render_settings_calls[-1]
    assert rs.measurement_color == (0.0, 0.0, 0.0)
    assert rs.measurement_label_size == 12


# ----------------------------------------------------------------------
# Surface dialog (non-modal, live preview, termination selection)
# ----------------------------------------------------------------------


def test_surface_dialog_opens_with_preview(qtbot, srtio3):
    model = StructureModel(srtio3)
    view = _FakeViewport()
    dlg = SurfaceDialog(model, view)
    qtbot.addWidget(dlg)

    assert len(view.rendered) == 1  # preview pushed on open
    atoms, reset_view, bonds = view.rendered[-1]
    assert reset_view is False
    assert bonds is None  # auto-detected — model bond indices are invalid
    assert len(atoms) > 0
    assert tuple(atoms.get_pbc()) == (True, True, True)


def test_surface_dialog_termination_combo_and_preview_switch(qtbot, gaas):
    """GaAs (111) is polar: two unique terminations (Ga / As faces)."""
    model = StructureModel(gaas)
    view = _FakeViewport()
    dlg = SurfaceDialog(model, view)  # default Miller (1,1,1)
    qtbot.addWidget(dlg)

    combo = dlg._termination_combo
    assert combo.count() == 2
    items = [combo.itemText(i) for i in range(combo.count())]
    assert any("Ga" in t and "As" in t for t in items)
    assert any(t.startswith("1/2") for t in items)
    assert any(t.startswith("2/2") for t in items)

    rendered_before = len(view.rendered)
    combo.setCurrentIndex(1)
    assert len(view.rendered) == rendered_before + 1
    assert np.allclose(view.rendered[-1][0].get_positions(),
                       dlg._slab_infos[1].atoms.get_positions())


def test_surface_dialog_param_change_debounced(qtbot, srtio3):
    """Parameter spins debounce through a 150 ms timer before recompute."""
    model = StructureModel(srtio3)
    view = _FakeViewport()
    dlg = SurfaceDialog(model, view)
    qtbot.addWidget(dlg)

    n_before = len(view.rendered)
    dlg._layers_spin.setValue(5)
    dlg._vacuum_spin.setValue(20.0)
    assert len(view.rendered) == n_before  # still debounced
    qtbot.wait(250)
    assert len(view.rendered) == n_before + 1  # one recompute for both spins


def test_surface_dialog_termination_switch_reuses_cache(qtbot, gaas):
    model = StructureModel(gaas)
    view = _FakeViewport()
    dlg = SurfaceDialog(model, view)
    qtbot.addWidget(dlg)

    key = dlg._cache_key
    dlg._termination_combo.setCurrentIndex(1)
    dlg._termination_combo.setCurrentIndex(0)
    assert dlg._cache_key == key  # no recompute — cached SlabInfo list


def test_surface_dialog_accept_sets_result(qtbot, gaas):
    model = StructureModel(gaas)
    view = _FakeViewport()
    dlg = SurfaceDialog(model, view)
    qtbot.addWidget(dlg)

    dlg._termination_combo.setCurrentIndex(1)
    dlg._on_accept()
    assert dlg.result_structure is not None
    assert np.allclose(dlg.result_structure.atoms.get_positions(),
                       dlg._slab_infos[1].atoms.get_positions())
    # model untouched until MainWindow applies the result
    assert np.allclose(model.positions, gaas.get_positions())


def test_surface_dialog_reject_restores_viewport(qtbot, srtio3):
    model = StructureModel(srtio3)
    view = _FakeViewport()
    dlg = SurfaceDialog(model, view)
    qtbot.addWidget(dlg)

    dlg.reject()
    atoms, reset_view, bonds = view.rendered[-1]
    assert reset_view is False
    assert np.allclose(atoms.get_positions(), model.positions)
    assert bonds is not None  # the model's own bond list


def test_surface_dialog_zero_miller_warns(qtbot, srtio3, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "vaspen.ui.surface_dialog.QMessageBox.warning",
        staticmethod(lambda *a, **k: warnings.append(a)))

    model = StructureModel(srtio3)
    view = _FakeViewport()
    dlg = SurfaceDialog(model, view)
    qtbot.addWidget(dlg)

    dlg._h_spin.setValue(0)
    dlg._k_spin.setValue(0)
    dlg._l_spin.setValue(0)
    dlg._ensure_slabs()

    assert not dlg._ok_button.isEnabled()
    assert dlg._hint.text() != ""
    assert dlg._termination_combo.count() == 0

    dlg._on_accept()  # no slab to accept — warns and stays open
    assert dlg.result_structure is None
    assert len(warnings) == 1


def test_surface_dialog_supercell_expands_preview(qtbot, gaas):
    model = StructureModel(gaas)
    view = _FakeViewport()
    dlg = SurfaceDialog(model, view)
    qtbot.addWidget(dlg)

    base_n = dlg._slab_infos[0].n_atoms
    assert len(view.rendered[-1][0]) == base_n  # 1x1 default

    key = dlg._cache_key
    dlg._supercell_a_spin.setValue(2)
    assert dlg._cache_key == key  # no SlabGenerator recompute (repeat only)
    assert len(view.rendered[-1][0]) == 2 * base_n

    dlg._supercell_b_spin.setValue(2)
    assert len(view.rendered[-1][0]) == 4 * base_n
    assert str(4 * base_n) in dlg._status_label.text()

    dlg._on_accept()
    assert dlg.result_structure is not None
    assert dlg.result_structure.n_atoms == 4 * base_n
    assert model.n_atoms == len(gaas)  # model untouched until applied


# ----------------------------------------------------------------------
# Settings dialog — POSCAR coordinate format
# ----------------------------------------------------------------------

def test_settings_dialog_poscar_coords_default_direct(qtbot):
    """Unset config preselects fractional (Direct) coordinates."""
    dlg = SettingsDialog()
    qtbot.addWidget(dlg)
    assert dlg._poscar_coords_combo.currentIndex() == 0


def test_settings_dialog_poscar_coords_loads_config(qtbot):
    config = AppConfig()
    config.poscar_coords_direct = False
    dlg = SettingsDialog()
    qtbot.addWidget(dlg)
    assert dlg._poscar_coords_combo.currentIndex() == 1  # Cartesian


def test_settings_dialog_poscar_coords_accept_persists(qtbot):
    config = AppConfig()
    config.poscar_coords_direct = True
    dlg = SettingsDialog()
    qtbot.addWidget(dlg)
    dlg._poscar_coords_combo.setCurrentIndex(1)  # Cartesian
    dlg._on_accept()
    assert config.poscar_coords_direct is False


# ----------------------------------------------------------------------
# Every dialog dropdown is a MenuButton (QComboBox popups ghost here)
# ----------------------------------------------------------------------

def test_dialog_dropdowns_are_menu_buttons(qtbot, periodic_model, srtio3):
    from PySide6.QtWidgets import QComboBox

    from vaspen.ui.incar_editor import IncarEditorDialog
    from vaspen.ui.kpoints_editor import KpointsEditorDialog
    from vaspen.ui.menu_button import MenuButton
    from vaspen.ui.potcar_dialog import PotcarDialog
    from vaspen.ui.transform_dialog import TransformDialog

    checkables: list = []

    dlg = SettingsDialog()
    qtbot.addWidget(dlg)
    checkables += [dlg._language_combo, dlg._calc_type_combo,
                   dlg._poscar_coords_combo]
    dlg.close()

    dlg = IncarEditorDialog()
    qtbot.addWidget(dlg)
    checkables.append(dlg._preset_combo)
    dlg.close()

    dlg = KpointsEditorDialog(periodic_model)
    qtbot.addWidget(dlg)
    checkables += [dlg._mode_combo, dlg._gamma_auto, dlg._gamma_manual]
    dlg.close()

    model = StructureModel()
    model.load_atoms(srtio3)
    dlg = SurfaceDialog(model, _FakeViewport())
    qtbot.addWidget(dlg)
    checkables.append(dlg._termination_combo)
    dlg.close()

    dlg = TransformDialog(periodic_model)
    qtbot.addWidget(dlg)
    checkables.append(dlg._axis_combo)
    dlg.close()

    dlg = PotcarDialog()
    qtbot.addWidget(dlg)
    checkables.append(dlg._functional_combo)
    dlg.close()

    for widget in checkables:
        assert isinstance(widget, MenuButton)
        assert not isinstance(widget, QComboBox)


# ----------------------------------------------------------------------
# INCAR editor — preview format & duplicate-tag validation
# ----------------------------------------------------------------------

def test_incar_editor_preview_uses_aligned_format(qtbot):
    from vaspen.ui.incar_editor import IncarEditorDialog

    dlg = IncarEditorDialog()
    qtbot.addWidget(dlg)
    dlg._load_preset("neb")  # has LCLIMB=True → .TRUE. in preview

    preview = dlg._preview.toPlainText()
    assert ".TRUE." in preview
    assert "True" not in preview
    # comment alignment: active comment lines have '(' at column 26
    for line in preview.splitlines():
        if "(" in line and not line.startswith("  #"):
            assert line[25] == "(", repr(line)
    # suggestion block is commented out
    assert any(l.startswith("  # MAGMOM") for l in preview.splitlines())
    dlg.close()


def test_incar_editor_duplicate_tags_detected(qtbot):
    from vaspen.ui.incar_editor import IncarEditorDialog

    dlg = IncarEditorDialog()
    qtbot.addWidget(dlg)
    dlg._load_preset("scf")

    dlg._add_custom_tag()
    row = dlg._table.rowCount() - 1
    dlg._table.item(row, 0).setText("ENCUT")  # collides with preset ENCUT
    assert "ENCUT" in dlg._validate_tags()
    dlg.close()


def test_incar_editor_add_tag_focuses_pending_row(qtbot):
    from vaspen.ui.incar_editor import IncarEditorDialog

    dlg = IncarEditorDialog()
    qtbot.addWidget(dlg)
    dlg._load_preset("scf")

    dlg._add_custom_tag()
    rows_before = dlg._table.rowCount()
    dlg._add_custom_tag()  # NEW_TAG row already pending → no new row
    assert dlg._table.rowCount() == rows_before
    dlg.close()


# ----------------------------------------------------------------------
# INCAR editor — editable preview, sync, save-with-hand-edits
# ----------------------------------------------------------------------

def test_incar_editor_preview_is_editable(qtbot):
    from vaspen.ui.incar_editor import IncarEditorDialog

    dlg = IncarEditorDialog()
    qtbot.addWidget(dlg)
    assert not dlg._preview.isReadOnly()
    dlg.close()


def test_incar_editor_sync_from_preview(qtbot):
    from vaspen.ui.incar_editor import IncarEditorDialog

    dlg = IncarEditorDialog()
    qtbot.addWidget(dlg)
    dlg._load_preset("scf")

    # hand-edit: change ENCUT, add a new tag, keep a commented suggestion
    dlg._preview.setPlainText(
        "SYSTEM = edited\n"
        "  ENCUT  =  500          (my cutoff)\n"
        "  IVDW   =  11\n"
        "  # MAGMOM =               (initial magnetic moments per atom)\n"
    )
    dlg._on_sync_from_preview()

    # table now holds the parsed tags (ENCUT updated, IVDW added)
    table_tags = {
        dlg._table.item(r, 0).text(): dlg._table.item(r, 1).text()
        for r in range(dlg._table.rowCount())
    }
    assert table_tags["ENCUT"] == "500"
    assert "IVDW" in table_tags and table_tags["IVDW"] == "11"
    assert "MAGMOM" not in table_tags  # commented line stays a suggestion
    # preview was re-aligned/normalized by the sync (canonical comment
    # table replaces the hand-written one)
    preview = dlg._preview.toPlainText()
    assert "  ENCUT  =  500          (plane-wave cutoff in eV; set to 1.3 x ENMAX of POTCAR)" in preview
    assert not dlg._preview_dirty
    dlg.close()


def test_incar_editor_sync_rejects_bad_lines(qtbot, monkeypatch):
    from vaspen.ui.incar_editor import IncarEditorDialog

    dlg = IncarEditorDialog()
    qtbot.addWidget(dlg)
    dlg._load_preset("scf")
    warnings = []
    monkeypatch.setattr(
        "vaspen.ui.incar_editor.QMessageBox.warning",
        lambda *a, **k: warnings.append(a),
    )

    dlg._preview.setPlainText("ENCUT = 500\nnot a tag line\n")
    dlg._on_sync_from_preview()

    assert len(warnings) == 1
    # table untouched by the failed sync
    table_tags = {
        dlg._table.item(r, 0).text() for r in range(dlg._table.rowCount())
    }
    assert "ENCUT" in table_tags  # still preset value, not synced
    dlg.close()


def test_incar_editor_accept_saves_hand_edited_preview(qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog, QFileDialog

    from vaspen.ui.incar_editor import IncarEditorDialog

    dlg = IncarEditorDialog()
    qtbot.addWidget(dlg)
    dlg._load_preset("scf")

    out = tmp_path / "INCAR"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        lambda *a, **k: (str(out), ""),
    )

    # user hand-edits the preview, then saves without touching the table
    dlg._preview.setPlainText("SYSTEM = hand edited\n  ENCUT = 520\n")
    dlg._on_accept()

    assert out.read_text(encoding="utf-8") == "SYSTEM = hand edited\n  ENCUT = 520\n"
    assert dlg.result() == QDialog.Accepted  # noqa: F821
    dlg.close()


def test_incar_editor_accept_blocks_duplicate_in_preview(qtbot, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from vaspen.ui.incar_editor import IncarEditorDialog

    dlg = IncarEditorDialog()
    qtbot.addWidget(dlg)
    dlg._load_preset("scf")

    warnings = []
    calls = []
    monkeypatch.setattr(
        "vaspen.ui.incar_editor.QMessageBox.warning",
        lambda *a, **k: warnings.append(a),
    )
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        lambda *a, **k: calls.append(a) or ("", ""),
    )

    dlg._preview.setPlainText("ENCUT = 400\nencut = 500\n")
    dlg._on_accept()

    assert len(warnings) == 1
    assert calls == []  # save dialog never opened
    dlg.close()
