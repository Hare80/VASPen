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
from vaspen.ui.supercell_dialog import SupercellDialog
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

    def set_structure(self, atoms, reset_view=False, bonds=None):
        self.rendered.append((atoms, reset_view, bonds))

    def set_highlight(self, indices):
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
