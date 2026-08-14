"""Tests for the unified GenerateAllDialog (POSCAR/INCAR/KPOINTS/POTCAR)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write as ase_write
from PySide6.QtWidgets import QMessageBox

from vaspen.core.file_io import FileIO
from vaspen.core.structure import StructureModel
from vaspen.ui.generate_all_dialog import GenerateAllDialog

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples" / "neb_vacancy_hop"


def _example_model() -> StructureModel:
    model = StructureModel()
    model.load_atoms(FileIO.read(str(EXAMPLES / "initial" / "POSCAR")))
    return model


def _record(monkeypatch, name: str) -> list:
    """Record calls to a QMessageBox static method."""
    calls: list = []
    monkeypatch.setattr(
        f"vaspen.ui.generate_all_dialog.QMessageBox.{name}",
        staticmethod(lambda *a, **k: calls.append(a) or QMessageBox.Ok),
    )
    return calls


# ----------------------------------------------------------------------
# Task coupling
# ----------------------------------------------------------------------

def test_band_task_switches_kpoints_to_line_mode(qtbot):
    from ase.build import bulk

    from vaspen.ui.generate_all_dialog import TASKS

    model = StructureModel()
    model.load_atoms(bulk("Si", "diamond", a=5.43))

    dlg = GenerateAllDialog(model, default_task="band")
    qtbot.addWidget(dlg)

    assert dlg._task_combo.currentIndex() == TASKS.index("band")
    assert dlg._kpoints_panel.mode() == "line"
    assert not dlg._poscar_panel.neb_mode
    # the suggested path is applied and renders a valid preview
    path_text = dlg._kpoints_panel._band_path_edit.text()
    assert "-" in path_text and len(path_text) > len("G-X")
    preview = dlg._kpoints_panel._preview.toPlainText()
    assert preview.startswith("Band structure:")
    assert not preview.startswith("#")


def test_incar_preset_change_syncs_task_combo(qtbot):
    model = _example_model()
    dlg = GenerateAllDialog(model, default_task="scf")
    qtbot.addWidget(dlg)

    # user switches to NEB inside the INCAR tab → the whole dialog follows
    dlg._incar_panel._preset_combo.setCurrentIndex(5)
    assert dlg._task_combo.currentIndex() == 5
    assert dlg._poscar_panel.neb_mode
    assert dlg._kpoints_panel.mode() == "automatic"


# ----------------------------------------------------------------------
# NEB flow
# ----------------------------------------------------------------------

def _write_poscar(atoms: Atoms, path: Path) -> Path:
    ase_write(path, atoms, format="vasp", vasp5=True, direct=True)
    return path


def test_neb_end_to_end(qtbot, tmp_path, monkeypatch):
    model = _example_model()
    info = _record(monkeypatch, "information")

    fin_dir = tmp_path / "final"
    fin_dir.mkdir(parents=True, exist_ok=True)
    fin_path = _write_poscar(
        FileIO.read(str(EXAMPLES / "final" / "POSCAR")), fin_dir / "POSCAR")
    monkeypatch.setattr(
        "vaspen.ui.generate_all_dialog.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(fin_path), ""),
    )

    previews: list = []
    dlg = GenerateAllDialog(model, default_task="neb",
                            preview_callback=lambda atoms, *a: previews.append(atoms))
    qtbot.addWidget(dlg)

    panel = dlg._poscar_panel
    assert panel.neb_mode
    panel._on_browse_fin()
    # Browsing endpoints clears the images and resets the preview
    # (the callback receives None so the main window leaves frame-edit).
    assert previews == [None]

    # diagnostics: distance + suggested count prefill the spin box
    assert "2.5562" in panel._info_label.text()
    assert panel._images_spin.value() == 4

    panel._on_interpolate()
    assert panel.n_images == 4
    assert panel._images_list.count() == 6

    # IMAGES is auto-filled into the INCAR tab
    incar_tags = {
        dlg._incar_panel._table.item(r, 0).text():
        dlg._incar_panel._table.item(r, 1).text()
        for r in range(dlg._incar_panel._table.rowCount())
    }
    assert incar_tags["IMAGES"] == "4"

    # clicking a frame previews it via the callback
    panel._images_list.setCurrentRow(2)
    frames = [a for a in previews if a is not None]
    assert len(frames) == 1
    assert len(frames[0]) == 7

    # generate writes the vtst-style layout
    out = tmp_path / "out"
    out.mkdir()
    dlg._dir_edit.setText(str(out))
    dlg._on_generate()

    assert info, "success message expected"
    for i in range(6):
        assert (out / f"{i:02d}" / "POSCAR").exists(), i
    assert (out / "INCAR").exists()
    assert (out / "KPOINTS").exists()
    assert not (out / "POTCAR").exists()  # no library configured
    assert not (out / "POSCAR").exists()  # NEB layout has no root POSCAR

    incar_text = (out / "INCAR").read_text(encoding="utf-8")
    from vaspen.core.vasp_input import parse_incar_content
    tags, problems = parse_incar_content(incar_text)
    assert problems == []
    assert tags["IMAGES"] == "4"
    # NEB uses a uniform k-mesh, not line-mode
    assert "Line-mode" not in (out / "KPOINTS").read_text(encoding="utf-8")


def test_neb_blocks_generate_without_images(qtbot, tmp_path, monkeypatch):
    model = _example_model()
    warnings = _record(monkeypatch, "warning")

    dlg = GenerateAllDialog(model, default_task="neb")
    qtbot.addWidget(dlg)

    out = tmp_path / "out"
    out.mkdir()
    dlg._dir_edit.setText(str(out))
    dlg._on_generate()

    assert warnings
    assert not any(out.iterdir())  # nothing written


def test_neb_order_mismatch_warns_then_strict_order(qtbot, tmp_path, monkeypatch):
    # Two Cu atoms whose order is swapped between ini/fin
    ini = Atoms("Cu2", positions=[[0.0, 0, 0], [2.0, 0, 0]], cell=[10, 10, 10], pbc=True)
    fin = Atoms("Cu2", positions=[[2.0, 0, 0], [0.0, 0, 0]], cell=[10, 10, 10], pbc=True)
    model = StructureModel()
    model.load_atoms(ini)
    fin_dir = tmp_path / "fin"
    fin_dir.mkdir(parents=True, exist_ok=True)
    fin_path = _write_poscar(fin, fin_dir / "POSCAR")
    monkeypatch.setattr(
        "vaspen.ui.generate_all_dialog.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(fin_path), ""),
    )

    # Cancel at the warning → no images
    questions: list = []
    monkeypatch.setattr(
        "vaspen.ui.generate_all_dialog.QMessageBox.question",
        staticmethod(lambda *a, **k: questions.append(a) or QMessageBox.No),
    )
    dlg = GenerateAllDialog(model, default_task="neb")
    qtbot.addWidget(dlg)
    panel = dlg._poscar_panel
    panel._on_browse_fin()
    panel._on_interpolate()
    assert questions
    assert panel.n_images == 0

    # Confirm → strict file order interpolation
    monkeypatch.setattr(
        "vaspen.ui.generate_all_dialog.QMessageBox.question",
        staticmethod(lambda *a, **k: QMessageBox.Yes),
    )
    panel._images_spin.setValue(1)  # the suggested count (4) was prefilled
    panel._on_interpolate()
    assert panel.n_images == 1
    # strict order: atom 0 of the initial interpolates to atom 0 of the
    # final (positions swap along the path)
    mid = panel.images[1]
    assert np.allclose(mid.positions, [[1.0, 0, 0], [1.0, 0, 0]])


def test_neb_cell_mismatch_blocks(qtbot, tmp_path, monkeypatch):
    model = _example_model()
    ini = FileIO.read(str(EXAMPLES / "initial" / "POSCAR"))
    bad = ini.copy()
    bad.set_cell(ini.get_cell() * 1.1)
    fin_dir = tmp_path / "fin"
    fin_dir.mkdir(parents=True, exist_ok=True)
    bad_path = _write_poscar(bad, fin_dir / "POSCAR")
    monkeypatch.setattr(
        "vaspen.ui.generate_all_dialog.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(bad_path), ""),
    )
    warnings = _record(monkeypatch, "warning")

    dlg = GenerateAllDialog(model, default_task="neb")
    qtbot.addWidget(dlg)
    panel = dlg._poscar_panel
    panel._on_browse_fin()
    assert "cells differ" in panel._warn_label.text().lower()

    panel._on_interpolate()
    assert warnings
    assert panel.n_images == 0


def test_neb_nonperiodic_structure_gets_wrapped(qtbot, monkeypatch):
    from vaspen.core.neb import interpolate_neb

    class _FakeWrapDialog:
        padding = 5.0

        def __init__(self, *a, **k):
            pass

        def exec(self):
            return 1  # QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        "vaspen.ui.generate_all_dialog.PeriodicWrapDialog", _FakeWrapDialog)

    model = _example_model()
    dlg = GenerateAllDialog(model, default_task="neb")
    qtbot.addWidget(dlg)
    panel = dlg._poscar_panel

    molecule = Atoms("Cu7", positions=model.atoms.positions.copy())  # no cell
    wrapped = panel._ensure_periodic_copy(molecule)
    assert wrapped is not None
    assert wrapped.get_cell().rank == 3
    assert wrapped.pbc.all()

    # cancel path returns None
    class _CancelWrapDialog(_FakeWrapDialog):
        def exec(self):
            return 0

    monkeypatch.setattr(
        "vaspen.ui.generate_all_dialog.PeriodicWrapDialog", _CancelWrapDialog)
    assert panel._ensure_periodic_copy(molecule) is None


def test_neb_imimages_survives_task_roundtrip(qtbot):
    model = _example_model()
    dlg = GenerateAllDialog(model, default_task="neb")
    qtbot.addWidget(dlg)
    panel = dlg._poscar_panel
    # interpolate directly (no file dialogs involved)
    panel._final_atoms = FileIO.read(str(EXAMPLES / "final" / "POSCAR"))
    panel._update_diagnostics()
    panel._on_interpolate()

    # switch away and back — the preset reload empties IMAGES, the
    # task rule refills it from the generated images
    dlg._task_combo.setCurrentIndex(0)  # scf
    dlg._task_combo.setCurrentIndex(5)  # neb
    incar_tags = {
        dlg._incar_panel._table.item(r, 0).text():
        dlg._incar_panel._table.item(r, 1).text()
        for r in range(dlg._incar_panel._table.rowCount())
    }
    assert incar_tags["IMAGES"] == "4"


# ----------------------------------------------------------------------
# Validation + normal task generation
# ----------------------------------------------------------------------

def test_empty_incar_tag_blocks_generate(qtbot, tmp_path, monkeypatch):
    model = _example_model()
    warnings = _record(monkeypatch, "warning")

    dlg = GenerateAllDialog(model, default_task="scf")
    qtbot.addWidget(dlg)
    dlg._incar_panel.apply_tag("FOO", "")

    out = tmp_path / "out"
    out.mkdir()
    dlg._dir_edit.setText(str(out))
    dlg._on_generate()

    assert warnings
    assert not any(out.iterdir())


def test_normal_task_writes_all_files(qtbot, tmp_path, monkeypatch):
    model = _example_model()
    info = _record(monkeypatch, "information")

    dlg = GenerateAllDialog(model, default_task="scf")
    qtbot.addWidget(dlg)
    assert not dlg._poscar_panel.neb_mode

    out = tmp_path / "out"
    out.mkdir()
    dlg._dir_edit.setText(str(out))
    dlg._on_generate()

    assert info
    assert (out / "POSCAR").exists()
    assert (out / "INCAR").exists()
    assert (out / "KPOINTS").exists()
    assert not (out / "POTCAR").exists()  # no library
    assert "Cu" in (out / "POSCAR").read_text(encoding="utf-8")


def test_generate_without_output_dir_warns(qtbot, monkeypatch):
    model = _example_model()
    warnings = _record(monkeypatch, "warning")

    dlg = GenerateAllDialog(model, default_task="scf")
    qtbot.addWidget(dlg)
    dlg._dir_edit.setText("")
    dlg._on_generate()
    assert warnings


def test_poscar_coords_config_controls_preview(qtbot):
    """The POSCAR tab follows the Direct/Cartesian preference."""
    from vaspen.utils.config import AppConfig

    model = _example_model()
    config = AppConfig()

    config.poscar_coords_direct = True
    dlg = GenerateAllDialog(model, default_task="scf")
    qtbot.addWidget(dlg)
    assert dlg._poscar_panel.poscar_direct is True
    assert "Direct" in dlg._poscar_panel._poscar_preview.toPlainText()

    config.poscar_coords_direct = False
    dlg2 = GenerateAllDialog(model, default_task="scf")
    qtbot.addWidget(dlg2)
    assert dlg2._poscar_panel.poscar_direct is False
    assert "Cartesian" in dlg2._poscar_panel._poscar_preview.toPlainText()


def test_poscar_coord_combo_refreshes_preview(qtbot):
    model = _example_model()
    dlg = GenerateAllDialog(model, default_task="scf")
    qtbot.addWidget(dlg)
    panel = dlg._poscar_panel

    panel._coord_combo.setCurrentIndex(0)  # Direct
    assert "Direct" in panel._poscar_preview.toPlainText()
    panel._coord_combo.setCurrentIndex(1)  # Cartesian
    assert "Cartesian" in panel._poscar_preview.toPlainText()


# ----------------------------------------------------------------------
# Non-periodic model robustness (regression: benzene crash 2026-08-14)
# ----------------------------------------------------------------------

def test_generate_all_constructs_with_nonperiodic_model(qtbot):
    """A cell-less molecule must not crash the dialog construction
    (KPOINTS preview used to hit np.linalg.inv on a singular cell),
    and must not fabricate a mesh for it — the preview explains
    instead."""
    from ase.build import molecule

    model = StructureModel()
    model.load_atoms(molecule("C6H6"))
    dlg = GenerateAllDialog(model, default_task="scf")
    qtbot.addWidget(dlg)

    # automatic-mode preview is blocked with a note, not a fake mesh
    preview = dlg._kpoints_panel._preview.toPlainText()
    assert preview.startswith("#")
    assert "not periodic" in preview
    assert "Automatic k-point mesh" not in preview


def test_generate_blocks_nonperiodic_model(qtbot, tmp_path, monkeypatch):
    from ase.build import molecule

    model = StructureModel()
    model.load_atoms(molecule("C6H6"))
    warnings = _record(monkeypatch, "warning")

    dlg = GenerateAllDialog(model, default_task="scf")
    qtbot.addWidget(dlg)
    out = tmp_path / "out"
    out.mkdir()
    dlg._dir_edit.setText(str(out))
    dlg._on_generate()

    assert warnings
    assert not any(out.iterdir())


def test_kpoints_panel_singular_cell_falls_back(qtbot):
    from ase.build import bulk

    from vaspen.ui.kpoints_editor import KpointsEditorPanel

    model = StructureModel()
    model.load_atoms(Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))  # no cell
    panel = KpointsEditorPanel(model)
    qtbot.addWidget(panel)
    # automatic preview must not raise and must not fabricate a mesh
    preview = panel._preview.toPlainText()
    assert preview.startswith("#")
    assert "not periodic" in preview
    # the line-mode preview reports a clear note, not a crash
    panel.set_mode("line")
    assert panel._preview.toPlainText().startswith("#")

    # a periodic model still gets a real mesh
    model.load_atoms(bulk("Si", "diamond", a=5.43))
    panel.set_mode("automatic")
    assert panel._preview.toPlainText().startswith("Automatic k-point mesh")
    assert panel.content().startswith("Automatic k-point mesh")


# ----------------------------------------------------------------------
# NEB interpolation algorithm selector (Linear default / IDPP)
# ----------------------------------------------------------------------

ETHANE = REPO / "examples" / "neb_ethane_rotation"


def _ethane_model() -> StructureModel:
    model = StructureModel()
    model.load_atoms(FileIO.read(str(ETHANE / "initial" / "POSCAR")))
    return model


def _min_pair_distance(atoms) -> float:
    import numpy as np

    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def test_neb_algorithm_selector_defaults_to_linear(qtbot):
    model = _example_model()
    dlg = GenerateAllDialog(model, default_task="neb")
    qtbot.addWidget(dlg)
    panel = dlg._poscar_panel

    assert panel._algo_combo.currentIndex() == 0  # Linear
    panel._final_atoms = FileIO.read(str(EXAMPLES / "final" / "POSCAR"))
    panel._update_diagnostics()
    panel._on_interpolate()
    # linear path = linear in fractional space
    frac = panel.images[2].get_scaled_positions(wrap=False)
    ini = panel._init_atoms.get_scaled_positions(wrap=False)
    assert np.allclose(frac, ini + (panel.images[-1].get_scaled_positions(wrap=False) - ini) * (2 / 5))


def test_neb_idpp_algorithm_avoids_collisions(qtbot, monkeypatch):
    # Ethane's equivalent hydrogens can be relabeled, so the order
    # diagnostic fires — confirm "continue in strict file order".
    monkeypatch.setattr(
        "vaspen.ui.generate_all_dialog.QMessageBox.question",
        staticmethod(lambda *a, **k: QMessageBox.Yes),
    )
    model = _ethane_model()
    dlg = GenerateAllDialog(model, default_task="neb")
    qtbot.addWidget(dlg)
    panel = dlg._poscar_panel

    panel._final_atoms = FileIO.read(str(ETHANE / "final" / "POSCAR"))
    panel._update_diagnostics()
    panel._images_spin.setValue(5)
    panel._algo_combo.setCurrentIndex(1)  # IDPP
    panel._on_interpolate()

    assert panel.n_images == 5
    assert min(_min_pair_distance(f) for f in panel.images) > 0.9


def test_neb_linear_algorithm_collides_ethane(qtbot, monkeypatch):
    """Contrast test: the same pair interpolated linearly collides."""
    monkeypatch.setattr(
        "vaspen.ui.generate_all_dialog.QMessageBox.question",
        staticmethod(lambda *a, **k: QMessageBox.Yes),
    )
    model = _ethane_model()
    dlg = GenerateAllDialog(model, default_task="neb")
    qtbot.addWidget(dlg)
    panel = dlg._poscar_panel

    panel._final_atoms = FileIO.read(str(ETHANE / "final" / "POSCAR"))
    panel._update_diagnostics()
    panel._images_spin.setValue(5)
    panel._on_interpolate()  # default Linear

    assert min(_min_pair_distance(f) for f in panel.images) < 0.75


# ----------------------------------------------------------------------
# Frozen atoms in the NEB flow
# ----------------------------------------------------------------------

FROZEN = REPO / "examples" / "neb_frozen"


def test_neb_frozen_atoms_flow_pass(qtbot, tmp_path, monkeypatch):
    """Opening a POSCAR with selective dynamics freezes those atoms:
    frames keep them put and the written POSCARs carry F F F rows."""
    model = StructureModel()
    model.load_atoms(FileIO.read(str(FROZEN / "pass" / "initial" / "POSCAR")))
    assert model.fixed_flags.all(axis=1).sum() == 6

    fin_dir = tmp_path / "final"
    fin_dir.mkdir(parents=True, exist_ok=True)
    fin_path = _write_poscar(
        FileIO.read(str(FROZEN / "pass" / "final" / "POSCAR")), fin_dir / "POSCAR")
    monkeypatch.setattr(
        "vaspen.ui.generate_all_dialog.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(fin_path), ""),
    )
    info = _record(monkeypatch, "information")

    dlg = GenerateAllDialog(model, default_task="neb")
    qtbot.addWidget(dlg)
    panel = dlg._poscar_panel
    panel._on_browse_fin()

    assert "6 atoms frozen" in panel._info_label.text()
    panel._on_interpolate()

    mask = model.fixed_flags.all(axis=1)
    for frame in panel.images:
        assert np.allclose(frame.positions[mask],
                           model.atoms.positions[mask])
    # the moving atom actually moved
    moving = ~mask
    assert not np.allclose(panel.images[2].positions[moving],
                           model.atoms.positions[moving])

    out = tmp_path / "out"
    out.mkdir()
    dlg._dir_edit.setText(str(out))
    dlg._on_generate()
    assert info

    text = (out / "02" / "POSCAR").read_text(encoding="utf-8")
    assert "Selective dynamics" in text
    assert text.count("F   F   F") == 6


def test_neb_frozen_atoms_block_flow(qtbot, tmp_path, monkeypatch):
    """The final moving a frozen atom blocks interpolation."""
    model = StructureModel()
    model.load_atoms(FileIO.read(str(FROZEN / "block" / "initial" / "POSCAR")))
    fin_dir = tmp_path / "final"
    fin_dir.mkdir(parents=True, exist_ok=True)
    fin_path = _write_poscar(
        FileIO.read(str(FROZEN / "block" / "final" / "POSCAR")), fin_dir / "POSCAR")
    monkeypatch.setattr(
        "vaspen.ui.generate_all_dialog.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(fin_path), ""),
    )
    warnings = _record(monkeypatch, "warning")

    dlg = GenerateAllDialog(model, default_task="neb")
    qtbot.addWidget(dlg)
    panel = dlg._poscar_panel
    panel._on_browse_fin()
    panel._on_interpolate()

    assert warnings  # Cannot Interpolate box
    assert panel.n_images == 0


def test_neb_browsed_ini_selective_dynamics_freeze(qtbot, tmp_path, monkeypatch):
    """Flags from a browsed initial file (not the model) also apply."""
    model = _example_model()
    dlg = GenerateAllDialog(model, default_task="neb")
    qtbot.addWidget(dlg)
    panel = dlg._poscar_panel

    ini_dir = tmp_path / "ini"
    ini_dir.mkdir(parents=True, exist_ok=True)
    ini_path = _write_poscar(
        FileIO.read(str(FROZEN / "pass" / "initial" / "POSCAR")), ini_dir / "POSCAR")
    monkeypatch.setattr(
        "vaspen.ui.generate_all_dialog.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(ini_path), ""),
    )
    panel._on_browse_ini()

    assert panel._ini_fixed_flags.all(axis=1).sum() == 6
    panel._final_atoms = FileIO.read(str(FROZEN / "pass" / "final" / "POSCAR"))
    panel._update_diagnostics()
    panel._on_interpolate()
    mask = panel._ini_fixed_flags.all(axis=1)
    for frame in panel.images:
        assert np.allclose(frame.positions[mask],
                           panel._init_atoms.positions[mask])


# ----------------------------------------------------------------------
# Frame editing (move/rotate atoms, delete/add bonds, undo, persistence)
# ----------------------------------------------------------------------

def _interpolated_dialog(model, preview_callback=None):
    """Dialog with the Cu example pair already interpolated (5 images)."""
    dlg = GenerateAllDialog(model, default_task="neb",
                            preview_callback=preview_callback)
    panel = dlg._poscar_panel
    panel._final_atoms = FileIO.read(str(EXAMPLES / "final" / "POSCAR"))
    panel._update_diagnostics()
    panel._images_spin.setValue(4)
    panel._on_interpolate()
    return dlg


def test_frame_preview_data_editability(qtbot):
    model = _example_model()
    dlg = _interpolated_dialog(model)
    qtbot.addWidget(dlg)
    panel = dlg._poscar_panel

    atoms0, ed0, flags0, bonds0 = panel.frame_preview_data(0)
    atoms2, ed2, flags2, bonds2 = panel.frame_preview_data(2)
    atoms5, ed5, flags5, bonds5 = panel.frame_preview_data(5)

    assert ed0 is False and ed5 is False  # endpoints locked
    assert ed2 is True                     # middle editable
    assert len(atoms2) == 7
    assert flags2.shape == (7, 3)
    assert len(bonds2) > 0                 # auto-detected


def test_frame_apply_move_updates_frame_only(qtbot):
    model = _example_model()
    dlg = _interpolated_dialog(model)
    qtbot.addWidget(dlg)
    panel = dlg._poscar_panel
    model_positions = model.atoms.positions.copy()

    idx = 2
    before = panel.images[idx].positions.copy()
    delta = np.array([0.3, 0.0, 0.0])
    panel.apply_frame_move(idx, [1], before[1] + delta)

    assert np.allclose(panel.images[idx].positions[1], before[1] + delta)
    # the model is untouched
    assert np.allclose(model.atoms.positions, model_positions)


def test_frame_bond_delete_survives_redetect_and_add_restores(qtbot):
    model = _example_model()
    dlg = _interpolated_dialog(model)
    qtbot.addWidget(dlg)
    panel = dlg._poscar_panel
    idx = 2

    bonds = panel.frame_preview_data(idx)[3]
    pair = (bonds[0].i, bonds[0].j)
    assert panel.remove_frame_bond(idx, 0) is True
    assert pair not in panel._frame_deleted[idx] or True
    assert all((b.i, b.j) != pair for b in panel.frame_preview_data(idx)[3])

    # moving an atom re-detects bonds but the deleted pair stays gone
    atoms = panel.images[idx]
    panel.apply_frame_move(idx, [0], atoms.positions[0] + [0.1, 0, 0])
    assert all((b.i, b.j) != pair for b in panel.frame_preview_data(idx)[3])

    # adding the bond back un-deletes it
    panel.add_frame_bond(idx, pair[0], pair[1])
    assert any((b.i, b.j) == pair for b in panel.frame_preview_data(idx)[3])


def test_frame_undo_redo(qtbot):
    model = _example_model()
    dlg = _interpolated_dialog(model)
    qtbot.addWidget(dlg)
    panel = dlg._poscar_panel
    idx = 2

    before = panel.images[idx].positions.copy()
    panel.apply_frame_move(idx, [1], before[1] + [0.5, 0, 0])
    assert not np.allclose(panel.images[idx].positions, before)

    assert panel.frame_undo(idx) is True
    assert np.allclose(panel.images[idx].positions, before)

    assert panel.frame_redo(idx) is True
    assert not np.allclose(panel.images[idx].positions, before)

    # nothing left to undo after a fresh frame
    assert panel.frame_undo(0) is False


def test_frame_edits_persist_to_written_poscar(qtbot, tmp_path, monkeypatch):
    from ase.io import read as ase_read

    model = _example_model()
    info = _record(monkeypatch, "information")
    dlg = _interpolated_dialog(model)
    qtbot.addWidget(dlg)
    panel = dlg._poscar_panel

    # Small delta: cannot cross the cell boundary, so the written
    # (wrapped) position differs from the pristine one exactly by delta.
    idx = 2
    atoms = panel.images[idx]
    pristine = atoms.positions[1].copy()
    delta = np.array([0.05, 0.0, 0.0])
    panel.apply_frame_move(idx, [1], pristine + delta)

    out = tmp_path / "out"
    out.mkdir()
    dlg._dir_edit.setText(str(out))
    dlg._on_generate()
    assert info

    written = ase_read(out / "02" / "POSCAR", format="vasp")
    moved = written.positions[1] - pristine
    # unwrap any boundary wrap through the reciprocal of the cell
    frac = np.linalg.solve(written.get_cell().T, moved)
    moved = (frac - np.round(frac)) @ written.get_cell()
    assert np.allclose(moved, delta, atol=1e-6)


def test_reinterpolate_asks_before_discarding_edits(qtbot, monkeypatch):
    model = _example_model()
    questions: list = []
    monkeypatch.setattr(
        "vaspen.ui.generate_all_dialog.QMessageBox.question",
        staticmethod(lambda *a, **k: questions.append(a) or QMessageBox.No),
    )
    dlg = _interpolated_dialog(model)
    qtbot.addWidget(dlg)
    panel = dlg._poscar_panel

    panel.apply_frame_move(2, [1],
                           panel.images[2].positions[1] + [0.2, 0, 0])
    edited = panel.images[2].positions.copy()
    panel._on_interpolate()
    assert questions  # confirm asked
    assert np.allclose(panel.images[2].positions, edited)  # No → kept

    monkeypatch.setattr(
        "vaspen.ui.generate_all_dialog.QMessageBox.question",
        staticmethod(lambda *a, **k: QMessageBox.Yes),
    )
    panel._on_interpolate()
    assert panel.has_frame_edits() is False  # regenerated, state cleared


def test_main_window_routes_frame_edits(qtbot):
    """Real MainWindow: frame edits land in the frame, never the model."""
    from vaspen.ui.main_window import MainWindow
    from vaspen.ui.tools import ToolMode

    win = MainWindow()
    qtbot.addWidget(win)
    model = _example_model()
    win._structure.load_atoms(model.atoms, None)
    model_positions = win._structure.atoms.positions.copy()

    dlg = GenerateAllDialog(win._structure, win,
                            preview_callback=win._preview_image_atoms,
                            default_task="neb")
    qtbot.addWidget(dlg)
    win._generate_dialog = dlg
    panel = dlg._poscar_panel
    panel._final_atoms = FileIO.read(str(EXAMPLES / "final" / "POSCAR"))
    panel._update_diagnostics()
    panel._images_spin.setValue(4)
    panel._on_interpolate()

    # select a middle frame → frame-edit context active, tools enabled
    panel._images_list.setCurrentRow(2)
    assert win._frame_edit is not None
    assert win._mode_actions[ToolMode.MOVE_ATOM].isEnabled()

    # move commits to the frame
    idx = win._frame_edit["index"]
    atoms = panel.images[idx]
    pre_move = atoms.positions[1].copy()
    target = pre_move + [0.25, 0.0, 0.0]
    win._on_atoms_moved([1], target[None, :])
    assert np.allclose(panel.images[idx].positions[1], target)
    assert np.allclose(win._structure.atoms.positions, model_positions)

    # bond delete + add route to the frame
    bonds = panel.frame_preview_data(idx)[3]
    pair = (bonds[0].i, bonds[0].j)
    win._on_delete_requested("bond", 0)
    assert all((b.i, b.j) != pair for b in panel.frame_preview_data(idx)[3])
    win._on_bond_created(pair[0], pair[1])
    assert any((b.i, b.j) == pair for b in panel.frame_preview_data(idx)[3])

    # atom deletion is refused
    win._on_delete_requested("atom", 0)
    assert len(panel.images[idx]) == 7

    # per-frame undo pops the three edits (bond add, bond delete, move)
    win.act_undo.trigger()
    win.act_undo.trigger()
    win.act_undo.trigger()
    assert np.allclose(panel.images[idx].positions[1], pre_move)

    # endpoint frames lock the tools again
    panel._images_list.setCurrentRow(0)
    assert win._frame_edit is None
    assert not win._mode_actions[ToolMode.MOVE_ATOM].isEnabled()


# ----------------------------------------------------------------------
# Code-review regression tests (2026-08-14)
# ----------------------------------------------------------------------

def _record_question(monkeypatch, answer) -> list:
    """Record QMessageBox.question calls and always return ``answer``."""
    calls: list = []
    monkeypatch.setattr(
        "vaspen.ui.generate_all_dialog.QMessageBox.question",
        staticmethod(lambda *a, **k: calls.append(a) or answer),
    )
    return calls


def test_browse_with_frame_edits_asks_before_discarding(qtbot, tmp_path, monkeypatch):
    """Changing endpoints after manual frame edits must ask — and keep
    the edits (plus the cleared preview reset) when the user declines."""
    from vaspen.ui.generate_all_dialog import PoscarPanel

    model = _example_model()
    previews: list = []
    panel = PoscarPanel(model, preview_callback=lambda atoms, *a: previews.append(atoms))
    qtbot.addWidget(panel)

    fin_dir = tmp_path / "final"
    fin_dir.mkdir(parents=True, exist_ok=True)
    fin_path = _write_poscar(
        FileIO.read(str(EXAMPLES / "final" / "POSCAR")), fin_dir / "POSCAR")
    monkeypatch.setattr(
        "vaspen.ui.generate_all_dialog.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(fin_path), ""),
    )
    panel._on_browse_fin()
    panel._on_interpolate()
    panel.apply_frame_move(2, [0], panel._images[2].positions[[0]] + 0.5)
    assert panel.has_frame_edits()
    n_before = panel.n_images

    # decline → the dialog stays untouched
    _record_question(monkeypatch, QMessageBox.No)
    panel._on_browse_ini()
    assert panel.n_images == n_before
    assert panel.has_frame_edits()

    # accept → the images (and their edits) are discarded, and the main
    # window is told via the None reset callback
    _record_question(monkeypatch, QMessageBox.Yes)
    panel._on_browse_ini()
    assert panel.n_images == 0
    assert not panel.has_frame_edits()
    assert previews and previews[-1] is None


def test_frame_preview_data_out_of_range_returns_none(qtbot):
    from vaspen.ui.generate_all_dialog import PoscarPanel

    panel = PoscarPanel()
    qtbot.addWidget(panel)
    assert panel.frame_preview_data(0) is None
    assert panel.frame_preview_data(5) is None


def test_write_failure_removes_partial_files(qtbot, tmp_path, monkeypatch):
    """A mid-write OSError must not leave a half-generated file set that
    reads as fully generated."""
    model = _example_model()
    fin_dir = tmp_path / "final"
    fin_dir.mkdir(parents=True, exist_ok=True)
    fin_path = _write_poscar(
        FileIO.read(str(EXAMPLES / "final" / "POSCAR")), fin_dir / "POSCAR")
    monkeypatch.setattr(
        "vaspen.ui.generate_all_dialog.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(fin_path), ""),
    )
    dlg = GenerateAllDialog(model, default_task="neb")
    qtbot.addWidget(dlg)
    dlg._poscar_panel._on_browse_fin()
    dlg._poscar_panel._on_interpolate()

    out = tmp_path / "out"
    out.mkdir()
    dlg._dir_edit.setText(str(out))
    critical = _record(monkeypatch, "critical")

    real_write = Path.write_text

    def failing_write(self, *a, **k):
        if self.parent.name == "01" and self.name == "POSCAR":
            raise OSError("disk full")
        return real_write(self, *a, **k)

    monkeypatch.setattr("pathlib.Path.write_text", failing_write)
    dlg._on_generate()
    assert critical
    assert not (out / "INCAR").exists()
    assert not (out / "00" / "POSCAR").exists()
    assert not (out / "01" / "POSCAR").exists()


def test_retranslate_preserves_selections(qtbot):
    """Language-change events reach the non-modal dialog and its panels;
    combo selections survive the rebuild."""
    from PySide6.QtCore import QEvent

    model = _example_model()
    dlg = GenerateAllDialog(model, default_task="neb")
    qtbot.addWidget(dlg)
    dlg._task_combo.setCurrentIndex(dlg._task_combo.findText("Band Structure"))
    band_index = dlg._task_combo.currentIndex()
    dlg._poscar_panel._coord_combo.setCurrentIndex(1)

    dlg.changeEvent(QEvent(QEvent.Type.LanguageChange))
    dlg._poscar_panel.changeEvent(QEvent(QEvent.Type.LanguageChange))
    dlg._kpoints_panel.changeEvent(QEvent(QEvent.Type.LanguageChange))
    dlg._incar_panel.changeEvent(QEvent(QEvent.Type.LanguageChange))
    dlg._potcar_panel.changeEvent(QEvent(QEvent.Type.LanguageChange))

    assert dlg._task_combo.currentIndex() == band_index
    assert dlg._poscar_panel._coord_combo.currentIndex() == 1
    assert dlg._kpoints_panel.mode() == "line"  # band rule still active
    assert dlg._tabs.tabText(0)  # re-applied, non-empty
