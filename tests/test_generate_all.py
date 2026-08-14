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
                            preview_callback=previews.append)
    qtbot.addWidget(dlg)

    panel = dlg._poscar_panel
    assert panel.neb_mode
    panel._on_browse_fin()

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
    assert len(previews) == 1
    assert len(previews[0]) == 7

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
