"""Tests for the run editor (§7.15 P1.7) and its panel loaders.

The editor embeds the Generate-All panels (§7.7: one implementation
per file type); Apply is WYSIWYG — panel content writes straight to
the run's files, and the provenance card records an edit entry.
"""

import json
from pathlib import Path

import pytest
from ase.build import bulk
from PySide6.QtWidgets import QMessageBox

from vaspen.core.project import Project
from vaspen.core.vasp_input import parse_incar_content
from vaspen.ui.kpoints_editor import KpointsEditorPanel
from vaspen.ui.run_editor import RunEditorDialog


@pytest.fixture
def project_with_runs(tmp_path):
    proj = Project.create(tmp_path / "proj", "Editor Demo")
    f = tmp_path / "si.vasp"
    bulk("Si", "diamond", a=5.43, cubic=True).write(f, format="vasp")
    s_entry = proj.add_structure(f)
    run = proj.add_run("scf", "scf", {"preset": "scf",
                                      "kpoints_mode": "automatic",
                                      "kpoints_params": {"k_spacing": 0.04},
                                      "functional": "PBE"},
                       s_entry["id"], "sha256:x", "gui")
    # standard run inputs
    run_dir = proj.run_dir(run["id"])
    (run_dir / "INCAR").write_text(
        "SYSTEM = demo\nENCUT = 400\nIMAGES = 3\n", encoding="utf-8")
    (run_dir / "KPOINTS").write_text(
        "Automatic mesh\n0\nGamma\n4 4 4\n0 0 0\n", encoding="utf-8")
    # neb run with frames
    neb = proj.add_run("neb", "neb", {"method": "idpp"}, s_entry["id"],
                       "sha256:x", "gui")
    neb_dir = proj.run_dir(neb["id"])
    (neb_dir / "INCAR").write_text(
        "SYSTEM = neb demo\nIMAGES = 3\nSPRING = -5\n", encoding="utf-8")
    (neb_dir / "KPOINTS").write_text(
        "Automatic mesh\n0\nGamma\n2 2 2\n0 0 0\n", encoding="utf-8")
    for i in range(5):
        frame = neb_dir / f"{i:02d}"
        frame.mkdir()
        (frame / "POSCAR").write_text("frame", encoding="utf-8")
    return proj


# ----------------------------------------------------------------------
# Panel loaders
# ----------------------------------------------------------------------


def test_incar_set_tags_loads_custom(qtbot, project_with_runs):
    from vaspen.ui.incar_editor import IncarEditorPanel

    panel = IncarEditorPanel()
    qtbot.addWidget(panel)
    tags, _problems = parse_incar_content(
        (project_with_runs.run_dir("001-scf") / "INCAR").read_text("utf-8"))
    panel.set_tags(tags)
    assert panel.preset_key == "custom"
    assert "IMAGES" in panel.content()
    assert "SPRING" not in panel.content()


def test_kpoints_load_content_manual_and_note(qtbot):
    panel = KpointsEditorPanel()
    qtbot.addWidget(panel)
    mode, note = panel.load_content(
        "Automatic k-point mesh (KSPACING = 0.040, 2π/Å)\n0\nGamma\n"
        "5 5 5\n0 0 0\n")
    assert mode == "manual"
    assert "equivalent" in note
    assert panel._k1_spin.value() == 5
    mode, note = panel.load_content(
        "Band structure: G-X|X-M|M-G\n60\nLine-mode\nReciprocal\n")
    assert mode == "line" and note == ""
    assert panel._band_path_edit.text() == "G-X|X-M|M-G"


# ----------------------------------------------------------------------
# Editor: load + apply
# ----------------------------------------------------------------------


def test_editor_loads_files_into_panels(qtbot, project_with_runs):
    dlg = RunEditorDialog(project_with_runs, "001-scf")
    qtbot.addWidget(dlg)
    assert "IMAGES" in dlg._incar_panel.content()
    assert dlg._kpoints_panel.mode() == "manual"
    assert dlg._kpoints_panel._k1_spin.value() == 4


def test_editor_apply_writes_and_records(qtbot, project_with_runs,
                                          monkeypatch):
    project = project_with_runs
    dlg = RunEditorDialog(project, "001-scf")
    qtbot.addWidget(dlg)
    dlg._incar_panel.apply_tag("IVDW", "11")
    dlg._status_combo.setCurrentText("submitted")
    dlg._on_apply()
    assert dlg.result()  # accepted
    incar = (project.run_dir("001-scf") / "INCAR").read_text("utf-8")
    assert "IVDW" in incar and "IMAGES" in incar
    assert project.run_entry("001-scf")["status"] == "submitted"
    card = json.loads(
        (project.run_dir("001-scf") / "run.json").read_text("utf-8"))
    assert card["edits"][-1]["via"] == "gui"


def test_editor_apply_confirms_over_outputs(qtbot, project_with_runs,
                                             monkeypatch):
    project = project_with_runs
    (project.run_dir("001-scf") / "CONTCAR").write_text("x",
                                                        encoding="utf-8")
    asked = []
    monkeypatch.setattr(
        "vaspen.ui.run_editor.QMessageBox.question",
        staticmethod(lambda *a, **k: asked.append(True) or QMessageBox.No))
    dlg = RunEditorDialog(project, "001-scf")
    qtbot.addWidget(dlg)
    dlg._on_apply()  # user declined → nothing written
    assert asked
    incar = (project.run_dir("001-scf") / "INCAR").read_text("utf-8")
    assert "IVDW" not in incar


def test_editor_neb_tabs_and_no_poscar_page(qtbot, project_with_runs):
    dlg = RunEditorDialog(project_with_runs, "002-neb")
    qtbot.addWidget(dlg)
    tab_names = [dlg._tabs.tabText(i)
                 for i in range(dlg._tabs.count())]
    assert "POSCAR" not in tab_names
    assert tab_names[:3] == ["INCAR", "KPOINTS", "POTCAR"]
    # the run's real NEB tags load (IMAGES from the file)
    content = dlg._incar_panel.content()
    assert "IMAGES" in content and "SPRING" in content


# ----------------------------------------------------------------------
# POTCAR page (per-element pseudopotentials, pseudo-H flagging)
# ----------------------------------------------------------------------

_FAKE_POTCAR = """\
    TITEL  = PAW_PBE Fe_pv 06Sep2007
    ZVAL   =  8.000
    ENMAX  = 267.983 eV; the recommended cutoff
    VRHFIN =Fe: 3d64s2
    End of Dataset
    TITEL  = PAW_PBE O 08Apr2002
    ZVAL   =  6.000
    ENMAX  = 400.000 eV
    VRHFIN =O: 2s2 2p4
    End of Dataset
    TITEL  = PAW_PBE V_h 08Apr2002
    ZVAL   =  9.019   1.000
    ENMAX  = 177.443 eV
    VRHFIN =V: 3d34s2
    VRHFIN =H: 1s1
    End of Dataset
"""


def test_potcar_page_lists_sections_and_flags_pseudo_h(
        qtbot, project_with_runs):
    run_dir = project_with_runs.run_dir("001-scf")
    (run_dir / "POTCAR").write_text(_FAKE_POTCAR, encoding="utf-8")
    dlg = RunEditorDialog(project_with_runs, "001-scf")
    qtbot.addWidget(dlg)
    table = dlg._potcar_table
    assert table.rowCount() == 3
    assert table.item(0, 0).text() == "Fe"
    assert table.item(0, 1).text() == "Fe_pv"
    assert table.item(0, 2).text() == "8"
    assert table.item(0, 3).text() == "267.983"
    assert table.item(1, 0).text() == "O"
    # the V_h section embeds a pseudo-hydrogen — flagged, not read as
    # structure hydrogen
    assert table.item(2, 0).text() == "V + pseudo-H"
    assert table.item(2, 1).text() == "V_h"
    assert not dlg._pseudo_note.isHidden()


def test_potcar_table_hidden_without_file(qtbot, project_with_runs):
    dlg = RunEditorDialog(project_with_runs, "001-scf")
    qtbot.addWidget(dlg)
    assert dlg._potcar_table.isHidden()
    assert dlg._pseudo_note.isHidden()


# ----------------------------------------------------------------------
# Info bar: NEB runs show Initial/Final, not one "structure"
# ----------------------------------------------------------------------


def test_neb_info_bar_shows_initial_and_final(qtbot, project_with_runs):
    project = project_with_runs
    s_id = project.structures[0]["id"]
    project.update_run_parameters("002-neb", {"final_structure": s_id})
    dlg = RunEditorDialog(project, "002-neb")
    qtbot.addWidget(dlg)
    assert dlg._structure_label.text() == f"Initial: {s_id}"
    assert dlg._final_label.text() == f"Final: {s_id}"


def test_neb_info_bar_final_missing_shows_placeholder(
        qtbot, project_with_runs):
    dlg = RunEditorDialog(project_with_runs, "002-neb")
    qtbot.addWidget(dlg)
    assert dlg._final_label.text() == "Final: —"


def test_scf_info_bar_keeps_structure_label(qtbot, project_with_runs):
    dlg = RunEditorDialog(project_with_runs, "001-scf")
    qtbot.addWidget(dlg)
    assert dlg._structure_label.text() == "Structure: si"
    assert dlg._final_label is None


def test_regen_potcar_preserves_variants(qtbot, project_with_runs,
                                         tmp_path, monkeypatch):
    """Regenerating a run's POTCAR keeps its parsed variants (Si_d
    stays Si_d) instead of degrading to the recommended default."""
    project = project_with_runs
    lib = tmp_path / "lib" / "potpaw_PBE.54"
    for name, tag in (("Si", "Si_plain"), ("Si_d", "Si_d_variant")):
        d = lib / name
        d.mkdir(parents=True)
        (d / "POTCAR").write_text(f"# {tag}\n", encoding="utf-8")
    monkeypatch.setattr(
        "vaspen.utils.config.AppConfig.potcar_library_path",
        property(lambda self: str(tmp_path / "lib")))
    run_dir = project.run_dir("001-scf")
    (run_dir / "POTCAR").write_text(
        "    TITEL  = PAW_PBE Si_d 05May2015\n"
        "    ZVAL   =  4.000\n"
        "    VRHFIN =Si: 3s2 3p2\n"
        "    End of Dataset\n", encoding="utf-8")
    dlg = RunEditorDialog(project, "001-scf")
    qtbot.addWidget(dlg)
    dlg._on_regen_potcar()
    content = (run_dir / "POTCAR").read_text("utf-8")
    assert "# Si_d_variant" in content
    card = json.loads((run_dir / "run.json").read_text("utf-8"))
    assert card["parameters"]["potcar_variants"] == {"Si": "Si_d"}


def test_regen_poscar_rewrites_from_registered_structure(
        qtbot, project_with_runs, monkeypatch):
    project = project_with_runs
    run_dir = project.run_dir("001-scf")
    dlg = RunEditorDialog(project, "001-scf")
    qtbot.addWidget(dlg)
    before = len(project.run_entry("001-scf").get("edits", [])
                 if "edits" in project.run_entry("001-scf") else [])
    dlg._on_regen_poscar()
    from vaspen.core.vasp_input import generate_poscar
    expected = generate_poscar(dlg._structure_model)
    assert (run_dir / "POSCAR").read_text("utf-8") == expected
    card = json.loads((run_dir / "run.json").read_text("utf-8"))
    assert card["edits"][-1]["via"] == "gui"


def test_regen_poscar_missing_structure_errors(qtbot, tmp_path,
                                               monkeypatch):
    import shutil

    criticals = []
    monkeypatch.setattr(
        "vaspen.ui.run_editor.QMessageBox.critical",
        staticmethod(lambda *a, **k: criticals.append(a) or 1))
    project = Project.create(tmp_path / "proj", "T")
    f = tmp_path / "si.vasp"
    bulk("Si", "diamond", a=5.43, cubic=True).write(f, format="vasp")
    s_entry = project.add_structure(f)
    project.add_run("scf", "scf", {}, s_entry["id"], "sha256:x", "gui")
    project.structure_path(s_entry["id"]).unlink()
    # the registered COPY inside structures/ is gone from disk
    dlg = RunEditorDialog(project, "001-scf")
    qtbot.addWidget(dlg)
    dlg._on_regen_poscar()
    assert criticals
    assert not (project.run_dir("001-scf") / "POSCAR").exists()
