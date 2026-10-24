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
