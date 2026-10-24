"""Tests for the run parameters dialog (§7.15 P1.5)."""

import json
from pathlib import Path

import pytest
from ase.build import bulk
from PySide6.QtWidgets import QMessageBox

from vaspen.core.project import Project
from vaspen.ui.run_dialog import RunDialog, _parse_overrides


@pytest.fixture
def project_with_run(tmp_path):
    proj = Project.create(tmp_path / "proj", "Run Demo")
    f = tmp_path / "si.vasp"
    bulk("Si", "diamond", a=5.43, cubic=True).write(f, format="vasp")
    entry = proj.add_structure(f)
    run = proj.add_run("scf", "scf", {"preset": "scf",
                                      "kpoints_mode": "automatic",
                                      "kpoints_params": {"k_spacing": 0.04},
                                      "functional": "PBE"},
                       entry["id"], "sha256:x", "gui")
    return proj, run["id"]


def test_parse_overrides():
    parsed, err = _parse_overrides("IVDW = 11\n\nMAGMOM = 0.6*56")
    assert parsed == {"IVDW": "11", "MAGMOM": "0.6*56"}
    assert err is None
    parsed, err = _parse_overrides("no equals sign")
    assert err and "missing '='" in err
    parsed, err = _parse_overrides("= 5")
    assert err and "empty tag" in err


def test_dialog_loads_card_fields(qtbot, project_with_run):
    proj, run_id = project_with_run
    dlg = RunDialog(proj, run_id)
    qtbot.addWidget(dlg)
    assert dlg._info_id.text() == run_id
    assert dlg._preset_combo.currentText() == "scf"
    assert dlg._functional_combo.currentText() == "PBE"
    assert dlg._kpoints_combo.currentText() == "automatic"
    assert dlg._kspacing_spin.value() == pytest.approx(0.04)
    assert dlg._status_combo.currentText() == "inputs-ready"


def test_dialog_kpoints_mode_switches_fields(qtbot, project_with_run):
    proj, run_id = project_with_run
    dlg = RunDialog(proj, run_id)
    qtbot.addWidget(dlg)
    dlg._kpoints_combo.setCurrentText("manual")
    # offscreen dialogs are never "visible" — assert the hidden FLAG
    assert dlg._kspacing_spin.isHidden()
    assert not dlg._k1.isHidden() and not dlg._k2.isHidden()         and not dlg._k3.isHidden()


def test_dialog_regenerate_writes_files(qtbot, project_with_run,
                                        monkeypatch):
    proj, run_id = project_with_run
    dlg = RunDialog(proj, run_id)
    qtbot.addWidget(dlg)
    dlg._preset_combo.setCurrentText("opt")
    dlg._overrides.setPlainText("IVDW = 11")
    dlg._on_regenerate()  # no outputs → no confirm
    assert dlg.result()
    incar = (proj.run_dir(run_id) / "INCAR").read_text()
    assert "EDIFFG" in incar  # opt preset applied
    assert "IVDW" in incar    # override applied
    card = json.loads((proj.run_dir(run_id) / "run.json").read_text("utf-8"))
    assert card["parameters"]["preset"] == "opt"
    assert card["parameters"]["IVDW"] == "11"


def test_dialog_regenerate_confirms_over_outputs(qtbot, project_with_run,
                                                 monkeypatch):
    proj, run_id = project_with_run
    run_dir = proj.run_dir(run_id)
    (run_dir / "CONTCAR").write_text("x", encoding="utf-8")
    asked = []
    monkeypatch.setattr(
        "vaspen.ui.run_dialog.QMessageBox.question",
        staticmethod(lambda *a, **k: asked.append(True)
                     or QMessageBox.No))
    dlg = RunDialog(proj, run_id)
    qtbot.addWidget(dlg)
    dlg._on_regenerate()  # user said No → nothing regenerated
    assert asked
    assert not dlg.result()  # rejected (0) vs Accepted (1)
    incar = run_dir / "INCAR"
    assert not incar.exists() or "EDIFFG" not in incar.read_text()


def test_dialog_override_error_blocks_regenerate(qtbot, project_with_run,
                                                 monkeypatch):
    proj, run_id = project_with_run
    dlg = RunDialog(proj, run_id)
    qtbot.addWidget(dlg)
    dlg._overrides.setPlainText("broken line without equals")
    warned = []
    monkeypatch.setattr(
        "vaspen.ui.run_dialog.QMessageBox.warning",
        staticmethod(lambda *a, **k: warned.append(True) or QMessageBox.Ok))
    dlg._on_regenerate()
    assert warned
    assert not dlg.result()


def test_text_viewer_loads_and_truncates(tmp_path, qtbot):
    from vaspen.ui.text_viewer import TextViewerDialog

    small = tmp_path / "INCAR"
    small.write_text("ENCUT = 400\n", encoding="utf-8")
    dlg = TextViewerDialog(small)
    qtbot.addWidget(dlg)
    assert "ENCUT = 400" in dlg._text.toPlainText()

    big = tmp_path / "OUTCAR"
    big.write_text("x" * (600 * 1024), encoding="utf-8")  # > 512 KB
    dlg2 = TextViewerDialog(big)
    qtbot.addWidget(dlg2)
    text = dlg2._text.toPlainText()
    assert "truncated" in text
    assert len(text) < 600 * 1024
