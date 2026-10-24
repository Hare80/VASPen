"""Tests for the project structure picker (§7.15 P2).

A modal chooser over the open project's registered structures, used by
the NEB endpoint rows in the Generate All dialog.
"""

from __future__ import annotations

import pytest
from ase.build import bulk
from PySide6.QtWidgets import QDialogButtonBox

from vaspen.core.project import Project
from vaspen.ui.structure_picker import StructurePickerDialog


@pytest.fixture
def project(tmp_path):
    proj = Project.create(tmp_path / "proj", "T")
    src_a = tmp_path / "initial.vasp"
    bulk("Cu", "fcc", a=3.6).write(src_a, format="vasp")
    proj.add_structure(src_a, metadata={"formula": "Cu"})
    src_b = tmp_path / "final.vasp"
    bulk("Ni", "fcc", a=3.5).write(src_b, format="vasp")
    proj.add_structure(src_b, metadata={"formula": "Ni"})
    return proj


def test_lists_structures_with_formula_and_selection(qtbot, project):
    dlg = StructurePickerDialog(project)
    qtbot.addWidget(dlg)
    assert dlg._list.count() == 2
    assert dlg._list.item(0).text() == "initial  (Cu)"
    assert dlg._list.item(1).text() == "final  (Ni)"
    assert dlg._list.currentRow() == 0  # first entry preselected
    assert dlg._buttons.button(
        QDialogButtonBox.StandardButton.Ok).isEnabled()


def test_selected_structure_path(qtbot, project):
    dlg = StructurePickerDialog(project)
    qtbot.addWidget(dlg)
    assert dlg.selected_structure_id() == "initial"
    assert dlg.selected_structure_path() == project.structure_path("initial")
    dlg._list.setCurrentRow(1)
    assert dlg.selected_structure_path() == project.structure_path("final")


def test_empty_project_shows_hint_and_disables_ok(qtbot, tmp_path):
    proj = Project.create(tmp_path / "empty", "E")
    dlg = StructurePickerDialog(proj)
    qtbot.addWidget(dlg)
    assert dlg._list.isHidden()
    assert not dlg._empty_label.isHidden()
    assert not dlg._buttons.button(
        QDialogButtonBox.StandardButton.Ok).isEnabled()
    assert dlg.selected_structure_path() is None
