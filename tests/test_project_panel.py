"""Tests for the project panel and MainWindow project wiring (§7.15)."""

from pathlib import Path

import pytest
from ase.build import bulk
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from vaspen.core.project import Project
from vaspen.ui.main_window import MainWindow
from vaspen.ui.project_panel import ProjectPanel


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


@pytest.fixture
def project(tmp_path):
    proj = Project.create(tmp_path / "proj", "Panel Demo")
    f = tmp_path / "Fe.vasp"
    bulk("Fe").write(f, format="vasp")
    proj.add_structure(f, metadata={"formula": "Fe"})
    proj.add_run("scf", "scf", {"preset": "scf", "kpoints_mode": "automatic", "kpoints_params": {"k_spacing": 0.04}, "functional": "PBE"}, "fe", "sha256:x", "gui")
    return proj


def _run_items(panel: ProjectPanel):
    root = panel._tree.topLevelItem(1)  # Runs group
    return [root.child(i).text(0) for i in range(root.childCount())]


def test_window_has_project_dock_and_menu(window):
    assert window._dock_project.objectName() == "dock_project"
    texts = [a.text() for a in window._menu_file.actions()]
    assert any("Project" in t for t in texts)  # new/open/close actions
    assert window._project is None


def test_adopt_project_refreshes_panel_and_recents(window, project):
    window._adopt_project(project)
    assert window._project_panel._header.text() == "Panel Demo"
    assert window._config.recent_projects == [str(project.root)]
    structs = window._project_panel._tree.topLevelItem(0)
    assert structs.childCount() == 1
    assert _run_items(window._project_panel) == ["001-scf"]


def test_panel_double_click_opens_structure(window, project, qtbot):
    window._adopt_project(project)
    panel = window._project_panel
    structs = panel._tree.topLevelItem(0)
    item = structs.child(0)
    item.setSelected(True)
    seen = []
    panel.structure_open_requested.connect(seen.append)
    panel._on_double_clicked(item, 0)
    assert seen and seen[0].endswith("Fe.vasp")
    # wired into the main window's open path
    qtbot.wait(50)
    assert window._structure.n_atoms == 1


def test_run_status_change_updates_project(window, project):
    window._adopt_project(project)
    window._on_run_status_change("001-scf", "running")
    assert project.run_entry("001-scf")["status"] == "running"
    assert _run_items(window._project_panel) == ["001-scf"]
    # statuses render in the tree's second column
    runs_root = window._project_panel._tree.topLevelItem(1)
    assert runs_root.child(0).text(1) == "running"


def test_close_project_clears_panel(window, project):
    window._adopt_project(project)
    window._on_close_project()
    assert window._project is None
    assert window._project_panel._project is None


def test_add_current_structure_guard(window):
    window._adopt_project(Project.create(
        window._config.last_directory or ".", "x")) \
        if False else None
    # no project / empty model → silent no-op
    window._on_add_current_structure()
    assert window._project is None


def test_run_parameters_menu_signal(window, project, monkeypatch):
    """The panel signal opens the RunDialog wired to the right run
    (fake the modal — a real exec() would block the test)."""
    opened = []

    class _FakeRunEditor:
        DialogCode = 1  # Accepted

        def __init__(self, project, run_id, parent=None):
            opened.append(run_id)

        def exec(self):
            return self.DialogCode

    monkeypatch.setattr(
        "vaspen.ui.run_editor.RunEditorDialog", _FakeRunEditor)
    window._adopt_project(project)
    panel = window._project_panel
    panel.run_parameters_requested.emit("001-scf")
    assert opened == ["001-scf"]
    # accepted → the panel resynced via _sync_project_tools (the
    # bridge-specific PROJECT_REF rebind is covered in test_bridge.py)


def test_run_tooltip_shows_parameters(window, project):
    window._adopt_project(project)
    panel = window._project_panel
    tooltip = panel._run_tooltip("001-scf")
    assert "preset: scf" in tooltip
    assert "functional: PBE" in tooltip


def test_run_expands_to_file_tree(window, project):
    window._adopt_project(project)
    panel = window._project_panel
    runs_root = panel._tree.topLevelItem(1)
    run_item = runs_root.child(0)
    # add_run wrote run.json; the tree lists it as a file child
    files = [run_item.child(i).text(0)
             for i in range(run_item.childCount())]
    assert "run.json" in files
    child = next(c for c in (run_item.child(i) for i in
                             range(run_item.childCount()))
                 if c.text(0) == "run.json")
    assert child.data(0, Qt.ItemDataRole.UserRole) == "run-file"


def test_run_file_double_click_routes_by_kind(window, project, qtbot):
    window._adopt_project(project)
    panel = window._project_panel
    runs_root = panel._tree.topLevelItem(1)
    run_item = runs_root.child(0)
    json_child = next(c for c in (run_item.child(i) for i in
                                  range(run_item.childCount()))
                      if c.text(0) == "run.json")
    seen = []
    panel.file_open_requested.connect(seen.append)
    panel._on_double_clicked(json_child, 0)  # run.json → text viewer
    assert seen and seen[0].endswith("run.json")


def test_delete_run_menu_flow(window, project, monkeypatch):
    window._adopt_project(project)
    panel = window._project_panel
    run_dir = project.run_dir("001-scf")
    asked = []
    monkeypatch.setattr(
        "vaspen.ui.project_panel.QMessageBox.question",
        staticmethod(lambda *a, **k: asked.append(True) or QMessageBox.Yes))
    deleted = []
    panel.run_deleted.connect(deleted.append)
    panel._on_delete_run("001-scf")
    assert asked
    assert project.runs == []
    assert not run_dir.exists()
    assert deleted == ["001-scf"]


def test_delete_run_declined_keeps_run(window, project, monkeypatch):
    window._adopt_project(project)
    panel = window._project_panel
    monkeypatch.setattr(
        "vaspen.ui.project_panel.QMessageBox.question",
        staticmethod(lambda *a, **k: QMessageBox.No))
    panel._on_delete_run("001-scf")
    assert project.runs  # still registered


def test_delete_active_run_refused(window, project, monkeypatch):
    window._adopt_project(project)
    project.set_run_status("001-scf", "submitted")
    warned = []
    monkeypatch.setattr(
        "vaspen.ui.project_panel.QMessageBox.warning",
        staticmethod(lambda *a, **k: warned.append(True) or QMessageBox.Ok))
    window._project_panel._on_delete_run("001-scf")
    assert warned
    assert project.runs  # untouched
