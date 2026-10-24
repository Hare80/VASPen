"""Tests for the project panel and MainWindow project wiring (§7.15)."""

from pathlib import Path

import pytest
from ase.build import bulk

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

    class _FakeRunDialog:
        DialogCode = 1  # Accepted

        def __init__(self, project, run_id, parent=None):
            opened.append(run_id)

        def exec(self):
            return self.DialogCode

    monkeypatch.setattr(
        "vaspen.ui.run_dialog.RunDialog", _FakeRunDialog)
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
