"""UI smoke tests — main window launch, live language switch, open path."""

import pytest
from ase import Atoms

from vaspen.core import file_io as fi
from vaspen.ui.main_window import MainWindow


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_main_window_launch(window):
    assert window.windowTitle() == "VASPen"
    assert window._viewport is not None


def test_language_switch_is_live_and_roundtrips(window):
    window._switch_language("zh")
    assert window.act_open.text() == "打开(&O)..."
    assert window._menu_calc.title() == "计算(&C)"
    window._switch_language("en")
    assert window.act_open.text() == "&Open..."


def test_open_file_builds_scene_once(window, monkeypatch):
    """Regression: _open_file used to emit structure_modified AND
    structure_loaded, rebuilding the viewport twice."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10])
    monkeypatch.setattr(
        fi.FileIO, "read", classmethod(lambda cls, p: atoms)
    )
    loaded = []
    modified = []
    window._structure.structure_loaded.connect(lambda: loaded.append(1))
    window._structure.structure_modified.connect(lambda: modified.append(1))

    window._open_file("fake.xyz")

    assert len(loaded) == 1
    assert len(modified) == 0
    assert window._structure.filepath == "fake.xyz"
    assert window._structure.is_dirty is False


def test_open_failure_shows_error(window, monkeypatch):
    def boom(cls, p):
        raise RuntimeError("bad file")
    monkeypatch.setattr(fi.FileIO, "read", classmethod(boom))
    # _open_file shows a modal error box — stub it so the test does not block
    monkeypatch.setattr(
        "vaspen.ui.main_window.QMessageBox.critical",
        staticmethod(lambda *args, **kwargs: None),
    )
    window._open_file("fake.xyz")  # must not raise
    assert window._structure.n_atoms == 0
