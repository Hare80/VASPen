"""Tests for single-instance startup (§7.14, VESTA-style).

A second launch forwards its CLI file to the running window through a
QLocalServer named pipe and exits — no second process or window.
"""

from pathlib import Path

import threading

import pytest
from ase.build import bulk
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from vaspen.ui.main_window import MainWindow
from vaspen.ui.single_instance import SingleInstanceServer, forward_to_running


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _forward(filepath):
    """forward_to_running from a worker thread — the same-process
    server can only read while the main thread pumps events."""
    result = {}

    def run():
        result["r"] = forward_to_running(filepath)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    while worker.is_alive():
        QTest.qWait(25)
    return result["r"]


def test_forward_without_running_instance_returns_false(tmp_path):
    # No window/server in this test yet — the connect must fail and
    # report "we are the first instance".
    assert _forward(str(tmp_path / "whatever.vasp")) is False
    assert _forward(None) is False


def test_forward_hands_file_to_running_window(window, tmp_path):
    f = tmp_path / "forwarded.vasp"
    bulk("Si", "diamond", a=5.43, cubic=True).write(f, format="vasp")
    assert _forward(str(f)) is True
    QTest.qWait(300)  # let the server deliver open_requested
    assert window._structure.n_atoms == 8
    assert window._structure.filepath == str(f)


def test_forward_without_file_only_raises(window):
    """A bare second launch raises the window and loads nothing."""
    _si_path = Path("nonexistent-for-forward.vasp")
    assert _forward(None) is True
    QTest.qWait(300)
    assert window._structure.n_atoms == 0  # nothing loaded


def test_second_server_same_name_does_not_crash(window):
    """Two servers on one name: POSIX rejects the second listen;
    Windows named pipes allow multiple instances — single-instance-ness
    comes from the main() guard, not from pipe exclusivity."""
    second = SingleInstanceServer(window)
    assert second.listen() in (True, False)
    # Windows named pipes allow several listeners on one name and a
    # new connection may land on ANY of them — close the extra one so
    # later forwards still reach the window's own server.
    second._server.close()
    # the window and its own server keep working either way
    assert _forward(None) is True


def test_forwarded_nonexistent_path_ignored(window, tmp_path, monkeypatch):
    """A second launch with a nonexistent file just raises the window;
    the running window shows the standard open-failure box (the user
    is at the keyboard) and loads nothing."""
    asked = []
    monkeypatch.setattr(
        "vaspen.ui.main_window.QMessageBox.critical",
        staticmethod(lambda *a, **k: asked.append(True) or QMessageBox.Ok))
    assert _forward(str(tmp_path / "nope.vasp")) is True
    QTest.qWait(300)
    assert window._structure.n_atoms == 0
    assert asked  # standard open-failure feedback
