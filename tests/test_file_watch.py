"""Tests for external file-change detection (GUI, §7.12).

Covers the FileChangeMonitor unit behavior (debounce, baseline,
rebase, delete+recreate) and the MainWindow integration (silent
reload on a clean model, confirm on a dirty model, smart camera,
self-save suppression, watcher lifetime across derived structures).
"""

import os
import time
from pathlib import Path

from PySide6.QtTest import QTest

import numpy as np
import pytest
from ase.build import bulk
from PySide6.QtWidgets import QMessageBox

from vaspen.ui.file_watch import FileChangeMonitor
from vaspen.ui.main_window import MainWindow


# ----------------------------------------------------------------------
# FileChangeMonitor unit behavior
# ----------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    # NTFS/FAT mtime resolution is finite; make sure consecutive writes
    # differ in mtime_ns or size.
    time.sleep(0.01)
    path.write_text(text, encoding="utf-8")


def _record(mon: FileChangeMonitor) -> list:
    """Collect file_changed emissions (a SignalInstance is always
    truthy — negative assertions need an explicit recorder)."""
    seen: list = []
    mon.file_changed.connect(seen.append)
    return seen


def test_monitor_fires_after_external_write(qtbot, tmp_path):
    f = tmp_path / "s.vasp"
    f.write_text("one")
    mon = FileChangeMonitor()
    seen = _record(mon)
    mon.watch(str(f))
    with qtbot.waitSignal(mon.file_changed, timeout=3000):
        _write(f, "two")
    assert mon.watched_path == str(f)


def test_monitor_ignores_pre_watch_changes(qtbot, tmp_path):
    f = tmp_path / "s.vasp"
    f.write_text("one")
    f.write_text("churn")  # happened BEFORE watch() — not our baseline
    mon = FileChangeMonitor()
    seen = _record(mon)
    mon.watch(str(f))  # adopts the current mtime+size as baseline
    QTest.qWait(700)
    assert seen == []


def test_monitor_rebase_suppresses_current_change(qtbot, tmp_path):
    f = tmp_path / "s.vasp"
    f.write_text("one")
    mon = FileChangeMonitor()
    seen = _record(mon)
    mon.watch(str(f))
    _write(f, "two")
    mon.rebase()  # user chose "ignore": adopt current disk state
    QTest.qWait(700)
    assert seen == []
    # A FURTHER change still fires.
    with qtbot.waitSignal(mon.file_changed, timeout=3000):
        _write(f, "three")


def test_monitor_handles_delete_and_recreate(qtbot, tmp_path):
    f = tmp_path / "s.vasp"
    f.write_text("one")
    mon = FileChangeMonitor()
    seen = _record(mon)
    mon.watch(str(f))
    with qtbot.waitSignal(mon.file_changed, timeout=3000):
        os.remove(f)
        _write(f, "recreated")


def test_monitor_silent_while_file_missing(qtbot, tmp_path):
    f = tmp_path / "s.vasp"
    f.write_text("one")
    mon = FileChangeMonitor()
    seen = _record(mon)
    mon.watch(str(f))
    os.remove(f)  # gone for good (no recreate)
    QTest.qWait(700)
    assert seen == []


def test_monitor_clear_and_rewatch(qtbot, tmp_path):
    a = tmp_path / "a.vasp"
    b = tmp_path / "b.vasp"
    a.write_text("a")
    b.write_text("b")
    mon = FileChangeMonitor()
    seen = _record(mon)
    mon.watch(str(a))
    mon.clear()
    assert seen == []
    _write(a, "changed")
    QTest.qWait(700)
    assert seen == []
    mon.watch(str(b))
    with qtbot.waitSignal(mon.file_changed, timeout=3000):
        _write(b, "changed")


# ----------------------------------------------------------------------
# MainWindow integration
# ----------------------------------------------------------------------


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _ask(monkeypatch, answers: list) -> list:
    """Replace MainWindow's QMessageBox.question with a canned-answer
    stub (existing _record pattern from test_generate_all)."""
    calls: list = []
    iterator = iter(answers)
    monkeypatch.setattr(
        "vaspen.ui.main_window.QMessageBox.question",
        staticmethod(
            lambda *a, **k: calls.append(a) or next(iterator)),
    )
    return calls


def _si_file(tmp_path: Path, a: float = 5.43) -> Path:
    f = tmp_path / "si.vasp"
    bulk("Si", "diamond", a=a, cubic=True).write(f, format="vasp")
    return f


def test_open_watches_the_loaded_file(window, tmp_path):
    f = _si_file(tmp_path)
    window._open_file(str(f))
    assert window._file_monitor.watched_path == str(f)


def test_clean_model_reloads_silently(window, qtbot, tmp_path):
    f = _si_file(tmp_path, a=5.43)
    window._open_file(str(f))
    bulk("Si", "diamond", a=5.60, cubic=True).write(f, format="vasp")
    qtbot.waitUntil(
        lambda: window._structure.cell_lengths[0] == pytest.approx(5.60),
        timeout=5000)
    assert not window._structure.is_dirty
    assert window._file_monitor.watched_path == str(f)


def test_reload_smart_camera(window, qtbot, tmp_path, monkeypatch):
    f = _si_file(tmp_path)
    window._open_file(str(f))
    calls = []
    orig = window._viewport.set_structure
    monkeypatch.setattr(
        window._viewport, "set_structure",
        lambda *a, **k: calls.append(k) or orig(*a, **k))
    # Same composition/count/cell, nudged positions → keep camera.
    moved = bulk("Si", "diamond", a=5.43, cubic=True)
    moved.positions[0] += [0.05, 0.0, 0.0]
    moved.write(f, format="vasp")
    qtbot.waitUntil(
        lambda: bool(calls) and np.allclose(
            window._structure.positions[0], moved.positions[0]),
        timeout=5000)
    assert calls[-1].get("reset_view") is False
    # A cell change re-fits.
    calls.clear()
    bulk("Si", "diamond", a=5.90, cubic=True).write(f, format="vasp")
    qtbot.waitUntil(
        lambda: window._structure.cell_lengths[0] == pytest.approx(5.90),
        timeout=5000)
    assert calls[-1].get("reset_view") is True


def test_dirty_model_confirm_yes_reloads(window, qtbot, tmp_path,
                                         monkeypatch):
    f = _si_file(tmp_path, a=5.43)
    window._open_file(str(f))
    window._structure.set_magmom([0], 1.0)  # unsaved change
    assert window._structure.is_dirty
    asked = _ask(monkeypatch, [QMessageBox.Yes])
    bulk("Si", "diamond", a=5.70, cubic=True).write(f, format="vasp")
    qtbot.waitUntil(
        lambda: window._structure.cell_lengths[0] == pytest.approx(5.70),
        timeout=5000)
    assert asked  # the confirm was shown, user chose Reload


def test_dirty_model_confirm_no_keeps_memory(window, qtbot, tmp_path,
                                             monkeypatch):
    f = _si_file(tmp_path, a=5.43)
    window._open_file(str(f))
    window._structure.set_magmom([0], 1.0)
    _ask(monkeypatch, [QMessageBox.No])
    bulk("Si", "diamond", a=5.70, cubic=True).write(f, format="vasp")
    QTest.qWait(700)
    # Memory untouched (old geometry, still dirty)…
    assert window._structure.cell_lengths[0] == pytest.approx(5.43)
    assert window._structure.is_dirty
    # …and a NEW change still prompts (answered Yes this time).
    _ask(monkeypatch, [QMessageBox.Yes])
    bulk("Si", "diamond", a=5.80, cubic=True).write(f, format="vasp")
    qtbot.waitUntil(
        lambda: window._structure.cell_lengths[0] == pytest.approx(5.80),
        timeout=5000)


def test_own_save_does_not_prompt(window, qtbot, tmp_path, monkeypatch):
    f = _si_file(tmp_path)
    window._open_file(str(f))
    window._structure.set_magmom([0], 1.0)
    asked = _ask(monkeypatch, [QMessageBox.Yes] * 10)
    window._on_save()  # writes the file the monitor is watching
    assert not window._structure.is_dirty
    QTest.qWait(700)
    assert not asked  # our own write must not trigger the confirm


def test_derived_structure_detaches_watcher(window, tmp_path):
    f = _si_file(tmp_path)
    window._open_file(str(f))
    assert window._file_monitor.watched_path == str(f)
    # Derived structure: the dialog sites call reset_filepath +
    # _sync_file_watcher — replay that contract here.
    window._structure.replace_atoms(window._structure.atoms * (2, 1, 1))
    window._structure.reset_filepath()
    window._sync_file_watcher()
    assert window._file_monitor.watched_path is None
