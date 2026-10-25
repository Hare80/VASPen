"""Tests for the GUI live bridge (§7.13).

The bridge executes MCP tools on the Qt main thread against the
window's LIVE structure; the vaspen-mcp proxy forwards to it when the
GUI is up and falls back to the headless session when it is not.

Test-shape note: a bridge call must be issued from a NON-main thread
with the Qt event loop pumping (qWait) — the request is delivered to
the main thread as a queued signal, which never fires while the test
function blocks. That mirrors production, where the proxy runs in its
own process.
"""

import base64
import json
import socket
import threading
import time
from pathlib import Path

import pytest
from ase.build import bulk
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from vaspen.mcp_server import server as mcp_server
from vaspen.mcp_server import tools as mcp_tools
from vaspen.mcp_server.bridge import UNAVAILABLE, forward
from vaspen.mcp_server.session import ServerSession
from vaspen.ui.bridge_server import LiveSession
from vaspen.ui.main_window import MainWindow


@pytest.fixture
def bridge_window(qtbot):
    """MainWindow with the live bridge enabled; restores the global
    tool session binding afterwards (the bridge points tools.SESSION at
    its LiveSession, which must not leak into other tests)."""
    from vaspen.utils.config import AppConfig

    AppConfig().mcp_live_bridge = True
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._bridge is not None
    yield window
    mcp_tools.SESSION = ServerSession()
    mcp_tools.PROJECT_REF = mcp_tools._ProjectRef()


def _si_file(tmp_path: Path, a: float = 5.43) -> Path:
    f = tmp_path / "si.vasp"
    bulk("Si", "diamond", a=a, cubic=True).write(f, format="vasp")
    return f


def _raw_call(window, tool, args, token=None):
    port = window._config.bridge_port
    tok = window._config.bridge_token if token is None else token
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall((json.dumps({"token": tok, "tool": tool,
                               "args": args}) + "\n").encode("utf-8"))
        buf = bytearray()
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf.extend(chunk)
    return json.loads(bytes(buf).decode("utf-8"))


def _call(window, tool, args, token=None):
    """Issue one bridge call from a worker thread while the test thread
    pumps the Qt event loop (see module docstring)."""
    result: dict = {}

    def run():
        try:
            result["response"] = _raw_call(window, tool, args, token)
        except Exception as exc:  # pragma: no cover - surfaced below
            result["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    while worker.is_alive():
        QTest.qWait(25)
    if "error" in result:
        raise result["error"]
    return result["response"]


def _forward(window, tool, args):
    """forward() from a worker thread (same event-loop constraint)."""
    result: dict = {}

    def run():
        try:
            result["response"] = forward(tool, args)
        except Exception as exc:
            result["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    while worker.is_alive():
        QTest.qWait(25)
    if "error" in result:
        raise result["error"]
    return result["response"]


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def test_bridge_starts_and_publishes_discovery(bridge_window):
    from vaspen.utils.config import AppConfig

    cfg = AppConfig()
    assert cfg.bridge_port > 0
    assert cfg.bridge_token
    # the tool core now sees the live session
    assert isinstance(mcp_tools.SESSION, LiveSession)


def test_bridge_disabled_by_suite_default(qtbot):
    """The suite-wide default (conftest) keeps the bridge off — a plain
    MainWindow never spawns a listener."""
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._bridge is None


# ----------------------------------------------------------------------
# Live tool execution
# ----------------------------------------------------------------------


def test_open_structure_loads_into_window(bridge_window, tmp_path):
    f = _si_file(tmp_path)
    response = _call(bridge_window, "open_structure", {"path": str(f)})
    assert response["ok"] is True
    assert response["result"]["formula"] == "Si8"
    assert bridge_window._structure.n_atoms == 8
    assert bridge_window._structure.filepath == str(f)


def test_mutations_are_live(bridge_window, tmp_path):
    f = _si_file(tmp_path)
    _call(bridge_window, "open_structure", {"path": str(f)})
    response = _call(bridge_window, "make_supercell",
                     {"na": 2, "nb": 2, "nc": 2})
    assert response["ok"] is True
    assert bridge_window._structure.n_atoms == 64  # the window sees it


def test_run_python_operates_on_live_model(bridge_window, tmp_path):
    f = _si_file(tmp_path)
    _call(bridge_window, "open_structure", {"path": str(f)})
    response = _call(bridge_window, "run_python", {"code": "model.n_atoms"})
    assert response["result"]["result"] == 8


def test_render_preview_returns_image_payload(bridge_window, tmp_path):
    f = _si_file(tmp_path)
    _call(bridge_window, "open_structure", {"path": str(f)})
    response = _call(bridge_window, "render_preview",
                     {"view": "z", "max_px": 200})
    png = base64.b64decode(response["image"]["data"])
    assert png[:4] == b"\x89PNG"


def test_save_via_bridge_does_not_self_trigger_reload(bridge_window,
                                                      tmp_path):
    f = _si_file(tmp_path)
    _call(bridge_window, "open_structure", {"path": str(f)})
    seen: list = []
    bridge_window._file_monitor.file_changed.connect(seen.append)
    response = _call(bridge_window, "save_structure", {"path": str(f)})
    assert response["ok"] is True
    assert not bridge_window._structure.is_dirty
    QTest.qWait(700)  # longer than the watcher debounce
    assert seen == []  # the bridge's own save must not look external


def test_on_new_swaps_the_session_model(bridge_window, tmp_path,
                                        monkeypatch):
    f = _si_file(tmp_path)
    _call(bridge_window, "open_structure", {"path": str(f)})
    monkeypatch.setattr(
        "vaspen.ui.main_window.QMessageBox.question",
        staticmethod(lambda *a, **k: QMessageBox.Yes))
    bridge_window._on_new()  # replaces the model object entirely
    response = _call(bridge_window, "get_structure_info", {})
    assert response["ok"] is False
    assert "No structure loaded" in response["error"]


# ----------------------------------------------------------------------
# Refusals / errors
# ----------------------------------------------------------------------


def test_dirty_load_is_refused(bridge_window, tmp_path, monkeypatch):
    f = _si_file(tmp_path)
    _call(bridge_window, "open_structure", {"path": str(f)})
    bridge_window._structure.set_magmom([0], 1.0)  # unsaved user work
    # The refusal is a clean tool error — not a dialog. (The second
    # structure goes to a DIFFERENT path: rewriting the watched file
    # with a dirty model would legitimately trigger the reload prompt
    # instead — that interplay is covered by test_file_watch.py.)
    asked = []
    monkeypatch.setattr(
        "vaspen.ui.main_window.QMessageBox.question",
        staticmethod(lambda *a, **k: asked.append(True) or QMessageBox.No))
    other = tmp_path / "other.vasp"
    bulk("Si", "diamond", a=6.0, cubic=True).write(other, format="vasp")
    response = _call(bridge_window, "open_structure", {"path": str(other)})
    assert response["ok"] is False
    assert "unsaved" in response["error"]
    assert bridge_window._structure.any_magmom is True
    assert asked == []


def test_token_mismatch_rejected(bridge_window, tmp_path):
    response = _call(bridge_window, "get_structure_info", {},
                     token="wrong-token")
    assert response["ok"] is False
    assert "token" in response["error"]


def test_unknown_tool_rejected(bridge_window):
    response = _call(bridge_window, "not_a_tool", {})
    assert response["ok"] is False
    assert "unknown tool" in response["error"]


# ----------------------------------------------------------------------
# Proxy routing: live forward vs headless fallback
# ----------------------------------------------------------------------


def test_proxy_forwards_when_bridge_up(bridge_window, tmp_path):
    f = _si_file(tmp_path)
    _call(bridge_window, "open_structure", {"path": str(f)})
    result = _forward(bridge_window, "get_structure_info", {})
    assert result is not UNAVAILABLE
    assert result["formula"] == "Si8"
    # mutations land in the window through the proxy path too
    _forward(bridge_window, "make_supercell", {"na": 2, "nb": 1, "nc": 1})
    assert bridge_window._structure.n_atoms == 16


def test_proxy_falls_back_when_no_bridge(qtbot):
    """No GUI bridge published (port 0) → forward returns UNAVAILABLE
    and the stdio wrapper executes headlessly."""
    from vaspen.utils.config import AppConfig

    AppConfig().bridge_port = 0
    AppConfig().bridge_token = ""
    result = _forward(None, "get_structure_info", {})
    assert result is UNAVAILABLE
    # headless fallback still works through the wrapper
    response = mcp_tools.get_structure_info.__wrapped__() \
        if hasattr(mcp_tools.get_structure_info, "__wrapped__") else None
    with pytest.raises(ValueError, match="No structure loaded"):
        mcp_tools.get_structure_info()


# ----------------------------------------------------------------------
# Projects over the bridge (§7.15)
# ----------------------------------------------------------------------


def test_bridge_create_run_shows_in_panel(bridge_window, qtbot, tmp_path):
    from vaspen.core.project import Project

    f = tmp_path / "si.vasp"
    bulk("Si", "diamond", a=5.43, cubic=True).write(f, format="vasp")
    project = Project.create(tmp_path / "proj", "Bridge Demo")

    # AI opens the project — the WINDOW adopts it (panel + recents)
    assert _call(bridge_window, "open_project",
                 {"dir": str(project.root)})["ok"] is True
    assert bridge_window._project is not None
    assert bridge_window._project_panel._header.text() == "Bridge Demo"

    # AI loads a structure and creates a run — the panel refreshes live
    _call(bridge_window, "open_structure", {"path": str(f)})
    response = _call(bridge_window, "create_run",
                     {"name": "scf bulk", "preset": "scf"})
    assert response["ok"] is True
    run_id = response["result"]["run"]["id"]

    def panel_has_run():
        runs_root = bridge_window._project_panel._tree.topLevelItem(1)
        return any(
            runs_root.child(i).text(0) == run_id
            for i in range(runs_root.childCount()))

    qtbot.waitUntil(panel_has_run, timeout=5000)

    # status changes through the bridge land in the manifest on disk
    # (the window reloaded its OWN Project instance from disk on
    # open_project — the test-local object is deliberately stale)
    assert _call(bridge_window, "set_run_status",
                 {"run_id": run_id, "status": "submitted"})["ok"] is True
    from vaspen.core.project import Project as _Project
    on_disk = _Project.load(project.root)
    assert on_disk.run_entry(run_id)["status"] == "submitted"


def test_forward_retries_once_when_bridge_not_accepting_yet(
    monkeypatch,
):
    """GUI-startup race: the first connect is refused, the retry (one
    pause later) reaches the bridge — forward must NOT silently fall
    back to headless (the AI would believe it opened a structure while
    the window shows nothing).

    Pure-socket fake server (no Qt): bound but not listening at first —
    connects are refused until the thread calls listen() after 0.3 s,
    i.e. inside the 0.5 s retry window.
    """
    from vaspen.utils.config import AppConfig

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    # deliberately NOT listening yet — connect is refused for now

    def _accept_later():
        time.sleep(0.3)
        srv.listen(1)
        conn, _ = srv.accept()
        with conn:
            payload = json.loads(conn.recv(65536).decode("utf-8"))
            response = json.dumps({
                "ok": True,
                "result": {"formula": "H2O", "echo": payload["tool"]},
            }) + "\n"
            conn.sendall(response.encode("utf-8"))

    thread = threading.Thread(target=_accept_later, daemon=True)
    thread.start()
    monkeypatch.setattr(
        AppConfig, "bridge_port", property(lambda self: port))
    monkeypatch.setattr(
        AppConfig, "bridge_token", property(lambda self: "tok"))

    result = forward("open_structure", {"path": "x"})
    assert result == {"formula": "H2O", "echo": "open_structure"}
    thread.join(5)
    srv.close()


def test_forward_never_retries_after_delivery(monkeypatch):
    """A delivered request must never be retried or fall back to
    headless — a fake bridge accepts, reads the request, then closes
    WITHOUT responding: exactly ONE delivery, forward raises ToolError
    (not UNAVAILABLE)."""
    from mcp.server.mcpserver.exceptions import ToolError

    from vaspen.utils.config import AppConfig

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    deliveries: list[str] = []

    def _accept_and_vanish():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            with conn:
                data = conn.recv(65536)
                if data:
                    deliveries.append(data.decode("utf-8"))
                # close without any response

    thread = threading.Thread(target=_accept_and_vanish, daemon=True)
    thread.start()
    monkeypatch.setattr(
        AppConfig, "bridge_port", property(lambda self: port))
    monkeypatch.setattr(
        AppConfig, "bridge_token", property(lambda self: "tok"))

    with pytest.raises(ToolError, match="delivered"):
        forward("make_supercell", {"na": 2, "nb": 1, "nc": 1})
    time.sleep(0.3)  # let a (wrong) second delivery arrive if retried
    assert len(deliveries) == 1  # exactly one delivery
    srv.close()
