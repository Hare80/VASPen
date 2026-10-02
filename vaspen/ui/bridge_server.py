"""Live bridge: the running GUI hosts the MCP tool surface.

A localhost TCP listener (daemon thread) receives tool requests from
the ``vaspen-mcp`` proxy and executes them on the Qt main thread
against the window's LIVE structure — every model mutation goes
through ``StructureModel``'s own signals, so the viewport, structure
tree, measurements and the file watcher all update exactly as if the
user had performed the action. When the GUI closes, the proxy falls
back to the headless standalone session (see §7.13).

Threading follows the established idiom (surface_dialog's
QObject-signal relay): the worker thread emits a queued signal carrying
the request; the main-thread slot runs the tool and fills the response;
the worker waits on a ``threading.Event`` with a timeout so a wedged
GUI surfaces as a clean "busy" error instead of hanging the AI client.

Session semantics: tools run with ``tools.SESSION`` bound to a
:class:`LiveSession` that reads ``window._structure`` fresh on every
access (the window swaps the model object on New) and refuses
``open_structure`` while the user has unsaved changes — the AI never
silently discards in-window work.
"""

from __future__ import annotations

import base64
import json
import logging
import secrets
import socket
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QObject, Signal

from vaspen.mcp_server import tools as mcp_tools
from vaspen.utils.config import AppConfig

logger = logging.getLogger("vaspen.bridge")

DEFAULT_PORT = 8765
#: Ports tried before giving up (each attempt writes the winner back to
#: the config, so the proxy always finds the actual port).
_PORT_ATTEMPTS = 20

#: Mirrors the bridge proxy's read ceiling; a tool exceeding this
#: surfaces as "GUI busy" rather than hanging the AI client.
EXEC_TIMEOUT_S = 300.0

#: Same exception set the stdio server maps onto ToolError.
_REQUEST_EXC = (ValueError, KeyError, IndexError, FileNotFoundError,
                OSError, RuntimeError)


class LiveSession:
    """``tools.ServerSession``-compatible view of the window's model."""

    def __init__(self, window) -> None:
        self._window = window
        # Tool bodies take it (transport compatibility); the bridge
        # executes serially on the GUI thread, so it never contends.
        self.lock = threading.Lock()

    # -- accessed per call (the window swaps the model on New) ---------

    @property
    def model(self):
        return self._window._structure

    @property
    def filepath(self) -> str | None:
        return self._window._structure.filepath

    def require_model(self):
        model = self._window._structure
        if model is None or model.n_atoms == 0:
            raise ValueError(
                "No structure loaded — call open_structure(path) first.")
        return model

    def replace_atoms(self, atoms, fixed_flags=None, magmoms=None):
        model = self._window._structure
        model.replace_atoms(atoms, fixed_flags=fixed_flags,
                            magmoms=magmoms)
        return model

    def load(self, path: str | Path):
        """Open a file in the window — manual-open semantics minus the
        dialogs (failures raise so the AI gets a clean error).

        Refuses while the user has unsaved changes: an AI-triggered
        open must never silently discard in-window work.
        """
        model = self._window._structure
        if model.is_dirty:
            raise ValueError(
                "The VASPen window has unsaved changes — save or "
                "discard them first (ask the user), then retry "
                "open_structure.")
        model.load(path)
        # Same registration side effects as MainWindow._open_file.
        self._window._config.add_recent_file(str(path))
        self._window._config.last_directory = str(Path(path).parent)
        self._window._update_recent_ui()
        self._window._set_status(QCoreApplication.translate(
            "MainWindow", "Loaded: {}").format(str(path)))
        logger.info("Opened file via bridge: %s", path)
        return model


class BridgeServer(QObject):
    """Localhost listener that runs MCP tools on the GUI main thread."""

    #: worker thread → main thread; the payload dict carries the request
    #: and receives the response (queued — receivers live on the main
    #: thread, the emitter does not).
    _execute = Signal(object)

    def __init__(self, window) -> None:
        super().__init__(window)
        self._window = window
        self._session = LiveSession(window)
        self._token = ""
        self._listen_socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._execute.connect(self._run_tool)

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        """Bind, publish port+token, start the accept loop."""
        config = AppConfig()
        base_port = config.bridge_port or DEFAULT_PORT
        token = secrets.token_hex(16)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # SO_REUSEADDR must stay POSIX-only: on Windows it permits
        # binding a port another process is ACTIVELY listening on
        # (hijack), which misroutes bridge requests to a stale server.
        # The bump-port loop below handles conflicts instead.
        if sys.platform != "win32":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        bound: int | None = None
        for attempt in range(_PORT_ATTEMPTS):
            candidate = int(base_port) + attempt
            try:
                sock.bind(("127.0.0.1", candidate))
                bound = candidate
                break
            except OSError:
                continue  # port taken — try the next one
        if bound is None:
            sock.close()
            raise OSError(
                f"no free MCP bridge port in {base_port}.."
                f"{base_port + _PORT_ATTEMPTS - 1}")
        sock.listen(1)
        self._listen_socket = sock
        self._token = token
        config.bridge_port = bound
        config.bridge_token = token
        config.sync()
        # The window's model is now the session every bridge tool sees.
        mcp_tools.SESSION = self._session
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name="vaspen-bridge")
        self._thread.start()
        logger.info("MCP live bridge listening on 127.0.0.1:%d", bound)
        self._window._set_status(QCoreApplication.translate(
            "MainWindow", "MCP bridge listening on port {}").format(bound))

    def stop(self) -> None:
        # Disconnect FIRST: an emit from the worker thread into a
        # half-deleted QObject is a native crash (queued metacalls
        # targeting a dead receiver). Qt purges pending events for the
        # destroyed receiver; in-flight requests get the shutdown
        # error from the guards below.
        try:
            self._execute.disconnect()
        except (RuntimeError, TypeError):
            pass
        self._stopping.set()
        if self._listen_socket is not None:
            try:
                self._listen_socket.close()
            except OSError:
                pass
            self._listen_socket = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- worker thread -------------------------------------------------

    def _serve(self) -> None:
        sock = self._listen_socket
        while not self._stopping.is_set():
            try:
                conn, _addr = sock.accept()
            except OSError:
                return  # socket closed by stop()
            try:
                self._handle(conn)
            except Exception:
                logger.exception("bridge request failed")
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(5.0)  # the request follows immediately
        buf = bytearray()
        while not buf.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                return
            buf.extend(chunk)
            if len(buf) > 16 * 1024 * 1024:
                return  # oversized request — drop the connection
        try:
            request = json.loads(bytes(buf).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            response = {"ok": False, "error": f"bad request: {exc}"}
        else:
            response = self._dispatch(request)
        conn.sendall((json.dumps(response) + "\n").encode("utf-8"))

    def _dispatch(self, request: dict) -> dict:
        if request.get("token") != self._token:
            return {"ok": False, "error": "bridge token mismatch"}
        if self._stopping.is_set():
            return {"ok": False, "error": "GUI is closing — bridge is down"}
        name = request.get("tool", "")
        fn = next((f for f in mcp_tools._TOOLS if f.__name__ == name),
                  None)
        if fn is None:
            return {"ok": False, "error": f"unknown tool: {name}"}
        payload = {"fn": fn, "args": request.get("args") or {},
                   "done": threading.Event(), "response": None}
        self._execute.emit(payload)  # queued → main thread
        if not payload["done"].wait(EXEC_TIMEOUT_S):
            return {"ok": False,
                    "error": f"GUI busy — tool {name} timed out after "
                             f"{EXEC_TIMEOUT_S:.0f}s"}
        return payload["response"]

    # -- main thread ---------------------------------------------------

    def _run_tool(self, payload: dict) -> None:
        """Execute one tool against the live session (GUI thread)."""
        if self._stopping.is_set():
            # Window is closing — never touch widgets after closeEvent.
            payload["response"] = {"ok": False,
                                   "error": "GUI is closing — bridge is down"}
            payload["done"].set()
            return
        fn, args = payload["fn"], payload["args"]
        name = fn.__name__
        logger.info("bridge tool %s", name)
        self._window._set_status(QCoreApplication.translate(
            "MainWindow", "MCP: {} (running)").format(name))
        try:
            result = fn(**args)
        except _REQUEST_EXC as exc:
            logger.warning("bridge tool %s failed: %s", name, exc)
            payload["response"] = {"ok": False, "error": str(exc)}
        else:
            if isinstance(result, bytes):  # PNG (see tools.py docstring)
                payload["response"] = {
                    "ok": True,
                    "image": {"data": base64.b64encode(result).decode(),
                              "format": "png"},
                }
            else:
                payload["response"] = {"ok": True, "result": result}
            self._window._set_status(QCoreApplication.translate(
                "MainWindow", "MCP: {} (done)").format(name))
            # Tools that wrote the file (save_structure) must refresh the
            # watcher baseline — StructureModel.save emits no signal, so
            # without this the GUI would treat its own bridge save as an
            # external change and prompt. Idempotent for all other tools.
            self._window._sync_file_watcher()
        payload["done"].set()
