"""Single-instance startup (§7.14, VESTA-style).

Launching VASPen again while it is running must not spawn a second
process or window: the new process hands its command-line file (if
any) to the running instance through a local named pipe
(``QLocalServer``/``QLocalSocket`` — per-user namespace, no token
needed) and exits. The running window then raises itself and opens
the file through the same ``_open_file`` path as File → Open.

This channel is independent of the MCP live bridge (§7.13) and works
even when the bridge is disabled in Settings.
"""

from __future__ import annotations

import json
import logging

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

logger = logging.getLogger(__name__)

SERVER_NAME = "VASPen"
_CONNECT_TIMEOUT_MS = 300


def forward_to_running(filepath: str | None) -> bool:
    """Hand ``filepath`` to the already-running VASPen, if any.

    Returns True when a running instance accepted the handoff (the
    caller should exit); False when no instance is running — the
    caller becomes the single instance.
    """
    sock = QLocalSocket()
    sock.connectToServer(SERVER_NAME)
    if not sock.waitForConnected(_CONNECT_TIMEOUT_MS):
        # No running instance (or a stale pipe — the fresh listen in
        # the new process will clean it up on POSIX).
        sock.abort()
        return False
    payload = json.dumps({"path": filepath}) + "\n"
    sock.write(payload.encode("utf-8"))
    sock.flush()
    if not sock.waitForBytesWritten(_CONNECT_TIMEOUT_MS):
        # On Windows the pipe write completes when the PEER reads; the
        # peer lives in another process with a running event loop, so
        # wait for the graceful close instead of dropping the bytes.
        logger.debug("single-instance write pending — flushing via "
                     "graceful disconnect")
    sock.disconnectFromServer()
    if sock.state() != QLocalSocket.LocalSocketState.UnconnectedState:
        sock.waitForDisconnected(_CONNECT_TIMEOUT_MS)
    return True


class SingleInstanceServer(QObject):
    """Listens for second-launch handoffs on the named pipe."""

    #: a second launch arrived; path may be None (just raise the window)
    open_requested = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._server = QLocalServer(self)

    @staticmethod
    def remove_stale_server() -> None:
        """Drop a POSIX socket node left behind by a crashed instance.

        Only call on the STARTUP path (we are the first instance) — on
        POSIX this unlinks the active socket node of a running
        instance, breaking its handoff channel (a no-op on Windows).
        """
        QLocalServer.removeServer(SERVER_NAME)

    def listen(self) -> bool:
        """Start listening; False when the name is taken (a second
        window inside the same process — degrade gracefully)."""
        if not self._server.listen(SERVER_NAME):
            logger.warning("single-instance listen failed: %s",
                           self._server.errorString())
            return False
        self._server.newConnection.connect(self._on_new_connection)
        return True

    def stop(self) -> None:
        """Deterministically release the pipe (window close). Without
        this the name lingers until the QObject is actually destroyed,
        and on Windows a second server's connections can still land on
        the stale listener."""
        self._server.close()

    def _on_new_connection(self) -> None:
        while self._server.hasPendingConnections():
            sock = self._server.nextPendingConnection()
            sock.readyRead.connect(lambda s=sock: self._read(s))
            # The second launch may have written and disconnected before
            # this slot ran — buffered data does not re-trigger
            # readyRead, so read whatever is already there right now.
            if sock.bytesAvailable():
                self._read(sock)

    def _read(self, sock) -> None:
        payload = bytes(sock.readAll())
        filepath: str | None = None
        try:
            filepath = json.loads(payload.decode("utf-8")).get("path")
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.warning("single-instance payload unreadable")
        if filepath is not None and not isinstance(filepath, str):
            filepath = None
        self.open_requested.emit(filepath)
