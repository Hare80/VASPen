"""Live-bridge proxy: offer MCP tool calls to a running VASPen GUI.

Discovery: the GUI publishes a localhost port and a per-session token
through the shared QSettings store (``AppConfig.bridge_port`` /
``bridge_token``), so the proxy finds the bridge without any
fixed-port contract. When no bridge is reachable — GUI closed, live
bridge disabled, stale port — ``forward`` returns ``UNAVAILABLE`` and
the caller falls back to this process's own headless session (the
shipped standalone behavior).

Safety: calls execute on the GUI's main thread with the user's
privileges; the loopback binding plus the per-session token keep other
machines (and uninvited local processes without the token) out.
"""

from __future__ import annotations

import base64
import json
import logging
import socket

from mcp.server.mcpserver import Image
from mcp.server.mcpserver.exceptions import ToolError

from vaspen.utils.config import AppConfig

logger = logging.getLogger("vaspen.mcp.bridge")

#: Loopback connect must be instant when the GUI is up; anything else
#: reads as "no bridge" and falls back to headless execution.
_CONNECT_TIMEOUT_S = 0.5

#: Tool calls can be genuinely slow (slab cutting, IDPP); the GUI side
#: enforces the same ceiling, so this only guards a dead connection.
_READ_TIMEOUT_S = 300.0

_MAX_RESPONSE_BYTES = 256 * 1024 * 1024


class _Unavailable:
    """Sentinel: no GUI bridge reachable — fall back to headless."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "UNAVAILABLE"


UNAVAILABLE = _Unavailable()


def forward(tool: str, args: dict):
    """Forward one tool call to the running GUI bridge.

    Returns ``UNAVAILABLE`` when no bridge is reachable; otherwise the
    tool result (dict, or Image for renders). Raises ``ToolError`` with
    the GUI-side message when the tool fails there.
    """
    config = AppConfig()
    port = config.bridge_port
    token = config.bridge_token
    if not port or not token:
        return UNAVAILABLE
    try:
        with socket.create_connection(("127.0.0.1", int(port)),
                                      timeout=_CONNECT_TIMEOUT_S) as sock:
            sock.settimeout(_READ_TIMEOUT_S)
            request = json.dumps({"token": token, "tool": tool,
                                  "args": args}) + "\n"
            sock.sendall(request.encode("utf-8"))
            buf = bytearray()
            while not buf.endswith(b"\n"):
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) > _MAX_RESPONSE_BYTES:
                    raise ToolError("bridge response too large")
    except (OSError, ValueError):
        # Refused / stale port / GUI died mid-connect: no bridge here.
        return UNAVAILABLE
    if not buf:
        return UNAVAILABLE
    # The request was already delivered — from here on a failure must
    # NOT fall back (the GUI may have executed a mutating tool already).
    try:
        response = json.loads(bytes(buf).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ToolError(f"bridge protocol error: {exc}") from exc
    if not response.get("ok", False):
        raise ToolError(str(response.get("error", "bridge call failed")))
    if "image" in response:
        image = response["image"]
        return Image(data=base64.b64decode(image["data"]),
                     format=image.get("format", "png"))
    return response.get("result")
