"""MCP stdio server for VASPen — SDK adapter over the tool core.

This module owns the `mcp` SDK surface: it wraps the transport-free
tool functions from ``vaspen.mcp_server.tools`` (ToolError mapping,
bytes→Image, live-bridge routing) and registers them on MCPServer.
Tool descriptions/responses are English-only by design (AI-facing,
not UI): nothing here calls ``_tr()``/``tr()`` and this package is
deliberately absent from the i18n parity tests.

Error convention: core/tools raise ValueError/FileNotFoundError/...
with clean messages. The ``mcp_tool`` wrapper converts those into
``ToolError`` — the SDK turns that into an isError result whose text is
the clean message. Without it the SDK wraps unexpected exceptions as
the generic "Error executing tool …", which hides the real message.

Live bridge (§7.13): each call is first offered to a running VASPen
GUI (``bridge.forward``); when no GUI bridge is reachable the call
falls back to this process's own headless session below — the shipped
standalone behavior, unchanged.
"""

from __future__ import annotations

import functools
import json
import logging
import time
from typing import Any

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from vaspen import __version__
from vaspen.core.file_io import FileIO
from vaspen.mcp_server import bridge, tools
from vaspen.mcp_server.tools import tool  # noqa: F401  (re-export)

logger = logging.getLogger("vaspen.mcp")

# Wrapped tool registry handed to the SDK (and re-exported below, so
# tests and the selftest see name/annotations/docstring of the
# original signature through functools.wraps).
_TOOLS: list = []


def mcp_tool(fn):
    """Route a raw tool function: live GUI bridge first, then the
    local headless session; map expected core errors onto ToolError.

    functools.wraps keeps name/annotations/docstring, so the SDK's
    schema generation sees the original signature.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not args:  # the SDK (and tests with kwargs) can be forwarded
            forwarded = bridge.forward(fn.__name__, kwargs)
            if forwarded is not bridge.UNAVAILABLE:
                return forwarded
        start = time.monotonic()
        try:
            result = fn(*args, **kwargs)
        except (ValueError, KeyError, IndexError, FileNotFoundError,
                OSError, RuntimeError) as exc:
            logger.warning("tool %s failed in %.3fs: %s",
                           fn.__name__, time.monotonic() - start, exc)
            raise ToolError(str(exc)) from exc
        logger.info("tool %s ok in %.3fs",
                    fn.__name__, time.monotonic() - start)
        if isinstance(result, bytes):  # PNG from render_preview
            return Image(data=result, format="png")
        return result

    _TOOLS.append(wrapper)
    return wrapper


# Wrap + re-export every registered tool at module level so direct
# callers (tests, selftest) exercise the same routed/error-mapped path
# the SDK sees.
for _fn in tools._TOOLS:
    _wrapped = mcp_tool(_fn)
    if _fn.__name__ == "render_preview":
        # bytes→Image happens in the wrapper; annotate it so the SDK's
        # output model (built from annotations) accepts the return.
        _wrapped.__annotations__["return"] = Image
    globals()[_fn.__name__] = _wrapped


# ----------------------------------------------------------------------
# Server assembly
# ----------------------------------------------------------------------


def build_server() -> MCPServer:
    """Create the MCPServer instance with every registered tool and the
    static resources."""
    server = MCPServer(
        "VASPen",
        version=__version__,
        instructions=(
            "VASPen tools for computational materials science: "
            "open/analyze/edit structure files (CIF, XYZ, POSCAR, XSF, "
            "PDB, ...), analyze symmetry, cut surface slabs, build "
            "supercells, prepare NEB paths and generate VASP input "
            "files (INCAR/KPOINTS/POSCAR/POTCAR). Tools operate on the "
            "structure loaded with open_structure — in a running VASPen "
            "GUI they act on the window's live structure; atom indices "
            "are 0-based; lengths are Angstroms, angles degrees."
        ),
    )
    for fn in _TOOLS:
        server.tool()(fn)

    @server.resource("vaspen://formats")
    def formats() -> str:
        """Structure file formats VASPen can read and write."""
        return json.dumps({
            "read": sorted(FileIO.supported_read_formats()),
            "write": sorted(FileIO.supported_write_formats()),
        }, indent=2)

    @server.resource("vaspen://incar-presets")
    def incar_presets() -> str:
        """INCAR presets with their default tags."""
        return json.dumps(tools._incar_presets_payload(), indent=2)

    return server
