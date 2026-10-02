"""Entry point for the VASPen MCP server (headless, stdio).

Console script ``vaspen-mcp`` / ``python -m vaspen.mcp_server``. The
stdio protocol owns stdout — application logging goes to stderr only
(:func:`vaspen.utils.logger.setup_logger` console handler). stdin and
stdout are reconfigured to UTF-8 explicitly: the MCP stdio framing is
UTF-8 JSON lines, and the platform default (e.g. GBK on Windows) would
corrupt non-ASCII content.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Run the MCP server over stdio. Returns a process exit code."""
    from vaspen import __version__

    parser = argparse.ArgumentParser(
        prog="vaspen-mcp",
        description="VASPen MCP server — headless VASPen tools "
                    "(structures, VASP inputs, NEB) for AI clients "
                    "over stdio.",
    )
    parser.add_argument("--version", action="version",
                        version=f"VASPen {__version__}")
    parser.add_argument(
        "--selftest", action="store_true",
        help="import the full tool surface and exit — frozen-build "
             "smoke check (exercises the complete import chain)")
    args = parser.parse_args(argv)

    try:
        from mcp.server.mcpserver import MCPServer  # noqa: F401
    except ImportError:
        print(
            'The MCP SDK is not installed. Run: pip install "vaspen[mcp]"',
            file=sys.stderr,
        )
        return 1

    # stdin/stdout carry the protocol; reconfigure before any I/O.
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    from vaspen.utils.logger import setup_logger

    setup_logger(name="vaspen.mcp")

    from vaspen.mcp_server.server import _TOOLS, build_server

    if args.selftest:
        # Every import above + all tool modules resolved (a missing
        # frozen submodule shows up here, not at AI-client runtime).
        build_server()
        print(f"vaspen-mcp selftest OK: {len(_TOOLS)} tools registered")
        return 0

    server = build_server()
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
