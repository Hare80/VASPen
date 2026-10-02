"""``python -m vaspen.mcp_server`` — MCP server entry point.

Also serves as the PyInstaller entry script for the optional
``vaspen-mcp`` console executable (scripts/build.py --with-mcp).
"""

import sys

from vaspen.mcp_server.main import main

if __name__ == "__main__":
    sys.exit(main())
