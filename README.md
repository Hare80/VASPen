# VASPen

A cross-platform desktop GUI for VASP first-principles calculations. Provides visual structure modeling, VASP input file generation (INCAR, KPOINTS, POSCAR, POTCAR), surface/slab building, and a NEB workflow.

## Download & Run (Windows)

- Download the latest `VASPen-v<version>-win64.zip` from
  [GitHub Releases](https://github.com/Hare80/VASPen/releases/latest).
- Unzip anywhere; run `VASPen/VASPen.exe`. **Keep `VASPen.exe` and the
  `_internal/` folder together** — moving the exe alone breaks it.
- The exe is not code-signed: on first run Windows SmartScreen may
  show a warning → **More info → Run anyway**.
- No Python installation required. See the user guides:
  [English](docs/user-guide-en.md) · [中文](docs/user-guide-zh.md)

Linux users run from source (Quick Start below).

## Features

- **Structure visualization** — ball-and-stick 3D rendering with mouse rotation, zoom, and pan (Qt native OpenGL)
- **Multi-format support** — open CIF, XYZ, POSCAR, CONTCAR, XSF, PDB and more (via ASE)
- **Structure editing** — add/remove/replace atoms, create supercells, sort/wrap atoms
- **Surface/slab cutting** — specify Miller indices (hkl), layers, and vacuum thickness; re-box slabs that already carry vacuum
- **Symmetry analysis & cell conversion** — space group / point group, symmetrize, conventional ↔ primitive cells
- **Periodic wrap** — molecules wrapped into a vacuum box before saving to VASP/CIF formats
- **INCAR generation** — presets for SCF, Optimization, Band, DOS, Optical, and NEB (community-standard defaults)
- **KPOINTS generation** — Automatic KSPACING, manual mesh, and line-mode for band structure
- **POTCAR generation** — per-element pseudopotential selection with recommended semi-core variants, auto-concatenation
- **One-click input generation** — generate all four VASP input files at once
- **NEB workflow** — Linear & IDPP interpolation, frozen-atom handling, per-frame editing, standard VASP image layout
- **Measurements** — distances, angles, dihedrals in the 3D viewport
- **MCP server** — headless `vaspen-mcp` stdio server exposing the full toolkit to AI clients (Claude Desktop, ZCode, Cursor, …)
- **i18n** — English and Chinese interface (switchable live)
- **Theming** — light/dark themes with live switching
- **Welcome page** — recent files (click selects, double-click opens) and drag-and-drop opening
- **Cross-platform** — Windows (primary) and Linux

## Requirements

- Python 3.11+
- PySide6 ≥ 6.6
- ASE ≥ 3.29
- pymatgen ≥ 2024.1
- spglib ≥ 2.3
- NumPy ≥ 1.26 / SciPy ≥ 1.12

See [pyproject.toml](pyproject.toml) for full dependency list.

## Quick Start

```bash
# Clone and enter project
git clone https://github.com/Hare80/VASPen.git VASPen
cd VASPen

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1
# Activate (Linux)
source .venv/bin/activate

# Install in editable mode
pip install -e ".[dev]"

# Run
python -m vaspen.main
```

## Configuring POTCAR Library

VASPen needs a local pseudopotential library to generate POTCAR files. The library uses the standard VASP pseudopotential directory layout:

```
potcar/
├── PBE.54/
│   ├── Fe/POTCAR
│   ├── O/POTCAR
│   ├── Ga_d/POTCAR
│   └── ...
├── PBE.64/
│   └── ...
└── LDA.54/
    └── ...
```

Set the path in **Edit → Preferences → Pseudopotential Library**.

If you already have such a library, point VASPen to the same directory.

## MCP Server (for AI clients)

VASPen ships a headless [Model Context Protocol](https://modelcontextprotocol.io)
server that exposes its structure engine and VASP input generation to AI
assistants — 23 tools covering file open/save, structure analysis
(symmetry, bonds, measurements), supercells, surface cutting, NEB
preparation and one-shot INCAR/KPOINTS/POSCAR/POTCAR generation. The
pseudopotential library path configured in the GUI preferences is
shared with the server.

**From source (Windows & Linux):**

```bash
pip install -e ".[mcp]"
vaspen-mcp          # stdio MCP server; --version for the version
```

**From the release zip (Windows):** download the
`VASPen-v<version>-win64.zip` built with `--with-mcp` and use
`VASPen/vaspen-mcp/vaspen-mcp.exe` as the command below.

Register it with your AI client, e.g. Claude Desktop
(`claude_desktop_config.json`) or ZCode:

```json
{
  "mcpServers": {
    "vaspen": {
      "command": "D:/path/to/vaspen-mcp.exe",
      "args": []
    }
  }
}
```

(For a source install, `"command": "vaspen-mcp"` usually suffices.)

## Project Structure

```
VASPen/
├── pyproject.toml
├── README.md
├── CLAUDE.md               # Developer documentation
├── docs/
│   ├── user-guide-en.md    # User manual (English)
│   ├── user-guide-zh.md    # User manual (Chinese)
│   └── reviews/            # Internal code-review reports
├── vaspen/
│   ├── main.py             # Entry point
│   ├── ui/                 # PySide6 widgets
│   │   ├── main_window.py
│   │   ├── incar_editor.py
│   │   ├── kpoints_editor.py
│   │   ├── potcar_dialog.py
│   │   ├── surface_dialog.py
│   │   ├── settings_dialog.py
│   │   └── viewport3d.py   # 3D viewport (Qt native OpenGL)
│   ├── core/               # Business logic
│   │   ├── structure.py
│   │   ├── file_io.py
│   │   ├── surface.py
│   │   ├── builder.py
│   │   └── vasp_input.py
│   ├── resources/
│   │   ├── i18n/           # Translation files (.ts/.qm)
│   │   ├── templates/
│   │   └── icons/
│   ├── utils/
│   │   ├── config.py
│   │   └── logger.py
│   ├── mcp_server/           # Headless MCP server for AI clients
│   │   ├── server.py         # Tool surface over the core layer
│   │   ├── session.py        # Current-structure session state
│   │   └── main.py           # vaspen-mcp entry (stdio)
├── tests/
└── scripts/
    └── build.py
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Update translations (hand-edit vaspen_zh.ts, then sync + compile)
python scripts/sync_en_ts.py
pyside6-lrelease vaspen/resources/i18n/*.ts

# Build Windows executable
python scripts/build.py
```

## Roadmap

- [x] Project skeleton & CLAUDE.md
- [x] Core modules (structure, file I/O, VASP input generation)
- [x] Main window with menus, toolbars, status bar
- [x] INCAR/KPOINTS/POTCAR editor dialogs
- [x] Surface/slab cutting
- [x] i18n infrastructure (en/zh, live switching)
- [x] 3D viewport (Qt native OpenGL ball-and-stick rendering)
- [x] Structure tree panel (atom list, cell parameters, selection sync)
- [x] Structure editing (add/delete atoms, position editing, undo/redo)
- [x] Test suite (unit + UI smoke tests)
- [x] Welcome page with recent files
- [x] QSS theming
- [x] PyInstaller Windows build (version resource, bundled docs, release zip, smoke test)
- [x] User documentation (bilingual en/zh user guides)
- [x] MCP server for AI clients (headless stdio, 23 tools)
- [ ] SSH remote server connection + job submission (future)
- [ ] Post-processing (band structure / DOS plotting)
- [ ] Plugin system

## License

VASPen is free and open source, released under the [MIT License](LICENSE).

## Acknowledgments

Structure I/O is powered by [ASE](https://wiki.fysik.dtu.dk/ase/) and [pymatgen](https://pymatgen.org/); high-symmetry k-path generation uses [spglib](https://spglib.readthedocs.io/).
