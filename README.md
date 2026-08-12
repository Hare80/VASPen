# VASPen

A cross-platform desktop GUI for VASP first-principles calculations. Provides visual structure modeling, VASP input file generation (INCAR, KPOINTS, POSCAR, POTCAR), and surface/slab building — similar to Materials Studio, BURAI, and MedeA.

## Features

- **Structure visualization** — ball-and-stick 3D rendering with mouse rotation, zoom, and pan (Qt native OpenGL)
- **Multi-format support** — open CIF, XYZ, POSCAR, CONTCAR, XSF, PDB and more (via ASE)
- **Structure editing** — add/remove/replace atoms, create supercells, sort/wrap atoms
- **Surface/slab cutting** — specify Miller indices (hkl), layers, and vacuum thickness
- **INCAR generation** — presets for SCF, Optimization, Band, DOS, Optical, and NEB (defaults based on [vaspkit](https://vaspkit.com))
- **KPOINTS generation** — Automatic KSPACING, manual mesh, and line-mode for band structure
- **POTCAR generation** — per-element pseudopotential selection with vaspkit-recommended variants, auto-concatenation
- **One-click input generation** — generate all four VASP input files at once
- **i18n** — English and Chinese interface (switchable)
- **Cross-platform** — Windows (primary) and Linux

## Requirements

- Python 3.11+
- PySide6 ≥ 6.6
- ASE ≥ 3.22
- pymatgen ≥ 2024.1
- spglib ≥ 2.3
- NumPy ≥ 1.26 / SciPy ≥ 1.12

See [pyproject.toml](pyproject.toml) for full dependency list.

## Quick Start

```bash
# Clone and enter project
git clone <repo-url> VASPen
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

VASPen needs a local pseudopotential library to generate POTCAR files. The library should follow vaspkit's directory structure:

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

If you already use vaspkit, point VASPen to the same POTCAR directory.

## Project Structure

```
VASPen/
├── pyproject.toml
├── README.md
├── CLAUDE.md               # Developer documentation
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
│   └── utils/
│       ├── config.py
│       └── logger.py
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

# Update translations
pyside6-lupdate vaspen/ -ts vaspen/resources/i18n/vaspen_zh.ts
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
- [x] i18n infrastructure (en/zh)
- [x] 3D viewport (Qt native OpenGL ball-and-stick rendering)
- [ ] Structure tree panel (atom list, cell parameters)
- [ ] Welcome page with recent files
- [ ] QSS theming
- [ ] SSH remote server connection + job submission (future)
- [ ] Post-processing (band structure / DOS plotting)
- [ ] Plugin system

## License

VASPen is free and open source, released under the [MIT License](LICENSE).

## Acknowledgments

Default parameters are based on [vaspkit](https://vaspkit.com) recommendations — the most widely used VASP pre/post-processing toolkit. Structure I/O is powered by [ASE](https://wiki.fysik.dtu.dk/ase/) and [pymatgen](https://pymatgen.org/).
