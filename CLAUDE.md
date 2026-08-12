# CLAUDE.md — VASPen

## 1. Project Overview

**VASPen** is a cross-platform desktop GUI application for VASP (Vienna Ab initio Simulation Package) first-principles calculations. It provides visual structure modeling, input file generation, and job management — similar in scope to Materials Studio, BURAI, and MedeA.

**Target users:** Computational materials science researchers who use VASP.

**License:** MIT — free and open source.

---

## 2. Tech Stack

| Layer | Technology | Version | Notes |
|-------|-----------|---------|-------|
| Language | Python | ≥3.11 | |
| GUI framework | PySide6 | ≥6.6 | Qt 6.x, LGPL — commercial-friendly |
| 3D rendering | Qt native OpenGL | (Qt 6) | Hand-written GL 3.3 via QOpenGLWidget — no extra dependency |
| Structure engine | ASE | ≥3.22 | Reads/writes 100+ structure formats |
| Materials analysis | pymatgen | ≥2024.1 | VASP I/O, symmetry, band paths |
| Symmetry | spglib | ≥2.3 | Space group, k-path generation |
| Numerics | numpy + scipy | ≥1.26 / ≥1.12 | |
| i18n | Qt Linguist | (part of Qt) | `.ts` → `.qm` workflow |
| Packaging | PyInstaller → Nuitka | latest | Dev: PyInstaller; Release: Nuitka |
| Testing | pytest + pytest-qt | ≥8 / ≥4 | |

### Key Architecture Decision

VASP's file-format ecosystem lives entirely in Python (ASE + pymatgen, 15+ years of development). Rewriting this in C++/Rust/C# would be impractical for a solo developer. Python is the pragmatic choice; performance hotspots (3D rendering, large-file parsing) can be replaced with compiled extensions later (PyO3/Rust or Cython) without touching the UI layer.

---

## 3. Project Structure

```
VASPen/
├── pyproject.toml              # PEP 621 project metadata + dependencies
├── README.md
├── CLAUDE.md                   # This file — project documentation for AI assistants
├── vaspen/
│   ├── __init__.py
│   ├── main.py                 # Entry point: QApplication + MainWindow
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── main_window.py      # QMainWindow: menus, toolbars, status bar, docks
│   │   ├── viewport3d.py       # 3D viewport — QOpenGLWidget, hand-written GL
│   │   ├── structure_tree.py   # Atom list + cell parameters panel (left dock)
│   │   ├── incar_editor.py     # INCAR tag table editor with presets
│   │   ├── kpoints_editor.py   # KPOINTS mode selector + parameters
│   │   ├── potcar_dialog.py    # Pseudopotential selection + POTCAR concatenation
│   │   ├── surface_dialog.py   # Miller index input, vacuum, slab preview
│   │   ├── settings_dialog.py  # Preferences (paths, language, defaults)
│   │   └── about_dialog.py     # About dialog
│   ├── core/
│   │   ├── __init__.py
│   │   ├── structure.py        # StructureModel — wraps ASE Atoms with signals
│   │   ├── file_io.py          # Unified file read/write with format registry
│   │   ├── surface.py          # Miller-index slab cutting via ASE
│   │   ├── builder.py          # Add/remove/replace atoms, supercell, symmetry
│   │   └── vasp_input.py       # INCAR/KPOINTS/POTCAR generation with presets
│   ├── resources/
│   │   ├── i18n/
│   │   │   ├── vaspen_en.ts   # English source translations
│   │   │   └── vaspen_zh.ts   # Chinese translations
│   │   ├── templates/          # Default input file templates
│   │   └── icons/              # SVG icons
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # AppConfig — QSettings wrapper
│       └── logger.py           # Centralized logging
├── tests/
│   ├── __init__.py
│   ├── test_structure.py
│   ├── test_file_io.py
│   ├── test_vasp_input.py
│   └── test_ui.py
└── scripts/
    └── build.py                # PyInstaller / Nuitka build script
```

---

## 4. Getting Started

### 4.1 Prerequisites

- Python 3.11 or later
- Git
- A VASP pseudopotential library (optional — only needed for POTCAR generation)

### 4.2 Setup

```bash
# Clone (if applicable)
git clone <repo-url> VASPen
cd VASPen

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1
# Activate (Linux/macOS)
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run the application
python -m vaspen.main
```

### 4.3 Development Tools

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=vaspen

# Update translation files
pyside6-lupdate vaspen/ -ts vaspen/resources/i18n/vaspen_zh.ts
pyside6-lrelease vaspen/resources/i18n/*.ts

# Build Windows executable
python scripts/build.py
```

---

## 5. Coding Conventions

### 5.1 Naming (PEP 8)

| Element | Convention | Example |
|---------|-----------|---------|
| Module / file | `snake_case` | `main_window.py`, `vasp_input.py` |
| Class | `PascalCase` | `MainWindow`, `StructureModel`, `FileIO` |
| Function / method | `snake_case` | `load_structure()`, `generate_incar()` |
| Constant | `UPPER_SNAKE_CASE` | `INCAR_PRESETS`, `SUPPORTED_FORMATS` |
| Qt signal | `snake_case` verb-describing | `structure_loaded`, `atom_selected` |
| Qt slot | `_on_<widget>_<event>` | `_on_open_clicked`, `_on_atom_selected` |
| Private member | `_leading_underscore` | `self._atoms`, `self._current_file` |

### 5.2 File Organization

- One class per file (except small helper/data classes).
- File name matches the primary class name: `MainWindow` → `main_window.py`.
- UI widget files go in `ui/`. Pure logic/data files go in `core/`. Cross-cutting utilities go in `utils/`.

### 5.3 Imports

Order with a blank line between each group:

```python
# 1. Standard library
import os
from pathlib import Path

# 2. Third-party
import numpy as np
from PySide6.QtWidgets import QMainWindow, QMenu
from ase.io import read, write

# 3. Local
from vaspen.core.structure import StructureModel
from vaspen.utils.config import AppConfig
```

### 5.4 Type Annotations

All public API methods MUST have type annotations. Use `from __future__ import annotations` if needed for forward references.

```python
def load_structure(self, filepath: str | Path) -> StructureModel:
    """Load a structure from file and return a StructureModel."""
    ...
```

### 5.5 Docstrings

Use Google-style docstrings:

```python
def generate_incar(
    preset: str,
    structure: StructureModel,
    overrides: dict[str, str] | None = None,
) -> str:
    """Generate INCAR content from a preset.

    Args:
        preset: One of "scf", "opt", "band", "dos", "optical".
        structure: The structure model (used for ENCUT estimation).
        overrides: Optional tag-value overrides for the preset.

    Returns:
        INCAR file content as a string.

    Raises:
        ValueError: If preset is not recognized.
    """
```

### 5.6 Qt-Specific Patterns

**Signals and Slots:**

```python
class StructureModel(QObject):
    structure_loaded = Signal()          # emitted after load/save
    atoms_modified = Signal()            # emitted after any structural change
    atom_selected = Signal(int)          # emits atom index

class MainWindow(QMainWindow):
    def _connect_signals(self) -> None:
        self._structure.structure_loaded.connect(self._on_structure_loaded)
        self._structure.atom_selected.connect(self._on_atom_selected)
```

**i18n:** Every user-visible string MUST be wrapped in `self.tr()`:

```python
self.setWindowTitle(self.tr("VASPen"))
action_open.setText(self.tr("&Open..."))
```

**Long-running operations:** Use `QThread` or `QProgressDialog` — never block the GUI thread.

---

## 6. Architecture Patterns

### 6.1 MVC Layering

```
┌──────────────────────┐
│   UI Layer (ui/)     │  ← PySide6 widgets, signals/slots, tr()
│   Displays data,     │
│   handles user input │
├──────────────────────┤
│   Core Layer (core/) │  ← Pure Python, no Qt imports
│   Business logic,    │     (except QObject for signals in
│   data models        │      StructureModel)
├──────────────────────┤
│   Utils (utils/)     │  ← Cross-cutting: config, logging
└──────────────────────┘
```

- **UI layer** knows about Core and Utils.
- **Core layer** does NOT import from UI. It may import `PySide6.QtCore.QObject` for signal support only.
- **Utils** may be imported by any layer.

### 6.2 Signal-Driven Updates

Structure changes propagate via Qt signals, not direct method calls:

```
User action (UI)
    → StructureModel method (Core)
    → StructureModel emits signal
    → All observers update:
        ├── Viewport3D (re-render)
        ├── StructureTree (refresh table)
        └── MainWindow status bar (atom count)
```

### 6.3 Format Registry Pattern

`FileIO` uses a registry of handlers. Adding support for a new format means registering a read/write handler — no other code changes needed:

```python
class FileIO:
    _readers: dict[str, Callable] = {}
    _writers: dict[str, Callable] = {}

    @classmethod
    def register(cls, ext: str, reader: Callable, writer: Callable) -> None:
        cls._readers[ext] = reader
        cls._writers[ext] = writer
```

---

## 7. Key Design Decisions

### 7.1 Why PySide6 over PyQt6?

PySide6 is the **official** Qt for Python binding by the Qt Company, licensed LGPL. PyQt6 is Riverbank's independent binding, licensed GPLv3 (or commercial). PySide6 is more permissive for commercial distribution.

### 7.2 Why ASE + pymatgen together?

- **ASE** handles structure I/O and atomic manipulation (read/write cif, xyz, POSCAR, CONTCAR, 100+ formats).
- **pymatgen** handles VASP-specific input/output generation and symmetry analysis. It builds on ASE for some operations but adds VASP-specific knowledge (INCAR validation, POTCAR handling, band path generation).

Both are mature (15+ years) and maintained by the computational materials community.

### 7.3 Why hand-written Qt native OpenGL?

History: the 3D viewport was first built on **VisPy**, which failed empirically on this project — its transform chain drifted out of sync with the actual GPU render (atoms rendered offset from where the picking math said they were, so click-to-select stopped working), and the bug is inside VisPy's Qt integration layer where we cannot fix it.

The replacement is **hand-written OpenGL 3.3 on QOpenGLWidget** (Qt official class, LGPL, ships with PySide6 — zero new dependencies):

- Same pattern as OVITO / VESTA / Mercury — the established commercial approach for crystal visualization.
- Camera, projection and picking share ONE set of matrices computed by our own code — the transform-desync failure mode is structurally impossible.
- Ball-and-stick scenes (<1000 atoms) are trivial GPU work; Python overhead is irrelevant.
- Picking uses ID-color framebuffer readback (render each atom in a unique color to an offscreen FBO, read the pixel under the cursor) — bypasses all coordinate math, exact under occlusion and for same-element structures.
- The renderer is isolated in `ui/viewport3d.py` (~450 lines); if a compiled extension is ever needed, only this file changes.

PyVista was also considered (VTK-based) but is heavier and harder to embed in Qt.

### 7.3.1 Camera & interaction policy (settled decisions — do not re-litigate)

| Decision | Value | Notes |
|----------|-------|-------|
| Projection | **Orthographic** (default) | Perspective rejected by user ("看着有点歪") |
| Default view on open | c-axis top-down fit (VESTA convention: b up, a left); molecules get 45°/30° isometric | `_fit_camera()` |
| Reset View | Toolbar button + View menu → `reset_view()` → re-fit | Added 2026-08-13 |
| In-place edits (supercell/surface/add/delete) | **Camera preserved** — `set_structure(reset_view=False)` | Far plane is computed per-frame from current atom positions so no clipping |
| Zoom | **Completely free** (VESTA-style, decided 2026-08-13) | No wall/atom limits; camera may pass through frame and atoms. Floor 1e-3 only to avoid singular view math. (Earlier wall-limited zoom was rejected as too strict.) |
| Highlight on click | Color-only (amber), **no size change** | |
| Cell frame | Depth-tested, semi-transparent blue — occludable by atoms; drawn with `GL_DEPTH_CLAMP` so it stays fully visible when the camera is inside the cell (no near-plane cross-section) | |
| Near plane | Fixed 0.01 | |
| Picking | ID-color FBO readback; pick pass draws edge-scale spheres so rim clicks hit | |

### 7.4 Why QSettings for configuration?

`QSettings` is Qt's built-in persistent key-value store. It automatically picks the right backend (Windows registry, Linux `~/.config/`, macOS plist). No extra dependency needed.

---

## 8. Default-Value Reference (community standards)

### 8.1 INCAR Presets

The INCAR strategy uses **layered defaults by calculation type**. Implemented in `vaspen/core/vasp_input.py`:

| Tag | SCF | Optimization | Band | DOS | Optical |
|-----|-----|-------------|------|-----|---------|
| ENCUT | 400 | 400 | 400 | 400 | 400 |
| ISMEAR | 0 | 0 | 0 | -5 | 0 |
| SIGMA | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| EDIFF | 1E-6 | 1E-6 | 1E-6 | 1E-6 | 1E-6 |
| EDIFFG | — | -0.01 | — | — | — |
| IBRION | — | 2 | — | — | — |
| ISIF | — | 3 | — | — | — |
| NSW | — | 100 | — | — | — |
| PREC | Normal | Normal | Normal | Normal | Normal |
| LREAL | Auto | Auto | Auto | Auto | Auto |
| LWAVE | F | F | F | F | F |
| LCHARG | F | F | F | F | F |
| LORBIT | — | — | — | 11 | — |
| NEDOS | — | — | — | 2000 | 2000 |
| LOPTICS | — | — | — | — | T |
| CSHIFT | — | — | — | — | 0.1 |

### 8.2 KPOINTS Modes

| Mode | Key Parameter | Default |
|------|--------------|-----------------|
| Automatic (KSPACING) | KSPACING = 0.04 | Insulators: 0.04, Metals: 0.03 |
| Manual Mesh | n1, n2, n3 | User-specified |
| Line-mode (Band) | High-symmetry path + points per segment | 20 points/segment |

### 8.3 POTCAR Recommendations

| Element | Recommended Variant | Reason |
|---------|-------------------|--------|
| Most elements | PBE_54 (PBE.54) | default |
| Ga, In, Sn, Pb, Ge | *_d variants | Semi-core d electrons |
| Alkali / Alkaline earth | *_sv variants | Semi-core s/p electrons |

Default functional: PBE. Pseudopotential library root path is configured by the user (pointing to a standard pseudopotential directory or their own).

---

## 9. Testing Strategy

### 9.1 Unit Tests (pytest)

All modules in `core/` and `utils/` must have corresponding tests:

```bash
pytest tests/test_structure.py
pytest tests/test_vasp_input.py
pytest tests/test_file_io.py
```

### 9.2 UI Tests (pytest-qt)

Smoke tests for critical UI paths:

```python
def test_main_window_launch(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    assert window.windowTitle() == "VASPen"

def test_open_poscar(qtbot, tmp_path):
    # Create a minimal POSCAR, open it, verify 3D view updates
    ...
```

### 9.3 Manual Testing

- 3D rendering quality (visual inspection)
- File format compatibility with real VASP output files
- Cross-platform behavior (test on both Windows and Linux before release)

---

## 10. Build & Packaging

### 10.1 Development

```bash
python -m vaspen.main     # Run from source
```

### 10.2 PyInstaller (quick build, dev use)

```bash
pyinstaller --name VASPen \
    --windowed \
    --icon vaspen/resources/icons/app.ico \
    --add-data "vaspen/resources:resources" \
    vaspen/main.py
```

### 10.3 Nuitka (optimized build, release use)

```bash
python -m nuitka --standalone --windows-console-mode=disable \
    --enable-plugin=pyside6 \
    --include-data-dir=vaspen/resources=resources \
    --output-dir=dist \
    vaspen/main.py
```

**Important:** The pseudopotential library is NEVER bundled — users configure its path in Settings.

---

## 11. Internationalization (i18n)

### 11.1 Workflow

1. Wrap all user-visible strings in `self.tr("...")`.
2. Generate `.ts` files: `pyside6-lupdate vaspen/ -ts vaspen/resources/i18n/vaspen_zh.ts`
3. Open `.ts` in Qt Linguist, fill translations.
4. Compile: `pyside6-lrelease vaspen/resources/i18n/*.ts`
5. Load `.qm` at startup via `QTranslator`.

**Warning (2026-08-13):** `pyside6-lupdate`'s Python extractor is broken in
this environment (silently extracts 0 strings and marks all existing
entries `type="vanished"`). Do NOT run it here — edit the `.ts` files by
hand, then `pyside6-lrelease` (via `.venv/Scripts/python.exe
.venv/Scripts/pyside6-lrelease.exe`). Also note: dynamic strings via
`QCoreApplication.translate("FileIO", label)` are not statically
extractable; the FileIO context is maintained by hand in the `.ts` files.

### 11.2 Language Switching (live — no restart)

`MainWindow` owns the `QTranslator` and loads it in `__init__` (before UI creation).
Switching = reload the `.qm` + re-install the translator; Qt then posts
`QEvent.LanguageChange` to all widgets (we also broadcast it explicitly via
`app.allWidgets()`). `MainWindow.changeEvent()` catches it and calls
`_retranslate_ui()`, which re-applies every `tr()` string (window title,
actions, menus, toolbar, docks, status bar, recent-files menu).

Key pattern:

```python
def _switch_language(self, lang: str) -> None:
    app = QApplication.instance()
    app.removeTranslator(self._translator)
    self._translator.load(str(qm_path))
    app.installTranslator(self._translator)
    for widget in app.allWidgets():
        app.sendEvent(widget, QEvent(QEvent.Type.LanguageChange))

def changeEvent(self, event: QEvent) -> None:
    if event.type() == QEvent.Type.LanguageChange:
        self._retranslate_ui()
    super().changeEvent(event)
```

Rules for new code:

- Dialogs are created fresh per invocation (`dlg.exec()`) and are modal, so
  they pick up the current language automatically — no retranslate needed.
- Any future **non-modal/persistent** widget must implement `changeEvent()` +
  a `_retranslate()` method and re-apply its `tr()` strings there.
- After adding/editing user-visible strings, re-run `pyside6-lupdate` so the
  `.ts` files stay in sync.

### 11.3 Default Language

English (`en`). Chinese (`zh`) available via View → Language menu.

---

## 12. Roadmap

### v0.1 — MVP (Sprint 1-3)
- [x] Project skeleton
- [x] Main window with menus
- [x] 3D structure viewer (ball-and-stick)
- [x] Open/save cif, xyz, POSCAR, CONTCAR
- [x] INCAR/KPOINTS generation with preset defaults
- [x] POTCAR generation from local pseudopotential library
- [x] Basic structure editing (add/delete atoms, position editing, undo/redo, supercell)
- [x] Surface/slab cutting

### v0.2 — Polish (Sprint 4)
- [ ] i18n (en/zh)
- [ ] QSS theming
- [ ] Welcome page with recent files
- [ ] Drag-and-drop file opening

### v0.3 — Release (Sprint 5)
- [ ] Test suite
- [ ] PyInstaller Windows build
- [ ] User documentation

### v1.0+ — Future
- [ ] Remote SSH server connection + job submission
- [ ] Job queue management
- [ ] Band structure / DOS plotting (post-processing)
- [ ] Plugin system (Python entry points)
- [ ] Nuitka compiled release
