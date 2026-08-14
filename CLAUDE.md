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
│   │   ├── periodic_wrap_dialog.py  # Vacuum padding dialog before saving molecules to periodic formats
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
| Default view on open | c-axis top-down fit (VESTA convention: b up, a left); molecules get 45°/30° isometric; periodic fit anchors on the **geometric cell center** (not the atom centroid) and frames the whole cell, corners included | `_fit_camera()` |
| Reset View | Toolbar button + View menu → `reset_view()` → re-fit | Added 2026-08-13 |
| In-place edits (supercell/surface/add/delete) | **Camera preserved** — `set_structure(reset_view=False)` | Far plane is computed per-frame from current atom positions so no clipping |
| Zoom | **Completely free** (VESTA-style, decided 2026-08-13) | No wall/atom limits; camera may pass through frame and atoms. Floor 1e-3 only to avoid singular view math. (Earlier wall-limited zoom was rejected as too strict.) |
| Atom near-plane handling | **Near-distance fade** (decided 2026-08-13) | Sphere shader fades alpha in `[NEAR_FADE_END=0.25, NEAR_FADE_START=0.8]` Å view depth + two-sided lighting — zooming into atoms never shows a cross-section disk |
| Highlight on click | Color-only (amber), **no size change** | |
| Cell frame | Depth-tested, semi-transparent blue — occludable by atoms; drawn with `GL_DEPTH_CLAMP` so it stays fully visible when the camera is inside the cell (no near-plane cross-section) | |
| Near plane | Fixed 0.01 | |
| Picking | ID-color FBO readback; pick pass draws edge-scale spheres so rim clicks hit | |

### 7.4 Why QSettings for configuration?

`QSettings` is Qt's built-in persistent key-value store. It automatically picks the right backend (Windows registry, Linux `~/.config/`, macOS plist). No extra dependency needed.

### 7.5 Non-periodic → periodic wrap policy (settled 2026-08-13)

- **Structure file formats** — resolved through the module-level `resolve_format()` in file_io.py, not ASE's glob matching: extensionless `POSCAR`/`CONTCAR` (any case) and `.poscar`/`.contcar` map to `vasp` on every platform. `FileIO.read` also fixes ASE's CIF reader never setting pbc (cif + full-rank cell → pbc=True); scoped to cif only so xyz molecules round-trip unchanged. The Save As filter advertises `*.vasp *.poscar *.contcar POSCAR CONTCAR` as one VASP entry.
- **Saving a molecule to a periodic format** (vasp-family / cif) converts the **model in place** (方案 B, MS-style): `MainWindow._ensure_periodic_for()` shows `PeriodicWrapDialog` (vacuum padding, default **5 Å**, range 0.5–50, "remember" checkbox → `periodic_wrap_padding` / `remember_wrap_padding` in QSettings). On OK, `StructureModel.make_periodic(padding)` wraps: cell = diag(bbox extent + 2×padding), atoms shifted so bbox center = cell center (exactly `padding` vacuum on every face), pbc=(T,T,T). Undoable, emits `structure_modified`, camera preserved, filepath NOT reset. Cancel aborts the save.
- The same flow runs before **Generate All Input Files** (POSCAR + KPOINTS mesh need a cell). Saving to xyz never triggers it.
- **Slabs** (pbc partially True with a full-rank cell) count as periodic and are never re-boxed. `is_periodic` = `cell.rank == 3 and pbc.any()` (structure.py).
- **`FileIO.write` stays pure** (no dialogs, no mutation): saving a rank-<3 cell to vasp/cif raises a translatable `ValueError` before any file is created — the UI wrap makes this unreachable in normal flow.

### 7.6 Materials Project Integration (planned — security policy, 2026-08-13)

Materials Project (MP) integration lets users search and download structures by formula or `mp-XXXX` ID and look up computed properties (band gap, formation energy, …) via the MP API (pymatgen `MPRester` — pymatgen is already a dependency, no new package). It is offline-first and optional: the app is fully functional without it, and any network failure degrades to a status-bar message.

**API key handling (settled — never re-litigate):**
- The key is the user's personal credential. Store it only in `QSettings` (OS-native backend: Windows registry / `~/.config` / macOS plist) — never in the repo, project files, logs, crash reports, or generated VASP input files.
- Settings UI displays the key masked (last 4 characters) with a note that it is a personal credential.
- `MP_API_KEY` environment variable is an optional fallback. **No built-in default key** — a key bundled in an open-source repo would be public.
- Logging redacts the key (centralized in `utils/logger.py`).

**Network policy:**
- HTTPS only; TLS certificate verification is never disabled.
- All requests run on a worker (`QThread`) with timeouts — never block the GUI thread.

**Input sanitization (injection defenses):**
- Formulas are validated by parsing with pymatgen `Composition` before any request is built; invalid input is rejected client-side.
- API criteria are constructed only through a whitelist UI (property + numeric range). User text is never `eval()`'d / `exec()`'d into criteria.
- Identifiers must match `^mp-\d+$` before use in an API call or filename.

**Response handling:**
- Responses are parsed strictly as JSON/structures and treated as data — never executed, never rendered as HTML/JS in rich-text widgets.
- Downloaded files are named from the sanitized materials_id (e.g. `mp-1234.cif`), never from raw formula or user input → no path traversal.

**Attribution:** MP terms of use require citation in publications and forbid bulk redistribution; the UI labels downloaded data with its MP source and materials_id.

Planned module: `vaspen/core/materials_project.py` (pure logic, no Qt); UI is a modal search/download dialog created fresh per invocation (existing i18n rules in §11.2 apply).

### 7.7 Unified "Generate All Input Files" dialog (settled 2026-08-14)

Calculate → Generate All Input Files opens one dialog
(`vaspen/ui/generate_all_dialog.py`) with POSCAR/INCAR/KPOINTS/POTCAR
tabs; files are written only when its Generate button is clicked
(output directory chosen inside the dialog). The INCAR/KPOINTS/POTCAR
tabs embed the **same panels** as the standalone dialogs — each editor
is now a QWidget panel (`IncarEditorPanel`/`KpointsEditorPanel`/
`PotcarPanel`) wrapped in a thin QDialog shell that delegates unknown
attribute lookups to the panel (`__getattr__`), so there is exactly one
implementation per file type.

**Dialog modality (settled 2026-08-14):** the generate dialog is
**non-modal** (same pattern as the surface dialog) — a modal dialog
centered over the main window would cover the 3D viewport and hide the
NEB frame preview. While it is open, structure-editing entry points are
paused via `_set_preview_editing_enabled` and restored on `finished`;
the viewport is rebound to the model when the dialog closes. Clicking a
frame previews it via `_preview_image_atoms`, which forces a
synchronous `repaint()` — the GL viewport can otherwise defer a
scheduled repaint for seconds, which reads as "clicking does nothing".

**Non-periodic structures (settled 2026-08-14):** KPOINTS panels never
fabricate a mesh for a molecule — when the model has no full-rank cell,
the automatic-mode preview shows a '#'-prefixed note ("the structure
is not periodic — wrap it in a periodic cell first") and generation is
blocked ('#'-content is rejected, matching the line-mode error
pattern). The normal flow wraps molecules via the periodic gate
(`_ensure_periodic_for`) before the dialog opens; the dialog itself
also refuses Generate on a non-periodic model (the model can change
underneath a non-modal dialog via drag-and-drop open) and its
construction can never crash on a degenerate cell.

**Task coupling** (task selector syncs the tabs; user may override
afterwards):

| Task | Rule |
|------|------|
| band | KPOINTS switches to line-mode, pre-filled with a lattice-aware path from pymatgen `HighSymmKpath` (Setyawan-Curtarolo convention; symmetry-point coordinates transformed from the standardized primitive reciprocal basis into the input cell's basis — `suggest_band_path()` in vasp_input.py). Fallback: the default `G-X|X-M|M-G` path. |
| neb | POSCAR tab switches to initial/final selection + interpolation; KPOINTS switches to the automatic mesh; after interpolation `IMAGES` is filled into the INCAR tab (re-applied whenever the task re-selects NEB). |
| custom | Nothing forced. |
| other | POSCAR tab normal, KPOINTS automatic. |

**NEB interpolation** (`vaspen/core/neb.py`, pure logic, no Qt):
- Linear per-atom interpolation in fractional coordinates using the
  per-component minimal-image displacement; the lattice of every frame
  equals the initial cell; endpoints included (n_images + 2 frames,
  1–98).
- Distance metric = Euclidean norm of the full 3N-atom minimal-image
  displacement vector (Å). Suggested images = `ceil(distance / 0.8)`.
  The implementation was cross-checked against the classic reference
  tooling during development (distance identical to ~1e-15, frame
  coordinates to ~6e-15); the comparison values are pinned in
  `tests/test_neb.py`. No third-party tool names appear in code/docs.
- **Strict file order, never auto-reordered.** A diagnostic
  (same-element optimal assignment via scipy, warning only) reports
  when reordering would shorten the path by >25% AND >1 Å; the user
  may then continue in file order or cancel. A non-blocking warning
  also appears when >25 images are suggested.
- Initial/final must have equal per-element counts and equal cells
  (variable-cell NEB is blocked with a clear message). Non-periodic
  endpoints run the standard wrap flow (`PeriodicWrapDialog`, on a
  copy — the model is untouched).
- Output layout: `00/POSCAR`…`0N/POSCAR` subdirectories + INCAR /
  KPOINTS / POTCAR in the root (standard VASP NEB run layout).
  Clicking a frame previews it in the 3D viewport (temporary; the
  viewport is rebound to the model when the dialog closes). An
  IMAGES/actual-count mismatch is confirmed before writing.

---

## 8. Default-Value Reference (community standards)

### 8.1 INCAR Presets

The INCAR strategy uses **layered defaults by calculation type**. Implemented in `vaspen/core/vasp_input.py`:

| Tag | SCF | Optimization | Band | DOS | Optical |
|-----|-----|-------------|------|-----|---------|
| ENCUT | 400 | 400 | 400 | 400 | 400 |
| ISMEAR | 0 | 0 | 0 | -5 | 0 |
| SIGMA | 0.05 | 0.05 | 0.05 | 0.05 | 0.01 |
| EDIFF | 1E-6 | 1E-6 | 1E-6 | 1E-6 | 1E-6 |
| EDIFFG | — | -0.01 | — | — | — |
| IBRION | — | 2 | — | — | — |
| ISIF | — | 3 | — | — | — |
| NSW | — | 100 | — | — | — |
| NELM | 60 | — | — | — | — |
| ISPIN | 1 | 1 | 1 | 1 | 1 |
| PREC | Normal | Normal | Normal | Normal | Normal |
| LREAL | Auto | Auto | Auto | Auto | Auto |
| LWAVE | F | F | F | F | F |
| LCHARG | F | F | T¹ | F | F |
| LORBIT | — | — | — | 11 | — |
| NEDOS | — | — | — | 2001 | 2000 |
| LOPTICS | — | — | — | — | T |
| CSHIFT | — | — | — | — | 0.1 |

¹ Band LCHARG=T: the preset uses ICHARG=11, which reads the CHGCAR the
previous SCF run must have written.

NEB preset (VTST-style, settled 2026-08-14): SCF base + EDIFFG -0.02,
IBRION 3, POTIM 0, IOPT 1 (LBFGS), ICHAIN 0, NSW 500, SPRING -5,
LCLIMB T; IMAGES defaults to EMPTY and must be filled in by the user
(both generation paths block on empty-value tags).

**INCAR text format (settled 2026-08-14, values confirmed with the
user):** `format_incar_content()` renders every tag exactly once
(insertion order) as `  TAG    =  value          (comment)` — trailing
parenthesized English comments from `INCAR_TAG_COMMENTS`, "(" aligned
at column 26; booleans as `.TRUE.`/`.FALSE.`; `SYSTEM` is a plain line
(VASP reads the whole line after "=" as the name, so no comment).
Commented-out suggestion lines (`# MAGMOM`, `# IVDW = 11`,
`# ISTART = 1`) follow after a blank line. The INCAR editor preview
uses the same renderer, and its accept path rejects duplicate tag rows.

### 8.2 KPOINTS Modes

| Mode | Key Parameter | Default |
|------|--------------|-----------------|
| Automatic (KSPACING) | KSPACING = 0.04 | Insulators: 0.04, Metals: 0.03 |
| Manual Mesh | n1, n2, n3 | User-specified |
| Line-mode (Band) | High-symmetry path + points per segment | 20 points/segment |

**KSPACING convention (settled 2026-08-13):** vaspkit-style units of
**2π/Å**. The automatic mode writes an explicit regular mesh (Gamma /
Monkhorst-Pack, shift `0 0 0`) with
`N_i = max(1, ceil(|b_i| / KSPACING))` where `b_i` are the normalized
reciprocal lattice vectors (`b_i·a_j = δ_ij`). Equivalent to VASP's
`KSPACING` INCAR tag (https://vasp.at/wiki/KSPACING) with
`KSPACING_vasp = 2π × KSPACING_input`. Worked examples: Si cubic
(0.04 → 5×5×5), GaAs FCC primitive (0.030 → 11×11×11, 0.020 →
16×16×16), ZnO hexagonal (0.040 → 9 9 5).

**Band tasks (settled 2026-08-14):** in the unified input-file dialog,
the band task auto-switches KPOINTS to line-mode with a pymatgen
`HighSymmKpath`-suggested path (§7.7).

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
- [ ] Materials Project integration (structure search by formula / mp-id, download & open, property lookup) — security policy in §7.6
- [ ] Remote SSH server connection + job submission
- [ ] Job queue management
- [ ] Band structure / DOS plotting (post-processing)
- [ ] Plugin system (Python entry points)
- [ ] Nuitka compiled release
