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
│   │   ├── measurement.py            # Measurement manager + dock panel (right side)
│   │   ├── display_options_dialog.py # Render settings dialog (live preview)
│   │   ├── welcome_page.py           # Empty-state landing page (recent files + quick actions, §7.10)
│   │   └── tools.py                  # Viewport interaction tools (select/move/rotate/add/delete/bond/measure)
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
│   │   ├── themes/             # light.qss + dark.qss — Fusion theme stylesheets (§7.9)
│   │   └── icons/              # App icon set — PNG + ICO (regenerate via scripts/make_icon.py)
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # AppConfig — QSettings wrapper
│       ├── logger.py           # Centralized logging
│       └── theme.py            # QSS theming engine (Fusion + palette + QSS, §7.9)
├── examples/                    # Curated demo structures + NEB trees (README inside; tests never read from here — §9)
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared pytest fixtures — structures built in code (§9)
│   ├── test_structure.py
│   ├── test_file_io.py
│   ├── test_vasp_input.py
│   ├── test_surface.py
│   ├── test_neb.py
│   ├── test_generate_all.py
│   ├── test_dialogs.py
│   ├── test_ui.py
│   ├── test_ui_tools.py
│   ├── test_theme.py
│   ├── test_welcome_page.py
│   ├── test_i18n.py
│   ├── test_bonds.py
│   ├── test_measure.py
│   ├── test_menu_button.py
│   ├── test_render_settings.py
│   ├── test_symmetry.py
│   ├── test_transform.py
│   ├── test_viewport_geometry.py
│   └── test_icon_assets.py      # icon letterbox zero-crop + asset pins
└── scripts/
    ├── build.py                # PyInstaller / Nuitka build script
    ├── make_icon.py            # Regenerate the app icon set (never crops)
    └── sync_en_ts.py           # Regenerate vaspen_en.ts as a mirror of zh.ts
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
| Projection | **Orthographic** (default) | Perspective evaluated and rejected — orthographic is the user's preferred projection |
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
- **Saving a molecule to a periodic format** (vasp-family / cif) converts the **model in place** (option B, in-place conversion): `MainWindow._ensure_periodic_for()` shows `PeriodicWrapDialog` (vacuum padding, default **5 Å**, range 0.5–50, "remember" checkbox → `periodic_wrap_padding` / `remember_wrap_padding` in QSettings). On OK, `StructureModel.make_periodic(padding)` wraps: cell = diag(bbox extent + 2×padding), atoms shifted so bbox center = cell center (exactly `padding` vacuum on every face), pbc=(T,T,T). Undoable, emits `structure_modified`, camera preserved, filepath NOT reset. Cancel aborts the save.
- The same flow runs before **Generate All Input Files** (POSCAR + KPOINTS mesh need a cell). Saving to xyz never triggers it.
- **Slabs** (pbc partially True with a full-rank cell) count as periodic and are never re-boxed. `is_periodic` = `cell.rank == 3 and pbc.any()` (structure.py).
- **`FileIO.write` stays pure** (no dialogs, no mutation): saving a rank-<3 cell to vasp/cif raises a translatable `ValueError` before any file is created — the UI wrap makes this unreachable in normal flow.
- **CIF space-group headers (settled 2026-08-16):** the CIF reader sanitizes unusable space-group headers before ASE parses — a number tag (`_symmetry_Int_Tables_number` etc.) valued `0`/`?`/`.`/blank is dropped, and a blank/unknown H-M symbol is rewritten to `'P 1'` (identity database metadata; the file's explicit symmetry operations then define the structure — a `0`-numbered export with its own ops parses exactly). Usable headers pass through byte-identical. Pinned by tests in test_file_io.py.

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
- **Algorithm selector** (settled 2026-08-14): the NEB page offers
  Linear (default) and IDPP. Linear per-atom interpolation in
  fractional coordinates using the per-component minimal-image
  displacement; the lattice of every frame equals the initial cell;
  endpoints included (n_images + 2 frames, 1–98).
- **IDPP** (`interpolate_idpp`): starts from the linear path and
  relaxes the intermediate images with the image-dependent pair
  potential so the interatomic-distance change between adjacent
  images is uniform — atoms never pass through each other (the linear
  path of a methyl rotation collides hydrogens; IDPP does not). Uses
  the built-in IDPP relaxation of ASE (ase.mep.idpp_interpolate,
  mic=True, fmax=0.1, steps=100, no traj/log files); the endpoints
  are restored to exact copies and frames are wrapped back into the
  cell. `examples/neb_ethane_rotation/` demonstrates the contrast.
- **Frozen atoms (settled 2026-08-14):** fully-frozen atoms of the
  initial structure (fixed flags on all three directions) keep their
  initial position in EVERY frame (linear delta=0; IDPP honours the
  attached FixAtoms + a post-relaxation snap-back). Partially-frozen
  atoms interpolate normally — their FixScaled flags are enforced by
  VASP at run time. Every written image POSCAR carries the initial
  structure's Selective dynamics flags (via
  `atoms_with_fixed_constraints`). If a frozen atom has different
  positions in the initial and final structures, interpolation is
  BLOCKED with a message listing the atom (1-based) — a frozen atom
  cannot move during the NEB run either. The frozen pass/block case
  pairs are built in code and pinned by the tests (conftest fixture
  `frozen_pass_block_pair`).
- **Frame editing (settled 2026-08-14):** clicking a MIDDLE frame in
  the image list enters frame-edit mode — the viewport's move/rotate/
  delete/create-bond tools (and per-frame Ctrl+Z/Ctrl+Shift+Z) become
  active while the generate dialog is open, and every edit signal is
  routed to the frame (MainWindow `_frame_edit` context) instead of
  the model. Initial and final frames are locked (tools disabled).
  Frozen atoms are guarded by the viewport tools (frame fixed flags
  passed to `set_structure`); atoms cannot be deleted (rejected with a
  status message); bonds are per-frame state (auto-detected minus
  user-deleted pairs, kept deleted through re-detection after moves,
  manual adds un-delete) with a per-frame snapshot history (depth 20).
  Frame edits persist into the written `0N/POSCAR` files; clicking
  Interpolate again asks before discarding manual edits.
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

## 7.8 Code Review 2026-08-14 — Settled Behaviors (do not re-litigate)

First full layered review (4 review agents + light sweep, ~11.7k lines, all
findings adversarially verified; dispositions in `docs/review-2026-08-14.md`).
Behaviors pinned by the fixes:

**Preview dialog state machine (surface / generate-all):**
- Opening a preview dialog leaves editing modes entirely:
  `cancel_active_tool()` **plus `set_mode(SELECT)`** — a still-active
  ADD/MOVE/DELETE tool keeps consuming clicks and mutating the model
  during the "paused" preview.
- Both preview dialogs can be open at once; editing entry points resume
  only when the **last** one closes (each finished handler re-enables
  only if the other dialog is `None`).
- `_open_file` closes any live preview dialog first (their finished
  handlers rebind the viewport) — dropEvent and the recent-files menu
  reach it while entry points are paused.
- Selection actions (select all/none/invert/neighbors/connected) are in
  the pause list; model-selection signals never touch the viewport
  highlight during frame edit; selection clicks during a non-frame
  preview stay viewport-local (frame indices ≠ model indices).
- Browsing NEB endpoints with frame edits asks before discarding;
  clearing images notifies MainWindow via `preview_callback(None, …)`
  which exits frame-edit and rebinds the viewport.

**Model invariants:**
- `StructureModel(atoms)` direct construction initializes derived state
  (IDs/bonds/occupancy) exactly like `load_atoms` — usable immediately.
- Undo/redo snapshots carry the `dirty` flag and restore it: undoing
  every edit after a save leaves the model clean (no false
  "unsaved changes" prompt).
- `remove_bond` on an absent bond is a true no-op (no undo entry, no
  dirty flag, no signal).
- `StructureModel.load()` delegates to `FileIO.read` (same registry
  behavior as the UI open path).

**Core API contracts (translatable `ValueError`, never raw numpy/spglib):**
- `estimate_k_mesh` rejects spacing ≤ 0; `generate_all_inputs` rejects
  non-periodic models in automatic/line modes; `_d_hkl` rejects the
  zero Miller index; `align` rejects zero-length directions (NaN basis).
- `extract_fixed_flags` broadcasts one `FixScaled` (3,) mask over all
  group indices; CIF occupancy sanitizing keeps scientific notation.
- `_check_frozen_consistent` compares frozen endpoints by minimal image
  (the same site written with a different lattice translation is not
  movement).
- POTCAR generation: the standalone dialog ALWAYS regenerates on OK
  (cached content goes stale after variant/functional changes);
  `available_variants` matches repeated `_suffix` groups (`Fe_sv_GW`).
- Viewport: camera basis is NaN-safe for ±Y → Front/Back view
  sequences; `project_to_screen` rejects points behind the camera
  (ortho clip.w is always 1); `_fit_camera` ignores NaN coordinates.
- Coordinate/magmom text inputs reject non-finite values (`nan`/`inf`).
- Removed dead code: `SurfaceCutter.cut_arbitrary` (broken under ASE
  3.29, no callers), the never-emitted `bond_removed` signal.
- Generate-all write loop removes partially written files on failure
  (no half-generated set that reads as complete).

**Surface dialog responsiveness (settled 2026-08-14, user-reported hang):**
- Slab generation (pymatgen, seconds per call) NEVER runs on the GUI
  thread — `SurfaceDialog` computes on a QThread worker (`_SlabWorker`)
  with a generation counter: only the newest parameter set applies its
  result; stale workers finishing late are discarded. While computing,
  OK is disabled and the status label shows "Computing slab…".
- Auto bond detection (`find_bonds`) is capped at
  `MAX_AUTO_BOND_ATOMS = 1000` atoms — the O(N²) minimum-image distance
  matrix blows up beyond that (a 10×10 slab supercell froze the app for
  minutes / hit MemoryError at ~4.8 GB). Larger structures render
  without auto bonds; manual bonds still work per pair.

**Slab presentation (settled 2026-08-15, user request):** pymatgen's raw
slab box is sheared (for cubic (111) the c axis is NOT perpendicular to
a and b — the vacuum tilts relative to the surface normal). `slabs()`
re-expresses every slab via `get_orthogonal_c_slab()` +
`_standard_slab_cell()`: shortest in-plane basis (square for (100), hex
for (111)), c exactly along the surface normal (vacuum ⊥ ab), c length
= slab thickness + the REQUESTED vacuum (exact), slab centered along c.
All steps are unimodular re-bases + a pure rotation — distances
preserved. The layer-projection compositions were re-verified after the
fix (e.g. rocksalt (100) now correctly alternates pure O/Mg planes).

**Symmetry cell conversion (settled 2026-08-15, user request):** the
Symmetry dialog offers Conventional cell (spglib standardize,
`to_primitive=False` — the default) and Primitive cell (spglib
`to_primitive=True` + ASE `cellpar_to_cell` standard orientation:
a ∥ x, b in the xy-plane — the VESTA-style presentation; spglib's own
basis reads as a skewed box). Pure rotation — atom distances unchanged.
See examples/Cu_bulk_primitive.vasp (fcc primitive, a=3.61/√2, 60°).

**Cleave vs re-box are separate features, no vacuum detection
(settled 2026-08-15, user decision):** pymatgen SlabGenerator assumes a
dense bulk (per its manual); an automatic vacuum-detection heuristic
mis-flagged dense bulks (e.g. a 134-atom cell with near-coplanar
layers), so detection is deliberately REMOVED — no such code may be
re-introduced.
- **Calculate → Cleave Surface** (bulk inputs only; the dialog says so
  and points to the re-box feature). `slabs()` is a pure pymatgen
  pipeline (orthogonal c + standard presentation + exact vacuum).
  **Performance (settled 2026-08-15):** pymatgen's `center_slab` is a
  per-atom neighbor search (~10× slower than everything else,
  profiled) — the generator runs with `center_slab=False` and
  `_standard_slab_cell` unwraps (`_unwrap_layers`) + centers exactly;
  do NOT set center_slab back to True. Terminations are enumerated
  incrementally: `_possible_terminations` (ported from pymatgen's
  get_slabs internals, exact shift list) + `iter_slabs` generator;
  the dialog announces the total first (placeholder slots), fills
  items progressively and a click on an unloaded item jumps the
  compute queue (shared `_ComputeOrder.priority`, GIL-atomic, no
  locks). Whole-list `get_slabs` remains the fallback for unknown
  pymatgen versions. **The termination menu is built ONCE at `total`
  and never rebuilt** — arriving items update their label IN PLACE
  via `MenuButton.setItemText` (an open menu must not flicker);
  `setItemText` never emits and keeps the selection/popup state.
- **Tools → Re-box Slab** (`rebox_slab(atoms, vacuum)` +
  `vaspen/ui/rebox_dialog.py`, next to Wrap in Periodic Cell) for
  structures that already carry vacuum: unwraps layers split across
  the periodic boundary (±c translations), re-applies the vacuum along
  c with the slab centered; the in-plane cell and the atom ORDER are
  unchanged. Applied via `replace_atoms` with the frozen flags and
  magmoms carried over 1:1 (same atoms — unlike cleave, which builds a
  new atom set and clears them); filepath reset, one undo step.
- Both features confirm before discarding partial occupancy (disorder)
  — same message pattern as the symmetrize confirm.

## 7.8.1 Code Review 2026-08-15 — Settled Behaviors (do not re-litigate)

Follow-up review of the 9 commits after the 2026-08-14 review (report:
`docs/review-2026-08-15.md`). Behaviors pinned by the fixes:

**Surface computation:**
- `iter_slabs` yields every index **exactly once** — a priority jump
  advances the sequential cursor past the computed index (it used to
  rebuild the clicked slab twice).
- Closing or accepting the SurfaceDialog **cancels the pool task**
  (`_cancel_compute`: generation bump + cooperative cancel); a late
  item must never re-push a preview over the restored model.
- `termination_labels` is **deleted**. Instant exact chemistry labels
  would require replicating pymatgen's slab pipeline in miniature (the
  ouc-plane shortcut was wrong on 10/13 benchmark structures — see the
  gate table in the review report). Labels come from the built slabs
  only (stream in per item, click jumps the queue) — that path can
  never lie. Do not re-introduce a cluster/ouc-based label shortcut.
- `_surface_compositions` tolerance has a **0.1 Å floor** (merges
  rumpled mixed faces; zincblende (111) double layers stay distinct —
  pinned by test).
- `rebox_slab` raises translatable ValueError for non-periodic input or
  vacuum ≤ 0. Rebox vacuum range 0.5–100 Å (vs wrap 0.5–50) is
  intentional — slab vacuum can be larger than molecule padding.
- Per-termination c lengths may differ between shifts of the same
  cleave (pymatgen's raw material extent depends on the shift's wrap
  position); the requested vacuum is exact in every case — documented
  quirk, not a bug.

**Preview pause (extends the §7.8 list):**
- `act_auto_bonds` and the Bond Order submenu are in the pause tuple;
  bond/background viewport clicks are guarded viewport-local during
  previews (`_on_bond_clicked`/`_on_background_clicked`); on resume,
  stateful actions are re-derived (`_update_edit_actions` +
  `_sync_auto_bonds_action` + selection handlers) — never a blind
  `setEnabled(True)`.

**Dialogs:**
- SymmetryDialog's Conventional/Primitive cell selector is **periodic
  only** (hidden for point-group molecules).
- Rebox dialog remember semantics mirror the wrap dialog:
  `rebox_vacuum` / `remember_rebox` config properties (value always
  saved; the checkbox decides whether it is the default next time).
- Symmetrize resets the filepath (derived structure → Save As), like
  cleave/supercell/rebox.
- The disorder confirm lives in ONE shared helper
  `confirm_disorder_loss(parent, operation)` in `vaspen/ui/tools.py`
  (strings in the MainWindow .ts context); message pattern unchanged.
  `_confirm_disorder_poscar_save` is a different message and stays
  separate.

**i18n (extends §11.1):**
- `vaspen_en.ts` is now a **generated full mirror of zh.ts**
  (translation = source — en IS the source language): `scripts/
  sync_en_ts.py` is the ONLY regeneration path — never hand-edit both;
  the mirror property (translation == source) is pinned by a test.
- `tests/test_i18n.py` enforces two-way ast parity over **every**
  tr()/translate() context — all UI modules plus the core `_tr`
  contexts (Neb / StructureModel / VaspInput; FileIO is
  hand-maintained because its strings are dynamic `_tr(dict_value)`s,
  pinned by a dedicated test) — plus a compiled-zh.qm runtime check.
  Dynamically-translated strings (`self.tr(text)` over data tuples,
  translate callables passed to shared helpers) are whitelisted in
  `DYNAMIC_LITERALS` with a presence check. Run it after every
  string change.

## 7.8.2 App Icon Policy (settled 2026-08-15 — do not re-litigate)

- **Source artwork** = `vaspen/resources/icons/vaspen_icon.png`
  (1536×1024, 3:2 landscape, RGBA; committed to the repo — regeneration
  never depends on a file outside it). It carries a slight global
  transparency (alpha max 254) and semi-transparent rounded corners;
  the art floats inside a transparent margin. `make_icon.py`
  normalizes the alpha channel to 255 at generation time (a pure
  alpha stretch, spatial pixels untouched — NOT a crop), so the icon
  renders fully opaque; the committed source stays byte-identical to
  the original.
- **The artwork is NEVER cropped and NEVER zoomed.** Square sizes are
  produced only by letterboxing the ENTIRE source canvas — including
  the semi-transparent corners — centered on a transparent square
  canvas. The earlier center-zoom attempts (81d1041, 62e1440, bdd22de)
  and the About-dialog logo (59a1747) were all reverted by the user.
  Content loss is a hard failure pinned by `tests/test_icon_assets.py`
  (letterbox = zero-crop is verified pixel-exact).
- **`scripts/make_icon.py` is the only regeneration path** (Pillow;
  letterbox → resize → save). Outputs: `app_16/24/32/48/64/128/256.png`
  (24 added for Windows small-icon contexts), `app.png` (256×256
  **square** letterboxed fallback — never the raw 3:2 art, Qt would
  stretch it), `app.ico` (7 sizes, PNG-compressed entries — PyInstaller
  re-encodes ICO input into PNG exe resources regardless, verified
  byte-identical, so PNG is the size-optimal choice). The 1.5 MB
  source PNG rides along in the PyInstaller dist by design.
- **Scope: application icon only** (window title bar, taskbar, exe
  file icon). No toolbar/menu/About icons.
- **Windows taskbar identity**: AUMID is `"VASPen"` — never versioned
  (grouping stays stable across upgrades) — and must be set BEFORE the
  first window is created (`_set_windows_app_id()` in main.py).
  Explorer binds the taskbar button to the host process when the first
  window is shown; the previous after-show call was the "icon sometimes
  displays, sometimes doesn't" bug. The old WM_SETICON/LoadImageW hook
  was deleted — Qt propagates the window icon to the HWND itself when
  the icon is set before show. Icon failures are logged, never silently
  swallowed.
- `build.py` fails loudly when `app.ico` is missing (run
  `python scripts/make_icon.py` first).
- **Frozen layout gotcha**: PyInstaller onedir flattens the entry
  script to `_internal/main.py` while bundled data keeps the package
  layout (`_internal/vaspen/resources`) — `__file__`-relative paths
  resolve wrong when frozen. Icon loading uses `_icons_dir()` in
  main.py (sys._MEIPASS + package layout when frozen); any new
  resource-path code must do the same.

## 7.9 QSS Theming Policy (settled 2026-08-15 — do not re-litigate)

- **Two themes, "light" and "dark"** (no system-follow — user choice).
  Default light. **Both run on Qt "Fusion" style + QPalette +
  app-level QSS** (`vaspen/resources/themes/{light,dark}.qss`) — the
  light theme deliberately approximates the previous native Windows
  look; the user accepted that light is no longer the native style.
- **The engine is `vaspen/utils/theme.py`**: `apply_theme(app, name)`
  = setStyle("Fusion") → setPalette (all three ColorGroups, Disabled
  mandatory) → setStyleSheet, then records `current_theme()`.
  `main.py` applies the persisted theme before the window exists;
  `MainWindow._on_preferences` re-applies after the Settings dialog
  (live switch, no restart — same pattern as language).
  `themes_dir()` is `__file__`-relative **inside theme.py** (the
  package layout is intact when frozen; only the entry script
  flattens — do not move the resolution into main.py).
- **The two QSS files are structural mirrors** (same selector list,
  mirrored values; pinned by a selector-parity test) — edit both. No
  font rules, no scrollbar rules (the palette Light…Shadow roles ARE
  set per theme and drive Fusion scrollbars). `:default` borders stay
  1px (widening shifts dialog layouts). `QMenu::item` padding keeps
  24px left for checkable indicators. Dock title bars keep their
  DEFAULT close/float icons, and the View menu carries a
  `toggleViewAction` checkable per dock (`_add_dock_toggle_actions`)
  so a closed dock can always be re-opened; the toggle texts are
  re-synced from the dock titles in `_retranslate_ui` (Qt does not
  follow windowTitle changes).
- **No new hardcoded widget colors.** The three old hardcoded hint
  colors were replaced by `setProperty("hintKind", "error"|"warn")`
  + `QLabel[hintKind=…]` rules in both themes (property selectors
  re-evaluate on live switch). Two per-widget stylesheets remain by
  design: periodic-table element buttons (pastel fills + explicit
  dark `color: rgb(28,28,28)` — the dark palette's near-white
  ButtonText would be unreadable) and the display-options color
  swatches (textless, border works in both themes).
- **GL viewport coherence**: when the render background is still the
  OTHER theme's preset default (white ↔ `#1e1e24`), switching themes
  swaps it to the new default — and persists it
  (`MainWindow._sync_background_to_theme`, also run once at startup).
  The swap derives from the **LIVE viewport settings**, not the
  persisted config (live View-menu toggles like Show Cell are not
  persisted — pushing a stale snapshot would silently re-enable
  them). Custom backgrounds are never touched. Gradient presets are
  user choices and never swapped.
- The structure-tree element column uses
  `element_text_color(sym, dark=is_dark())` (bright colors clamped
  down for light panels, dark colors lifted for dark panels);
  `_apply_theme` refreshes the panel after switching.
- Settings UI: Theme row in the Settings dialog (MenuButton, like
  every dropdown), persisted via `AppConfig.theme` (string, clamps
  to light on corrupt values — the config module stays Qt-free).
- i18n applies: new tr() strings go through the §11.2 workflow
  (hand-edit zh.ts → sync_en_ts.py → lrelease → test_i18n).
- Tests: `tests/test_theme.py` pins the engine (palette effects,
  background-swap semantics, element-color variants, config
  clamping); every theme-mutating test restores "light" — the
  QApplication is session-wide.

## 7.10 Welcome Page Policy (settled 2026-08-15 — do not re-litigate)

- **`vaspen/ui/welcome_page.py`** (WelcomePage, context in i18n): title
  + tagline, recent-files list, New Structure / Open / Browse buttons,
  drag-and-drop hint. Standard scope — user chose it over an
  examples-section variant.
- **Interaction (user-decided, do not re-litigate)**: a click SELECTS
  a recent file; **double-click or Enter opens it** (`itemActivated`).
  **Open = opens the SELECTED recent file** (no dialog; disabled when
  nothing is selected — `selected_file()` drives it). **Browse... =
  the standard file dialog** (same `_on_open` as the File menu /
  toolbar). A small "Double-click to open a file" hint sits under the
  list.
- **Central stack**: `MainWindow._central_stack` = QStackedWidget
  (index 0 welcome page, index 1 viewport), replacing the plain
  viewport slot in `_central_layout`; the edit toolbar's
  `insertWidget(0, …)` is unaffected. `window._viewport` keeps its
  attribute name (tests depend on it).
- **Visibility keyed on the MODEL**: `_update_welcome_visibility()`
  shows the welcome page iff `self._structure.n_atoms == 0` — NEVER on
  viewport buffers, which previews (slab/NEB frames) replace
  temporarily. Called from `_on_structure_loaded`,
  `_on_structure_modified` (all atoms can be deleted) and once at the
  end of `__init__`. `_on_new` sets index 1 INLINE (the helper would
  select the welcome page for an empty model — New is an explicit
  empty session, not a return to the welcome page).
- **The viewport's own empty-state overlay** ("Open a structure file
  to begin…") is KEPT — still needed when the viewport shows a
  temporary/empty preview scene; it simply never paints while hidden
  behind the welcome page.
- **Recent files refresh**: `_update_recent_ui()` refreshes BOTH
  surfaces (File menu + welcome list) and replaces the old
  `_update_recent_menu()` call sites (open, save-as, retranslate,
  clear); the menu-bar creation call stays menu-only (the welcome
  page does not exist yet there — it is seeded in
  `_create_central_widget`).
- Welcome buttons reuse `_on_new` / `_on_welcome_open` /
  `_on_open` (Browse); list activation reuses `_open_file`. Zero new
  QSS rules — the page composes from the theme palette + existing
  selectors; the title font is set in code (`QFont`, not QSS — the
  no-font-rules policy covers QSS only).
- **New Structure semantics (settled)**: New is an explicit empty
  SESSION — it enters the viewport workspace (the viewport's own
  empty-state overlay takes over), skips the discard-confirm when
  `n_atoms == 0` (nothing to discard — asking on the welcome page was
  noise), and the welcome page returns when the model is emptied
  again (delete-all).
- i18n: WelcomePage context is in `CHECKED_MODULES`
  (tests/test_i18n.py); the page is persistent → changeEvent +
  `_retranslate()` per §11.2.
- CLI file open briefly shows the welcome page before the file loads
  (window shows before `_open_file`) — accepted quirk.

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

**Structure independence (settled 2026-08-16):** tests build their own
structures in code — shared builders live as pytest fixtures in
`tests/conftest.py` (NEB demo pairs, slab, molecules, bulks). The test
suite must **never read files from `examples/`** — that folder is
user-facing and freely editable/deletable without affecting tests.

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

`python scripts/build.py` is the canonical path (also collects ASE/
pymatgen data that the bare command below misses). The bare equivalent:

```bash
pyinstaller --name VASPen \
    --windowed \
    --icon vaspen/resources/icons/app.ico \
    --add-data "vaspen/resources;vaspen/resources" \
    vaspen/main.py
```

(The add-data dest must mirror the package layout — `vaspen/resources`,
not `resources` — the code resolves resources via `__file__`.)

**Release pipeline (settled 2026-08-16):** `scripts/build.py` now does
the full release steps after the PyInstaller build (Nuitka branch
unchanged): generates the Windows VERSIONINFO file from
`vaspen/__init__.py __version__` (single source of truth, written to
gitignored `build/version_info.txt`, passed via `--version-file`;
`--specpath build` keeps the root clean); copies LICENSE + README +
both user guides + `examples/` into `dist/VASPen/` (zip must be
self-contained; MIT text travels with the binary; the pseudopotential
library is never bundled); zips it as `dist/VASPen-v<version>-win64.zip`
with a single top-level `VASPen/` folder; verifies the key files exist
(hard fail). `python scripts/build.py --smoke` additionally launches
the exe, waits (≤90 s) for the frozen log
(`%LOCALAPPDATA%\VASPen\vaspen.log`) to grow with the startup line,
then `taskkill /T /F` — it aborts if a VASPen.exe was already running
(never kills a live session).

### 10.3 Nuitka (optimized build, release use)

```bash
python -m nuitka --standalone --windows-console-mode=disable \
    --enable-plugin=pyside6 \
    --windows-icon-from-ico=vaspen/resources/icons/app.ico \
    --include-data-dir=vaspen/resources=vaspen/resources \
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
- After adding/editing user-visible strings, hand-edit `vaspen_zh.ts`
  (new `<message>` with source + zh translation), then run
  `python scripts/sync_en_ts.py` (regenerates the en mirror), compile
  with `pyside6-lrelease` via `.venv/Scripts/python.exe
  .venv/Scripts/pyside6-lrelease.exe`, and run
  `pytest tests/test_i18n.py`. Do NOT run `pyside6-lupdate` — broken
  on this machine (§11.1).

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
- [x] App icon (letterboxed full source art — never cropped; title bar / taskbar / exe; §7.8.2)
- [x] i18n (en/zh)
- [x] QSS theming (light/dark, Fusion + QSS, live switch; §7.9)
- [x] Welcome page with recent files (central stack; click selects, double-click/Enter opens — §7.10)
- [x] Drag-and-drop file opening (MainWindow-level, any URL, first file)

### v0.3 — Release (Sprint 5)
- [x] Test suite
- [x] PyInstaller Windows build
- [x] User documentation

### v1.0+ — Future
- [ ] Materials Project integration (structure search by formula / mp-id, download & open, property lookup) — security policy in §7.6
- [ ] Remote SSH server connection + job submission
- [ ] Job queue management
- [ ] Band structure / DOS plotting (post-processing)
- [ ] Plugin system (Python entry points)
- [ ] Nuitka compiled release
