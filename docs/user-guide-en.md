# VASPen User Guide

## 1. Introduction

VASPen is a cross-platform desktop GUI for [VASP](https://www.vasp.at)
first-principles calculations. It provides visual 3D structure modeling,
surface/slab building, and generation of the four VASP input files
(POSCAR, INCAR, KPOINTS, POTCAR), plus a NEB (nudged elastic band)
workflow — everything you need between "I have a crystal structure"
and "I can submit a job".

Version 0.3.0 is the first packaged Windows release. VASPen is free
and open source (MIT License); the source code lives at
<https://github.com/Hare80/VASPen>.

## 2. System Requirements

**Windows release (recommended):**

- Windows 10 or 11, 64-bit
- ~1 GB of free disk space (the unpacked bundle)
- A graphics driver supporting OpenGL 3.3 (any driver from the last
  decade)
- No Python installation required — the bundle is self-contained

**Running from source** (developers, Linux users):

- Python ≥ 3.11
- PySide6 ≥ 6.6, ASE ≥ 3.29, pymatgen ≥ 2024.1, spglib ≥ 2.3,
  NumPy ≥ 1.26, SciPy ≥ 1.12
- See the repository `README.md` for instructions

## 3. Download & Run

1. Go to <https://github.com/Hare80/VASPen/releases> and download
   `VASPen-v0.3.0-win64.zip`.
2. Unzip it anywhere you like. You get one folder, `VASPen/`,
   containing `VASPen.exe` and an `_internal/` folder.
   **Keep `VASPen.exe` and `_internal/` together** — the program will
   not start without it. Moving the exe alone breaks the installation.
3. Double-click `VASPen.exe`.

Notes:

- No installer, no registry dependencies — "portable" software.
  Updating means replacing the whole folder.
- Your settings (language, theme, recent files) are stored outside the
  program folder, so they survive updates.
- The exe is not code-signed. On first run Windows SmartScreen may
  show a warning: click **More info → Run anyway**. Only run downloads
  from the official GitHub repository.
- Linux: run from source (see §2); the release zip is Windows-only.

## 4. Quick Start

1. **Set up your pseudopotential library** (needed for POTCAR
   generation only): Edit → Preferences → *Pseudopotential Library* →
   set the library root (see §13.3).
2. **Open a structure**: File → Open, or drag a structure file onto
   the window. Structure files ship with the source repository in the
   `examples/` folder.
3. **Explore it in 3D**: left-drag to rotate, right-drag to pan, wheel
   to zoom.
4. **Build a slab**: Tools → Cleave Surface → choose Miller indices,
   layers and vacuum → OK (see §8).
5. **Generate inputs**: Calculate → Generate All Input Files → choose
   a task (e.g. SCF) → set an output directory → **Generate**. The
   four files are written to that directory (see §13).

## 5. Interface Tour

### Menus

| Menu | Contents |
|------|----------|
| **File** | New Structure (Ctrl+N), Open (Ctrl+O), Save (Ctrl+S), Save As (Ctrl+Shift+S), Export as POSCAR, Recent Files (10 entries, clearable), Quit |
| **Edit** | Undo (Ctrl+Z), Redo (Ctrl+Y / Ctrl+Shift+Z), Select All (Ctrl+A), Select None, Invert Selection, Select Neighbors, Select Connected, Freeze, Unfreeze, Bond Order (Single/Double/Triple/Aromatic), Auto Detect Bonds (off by default), Preferences |
| **View** | Reset View, Display Style (Ball & Stick / Space Filling (CPK) / Wireframe), Show Unit Cell, Atom Labels, Display Options, Language (English / 中文), dock toggles (Structure / Measurements / Properties) |
| **Calculate** | Generate INCAR, Generate KPOINTS, Generate POTCAR, Generate All Input Files |
| **Tools** | Cleave Surface, Supercell, Wrap in Periodic Cell, Re-box Slab, Edit Lattice, Find Symmetry |
| **Help** | About VASPen, About Qt |

**Reload on external changes:** while a file is loaded, VASPen watches
it on disk. If another program modifies it (a text editor, a script,
or the VASPen MCP server), the view reloads automatically — if you
have unsaved changes, VASPen asks first, because reloading discards
them.

### Toolbars

- **Main toolbar**: New, Open, Save, Generate All, Cleave Surface,
  Supercell, Reset View.
- **Edit toolbar**: Select, Add Atom (element shortlist + full
  periodic-table picker), Move, Rotate, Delete, Create Bond, Freeze,
  Unfreeze, Detect Bonds, Distance, Angle, Dihedral, view-direction
  presets (Front/Back/Left/Right/Top/Bottom), Undo, Redo.

### Docks & status bar

- **Structure tree** (left): atom list and cell parameters, kept in
  sync with the selection.
- **Measurements** (right): the distance/angle/dihedral list.
- **Properties** (right): per-atom element, position, fixed flag,
  magnetic moment (MAGMOM) and composition, plus charge/force/velocity
  display.
- **Status bar**: atom count and formula, cell parameters a/b/c/α/β/γ.

Any closed dock can be re-opened from the View menu.

### Welcome page

When no structure is loaded, a welcome page shows recent files and
buttons for New Structure / Open / Browse. **Click selects a recent
file; double-click (or Enter) opens it**; Open opens the selected
file; Browse opens the standard file dialog. You can also drag & drop
a file onto the window at any time.

## 6. 3D Viewport

| Mouse | Action |
|-------|--------|
| Left-drag | Rotate the camera |
| Right-drag / middle-drag | Pan |
| Wheel | Zoom — completely free: the camera may pass through atoms and the cell frame; atoms fade out as they get very close to the camera |
| Click on an atom | Select it |
| Ctrl/Shift + click | Add/remove from the selection |
| Shift + drag (in Select mode) | Box-select atoms |
| Esc | Cancel the active Move/Rotate operation |

Other notes:

- The projection is **orthographic** (no perspective distortion).
- **Default view on open**: periodic structures open with the c-axis
  on top (b up, a left); molecules open in a 45°/30° isometric view.
- **Reset View** (toolbar / View menu) re-fits the camera.
- In-place edits (supercell, cleave, add/delete atoms) preserve the
  camera.
- The View menu toggles the unit-cell frame, atom labels, and the
  display style; Display Options opens the render settings dialog
  (colors, background, atom radii) with live preview.
- Clicking a selected bond shows its length; the Measurements tools
  (Distance/Angle/Dihedral) add entries to the Measurements dock.

## 7. Structure Editing

Editing tools live in the Edit toolbar (or the Edit menu). Click a
tool to activate it; click Select (or Esc) to return to browsing.

| Tool | What it does |
|------|--------------|
| Select | Click / box-select atoms; selection operations (All, None, Invert, Neighbors, Connected) live in the Edit menu |
| Add Atom | Pick an element (shortlist or full periodic-table picker), click in the viewport to place the atom |
| Move | Drag a selected atom to a new position |
| Rotate | Rotate the selected atoms |
| Delete | Click an atom to delete it (the Delete key also works) |
| Create Bond | Click two atoms to add a bond; bond order (Single/Double/Triple/Aromatic) is set from the Edit menu |
| Freeze / Unfreeze | Toggle VASP selective-dynamics flags on the selected atoms (all three directions). Frozen atoms cannot be moved and are written into POSCAR with `F F F` flags |
| Detect Bonds | One-shot bond detection for the current structure |
| Auto Detect Bonds | (off by default) re-detects bonds after every edit; automatic detection is skipped for structures above 1000 atoms |

Notes:

- **Undo/Redo** (Ctrl+Z / Ctrl+Y) covers structural edits, each
  taking one step.
- The **Properties** dock shows the selected atom's element, position
  (fractional for periodic structures), fixed flag and magnetic moment.
  Setting MAGMOM values turns on `ISPIN = 2` in the generated INCAR
  (see §13.1).
- **Edit Lattice** (Tools menu) edits cell parameters with a live
  preview; periodic structures only.
- **Supercell** (Tools menu) repeats the cell by (n₁, n₂, n₃).
- Structures with partial site occupancy (disorder) are supported;
  operations that discard disorder (e.g. symmetrize, cleave) ask for
  confirmation first.

## 8. Cleave a Surface

Tools → **Cleave Surface** builds a slab from a **bulk periodic**
structure (the dialog tells you so if the current structure does not
qualify and points to Re-box Slab for vacuum-bearing slabs).

1. Enter the **Miller indices** (hkl).
2. Choose the **number of layers** and the **vacuum thickness**.
3. The termination list computes in the background (the dialog is
   non-modal — you can keep rotating the view while it works; clicking
   a termination jumps it to the front of the queue).
4. Pick a termination (each is previewed in the 3D viewport) and
   click OK.

The resulting slab has an orthogonal box: the c-axis lies exactly
along the surface normal, the requested vacuum is exact, and the slab
is centered. Applied as one undo step; saved as a new file. If the
slab carries partial occupancy, a confirmation asks before discarding
the disorder.

## 9. Re-box Slab

Tools → **Re-box Slab** re-applies the vacuum along c for a structure
that **already carries vacuum** (e.g. a slab you edited, or one
imported with an inconvenient vacuum). The in-plane cell, the atom
order, the frozen flags and the magnetic moments are all preserved —
unlike Cleave, which builds a new atom set. Vacuum range 0.5–100 Å
(default 15 Å, with a "remember" checkbox). One undo step.

## 10. Symmetry

Tools → **Find Symmetry** analyzes the structure (space group or
molecular point group) and offers:

- **Symmetrize** — snap the structure to its ideal symmetry.
- **Conventional cell** — standardize to the conventional cell
  (the default choice).
- **Primitive cell** — reduce to the primitive cell in the standard
  orientation (a along x, b in the xy-plane).

Cell conversion is available for periodic structures only. Results
are applied as one undo step and saved as a new file.

## 11. Wrap in Periodic Cell

Molecules (no periodic cell) cannot be saved directly to VASP/CIF
formats. VASPen handles this automatically:

- **Saving** a molecule to a periodic format (or running Generate All
  Input Files) opens the Wrap dialog: choose the vacuum padding
  (0.5–50 Å, default 5 Å, with a "remember" checkbox). The molecule is
  centered in a cubic box with exactly the requested padding on every
  face.
- Tools → **Wrap in Periodic Cell** runs the same step explicitly.

The wrap is undoable and preserves the camera.

## 12. NEB Workflow

Calculate → Generate All Input Files → Calculation Type = **NEB**.

1. In the POSCAR tab, browse for the **initial** and **final**
   structures. The two must have equal atom counts and elements and
   identical cells (variable-cell NEB is rejected with a message).
2. The dialog shows the path length (Å). The suggested image count is
   `ceil(distance / 0.8 Å)`; you can set 1–98 images.
3. Choose the interpolation:
   - **Linear** (default) — per-atom interpolation in fractional
     coordinates along the minimal-image displacement.
   - **IDPP** — relaxes the images so the interatomic-distance change
     between adjacent images is uniform; atoms never pass through each
     other. This is the mode to prefer when the linear path would
     collide atoms (e.g. rotating groups).
4. Click **Interpolate**; the image list appears. Clicking a frame
   previews it in the 3D viewport.
5. **Frozen atoms** (selective-dynamics flags on the initial
   structure): fully-frozen atoms keep their initial position in
   *every* image, and every written POSCAR carries the flags. If a
   frozen atom sits at a different position in the final structure,
   interpolation is blocked with a message. Partially-frozen atoms
   interpolate normally (their flags are enforced by VASP at run
   time).
6. **Frame editing**: clicking a *middle* frame enters edit mode —
   you can move/rotate atoms and edit bonds of that frame (per-frame
   undo/redo). Initial and final frames are locked; atoms cannot be
   deleted from a frame. Clicking Interpolate again asks before
   discarding manual edits.
7. The INCAR tab is filled with the NEB preset, including `IMAGES`.
8. Generate writes `00/POSCAR` … `0N/POSCAR` subdirectories plus
   INCAR / KPOINTS / POTCAR in the chosen output directory — the
   standard VASP NEB layout.

See the `examples/neb_*` folders for ready-made endpoint pairs.

## 13. Generating VASP Inputs

Calculate → **Generate All Input Files** is the one-stop dialog: a
task selector at the top syncs the POSCAR/INCAR/KPOINTS/POTCAR tabs,
which you can still override afterwards. Files are written only when
you click **Generate**, into the output directory chosen in the
dialog. The standalone Generate INCAR / KPOINTS / POTCAR dialogs use
the same panels.

### 13.1 INCAR presets

| Tag | SCF | Optimization | Band | DOS | Optical |
|-----|-----|--------------|------|-----|---------|
| ENCUT | 400 | 400 | 400 | 400 | 400 |
| ISMEAR | 0 | 0 | 0 | −5 | 0 |
| SIGMA | 0.05 | 0.05 | 0.05 | 0.05 | 0.01 |
| EDIFF | 1E-6 | 1E-6 | 1E-6 | 1E-6 | 1E-6 |
| EDIFFG | — | −0.01 | — | — | — |
| IBRION | — | 2 | — | — | — |
| ISIF | — | 3 | — | — | — |
| NSW | — | 100 | — | — | — |
| ISPIN | 1 | 1 | 1 | 1 | 1 |
| LORBIT | — | — | — | 11 | — |
| NEDOS | — | — | — | 2001 | 2000 |
| LOPTICS | — | — | — | — | T |
| CSHIFT | — | — | — | — | 0.1 |

Band: `ICHARG = 11`, `LCHARG = T` — the band run reads the CHGCAR of
a preceding SCF run (the SCF preset writes `LCHARG = F`; switch it to
T there).

NEB preset (VTST-style): SCF base plus `EDIFFG = -0.02`,
`IBRION = 3`, `POTIM = 0`, `IOPT = 1` (LBFGS), `ICHAIN = 0`,
`NSW = 500`, `SPRING = -5`, `LCLIMB = .TRUE.`. `IMAGES` starts empty
and must be filled in (the NEB task fills it automatically; both
generation paths refuse empty required tags).

Hints:

- Raise `ENCUT` to 1.3 × the largest `ENMAX` in your POTCARs when
  required.
- Every generated line carries an aligned comment; suggested-but-inactive
  lines (`# MAGMOM`, `# IVDW = 11`, `# ISTART = 1`) follow after a blank
  line and apply only when uncommented.
- The preview panel is the same renderer that writes the file — what
  you see is what you get.

### 13.2 KPOINTS modes

| Mode | Key parameter | Default |
|------|---------------|---------|
| Automatic | spacing (2π/Å convention) | insulators 0.04, metals 0.03 (fine 0.02, coarse 0.05) |
| Manual mesh | n₁ n₂ n₃ | your values |
| Line-mode (band) | high-symmetry path, points per segment | 20 points/segment |

The automatic mode writes an explicit regular mesh (Γ-centered or
Monkhorst-Pack, shift 0 0 0) whose density corresponds to the chosen
spacing in units of 2π/Å. The band task automatically switches to
line-mode and pre-fills a lattice-appropriate high-symmetry path from
the structure's symmetry (fallback: `G-X|X-M|M-G`).

Non-periodic structures have no k-mesh: the automatic-mode preview
shows a note instead, and generation is blocked until the structure
is wrapped in a periodic cell (§11).

### 13.3 POTCAR

First configure the library root in Preferences. The expected layout
is the standard VASP pseudopotential library:

```
<library root>/
  potpaw_PBE.54/   (or PBE.54/ etc.)
    Fe/
      POTCAR
    O/
      POTCAR
```

The dialog offers a functional selector (**PBE** is the default;
PBE_new, LDA and PW91 are available) and pre-selects the
recommended variant per element (semi-core `_d` / `_sv` variants for
elements that need them); the alternatives are read from your library
itself. The selected POTCARs are concatenated in POSCAR element order.
The library is never bundled with VASPen — you supply your own
licensed VASP potentials.

### 13.4 POSCAR

Coordinates are written in **Fractional (Direct)** by default (the
Preferences setting switches to Cartesian). Fixed atoms carry
`Selective dynamics` lines with `F F F` / `T T T` flags.

## 14. Settings

Edit → **Preferences**:

| Setting | Meaning |
|---------|---------|
| Language | English / 中文 — switches live (also available from View → Language) |
| Theme | Light / Dark — switches live |
| Default Calculation Type | pre-fills the task selector of the generate dialogs |
| POSCAR Coordinates | Fractional (Direct) or Cartesian |
| Pseudopotential Library Root | path to your VASP library (see §13.3) |

Settings are persisted outside the program folder (the Windows
registry on Windows) and survive updates.

## 15. Supported Formats

Read & write: CIF, XYZ, EXTXYZ, POSCAR/CONTCAR (any letter case,
extensionless files recognized), XSF, PDB, JSON, CUBE, TRAJ, DB.

- Saving a **molecule** to VASP/CIF triggers the periodic-wrap dialog
  (§11).
- Saving a structure with **partial occupancy** to POSCAR asks for
  confirmation (POSCAR cannot represent disorder).
- CIF reading sanitizes unknown occupancy markers and unusable
  space-group headers — files exported by external modelers with an
  "unknown group" number still open correctly.

## 16. Examples

The source repository ships a curated `examples/` folder:

- **Structure files**: fcc Cu primitive (symmetry conversion), bcc Fe
  2×2×2 (spin INCAR), hcp Mg, diamond (band/DOS), NaCl rocksalt
  ((100) cleave, two-element POTCAR), GaAs zincblende (band path),
  ZnO wurtzite, TiO₂ rutile, BaTiO₃ perovskite (three-element POTCAR,
  polar terminations), MFI zeolite 288 atoms (large-cell rendering),
  Cu (111) slab, graphene 3×3 sheet, benzene (wrap flow), PZT
  disordered perovskite (occupancy from CIF).
- **NEB endpoint pairs**: ethane rotation (Linear vs IDPP contrast),
  Au(111) vacancy hop with a frozen slab, P–O–H adsorbate migration.

Each NEB folder has its own README with step-by-step instructions.

## 17. FAQ

**Windows SmartScreen / antivirus flags the exe.**
The release is not code-signed, so Windows warns on first run
(More info → Run anyway) and some antivirus products may be
suspicious of unsigned downloads. Always download from the official
GitHub repository and verify the SHA-256 checksum published in the
release notes.

**The first launch is slow.**
One-time cost: Windows/antivirus scans the large bundle. Later
launches are fast.

**Where do my generated files go?**
Into the output directory you chose in the dialog — nothing is
written anywhere until you click Generate.

**"Pseudopotential library not found."**
Set the library root in Edit → Preferences → Pseudopotential Library
(§13.3).

**The INCAR tab complains about a tag without a value (NEB).**
`IMAGES` must be filled in. With the NEB task this happens
automatically after interpolation.

**I cannot move an atom.**
It is frozen (selective dynamics). Unfreeze it first (Edit →
Unfreeze or the Unfreeze toolbar button).

**The welcome page is showing.**
No structure is loaded. Open a file or press New Structure to enter
the empty workspace.

**The 3D view is black / broken.**
Update your graphics driver; VASPen needs OpenGL 3.3.

**The window title shows an unexpected language.**
View → Language switches between English and 中文 live.

**How do I update?**
Download the new zip and replace the whole `VASPen/` folder. Your
settings live outside the folder and are kept.

**I need to report a problem.**
Attach the log file `%LOCALAPPDATA%\VASPen\vaspen.log` (Windows
release builds only).

**Linux?**
Run from source (see §2). The release zip is Windows-only.

## 18. MCP Server (for AI clients)

VASPen includes a headless [Model Context Protocol](https://modelcontextprotocol.io)
(MCP) server that lets AI assistants — Claude Desktop, ZCode, Cursor and
other MCP clients — drive the toolkit directly: load and analyze
structures, cut surfaces, build supercells, prepare NEB runs and
generate complete VASP input sets, all from a chat prompt.

### 18.1 Installation

- **From source** (Windows & Linux): `pip install -e ".[mcp]"`, then the
  command `vaspen-mcp` is available (`vaspen-mcp --version` to check).
- **From the release zip** (Windows): use
  `VASPen/vaspen-mcp/vaspen-mcp.exe` from a zip built with `--with-mcp`.

The server is a plain stdio process: your AI client launches it on
demand and talks JSON-RPC over stdin/stdout — nothing listens on the
network.

### 18.2 Client configuration

Claude Desktop (`claude_desktop_config.json`) or the ZCode / Cursor MCP
settings:

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

For a source install, `"command": "vaspen-mcp"` usually suffices.

### 18.3 What the AI can do

| Group | Tools |
|-------|-------|
| Structures | open_structure, get_structure_info, list_atoms, save_structure, make_periodic |
| Editing | make_supercell, set_fixed_atoms, set_magmoms, translate_atoms, rotate_atoms |
| Analysis | measure (distance/angle/dihedral), find_bonds, analyze_symmetry, suggest_band_path, estimate_k_mesh |
| Symmetry & surfaces | symmetrize_cell, list_slab_terminations, cut_surface, rebox_slab |
| VASP inputs | list_incar_presets, generate_inputs |
| NEB | neb_check, neb_setup |
| Visualization | render_preview — a ball-and-stick image the AI can look at |
| Escape hatch | run_python — execute Python against the loaded structure (Settings-gated) |

A typical prompt: *"Open TiO2_rutile.vasp, tell me the space group, then
generate an SCF input set with a 0.03 k-spacing into ./vasp-run"* — the
assistant calls the tools itself and reports the results.

### 18.4 Notes

- Tool responses are in English regardless of the GUI language.
- Atom indices are 0-based; lengths are Å, angles in degrees.
- The pseudopotential library path configured in **Settings →
  Pseudopotential Library** is shared between the GUI and the MCP
  server (the library itself is never bundled).
- Guarded operations mirror the GUI's confirmation dialogs: the AI must
  pass explicit confirm flags for discarding partial occupancies
  (`allow_disorder_loss`) or continuing despite a NEB atom-order
  warning (`force`), so nothing destructive happens silently.
- **Live mode** — While VASPen is open, MCP tools act on the **window's structure** in real time (Settings → MCP Server → Live bridge, on by default): what the AI cuts, supercells or saves, you see rendered immediately. With VASPen closed, the same server works headlessly on files.
- When the server writes to a file that is currently open in the
  VASPen GUI, the GUI notices and reloads (asking first if you have
  unsaved changes) — see the reload note in the Menus section.

## 19. License & Credits

VASPen is free and open source, released under the MIT License
(`LICENSE` in the repository).

Built with [PySide6](https://www.qt.io) (Qt, LGPL) for the GUI,
[ASE](https://wiki.fysik.dtu.dk/ase) and [pymatgen](https://pymatgen.org)
for structure handling, and [spglib](https://spglib.github.io/spglib/)
for symmetry analysis. 3D rendering is native Qt OpenGL.

VASPen does **not** include VASP itself or any pseudopotential
library — users supply their own licensed VASP installation and PAW
potentials. Please cite VASP (and any structure database you used) in
your publications as required by their licenses.
