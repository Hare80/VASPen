"""Unified "Generate All Input Files" dialog.

One window with tabs for POSCAR, INCAR, KPOINTS and POTCAR. The INCAR,
KPOINTS and POTCAR tabs embed the existing editor panels (single source
of truth shared with the standalone dialogs); the POSCAR tab renders
the current structure and, for the NEB task, switches to initial/final
structure selection with linear path interpolation.

Task coupling (settled 2026-08-14):
- band  → KPOINTS switches to line-mode, pre-filled with a
  lattice-aware high-symmetry path suggestion (fallback: the default
  path);
- neb   → the POSCAR tab switches to NEB mode; KPOINTS switches to the
  automatic mesh; interpolated images are written as 00/POSCAR…
  0N/POSCAR subdirectories and IMAGES is filled into the INCAR tab;
- other → POSCAR tab in normal mode, KPOINTS automatic (custom leaves
  everything to the user).

Files are written only when Generate is clicked.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from ase import Atoms
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.file_io import FileIO
from vaspen.core.bonds import Bond, find_bonds
from vaspen.core.neb import (
    constraints_to_fixed_flags,
    detect_order_mismatch,
    interpolate_idpp,
    interpolate_neb,
    neb_distance,
    suggest_n_images,
)
from vaspen.core.structure import wrap_in_padded_cell
from vaspen.core.vasp_input import (
    generate_poscar,
    parse_incar_content,
    suggest_band_path,
)
from vaspen.ui.incar_editor import IncarEditorPanel
from vaspen.ui.kpoints_editor import KpointsEditorPanel
from vaspen.ui.menu_button import MenuButton
from vaspen.ui.periodic_wrap_dialog import PeriodicWrapDialog
from vaspen.ui.potcar_dialog import PotcarPanel
from vaspen.utils.config import AppConfig

# Task keys, shared with the INCAR preset keys (index-aligned with the
# task selector).
TASKS = ["scf", "opt", "band", "dos", "optical", "neb", "custom"]

_POSCAR_FILTER = (
    "VASP POSCAR (POSCAR CONTCAR *.vasp *.poscar *.contcar);;All files (*)"
)


class PoscarPanel(QWidget):
    """POSCAR tab: structure preview, or NEB interpolation for NEB tasks."""

    #: Emitted with the number of intermediate images after interpolation.
    images_generated = Signal(int)

    def __init__(self, structure_model=None, preview_callback=None, parent=None) -> None:
        super().__init__(parent)
        self._structure_model = structure_model
        self._preview_callback = preview_callback
        self._config = AppConfig()
        self._init_atoms: Atoms | None = None
        self._final_atoms: Atoms | None = None
        self._images: list[Atoms] = []
        # Per-atom per-direction fixed flags of the INITIAL structure:
        # fully-fixed atoms stay put during interpolation; all flags
        # are written into every image's Selective dynamics block.
        self._ini_fixed_flags = np.zeros((0, 3), dtype=bool)
        # Per-frame edit state — middle frames are editable in the 3D
        # viewport (move/rotate atoms, delete/add bonds; endpoints are
        # locked). Bonds persist per frame (auto-detected minus the
        # user-deleted pairs); history snapshots back Ctrl+Z per frame.
        self._frame_bonds: dict[int, list[Bond]] = {}
        self._frame_deleted: dict[int, set[tuple[int, int]]] = {}
        self._frame_history: dict[int, list[tuple]] = {}
        self._frame_redo: dict[int, list[tuple]] = {}
        self._has_frame_edits = False

        if structure_model is not None and structure_model.n_atoms:
            self._init_atoms = structure_model.atoms.copy()
            self._init_atoms.info = {}
            self._ini_fixed_flags = np.asarray(
                structure_model.fixed_flags, dtype=bool).copy()

        self._build_ui()
        self._refresh_poscar_preview()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_normal_page())
        self._stack.addWidget(self._build_neb_page())
        layout.addWidget(self._stack)

    def _build_normal_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        coord_row = QHBoxLayout()
        coord_row.addWidget(QLabel(self.tr("Coordinates:")))
        self._coord_combo = MenuButton()
        self._coord_combo.addItems([
            self.tr("Direct (fractional)"),
            self.tr("Cartesian"),
        ])
        direct_index = 0 if self._config.poscar_coords_direct else 1
        self._coord_combo.setCurrentIndex(direct_index)
        self._coord_combo.currentIndexChanged.connect(self._refresh_poscar_preview)
        coord_row.addWidget(self._coord_combo)
        coord_row.addStretch()
        layout.addLayout(coord_row)

        layout.addWidget(QLabel(self.tr("Preview:")))
        self._poscar_preview = QTextEdit()
        self._poscar_preview.setReadOnly(True)
        self._poscar_preview.setFontFamily("Consolas, monospace")
        layout.addWidget(self._poscar_preview, 1)
        return page

    def _build_neb_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        # Initial structure
        ini_row = QHBoxLayout()
        ini_row.addWidget(QLabel(self.tr("Initial structure:")))
        self._ini_edit = QLineEdit(
            self.tr("(current structure)") if self._init_atoms is not None else "")
        self._ini_edit.setReadOnly(True)
        ini_row.addWidget(self._ini_edit, 1)
        self._ini_browse = QPushButton(self.tr("Browse..."))
        self._ini_browse.clicked.connect(self._on_browse_ini)
        ini_row.addWidget(self._ini_browse)
        layout.addLayout(ini_row)

        # Final structure
        fin_row = QHBoxLayout()
        fin_row.addWidget(QLabel(self.tr("Final structure:")))
        self._fin_edit = QLineEdit("")
        self._fin_edit.setReadOnly(True)
        fin_row.addWidget(self._fin_edit, 1)
        self._fin_browse = QPushButton(self.tr("Browse..."))
        self._fin_browse.clicked.connect(self._on_browse_fin)
        fin_row.addWidget(self._fin_browse)
        layout.addLayout(fin_row)

        # Diagnostics
        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)
        self._warn_label = QLabel("")
        self._warn_label.setWordWrap(True)
        self._warn_label.setStyleSheet("color: #b06000;")
        layout.addWidget(self._warn_label)

        # Interpolation algorithm (linear is the default; IDPP avoids
        # atoms passing through each other on the straight-line path)
        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel(self.tr("Algorithm:")))
        self._algo_combo = MenuButton()
        self._algo_combo.addItems([self.tr("Linear"), self.tr("IDPP")])
        algo_row.addWidget(self._algo_combo)
        algo_row.addStretch()
        layout.addLayout(algo_row)

        # Interpolation controls
        interp_row = QHBoxLayout()
        interp_row.addWidget(QLabel(self.tr("Intermediate images:")))
        self._images_spin = QSpinBox()
        self._images_spin.setRange(1, 98)
        self._images_spin.setValue(1)
        interp_row.addWidget(self._images_spin)
        self._interpolate_btn = QPushButton(self.tr("Interpolate"))
        self._interpolate_btn.clicked.connect(self._on_interpolate)
        interp_row.addWidget(self._interpolate_btn)
        interp_row.addStretch()
        layout.addLayout(interp_row)

        # Image list (click to preview in the 3D viewport)
        layout.addWidget(QLabel(self.tr("Image frames (click to preview):")))
        self._images_list = QListWidget()
        self._images_list.currentItemChanged.connect(self._on_image_selected)
        layout.addWidget(self._images_list, 1)
        return page

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def neb_mode(self) -> bool:
        """True when the NEB page is active."""
        return self._stack.currentIndex() == 1

    def set_neb_mode(self, enabled: bool) -> None:
        self._stack.setCurrentIndex(1 if enabled else 0)

    @property
    def poscar_direct(self) -> bool:
        """Normal-mode coordinate style (True = Direct)."""
        return self._coord_combo.currentIndex() == 0

    @property
    def images(self) -> list[Atoms]:
        """Interpolated NEB frames (empty before interpolation)."""
        return self._images

    @property
    def n_images(self) -> int:
        """Number of intermediate images (0 when none generated)."""
        return max(0, len(self._images) - 2)

    # ------------------------------------------------------------------
    # Frame editing (middle images only)
    # ------------------------------------------------------------------

    def frame_preview_data(self, index: int):
        """Preview data for the frame at ``index``.

        Returns (atoms, editable, fixed_flags, bonds). ``editable`` is
        True only for intermediate frames — the initial and final
        frames are locked. Bonds are auto-detected on first access
        (minus user-deleted pairs) and cached per frame.
        """
        atoms = self._images[index]
        editable = 0 < index < len(self._images) - 1
        return atoms, editable, self._ini_fixed_flags, self._frame_bonds_for(index)

    def has_frame_edits(self) -> bool:
        """True when any frame was edited manually since interpolation."""
        return self._has_frame_edits

    def _frame_bonds_for(self, index: int) -> list[Bond]:
        if index not in self._frame_bonds:
            self._frame_bonds[index] = self._detect_frame_bonds(index)
        return self._frame_bonds[index]

    def _detect_frame_bonds(self, index: int) -> list[Bond]:
        atoms = self._images[index]
        deleted = self._frame_deleted.get(index, set())
        bonds = find_bonds(
            atoms.positions,
            list(atoms.get_chemical_symbols()),
            atoms.get_cell().array,
            tuple(atoms.get_pbc()),
        )
        return [b for b in bonds if (b.i, b.j) not in deleted]

    def _frame_snapshot(self, index: int) -> tuple:
        return (
            self._images[index].positions.copy(),
            list(self._frame_bonds.get(index, [])),
            set(self._frame_deleted.get(index, set())),
        )

    def _push_frame_history(self, index: int) -> None:
        self._frame_history.setdefault(index, []).append(
            self._frame_snapshot(index))
        if len(self._frame_history[index]) > 20:
            self._frame_history[index].pop(0)
        self._frame_redo[index] = []
        self._has_frame_edits = True

    def _restore_frame_snapshot(self, index: int, snapshot: tuple) -> None:
        positions, bonds, deleted = snapshot
        self._images[index].positions[:] = positions
        self._frame_bonds[index] = list(bonds)
        self._frame_deleted[index] = set(deleted)

    def apply_frame_move(self, index: int, indices: list[int],
                         positions) -> None:
        """Move atoms of a middle frame (frozen atoms are guarded by
        the viewport tools; the caller checks editability)."""
        self._push_frame_history(index)
        self._images[index].positions[indices] = np.asarray(
            positions, dtype=float)
        # connectivity changed — re-detect, keeping user-deleted pairs
        self._frame_bonds[index] = self._detect_frame_bonds(index)

    def remove_frame_bond(self, index: int, bond_list_index: int) -> bool:
        """Delete a bond of a middle frame (by bond-list index)."""
        bonds = self._frame_bonds_for(index)
        if not (0 <= bond_list_index < len(bonds)):
            return False
        self._push_frame_history(index)
        removed = bonds.pop(bond_list_index)
        self._frame_deleted.setdefault(index, set()).add(
            (removed.i, removed.j))
        return True

    def add_frame_bond(self, index: int, i: int, j: int) -> None:
        """Manually add a bond to a middle frame (un-deletes the pair)."""
        self._frame_bonds_for(index)
        pair = (min(i, j), max(i, j))
        if any((b.i, b.j) == pair for b in self._frame_bonds[index]):
            return  # already present
        self._push_frame_history(index)
        self._frame_bonds[index].append(Bond(i, j, order=1))
        self._frame_deleted.setdefault(index, set()).discard(pair)

    def frame_undo(self, index: int) -> bool:
        history = self._frame_history.get(index, [])
        if not history:
            return False
        self._frame_redo.setdefault(index, []).append(
            self._frame_snapshot(index))
        self._restore_frame_snapshot(index, history.pop())
        return True

    def frame_redo(self, index: int) -> bool:
        redo = self._frame_redo.get(index, [])
        if not redo:
            return False
        self._frame_history.setdefault(index, []).append(
            self._frame_snapshot(index))
        self._restore_frame_snapshot(index, redo.pop())
        return True

    # ------------------------------------------------------------------
    # Normal mode
    # ------------------------------------------------------------------

    def _refresh_poscar_preview(self) -> None:
        if self._structure_model is None or self._structure_model.n_atoms == 0:
            self._poscar_preview.setPlainText(
                self.tr("# No structure loaded."))
            return
        try:
            content = generate_poscar(
                self._structure_model, poscar_direct=self.poscar_direct)
        except Exception as e:
            content = self.tr("# {}").format(e)
        self._poscar_preview.setPlainText(content)

    # ------------------------------------------------------------------
    # NEB mode
    # ------------------------------------------------------------------

    def _on_browse_ini(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Initial Structure (POSCAR)"),
            self._config.last_directory,
            _POSCAR_FILTER,
        )
        if not filepath:
            return
        try:
            atoms = FileIO.read(filepath)
        except Exception as e:
            QMessageBox.warning(
                self, self.tr("Open Failed"),
                self.tr("Could not open file:\n{}").format(str(e)))
            return
        atoms = self._ensure_periodic_copy(atoms)
        if atoms is None:
            return
        self._init_atoms = atoms
        self._ini_fixed_flags = constraints_to_fixed_flags(atoms)
        self._ini_edit.setText(filepath)
        self._config.last_directory = str(Path(filepath).parent)
        self._clear_images()
        self._update_diagnostics()

    def _on_browse_fin(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Final Structure (POSCAR)"),
            self._config.last_directory,
            _POSCAR_FILTER,
        )
        if not filepath:
            return
        try:
            atoms = FileIO.read(filepath)
        except Exception as e:
            QMessageBox.warning(
                self, self.tr("Open Failed"),
                self.tr("Could not open file:\n{}").format(str(e)))
            return
        atoms = self._ensure_periodic_copy(atoms)
        if atoms is None:
            return
        self._final_atoms = atoms
        self._fin_edit.setText(filepath)
        self._config.last_directory = str(Path(filepath).parent)
        self._clear_images()
        self._update_diagnostics()

    def _ensure_periodic_copy(self, atoms: Atoms) -> Atoms | None:
        """Wrap a molecule in a padded cell (on a copy, model untouched).

        NEB runs need a periodic box on both endpoints. Slabs and other
        periodic structures pass through unchanged.
        """
        if atoms.get_cell().rank == 3 and atoms.pbc.any():
            return atoms
        dlg = PeriodicWrapDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return wrap_in_padded_cell(atoms, dlg.padding)

    def _update_diagnostics(self) -> None:
        """Recompute path distance + suggested image count for the pair."""
        self._info_label.setText("")
        self._warn_label.setText("")
        if self._init_atoms is None or self._final_atoms is None:
            return
        try:
            distance = neb_distance(self._init_atoms, self._final_atoms)
        except ValueError as e:
            self._warn_label.setText(str(e))
            return
        suggested = suggest_n_images(distance)
        self._images_spin.setValue(min(max(suggested, 1), 98))
        text = self.tr("Path distance: {:.4f} Å\n"
                       "Suggested images: {} (≈ 0.8 Å per image)")
        text = text.format(distance, suggested)
        n_frozen = int(self._ini_fixed_flags.all(axis=1).sum())
        if n_frozen:
            text += "\n" + self.tr(
                "{} atoms frozen — kept fixed in all images").format(n_frozen)
        self._info_label.setText(text)
        if suggested > 25:
            self._warn_label.setText(
                self.tr("The path is long ({} images suggested). Check that "
                        "the initial and final structures form a reasonable "
                        "path.").format(suggested))

    def _reset_frame_state(self) -> None:
        self._frame_bonds.clear()
        self._frame_deleted.clear()
        self._frame_history.clear()
        self._frame_redo.clear()
        self._has_frame_edits = False

    def _clear_images(self) -> None:
        self._images = []
        self._reset_frame_state()
        self._images_list.clear()

    def _on_interpolate(self) -> None:
        if self._init_atoms is None:
            QMessageBox.warning(
                self, self.tr("No Initial Structure"),
                self.tr("Load the initial structure first."))
            return
        if self._final_atoms is None:
            QMessageBox.warning(
                self, self.tr("No Final Structure"),
                self.tr("Load the final structure first."))
            return
        try:
            distance = neb_distance(self._init_atoms, self._final_atoms)
        except ValueError as e:
            QMessageBox.warning(
                self, self.tr("Cannot Interpolate"), str(e))
            return

        # Order mismatch is only a warning: the user may continue with
        # the strict file order (settled 2026-08-14).
        mismatch = detect_order_mismatch(self._init_atoms, self._final_atoms)
        if mismatch is not None:
            reply = QMessageBox.question(
                self,
                self.tr("Atom Order Mismatch"),
                self.tr("The initial and final structures may be in "
                        "different atom orders.\n\n"
                        "File-order path length: {:.4f} Å\n"
                        "After reordering atoms of the same element: "
                        "{:.4f} Å\n\n"
                        "Interpolate in strict file order anyway?")
                .format(distance, mismatch),
            )
            if reply != QMessageBox.Yes:
                return

        # Regenerating the images discards the user's manual frame
        # edits — confirm first.
        if self._has_frame_edits:
            reply = QMessageBox.question(
                self,
                self.tr("Discard Frame Edits"),
                self.tr("Regenerating the images will discard your "
                        "manual edits to the frames. Continue?"),
            )
            if reply != QMessageBox.Yes:
                return

        n = self._images_spin.value()
        # Fully-frozen atoms stay at their initial position in every
        # frame; partially-frozen atoms interpolate normally (their
        # FixScaled flags are enforced by VASP at run time).
        frozen_mask = self._ini_fixed_flags.all(axis=1)
        try:
            if self._algo_combo.currentIndex() == 1:  # IDPP
                self._images = interpolate_idpp(
                    self._init_atoms, self._final_atoms, n,
                    frozen_mask=frozen_mask)
            else:  # Linear (default)
                self._images = interpolate_neb(
                    self._init_atoms, self._final_atoms, n,
                    frozen_mask=frozen_mask)
        except ValueError as e:
            QMessageBox.warning(
                self, self.tr("Cannot Interpolate"), str(e))
            return
        self._reset_frame_state()  # fresh images, no stale frame state
        self._images_list.clear()
        for i, _atoms in enumerate(self._images):
            if i == 0:
                label = self.tr("00 — initial")
            elif i == len(self._images) - 1:
                label = self.tr("{:02d} — final").format(i)
            else:
                label = f"{i:02d}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, i)
            self._images_list.addItem(item)
        self.images_generated.emit(n)

    def _on_image_selected(self, current: QListWidgetItem | None,
                           _previous: QListWidgetItem | None) -> None:
        if current is None or self._preview_callback is None:
            return
        index = current.data(Qt.UserRole)
        if index is not None and 0 <= index < len(self._images):
            atoms, editable, flags, bonds = self.frame_preview_data(index)
            self._preview_callback(atoms, index, editable, flags, bonds, self)


class GenerateAllDialog(QDialog):
    """Unified preview/adjust/generate window for the four VASP inputs."""

    def __init__(self, structure_model, parent=None,
                 preview_callback=None, default_task: str = "scf") -> None:
        super().__init__(parent)
        self._structure_model = structure_model
        self._preview_callback = preview_callback
        self._config = AppConfig()
        self._syncing = False

        self.setWindowTitle(self.tr("Generate All Input Files"))
        self.resize(900, 700)
        self._build_ui()

        if default_task not in TASKS:
            default_task = "scf"
        self._sync_task(default_task, source="init")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Task selector ──
        task_row = QHBoxLayout()
        task_row.addWidget(QLabel(self.tr("Calculation Type:")))
        self._task_combo = MenuButton()
        self._task_combo.addItems([
            self.tr("SCF (Static)"),
            self.tr("Optimization"),
            self.tr("Band Structure"),
            self.tr("DOS"),
            self.tr("Optical"),
            self.tr("NEB"),
            self.tr("Custom"),
        ])
        self._task_combo.currentIndexChanged.connect(self._on_task_changed)
        task_row.addWidget(self._task_combo)
        task_row.addStretch()
        layout.addLayout(task_row)

        # ── Tabs ──
        self._tabs = QTabWidget()
        self._poscar_panel = PoscarPanel(
            self._structure_model, preview_callback=self._preview_callback)
        self._incar_panel = IncarEditorPanel(self._structure_model)
        self._kpoints_panel = KpointsEditorPanel(self._structure_model)
        self._potcar_panel = PotcarPanel(self._structure_model)
        self._tabs.addTab(self._poscar_panel, self.tr("POSCAR"))
        self._tabs.addTab(self._incar_panel, self.tr("INCAR"))
        self._tabs.addTab(self._kpoints_panel, self.tr("KPOINTS"))
        self._tabs.addTab(self._potcar_panel, self.tr("POTCAR"))
        layout.addWidget(self._tabs, 1)

        self._incar_panel.preset_changed.connect(self._on_incar_preset_changed)
        self._poscar_panel.images_generated.connect(self._on_images_generated)

        # ── Output directory ──
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel(self.tr("Output directory:")))
        self._dir_edit = QLineEdit(self._config.last_directory)
        dir_row.addWidget(self._dir_edit, 1)
        browse_btn = QPushButton(self.tr("Browse..."))
        browse_btn.clicked.connect(self._browse_output_dir)
        dir_row.addWidget(browse_btn)
        layout.addLayout(dir_row)

        # ── Buttons ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._generate_btn = QPushButton(self.tr("Generate"))
        self._generate_btn.clicked.connect(self._on_generate)
        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._generate_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Task coupling
    # ------------------------------------------------------------------

    def _on_task_changed(self, index: int) -> None:
        self._sync_task(TASKS[index], source="task")

    def _on_incar_preset_changed(self, key: str) -> None:
        # The user changed the preset inside the INCAR tab — the task
        # selector follows and the per-task rules re-apply.
        self._sync_task(key, source="incar")

    def _sync_task(self, key: str, source: str) -> None:
        if self._syncing or key not in TASKS:
            return
        self._syncing = True
        try:
            # re-entrant signals are blocked by the _syncing guard
            if source in ("task", "init"):
                self._task_combo.setCurrentIndex(TASKS.index(key))
                self._incar_panel.set_preset(key)
            elif source == "incar":
                self._task_combo.setCurrentIndex(TASKS.index(key))
            self._apply_task_rules(key)
        finally:
            self._syncing = False

    def _apply_task_rules(self, key: str) -> None:
        """Per-task defaults (the user can override them afterwards)."""
        if key == "band":
            self._poscar_panel.set_neb_mode(False)
            self._kpoints_panel.set_mode("line")
            suggestion = None
            if self._structure_model is not None and self._structure_model.n_atoms:
                suggestion = suggest_band_path(self._structure_model.atoms)
            if suggestion is not None:
                path, points = suggestion
                text = "|".join(f"{a}-{b}" for a, b in path)
                self._kpoints_panel.set_special_points_override(points)
                self._kpoints_panel.set_band_path_text(text)
            else:
                # keep the ASE default path (G-X|X-M|M-G)
                self._kpoints_panel.set_special_points_override(None)
        elif key == "neb":
            self._poscar_panel.set_neb_mode(True)
            self._kpoints_panel.set_mode("automatic")
            # re-fill IMAGES when images were generated earlier and the
            # user switched away and back (preset reload empties it)
            if self._poscar_panel.n_images:
                self._incar_panel.apply_tag(
                    "IMAGES", str(self._poscar_panel.n_images))
        elif key != "custom":
            self._poscar_panel.set_neb_mode(False)
            self._kpoints_panel.set_mode("automatic")

    def _on_images_generated(self, n: int) -> None:
        """Fill the interpolated image count into the INCAR tab."""
        self._incar_panel.apply_tag("IMAGES", str(n))

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    def _browse_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            self.tr("Choose Output Directory"),
            self._dir_edit.text() or self._config.last_directory,
        )
        if directory:
            self._dir_edit.setText(directory)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def _render_poscar_text(self, atoms: Atoms,
                            fixed_flags: np.ndarray | None = None) -> str:
        """Render a NEB frame as POSCAR text (with Selective dynamics)."""
        from ase.io import write as ase_write

        from vaspen.core.file_io import (
            atoms_with_fixed_constraints,
            vasp_write_atoms,
        )

        frame = atoms.copy()
        frame.constraints = []  # flags are the single source of truth
        out = atoms_with_fixed_constraints(
            vasp_write_atoms(frame, True), fixed_flags)
        buf = io.StringIO()
        ase_write(buf, out, format="vasp", vasp5=True, direct=True)
        return buf.getvalue()

    def _on_generate(self) -> None:
        out_text = self._dir_edit.text().strip()
        if not out_text:
            QMessageBox.warning(
                self, self.tr("No Output Directory"),
                self.tr("Choose an output directory first."))
            return
        if self._structure_model is None or self._structure_model.n_atoms == 0:
            QMessageBox.warning(
                self, self.tr("No Structure"),
                self.tr("Load a structure first."))
            return
        # The dialog is non-modal, so the model can change underneath it
        # (drag-and-drop open). POSCAR + KPOINTS need a periodic cell —
        # the NEB path validates its own initial/final pair.
        if not self._poscar_panel.neb_mode and not self._structure_model.is_periodic:
            QMessageBox.warning(
                self, self.tr("Not Periodic"),
                self.tr("The structure is not periodic. Close the dialog, "
                        "wrap the structure in a periodic cell and try "
                        "again."))
            return

        try:
            incar = self._incar_panel.content()
        except ValueError as e:
            QMessageBox.warning(self, self.tr("Invalid INCAR"), str(e))
            return
        try:
            kpoints = self._kpoints_panel.content()
        except ValueError as e:
            QMessageBox.warning(self, self.tr("Invalid KPOINTS"), str(e))
            return
        try:
            potcar = self._potcar_panel.generate_content()
        except FileNotFoundError as e:
            QMessageBox.critical(self, self.tr("POTCAR Not Found"), str(e))
            return

        files: dict[str, str] = {"INCAR": incar, "KPOINTS": kpoints}
        if self._poscar_panel.neb_mode:
            images = self._poscar_panel.images
            if not images:
                QMessageBox.warning(
                    self, self.tr("No NEB Images"),
                    self.tr("Generate the NEB images in the POSCAR tab first."))
                return
            parsed, _problems = parse_incar_content(incar)
            im_value = parsed.get("IMAGES", "").strip()
            n_images = len(images) - 2
            if im_value.isdigit() and int(im_value) != n_images:
                reply = QMessageBox.question(
                    self,
                    self.tr("IMAGES Mismatch"),
                    self.tr("INCAR IMAGES ({}) does not match the number of "
                            "interpolated images ({}).\n\nContinue?")
                    .format(im_value, n_images),
                )
                if reply != QMessageBox.Yes:
                    return
            for i, atoms in enumerate(images):
                files[f"{i:02d}/POSCAR"] = self._render_poscar_text(
                    atoms, self._poscar_panel._ini_fixed_flags)
        else:
            files["POSCAR"] = generate_poscar(
                self._structure_model, poscar_direct=self._poscar_panel.poscar_direct)

        if potcar:
            files["POTCAR"] = potcar

        out = Path(out_text)
        try:
            for name, content in files.items():
                target = out / name
                target.parent.mkdir(parents=True, exist_ok=True)
                # UTF-8 + LF: VASP input files must not carry the
                # locale encoding (GBK) or CRLF line endings
                target.write_text(content, encoding="utf-8", newline="\n")
        except OSError as e:
            QMessageBox.critical(
                self, self.tr("Generation Failed"), str(e))
            return

        self._config.last_directory = str(out)
        QMessageBox.information(
            self,
            self.tr("Success"),
            self.tr("Generated files in:\n{}\n\nFiles: {}").format(
                out_text, ", ".join(files)),
        )
        self.accept()
