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
from PySide6.QtCore import QEvent, Qt, Signal
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
    QRadioButton,
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

    def __init__(self, structure_model=None, preview_callback=None, parent=None,
                 project_provider=None) -> None:
        super().__init__(parent)
        self._structure_model = structure_model
        self._preview_callback = preview_callback
        #: Callable returning the open Project (or None) — enables the
        #: "From Project..." endpoint pickers (NEB page).
        self._project_provider = project_provider
        self._config = AppConfig()
        self._init_atoms: Atoms | None = None
        self._final_atoms: Atoms | None = None
        # Where each endpoint was loaded from (None = no source file,
        # e.g. the initial endpoint handed in from an unsaved structure).
        # Project-run creation registers these files in the project.
        self._init_source_path: str | None = None
        self._final_source_path: str | None = None
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
            self._init_source_path = structure_model.filepath
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
        self._coord_label = QLabel(self.tr("Coordinates:"))
        coord_row.addWidget(self._coord_label)
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

        self._preview_label = QLabel(self.tr("Preview:"))
        layout.addWidget(self._preview_label)
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
        self._ini_label = QLabel(self.tr("Initial structure:"))
        ini_row.addWidget(self._ini_label)
        self._ini_default = self._init_atoms is not None
        self._ini_edit = QLineEdit(
            self.tr("(current structure)") if self._ini_default else "")
        self._ini_edit.setReadOnly(True)
        ini_row.addWidget(self._ini_edit, 1)
        if self._project_provider is not None:
            self._ini_project_btn = QPushButton(
                self.tr("From Project..."))
            self._ini_project_btn.clicked.connect(self._on_project_ini)
            ini_row.addWidget(self._ini_project_btn)
        self._ini_browse = QPushButton(self.tr("Browse..."))
        self._ini_browse.clicked.connect(self._on_browse_ini)
        ini_row.addWidget(self._ini_browse)
        layout.addLayout(ini_row)

        # Final structure
        fin_row = QHBoxLayout()
        self._fin_label = QLabel(self.tr("Final structure:"))
        fin_row.addWidget(self._fin_label)
        self._fin_edit = QLineEdit("")
        self._fin_edit.setReadOnly(True)
        fin_row.addWidget(self._fin_edit, 1)
        if self._project_provider is not None:
            self._fin_project_btn = QPushButton(
                self.tr("From Project..."))
            self._fin_project_btn.clicked.connect(self._on_project_fin)
            fin_row.addWidget(self._fin_project_btn)
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
        # Theme-aware via QLabel[hintKind="warn"] in the app QSS.
        self._warn_label.setProperty("hintKind", "warn")
        layout.addWidget(self._warn_label)

        # Interpolation algorithm (linear is the default; IDPP avoids
        # atoms passing through each other on the straight-line path)
        algo_row = QHBoxLayout()
        self._algo_label = QLabel(self.tr("Algorithm:"))
        algo_row.addWidget(self._algo_label)
        self._algo_combo = MenuButton()
        self._algo_combo.addItems([self.tr("Linear"), self.tr("IDPP")])
        algo_row.addWidget(self._algo_combo)
        algo_row.addStretch()
        layout.addLayout(algo_row)

        # Interpolation controls
        interp_row = QHBoxLayout()
        self._images_label = QLabel(self.tr("Intermediate images:"))
        interp_row.addWidget(self._images_label)
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
        self._images_list_label = QLabel(self.tr("Image frames (click to preview):"))
        layout.addWidget(self._images_list_label)
        self._images_list = QListWidget()
        self._images_list.currentItemChanged.connect(self._on_image_selected)
        layout.addWidget(self._images_list, 1)
        return page

    # ------------------------------------------------------------------
    # Language switching (the panel lives inside the non-modal dialog)
    # ------------------------------------------------------------------

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.LanguageChange:
            self._retranslate()
        super().changeEvent(event)

    def _retranslate(self) -> None:
        self._coord_label.setText(self.tr("Coordinates:"))
        coord_index = self._coord_combo.currentIndex()
        self._coord_combo.blockSignals(True)
        self._coord_combo.clear()
        self._coord_combo.addItems([
            self.tr("Direct (fractional)"),
            self.tr("Cartesian"),
        ])
        self._coord_combo.setCurrentIndex(coord_index)
        self._coord_combo.blockSignals(False)
        self._preview_label.setText(self.tr("Preview:"))
        self._ini_label.setText(self.tr("Initial structure:"))
        self._fin_label.setText(self.tr("Final structure:"))
        if self._project_provider is not None:
            self._ini_project_btn.setText(self.tr("From Project..."))
            self._fin_project_btn.setText(self.tr("From Project..."))
        if self._ini_default:
            self._ini_edit.setText(self.tr("(current structure)"))
        self._ini_browse.setText(self.tr("Browse..."))
        self._fin_browse.setText(self.tr("Browse..."))
        self._algo_label.setText(self.tr("Algorithm:"))
        algo_index = self._algo_combo.currentIndex()
        self._algo_combo.blockSignals(True)
        self._algo_combo.clear()
        self._algo_combo.addItems([self.tr("Linear"), self.tr("IDPP")])
        self._algo_combo.setCurrentIndex(algo_index)
        self._algo_combo.blockSignals(False)
        self._images_label.setText(self.tr("Intermediate images:"))
        self._interpolate_btn.setText(self.tr("Interpolate"))
        self._images_list_label.setText(self.tr("Image frames (click to preview):"))
        self._update_diagnostics()
        self._refresh_poscar_preview()

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

    def initial_source_path(self) -> str | None:
        """File the initial endpoint was loaded from (None when it has
        no source file — e.g. an unsaved structure from the viewport)."""
        return self._init_source_path

    def final_source_path(self) -> str | None:
        """File the final endpoint was loaded from (None until one is
        browsed in)."""
        return self._final_source_path

    def neb_method(self) -> str:
        """Selected interpolation algorithm: "linear" or "idpp"."""
        return "idpp" if self._algo_combo.currentIndex() == 1 else "linear"

    # ------------------------------------------------------------------
    # Frame editing (middle images only)
    # ------------------------------------------------------------------

    def frame_preview_data(self, index: int):
        """Preview data for the frame at ``index``.

        Returns (atoms, editable, fixed_flags, bonds). ``editable`` is
        True only for intermediate frames — the initial and final
        frames are locked. Bonds are auto-detected on first access
        (minus user-deleted pairs) and cached per frame. Returns None
        when the images were cleared (new endpoints browsed).
        """
        if not (0 <= index < len(self._images)):
            return None
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

    def _confirm_discard_frame_edits(self) -> bool:
        """Ask before an action that silently discards manual frame edits."""
        if not self.has_frame_edits():
            return True
        reply = QMessageBox.question(
            self,
            self.tr("Discard Frame Edits"),
            self.tr("Changing the initial/final structure discards the "
                    "manual edits made to the images.\n\nContinue?"),
        )
        return reply == QMessageBox.Yes

    def _load_endpoint_file(self, filepath: str, final: bool) -> None:
        """Load an endpoint structure from ``filepath`` (shared by the
        file-browse and project-picker paths)."""
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
        if final:
            self._final_atoms = atoms
            self._final_source_path = filepath
            self._fin_edit.setText(filepath)
        else:
            self._init_atoms = atoms
            self._ini_fixed_flags = constraints_to_fixed_flags(atoms)
            self._init_source_path = filepath
            self._ini_edit.setText(filepath)
            self._ini_default = False
        self._config.last_directory = str(Path(filepath).parent)
        self._clear_images()
        self._update_diagnostics()

    def _on_browse_ini(self) -> None:
        if not self._confirm_discard_frame_edits():
            return
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Initial Structure (POSCAR)"),
            self._config.last_directory,
            _POSCAR_FILTER,
        )
        if not filepath:
            return
        self._load_endpoint_file(filepath, final=False)

    def _on_browse_fin(self) -> None:
        if not self._confirm_discard_frame_edits():
            return
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Final Structure (POSCAR)"),
            self._config.last_directory,
            _POSCAR_FILTER,
        )
        if not filepath:
            return
        self._load_endpoint_file(filepath, final=True)

    def _on_project_ini(self) -> None:
        if not self._confirm_discard_frame_edits():
            return
        path = self._pick_project_structure()
        if path is not None:
            self._load_endpoint_file(str(path), final=False)

    def _on_project_fin(self) -> None:
        if not self._confirm_discard_frame_edits():
            return
        path = self._pick_project_structure()
        if path is not None:
            self._load_endpoint_file(str(path), final=True)

    def _pick_project_structure(self):
        """Modal pick over the project's registered structures — the
        path of the chosen one, or None (cancelled / nothing picked /
        no project or empty project)."""
        from vaspen.ui.structure_picker import StructurePickerDialog

        project = (self._project_provider()
                   if self._project_provider is not None else None)
        if project is None or not project.structures:
            QMessageBox.information(
                self, self.tr("No Structures"),
                self.tr("The open project has no registered structures."))
            return None
        dlg = StructurePickerDialog(project, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return dlg.selected_structure_path()

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
        # Tell the main window the frames are gone — an active frame-edit
        # context would otherwise keep pointing at a cleared list.
        if self._preview_callback is not None:
            self._preview_callback(None, None, False, None, None, self)

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

    #: Emitted after a project run was created and filled (run id) —
    #: the main window refreshes the project panel on it.
    run_created = Signal(str)

    def __init__(self, structure_model, parent=None,
                 preview_callback=None, default_task: str = "scf",
                 project_provider=None) -> None:
        super().__init__(parent)
        self._structure_model = structure_model
        self._preview_callback = preview_callback
        #: Callable returning the currently open Project (or None),
        #: resolved at Generate time — the dialog is non-modal, so the
        #: project can be opened/closed while it sits there.
        self._project_provider = project_provider
        self._config = AppConfig()
        self._syncing = False

        self.setWindowTitle(self.tr("Generate All Input Files"))
        self.resize(900, 700)
        self._build_ui()

        if default_task not in TASKS:
            default_task = "scf"
        self._sync_task(default_task, source="init")

    # ------------------------------------------------------------------
    # Language switching (non-modal/persistent → live retranslate,
    # CLAUDE.md §11.2; the embedded panels handle their own strings)
    # ------------------------------------------------------------------

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.LanguageChange:
            self._retranslate()
        super().changeEvent(event)

    def _retranslate(self) -> None:
        self.setWindowTitle(self.tr("Generate All Input Files"))
        self._task_label.setText(self.tr("Calculation Type:"))
        index = self._task_combo.currentIndex()
        self._task_combo.blockSignals(True)  # no task re-sync on retranslate
        self._task_combo.clear()
        self._task_combo.addItems([
            self.tr("SCF (Static)"),
            self.tr("Optimization"),
            self.tr("Band Structure"),
            self.tr("DOS"),
            self.tr("Optical"),
            self.tr("NEB"),
            self.tr("Custom"),
        ])
        self._task_combo.setCurrentIndex(index)
        self._task_combo.blockSignals(False)
        self._tabs.setTabText(0, self.tr("POSCAR"))
        self._tabs.setTabText(1, self.tr("INCAR"))
        self._tabs.setTabText(2, self.tr("KPOINTS"))
        self._tabs.setTabText(3, self.tr("POTCAR"))
        if self._dest_project_radio is not None:
            self._dest_project_radio.setText(
                self.tr("Current project (new run)"))
            self._dest_folder_radio.setText(self.tr("Folder:"))
            self._run_name_label.setText(self.tr("Run name:"))
        self._dir_label.setText(self.tr("Output directory:"))
        self._dir_browse_btn.setText(self.tr("Browse..."))
        self._generate_btn.setText(self.tr("Generate"))
        self._cancel_btn.setText(self.tr("Cancel"))

    def _on_destination_changed(self) -> None:
        """Project destination: the folder picker is inert; folder
        destination: the run name is."""
        project_selected = self._dest_project_radio.isChecked()
        self._run_name_label.setEnabled(project_selected)
        self._run_name_edit.setEnabled(project_selected)
        self._dir_label.setEnabled(not project_selected)
        self._dir_edit.setEnabled(not project_selected)
        self._dir_browse_btn.setEnabled(not project_selected)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Task selector ──
        task_row = QHBoxLayout()
        self._task_label = QLabel(self.tr("Calculation Type:"))
        task_row.addWidget(self._task_label)
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
            self._structure_model, preview_callback=self._preview_callback,
            project_provider=self._project_provider)
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

        # ── Destination ──
        self._dest_project_radio: QRadioButton | None = None
        self._dest_folder_radio: QRadioButton | None = None
        self._run_name_edit: QLineEdit | None = None
        if self._project_provider is not None:
            radio_row = QHBoxLayout()
            self._dest_project_radio = QRadioButton(
                self.tr("Current project (new run)"))
            self._dest_project_radio.setChecked(True)
            self._dest_project_radio.toggled.connect(
                self._on_destination_changed)
            radio_row.addWidget(self._dest_project_radio)
            self._dest_folder_radio = QRadioButton(self.tr("Folder:"))
            radio_row.addWidget(self._dest_folder_radio)
            radio_row.addStretch()
            layout.addLayout(radio_row)

            name_row = QHBoxLayout()
            self._run_name_label = QLabel(self.tr("Run name:"))
            name_row.addWidget(self._run_name_label)
            self._run_name_edit = QLineEdit()
            self._run_name_edit.setToolTip(self.tr(
                "Names the new run — becomes the runs/<id>/ directory "
                "(empty: automatic from task and structure)."))
            name_row.addWidget(self._run_name_edit, 1)
            layout.addLayout(name_row)

        dir_row = QHBoxLayout()
        self._dir_label = QLabel(self.tr("Output directory:"))
        dir_row.addWidget(self._dir_label)
        self._dir_edit = QLineEdit(self._config.last_directory)
        dir_row.addWidget(self._dir_edit, 1)
        self._dir_browse_btn = QPushButton(self.tr("Browse..."))
        self._dir_browse_btn.clicked.connect(self._browse_output_dir)
        dir_row.addWidget(self._dir_browse_btn)
        layout.addLayout(dir_row)
        if self._dest_folder_radio is not None:
            self._on_destination_changed()

        # ── Buttons ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._generate_btn = QPushButton(self.tr("Generate"))
        self._generate_btn.clicked.connect(self._on_generate)
        self._cancel_btn = QPushButton(self.tr("Cancel"))
        self._cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._generate_btn)
        btn_row.addWidget(self._cancel_btn)
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
        project = None
        run_id: str | None = None
        if (self._dest_project_radio is not None
                and self._dest_project_radio.isChecked()):
            project = self._project_provider()
            if project is None:
                QMessageBox.warning(
                    self, self.tr("No Project Open"),
                    self.tr("No project is open — open or create one "
                            "first (File → New/Open Project)."))
                return
        else:
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

        if project is not None:
            prepared = self._prepare_project_run(project)
            if prepared is None:
                return
            out, run_id = prepared
        else:
            out = Path(out_text)
        written: list[Path] = []
        try:
            for name, content in files.items():
                target = out / name
                target.parent.mkdir(parents=True, exist_ok=True)
                # UTF-8 + LF: VASP input files must not carry the
                # locale encoding (GBK) or CRLF line endings
                target.write_text(content, encoding="utf-8", newline="\n")
                written.append(target)
        except OSError as e:
            # Best-effort cleanup of this run's partial output — do not
            # leave a half-written set that reads as fully generated.
            for path in reversed(written):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            QMessageBox.critical(
                self, self.tr("Generation Failed"),
                self.tr("{}\n\nPartially written files were removed.")
                .format(e))
            return

        if run_id is not None:
            self.run_created.emit(run_id)
            QMessageBox.information(
                self,
                self.tr("Success"),
                self.tr("Run {} created in the project.\n\nFiles: {}").format(
                    run_id, ", ".join(files)),
            )
        else:
            self._config.last_directory = str(out)
            QMessageBox.information(
                self,
                self.tr("Success"),
                self.tr("Generated files in:\n{}\n\nFiles: {}").format(
                    str(out), ", ".join(files)),
            )
        self.accept()

    def _prepare_project_run(self, project):
        """Register the run's structure(s), create the run entry and
        return ``(run_dir, run_id)`` — or None after showing an error
        (the caller then aborts before writing anything)."""
        try:
            task = TASKS[self._task_combo.currentIndex()]
            if self._poscar_panel.neb_mode:
                # both endpoints become project structures so the run's
                # provenance points at real registered files
                initial = self._poscar_panel.initial_source_path()
                if not initial:
                    QMessageBox.information(
                        self, self.tr("Structure Not Saved"),
                        self.tr("The NEB initial endpoint has no source "
                                "file — load it from a file (or save the "
                                "current structure) first."))
                    return None
                structure_id, digest = project.register_structure_file(
                    initial,
                    formula=self._poscar_panel.images[0]
                    .get_chemical_formula())
                final_id = ""
                final = self._poscar_panel.final_source_path()
                if final:
                    final_id, _ = project.register_structure_file(
                        final,
                        formula=self._poscar_panel.images[-1]
                        .get_chemical_formula())
                params = {
                    "preset": task,
                    "kpoints_mode": self._kpoints_panel.mode(),
                    "functional": self._potcar_panel.functional(),
                    "method": self._poscar_panel.neb_method(),
                    "n_images": self._poscar_panel.n_images,
                    "final_structure": final_id,
                }
            else:
                if not self._structure_model.filepath:
                    QMessageBox.information(
                        self, self.tr("Structure Not Saved"),
                        self.tr("Save the structure to a file first, "
                                "then generate the run."))
                    return None
                structure_id, digest = project.register_structure_file(
                    self._structure_model.filepath,
                    formula=self._structure_model.chemical_formula)
                params = {
                    "preset": task,
                    "kpoints_mode": self._kpoints_panel.mode(),
                    "functional": self._potcar_panel.functional(),
                }
            name = (self._run_name_edit.text().strip()
                    if self._run_name_edit is not None else "") \
                or self._auto_run_name(task)
            entry = project.add_run(name, task, params,
                                    structure_id=structure_id,
                                    structure_digest=digest, via="gui")
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, self.tr("Generation Failed"), str(e))
            return None
        return project.run_dir(entry["id"]), entry["id"]

    def _auto_run_name(self, task: str) -> str:
        """Default run name: <task>-<structure stem or formula>."""
        model = self._structure_model
        stem = ""
        if model is not None and model.filepath:
            stem = Path(model.filepath).stem
        elif model is not None and model.n_atoms:
            stem = model.chemical_formula
        return f"{task}-{stem}" if stem else task
