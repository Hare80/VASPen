"""Surface / slab cutting dialog with live preview in the 3D viewport.

Non-modal and persistent while open, so it implements changeEvent +
_retranslate (CLAUDE.md §11.2 pattern). Every parameter change renders
the selected slab from a cached SlabInfo list into the viewport (the
model is untouched); Accept exposes the slab as result_structure,
Cancel/Esc restores the viewport to the model state.
"""

from __future__ import annotations

from ase import Atoms
from PySide6.QtCore import (
    QEvent,
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.structure import StructureModel
from vaspen.core.surface import (
    SlabInfo,
    SurfaceCutter,
    _subscript_formula,
    supercell_in_plane,
)
from vaspen.ui.menu_button import MenuButton


class _SlabResultHolder(QObject):
    """Main-thread relay for pool results (auto-connection → queued)."""

    done = Signal(object, object, int, object)  # infos, error, gen, key


class _SlabTask(QRunnable):
    """One slab computation on the global thread pool.

    QThreadPool manages the threads — unlike a hand-rolled QThread with
    deleteLater/quit, there is no thread-lifecycle race (the pattern
    crashed intermittently when queued signals were delivered during
    processEvents).
    """

    def __init__(self, holder: _SlabResultHolder, atoms: Atoms,
                 key: tuple, generation: int) -> None:
        super().__init__()
        self._holder = holder
        self._atoms = atoms
        self._key = key
        self._generation = generation

    def run(self) -> None:
        try:
            # StructureModel wraps the copy (derived-state init is cheap
            # relative to pymatgen and keeps the core API surface).
            model = StructureModel(self._atoms.copy())
            infos = SurfaceCutter(model).slabs(*self._key)
            self._holder.done.emit(infos, None, self._generation, self._key)
        except Exception as e:  # noqa: BLE001 — surfaced in the dialog
            self._holder.done.emit(None, e, self._generation, self._key)


class SurfaceDialog(QDialog):
    """Cleave a surface/slab from a bulk structure with live preview.

    The main viewport shows the cut slab while the dialog is open and
    stays rotatable/zoomable; MainWindow disables structure-editing
    entry points while the dialog lives (see _set_surface_editing_enabled).
    """

    def __init__(
        self,
        structure_model: StructureModel,
        viewport,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model = structure_model
        self._viewport = viewport
        self.result_structure: StructureModel | None = None
        self._slab_infos: list[SlabInfo] = []
        self._cache_key: tuple[tuple[int, int, int], int, float] | None = None
        self._debounce: QTimer | None = None
        self._hint_state: str | None = None
        self._hint_detail = ""
        # Async computation on the global thread pool, one task at a
        # time: a generation counter guards against out-of-order
        # results; parameter changes arriving while a task runs are
        # queued as _pending_key and chained on completion.
        self._generation = 0
        self._busy = False
        self._pending_key: tuple | None = None
        self._result_holder = _SlabResultHolder()
        self._result_holder.done.connect(self._on_slabs_ready)

        self.setWindowTitle(self.tr("Cleave Surface / Slab"))
        self.setWindowFlags(self.windowFlags() | Qt.Tool)  # floats above parent
        self.resize(460, 320)
        self._build_ui()
        self._ensure_slabs()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Miller indices ──
        self._miller_group = QGroupBox(self.tr("Miller Indices"))
        miller_layout = QHBoxLayout()

        miller_layout.addWidget(QLabel("h:"))
        self._h_spin = QSpinBox()
        self._h_spin.setRange(-5, 5)
        self._h_spin.setValue(1)
        self._h_spin.valueChanged.connect(self._on_params_changed)
        miller_layout.addWidget(self._h_spin)

        miller_layout.addWidget(QLabel("k:"))
        self._k_spin = QSpinBox()
        self._k_spin.setRange(-5, 5)
        self._k_spin.setValue(1)
        self._k_spin.valueChanged.connect(self._on_params_changed)
        miller_layout.addWidget(self._k_spin)

        miller_layout.addWidget(QLabel("l:"))
        self._l_spin = QSpinBox()
        self._l_spin.setRange(-5, 5)
        self._l_spin.setValue(1)
        self._l_spin.valueChanged.connect(self._on_params_changed)
        miller_layout.addWidget(self._l_spin)

        self._miller_group.setLayout(miller_layout)
        layout.addWidget(self._miller_group)

        # ── Parameters ──
        params_layout = QFormLayout()

        self._layers_spin = QSpinBox()
        self._layers_spin.setRange(1, 100)
        self._layers_spin.setValue(4)
        self._layers_spin.setToolTip(self.tr(
            "Number of atomic layers in the slab.\n"
            "More layers = thicker slab, more computational cost."
        ))
        self._layers_spin.valueChanged.connect(self._on_params_changed)
        self._layers_label = QLabel(self.tr("Layers:"))
        params_layout.addRow(self._layers_label, self._layers_spin)

        self._vacuum_spin = QDoubleSpinBox()
        self._vacuum_spin.setRange(0.0, 100.0)
        self._vacuum_spin.setValue(15.0)
        self._vacuum_spin.setSuffix(" Å")
        self._vacuum_spin.setToolTip(self.tr(
            "Vacuum thickness added above the slab.\n"
            "Standard: 10–15 Å for surface calculations."
        ))
        self._vacuum_spin.valueChanged.connect(self._on_params_changed)
        self._vacuum_label = QLabel(self.tr("Vacuum:"))
        params_layout.addRow(self._vacuum_label, self._vacuum_spin)

        # In-plane supercell (a × b) — the vacuum direction (c) is
        # never expanded; a supercell along c is meaningless for a slab.
        supercell_box = QHBoxLayout()
        self._supercell_a_spin = QSpinBox()
        self._supercell_a_spin.setRange(1, 10)
        self._supercell_a_spin.setValue(1)
        self._supercell_a_spin.valueChanged.connect(self._on_supercell_changed)
        supercell_box.addWidget(self._supercell_a_spin)
        supercell_box.addWidget(QLabel("×"))
        self._supercell_b_spin = QSpinBox()
        self._supercell_b_spin.setRange(1, 10)
        self._supercell_b_spin.setValue(1)
        self._supercell_b_spin.valueChanged.connect(self._on_supercell_changed)
        supercell_box.addWidget(self._supercell_b_spin)
        supercell_box.addStretch(1)
        self._supercell_label = QLabel(self.tr("Supercell (a×b):"))
        self._supercell_label.setToolTip(self.tr(
            "In-plane supercell size (a × b). "
            "The vacuum direction (c) is never expanded."))
        params_layout.addRow(self._supercell_label, supercell_box)

        layout.addLayout(params_layout)

        # ── Termination ──
        self._termination_group = QGroupBox(self.tr("Termination"))
        term_layout = QVBoxLayout()
        self._termination_combo = MenuButton()
        self._termination_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._termination_combo.currentIndexChanged.connect(
            self._on_termination_changed)
        term_layout.addWidget(self._termination_combo)
        self._termination_label = QLabel("")
        self._termination_label.setWordWrap(True)
        term_layout.addWidget(self._termination_label)
        self._termination_group.setLayout(term_layout)
        layout.addWidget(self._termination_group)

        # ── Preview note ──
        self._preview_hint = QLabel(self.tr(
            "Preview: the main view shows the cut slab; rotate/zoom to "
            "inspect it. Structure editing is paused until you close this "
            "dialog."))
        self._preview_hint.setWordWrap(True)
        layout.addWidget(self._preview_hint)

        # ── Error hint ──
        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: #b00020;")
        layout.addWidget(self._hint)

        # ── Status ──
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # ── Buttons ──
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._ok_button = buttons.button(QDialogButtonBox.Ok)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Slab computation (debounced; termination switching reuses cache)
    # ------------------------------------------------------------------

    def _on_params_changed(self, *_args) -> None:
        """Restart the debounce timer so held spinbox arrows recompute once."""
        if self._debounce is None:
            self._debounce = QTimer(self)
            self._debounce.setSingleShot(True)
            self._debounce.setInterval(150)
            self._debounce.timeout.connect(self._ensure_slabs)
        self._debounce.start()

    def _ensure_slabs(self) -> None:
        """Recompute the termination list for the current parameters.

        Cached by (miller, layers, vacuum); termination switching only
        re-pushes the cached slab. On failure the OK button is disabled
        and the cause is shown in the hint label (no message boxes —
        the live preview flow should not be blocked).
        """
        key = ((self._h_spin.value(), self._k_spin.value(),
                self._l_spin.value()),
               self._layers_spin.value(),
               float(self._vacuum_spin.value()))
        if key == self._cache_key:
            return
        self._cache_key = key

        if key[0] == (0, 0, 0):
            # Invalidate any in-flight/queued computation — its result
            # must not overwrite the zero-index error state.
            self._generation += 1
            self._pending_key = None
            self._slab_infos = []
            self._ok_button.setEnabled(False)
            self._rebuild_termination_combo()
            self._set_hint("zero_miller")
            return

        self._start_compute(key)

    def _start_compute(self, key: tuple) -> None:
        """Run slab generation on the thread pool (the GUI must not
        freeze — pymatgen takes seconds per parameter change). At most
        one task runs at a time; a key requested mid-run is queued as
        ``_pending_key`` and chained on completion."""
        if self._busy:
            self._pending_key = key  # the running task chains it
            return
        self._pending_key = None
        self._busy = True
        self._generation += 1
        self._ok_button.setEnabled(False)
        self._status_label.setText(self.tr("Computing slab…"))
        task = _SlabTask(self._result_holder, self._model.atoms.copy(),
                         key, self._generation)
        QThreadPool.globalInstance().start(task)

    def _on_slabs_ready(self, infos, error, generation: int,
                        key: tuple) -> None:
        """Apply a finished computation; chain a queued newer one."""
        self._busy = False
        if generation == self._generation:
            if error is not None:
                self._slab_infos = []
                self._ok_button.setEnabled(False)
                self._rebuild_termination_combo()
                self._set_hint("error", str(error))
                self._status_label.setText("")
            else:
                self._slab_infos = infos
                self._ok_button.setEnabled(bool(infos))
                self._set_hint(None)
                self._rebuild_termination_combo()
                self._push_preview()
        if self._pending_key is not None:
            next_key = self._pending_key
            self._pending_key = None
            self._start_compute(next_key)

    def _rebuild_termination_combo(self) -> None:
        """Re-populate the termination combo, keeping the current index."""
        index = self._termination_combo.currentIndex()
        self._termination_combo.blockSignals(True)
        self._termination_combo.clear()
        for i, info in enumerate(self._slab_infos):
            self._termination_combo.addItem(
                self.tr("{i}/{n} — top: {top}, bottom: {bottom}").format(
                    i=i + 1,
                    n=len(self._slab_infos),
                    top=_subscript_formula(info.top_composition),
                    bottom=_subscript_formula(info.bottom_composition),
                ))
        self._termination_combo.setCurrentIndex(
            min(max(index, 0), len(self._slab_infos) - 1))
        self._termination_combo.blockSignals(False)

    def _on_termination_changed(self, _index: int) -> None:
        if self._slab_infos:
            self._push_preview()

    def _on_supercell_changed(self, *_args) -> None:
        """Supercell expansion reuses the cached slabs (repeat is cheap)."""
        if self._slab_infos:
            self._push_preview()

    def _displayed_atoms(self) -> Atoms | None:
        """The selected slab expanded to the requested in-plane supercell."""
        if not self._slab_infos:
            return None
        info = self._slab_infos[self._termination_combo.currentIndex()]
        return supercell_in_plane(
            info.atoms, self._supercell_a_spin.value(),
            self._supercell_b_spin.value())

    def _push_preview(self) -> None:
        """Render the selected slab into the viewport.

        bonds=None: the model's bond list is invalid for a different
        atom count — set_structure auto-detects (viewport3d.py:828).
        Model highlights are NOT replayed: indices may exceed the slab
        atom count (set_structure already cleared the selection sets).
        """
        atoms = self._displayed_atoms()
        if atoms is None:
            return
        self._viewport.set_structure(atoms, reset_view=False, bonds=None)
        self._update_info_labels()

    def _update_info_labels(self) -> None:
        atoms = self._displayed_atoms()
        if atoms is None:
            self._termination_label.setText("")
            self._status_label.setText("")
            return
        info = self._slab_infos[self._termination_combo.currentIndex()]
        self._termination_label.setText(
            self.tr("Top: {top}, bottom: {bottom}").format(
                top=_subscript_formula(info.top_composition),
                bottom=_subscript_formula(info.bottom_composition),
            ))
        note = (self.tr("  (re-boxed — input already carried vacuum)")
                if info.reboxed else "")
        self._status_label.setText(
            self.tr("{n} atoms · {v} Å vacuum").format(
                n=len(atoms), v=f"{self._vacuum_spin.value():.1f}") + note)

    def _set_hint(self, state: str | None, detail: str = "") -> None:
        """Show/clear the error hint; state is remembered for retranslate."""
        self._hint_state = state
        self._hint_detail = detail
        if state == "zero_miller":
            self._hint.setText(
                self.tr("At least one Miller index must be non-zero."))
        elif state == "error":
            self._hint.setText(
                self.tr("Could not create slab:\n{}").format(detail))
        else:
            self._hint.setText("")

    # ------------------------------------------------------------------
    # Accept / reject
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        if self._model.n_atoms == 0:
            QMessageBox.warning(
                self,
                self.tr("No Structure"),
                self.tr("Load a bulk structure before cleaving a surface."),
            )
            return
        # Cleaving rebuilds the structure from scratch — fractional
        # occupancy (disorder) cannot survive it (same policy as the
        # symmetrize confirm). Warn and confirm first.
        if self._model.has_disorder:
            reply = QMessageBox.warning(
                self, self.tr("Partial Occupancy"),
                self.tr(
                    "This structure has partial occupancy (disorder).\n"
                    "Cleaving will discard the fractional occupancy "
                    "information.\n\n"
                    "Continue?"),
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return
        atoms = self._displayed_atoms()
        if atoms is None:
            QMessageBox.warning(
                self,
                self.tr("Cleave Failed"),
                self.tr("Could not create slab."),
            )
            return
        self.result_structure = StructureModel(atoms.copy())
        self.accept()

    def reject(self) -> None:
        """Restore the viewport to the real model state before closing."""
        self._viewport.set_structure(self._model.atoms, reset_view=False,
                                     bonds=self._model.bonds,
                                     fixed=self._model.fixed_flags)
        self._viewport.set_highlight(self._model.selected_indices)
        self._viewport.set_bond_highlight(self._model.selected_bonds)
        super().reject()

    # ------------------------------------------------------------------
    # Language switching (non-modal/persistent → live retranslate)
    # ------------------------------------------------------------------

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.LanguageChange:
            self._retranslate()
        super().changeEvent(event)

    def _retranslate(self) -> None:
        self.setWindowTitle(self.tr("Cleave Surface / Slab"))
        self._miller_group.setTitle(self.tr("Miller Indices"))
        self._layers_label.setText(self.tr("Layers:"))
        self._layers_spin.setToolTip(self.tr(
            "Number of atomic layers in the slab.\n"
            "More layers = thicker slab, more computational cost."
        ))
        self._vacuum_label.setText(self.tr("Vacuum:"))
        self._vacuum_spin.setToolTip(self.tr(
            "Vacuum thickness added above the slab.\n"
            "Standard: 10–15 Å for surface calculations."
        ))
        self._supercell_label.setText(self.tr("Supercell (a×b):"))
        self._supercell_label.setToolTip(self.tr(
            "In-plane supercell size (a × b). "
            "The vacuum direction (c) is never expanded."))
        self._termination_group.setTitle(self.tr("Termination"))
        self._preview_hint.setText(self.tr(
            "Preview: the main view shows the cut slab; rotate/zoom to "
            "inspect it. Structure editing is paused until you close this "
            "dialog."))
        self._rebuild_termination_combo()
        self._set_hint(self._hint_state, self._hint_detail)
        self._update_info_labels()
