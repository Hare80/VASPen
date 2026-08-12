"""KPOINTS editor dialog.

Supports three modes:
- Automatic KSPACING (recommended)
- Manual k-mesh grid
- Line-mode for band structure
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.vasp_input import (
    generate_kpoints_automatic,
    generate_kpoints_manual,
    generate_kpoints_line_mode,
    get_high_symmetry_points,
    KSPACING_RECOMMEND,
    estimate_k_mesh,
)


class KpointsEditorDialog(QDialog):
    """Dialog for generating KPOINTS content."""

    def __init__(self, structure_model=None, parent=None) -> None:
        super().__init__(parent)
        self._structure_model = structure_model

        self.setWindowTitle(self.tr("Generate KPOINTS"))
        self.resize(650, 500)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Mode selector ──
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel(self.tr("KPOINTS Mode:")))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([
            self.tr("Automatic (KSPACING) — recommended"),
            self.tr("Manual Mesh (n1 × n2 × n3)"),
            self.tr("Line-mode (Band Structure)"),
        ])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo, 1)
        layout.addLayout(mode_row)

        # ── Stacked parameter pages ──
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_automatic_page())
        self._stack.addWidget(self._build_manual_page())
        self._stack.addWidget(self._build_line_page())
        layout.addWidget(self._stack)

        # ── Estimated k-mesh info ──
        self._estimate_label = QLabel("")
        layout.addWidget(self._estimate_label)

        # ── Preview ──
        layout.addWidget(QLabel(self.tr("Preview:")))
        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setFontFamily("Consolas, monospace")
        self._preview.setMaximumHeight(180)
        layout.addWidget(self._preview)

        # ── Buttons ──
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._on_mode_changed(0)

    # ------------------------------------------------------------------
    # Page 1: Automatic KSPACING
    # ------------------------------------------------------------------

    def _build_automatic_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        # KSPACING slider
        slider_row = QHBoxLayout()
        self._kspacing_slider = QSlider(Qt.Horizontal)
        self._kspacing_slider.setRange(1, 50)  # 0.01 to 0.50
        self._kspacing_slider.setValue(4)       # 0.04 default
        self._kspacing_slider.valueChanged.connect(self._kspacing_slider_changed)
        slider_row.addWidget(self._kspacing_slider)

        self._kspacing_label = QLabel("0.040")
        self._kspacing_label.setMinimumWidth(50)
        slider_row.addWidget(self._kspacing_label)
        form.addRow(self.tr("KSPACING (Å⁻¹):"), slider_row)

        # Recommendation notes
        rec_text = (
            self.tr("Recommended: Insulators = 0.04, Metals = 0.03")
            + "\n"
            + self.tr("Fine = 0.02, Coarse = 0.05")
        )
        self._kspacing_note = QLabel(rec_text)
        self._kspacing_note.setWordWrap(True)
        form.addRow("", self._kspacing_note)

        # Gamma-centered checkbox
        self._gamma_auto = QComboBox()
        self._gamma_auto.addItems([self.tr("Gamma-centered"), self.tr("Monkhorst-Pack")])
        form.addRow(self.tr("Scheme:"), self._gamma_auto)

        self._gamma_auto.currentIndexChanged.connect(self._update_preview)

        return page

    def _kspacing_slider_changed(self, value: int) -> None:
        spacing = value / 100.0
        self._kspacing_label.setText(f"{spacing:.3f}")
        self._update_estimate()
        self._update_preview()

    # ------------------------------------------------------------------
    # Page 2: Manual Mesh
    # ------------------------------------------------------------------

    def _build_manual_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        mesh_row = QHBoxLayout()
        self._k1_spin = QSpinBox()
        self._k1_spin.setRange(1, 50)
        self._k1_spin.setValue(1)
        self._k2_spin = QSpinBox()
        self._k2_spin.setRange(1, 50)
        self._k2_spin.setValue(1)
        self._k3_spin = QSpinBox()
        self._k3_spin.setRange(1, 50)
        self._k3_spin.setValue(1)
        mesh_row.addWidget(QLabel("n1:"))
        mesh_row.addWidget(self._k1_spin)
        mesh_row.addWidget(QLabel("n2:"))
        mesh_row.addWidget(self._k2_spin)
        mesh_row.addWidget(QLabel("n3:"))
        mesh_row.addWidget(self._k3_spin)
        mesh_row.addStretch()
        form.addRow(self.tr("k-mesh:"), mesh_row)

        self._gamma_manual = QComboBox()
        self._gamma_manual.addItems([self.tr("Gamma-centered"), self.tr("Monkhorst-Pack")])
        form.addRow(self.tr("Scheme:"), self._gamma_manual)

        for spin in (self._k1_spin, self._k2_spin, self._k3_spin):
            spin.valueChanged.connect(self._update_preview)
        self._gamma_manual.currentIndexChanged.connect(self._update_preview)

        return page

    # ------------------------------------------------------------------
    # Page 3: Line-mode (Band)
    # ------------------------------------------------------------------

    def _build_line_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self._band_path_edit = QLineEdit("G-X|X-M|M-G")
        self._band_path_edit.setToolTip(
            self.tr("High-symmetry k-path, e.g. G-X|X-W|W-L|L-G|G-K")
        )
        self._band_path_edit.textChanged.connect(self._update_preview)
        form.addRow(self.tr("k-path:"), self._band_path_edit)

        path_note = QLabel(
            self.tr("Format: <start>-<end>|<start>-<end>|...\n"
                    "Labels: G=Gamma, X, M, R, K, L, W, etc.")
        )
        path_note.setWordWrap(True)
        form.addRow("", path_note)

        self._band_npoints = QSpinBox()
        self._band_npoints.setRange(5, 100)
        self._band_npoints.setValue(20)
        self._band_npoints.valueChanged.connect(self._update_preview)
        form.addRow(self.tr("Points per segment:"), self._band_npoints)

        return page

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _on_mode_changed(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        self._update_estimate()
        self._update_preview()

    # ------------------------------------------------------------------
    # Estimate
    # ------------------------------------------------------------------

    def _update_estimate(self) -> None:
        if self._structure_model is None or self._structure_model.n_atoms == 0:
            self._estimate_label.setText("")
            return

        mode = self._mode_combo.currentIndex()
        if mode == 0:  # automatic
            spacing = self._kspacing_slider.value() / 100.0
            try:
                mesh = estimate_k_mesh(self._structure_model.cell, spacing)
                self._estimate_label.setText(
                    self.tr("Estimated mesh: {} × {} × {}").format(*mesh)
                )
            except Exception:
                self._estimate_label.setText("")
        else:
            self._estimate_label.setText("")

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_preview(self) -> None:
        mode = self._mode_combo.currentIndex()

        if mode == 0:  # automatic
            spacing = self._kspacing_slider.value() / 100.0
            gamma = self._gamma_auto.currentIndex() == 0
            content = generate_kpoints_automatic(
                np.eye(3) if not self._structure_model else self._structure_model.cell,
                k_spacing=spacing,
                gamma_centered=gamma,
            )
        elif mode == 1:  # manual
            k1 = self._k1_spin.value()
            k2 = self._k2_spin.value()
            k3 = self._k3_spin.value()
            gamma = self._gamma_manual.currentIndex() == 0
            content = generate_kpoints_manual(k1, k2, k3, gamma_centered=gamma)
        else:  # line mode
            raw = self._band_path_edit.text().strip()
            segments = []
            for seg in raw.split("|"):
                parts = seg.split("-")
                if len(parts) == 2:
                    segments.append((parts[0].strip(), parts[1].strip()))
            if not segments:
                segments = [("G", "X")]
            npts = self._band_npoints.value()
            content = self._line_mode_content(segments, npts)

        self._preview.setPlainText(content)

    def _line_mode_content(self, segments: list[tuple[str, str]], npts: int) -> str:
        """Line-mode KPOINTS preview, or a '#'-prefixed error note.

        '#'-prefixed content is intentionally invalid VASP input and is
        rejected by _on_accept.
        """
        if self._structure_model is None or self._structure_model.n_atoms == 0:
            return self.tr("# Open a structure to compute the k-path coordinates.")
        try:
            special = get_high_symmetry_points(self._structure_model.cell)
        except Exception:
            special = {}
        if not special:
            return self.tr("# Cannot determine high-symmetry points for this cell.")
        try:
            return generate_kpoints_line_mode(segments, npts, special_points=special)
        except ValueError as e:
            return self.tr("# {}").format(e)

    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        self._update_preview()
        content = self._preview.toPlainText()
        if content.startswith("#"):
            QMessageBox.warning(
                self,
                self.tr("Cannot Generate KPOINTS"),
                content.lstrip("# "),
            )
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save KPOINTS"),
            "KPOINTS",
            "KPOINTS files (*);;All files (*)",
        )
        if filepath:
            from pathlib import Path
            Path(filepath).write_text(content, encoding="utf-8", newline="\n")
            self.accept()


