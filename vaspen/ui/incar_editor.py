"""INCAR editor dialog.

Provides a table-based INCAR tag editor with presets for common
calculation types (SCF, Optimization, Band, DOS, Optical, NEB).
Defaults follow vaspkit recommendations.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.vasp_input import INCAR_PRESETS, INCAR_TAG_DESCRIPTIONS


class IncarEditorDialog(QDialog):
    """Dialog for editing and generating INCAR content.

    Features:
    - Preset selector (SCF, Opt, Band, DOS, Optical, NEB)
    - Tag table (Tag | Value | Description) with inline editing
    - Add / remove custom tags
    - Live preview of generated INCAR text
    - Save as defaults
    """

    def __init__(self, structure_model=None, parent=None) -> None:
        super().__init__(parent)
        self._structure_model = structure_model
        self._tags: dict[str, str] = {}
        self._custom_tags: set[str] = set()

        self.setWindowTitle(self.tr("Generate INCAR"))
        self.resize(800, 600)
        self._build_ui()
        self._load_preset("scf")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Top: Preset selector ──
        top_row = QHBoxLayout()

        top_row.addWidget(QLabel(self.tr("Calculation Type:")))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems([
            self.tr("SCF (Static)"),
            self.tr("Optimization"),
            self.tr("Band Structure"),
            self.tr("DOS"),
            self.tr("Optical"),
            self.tr("NEB"),
            self.tr("Custom"),
        ])
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        top_row.addWidget(self._preset_combo)
        top_row.addStretch()

        self._encut_estimate_btn = QPushButton(self.tr("Estimate ENCUT from POTCAR"))
        self._encut_estimate_btn.setToolTip(self.tr("Read ENMAX from POTCAR and set ENCUT = 1.3 × ENMAX"))
        self._encut_estimate_btn.clicked.connect(self._estimate_encut)
        top_row.addWidget(self._encut_estimate_btn)

        layout.addLayout(top_row)

        # ── Middle: Tag table ──
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels([
            self.tr("Tag"), self.tr("Value"), self.tr("Description")
        ])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.itemChanged.connect(self._update_preview)
        layout.addWidget(self._table, 1)

        # Tag manipulation buttons
        tag_btn_row = QHBoxLayout()
        self._add_tag_btn = QPushButton(self.tr("Add Tag"))
        self._add_tag_btn.clicked.connect(self._add_custom_tag)
        self._remove_tag_btn = QPushButton(self.tr("Remove Selected Tag"))
        self._remove_tag_btn.clicked.connect(self._remove_selected_tag)
        self._reset_btn = QPushButton(self.tr("Reset to Preset"))
        self._reset_btn.clicked.connect(self._reset_preset)
        tag_btn_row.addWidget(self._add_tag_btn)
        tag_btn_row.addWidget(self._remove_tag_btn)
        tag_btn_row.addWidget(self._reset_btn)
        tag_btn_row.addStretch()
        layout.addLayout(tag_btn_row)

        # ── Bottom: Preview ──
        preview_label = QLabel(self.tr("Preview:"))
        layout.addWidget(preview_label)

        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setFontFamily("Consolas, monospace")
        self._preview.setMaximumHeight(200)
        layout.addWidget(self._preview)

        # ── Dialog buttons ──
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Preset handling
    # ------------------------------------------------------------------

    def _on_preset_changed(self, index: int) -> None:
        preset_keys = ["scf", "opt", "band", "dos", "optical", "neb", "custom"]
        key = preset_keys[index]
        if key != "custom":
            self._load_preset(key)

    def _load_preset(self, preset_name: str) -> None:
        """Load an INCAR preset into the table."""
        preset = dict(INCAR_PRESETS.get(preset_name, {}))
        preset.pop("_description", None)  # remove metadata key

        self._tags = {k: str(v) for k, v in preset.items()}
        self._custom_tags = set()
        self._refresh_table()

    def _reset_preset(self) -> None:
        index = self._preset_combo.currentIndex()
        preset_keys = ["scf", "opt", "band", "dos", "optical", "neb", "custom"]
        key = preset_keys[index]
        if key != "custom":
            self._load_preset(key)

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------

    def _refresh_table(self) -> None:
        """Rebuild the table from self._tags."""
        self._table.blockSignals(True)
        self._table.setRowCount(0)

        for tag, value in self._tags.items():
            row = self._table.rowCount()
            self._table.insertRow(row)

            # Tag (non-editable for preset tags)
            tag_item = QTableWidgetItem(tag)
            tag_item.setFlags(tag_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 0, tag_item)

            # Value (editable)
            val_item = QTableWidgetItem(value)
            self._table.setItem(row, 1, val_item)

            # Description
            desc = INCAR_TAG_DESCRIPTIONS.get(tag, "")
            desc_item = QTableWidgetItem(desc)
            desc_item.setFlags(desc_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 2, desc_item)

        self._table.blockSignals(False)
        self._update_preview()

    def _add_custom_tag(self) -> None:
        """Add a new custom tag row."""
        row = self._table.rowCount()
        self._table.insertRow(row)

        tag_item = QTableWidgetItem("NEW_TAG")
        self._table.setItem(row, 0, tag_item)
        self._table.setItem(row, 1, QTableWidgetItem(""))
        self._table.setItem(row, 2, QTableWidgetItem(self.tr("Custom tag")))
        self._custom_tags.add("NEW_TAG")

    def _remove_selected_tag(self) -> None:
        rows = set(i.row() for i in self._table.selectedItems())
        for row in sorted(rows, reverse=True):
            tag = self._table.item(row, 0).text()
            self._tags.pop(tag, None)
            self._custom_tags.discard(tag)
            self._table.removeRow(row)
        self._update_preview()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_preview(self) -> None:
        """Rebuild preview from current table state."""
        self._tags.clear()
        for row in range(self._table.rowCount()):
            tag_item = self._table.item(row, 0)
            val_item = self._table.item(row, 1)
            if tag_item and val_item:
                self._tags[tag_item.text()] = val_item.text()

        lines = [f"{tag} = {val}" for tag, val in self._tags.items()]
        self._preview.setPlainText("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # ENCUT estimation
    # ------------------------------------------------------------------

    def _estimate_encut(self) -> None:
        """Estimate ENCUT from structure elements (1.3 × ENMAX)."""
        if self._structure_model is None or self._structure_model.n_atoms == 0:
            QMessageBox.information(
                self,
                self.tr("No Structure"),
                self.tr("Load a structure first to estimate ENCUT."),
            )
            return

        # Try to get ENMAX from pymatgen POTCAR data
        try:
            from pymatgen.io.vasp import Potcar

            config = __import__('vaspen.utils.config', fromlist=['AppConfig']).AppConfig()
            potcar_path = config.potcar_library_path
            if not potcar_path:
                raise ValueError("POTCAR library not configured")

            from pathlib import Path
            from vaspen.core.vasp_input import get_potcar_recommendation, POTCAR_FUNCTIONAL_VERSIONS

            elements = self._structure_model.unique_symbols
            library = Path(potcar_path)
            version = POTCAR_FUNCTIONAL_VERSIONS["PBE"]

            max_encut = 400
            for el in elements:
                variant = get_potcar_recommendation(el, "PBE")
                potcar_file = library / version / variant / "POTCAR"
                if potcar_file.exists():
                    p = Potcar.from_file(str(potcar_file))
                    if p:
                        enmax = max(s.enmax for s in p)
                        max_encut = max(max_encut, int(enmax * 1.3))

            # Update table
            for row in range(self._table.rowCount()):
                if self._table.item(row, 0).text() == "ENCUT":
                    self._table.item(row, 1).setText(str(max_encut))
                    break
            else:
                # ENCUT not in table yet
                row = self._table.rowCount()
                self._table.insertRow(row)
                self._table.setItem(row, 0, QTableWidgetItem("ENCUT"))
                self._table.setItem(row, 1, QTableWidgetItem(str(max_encut)))
                desc = INCAR_TAG_DESCRIPTIONS.get("ENCUT", "")
                self._table.setItem(row, 2, QTableWidgetItem(desc))

            self._update_preview()
            QMessageBox.information(
                self,
                self.tr("ENCUT Estimated"),
                self.tr("ENCUT = {} eV (1.3 × ENMAX).\n"
                        "Please verify this value for your calculation.").format(max_encut),
            )
        except Exception as e:
            QMessageBox.warning(
                self,
                self.tr("Estimation Failed"),
                self.tr("Could not estimate ENCUT:\n{}\n\n"
                        "Set POTCAR library path in Edit → Preferences.").format(str(e)),
            )

    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        self._update_preview()
        content = self._preview.toPlainText()

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save INCAR"),
            "INCAR",
            "INCAR files (*);;All files (*)",
        )
        if filepath:
            from pathlib import Path
            Path(filepath).write_text(content, encoding="utf-8", newline="\n")
            self.accept()
