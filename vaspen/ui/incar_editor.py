"""INCAR editor panel and dialog.

The editor is split into two layers:

- ``IncarEditorPanel`` — a plain QWidget with all the editing UI
  (preset selector, tag table, preview). Embeddable, so the unified
  "Generate All Input Files" dialog reuses it as the INCAR tab.
- ``IncarEditorDialog`` — a thin QDialog shell around the panel that
  adds OK/Cancel and the save-file step. Attribute lookups not found
  on the dialog are delegated to the panel, so external code (tests,
  tooling) accessing ``dlg._table`` / ``dlg._preview`` keeps working.

Defaults follow community-standard settings.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
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

from vaspen.core.vasp_input import (
    INCAR_PRESETS,
    INCAR_SUGGESTIONS,
    INCAR_TAG_DESCRIPTIONS,
    format_incar_content,
    magmom_line,
    parse_incar_content,
)
from vaspen.ui.menu_button import MenuButton
from vaspen.utils.config import AppConfig


class IncarEditorPanel(QWidget):
    """Embeddable INCAR editor.

    Features:
    - Preset selector (SCF, Opt, Band, DOS, Optical, NEB, Custom)
    - Tag table (Tag | Value | Description) with inline editing
    - Add / remove custom tags
    - Live preview of generated INCAR text (editable; sync back via
      the button)
    """

    #: Emitted with the preset key whenever the user changes the
    #: preset selector ("custom" included).
    preset_changed = Signal(str)

    def __init__(self, structure_model=None, parent=None) -> None:
        super().__init__(parent)
        self._structure_model = structure_model
        self._tags: dict[str, str] = {}
        self._custom_tags: set[str] = set()
        self._preset_key: str = "scf"
        self._preview_dirty = False

        self._build_ui()
        self._load_preset("scf")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Top: Preset selector ──
        top_row = QHBoxLayout()

        self._calc_type_label = QLabel(self.tr("Calculation Type:"))
        top_row.addWidget(self._calc_type_label)
        self._preset_combo = MenuButton()
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

        # ── Bottom: Preview (editable — sync back via the button) ──
        preview_row = QHBoxLayout()
        self._preview_label = QLabel(self.tr("Preview:"))
        preview_row.addWidget(self._preview_label)
        preview_row.addStretch()
        self._sync_btn = QPushButton(self.tr("Sync Table from Preview"))
        self._sync_btn.setToolTip(self.tr(
            "Parse the edited preview text back into the tag table"))
        self._sync_btn.clicked.connect(self._on_sync_from_preview)
        preview_row.addWidget(self._sync_btn)
        layout.addLayout(preview_row)

        self._preview = QTextEdit()
        self._preview.setFontFamily("Consolas, monospace")
        self._preview.setMaximumHeight(200)
        self._preview.textChanged.connect(self._on_preview_text_changed)
        layout.addWidget(self._preview)

    # ------------------------------------------------------------------
    # Language switching (embedded in the non-modal generate dialog)
    # ------------------------------------------------------------------

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.LanguageChange:
            self._retranslate()
        super().changeEvent(event)

    def _retranslate(self) -> None:
        """Re-apply translatable texts; table edits and the preset survive."""
        self._calc_type_label.setText(self.tr("Calculation Type:"))
        index = self._preset_combo.currentIndex()
        self._preset_combo.blockSignals(True)  # keep the user's table edits
        self._preset_combo.clear()
        self._preset_combo.addItems([
            self.tr("SCF (Static)"),
            self.tr("Optimization"),
            self.tr("Band Structure"),
            self.tr("DOS"),
            self.tr("Optical"),
            self.tr("NEB"),
            self.tr("Custom"),
        ])
        self._preset_combo.setCurrentIndex(index)
        self._preset_combo.blockSignals(False)
        self._encut_estimate_btn.setText(self.tr("Estimate ENCUT from POTCAR"))
        self._encut_estimate_btn.setToolTip(
            self.tr("Read ENMAX from POTCAR and set ENCUT = 1.3 × ENMAX"))
        self._table.setHorizontalHeaderLabels([
            self.tr("Tag"), self.tr("Value"), self.tr("Description")
        ])
        self._add_tag_btn.setText(self.tr("Add Tag"))
        self._remove_tag_btn.setText(self.tr("Remove Selected Tag"))
        self._reset_btn.setText(self.tr("Reset to Preset"))
        self._preview_label.setText(self.tr("Preview:"))
        self._sync_btn.setText(self.tr("Sync Table from Preview"))
        self._sync_btn.setToolTip(self.tr(
            "Parse the edited preview text back into the tag table"))
        # Description column follows the new language (custom rows keep
        # their "Custom tag" label).
        preset_tags = {
            k.casefold() for k in INCAR_PRESETS.get(self._preset_key, {})
        }
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            desc_item = self._table.item(row, 2)
            if item is None or desc_item is None:
                continue
            tag = item.text()
            desc_item.setText(
                INCAR_TAG_DESCRIPTIONS.get(tag, "")
                if tag.casefold() in preset_tags or tag not in self._custom_tags
                else self.tr("Custom tag"))
        self._update_preview()

    # ------------------------------------------------------------------
    # Preset handling
    # ------------------------------------------------------------------

    @property
    def preset_key(self) -> str:
        """The currently loaded preset key."""
        return self._preset_key

    def set_preset(self, key: str) -> None:
        """Select a preset programmatically (combo follows, no signal)."""
        preset_keys = ["scf", "opt", "band", "dos", "optical", "neb", "custom"]
        if key not in preset_keys:
            return
        self._preset_combo.blockSignals(True)
        self._preset_combo.setCurrentIndex(preset_keys.index(key))
        self._preset_combo.blockSignals(False)
        if key != "custom":
            self._load_preset(key)

    def apply_tag(self, tag: str, value: str) -> None:
        """Set a tag value in the table (adds the row when absent).

        Table edits are collected first so in-flight cell edits are not
        lost. The preview is rebuilt from the table, which also resets
        the preview dirty flag (a structural change takes precedence
        over hand-edited preview text).
        """
        self._collect_tags()
        self._tags[tag] = value
        self._refresh_table()

    def _on_preset_changed(self, index: int) -> None:
        preset_keys = ["scf", "opt", "band", "dos", "optical", "neb", "custom"]
        key = preset_keys[index]
        if key != "custom":
            self._load_preset(key)
        self.preset_changed.emit(key)

    def _load_preset(self, preset_name: str) -> None:
        """Load an INCAR preset into the table."""
        self._preset_key = preset_name
        preset = dict(INCAR_PRESETS.get(preset_name, {}))
        preset.pop("_description", None)  # remove metadata key

        self._tags = {k: str(v) for k, v in preset.items()}
        # GUI-set initial moments are injected into every preset load
        # (the user can still hand-edit the rows / preview afterwards).
        if (self._structure_model is not None
                and self._structure_model.any_magmom):
            self._tags["MAGMOM"] = magmom_line(self._structure_model.magmoms)
            self._tags["ISPIN"] = "2"
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

    def _collect_tags(self) -> None:
        """Rebuild self._tags from the current table rows."""
        self._tags.clear()
        for row in range(self._table.rowCount()):
            tag_item = self._table.item(row, 0)
            val_item = self._table.item(row, 1)
            if tag_item and val_item:
                self._tags[tag_item.text()] = val_item.text()

    def _add_custom_tag(self) -> None:
        """Add a new custom tag row."""
        # Focus the pending row instead of stacking duplicate NEW_TAG rows
        for r in range(self._table.rowCount()):
            item = self._table.item(r, 0)
            if item and item.text() == "NEW_TAG":
                self._table.setCurrentCell(r, 1)
                return

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
        self._collect_tags()

        # Same renderer as file generation — preview == generated INCAR
        self._preview.blockSignals(True)
        self._preview.setPlainText(format_incar_content(
            self._tags, suggestions=INCAR_SUGGESTIONS.get(self._preset_key)
        ))
        self._preview.blockSignals(False)
        self._preview_dirty = False  # programmatic rebuild is not a user edit

    def _on_preview_text_changed(self) -> None:
        """The user typed in the preview — it becomes the source of truth."""
        self._preview_dirty = True

    def _on_sync_from_preview(self) -> None:
        """Parse the edited preview text back into the tag table."""
        tags, problems = parse_incar_content(self._preview.toPlainText())
        if problems:
            lines = [
                self.tr("line {}: {}").format(no, msg)
                for no, _kind, msg in problems
            ]
            QMessageBox.warning(
                self,
                self.tr("Cannot Sync from Preview"),
                self.tr("Fix the INCAR preview first:\n{}").format("\n".join(lines)),
            )
            return
        self._tags = tags
        preset_tags = INCAR_PRESETS.get(self._preset_key, {})
        self._custom_tags = {
            t for t in tags
            if t.casefold() not in {k.casefold() for k in preset_tags}
        }
        self._refresh_table()  # preview is re-aligned/normalized

    def _validate_tags(self) -> list[str]:
        """Tag names that appear more than once (case-insensitive)."""
        seen: dict[str, str] = {}
        duplicates: list[str] = []
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if not item:
                continue
            name = item.text().strip()
            if not name:
                continue
            key = name.casefold()
            if key in seen:
                if seen[key] is not None:
                    duplicates.append(seen[key])
                    seen[key] = None
            else:
                seen[key] = name
        return duplicates

    def content(self) -> str:
        """Validated INCAR text.

        The edited preview is the source of truth; a clean preview is
        first refreshed from the table (pending cell edits committed).

        Raises:
            ValueError: With a user-facing message when the content has
                duplicate tags, tags without a value, or malformed
                lines.
        """
        if not self._preview_dirty:
            duplicates = self._validate_tags()
            if duplicates:
                raise ValueError(
                    self.tr("Duplicate INCAR tags: {}. Remove or rename "
                            "the extra rows.").format(", ".join(duplicates)))
            self._update_preview()

        text = self._preview.toPlainText()
        _tags, problems = parse_incar_content(text)
        duplicates = [msg for _no, kind, msg in problems if kind == "duplicate"]
        if duplicates:
            raise ValueError(
                self.tr("Duplicate INCAR tags: {}. Remove or rename the "
                        "extra rows.").format(", ".join(duplicates)))
        empty_tags = [tag for tag, val in _tags.items() if not val.strip()]
        if empty_tags:
            raise ValueError(
                self.tr("Tags without a value: {}").format(", ".join(empty_tags)))
        malformed = [msg for _no, kind, msg in problems if kind == "malformed"]
        if malformed:
            raise ValueError(
                self.tr("These lines are not valid INCAR tag lines:\n{}")
                .format("\n".join(malformed)))
        return text

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
            import warnings
            from pymatgen.io.vasp import Potcar

            # pymatgen's POTCAR database does not know variant names like
            # Ba_sv — the warning is noise; ENMAX is read from the header.
            warnings.filterwarnings(
                "ignore", message="POTCAR data with symbol .* is not known to pymatgen"
            )

            config = AppConfig()
            potcar_path = config.potcar_library_path
            if not potcar_path:
                raise ValueError("POTCAR library not configured")

            from vaspen.core.vasp_input import (
                get_potcar_recommendation,
                resolve_potcar_dir,
            )

            elements = self._structure_model.unique_symbols
            library = Path(potcar_path)
            potcar_dir = resolve_potcar_dir(library, "PBE")

            max_encut = 400
            for el in elements:
                variant = get_potcar_recommendation(el, "PBE")
                potcar_file = potcar_dir / variant / "POTCAR"
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


class IncarEditorDialog(QDialog):
    """Dialog shell around IncarEditorPanel (OK saves to file, Cancel)."""

    def __init__(self, structure_model=None, parent=None) -> None:
        super().__init__(parent)
        self._panel = IncarEditorPanel(structure_model, self)

        self.setWindowTitle(self.tr("Generate INCAR"))
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        layout.addWidget(self._panel)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the embedded panel.

        Keeps external attribute access (tests, tooling) working after
        the editor moved from the dialog into the panel.
        """
        panel = self.__dict__.get("_panel")
        if panel is not None:
            try:
                return getattr(panel, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}")

    def _on_accept(self) -> None:
        # The edited preview is the source of truth; a clean preview is
        # first refreshed from the table (pending cell edits committed).
        panel = self._panel
        if not panel._preview_dirty:
            duplicates = panel._validate_tags()
            if duplicates:
                QMessageBox.warning(
                    self,
                    self.tr("Duplicate Tag"),
                    self.tr("Duplicate INCAR tags: {}. Remove or rename the extra rows.")
                    .format(", ".join(duplicates)),
                )
                return
            panel._update_preview()

        content = panel._preview.toPlainText()
        _tags, problems = parse_incar_content(content)
        duplicates = [msg for _no, kind, msg in problems if kind == "duplicate"]
        if duplicates:
            QMessageBox.warning(
                self,
                self.tr("Duplicate Tag"),
                self.tr("Duplicate INCAR tags: {}. Remove or rename the extra rows.")
                .format(", ".join(duplicates)),
            )
            return
        empty_tags = [tag for tag, val in _tags.items() if not val.strip()]
        if empty_tags:
            QMessageBox.warning(
                self,
                self.tr("Invalid INCAR Lines"),
                self.tr("Tags without a value: {}").format(", ".join(empty_tags)),
            )
            return
        malformed = [
            self.tr("line {}: {}").format(no, msg)
            for no, kind, msg in problems if kind == "malformed"
        ]
        if malformed:
            reply = QMessageBox.question(
                self,
                self.tr("Invalid INCAR Lines"),
                self.tr("These lines are not valid INCAR tag lines:\n{}\n\nSave anyway?")
                .format("\n".join(malformed)),
            )
            if reply != QMessageBox.Yes:
                return

        config = AppConfig()
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save INCAR"),
            str(Path(config.last_directory) / "INCAR"),
            "INCAR files (*);;All files (*)",
        )
        if filepath:
            Path(filepath).write_text(content, encoding="utf-8", newline="\n")
            config.last_directory = str(Path(filepath).parent)
            self.accept()
