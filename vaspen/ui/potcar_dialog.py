"""POTCAR generation panel and dialog.

Split into ``PotcarPanel`` (embeddable widget, reused as the POTCAR tab
of the unified input-file dialog) and ``PotcarDialog`` (a thin QDialog
shell that adds OK/Cancel and the save-file step; unknown attribute
lookups delegate to the panel).

Allows per-element selection of pseudopotential variants, then
concatenates the selected POTCAR files.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.vasp_input import (
    POTCAR_FUNCTIONAL_VERSIONS,
    available_variants,
    generate_potcar,
    get_potcar_recommendation,
    resolve_potcar_dir,
)
from vaspen.ui.menu_button import MenuButton
from vaspen.utils.config import AppConfig


class PotcarPanel(QWidget):
    """Embeddable POTCAR variant selector + concatenation preview."""

    def __init__(self, structure_model=None, parent=None) -> None:
        super().__init__(parent)
        self._structure_model = structure_model
        self._config = AppConfig()
        self._element_combos: dict[str, MenuButton] = {}
        self._potcar_content: str = ""
        self._paths_used: list[str] = []

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Functional ──
        func_row = QHBoxLayout()
        self._functional_label = QLabel(self.tr("Functional:"))
        func_row.addWidget(self._functional_label)
        self._functional_combo = MenuButton()
        self._functional_combo.addItems(list(POTCAR_FUNCTIONAL_VERSIONS.keys()))
        self._functional_combo.currentTextChanged.connect(self._refresh_elements)
        func_row.addWidget(self._functional_combo)
        func_row.addStretch()
        layout.addLayout(func_row)

        # ── Library path ──
        path_row = QHBoxLayout()
        self._library_label = QLabel(self.tr("POTCAR Library:"))
        path_row.addWidget(self._library_label)
        self._path_edit = QLineEdit(self._config.potcar_library_path)
        self._path_edit.setPlaceholderText(
            self.tr("Path to pseudopotential library root (e.g. /path/to/potcar)")
        )
        path_row.addWidget(self._path_edit, 1)
        self._browse_btn = QPushButton(self.tr("Browse..."))
        self._browse_btn.clicked.connect(self._browse_library)
        path_row.addWidget(self._browse_btn)
        layout.addLayout(path_row)

        # ── Elements ──
        self._elements_group = QGroupBox(self.tr("Elements (in POSCAR order)"))
        self._elements_layout = QFormLayout()
        self._elements_group.setLayout(self._elements_layout)
        layout.addWidget(self._elements_group)

        self._refresh_btn = QPushButton(self.tr("Refresh Elements from Structure"))
        self._refresh_btn.clicked.connect(self._refresh_elements)
        layout.addWidget(self._refresh_btn)

        # ── Preview ──
        self._preview_label = QLabel(self.tr(
            "POTCAR Preview (shows which files will be concatenated):"))
        layout.addWidget(self._preview_label)
        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setFontFamily("Consolas, monospace")
        self._preview.setMaximumHeight(200)
        layout.addWidget(self._preview, 1)

        # ── Generate button ──
        self._gen_btn = QPushButton(self.tr("Generate POTCAR Preview"))
        self._gen_btn.clicked.connect(self._on_generate_clicked)
        layout.addWidget(self._gen_btn)

        # Populate
        self._refresh_elements()

    # ------------------------------------------------------------------
    # Language switching (embedded in the non-modal generate dialog)
    # ------------------------------------------------------------------

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.LanguageChange:
            self._retranslate()
        super().changeEvent(event)

    def _retranslate(self) -> None:
        self._functional_label.setText(self.tr("Functional:"))
        self._library_label.setText(self.tr("POTCAR Library:"))
        self._path_edit.setPlaceholderText(
            self.tr("Path to pseudopotential library root (e.g. /path/to/potcar)"))
        self._browse_btn.setText(self.tr("Browse..."))
        self._elements_group.setTitle(self.tr("Elements (in POSCAR order)"))
        self._refresh_btn.setText(self.tr("Refresh Elements from Structure"))
        self._preview_label.setText(self.tr(
            "POTCAR Preview (shows which files will be concatenated):"))
        self._gen_btn.setText(self.tr("Generate POTCAR Preview"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_content(self) -> str:
        """Concatenate POTCAR text from the current selections.

        Returns "" when no library is configured or no elements are
        known (POTCAR is simply skipped in the unified flow).

        Raises:
            FileNotFoundError: When a selected POTCAR file is missing.
        """
        library = self._path_edit.text()
        if not library or not self._element_combos:
            return ""

        functional = self._functional_combo.currentText()
        potcar_dir = resolve_potcar_dir(Path(library), functional)
        contents: list[str] = []
        self._paths_used = []

        for el, combo in self._element_combos.items():
            variant = combo.currentText()
            potcar_path = potcar_dir / variant / "POTCAR"
            if not potcar_path.exists():
                # Fallback: try element name
                potcar_path = potcar_dir / el / "POTCAR"
            if not potcar_path.exists():
                raise FileNotFoundError(
                    f"POTCAR for {el} not found.\n"
                    f"Tried: {potcar_dir / variant / 'POTCAR'}\n"
                    f"       {potcar_dir / el / 'POTCAR'}"
                )
            with open(potcar_path, "r") as f:
                contents.append(f.read())
            self._paths_used.append(str(potcar_path))

        return "".join(contents)

    def _on_generate_clicked(self) -> None:
        """Generate-preview button: concatenate or explain why not."""
        library = self._path_edit.text()
        if not library:
            QMessageBox.warning(
                self,
                self.tr("Missing Library Path"),
                self.tr("Please set the POTCAR library path first."),
            )
            return
        if not self._element_combos:
            QMessageBox.warning(
                self,
                self.tr("No Elements"),
                self.tr("Load a structure to detect elements."),
            )
            return
        # Regenerate from scratch — a failed run must not leave a stale
        # concatenation from an earlier selection behind.
        self._potcar_content = ""
        try:
            self._potcar_content = self.generate_content()
        except FileNotFoundError as e:
            QMessageBox.critical(self, self.tr("POTCAR Not Found"), str(e))
            return
        self._preview.setPlainText(
            self.tr("POTCAR generated successfully! {} elements, {:,} bytes.\n\n"
                    "Files used:\n{}").format(
                len(self._element_combos),
                len(self._potcar_content),
                "\n".join(self._paths_used),
            )
        )

    # ------------------------------------------------------------------

    def _browse_library(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select POTCAR Library Root"),
            self._path_edit.text() or str(Path.home()),
        )
        if directory:
            self._path_edit.setText(directory)
            self._config.potcar_library_path = directory

    def _refresh_elements(self) -> None:
        """Rebuild element combo boxes from current structure."""
        # Clear
        while self._elements_layout.rowCount() > 0:
            self._elements_layout.removeRow(0)
        self._element_combos.clear()

        if self._structure_model is None or self._structure_model.n_atoms == 0:
            placeholder = QLabel(self.tr("No structure loaded. Elements cannot be detected."))
            placeholder.setWordWrap(True)
            self._elements_layout.addRow(placeholder)
            return

        functional = self._functional_combo.currentText()
        elements = self._structure_model.unique_symbols
        library = self._path_edit.text()

        for el in elements:
            combo = MenuButton()
            # Discover what the library actually offers (all wiki
            # variants: _d/_sv/_pv/_s/_h/_GW/_AE/_2/_3/fractional...);
            # fall back to the recommendation + common suffixes when
            # no library is configured yet.
            variants = available_variants(library, functional, el) if library else []
            if not variants:
                recommendation = get_potcar_recommendation(el, functional)
                variants = [recommendation]
                if el not in variants:
                    variants.append(el)
                for suffix in ("_d", "_sv", "_pv", "_h", "_s"):
                    v = el + suffix
                    if v not in variants:
                        variants.append(v)
            combo.addItems(variants)
            # Pre-select the wiki-recommended default when available
            idx = combo.findText(get_potcar_recommendation(el, functional))
            if idx >= 0:
                combo.setCurrentIndex(idx)
            self._elements_layout.addRow(f"{el}:", combo)
            self._element_combos[el] = combo
            combo.currentIndexChanged.connect(self._update_preview)

        self._update_preview()

    def _update_preview(self) -> None:
        library = self._path_edit.text()
        functional = self._functional_combo.currentText()

        if not library or not self._element_combos:
            self._preview.setPlainText("")
            return

        version_dir = resolve_potcar_dir(Path(library), functional)
        lines = [f"# POTCAR for functional = {functional} ({version_dir.name})"]
        lines.append(f"# Library root: {library}")
        lines.append("")

        for el, combo in self._element_combos.items():
            variant = combo.currentText()
            expected_path = version_dir / variant / "POTCAR"
            exists = "✓" if expected_path.exists() else "✗"
            lines.append(f"{exists} {el}: {variant}  →  {expected_path}")

        self._preview.setPlainText("\n".join(lines))


class PotcarDialog(QDialog):
    """Dialog shell around PotcarPanel (OK saves to file, Cancel)."""

    def __init__(self, structure_model=None, parent=None) -> None:
        super().__init__(parent)
        self._panel = PotcarPanel(structure_model, self)

        self.setWindowTitle(self.tr("Generate POTCAR"))
        self.resize(600, 400)
        layout = QVBoxLayout(self)
        layout.addWidget(self._panel)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the embedded panel."""
        panel = self.__dict__.get("_panel")
        if panel is not None:
            try:
                return getattr(panel, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}")

    def _on_accept(self) -> None:
        panel = self._panel
        # Always regenerate from the CURRENT selections — the cached
        # content goes stale when the variant/functional/library is
        # changed after a preview (those only refresh the preview text).
        panel._on_generate_clicked()

        if panel._potcar_content:
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                self.tr("Save POTCAR"),
                "POTCAR",
                "POTCAR files (*);;All files (*)",
            )
            if filepath:
                Path(filepath).write_text(
                    panel._potcar_content, encoding="utf-8", newline="\n")
                self.accept()
