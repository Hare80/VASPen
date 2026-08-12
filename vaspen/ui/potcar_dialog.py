"""POTCAR generation dialog.

Allows per-element selection of pseudopotential variants,
then concatenates the selected POTCAR files.
"""

from __future__ import annotations

from pathlib import Path

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
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.vasp_input import (
    POTCAR_FUNCTIONAL_VERSIONS,
    POTCAR_SPECIAL_RECOMMENDATIONS,
    generate_potcar,
    get_potcar_recommendation,
)
from vaspen.utils.config import AppConfig


class PotcarDialog(QDialog):
    """Dialog for selecting POTCAR variants and generating POTCAR."""

    def __init__(self, structure_model=None, parent=None) -> None:
        super().__init__(parent)
        self._structure_model = structure_model
        self._config = AppConfig()
        self._element_combos: dict[str, QComboBox] = {}
        self._potcar_content: str = ""
        self._paths_used: list[str] = []

        self.setWindowTitle(self.tr("Generate POTCAR"))
        self.resize(600, 400)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Functional ──
        func_row = QHBoxLayout()
        func_row.addWidget(QLabel(self.tr("Functional:")))
        self._functional_combo = QComboBox()
        self._functional_combo.addItems(list(POTCAR_FUNCTIONAL_VERSIONS.keys()))
        self._functional_combo.currentTextChanged.connect(self._refresh_elements)
        func_row.addWidget(self._functional_combo)
        func_row.addStretch()
        layout.addLayout(func_row)

        # ── Library path ──
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel(self.tr("POTCAR Library:")))
        self._path_edit = QLineEdit(self._config.potcar_library_path)
        self._path_edit.setPlaceholderText(
            self.tr("Path to pseudopotential library root (e.g. /path/to/potcar)")
        )
        path_row.addWidget(self._path_edit, 1)
        browse_btn = QPushButton(self.tr("Browse..."))
        browse_btn.clicked.connect(self._browse_library)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        # ── Elements ──
        self._elements_group = QGroupBox(self.tr("Elements (in POSCAR order)"))
        self._elements_layout = QFormLayout()
        self._elements_group.setLayout(self._elements_layout)
        layout.addWidget(self._elements_group)

        refresh_btn = QPushButton(self.tr("Refresh Elements from Structure"))
        refresh_btn.clicked.connect(self._refresh_elements)
        layout.addWidget(refresh_btn)

        # ── Preview ──
        layout.addWidget(QLabel(self.tr("POTCAR Preview (shows which files will be concatenated):")))
        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setFontFamily("Consolas, monospace")
        self._preview.setMaximumHeight(200)
        layout.addWidget(self._preview, 1)

        # ── Generate button ──
        gen_btn = QPushButton(self.tr("Generate POTCAR Preview"))
        gen_btn.clicked.connect(self._generate)
        layout.addWidget(gen_btn)

        # ── Dialog buttons ──
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Populate
        self._refresh_elements()

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

        for el in elements:
            combo = QComboBox()
            # Available variants for this element
            recommendation = get_potcar_recommendation(el, functional)
            variants = [el, recommendation] if recommendation != el else [el]
            # Add common suffixes
            for suffix in ["", "_d", "_sv", "_pv"]:
                variant = el + suffix
                if variant not in variants:
                    variants.append(variant)
            combo.addItems(variants)
            # Select recommended
            idx = combo.findText(recommendation)
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

        version = POTCAR_FUNCTIONAL_VERSIONS[functional]
        lines = [f"# POTCAR for functional = {functional} ({version})"]
        lines.append(f"# Library root: {library}")
        lines.append("")

        for el, combo in self._element_combos.items():
            variant = combo.currentText()
            expected_path = Path(library) / version / variant / "POTCAR"
            exists = "✓" if expected_path.exists() else "✗"
            lines.append(f"{exists} {el}: {variant}  →  {expected_path}")

        self._preview.setPlainText("\n".join(lines))

    def _generate(self) -> None:
        library = self._path_edit.text()
        functional = self._functional_combo.currentText()

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

        # Map elements to selected variants (respecting user choice)
        # Note: generate_potcar uses automatic recommendation;
        # we use a manual file-by-file concatenation here.
        contents: list[str] = []
        self._paths_used = []
        version = POTCAR_FUNCTIONAL_VERSIONS[functional]
        potcar_dir = Path(library) / version

        for el, combo in self._element_combos.items():
            variant = combo.currentText()
            potcar_path = potcar_dir / variant / "POTCAR"
            if not potcar_path.exists():
                # Fallback: try element name
                potcar_path = potcar_dir / el / "POTCAR"
            if not potcar_path.exists():
                QMessageBox.critical(
                    self,
                    self.tr("POTCAR Not Found"),
                    self.tr("POTCAR for {el} not found.\n"
                            "Tried: {p1}\n{p2}").format(
                        el=el,
                        p1=potcar_dir / variant / "POTCAR",
                        p2=potcar_dir / el / "POTCAR",
                    ),
                )
                return
            with open(potcar_path, "r") as f:
                contents.append(f.read())
            self._paths_used.append(str(potcar_path))

        self._potcar_content = "".join(contents)
        self._preview.setPlainText(
            self.tr("POTCAR generated successfully! {} elements, {:,} bytes.\n\n"
                    "Files used:\n{}").format(
                len(self._element_combos),
                len(self._potcar_content),
                "\n".join(self._paths_used),
            )
        )

    def _on_accept(self) -> None:
        if not self._potcar_content:
            self._generate()

        if self._potcar_content:
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                self.tr("Save POTCAR"),
                "POTCAR",
                "POTCAR files (*);;All files (*)",
            )
            if filepath:
                Path(filepath).write_text(self._potcar_content)
                self.accept()
