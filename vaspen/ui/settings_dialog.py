"""Settings / Preferences dialog."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vaspen.ui.menu_button import MenuButton
from vaspen.utils.config import AppConfig


class SettingsDialog(QDialog):
    """Application preferences dialog."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._config = AppConfig()

        self.setWindowTitle(self.tr("Preferences"))
        self.resize(500, 350)
        self._build_ui()
        self._load_settings()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── General ──
        gen_group = QGroupBox(self.tr("General"))
        gen_form = QFormLayout()

        self._language_combo = MenuButton()
        self._language_combo.addItems(["English", "中文"])
        gen_form.addRow(self.tr("Language:"), self._language_combo)

        self._theme_combo = MenuButton()
        self._theme_combo.addItems([self.tr("Light"), self.tr("Dark")])
        gen_form.addRow(self.tr("Theme:"), self._theme_combo)

        self._calc_type_combo = MenuButton()
        self._calc_type_combo.addItems([
            self.tr("SCF (Static)"),
            self.tr("Optimization"),
            self.tr("Band Structure"),
            self.tr("DOS"),
            self.tr("Optical"),
            self.tr("NEB"),
        ])
        gen_form.addRow(self.tr("Default Calculation Type:"), self._calc_type_combo)

        self._poscar_coords_combo = MenuButton()
        self._poscar_coords_combo.addItems([
            self.tr("Fractional (Direct)"),
            self.tr("Cartesian"),
        ])
        gen_form.addRow(self.tr("POSCAR Coordinates:"), self._poscar_coords_combo)

        coords_note = QLabel(self.tr(
            "Fractional (Direct) is the VASP convention (official "
            "CONTCAR files use it); VASP accepts both forms."
        ))
        coords_note.setWordWrap(True)
        gen_form.addRow("", coords_note)

        gen_group.setLayout(gen_form)
        layout.addWidget(gen_group)

        # ── POTCAR ──
        potcar_group = QGroupBox(self.tr("Pseudopotential Library"))
        potcar_form = QFormLayout()

        path_row = QHBoxLayout()
        self._potcar_path_edit = QLineEdit()
        self._potcar_path_edit.setPlaceholderText(
            self.tr("e.g. /home/user/vasp/potcar  or  C:\\vasp\\potcar")
        )
        path_row.addWidget(self._potcar_path_edit, 1)
        browse_btn = QPushButton(self.tr("Browse..."))
        browse_btn.clicked.connect(self._browse_potcar)
        path_row.addWidget(browse_btn)
        potcar_form.addRow(self.tr("Library Root:"), path_row)

        potcar_note = QLabel(self.tr(
            "The library should have subdirectories like:\n"
            "  PBE.54/Fe/POTCAR, PBE.54/O/POTCAR, ...\n"
            "This is the standard VASP pseudopotential directory layout."
        ))
        potcar_note.setWordWrap(True)
        potcar_form.addRow("", potcar_note)

        potcar_group.setLayout(potcar_form)
        layout.addWidget(potcar_group)

        # ── MCP server ──
        mcp_group = QGroupBox(self.tr("MCP Server (AI clients)"))
        mcp_form = QFormLayout()

        self._mcp_run_python_check = QCheckBox(self.tr(
            "Allow code execution (run_python)"))
        mcp_form.addRow(self.tr("AI code execution:"), self._mcp_run_python_check)

        mcp_note = QLabel(self.tr(
            "run_python lets an AI client run Python against the "
            "loaded structure (same privileges as VASPen itself). "
            "Disable it to restrict the MCP server to the fixed tools."
        ))
        mcp_note.setWordWrap(True)
        mcp_form.addRow("", mcp_note)

        mcp_group.setLayout(mcp_form)
        layout.addWidget(mcp_group)

        # ── Buttons ──
        layout.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_potcar(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select POTCAR Library Root"),
            self._potcar_path_edit.text() or str(Path.home()),
        )
        if directory:
            self._potcar_path_edit.setText(directory)

    def _load_settings(self) -> None:
        lang = self._config.language
        self._language_combo.setCurrentIndex(0 if lang == "en" else 1)

        self._theme_combo.setCurrentIndex(0 if self._config.theme == "light" else 1)

        calc_map = {"scf": 0, "opt": 1, "band": 2, "dos": 3, "optical": 4, "neb": 5}
        self._calc_type_combo.setCurrentIndex(
            calc_map.get(self._config.default_calc_type, 0)
        )

        self._poscar_coords_combo.setCurrentIndex(
            0 if self._config.poscar_coords_direct else 1
        )

        self._potcar_path_edit.setText(self._config.potcar_library_path)
        self._mcp_run_python_check.setChecked(self._config.mcp_allow_run_python)

    def _on_accept(self) -> None:
        self._config.language = "en" if self._language_combo.currentIndex() == 0 else "zh"
        self._config.theme = ("light", "dark")[self._theme_combo.currentIndex()]
        calc_keys = ["scf", "opt", "band", "dos", "optical", "neb"]
        self._config.default_calc_type = calc_keys[self._calc_type_combo.currentIndex()]
        self._config.poscar_coords_direct = (
            self._poscar_coords_combo.currentIndex() == 0
        )
        self._config.potcar_library_path = self._potcar_path_edit.text()
        self._config.mcp_allow_run_python = self._mcp_run_python_check.isChecked()
        self._config.sync()
        self.accept()
