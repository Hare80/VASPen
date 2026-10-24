"""Run editor — edit a project run's input files (§7.15 P1.7).

Embeds the SAME editor panels as the Generate All dialog (§7.7's "one
implementation per file type"): the run's existing INCAR/KPOINTS load
into the panels and Apply writes them back verbatim (WYSIWYG — no
parameter round-trip). The provenance card gains an edit-history entry;
raw text stays read-only in the viewer (edits bypassing validation and
provenance are deliberately not offered).

NEB runs show only INCAR/KPOINTS/POTCAR — frames are the structures
(open them from the project panel's file tree); regenerating a NEB run
never touches frame directories.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.project import Project, RUN_STATUSES
from vaspen.core.vasp_input import parse_incar_content
from vaspen.ui.incar_editor import IncarEditorPanel
from vaspen.ui.kpoints_editor import KpointsEditorPanel
from vaspen.utils.config import AppConfig
from vaspen.utils.logger import logger


class RunEditorDialog(QDialog):
    """View and edit one project run's input files."""

    def __init__(self, project: Project, run_id: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project = project
        self._run_id = run_id
        self._entry = dict(project.run_entry(run_id))
        self._is_neb = self._entry.get("type") == "neb"
        # the registered structure powers panel estimates (k-mesh etc.)
        self._structure_model = None
        try:
            from vaspen.core.structure import StructureModel
            self._structure_model = StructureModel()
            self._structure_model.load(project.structure_path(
                self._entry.get("structure", "")))
        except Exception:
            self._structure_model = None

        self.setWindowTitle(self.tr("Run: {}").format(run_id))
        self.resize(760, 560)
        self._build_ui()
        self._load()

    # -- UI --------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info = QGroupBox(self.tr("Run"))
        form = QFormLayout(info)
        self._type_label = QLabel(self._entry.get("type", ""))
        self._structure_label = QLabel(self._entry.get("structure", ""))
        form.addRow(self.tr("Type:"), self._type_label)
        form.addRow(self.tr("Structure:"), self._structure_label)
        self._status_combo = QComboBox()
        for status in RUN_STATUSES:
            self._status_combo.addItem(status)  # canonical manifest values
        form.addRow(self.tr("Status:"), self._status_combo)
        layout.addWidget(info)

        self._tabs = QTabWidget()
        self._incar_panel = IncarEditorPanel(
            structure_model=self._structure_model)
        self._tabs.addTab(self._incar_panel, self.tr("INCAR"))
        self._kpoints_panel = KpointsEditorPanel(
            structure_model=self._structure_model)
        self._kpoints_note = QLabel("")
        self._kpoints_note.setWordWrap(True)
        kpoints_page = QWidget()
        kpoints_layout = QVBoxLayout(kpoints_page)
        kpoints_layout.setContentsMargins(0, 0, 0, 0)
        kpoints_layout.addWidget(self._kpoints_note)
        kpoints_layout.addWidget(self._kpoints_panel)
        self._tabs.addTab(kpoints_page, self.tr("KPOINTS"))
        self._tabs.addTab(self._build_potcar_page(), self.tr("POTCAR"))
        if not self._is_neb:
            self._tabs.addTab(self._build_poscar_page(), self.tr("POSCAR"))
        layout.addWidget(self._tabs, 1)

        buttons = QHBoxLayout()
        apply_btn = QPushButton(self.tr("&Apply"))
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._on_apply)
        buttons.addStretch(1)
        buttons.addWidget(apply_btn)
        close_btn = QPushButton(self.tr("Close"))
        close_btn.clicked.connect(self.reject)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    def _build_potcar_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self._potcar_info = QLabel("")
        self._potcar_info.setWordWrap(True)
        layout.addWidget(self._potcar_info)
        regen = QPushButton(self.tr("Regenerate POTCAR (current library)"))
        regen.clicked.connect(self._on_regen_potcar)
        layout.addWidget(regen)
        layout.addStretch(1)
        return page

    def _build_poscar_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(self.tr(
            "POSCAR is generated from the registered structure — to "
            "change it, edit the structure (structures/) and regenerate "
            "here."))
        note.setWordWrap(True)
        layout.addWidget(note)
        regen = QPushButton(self.tr("Regenerate POSCAR"))
        regen.clicked.connect(self._on_regen_poscar)
        layout.addWidget(regen)
        layout.addStretch(1)
        return page

    # -- load --------------------------------------------------------------

    def _load(self) -> None:
        run_dir = self._project.run_dir(self._run_id)
        entry = self._entry
        self._status_combo.setCurrentIndex(
            RUN_STATUSES.index(entry["status"])
            if entry["status"] in RUN_STATUSES else 0)

        incar_path = run_dir / "INCAR"
        if incar_path.exists():
            tags, problems = parse_incar_content(
                incar_path.read_text("utf-8"))
            if tags:
                self._incar_panel.set_tags(tags)
        else:
            self._incar_panel.set_preset("scf")

        kpoints_path = run_dir / "KPOINTS"
        if kpoints_path.exists():
            _mode, note = self._kpoints_panel.load_content(
                kpoints_path.read_text("utf-8"))
            if note:
                self._kpoints_note.setText(note)

        card_path = run_dir / "run.json"
        potcar_info = ""
        if card_path.exists():
            try:
                card = json.loads(card_path.read_text("utf-8"))
                params = card.get("parameters", {})
                potcar_info = self.tr("Functional: {}").format(
                    params.get("functional", "?"))
            except (OSError, json.JSONDecodeError):
                pass
        potcar = run_dir / "POTCAR"
        if potcar.exists():
            potcar_info += "\n" + self.tr(
                "POTCAR present ({} KB)").format(
                potcar.stat().st_size // 1024)
        else:
            potcar_info += "\n" + self.tr(
                "No POTCAR — configure the library in Settings and "
                "regenerate.")
        self._potcar_info.setText(potcar_info)

    # -- actions -----------------------------------------------------------

    def _confirm_overwrite(self) -> bool:
        run_dir = self._project.run_dir(self._run_id)
        if (run_dir / "CONTCAR").exists() or (run_dir / "OSZICAR").exists():
            reply = QMessageBox.question(
                self, self.tr("Run: {}").format(self._run_id),
                self.tr("This run directory already contains calculation "
                        "outputs — applying will overwrite input files. "
                        "Continue?"))
            return reply == QMessageBox.Yes
        return True

    def _on_apply(self) -> None:
        """Write INCAR/KPOINTS from the panels (WYSIWYG) + status."""
        if not self._confirm_overwrite():
            return
        run_dir = self._project.run_dir(self._run_id)
        try:
            incar = self._incar_panel.content()
            kpoints = self._kpoints_panel.content()
        except ValueError as exc:
            QMessageBox.warning(self, self.tr("Run: {}").format(
                self._run_id), str(exc))
            return
        (run_dir / "INCAR").write_text(incar + "\n", encoding="utf-8",
                                       newline="\n")
        (run_dir / "KPOINTS").write_text(kpoints + "\n", encoding="utf-8",
                                         newline="\n")
        self._project.set_run_status(
            self._run_id, self._status_combo.currentText())
        self._project.add_edition(self._run_id, "gui")
        logger.info("Run inputs edited: %s", self._run_id)
        self.accept()

    def _on_regen_potcar(self) -> None:
        from vaspen.core.vasp_input import generate_potcar

        library = AppConfig().potcar_library_path
        if not library:
            QMessageBox.information(
                self, self.tr("POTCAR"),
                self.tr("No pseudopotential library configured — set it "
                        "in Edit → Preferences."))
            return
        if not self._confirm_overwrite():
            return
        try:
            content, _paths = generate_potcar(
                self._structure_model.unique_symbols, library,
                self._project.run_parameters(self._run_id).get(
                    "functional", "PBE"))
        except Exception as exc:
            QMessageBox.critical(self, self.tr("POTCAR"), str(exc))
            return
        (self._project.run_dir(self._run_id) / "POTCAR").write_text(
            content, encoding="utf-8", newline="\n")
        self._project.add_edition(self._run_id, "gui")
        self._load()

    def _on_regen_poscar(self) -> None:
        if self._structure_model is None:
            QMessageBox.critical(
                self, self.tr("POSCAR"),
                self.tr("Registered structure not found."))
            return
        if not self._confirm_overwrite():
            return
        from vaspen.core.vasp_input import generate_poscar

        poscar = generate_poscar(self._structure_model)
        (self._project.run_dir(self._run_id) / "POSCAR").write_text(
            poscar, encoding="utf-8", newline="\n")
        self._project.add_edition(self._run_id, "gui")
        logger.info("POSCAR regenerated for run %s", self._run_id)
