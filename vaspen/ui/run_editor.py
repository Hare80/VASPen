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
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.potcar_info import parse_potcar_sections
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
        # for NEB runs the manifest's structure is the INITIAL endpoint;
        # the final one is recorded on the card's parameters
        self._final_structure = str(
            project.run_parameters(run_id).get("final_structure") or "")
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
        self._build_ui()
        self._restore_geometry()
        self._load()

    # -- UI --------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # one compact line — every pixel goes to the editor panels
        info_bar = QHBoxLayout()
        info_bar.addWidget(QLabel(self.tr("Type: {}").format(
            self._entry.get("type", ""))))
        if self._is_neb:
            # a NEB run has TWO structures — naming one "the structure"
            # would be misleading; the manifest entry is the initial
            # endpoint, the final one lives on the provenance card
            self._structure_label = QLabel(self.tr("Initial: {}").format(
                self._entry.get("structure", "")))
            self._final_label = QLabel(self.tr("Final: {}").format(
                self._final_structure or "—"))
            info_bar.addWidget(self._structure_label)
            info_bar.addWidget(self._final_label)
        else:
            self._structure_label = QLabel(self.tr("Structure: {}").format(
                self._entry.get("structure", "")))
            self._final_label = None
            info_bar.addWidget(self._structure_label)
        self._status_combo = QComboBox()
        for status in RUN_STATUSES:
            self._status_combo.addItem(status)  # canonical manifest values
        info_bar.addStretch(1)
        info_bar.addWidget(QLabel(self.tr("Status:")))
        info_bar.addWidget(self._status_combo)
        layout.addLayout(info_bar)

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
        # one row per POTCAR dataset section: which element uses which
        # pseudopotential, its valence count and recommended cutoff
        self._potcar_table = QTableWidget(0, 4)
        self._potcar_table.setHorizontalHeaderLabels([
            self.tr("Element"), self.tr("Potential"),
            self.tr("Z(e)"), self.tr("ENMAX (eV)")])
        self._potcar_table.verticalHeader().setVisible(False)
        self._potcar_table.horizontalHeader().setStretchLastSection(True)
        self._potcar_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self._potcar_table, 1)
        self._pseudo_note = QLabel("")
        self._pseudo_note.setWordWrap(True)
        self._pseudo_note.hide()
        layout.addWidget(self._pseudo_note)
        regen = QPushButton(self.tr("Regenerate POTCAR (current library)"))
        regen.clicked.connect(self._on_regen_potcar)
        layout.addWidget(regen)
        return page

    def _fill_potcar_table(self, sections: list[dict]) -> None:
        self._potcar_table.setRowCount(len(sections))
        has_pseudo = False
        for row, section in enumerate(sections):
            element = section["elements"][0] if section["elements"] else "?"
            if section["pseudo_h"]:
                has_pseudo = True
                item = QTableWidgetItem(element + self.tr(" + pseudo-H"))
                item.setToolTip(self.tr(
                    "This potential embeds a pseudo-hydrogen (it fills "
                    "the empty d-band) — not a real hydrogen in the "
                    "structure."))
            else:
                item = QTableWidgetItem(element)
            self._potcar_table.setItem(row, 0, item)
            self._potcar_table.setItem(
                row, 1, QTableWidgetItem(section["variant"]))
            self._potcar_table.setItem(row, 2, QTableWidgetItem(
                f"{section['zval']:g}"
                if section["zval"] is not None else "—"))
            self._potcar_table.setItem(row, 3, QTableWidgetItem(
                f"{section['enmax']:g}"
                if section["enmax"] is not None else "—"))
        self._potcar_table.setVisible(bool(sections))
        self._pseudo_note.setVisible(has_pseudo)
        if has_pseudo:
            self._pseudo_note.setText(self.tr(
                "pseudo-H marks a pseudo-hydrogen embedded in an *_h "
                "potential (it fills the empty d-band) — it is not a "
                "real hydrogen in the structure."))

    def _build_poscar_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(self.tr(
            "POSCAR is written out from the structure registered in "
            "this project (structures/). This button only rewrites the "
            "run's POSCAR file to match it — it never modifies the "
            "structure itself. Use it after updating the registered "
            "structure, or to restore a lost POSCAR."))
        note.setWordWrap(True)
        layout.addWidget(note)
        regen = QPushButton(self.tr("Regenerate POSCAR"))
        regen.clicked.connect(self._on_regen_poscar)
        layout.addWidget(regen)
        layout.addStretch(1)
        return page

    # -- geometry ------------------------------------------------------------

    def _restore_geometry(self) -> None:
        """Default 1100x800; remember the user's size across opens
        (via AppConfig — the test suite isolates its QSettings)."""
        geo = AppConfig().qt_settings().value("run_editor_geometry")
        if geo and self.restoreGeometry(geo):
            return
        self.resize(1100, 800)

    def closeEvent(self, event) -> None:
        AppConfig().qt_settings().setValue(
            "run_editor_geometry", self.saveGeometry())
        super().closeEvent(event)

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
            sections = parse_potcar_sections(
                potcar.read_text("utf-8", errors="replace"))
            self._fill_potcar_table(sections)
        else:
            potcar_info += "\n" + self.tr(
                "No POTCAR — configure the library in Settings and "
                "regenerate.")
            self._potcar_table.hide()
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
        try:
            (run_dir / "INCAR").write_text(incar + "\n", encoding="utf-8",
                                           newline="\n")
            (run_dir / "KPOINTS").write_text(
                kpoints + "\n", encoding="utf-8", newline="\n")
        except OSError as exc:
            # e.g. the run was deleted from the project panel while
            # this editor stayed open
            QMessageBox.critical(self, self.tr("Run: {}").format(
                self._run_id), str(exc))
            return
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
        # preserve the run's POTCAR variants (Fe_pv stays Fe_pv; an
        # *_h pseudo-hydrogen is not silently dropped): parse the
        # existing file, fall back to the card, then defaults
        run_dir = self._project.run_dir(self._run_id)
        variants: dict = {}
        potcar = run_dir / "POTCAR"
        if potcar.exists():
            for section in parse_potcar_sections(
                    potcar.read_text("utf-8", errors="replace")):
                if section["elements"] and section["variant"]:
                    variants[section["elements"][0]] = section["variant"]
        if not variants:
            variants = dict(self._project.run_parameters(
                self._run_id).get("potcar_variants") or {})
        try:
            content, _paths = generate_potcar(
                self._structure_model.unique_symbols, library,
                self._project.run_parameters(self._run_id).get(
                    "functional", "PBE"),
                variants=variants or None)
        except Exception as exc:
            QMessageBox.critical(self, self.tr("POTCAR"), str(exc))
            return
        try:
            (run_dir / "POTCAR").write_text(
                content, encoding="utf-8", newline="\n")
        except OSError as exc:
            QMessageBox.critical(self, self.tr("POTCAR"), str(exc))
            return
        if variants:
            self._project.update_run_parameters(
                self._run_id, {"potcar_variants": variants})
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
        try:
            (self._project.run_dir(self._run_id) / "POSCAR").write_text(
                poscar, encoding="utf-8", newline="\n")
        except OSError as exc:
            QMessageBox.critical(self, self.tr("POSCAR"), str(exc))
            return
        self._project.add_edition(self._run_id, "gui")
        logger.info("POSCAR regenerated for run %s", self._run_id)
