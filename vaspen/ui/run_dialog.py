"""Run parameters dialog — view and adjust a project run (§7.15).

Shows the provenance card (read-only) plus the editable generation
parameters; "Regenerate Inputs" rebuilds the four VASP files in the
run directory from the registered structure with the new parameters.
If the run directory already carries calculation outputs (CONTCAR /
OSZICAR), regeneration asks first — it would overwrite them.

Modal dialog, created fresh per invocation (§11.2 — picks up the
current language automatically).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.project import Project, RUN_STATUSES
from vaspen.core.structure import StructureModel
from vaspen.core.vasp_input import INCAR_PRESETS
from vaspen.utils.logger import logger

_FUNCTIONALS = ("PBE", "PBE_new", "LDA", "PW91")
_KPOINTS_MODES = ("automatic", "manual", "line")


def _parse_overrides(text: str) -> tuple[dict, str | None]:
    """Parse ``KEY = VALUE`` lines into a dict; None error on success."""
    overrides: dict = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if "=" not in line:
            return {}, f"line {lineno}: missing '=' — {raw!r}"
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            return {}, f"line {lineno}: empty tag name"
        overrides[key] = value.strip()
    return overrides, None


class RunDialog(QDialog):
    """View / adjust one project run and regenerate its inputs."""

    def __init__(self, project: Project, run_id: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project = project
        self._run_id = run_id
        self._entry = dict(project.run_entry(run_id))
        self._card_parameters = dict(project.run_parameters(run_id))

        self.setWindowTitle(self.tr("Run Parameters"))
        self.resize(560, 480)
        self._build_ui()
        self._load()

    # -- UI --------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── provenance (read-only) ──
        info_group = QGroupBox(self.tr("Provenance"))
        info_form = QFormLayout(info_group)
        self._info_id = QLabel(self._run_id)
        self._info_type = QLabel(self._entry.get("type", ""))
        self._info_structure = QLabel(self._entry.get("structure", ""))
        self._info_created = QLabel(self._entry.get("created", ""))
        digest = str(self._entry.get("structure_digest", ""))
        self._info_digest = QLabel(digest[:19] + "…" if len(digest) > 19
                                   else digest)
        self._info_digest.setToolTip(digest)
        by = self._entry  # generated_by lives on the card, not the entry
        card_by = self._card_parameters  # placeholder, replaced in _load
        for label, widget in ((self.tr("Run:"), self._info_id),
                              (self.tr("Type:"), self._info_type),
                              (self.tr("Structure:"), self._info_structure),
                              (self.tr("Created:"), self._info_created),
                              (self.tr("Structure digest:"),
                               self._info_digest)):
            info_form.addRow(label, widget)
        layout.addWidget(info_group)

        # ── status ──
        status_group = QGroupBox(self.tr("Status"))
        status_form = QFormLayout(status_group)
        self._status_combo = QComboBox()
        for status in RUN_STATUSES:
            self._status_combo.addItem(status)  # canonical manifest values
        status_form.addRow(self.tr("Status:"), self._status_combo)
        layout.addWidget(status_group)

        # ── generation parameters ──
        gen_group = QGroupBox(self.tr("Generation Parameters"))
        gen_form = QFormLayout(gen_group)

        self._preset_combo = QComboBox()
        for name in INCAR_PRESETS:
            if not name.startswith("_"):
                self._preset_combo.addItem(name)
        gen_form.addRow(self.tr("INCAR preset:"), self._preset_combo)

        self._functional_combo = QComboBox()
        self._functional_combo.addItems(_FUNCTIONALS)
        gen_form.addRow(self.tr("Functional:"), self._functional_combo)

        self._kpoints_combo = QComboBox()
        self._kpoints_combo.addItems(_KPOINTS_MODES)
        self._kpoints_combo.currentTextChanged.connect(
            self._on_kpoints_mode)
        gen_form.addRow(self.tr("KPOINTS mode:"), self._kpoints_combo)

        # automatic → single spacing; manual → three ints; line → P2
        self._kspacing_spin = QDoubleSpinBox()
        self._kspacing_spin.setRange(0.005, 1.0)
        self._kspacing_spin.setDecimals(3)
        self._kspacing_spin.setSingleStep(0.005)
        self._kspacing_label = QLabel(self.tr("KSPACING (2π/Å):"))
        gen_form.addRow(self._kspacing_label, self._kspacing_spin)

        self._k_manual_row = QHBoxLayout()
        self._k1 = QSpinBox(); self._k1.setRange(1, 99)
        self._k2 = QSpinBox(); self._k2.setRange(1, 99)
        self._k3 = QSpinBox(); self._k3.setRange(1, 99)
        for box in (self._k1, self._k2, self._k3):
            self._k_manual_row.addWidget(box)
        self._k_manual_label = QLabel(self.tr("Mesh (k1 k2 k3):"))
        self._k_manual_widget = QWidget()
        self._k_manual_widget.setLayout(self._k_manual_row)
        gen_form.addRow(self._k_manual_label, self._k_manual_widget)

        gen_form.addRow(self.tr("INCAR overrides:"), self._overrides_edit())
        layout.addWidget(gen_group)

        # ── buttons ──
        self._regen_button = QPushButton(self.tr("&Regenerate Inputs"))
        self._regen_button.clicked.connect(self._on_regenerate)
        layout.addWidget(self._regen_button)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _overrides_edit(self) -> QPlainTextEdit:
        self._overrides = QPlainTextEdit()
        self._overrides.setPlaceholderText(self.tr(
            "KEY = VALUE (one per line), e.g.\n  IVDW = 11\n"
            "  MAGMOM = 0.6*56"))
        self._overrides.setFixedHeight(72)
        return self._overrides

    def _on_kpoints_mode(self, mode: str) -> None:
        auto = mode == "automatic"
        manual = mode == "manual"
        self._kspacing_spin.setVisible(auto)
        self._kspacing_label.setVisible(auto)
        self._k_manual_label.setVisible(manual)
        self._k_manual_widget.setVisible(manual)

    # -- load / collect ----------------------------------------------------

    def _load(self) -> None:
        entry = self._entry
        params = self._card_parameters
        self._status_combo.setCurrentIndex(
            RUN_STATUSES.index(entry["status"])
            if entry["status"] in RUN_STATUSES else 0)
        preset = str(params.get("preset", "scf"))
        self._preset_combo.setCurrentText(
            preset if preset in INCAR_PRESETS else "scf")
        self._functional_combo.setCurrentText(
            str(params.get("functional", "PBE")))
        kmode = str(params.get("kpoints_mode", "automatic"))
        self._kpoints_combo.setCurrentText(
            kmode if kmode in _KPOINTS_MODES else "automatic")
        kparams = params.get("kpoints_params") or {}
        self._kspacing_spin.setValue(float(kparams.get("k_spacing", 0.04)))
        mesh = kparams.get("mesh") or (1, 1, 1)
        for box, value in zip((self._k1, self._k2, self._k3), mesh):
            box.setValue(int(value))
        self._on_kpoints_mode(self._kpoints_combo.currentText())
        # INCAR overrides = card parameters minus the structured keys
        structured = {"preset", "kpoints_mode", "kpoints_params",
                      "functional"}
        lines = [f"{key} = {value}" for key, value in params.items()
                 if key not in structured]
        self._overrides.setPlainText("\n".join(lines))

    def _collect_overrides(self) -> dict | None:
        overrides, error = _parse_overrides(
            self._overrides.toPlainText())
        if error:
            QMessageBox.warning(self, self.tr("Run Parameters"),
                                self.tr("INCAR overrides: {}").format(error))
            return None
        return overrides

    def _collect_kpoints_params(self) -> dict:
        mode = self._kpoints_combo.currentText()
        if mode == "automatic":
            return {"k_spacing": round(self._kspacing_spin.value(), 4)}
        if mode == "manual":
            return {"k1": self._k1.value(), "k2": self._k2.value(),
                    "k3": self._k3.value()}
        return {}

    # -- actions -----------------------------------------------------------

    def _on_regenerate(self) -> None:
        overrides = self._collect_overrides()
        if overrides is None:
            return
        run_dir = self._project.run_dir(self._run_id)
        if (run_dir / "CONTCAR").exists() or (run_dir / "OSZICAR").exists():
            reply = QMessageBox.question(
                self, self.tr("Run Parameters"),
                self.tr("This run directory already contains calculation "
                        "outputs — regenerating will overwrite the input "
                        "files. Continue?"))
            if reply != QMessageBox.Yes:
                return
        entry = self._project.run_entry(self._run_id)
        try:
            structure_path = self._project.structure_path(
                entry.get("structure", ""))
            model = StructureModel()
            model.load(structure_path)
            files = self._project.write_run_inputs(
                self._run_id, model,
                preset=self._preset_combo.currentText(),
                incar_overrides=overrides or None,
                kpoints_mode=self._kpoints_combo.currentText(),
                kpoints_params=self._collect_kpoints_params(),
                functional=self._functional_combo.currentText(),
            )
        except Exception as e:
            QMessageBox.critical(self, self.tr("Run Parameters"), str(e))
            return
        self._project.set_run_status(
            self._run_id, self._status_combo.currentText())
        logger.info("Regenerated run inputs: %s (%d files)",
                    self._run_id, len(files))
        self.accept()
