"""Main window — menu bar, toolbars, status bar, dock widgets.

This is the application shell. All views (3D viewport, structure tree,
editor panels) are docked or centrally embedded here.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QSettings, QEvent, QTranslator
from PySide6.QtGui import QAction, QKeySequence, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QStatusBar,
    QToolBar,
    QWidget,
    QVBoxLayout,
    QLabel,
)

from vaspen.core.structure import StructureModel
from vaspen.core.file_io import FileIO
from vaspen.ui.structure_tree import StructureTreePanel
from vaspen.ui.viewport3d import Viewport3D
from vaspen.utils.config import AppConfig
from vaspen.utils.logger import logger


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self) -> None:
        super().__init__()
        self._config = AppConfig()
        self._structure = StructureModel()

        # Translator — owned by the window so the language can switch live
        self._translator = QTranslator(self)
        i18n_dir = Path(__file__).parent.parent / "resources" / "i18n"
        qm_path = i18n_dir / f"vaspen_{self._config.language}.qm"
        if qm_path.exists():
            self._translator.load(str(qm_path))
        QApplication.instance().installTranslator(self._translator)

        # Window properties
        self.setWindowTitle(self.tr("VASPen"))
        self.resize(1280, 800)
        self.setAcceptDrops(True)

        # Build UI
        self._create_actions()
        self._create_menu_bar()
        self._create_toolbar()
        self._create_status_bar()
        self._create_central_widget()
        self._create_dock_widgets()
        self._connect_signals()
        self._restore_window_state()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _create_actions(self) -> None:
        """Create all QActions used in menus and toolbars."""

        # ── File ──
        self.act_new = QAction(self.tr("&New Structure..."), self)
        self.act_new.setShortcut(QKeySequence.New)
        self.act_new.setStatusTip(self.tr("Create a new empty structure"))
        self.act_new.triggered.connect(self._on_new)

        self.act_open = QAction(self.tr("&Open..."), self)
        self.act_open.setShortcut(QKeySequence.Open)
        self.act_open.setStatusTip(self.tr("Open a structure file"))
        self.act_open.triggered.connect(self._on_open)

        self.act_save = QAction(self.tr("&Save"), self)
        self.act_save.setShortcut(QKeySequence.Save)
        self.act_save.setStatusTip(self.tr("Save current structure"))
        self.act_save.triggered.connect(self._on_save)

        self.act_save_as = QAction(self.tr("Save &As..."), self)
        self.act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.act_save_as.setStatusTip(self.tr("Save to a new file"))
        self.act_save_as.triggered.connect(self._on_save_as)

        self.act_export_poscar = QAction(self.tr("Export as &POSCAR..."), self)
        self.act_export_poscar.triggered.connect(self._on_export_poscar)

        self.act_quit = QAction(self.tr("&Quit"), self)
        self.act_quit.setShortcut(QKeySequence.Quit)
        self.act_quit.triggered.connect(self.close)

        # ── Edit ──
        self.act_undo = QAction(self.tr("&Undo"), self)
        self.act_undo.setShortcut(QKeySequence.Undo)
        self.act_undo.setEnabled(False)
        self.act_undo.triggered.connect(lambda: self._structure.undo())

        self.act_redo = QAction(self.tr("&Redo"), self)
        self.act_redo.setShortcut(QKeySequence.Redo)
        self.act_redo.setEnabled(False)
        self.act_redo.triggered.connect(lambda: self._structure.redo())

        self.act_preferences = QAction(self.tr("&Preferences..."), self)
        self.act_preferences.setStatusTip(self.tr("Configure settings"))
        self.act_preferences.triggered.connect(self._on_preferences)

        # ── View ──
        self.act_lang_en = QAction(self.tr("English"), self)
        self.act_lang_en.setCheckable(True)
        self.act_lang_en.triggered.connect(lambda: self._switch_language("en"))

        self.act_lang_zh = QAction(self.tr("中文"), self)
        self.act_lang_zh.setCheckable(True)
        self.act_lang_zh.triggered.connect(lambda: self._switch_language("zh"))

        lang = self._config.language
        self.act_lang_en.setChecked(lang == "en")
        self.act_lang_zh.setChecked(lang == "zh")

        # ── Calculate ──
        self.act_gen_incar = QAction(self.tr("Generate &INCAR..."), self)
        self.act_gen_incar.triggered.connect(self._on_generate_incar)

        self.act_gen_kpoints = QAction(self.tr("Generate &KPOINTS..."), self)
        self.act_gen_kpoints.triggered.connect(self._on_generate_kpoints)

        self.act_gen_potcar = QAction(self.tr("Generate &POTCAR..."), self)
        self.act_gen_potcar.triggered.connect(self._on_generate_potcar)

        self.act_gen_all = QAction(self.tr("Generate &All Input Files..."), self)
        self.act_gen_all.setStatusTip(self.tr("Generate INCAR, KPOINTS, POSCAR, POTCAR at once"))
        self.act_gen_all.triggered.connect(self._on_generate_all)

        # ── Tools ──
        self.act_surface = QAction(self.tr("C&ut Surface..."), self)
        self.act_surface.setStatusTip(self.tr("Cut a surface/slab from the current structure"))
        self.act_surface.triggered.connect(self._on_surface)

        self.act_supercell = QAction(self.tr("&Supercell..."), self)
        self.act_supercell.setStatusTip(self.tr("Create a supercell"))
        self.act_supercell.triggered.connect(self._on_supercell)

        # ── Help ──
        self.act_about = QAction(self.tr("&About VASPen"), self)
        self.act_about.triggered.connect(self._on_about)

        self.act_about_qt = QAction(self.tr("About &Qt"), self)
        self.act_about_qt.triggered.connect(QApplication.instance().aboutQt)

    # ------------------------------------------------------------------
    # Menu Bar
    # ------------------------------------------------------------------

    def _create_menu_bar(self) -> None:
        mb: QMenuBar = self.menuBar()

        # File
        self._menu_file = mb.addMenu(self.tr("&File"))
        self._menu_file.addAction(self.act_new)
        self._menu_file.addAction(self.act_open)
        self._menu_file.addSeparator()
        self._menu_file.addAction(self.act_save)
        self._menu_file.addAction(self.act_save_as)
        self._menu_file.addAction(self.act_export_poscar)
        self._menu_file.addSeparator()
        self._recent_menu = self._menu_file.addMenu(self.tr("&Recent Files"))
        self._update_recent_menu()
        self._menu_file.addSeparator()
        self._menu_file.addAction(self.act_quit)

        # Edit
        self._menu_edit = mb.addMenu(self.tr("&Edit"))
        self._menu_edit.addAction(self.act_undo)
        self._menu_edit.addAction(self.act_redo)
        self._menu_edit.addSeparator()
        self._menu_edit.addAction(self.act_preferences)

        # View
        self._menu_view = mb.addMenu(self.tr("&View"))
        self._menu_lang = self._menu_view.addMenu(self.tr("&Language"))
        self._menu_lang.addAction(self.act_lang_en)
        self._menu_lang.addAction(self.act_lang_zh)

        # Calculate
        self._menu_calc = mb.addMenu(self.tr("&Calculate"))
        self._menu_calc.addAction(self.act_gen_incar)
        self._menu_calc.addAction(self.act_gen_kpoints)
        self._menu_calc.addAction(self.act_gen_potcar)
        self._menu_calc.addSeparator()
        self._menu_calc.addAction(self.act_gen_all)

        # Tools
        self._menu_tools = mb.addMenu(self.tr("&Tools"))
        self._menu_tools.addAction(self.act_surface)
        self._menu_tools.addAction(self.act_supercell)

        # Help
        self._menu_help = mb.addMenu(self.tr("&Help"))
        self._menu_help.addAction(self.act_about)
        self._menu_help.addAction(self.act_about_qt)

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _create_toolbar(self) -> None:
        self._toolbar = QToolBar(self.tr("Main Toolbar"), self)
        self._toolbar.setObjectName("main_toolbar")
        self._toolbar.setMovable(False)
        self._toolbar.addAction(self.act_new)
        self._toolbar.addAction(self.act_open)
        self._toolbar.addAction(self.act_save)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self.act_gen_all)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self.act_surface)
        self._toolbar.addAction(self.act_supercell)
        self.addToolBar(self._toolbar)

    # ------------------------------------------------------------------
    # Status Bar
    # ------------------------------------------------------------------

    def _create_status_bar(self) -> None:
        sb: QStatusBar = self.statusBar()

        self._status_label = QLabel(self.tr("Ready"))
        sb.addWidget(self._status_label, 1)

        self._atom_count_label = QLabel("")
        sb.addPermanentWidget(self._atom_count_label)

        self._cell_label = QLabel("")
        sb.addPermanentWidget(self._cell_label)

    # ------------------------------------------------------------------
    # Central Widget — 3D viewport
    # ------------------------------------------------------------------

    def _create_central_widget(self) -> None:
        self._viewport = Viewport3D(self)
        self.setCentralWidget(self._viewport)

    # ------------------------------------------------------------------
    # Dock Widgets
    # ------------------------------------------------------------------

    def _create_dock_widgets(self) -> None:
        # Structure tree dock (left side) — atom list + cell parameters
        self._dock_structure = QDockWidget(self.tr("Structure"), self)
        self._dock_structure.setObjectName("dock_structure")
        self._dock_structure.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._structure_tree = StructureTreePanel(self._structure)
        self._dock_structure.setWidget(self._structure_tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._dock_structure)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._connect_model_signals(self._structure)
        # Viewport ↔ model bidirectional wiring
        self._viewport.atom_clicked.connect(self._on_atom_clicked)
        self._viewport.background_clicked.connect(self._on_background_clicked)

    def _connect_model_signals(self, model: StructureModel) -> None:
        """Connect a StructureModel's signals to window/UI updates.

        Called once per model instance (models are replaced on New).
        """
        model.structure_loaded.connect(self._on_structure_loaded)
        model.structure_modified.connect(self._on_structure_modified)
        model.atom_selected.connect(self._on_atom_selected)
        model.selection_cleared.connect(self._on_selection_cleared)

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _on_new(self) -> None:
        """Create a new empty structure (for now, reset to empty)."""
        # TODO: Launch a "new structure" dialog (crystal builder, import molecule, etc.)
        reply = QMessageBox.question(
            self,
            self.tr("New Structure"),
            self.tr("Discard current changes and create a new empty structure?"),
        )
        if reply == QMessageBox.Yes:
            self._structure = StructureModel()
            self._connect_model_signals(self._structure)
            self._structure_tree.set_model(self._structure)
            self._viewport.set_structure(None)
            self._update_status_bar()
            self._update_edit_actions()
            self._set_status(self.tr("New structure created."))

    def _on_open(self) -> None:
        """Open a structure file."""
        file_filter = FileIO.file_filter(for_writing=False)
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open Structure File"),
            self._config.last_directory,
            file_filter,
        )
        if filepath:
            self._open_file(filepath)

    def _open_file(self, filepath: str) -> None:
        """Load a structure from the given path."""
        try:
            atoms = FileIO.read(filepath)
            self._structure.load_atoms(atoms, filepath)  # emits structure_loaded once
            self._config.add_recent_file(filepath)
            self._config.last_directory = str(Path(filepath).parent)
            self._update_recent_menu()
            self._set_status(self.tr("Loaded: {}").format(filepath))
            logger.info("Opened file: %s", filepath)
        except Exception as e:
            logger.exception("Failed to open file: %s", filepath)
            QMessageBox.critical(
                self,
                self.tr("Open Failed"),
                self.tr("Could not open file:\n{}").format(str(e)),
            )

    def _on_save(self) -> None:
        """Save the current structure."""
        fp = self._structure.filepath
        if fp:
            try:
                self._structure.save(fp)
                self._set_status(self.tr("Saved: {}").format(fp))
            except Exception as e:
                QMessageBox.critical(self, self.tr("Save Failed"), str(e))
        else:
            self._on_save_as()

    def _on_save_as(self) -> None:
        """Save the structure to a new file."""
        file_filter = FileIO.file_filter(for_writing=True)
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save Structure As"),
            self._config.last_directory,
            file_filter,
        )
        if filepath:
            try:
                self._structure.save(filepath)
                self._config.add_recent_file(filepath)
                self._config.last_directory = str(Path(filepath).parent)
                self._update_recent_menu()
                self._set_status(self.tr("Saved: {}").format(filepath))
            except Exception as e:
                QMessageBox.critical(self, self.tr("Save Failed"), str(e))

    def _on_export_poscar(self) -> None:
        """Export current structure as a POSCAR file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Export as POSCAR"),
            str(Path(self._config.last_directory) / "POSCAR"),
            "POSCAR (*.vasp);;All files (*)",
        )
        if filepath:
            try:
                self._structure.save(filepath, fmt="vasp")
                self._set_status(self.tr("Exported POSCAR: {}").format(filepath))
            except Exception as e:
                QMessageBox.critical(self, self.tr("Export Failed"), str(e))

    # ------------------------------------------------------------------
    # Calculate → generate VASP inputs
    # ------------------------------------------------------------------

    def _on_generate_incar(self) -> None:
        from vaspen.ui.incar_editor import IncarEditorDialog
        dlg = IncarEditorDialog(self._structure, self)
        dlg.exec()

    def _on_generate_kpoints(self) -> None:
        from vaspen.ui.kpoints_editor import KpointsEditorDialog
        dlg = KpointsEditorDialog(self._structure, self)
        dlg.exec()

    def _on_generate_potcar(self) -> None:
        from vaspen.ui.potcar_dialog import PotcarDialog
        dlg = PotcarDialog(self._structure, self)
        dlg.exec()

    def _on_generate_all(self) -> None:
        from vaspen.core.vasp_input import generate_all_inputs

        potcar_path = self._config.potcar_library_path
        if not potcar_path:
            reply = QMessageBox.question(
                self,
                self.tr("POTCAR Library Not Configured"),
                self.tr("POTCAR library path is not set. Generate without POTCAR?\n\n"
                        "You can configure it in Edit → Preferences."),
            )
            if reply != QMessageBox.Yes:
                return

        output_dir = QFileDialog.getExistingDirectory(
            self,
            self.tr("Choose Output Directory"),
            self._config.last_directory,
        )
        if not output_dir:
            return

        try:
            files = generate_all_inputs(
                self._structure,
                incar_preset=self._config.default_calc_type,
                potcar_library=potcar_path,
            )
            out = Path(output_dir)
            for name, content in files.items():
                if content:
                    # UTF-8 + LF: VASP input files must not carry the
                    # locale encoding (GBK) or CRLF line endings
                    (out / name).write_text(content, encoding="utf-8", newline="\n")

            self._set_status(self.tr("VASP input files generated in: {}").format(output_dir))
            QMessageBox.information(
                self,
                self.tr("Success"),
                self.tr("Generated files in:\n{}\n\nFiles: {}").format(
                    output_dir, ", ".join(f for f, c in files.items() if c)
                ),
            )
        except Exception as e:
            logger.exception("Failed to generate inputs")
            QMessageBox.critical(
                self,
                self.tr("Generation Failed"),
                str(e),
            )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _on_surface(self) -> None:
        from vaspen.ui.surface_dialog import SurfaceDialog
        dlg = SurfaceDialog(self._structure, self)
        if dlg.exec() == SurfaceDialog.Accepted and dlg.result_structure is not None:
            self._structure.replace_atoms(dlg.result_structure.atoms)
            self._structure.reset_filepath()  # never silently overwrite the bulk file
            self._set_status(self.tr("Surface cut applied."))

    def _on_supercell(self) -> None:
        from vaspen.core.builder import StructureBuilder

        # Simple input: use a dialog or inline for now
        from PySide6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(
            self,
            self.tr("Supercell"),
            self.tr("Enter scaling factors (n_a n_b n_c):"),
            text="1 1 1",
        )
        if ok and text:
            try:
                parts = [int(x) for x in text.split()]
                if len(parts) != 3:
                    raise ValueError(self.tr("Need exactly 3 integers"))
                StructureBuilder.make_supercell(self._structure, tuple(parts))
                self._structure.reset_filepath()  # derived structure → Save As
                self._set_status(self.tr("Supercell {}×{}×{} created.").format(*parts))
            except Exception as e:
                QMessageBox.critical(self, self.tr("Invalid Input"), str(e))

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    def _on_preferences(self) -> None:
        from vaspen.ui.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self)
        dlg.exec()
        # Apply any language change immediately (no restart needed);
        # _switch_language is a no-op when the language is unchanged.
        self._switch_language(self._config.language)

    # ------------------------------------------------------------------
    # Language switching
    # ------------------------------------------------------------------

    def changeEvent(self, event: QEvent) -> None:
        """Re-translate all texts when the application language changes."""
        if event.type() == QEvent.Type.LanguageChange:
            self._retranslate_ui()
        super().changeEvent(event)

    def _retranslate_ui(self) -> None:
        """Re-apply every translatable text (called on LanguageChange)."""
        self.setWindowTitle(self.tr("VASPen"))

        # Actions
        self.act_new.setText(self.tr("&New Structure..."))
        self.act_new.setStatusTip(self.tr("Create a new empty structure"))
        self.act_open.setText(self.tr("&Open..."))
        self.act_open.setStatusTip(self.tr("Open a structure file"))
        self.act_save.setText(self.tr("&Save"))
        self.act_save.setStatusTip(self.tr("Save current structure"))
        self.act_save_as.setText(self.tr("Save &As..."))
        self.act_save_as.setStatusTip(self.tr("Save to a new file"))
        self.act_export_poscar.setText(self.tr("Export as &POSCAR..."))
        self.act_quit.setText(self.tr("&Quit"))
        self.act_undo.setText(self.tr("&Undo"))
        self.act_redo.setText(self.tr("&Redo"))
        self.act_preferences.setText(self.tr("&Preferences..."))
        self.act_preferences.setStatusTip(self.tr("Configure settings"))
        self.act_lang_en.setText(self.tr("English"))
        self.act_lang_zh.setText(self.tr("中文"))
        self.act_gen_incar.setText(self.tr("Generate &INCAR..."))
        self.act_gen_kpoints.setText(self.tr("Generate &KPOINTS..."))
        self.act_gen_potcar.setText(self.tr("Generate &POTCAR..."))
        self.act_gen_all.setText(self.tr("Generate &All Input Files..."))
        self.act_gen_all.setStatusTip(self.tr("Generate INCAR, KPOINTS, POSCAR, POTCAR at once"))
        self.act_surface.setText(self.tr("C&ut Surface..."))
        self.act_surface.setStatusTip(self.tr("Cut a surface/slab from the current structure"))
        self.act_supercell.setText(self.tr("&Supercell..."))
        self.act_supercell.setStatusTip(self.tr("Create a supercell"))
        self.act_about.setText(self.tr("&About VASPen"))
        self.act_about_qt.setText(self.tr("About &Qt"))

        # Menus
        self._menu_file.setTitle(self.tr("&File"))
        self._recent_menu.setTitle(self.tr("&Recent Files"))
        self._menu_edit.setTitle(self.tr("&Edit"))
        self._menu_view.setTitle(self.tr("&View"))
        self._menu_lang.setTitle(self.tr("&Language"))
        self._menu_calc.setTitle(self.tr("&Calculate"))
        self._menu_tools.setTitle(self.tr("&Tools"))
        self._menu_help.setTitle(self.tr("&Help"))

        # Toolbar / docks / status bar
        self._toolbar.setWindowTitle(self.tr("Main Toolbar"))
        self._dock_structure.setWindowTitle(self.tr("Structure"))
        self._status_label.setText(self.tr("Ready"))
        self._update_recent_menu()
        if self._structure.n_atoms > 0:
            self._update_status_bar()

    def _switch_language(self, lang: str) -> None:
        """Switch the UI language immediately — no restart needed."""
        if lang == self._config.language:
            return
        self._config.language = lang
        app = QApplication.instance()

        # Reload the translator, then broadcast LanguageChange to every widget
        app.removeTranslator(self._translator)
        i18n_dir = Path(__file__).parent.parent / "resources" / "i18n"
        qm_path = i18n_dir / f"vaspen_{lang}.qm"
        if qm_path.exists():
            self._translator.load(str(qm_path))
        app.installTranslator(self._translator)
        for widget in app.allWidgets():
            app.sendEvent(widget, QEvent(QEvent.Type.LanguageChange))

        self.act_lang_en.setChecked(lang == "en")
        self.act_lang_zh.setChecked(lang == "zh")
        self._set_status("中文" if lang == "zh" else "English")

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            self.tr("About VASPen"),
            self.tr(
                "<h2>VASPen v0.1.0</h2>"
                "<p>A cross-platform GUI for VASP first-principles calculations.</p>"
                "<p><b>Built with:</b> PySide6, ASE, pymatgen, Qt native OpenGL</p>"
                "<p>Free and open source (MIT License).</p>"
            ),
        )

    # ------------------------------------------------------------------
    # Drag & Drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            filepath = urls[0].toLocalFile()
            if Path(filepath).exists():
                self._open_file(filepath)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _restore_window_state(self) -> None:
        geo = self._config.window_geometry
        if geo:
            self.restoreGeometry(geo)
        state = self._config.window_state
        if state:
            self.restoreState(state)

    def closeEvent(self, event) -> None:
        self._config.window_geometry = self.saveGeometry()
        self._config.window_state = self.saveState()
        self._config.sync()
        super().closeEvent(event)

    def _update_recent_menu(self) -> None:
        self._recent_menu.clear()
        recent = self._config.recent_files
        if not recent:
            self._recent_menu.addAction(self.tr("(No recent files)")).setEnabled(False)
        else:
            for fp in recent:
                action = QAction(Path(fp).name, self)
                action.setToolTip(fp)
                action.triggered.connect(lambda checked, p=fp: self._open_file(p))
                self._recent_menu.addAction(action)
            self._recent_menu.addSeparator()
            clear_action = QAction(self.tr("Clear Recent Files"), self)
            clear_action.triggered.connect(self._clear_recent)
            self._recent_menu.addAction(clear_action)

    def _clear_recent(self) -> None:
        self._config.recent_files = []
        self._update_recent_menu()

    def _on_structure_loaded(self) -> None:
        self._viewport.set_structure(self._structure.atoms)
        self._update_status_bar()
        self._update_edit_actions()

    def _on_structure_modified(self) -> None:
        """Structure changed (add/remove atoms, supercell, surface cut...)."""
        # Keep the user's camera — in-place edits must not snap the view back
        self._viewport.set_structure(self._structure.atoms, reset_view=False)
        self._update_status_bar()
        self._update_edit_actions()

    def _on_selection_cleared(self) -> None:
        """Selection cleared via the model (e.g. the selected atom was deleted)."""
        self._viewport.highlight_atom(None)

    def _update_edit_actions(self) -> None:
        """Enable Undo/Redo based on the model's edit history."""
        self.act_undo.setEnabled(self._structure.can_undo)
        self.act_redo.setEnabled(self._structure.can_redo)

    def _on_atom_clicked(self, index: int) -> None:
        """User clicked an atom in the 3D viewport."""
        self._structure.select_atom(index)

    def _on_background_clicked(self) -> None:
        """User clicked empty space in the 3D viewport."""
        self._structure.clear_selection()

    def _on_atom_selected(self, index: int) -> None:
        """Atom selection changed via the model (e.g. from structure tree)."""
        self._viewport.highlight_atom(index)
        if 0 <= index < self._structure.n_atoms:
            sym = self._structure.symbols[index]
            pos = self._structure.positions[index]
            self._set_status(self.tr("Atom {}: {} at ({:.3f}, {:.3f}, {:.3f}) Å").format(
                index, sym, pos[0], pos[1], pos[2]
            ))

    def _update_status_bar(self) -> None:
        n = self._structure.n_atoms
        self._atom_count_label.setText(
            self.tr("Atoms: {}  |  {}").format(n, self._structure.chemical_formula)
        )
        if self._structure.n_atoms > 0:
            a, b, c = self._structure.cell_lengths
            alpha, beta, gamma = self._structure.cell_angles
            self._cell_label.setText(
                self.tr("a={:.2f} b={:.2f} c={:.2f}  α={:.1f}° β={:.1f}° γ={:.1f}°").format(
                    a, b, c, alpha, beta, gamma
                )
            )

    def _set_status(self, message: str) -> None:
        self._status_label.setText(message)
        logger.debug(message)
