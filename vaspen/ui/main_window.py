"""Main window — menu bar, toolbars, status bar, dock widgets.

This is the application shell. All views (3D viewport, structure tree,
editor panels) are docked or centrally embedded here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QSettings, QEvent, QTranslator
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QToolButton,
    QWidget,
    QVBoxLayout,
)

from vaspen import __version__
from vaspen.core.structure import StructureModel
from vaspen.core.file_io import PERIODIC_FORMATS, FileIO, resolve_format
from vaspen.ui.file_watch import FileChangeMonitor
from vaspen.ui.generate_all_dialog import GenerateAllDialog
from vaspen.ui.menu_button import MenuButton
from vaspen.ui.structure_tree import StructureTreePanel
from vaspen.ui.tools import ToolMode, confirm_disorder_loss
from vaspen.ui.viewport3d import Viewport3D
from vaspen.ui.welcome_page import WelcomePage
from vaspen.utils.config import AppConfig
from vaspen.utils.logger import logger
from vaspen.utils import theme

# Characters that are invalid in Windows filenames.
_INVALID_FILENAME_CHARS = '<>:"/\\|?*'


def _sanitize_filename(name: str) -> str:
    """Make a formula safe to use as a Windows filename.

    Strips characters invalid on Windows and replaces the decimal points
    of disordered formulas ("Fe2.05Ni0.5Co0.2") with underscores so the
    extension dot stays unambiguous.
    """
    cleaned = "".join(
        c for c in name if c not in _INVALID_FILENAME_CHARS)
    return cleaned.replace(".", "_").strip() or "structure"


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self) -> None:
        super().__init__()
        self._config = AppConfig()
        self._structure = StructureModel()
        # Non-modal preview dialogs (kept alive while open; one at a time)
        self._surface_dialog = None
        self._generate_dialog = None
        # True while the generate-all dialog previews NEB frames in the
        # viewport (the model itself is untouched during the preview)
        self._previewing_neb = False
        # Frame-edit context while a middle NEB image is previewed:
        # {"panel": PoscarPanel, "index": i}; viewport edit signals are
        # routed to the frame instead of the model while set.
        self._frame_edit: dict | None = None
        self._frame_selection: set[int] = set()
        # External file-change detection (§7.12): watched while the
        # model has a filepath; _reload_keep_camera carries the smart
        # camera decision into _on_structure_loaded during a reload.
        self._file_monitor = FileChangeMonitor(self)
        self._file_monitor.file_changed.connect(self._on_file_changed_on_disk)
        self._reload_keep_camera = False

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
        # Persisted render settings MUST be applied before the edit
        # toolbar wires its display-style menu check states (they derive
        # their initial values from the viewport settings — the old
        # hardcoded states would clobber the user's saved preferences).
        self._viewport.set_render_settings(self._config.render_settings)
        # Theme coherence: a default-background render setting follows
        # the theme (custom backgrounds are left untouched).
        self._sync_background_to_theme(theme.current_theme())
        self._create_edit_toolbar()
        self._create_dock_widgets()
        self._add_dock_toggle_actions()
        self._connect_signals()
        self._update_welcome_visibility()
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
        # ApplicationShortcut: undo/redo/delete must work regardless of
        # which widget has focus (viewport has StrongFocus and consumes
        # key events through the active tool).
        self.act_undo = QAction(self.tr("&Undo"), self)
        self.act_undo.setShortcut(QKeySequence.Undo)
        self.act_undo.setShortcutContext(Qt.ApplicationShortcut)
        self.act_undo.setEnabled(False)
        self.act_undo.triggered.connect(self._on_undo)

        self.act_redo = QAction(self.tr("&Redo"), self)
        # QKeySequence.Redo maps to Ctrl+Y on this platform; Ctrl+Shift+Z
        # is the other common convention — bind both explicitly.
        self.act_redo.setShortcuts(
            [QKeySequence.Redo, QKeySequence("Ctrl+Shift+Z"), QKeySequence("Ctrl+Y")])
        self.act_redo.setShortcutContext(Qt.ApplicationShortcut)
        self.act_redo.setEnabled(False)
        self.act_redo.triggered.connect(self._on_redo)

        self.act_preferences = QAction(self.tr("&Preferences..."), self)
        self.act_preferences.setStatusTip(self.tr("Configure settings"))
        self.act_preferences.triggered.connect(self._on_preferences)

        # ── View ──
        self.act_reset_view = QAction(self.tr("&Reset View"), self)
        self.act_reset_view.setStatusTip(self.tr("Reset the camera to the default view"))
        self.act_reset_view.triggered.connect(lambda: self._viewport.reset_view())

        self.act_lang_en = QAction(self.tr("English"), self)
        self.act_lang_en.setCheckable(True)
        self.act_lang_en.triggered.connect(lambda: self._switch_language("en"))

        self.act_lang_zh = QAction(self.tr("中文"), self)
        self.act_lang_zh.setCheckable(True)
        self.act_lang_zh.triggered.connect(lambda: self._switch_language("zh"))

        lang = self._config.language
        self.act_lang_en.setChecked(lang == "en")
        self.act_lang_zh.setChecked(lang == "zh")

        # ── Display style (View menu) ──
        self._style_actions: dict[str, QAction] = {}
        self._style_group = QActionGroup(self)
        self._style_group.setExclusive(True)
        for style, text in [
            ("ball_stick", self.tr("Ball & Stick")),
            ("cpk", self.tr("Space Filling (CPK)")),
            ("wireframe", self.tr("Wireframe")),
        ]:
            act = QAction(text, self)
            act.setCheckable(True)
            self._style_group.addAction(act)
            self._style_actions[style] = act

        self.act_show_cell = QAction(self.tr("Show Unit &Cell"), self)
        self.act_show_cell.setCheckable(True)
        self.act_show_cell.setStatusTip(self.tr("Show or hide the unit cell frame"))

        self.act_show_labels = QAction(self.tr("Atom &Labels"), self)
        self.act_show_labels.setCheckable(True)
        self.act_show_labels.setStatusTip(self.tr("Show element labels on atoms"))

        self.act_display_options = QAction(self.tr("Display &Options..."), self)
        self.act_display_options.setStatusTip(
            self.tr("Background, lighting, colors and effects"))
        self.act_display_options.triggered.connect(self._on_display_options)

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
        self.act_surface = QAction(self.tr("&Cleave Surface..."), self)
        self.act_surface.setStatusTip(self.tr("Cleave a surface/slab from the current structure"))
        self.act_surface.triggered.connect(self._on_surface)

        self.act_supercell = QAction(self.tr("&Supercell..."), self)
        self.act_supercell.setStatusTip(self.tr("Create a supercell"))
        self.act_supercell.triggered.connect(self._on_supercell)

        self.act_wrap_periodic = QAction(self.tr("Wrap in &Periodic Cell..."), self)
        self.act_wrap_periodic.setStatusTip(
            self.tr("Convert a molecule into a periodic structure (vacuum box)"))
        self.act_wrap_periodic.triggered.connect(self._on_wrap_periodic)

        self.act_rebox = QAction(self.tr("&Re-box Slab..."), self)
        self.act_rebox.setStatusTip(self.tr(
            "Re-apply the vacuum of a structure that already carries it "
            "(e.g. a cut surface)"))
        self.act_rebox.triggered.connect(self._on_rebox)

        self.act_edit_lattice = QAction(self.tr("Edit &Lattice..."), self)
        self.act_edit_lattice.setStatusTip(
            self.tr("Edit the unit cell parameters with live preview"))
        self.act_edit_lattice.triggered.connect(self._on_edit_lattice)

        self.act_symmetry = QAction(self.tr("Find &Symmetry..."), self)
        self.act_symmetry.setStatusTip(
            self.tr("Analyze the space group and symmetrize the structure"))
        self.act_symmetry.triggered.connect(self._on_symmetry)

        # ── 3D edit modes (exclusive; live in the edit toolbar) ──
        self._mode_actions: dict[ToolMode, QAction] = {}
        self._mode_group = QActionGroup(self)
        self._mode_group.setExclusive(True)
        for mode, text, tip in [
            (ToolMode.SELECT, self.tr("Select"), self.tr("Select atoms (Ctrl/Shift for multi-select, Shift+drag box-select)")),
            (ToolMode.ADD_ATOM, self.tr("Add Atom"), self.tr("Click empty space to add an atom")),
            (ToolMode.MOVE_ATOM, self.tr("Move"), self.tr("Drag an atom to move it (Esc cancels)")),
            (ToolMode.ROTATE, self.tr("Rotate"), self.tr("Drag to rotate the selection or molecule (Esc cancels)")),
            (ToolMode.DELETE, self.tr("Delete"), self.tr("Click an atom or bond to delete it")),
            (ToolMode.CREATE_BOND, self.tr("Create Bond"), self.tr("Click two atoms to create a bond")),
            (ToolMode.MEASURE_DISTANCE, self.tr("Distance"), self.tr("Click two atoms to measure their distance")),
            (ToolMode.MEASURE_ANGLE, self.tr("Angle"), self.tr("Click three atoms to measure the angle")),
            (ToolMode.MEASURE_TORSION, self.tr("Dihedral"), self.tr("Click four atoms to measure the dihedral")),
        ]:
            act = QAction(text, self)
            act.setCheckable(True)
            act.setStatusTip(tip)
            self._mode_group.addAction(act)
            self._mode_actions[mode] = act
        self._mode_actions[ToolMode.SELECT].setChecked(True)

        self.act_delete_selection = QAction(self.tr("Delete Selection"), self)
        self.act_delete_selection.setShortcut(QKeySequence.Delete)
        self.act_delete_selection.setShortcutContext(Qt.ApplicationShortcut)
        self.act_delete_selection.triggered.connect(self._on_delete_selection)
        self.addAction(self.act_delete_selection)  # window-level shortcut

        self.act_auto_bonds = QAction(self.tr("&Auto Detect Bonds"), self)
        self.act_auto_bonds.setCheckable(True)
        self.act_auto_bonds.setStatusTip(
            self.tr("Automatically detect bonds whenever the structure changes"))
        self.act_auto_bonds.setChecked(False)  # settled: off by default
        self.act_auto_bonds.toggled.connect(self._on_auto_bonds_toggled)

        self.act_detect_bonds = QAction(self.tr("Detect &Bonds"), self)
        self.act_detect_bonds.setStatusTip(
            self.tr("Re-detect all bonds from the current geometry"))
        self.act_detect_bonds.triggered.connect(self._on_detect_bonds)

        # ── Selection operations ──
        self.act_select_all = QAction(self.tr("Select &All"), self)
        self.act_select_all.setShortcut(QKeySequence.SelectAll)
        self.act_select_all.setShortcutContext(Qt.ApplicationShortcut)
        self.act_select_all.triggered.connect(
            lambda: self._structure.select_all())

        self.act_select_none = QAction(self.tr("Select &None"), self)
        self.act_select_none.triggered.connect(
            lambda: self._structure.select_none())

        self.act_select_invert = QAction(self.tr("&Invert Selection"), self)
        self.act_select_invert.triggered.connect(
            lambda: self._structure.select_invert())

        self.act_select_neighbors = QAction(self.tr("Select &Neighbors"), self)
        self.act_select_neighbors.triggered.connect(
            lambda: self._structure.select_neighbors())

        self.act_select_connected = QAction(self.tr("Select &Connected"), self)
        self.act_select_connected.triggered.connect(
            lambda: self._structure.select_connected())

        self.act_freeze = QAction(self.tr("&Freeze"), self)
        self.act_freeze.setStatusTip(
            self.tr("Freeze the selected atoms (VASP selective dynamics)"))
        self.act_freeze.setEnabled(False)
        self.act_freeze.triggered.connect(self._on_freeze_selection)

        self.act_unfreeze = QAction(self.tr("&Unfreeze"), self)
        self.act_unfreeze.setStatusTip(
            self.tr("Unfreeze the selected atoms (clear all fixed directions)"))
        self.act_unfreeze.setEnabled(False)
        self.act_unfreeze.triggered.connect(self._on_unfreeze_selection)

        # ── Bond order (applies to the selected bond) ──
        self._bond_order_actions: dict[int, QAction] = {}
        for order, text in [(1, self.tr("Single")), (2, self.tr("Double")),
                            (3, self.tr("Triple")), (4, self.tr("Aromatic"))]:
            act = QAction(text, self)
            act.setEnabled(False)
            act.triggered.connect(
                lambda checked, o=order: self._on_set_bond_order(o))
            self._bond_order_actions[order] = act

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
        self._menu_edit.addAction(self.act_select_all)
        self._menu_edit.addAction(self.act_select_none)
        self._menu_edit.addAction(self.act_select_invert)
        self._menu_edit.addAction(self.act_select_neighbors)
        self._menu_edit.addAction(self.act_select_connected)
        self._menu_edit.addSeparator()
        self._menu_edit.addAction(self.act_freeze)
        self._menu_edit.addAction(self.act_unfreeze)
        self._menu_edit.addSeparator()
        self._bond_order_menu = self._menu_edit.addMenu(self.tr("Bond &Order"))
        for act in self._bond_order_actions.values():
            self._bond_order_menu.addAction(act)
        self._menu_edit.addSeparator()
        self._menu_edit.addAction(self.act_auto_bonds)
        self._menu_edit.addSeparator()
        self._menu_edit.addAction(self.act_preferences)

        # View
        self._menu_view = mb.addMenu(self.tr("&View"))
        self._menu_view.addAction(self.act_reset_view)
        self._menu_view.addSeparator()
        self._menu_style = self._menu_view.addMenu(self.tr("Display &Style"))
        for act in self._style_actions.values():
            self._menu_style.addAction(act)
        self._menu_view.addAction(self.act_show_cell)
        self._menu_view.addAction(self.act_show_labels)
        self._menu_view.addSeparator()
        self._menu_view.addAction(self.act_display_options)
        self._menu_view.addSeparator()
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
        self._menu_tools.addAction(self.act_wrap_periodic)
        self._menu_tools.addAction(self.act_rebox)
        self._menu_tools.addSeparator()
        self._menu_tools.addAction(self.act_edit_lattice)
        self._menu_tools.addAction(self.act_symmetry)

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
        self._toolbar.addSeparator()
        self._toolbar.addAction(self.act_reset_view)
        self.addToolBar(self._toolbar)

    def _create_edit_toolbar(self) -> None:
        """3D editing toolbar: interaction modes, element picker, view tools."""
        # Buttons live in a FlowLayout container: when the toolbar gets
        # too long the overflowing buttons WRAP to the next row instead
        # of collapsing into the "»" extension chevron. The container is
        # added via addToolBar(area, widget) so it STRETCHES across the
        # full window width (a plain QToolBar.addWidget would leave the
        # container at its size-hint width and wrap every button onto
        # its own row).
        from vaspen.ui.flow_layout import FlowLayout

        container = QWidget(self)
        # Id selector for the theme QSS: keeps these buttons flat
        # (overrides the popupMode="1" bordered-chooser rule).
        container.setObjectName("edit_toolbar")
        flow = FlowLayout(container, margin=0, h_spacing=2, v_spacing=2)
        self._edit_buttons: list[QToolButton] = []

        def add_action(act: QAction) -> None:
            btn = QToolButton(container)
            btn.setDefaultAction(act)
            btn.setAutoRaise(True)
            btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
            self._edit_buttons.append(btn)
            flow.addWidget(btn)

        # Interaction modes (exclusive)
        for mode in (ToolMode.SELECT, ToolMode.ADD_ATOM, ToolMode.MOVE_ATOM,
                     ToolMode.ROTATE, ToolMode.DELETE, ToolMode.CREATE_BOND):
            act = self._mode_actions[mode]
            act.toggled.connect(
                lambda checked, m=mode: self._viewport.set_mode(m) if checked else None
            )
            add_action(act)
            if mode is ToolMode.ADD_ATOM:
                # Element picker right of Add Atom — the add-atom
                # workflow reads left-to-right: mode → element → click.
                # MenuButton (QToolButton + InstantPopup, same as the
                # View button): a QComboBox popup glitched in the flow
                # toolbar (appeared to expand multiple times).
                self._element_label = QLabel(self.tr("Element:"))
                flow.addWidget(self._element_label)
                self._element_btn = MenuButton(self)
                self._element_btn.addItems(
                    ("C", "H", "O", "N", "S", "P", "F", "Cl", "Si"))
                self._element_btn.currentIndexChanged.connect(
                    self._on_element_shortlist_changed)
                flow.addWidget(self._element_btn)
                # Ellipsis button — opens the periodic table for any
                # element outside the shortlist
                self._element_more_btn = QToolButton(self)
                self._element_more_btn.setText("…")
                self._element_more_btn.setToolTip(self.tr(
                    "Pick any element from the periodic table"))
                self._element_more_btn.clicked.connect(self._on_pick_element)
                flow.addWidget(self._element_more_btn)
                self._current_element = "C"
                self._set_current_element("C")
        add_action(self.act_freeze)
        add_action(self.act_unfreeze)

        # One-shot bond detection — useful while Auto Detect Bonds is off
        add_action(self.act_detect_bonds)

        for mode in (ToolMode.MEASURE_DISTANCE, ToolMode.MEASURE_ANGLE,
                     ToolMode.MEASURE_TORSION):
            act = self._mode_actions[mode]
            act.toggled.connect(
                lambda checked, m=mode: self._viewport.set_mode(m) if checked else None
            )
            add_action(act)

        add_action(self.act_reset_view)

        self._view_dir_btn = QToolButton(self)
        self._view_dir_btn.setText(self.tr("View"))
        self._view_dir_btn.setToolTip(self.tr("Align the camera with a world axis"))
        self._view_dir_btn.setPopupMode(QToolButton.InstantPopup)
        self._view_dir_menu = QMenu(self)
        for label, az, el, up in [
            (self.tr("Front (+z)"), 0.0, 0.0, None),
            (self.tr("Back (−z)"), 180.0, 0.0, None),
            (self.tr("Left (−x)"), -90.0, 0.0, None),
            (self.tr("Right (+x)"), 90.0, 0.0, None),
            (self.tr("Top (+y)"), 0.0, 89.9, (0.0, 0.0, 1.0)),
            (self.tr("Bottom (−y)"), 0.0, -89.9, (0.0, 0.0, -1.0)),
        ]:
            action = QAction(label, self)
            # QMenu.addAction(text, callable) calls plain lambdas with NO
            # arguments — use a QAction with an optional checked instead.
            action.triggered.connect(
                lambda checked=False, a=az, e=el, u=up: self._set_view_direction(a, e, u)
            )
            self._view_dir_menu.addAction(action)
        self._view_dir_btn.setMenu(self._view_dir_menu)
        flow.addWidget(self._view_dir_btn)

        add_action(self.act_undo)
        add_action(self.act_redo)

        # A plain widget bar above the viewport (NOT a QToolBar — the
        # toolbar area sizes toolbars to their content, which would
        # leave the flow layout ~180px wide and wrap every button onto
        # its own row). Expanding policy makes it span the full width.
        container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._edit_toolbar = container
        self._central_layout.insertWidget(0, container)

        # Display style wiring (the viewport exists at this point).
        # Initial check states come from the (persisted) viewport
        # settings, not hardcoded values.
        for style, act in self._style_actions.items():
            act.toggled.connect(
                lambda checked, s=style: self._viewport.set_structure_style(s)
                if checked else None
            )
        rs = self._viewport.render_settings()
        self._style_actions[rs.style].setChecked(True)
        self.act_show_cell.toggled.connect(self._viewport.set_show_cell)
        self.act_show_cell.setChecked(rs.show_cell)
        self.act_show_labels.toggled.connect(self._viewport.set_show_labels)
        self.act_show_labels.setChecked(rs.show_labels)

    def _set_view_direction(self, azimuth: float, elevation: float,
                            up: tuple[float, float, float] | None = None) -> None:
        """Point the camera along a world-axis direction (View menu)."""
        self._viewport.set_view_direction(azimuth, elevation, up)

    def _on_element_shortlist_changed(self, index: int) -> None:
        """Shortlist menu item picked (-1 = cleared, e.g. by a table pick)."""
        if index >= 0:
            self._set_current_element(self._element_btn.itemText(index))

    def _set_current_element(self, symbol: str) -> None:
        """Set the active add-atom element; sync button text and checks."""
        self._current_element = symbol
        if self._element_btn.findText(symbol) >= 0:
            self._element_btn.setCurrentText(symbol)
        else:
            # Periodic-table picks are not in the shortlist — show the
            # symbol with no menu item checked.
            self._element_btn.setCurrentIndex(-1)
            self._element_btn.setText(symbol)
        self._viewport.current_element = symbol

    def _on_pick_element(self) -> None:
        """Open the periodic table to pick any element (shortlist "…")."""
        from vaspen.ui.periodic_table_dialog import PeriodicTableDialog

        dlg = PeriodicTableDialog(self)
        if dlg.exec() != PeriodicTableDialog.DialogCode.Accepted:
            return
        if dlg.selected_symbol:
            self._set_current_element(dlg.selected_symbol)

    def _on_display_options(self) -> None:
        """Open the Display Options dialog (live preview, persisted on OK)."""
        from vaspen.ui.display_options_dialog import DisplayOptionsDialog

        dlg = DisplayOptionsDialog(self._viewport, self)
        dlg.exec()
        # Re-sync the View menu with the viewport (the dialog may have
        # changed style / cell / labels; both paths are idempotent).
        self._sync_display_menu()

    def _sync_display_menu(self) -> None:
        """Mirror the viewport's render settings into the View menu."""
        rs = self._viewport.render_settings()
        for style, act in self._style_actions.items():
            act.setChecked(style == rs.style)
        self.act_show_cell.setChecked(rs.show_cell)
        self.act_show_labels.setChecked(rs.show_labels)

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
        # The central area hosts the viewport plus the wrapping edit
        # bar above it (a plain widget bar — the QToolBar area sizes
        # toolbars to their content and cannot host a full-width
        # wrapping layout).
        self._central_container = QWidget(self)
        self._central_layout = QVBoxLayout(self._central_container)
        self._central_layout.setContentsMargins(0, 0, 0, 0)
        self._central_layout.setSpacing(0)
        # Stack: welcome page (index 0, no structure loaded) / viewport
        # (index 1). Visibility is keyed on the MODEL's atom count —
        # never on viewport buffers, which previews replace temporarily.
        self._central_stack = QStackedWidget(self._central_container)
        self._welcome_page = WelcomePage(self._central_stack)
        self._central_stack.addWidget(self._welcome_page)
        self._viewport = Viewport3D(self._central_stack)
        self._central_stack.addWidget(self._viewport)
        self._central_layout.addWidget(self._central_stack, 1)
        self.setCentralWidget(self._central_container)
        # The recent menu was built before the welcome page existed —
        # seed the list here.
        self._welcome_page.set_recent_files(self._config.recent_files)

    def _update_welcome_visibility(self) -> None:
        """Show the welcome page while the model is empty, the viewport
        otherwise. Called on load, on every structural change (all
        atoms can be deleted) and after New (the only path back to an
        empty model — it emits no signal)."""
        self._central_stack.setCurrentIndex(
            0 if self._structure.n_atoms == 0 else 1)

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

        # Measurement dock (right side) — distance/angle/dihedral list
        from vaspen.ui.measurement import MeasurementManager, MeasurementPanel

        self._measurement_manager = MeasurementManager(self._structure)
        self._dock_measure = QDockWidget(self.tr("Measurements"), self)
        self._dock_measure.setObjectName("dock_measure")
        self._dock_measure.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self._measurement_panel = MeasurementPanel(self._measurement_manager)
        self._dock_measure.setWidget(self._measurement_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self._dock_measure)

        # Atom properties dock (right side) — edit the selected atom
        from vaspen.ui.atom_properties import AtomPropertiesPanel

        self._dock_props = QDockWidget(self.tr("Properties"), self)
        self._dock_props.setObjectName("dock_props")
        self._dock_props.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self._atom_props = AtomPropertiesPanel(self._structure)
        self._dock_props.setWidget(self._atom_props)
        self.addDockWidget(Qt.RightDockWidgetArea, self._dock_props)

    def _add_dock_toggle_actions(self) -> None:
        """View-menu checkables for the three docks (created after the
        menu bar, so appended here). The docks are closable via their
        title-bar X — without these toggles a closed dock could never
        be re-opened."""
        self._menu_view.addSeparator()
        for dock in (self._dock_structure, self._dock_measure, self._dock_props):
            act = dock.toggleViewAction()
            act.setText(dock.windowTitle())
            self._menu_view.addAction(act)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._connect_model_signals(self._structure)
        # Viewport ↔ model bidirectional wiring
        self._viewport.atom_clicked.connect(self._on_atom_clicked)
        self._viewport.background_clicked.connect(self._on_background_clicked)
        self._viewport.bond_clicked.connect(self._on_bond_clicked)
        self._viewport.atoms_selected.connect(self._on_atoms_selected)
        self._viewport.atom_place_requested.connect(self._on_atom_place_requested)
        self._viewport.atoms_moved.connect(self._on_atoms_moved)
        self._viewport.bond_created.connect(self._on_bond_created)
        self._viewport.delete_requested.connect(self._on_delete_requested)
        self._viewport.measurement_added.connect(self._on_measurement_added)
        self._viewport.mode_changed.connect(self._on_mode_changed)
        self._viewport.frozen_drag_blocked.connect(
            lambda: self._set_status(
                self.tr("Cannot move frozen atoms — unfreeze them first.")))
        # Any manager change (add/remove/clear/sync) pushes to the viewport
        self._measurement_manager.changed.connect(self._push_measurements)
        # Welcome page quick actions
        self._welcome_page.open_file_requested.connect(self._open_file)
        self._welcome_page.new_requested.connect(self._on_new)
        self._welcome_page.open_requested.connect(self._on_welcome_open)
        self._welcome_page.browse_requested.connect(self._on_open)

    def _connect_model_signals(self, model: StructureModel) -> None:
        """Connect a StructureModel's signals to window/UI updates.

        Called once per model instance (models are replaced on New).
        """
        model.structure_loaded.connect(self._on_structure_loaded)
        model.structure_modified.connect(self._on_structure_modified)
        model.atom_selected.connect(self._on_atom_selected)
        model.selection_cleared.connect(self._on_selection_cleared)
        model.selection_changed.connect(self._on_selection_changed)
        model.bond_selection_changed.connect(self._on_bond_selection_changed)

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _on_new(self) -> None:
        """Create a new empty structure (for now, reset to empty)."""
        # TODO: Launch a "new structure" dialog (crystal builder, import molecule, etc.)
        if self._structure.n_atoms > 0 or self._structure.is_dirty:
            # Only ask when there is something to discard — atoms, or
            # undo history (a structure whose atoms were all deleted
            # is dirty and Ctrl+Z could still restore them).
            reply = QMessageBox.question(
                self,
                self.tr("New Structure"),
                self.tr("Discard current changes and create a new empty structure?"),
            )
            if reply != QMessageBox.Yes:
                return
        self._structure = StructureModel()
        self._connect_model_signals(self._structure)
        self._structure_tree.set_model(self._structure)
        self._measurement_manager.clear()
        self._measurement_manager.set_model(self._structure)
        self._atom_props.set_model(self._structure)
        self._viewport.set_structure(None)
        # New = an explicit empty SESSION: enter the viewport workspace
        # (the welcome page stays for a never-started state; it comes
        # back when the model is emptied again).
        self._central_stack.setCurrentIndex(1)
        self._update_status_bar()
        self._update_edit_actions()
        self._set_status(self.tr("New structure created."))

    def _on_open(self) -> None:
        """Open a structure file via the file dialog (File menu,
        toolbar, and the welcome page's Browse button)."""
        file_filter = FileIO.file_filter(for_writing=False)
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open Structure File"),
            self._config.last_directory,
            file_filter,
        )
        if filepath:
            self._open_file(filepath)

    def _on_welcome_open(self) -> None:
        """Open the SELECTED recent file (welcome page's Open button —
        no dialog; the button is disabled without a selection)."""
        path = self._welcome_page.selected_file()
        if path:
            self._open_file(path)

    def _open_file(self, filepath: str) -> None:
        """Load a structure from the given path."""
        # Live preview dialogs hold a stale snapshot of the model
        # (cached slabs / interpolated images) — close them first; their
        # finished handlers rebind the viewport before the new structure
        # loads. dropEvent and the recent-files menu reach this method
        # while the editing entry points are paused.
        for dlg in (self._surface_dialog, self._generate_dialog):
            if dlg is not None:
                dlg.close()
        try:
            atoms = FileIO.read(filepath)
            self._structure.load_atoms(atoms, filepath)  # emits structure_loaded once
            self._config.add_recent_file(filepath)
            self._config.last_directory = str(Path(filepath).parent)
            self._update_recent_ui()
            self._set_status(self.tr("Loaded: {}").format(filepath))
            logger.info("Opened file: %s", filepath)
        except Exception as e:
            logger.exception("Failed to open file: %s", filepath)
            QMessageBox.critical(
                self,
                self.tr("Open Failed"),
                self.tr("Could not open file:\n{}").format(str(e)),
            )

    def _ensure_periodic_for(self, target_fmt: str) -> bool:
        """Ensure the model is periodic when the target format requires it.

        If the model is molecule-like and the target format is vasp-family or
        cif, shows the wrap dialog and converts the model in place (undoable).
        Slabs (partially periodic) pass through untouched — re-boxing them
        would destroy their lattice.

        Args:
            target_fmt: ASE format name ("vasp", "cif", "xyz", ...) or "".

        Returns:
            True if saving may proceed, False if the user cancelled.
        """
        if not target_fmt or target_fmt not in PERIODIC_FORMATS:
            return True
        if self._structure.is_periodic or self._structure.n_atoms == 0:
            return True  # empty structures fall through to FileIO's error
        return self._wrap_molecule_to_periodic()

    def _wrap_molecule_to_periodic(self) -> bool:
        """Show the wrap dialog and convert the molecule in place (undoable).

        Shared by the save-to-periodic-format flow (_ensure_periodic_for)
        and the Tools → Wrap in Periodic Cell action. The caller ensures
        the model is molecule-like (non-periodic, non-empty).

        Returns:
            True when the conversion happened, False on cancel.
        """
        from vaspen.ui.periodic_wrap_dialog import PeriodicWrapDialog

        dlg = PeriodicWrapDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False
        self._structure.make_periodic(dlg.padding)
        self._set_status(
            self.tr("Wrapped in periodic cell ({} Å vacuum).").format(dlg.padding))
        return True

    def _confirm_disorder_poscar_save(self) -> bool:
        """Warn before saving a partially-occupied structure to VASP.

        The POSCAR format has no occupancy field — the saved file will
        contain only the dominant species per site and the composition
        information is lost. Requires the user's confirmation (settled
        policy 2026-08-14: warn, do not block).
        """
        if not self._structure.has_disorder:
            return True
        reply = QMessageBox.warning(
            self,
            self.tr("Partial Occupancy"),
            self.tr(
                "This structure has partial occupancy (disorder).\n"
                "The VASP POSCAR format does not support fractional "
                "occupancy — the saved file will contain only the "
                "dominant species per site and the composition "
                "information will be lost.\n\n"
                "Continue saving?"),
            QMessageBox.Save | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return reply == QMessageBox.Save

    def _confirm_disorder_loss(self, operation: str) -> bool:
        """Warn that an operation will discard fractional-occupancy data.

        Some structure operations (supercell, symmetrize) rebuild the
        atoms from scratch and cannot carry the site compositions over.
        Requires the user's confirmation when the structure is
        disordered.
        """
        if not self._structure.has_disorder:
            return True
        # the shared helper (vaspen.ui.tools.confirm_disorder_loss) —
        # also used by the cleave / re-box / symmetrize dialogs
        return confirm_disorder_loss(self, operation)

    def _on_save(self) -> None:
        """Save the current structure."""
        fp = self._structure.filepath
        if fp:
            try:
                if not self._ensure_periodic_for(resolve_format(fp) or ""):
                    return
                if (resolve_format(fp) == "vasp"
                        and not self._confirm_disorder_poscar_save()):
                    return
                self._structure.save(fp, direct=self._config.poscar_coords_direct)
                self._sync_file_watcher()  # fresh baseline: our own write
                self._set_status(self.tr("Saved: {}").format(fp))
            except Exception as e:
                QMessageBox.critical(self, self.tr("Save Failed"), str(e))
        else:
            self._on_save_as()

    def _default_save_name(self) -> str:
        """Default filename for Save As: loaded stem, else the formula.

        The formula is sanitized for Windows filenames (invalid characters
        stripped, decimal points in disordered formulas become '_').
        """
        fp = self._structure.filepath
        if fp:
            stem = Path(fp).stem
        else:
            stem = _sanitize_filename(self._structure.chemical_formula)
        return stem + ".cif"

    def _on_save_as(self) -> None:
        """Save the structure to a new file."""
        file_filter = FileIO.file_filter(for_writing=True)
        # Default to the first (CIF) category so the suggested extension
        # matches the pre-selected filter.
        selected = file_filter.split(";;")[0]
        default_path = str(
            Path(self._config.last_directory) / self._default_save_name())
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save Structure As"),
            default_path,
            file_filter,
            selected,
        )
        if filepath:
            try:
                if not self._ensure_periodic_for(resolve_format(filepath) or ""):
                    return
                if (resolve_format(filepath) == "vasp"
                        and not self._confirm_disorder_poscar_save()):
                    return
                self._structure.save(filepath, direct=self._config.poscar_coords_direct)
                self._config.add_recent_file(filepath)
                self._config.last_directory = str(Path(filepath).parent)
                self._update_recent_ui()
                self._sync_file_watcher()  # fresh baseline: our own write
                self._set_status(self.tr("Saved: {}").format(filepath))
            except Exception as e:
                QMessageBox.critical(self, self.tr("Save Failed"), str(e))

    # -- External file changes (§7.12) ----------------------------------

    def _sync_file_watcher(self) -> None:
        """Point the file monitor at the model's current filepath.

        Called wherever the filepath changes: open (structure_loaded),
        the derived-structure reset_filepath sites, New (empty model),
        and after a successful save (fresh baseline — the GUI's own
        write must not prompt).
        """
        fp = self._structure.filepath
        if fp:
            self._file_monitor.watch(fp)
        else:
            self._file_monitor.clear()

    def _on_file_changed_on_disk(self, path: str) -> None:
        """The watched file changed on disk (external edit / MCP write).

        Godot-style policy (settled §7.12): a clean model reloads
        silently; unsaved changes trigger a confirm — reloading
        discards them, ignoring keeps memory and re-bases the monitor
        so only FUTURE changes fire again.
        """
        if path != self._structure.filepath or self._structure.n_atoms == 0:
            return  # stale signal — defensive
        try:
            new_atoms = FileIO.read(path)
        except Exception as e:
            # Most likely a mid-write state; do not touch the model —
            # the next settled write re-fires the monitor.
            logger.exception("Reload failed: %s", path)
            self._set_status(self.tr("Reload failed: {}").format(str(e)))
            return
        if self._structure.is_dirty:
            reply = QMessageBox.question(
                self,
                self.tr("File Changed on Disk"),
                self.tr(
                    "{}\n\nhas been modified on disk. Reload and "
                    "discard your unsaved changes?").format(path),
            )
            if reply != QMessageBox.Yes:
                self._file_monitor.rebase()
                return
        # Smart camera (settled): same composition, count and cell →
        # the geometry is only nudged (e.g. an MCP edit) — keep the
        # user's view; anything else re-fits.
        same_geometry = (
            len(new_atoms) == self._structure.n_atoms
            and new_atoms.get_chemical_formula()
            == self._structure.atoms.get_chemical_formula()
            and np.allclose(np.asarray(new_atoms.get_cell().array),
                            np.asarray(self._structure.cell)))
        self._reload_keep_camera = same_geometry
        try:
            self._structure.load_atoms(new_atoms, path)
        finally:
            self._reload_keep_camera = False
        self._set_status(self.tr("Reloaded from disk: {}").format(path))

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
                if not self._ensure_periodic_for("vasp"):
                    return
                if not self._confirm_disorder_poscar_save():
                    return
                self._structure.save(filepath, fmt="vasp",
                                     direct=self._config.poscar_coords_direct)
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
        # VASP input files require a periodic structure (POSCAR lattice +
        # KPOINTS mesh); offer the wrap dialog for molecules first.
        if not self._ensure_periodic_for("vasp"):
            return
        # POSCAR cannot carry fractional occupancy — confirm first.
        if not self._confirm_disorder_poscar_save():
            return

        if self._generate_dialog is not None:
            self._generate_dialog.raise_()
            self._generate_dialog.activateWindow()
            return

        # Non-modal (same pattern as the surface dialog): a modal dialog
        # would cover the 3D viewport, hiding the NEB frame preview.
        # Files are written only when the dialog's Generate is clicked.
        try:
            dlg = GenerateAllDialog(
                self._structure,
                self,
                preview_callback=self._preview_image_atoms,
                default_task=self._config.default_calc_type,
            )
        except Exception as e:
            # The dialog must never fail silently (e.g. a molecule that
            # reached it without being wrapped) — surface the error.
            logger.exception("Failed to open generate dialog")
            QMessageBox.critical(
                self, self.tr("Generation Failed"), str(e))
            return
        dlg.finished.connect(lambda _result: self._on_generate_dialog_finished(dlg))
        self._generate_dialog = dlg
        self._set_preview_editing_enabled(False)
        self._viewport.cancel_active_tool()
        # Leave editing modes entirely — an active ADD/MOVE/DELETE tool
        # would keep consuming clicks and mutate the model during the
        # "paused" preview (the mode buttons are disabled, not the tool).
        self._viewport.set_mode(ToolMode.SELECT)
        dlg.show()

    def _preview_image_atoms(self, atoms, index=None, editable=False,
                             fixed=None, bonds=None, panel=None) -> None:
        """Temporarily show a NEB image frame in the 3D viewport.

        The model is untouched; the viewport is rebound to the model
        when the dialog closes. The camera refits on the first preview
        and is preserved across subsequent frames so the images can be
        compared directly.

        Middle frames enter frame-edit mode: viewport edit signals are
        routed to the frame (move/rotate atoms, delete/add bonds)
        instead of the model; the initial and final frames stay locked.
        """
        if atoms is None:
            # The dialog cleared its images (new endpoints were
            # browsed) — exit frame-edit and re-show the model.
            self._frame_edit = None
            self._frame_selection = set()
            self._previewing_neb = False
            self._rebind_viewport_to_model()
            self._update_frame_edit_actions(False)
            self._update_edit_actions()
            return
        self._viewport.set_structure(
            atoms,
            reset_view=not self._previewing_neb,
            bonds=bonds,
            fixed=fixed,
        )
        # set_structure() only schedules a repaint, and the GL viewport
        # can defer it for seconds — force an immediate synchronous
        # repaint so clicking a frame updates the view instantly.
        self._viewport.repaint()
        self._previewing_neb = True

        if editable and panel is not None:
            self._frame_edit = {"panel": panel, "index": index}
            self._frame_selection = set()
            self._viewport.set_highlight(set())
            self._set_status(self.tr(
                "Editing NEB image {:02d} — frozen atoms locked, atoms "
                "cannot be deleted.").format(index))
        else:
            self._frame_edit = None
            if index is not None:
                self._set_status(
                    self.tr("The initial and final frames are locked."))
        self._update_frame_edit_actions(editable)

    def _on_generate_dialog_finished(self, dlg) -> None:
        """Restore viewport + editing entry points after the dialog closes."""
        self._frame_edit = None
        self._frame_selection = set()
        if self._generate_dialog is dlg:
            self._generate_dialog = None
            # Both preview dialogs can be open at once — editing stays
            # paused until the LAST one closes.
            if self._surface_dialog is None:
                self._set_preview_editing_enabled(True)
                # re-derive stateful actions after the unconditional
                # re-enable (see _on_surface_dialog_finished)
                self._update_edit_actions()
                self._sync_auto_bonds_action()
                self._on_selection_changed()
                self._on_bond_selection_changed()
        self._previewing_neb = False
        self._rebind_viewport_to_model()
        dlg.deleteLater()

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _on_surface(self) -> None:
        from vaspen.ui.surface_dialog import SurfaceDialog

        if self._structure.n_atoms == 0:
            QMessageBox.warning(
                self,
                self.tr("No Structure"),
                self.tr("Load a bulk structure before cleaving a surface."),
            )
            return
        if not self._structure.is_periodic:
            QMessageBox.information(
                self,
                self.tr("Cleave Surface"),
                self.tr("Cleaving a surface requires a periodic structure "
                        "(a full-rank cell with periodic boundary "
                        "conditions)."),
            )
            return
        if self._surface_dialog is not None:
            self._surface_dialog.raise_()
            self._surface_dialog.activateWindow()
            return

        # Non-modal: the viewport stays rotatable while the dialog is
        # open. Editing entry points are paused to prevent conflicts
        # with the live preview (see _set_preview_editing_enabled).
        dlg = SurfaceDialog(self._structure, self._viewport, self)
        dlg.accepted.connect(lambda: self._apply_surface_result(dlg))
        dlg.finished.connect(lambda _result: self._on_surface_dialog_finished(dlg))
        self._surface_dialog = dlg
        self._set_preview_editing_enabled(False)
        self._viewport.cancel_active_tool()
        self._viewport.set_mode(ToolMode.SELECT)
        dlg.show()

    def _apply_surface_result(self, dlg) -> None:
        """Apply the accepted slab to the model (one undo step)."""
        if dlg.result_structure is not None:
            self._structure.replace_atoms(dlg.result_structure.atoms)
            self._structure.reset_filepath()  # never silently overwrite the bulk file
            self._sync_file_watcher()
            self._set_status(self.tr("Cleaved surface applied."))

    def _on_surface_dialog_finished(self, dlg) -> None:
        """Re-enable editing entry points after the dialog closes."""
        if self._surface_dialog is dlg:
            self._surface_dialog = None
            # Both preview dialogs can be open at once — editing stays
            # paused until the LAST one closes.
            if self._generate_dialog is None:
                self._set_preview_editing_enabled(True)
                # re-enabling enables everything unconditionally —
                # re-derive the stateful actions (undo/redo stacks,
                # auto-bonds check state, selection-dependent tools)
                self._update_edit_actions()
                self._sync_auto_bonds_action()
                self._on_selection_changed()
                self._on_bond_selection_changed()
            dlg.deleteLater()

    def _set_preview_editing_enabled(self, enabled: bool) -> None:
        """Pause structure-editing entry points while the preview dialog is open.

        Rotate/zoom/pan stay active (mouse handlers, not actions) so the
        user can inspect the slab from any angle. Save/generate actions
        stay enabled — the model is untouched during preview. Pausing is
        belt-and-braces on top of the click guards in _on_bond_clicked /
        _on_background_clicked, which keep viewport clicks local to the
        preview structure.
        """
        for act in (self.act_new, self.act_open, self.act_surface,
                    self.act_supercell, self.act_edit_lattice,
                    self.act_symmetry,
                    self.act_wrap_periodic, self.act_rebox,
                    self.act_delete_selection, self.act_undo, self.act_redo,
                    self.act_detect_bonds, self.act_freeze, self.act_unfreeze,
                    self.act_select_all, self.act_select_none,
                    self.act_select_invert, self.act_select_neighbors,
                    self.act_select_connected,
                    self.act_auto_bonds,
                    *self._bond_order_actions.values()):
            act.setEnabled(enabled)
        for mode, act in self._mode_actions.items():
            if mode is not ToolMode.SELECT:
                act.setEnabled(enabled)
        self._element_label.setEnabled(enabled)
        self._element_btn.setEnabled(enabled)
        self._element_more_btn.setEnabled(enabled)
        self._dock_structure.setEnabled(enabled)
        self._dock_props.setEnabled(enabled)

    #: Modes usable while editing a middle NEB frame (add/measure/etc.
    #: stay paused; DELETE is bond-only — atom deletion is rejected in
    #: the handler).
    _FRAME_EDIT_MODES = (
        ToolMode.SELECT, ToolMode.MOVE_ATOM, ToolMode.ROTATE,
        ToolMode.DELETE, ToolMode.CREATE_BOND,
    )

    def _update_frame_edit_actions(self, editable: bool) -> None:
        """Enable/disable the frame-edit toolset inside the paused window.

        Called whenever the selected NEB frame changes: middle frames
        unlock move/rotate/delete/create-bond + per-frame undo/redo;
        endpoint frames (and everything else) keep them disabled.
        """
        for mode, act in self._mode_actions.items():
            if mode in self._FRAME_EDIT_MODES:
                act.setEnabled(editable)
        self.act_undo.setEnabled(editable)
        self.act_redo.setEnabled(editable)

    def _refresh_frame_edit_view(self) -> None:
        """Re-push the edited frame into the viewport (bonds + flags)."""
        if self._frame_edit is None:
            return
        panel = self._frame_edit["panel"]
        index = self._frame_edit["index"]
        data = panel.frame_preview_data(index)
        if data is None:
            # The dialog cleared its images (new endpoints) — the frame
            # context is gone.
            self._frame_edit = None
            return
        atoms, _editable, flags, bonds = data
        self._viewport.set_structure(atoms, reset_view=False,
                                     bonds=bonds, fixed=flags)
        self._viewport.set_highlight(self._frame_selection)
        self._viewport.repaint()

    def _on_supercell(self) -> None:
        from vaspen.core.builder import StructureBuilder
        from vaspen.ui.supercell_dialog import SupercellDialog

        dlg = SupercellDialog(self)
        if dlg.exec() == SupercellDialog.Accepted and dlg.factors is not None:
            # Building a supercell drops the fractional-occupancy info —
            # confirm first for disordered structures.
            if not self._confirm_disorder_loss(self.tr("Creating a supercell")):
                return
            StructureBuilder.make_supercell(self._structure, dlg.factors)
            self._structure.reset_filepath()  # derived structure → Save As
            self._sync_file_watcher()
            self._set_status(self.tr("Supercell {}×{}×{} created.").format(*dlg.factors))

    def _on_wrap_periodic(self) -> None:
        """Tools → Wrap in Periodic Cell: convert a molecule in place.

        Shares the wrap dialog + make_periodic kernel with the
        save-to-periodic-format flow (_wrap_molecule_to_periodic).
        """
        if self._structure.n_atoms == 0:
            QMessageBox.information(
                self,
                self.tr("Wrap in Periodic Cell"),
                self.tr("Load a structure first."),
            )
            return
        if self._structure.is_periodic:
            QMessageBox.information(
                self,
                self.tr("Wrap in Periodic Cell"),
                self.tr("The structure is already periodic."),
            )
            return
        self._wrap_molecule_to_periodic()

    def _on_rebox(self) -> None:
        """Tools → Re-box Slab: re-apply vacuum to a slab-like structure.

        The counterpart of Cleave Surface for structures that already
        carry vacuum. The atom order is unchanged (pure c-translations
        + re-vacuum), so frozen flags and magnetic moments are carried
        over 1:1 — unlike cleave, which builds a new atom set.
        """
        from vaspen.ui.rebox_dialog import ReBoxDialog

        if self._structure.n_atoms == 0:
            QMessageBox.information(
                self,
                self.tr("Re-box Slab"),
                self.tr("Load a structure first."),
            )
            return
        if not self._structure.is_periodic:
            QMessageBox.information(
                self,
                self.tr("Re-box Slab"),
                self.tr("Re-boxing requires a periodic structure "
                        "(a full-rank cell with periodic boundary "
                        "conditions)."),
            )
            return
        dlg = ReBoxDialog(self._structure, self)
        if (dlg.exec() == ReBoxDialog.Accepted
                and dlg.result_atoms is not None):
            # replace_atoms: one undo step; atom order preserved, so the
            # frozen flags and initial moments carry over 1:1.
            self._structure.replace_atoms(
                dlg.result_atoms,
                fixed_flags=self._structure.fixed_flags,
                magmoms=self._structure.magmoms)
            self._structure.reset_filepath()  # derived structure → Save As
            self._sync_file_watcher()
            self._set_status(self.tr("Slab re-boxed (vacuum along c)."))

    def _on_edit_lattice(self) -> None:
        from vaspen.ui.lattice_dialog import LatticeDialog

        if not self._structure.is_periodic:
            QMessageBox.information(
                self,
                self.tr("Edit Lattice"),
                self.tr("Lattice editing requires a periodic structure "
                        "(a full-rank cell with periodic boundary conditions)."),
            )
            return
        dlg = LatticeDialog(self._structure, self._viewport, self)
        dlg.exec()  # applies to the model on Accept (one undo step)

    def _on_symmetry(self) -> None:
        from vaspen.ui.symmetry_dialog import SymmetryDialog

        if self._structure.any_fixed:
            QMessageBox.warning(
                self,
                self.tr("Symmetry"),
                self.tr("Some atoms are frozen. Unfreeze them before symmetrizing."),
            )
            return
        dlg = SymmetryDialog(self._structure, self)
        if (dlg.exec() == SymmetryDialog.Accepted
                and dlg.result_atoms is not None):
            # replace_atoms: one undo step; bonds/selection reset (new indices)
            self._structure.replace_atoms(dlg.result_atoms)
            self._structure.reset_filepath()  # derived structure → Save As
            self._sync_file_watcher()
            self._set_status(self.tr("Structure symmetrized."))

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
        # Same for the theme: live switch, no-op when unchanged.
        self._apply_theme(self._config.theme)

    def _apply_theme(self, name: str) -> None:
        """Apply a theme app-wide (no restart), keep the GL background
        coherent and re-tint the element column."""
        if name == theme.current_theme():
            return
        theme.apply_theme(QApplication.instance(), name)
        self._sync_background_to_theme(name)
        self._structure_tree.refresh()  # element-symbol column colors

    def _sync_background_to_theme(self, name: str) -> None:
        """Swap the GL background to the theme default when the user
        never customized it (still the other theme's default); custom
        backgrounds are untouched. Persisted so a restart stays
        coherent. Derived from the LIVE viewport settings — the
        persisted config may lag behind unpersisted live toggles
        (View-menu cell/label switches), and pushing a stale snapshot
        would silently re-enable them."""
        rs = self._viewport.render_settings()
        new_rs = theme.sync_background_for_theme(rs, name)
        if new_rs is not rs:
            self._viewport.set_render_settings(new_rs)
            self._config.render_settings = new_rs
            self._config.sync()

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
        self.act_auto_bonds.setText(self.tr("&Auto Detect Bonds"))
        self.act_auto_bonds.setStatusTip(
            self.tr("Automatically detect bonds whenever the structure changes"))
        self.act_detect_bonds.setText(self.tr("Detect &Bonds"))
        self.act_detect_bonds.setStatusTip(
            self.tr("Re-detect all bonds from the current geometry"))
        self.act_select_all.setText(self.tr("Select &All"))
        self.act_select_none.setText(self.tr("Select &None"))
        self.act_select_invert.setText(self.tr("&Invert Selection"))
        self.act_select_neighbors.setText(self.tr("Select &Neighbors"))
        self.act_select_connected.setText(self.tr("Select &Connected"))
        self.act_freeze.setText(self.tr("&Freeze"))
        self.act_freeze.setStatusTip(
            self.tr("Freeze the selected atoms (VASP selective dynamics)"))
        self.act_unfreeze.setText(self.tr("&Unfreeze"))
        self.act_unfreeze.setStatusTip(
            self.tr("Unfreeze the selected atoms (clear all fixed directions)"))
        self._bond_order_menu.setTitle(self.tr("Bond &Order"))
        for order, text in [(1, self.tr("Single")), (2, self.tr("Double")),
                            (3, self.tr("Triple")), (4, self.tr("Aromatic"))]:
            self._bond_order_actions[order].setText(text)
        self.act_preferences.setText(self.tr("&Preferences..."))
        self.act_preferences.setStatusTip(self.tr("Configure settings"))
        self.act_reset_view.setText(self.tr("&Reset View"))
        self.act_reset_view.setStatusTip(self.tr("Reset the camera to the default view"))
        self.act_lang_en.setText(self.tr("English"))
        self.act_lang_zh.setText(self.tr("中文"))
        for style, text in [
            ("ball_stick", self.tr("Ball & Stick")),
            ("cpk", self.tr("Space Filling (CPK)")),
            ("wireframe", self.tr("Wireframe")),
        ]:
            self._style_actions[style].setText(text)
        self.act_show_cell.setText(self.tr("Show Unit &Cell"))
        self.act_show_cell.setStatusTip(self.tr("Show or hide the unit cell frame"))
        self.act_show_labels.setText(self.tr("Atom &Labels"))
        self.act_show_labels.setStatusTip(self.tr("Show element labels on atoms"))
        self.act_display_options.setText(self.tr("Display &Options..."))
        self.act_display_options.setStatusTip(
            self.tr("Background, lighting, colors and effects"))
        self.act_gen_incar.setText(self.tr("Generate &INCAR..."))
        self.act_gen_kpoints.setText(self.tr("Generate &KPOINTS..."))
        self.act_gen_potcar.setText(self.tr("Generate &POTCAR..."))
        self.act_gen_all.setText(self.tr("Generate &All Input Files..."))
        self.act_gen_all.setStatusTip(self.tr("Generate INCAR, KPOINTS, POSCAR, POTCAR at once"))
        self.act_surface.setText(self.tr("&Cleave Surface..."))
        self.act_surface.setStatusTip(self.tr("Cleave a surface/slab from the current structure"))
        self.act_supercell.setText(self.tr("&Supercell..."))
        self.act_supercell.setStatusTip(self.tr("Create a supercell"))
        self.act_wrap_periodic.setText(self.tr("Wrap in &Periodic Cell..."))
        self.act_wrap_periodic.setStatusTip(
            self.tr("Convert a molecule into a periodic structure (vacuum box)"))
        self.act_rebox.setText(self.tr("&Re-box Slab..."))
        self.act_rebox.setStatusTip(self.tr(
            "Re-apply the vacuum of a structure that already carries it "
            "(e.g. a cut surface)"))
        self.act_edit_lattice.setText(self.tr("Edit &Lattice..."))
        self.act_edit_lattice.setStatusTip(
            self.tr("Edit the unit cell parameters with live preview"))
        self.act_symmetry.setText(self.tr("Find &Symmetry..."))
        self.act_symmetry.setStatusTip(
            self.tr("Analyze the space group and symmetrize the structure"))
        self.act_about.setText(self.tr("&About VASPen"))
        self.act_about_qt.setText(self.tr("About &Qt"))

        # Menus
        self._menu_file.setTitle(self.tr("&File"))
        self._recent_menu.setTitle(self.tr("&Recent Files"))
        self._menu_edit.setTitle(self.tr("&Edit"))
        self._menu_view.setTitle(self.tr("&View"))
        self._menu_style.setTitle(self.tr("Display &Style"))
        self._menu_lang.setTitle(self.tr("&Language"))
        self._menu_calc.setTitle(self.tr("&Calculate"))
        self._menu_tools.setTitle(self.tr("&Tools"))
        self._menu_help.setTitle(self.tr("&Help"))

        # Toolbar / docks / status bar
        self._toolbar.setWindowTitle(self.tr("Main Toolbar"))
        self._edit_toolbar.setWindowTitle(self.tr("Edit Toolbar"))
        self._dock_structure.setWindowTitle(self.tr("Structure"))
        self._dock_measure.setWindowTitle(self.tr("Measurements"))
        self._dock_props.setWindowTitle(self.tr("Properties"))
        # toggleViewAction texts do NOT follow windowTitle changes —
        # re-sync them for the live language switch.
        for dock in (self._dock_structure, self._dock_measure, self._dock_props):
            dock.toggleViewAction().setText(dock.windowTitle())
        self._status_label.setText(self.tr("Ready"))
        for mode, (text, tip) in {
            ToolMode.SELECT: (self.tr("Select"),
                              self.tr("Select atoms (Ctrl/Shift for multi-select, Shift+drag box-select)")),
            ToolMode.ADD_ATOM: (self.tr("Add Atom"),
                                self.tr("Click empty space to add an atom")),
            ToolMode.MOVE_ATOM: (self.tr("Move"),
                                 self.tr("Drag an atom to move it (Esc cancels)")),
            ToolMode.ROTATE: (self.tr("Rotate"),
                              self.tr("Drag to rotate the selection or molecule (Esc cancels)")),
            ToolMode.DELETE: (self.tr("Delete"),
                              self.tr("Click an atom or bond to delete it")),
            ToolMode.CREATE_BOND: (self.tr("Create Bond"),
                                   self.tr("Click two atoms to create a bond")),
            ToolMode.MEASURE_DISTANCE: (self.tr("Distance"),
                                        self.tr("Click two atoms to measure their distance")),
            ToolMode.MEASURE_ANGLE: (self.tr("Angle"),
                                     self.tr("Click three atoms to measure the angle")),
            ToolMode.MEASURE_TORSION: (self.tr("Dihedral"),
                                       self.tr("Click four atoms to measure the dihedral")),
        }.items():
            act = self._mode_actions[mode]
            act.setText(text)
            act.setStatusTip(tip)
        self.act_delete_selection.setText(self.tr("Delete Selection"))
        self._element_label.setText(self.tr("Element:"))
        self._element_more_btn.setToolTip(self.tr(
            "Pick any element from the periodic table"))
        self._view_dir_btn.setText(self.tr("View"))
        self._view_dir_btn.setToolTip(self.tr("Align the camera with a world axis"))
        self._update_recent_ui()
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
        self._set_status(self.tr("中文") if lang == "zh" else self.tr("English"))

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            self.tr("About VASPen"),
            self.tr(
                "<h2>VASPen v%1</h2>"
                "<p>A cross-platform GUI for VASP first-principles calculations.</p>"
                "<p><b>Built with:</b> PySide6, ASE, pymatgen, Qt native OpenGL</p>"
                "<p>Free and open source (MIT License).</p>"
            ).arg(__version__),
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
        self._update_recent_ui()

    def _update_recent_ui(self) -> None:
        """Refresh both recent-files surfaces (File menu + welcome page)."""
        self._update_recent_menu()
        self._welcome_page.set_recent_files(self._config.recent_files)

    def _on_structure_loaded(self) -> None:
        # Any NEB frame-edit context refers to the PREVIOUS structure —
        # the viewport is rebound below and edit signals must never
        # reach a frame of the replaced model (backstop: _open_file
        # already closes the preview dialogs).
        self._frame_edit = None
        self._frame_selection = set()
        self._previewing_neb = False
        # Measurements belong to the PREVIOUS structure — their atom IDs
        # would silently resolve to unrelated atoms of the new one.
        self._measurement_manager.clear()
        # reset_view=False only during an external-file reload whose
        # geometry is unchanged (smart camera, §7.12) — normal opens fit.
        self._viewport.set_structure(self._structure.atoms,
                                     reset_view=not self._reload_keep_camera,
                                     bonds=self._structure.bonds,
                                     fixed=self._structure.fixed_flags)
        self._sync_auto_bonds_action()
        self._update_welcome_visibility()
        self._update_status_bar()
        self._update_edit_actions()
        self._sync_file_watcher()
        # Loading clears the selection without emitting selection_changed —
        # re-sync selection-dependent action states here.
        self.act_freeze.setEnabled(False)
        self.act_unfreeze.setEnabled(False)

    def _on_structure_modified(self) -> None:
        """Structure changed (add/remove atoms, supercell, surface cleave...)."""
        # Tools must not keep stale indices/preview state across edits
        self._viewport.cancel_active_tool()
        # Keep the user's camera — in-place edits must not snap the view back
        self._viewport.set_structure(
            self._structure.atoms, reset_view=False, bonds=self._structure.bonds,
            fixed=self._structure.fixed_flags)
        # set_structure resets the viewport highlight — re-apply the model
        # selection so it survives edits
        self._viewport.set_highlight(self._structure.selected_indices)
        self._viewport.set_bond_highlight(self._structure.selected_bonds)
        self._sync_auto_bonds_action()
        self._measurement_manager.sync_with_model()  # → changed → push
        self._update_welcome_visibility()
        self._update_status_bar()
        self._update_edit_actions()

    def _on_selection_cleared(self) -> None:
        """Selection cleared via the model (e.g. the selected atom was deleted)."""
        if self._frame_edit is not None:
            return  # frame selection is viewport-local
        self._viewport.highlight_atom(None)

    def _on_selection_changed(self) -> None:
        """Selection set changed (multi-select aware)."""
        if self._frame_edit is not None:
            return  # model selection must not overwrite the frame highlight
        self._viewport.set_highlight(self._structure.selected_indices)
        self._update_selection_status()
        has_sel = bool(self._structure.selected_indices)
        self.act_freeze.setEnabled(has_sel)
        self.act_unfreeze.setEnabled(has_sel)

    def _update_selection_status(self) -> None:
        """Show the selection in the status bar (single atom or count)."""
        sel = self._structure.selected_indices
        if not sel:
            return
        if len(sel) == 1:
            index = next(iter(sel))
            sym = self._structure.symbols[index]
            pos = self._structure.positions[index]
            self._set_status(self.tr("Atom {}: {} at ({:.3f}, {:.3f}, {:.3f}) Å").format(
                index, sym, pos[0], pos[1], pos[2]
            ))
        else:
            self._set_status(self.tr("{} atoms selected").format(len(sel)))

    def _update_edit_actions(self) -> None:
        """Enable Undo/Redo based on the model's edit history."""
        self.act_undo.setEnabled(self._structure.can_undo)
        self.act_redo.setEnabled(self._structure.can_redo)

    def _on_atom_clicked(self, index: int) -> None:
        """User clicked an atom in the 3D viewport (legacy single-select)."""
        self._structure.select_atom(index)

    def _on_background_clicked(self) -> None:
        """User clicked empty space in the 3D viewport."""
        if self._frame_edit is not None:
            self._frame_selection = set()
            self._viewport.set_highlight(set())
            return
        if self._previewing_neb or self._surface_dialog is not None:
            return  # preview clicks stay viewport-local (model untouched)
        self._structure.clear_selection()
        self._structure.clear_bond_selection()

    def _on_atom_selected(self, index: int) -> None:
        """Atom selection changed via the model (e.g. from structure tree)."""
        self._viewport.highlight_atom(index)
        if 0 <= index < self._structure.n_atoms:
            sym = self._structure.symbols[index]
            pos = self._structure.positions[index]
            self._set_status(self.tr("Atom {}: {} at ({:.3f}, {:.3f}, {:.3f}) Å").format(
                index, sym, pos[0], pos[1], pos[2]
            ))

    # ------------------------------------------------------------------
    # 3D edit-tool handlers (Interaction → Command → Model)
    # ------------------------------------------------------------------

    def _on_atoms_selected(self, indices: list, mode: str) -> None:
        """Selection from a viewport tool (click / box / modifiers).

        While a middle NEB frame is being edited, the selection stays
        viewport-local — the model's selection is untouched. During a
        non-frame preview (surface slab, endpoint images) the viewport
        shows a TEMPORARY structure — frame indices must never become
        model indices, so the selection stays viewport-local there too.
        """
        if self._frame_edit is not None:
            if mode == "add":
                self._frame_selection |= set(indices)
            elif mode == "toggle":
                for i in indices:
                    self._frame_selection.symmetric_difference_update({i})
            else:
                self._frame_selection = set(indices)
            self._viewport.set_highlight(self._frame_selection)
            return
        if self._previewing_neb or self._surface_dialog is not None:
            self._viewport.set_highlight(set(indices))
            return
        if mode == "add":
            self._structure.add_to_selection(indices)
        elif mode == "toggle":
            self._structure.toggle_selection(indices)
        else:
            self._structure.set_selection(indices)

    def _on_atom_place_requested(self, position, anchor) -> None:
        """Add-atom tool: place the chosen element at a world position.

        When the atom was grown from an anchor atom, the new bond is
        created too (in auto mode the recompute already finds it; in
        manual mode the bond is added explicitly).
        """
        from vaspen.core.builder import StructureBuilder

        model = self._structure
        pos = np.asarray(position, dtype=float)
        if model.is_periodic:
            # Wrap into the home cell (fractional %1 on periodic axes)
            cell = np.asarray(model.cell, dtype=float)
            frac = np.linalg.solve(cell.T, pos)
            for ax in range(3):
                if model.pbc[ax]:
                    frac[ax] %= 1.0
            pos = frac @ cell
        symbol = self._current_element
        StructureBuilder.add_atom(model, symbol, pos)
        new_index = model.n_atoms - 1
        if anchor is not None and model.bond_mode == "manual":
            model.add_bond(anchor, new_index)
        self._set_status(self.tr("Atom {} added at ({:.3f}, {:.3f}, {:.3f}) Å").format(
            symbol, pos[0], pos[1], pos[2]
        ))

    def _on_undo(self) -> None:
        """Undo: per-frame history while editing a NEB frame, else the model."""
        if self._frame_edit is not None:
            if self._frame_edit["panel"].frame_undo(self._frame_edit["index"]):
                self._refresh_frame_edit_view()
            else:
                self._set_status(self.tr("Nothing to undo for this image."))
            return
        self._structure.undo()

    def _on_redo(self) -> None:
        """Redo: per-frame history while editing a NEB frame, else the model."""
        if self._frame_edit is not None:
            if self._frame_edit["panel"].frame_redo(self._frame_edit["index"]):
                self._refresh_frame_edit_view()
            else:
                self._set_status(self.tr("Nothing to redo for this image."))
            return
        self._structure.redo()

    def _on_atoms_moved(self, indices: list, positions) -> None:
        """Move tool committed a drag — one undoable model call.

        While a middle NEB frame is being edited, the commit lands in
        the frame instead of the model. Frozen atoms are rejected here
        with a status message and the viewport is re-synced: the
        rejected call does not emit structure_modified, so the drag
        preview would otherwise stay on screen.
        """
        if self._frame_edit is not None:
            panel = self._frame_edit["panel"]
            panel.apply_frame_move(
                self._frame_edit["index"], list(indices),
                np.asarray(positions, dtype=float))
            self._refresh_frame_edit_view()
            return
        if any(self._structure.is_fixed(i) for i in indices):
            self._set_status(
                self.tr("Cannot move frozen atoms — unfreeze them first."))
            self._rebind_viewport_to_model()
            return
        try:
            self._structure.set_atom_positions(
                indices, np.asarray(positions, dtype=float))
        except ValueError as e:
            self._set_status(str(e))
            self._rebind_viewport_to_model()

    def _rebind_viewport_to_model(self) -> None:
        """Reset the viewport scene to the model state (drag-preview rebound)."""
        self._viewport.set_structure(
            self._structure.atoms, reset_view=False,
            bonds=self._structure.bonds, fixed=self._structure.fixed_flags)
        self._viewport.set_highlight(self._structure.selected_indices)
        self._viewport.set_bond_highlight(self._structure.selected_bonds)

    def _on_bond_created(self, i: int, j: int) -> None:
        if self._frame_edit is not None:
            panel = self._frame_edit["panel"]
            panel.add_frame_bond(self._frame_edit["index"], i, j)
            self._refresh_frame_edit_view()
            self._set_status(self.tr("Bond created: {}–{}").format(i, j))
            return
        try:
            self._structure.add_bond(i, j)
            self._set_status(self.tr("Bond created: {}–{}").format(i, j))
        except ValueError as e:
            self._set_status(str(e))

    def _on_delete_requested(self, kind: str, index: int) -> None:
        """Delete tool clicked an atom/bond, or deletes the selection.

        While a middle NEB frame is being edited, only bond deletion
        is allowed — atoms cannot be deleted from a NEB image.
        """
        if self._frame_edit is not None:
            if kind == "bond":
                panel = self._frame_edit["panel"]
                if panel.remove_frame_bond(self._frame_edit["index"], index):
                    self._refresh_frame_edit_view()
            else:
                self._set_status(
                    self.tr("Atoms cannot be deleted in NEB images."))
            return
        if kind == "selection":
            self._on_delete_selection()
        elif kind == "atom":
            self._structure.delete_atom(index)
        elif kind == "bond":
            try:
                b = self._structure.bonds[index]
                self._structure.remove_bond(b.i, b.j)
            except (IndexError, ValueError):
                pass

    def _on_bond_clicked(self, index: int) -> None:
        """Select tool clicked a bond → select it (bond list index)."""
        if self._frame_edit is not None:
            return  # bond selection is a model feature — frames skip it
        if self._previewing_neb or self._surface_dialog is not None:
            return  # preview bond indices are temporary — never model indices
        self._structure.select_bond(index)

    def _on_bond_selection_changed(self) -> None:
        """Bond selection changed via the model."""
        self._viewport.set_bond_highlight(self._structure.selected_bonds)
        sel = self._structure.selected_bonds
        for act in self._bond_order_actions.values():
            act.setEnabled(len(sel) == 1)
        if len(sel) == 1:
            k = next(iter(sel))
            try:
                b = self._structure.bonds[k]
                self._set_status(self.tr("Bond {}–{} (order {})").format(
                    b.i, b.j, b.order))
            except IndexError:
                pass

    def _on_set_bond_order(self, order: int) -> None:
        """Bond Order menu: change the selected bond's order (undoable)."""
        sel = self._structure.selected_bonds
        if len(sel) != 1:
            return
        k = next(iter(sel))
        try:
            b = self._structure.bonds[k]
        except IndexError:
            return
        self._structure.set_bond_order(b.i, b.j, order)
        # bond edits clear the selection — re-select so the user can
        # chain order changes (Double → Aromatic …)
        self._structure.select_bond(k)

    def _on_mode_changed(self, mode: ToolMode) -> None:
        """Keep the toolbar checked state in sync with the viewport mode."""
        for m, act in self._mode_actions.items():
            if act.isChecked() != (m == mode):
                act.setChecked(m == mode)

    def _on_delete_selection(self) -> None:
        """Delete key: remove all selected atoms in one undo step."""
        indices = sorted(self._structure.selected_indices, reverse=True)
        if indices:
            self._structure.delete_atoms(indices)

    def _on_freeze_selection(self) -> None:
        """Freeze the selected atoms (VASP selective dynamics).

        Freezes all three directions (the settled default); partially
        frozen atoms are overwritten to fully frozen. One undo step.
        """
        sel = sorted(self._structure.selected_indices)
        if not sel:
            return
        self._structure.set_fixed(sel, True)
        self._set_status(self.tr("Frozen {} atoms.").format(len(sel)))

    def _on_unfreeze_selection(self) -> None:
        """Unfreeze the selected atoms — clears ALL fixed directions.

        Partial per-direction flags are cleared too (the atom is fully
        free afterwards). One undo step.
        """
        sel = sorted(self._structure.selected_indices)
        if not sel:
            return
        self._structure.set_fixed(sel, False)
        self._set_status(self.tr("Unfroze {} atoms.").format(len(sel)))

    def _on_measurement_added(self, kind: str, indices: list) -> None:
        """Measure tool completed a pick sequence."""
        self._measurement_manager.add(kind, indices)  # sync → changed → push

    def _push_measurements(self) -> None:
        """Push the manager's payload to the viewport (no re-sync here —
        sync_with_model emits changed, so this must stay side-effect-free
        to avoid a signal loop)."""
        self._viewport.set_measurements(self._measurement_manager.payload())

    def _on_auto_bonds_toggled(self, checked: bool) -> None:
        """Auto Detect Bonds action: re-derive connectivity (undoable)."""
        mode = "auto" if checked else "manual"
        if mode != self._structure.bond_mode:
            self._structure.set_bond_mode(mode)
        self._sync_auto_bonds_action()

    def _sync_auto_bonds_action(self) -> None:
        """Mirror the model's bond mode in the checkable action."""
        self.act_auto_bonds.blockSignals(True)
        self.act_auto_bonds.setChecked(self._structure.bond_mode == "auto")
        self.act_auto_bonds.blockSignals(False)

    def _on_detect_bonds(self) -> None:
        """Detect Bonds button: one-shot connectivity refresh."""
        if self._structure.n_atoms == 0:
            return
        self._structure.detect_bonds()
        self._set_status(self.tr("Detected {} bonds").format(
            len(self._structure.bonds)))

    def _update_status_bar(self) -> None:
        n = self._structure.n_atoms
        self._atom_count_label.setText(
            self.tr("Atoms: {}  |  {}").format(n, self._structure.chemical_formula)
        )
        if self._structure.n_atoms > 0:
            if self._structure.is_periodic:
                a, b, c = self._structure.cell_lengths
                alpha, beta, gamma = self._structure.cell_angles
                self._cell_label.setText(
                    self.tr("a={:.2f} b={:.2f} c={:.2f}  α={:.1f}° β={:.1f}° γ={:.1f}°").format(
                        a, b, c, alpha, beta, gamma
                    )
                )
            else:
                self._cell_label.setText(self.tr("a=— b=— c=—"))
        else:
            # Empty model (welcome page / fresh New): drop the previous
            # structure's cell parameters instead of leaving them stale.
            self._cell_label.setText("")

    def _set_status(self, message: str) -> None:
        self._status_label.setText(message)
        logger.debug(message)
