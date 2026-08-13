"""Display Options dialog — background, lighting, colors, overlays, effects.

Modal with LIVE preview (same pattern as LatticeDialog): every control
change rebuilds a RenderSettings and applies it to the viewport
immediately; Accept persists it via AppConfig, Cancel restores the
original settings. Created fresh per invocation so it picks up the
current UI language automatically.

Choice controls (style, color scheme, add element) are QToolButton +
QMenu popups, NOT QComboBox: QComboBox popups inside this dialog's
QScrollArea reopened repeatedly on click (Qt popup re-entrancy), the
menu pattern is the one the main window already uses for its
view-direction button and has no such issue.
"""

from __future__ import annotations

from ase.data import chemical_symbols
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from vaspen.core.render_settings import (
    BACKGROUND_DARK,
    BACKGROUND_LIGHT,
    GRADIENT_DARK_BOTTOM,
    GRADIENT_DARK_TOP,
    GRADIENT_LIGHT_BOTTOM,
    GRADIENT_LIGHT_TOP,
    PALETTES,
    RenderSettings,
)
from vaspen.ui.viewport3d import element_color
from vaspen.utils.config import AppConfig

_STYLE_ITEMS = [
    ("ball_stick", "Ball & Stick"),
    ("cpk", "Space Filling (CPK)"),
    ("wireframe", "Wireframe"),
]

_SCHEME_ITEMS = [
    ("jmol", "Jmol (default)"),
    ("metal_nonmetal", "Metal / Non-metal"),
    ("block", "Periodic table blocks (s/p/d/f)"),
]


class DisplayOptionsDialog(QDialog):
    """Edit every adjustable render parameter with live 3D preview."""

    def __init__(self, viewport, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewport = viewport
        self._original = RenderSettings.from_dict(
            viewport.render_settings().to_dict())
        self._settings = RenderSettings.from_dict(self._original.to_dict())
        # per-element overrides: symbol → [r, g, b, a] (source of truth
        # for the color rows; mirrored into self._settings on apply)
        self._overrides: dict[str, list[float]] = {
            sym: list(rgba)
            for sym, rgba in self._settings.atom_colors.items()
        }
        self._bg_color = self._settings.background_color
        self._grad_top = self._settings.gradient_top
        self._grad_bottom = self._settings.gradient_bottom
        self._cell_color = self._settings.cell_color
        self._row_grid: QGridLayout | None = None

        self.setWindowTitle(self.tr("Display Options"))
        self.setModal(True)
        self.resize(540, 680)
        self._build_ui()
        self._load_settings_into_controls(self._settings)
        self._apply()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.addWidget(self._build_background_group())
        vbox.addWidget(self._build_lighting_group())
        vbox.addWidget(self._build_atoms_group())
        vbox.addWidget(self._build_colors_group())
        vbox.addWidget(self._build_overlay_group())
        vbox.addWidget(self._build_effects_group())
        vbox.addStretch(1)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

        self._reset_btn = QPushButton(self.tr("Reset to Defaults"))
        self._reset_btn.clicked.connect(self._on_reset_defaults)
        self._reset_btn.setToolTip(
            self.tr("Restore every setting to its default value"))
        layout.addWidget(self._reset_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_background_group(self) -> QGroupBox:
        group = QGroupBox(self.tr("Background"))
        vbox = QVBoxLayout(group)

        row = QHBoxLayout()
        self._bg_dark_radio = QRadioButton(self.tr("Dark"))
        self._bg_light_radio = QRadioButton(self.tr("Light"))
        self._bg_custom_radio = QRadioButton(self.tr("Custom"))
        self._bg_group = QButtonGroup(self)
        for radio in (self._bg_dark_radio, self._bg_light_radio,
                      self._bg_custom_radio):
            self._bg_group.addButton(radio)
            radio.toggled.connect(self._on_bg_preset_changed)
        row.addWidget(self._bg_dark_radio)
        row.addWidget(self._bg_light_radio)
        row.addWidget(self._bg_custom_radio)
        row.addStretch(1)
        self._bg_swatch = self._make_swatch(self._on_bg_color_picked)
        row.addWidget(self._bg_swatch)
        vbox.addLayout(row)

        row2 = QHBoxLayout()
        self._grad_check = QCheckBox(self.tr("Gradient"))
        self._grad_check.toggled.connect(self._on_changed)
        row2.addWidget(self._grad_check)
        row2.addWidget(QLabel(self.tr("Top:")))
        self._grad_top_swatch = self._make_swatch(self._on_grad_top_picked)
        row2.addWidget(self._grad_top_swatch)
        row2.addWidget(QLabel(self.tr("Bottom:")))
        self._grad_bottom_swatch = self._make_swatch(self._on_grad_bottom_picked)
        row2.addWidget(self._grad_bottom_swatch)
        row2.addStretch(1)
        vbox.addLayout(row2)
        return group

    def _build_lighting_group(self) -> QGroupBox:
        group = QGroupBox(self.tr("Lighting"))
        vbox = QVBoxLayout(group)
        self._az_slider = self._add_slider(
            vbox, self.tr("Azimuth"), -180.0, 180.0, 1.0, 25.0, "{:.0f}°")
        self._el_slider = self._add_slider(
            vbox, self.tr("Elevation"), -90.0, 90.0, 1.0, 33.6, "{:.0f}°")
        self._amb_slider = self._add_slider(
            vbox, self.tr("Ambient"), 0.0, 2.0, 0.01, 0.75)
        self._diff_slider = self._add_slider(
            vbox, self.tr("Diffuse"), 0.0, 2.0, 0.01, 0.80)
        self._spec_slider = self._add_slider(
            vbox, self.tr("Specular"), 0.0, 1.0, 0.01, 0.30)
        self._shin_slider = self._add_slider(
            vbox, self.tr("Shininess"), 1.0, 128.0, 1.0, 32.0, "{:.0f}")
        self._fill_slider = self._add_slider(
            vbox, self.tr("Fill light"), 0.0, 1.0, 0.01, 0.15)
        row = QHBoxLayout()
        self._headlight_check = QCheckBox(
            self.tr("Headlight (light follows camera)"))
        self._headlight_check.toggled.connect(self._on_changed)
        self._spec_check = QCheckBox(self.tr("Specular highlights"))
        self._spec_check.setToolTip(
            self.tr("Glossy highlights on all atoms (on/off)"))
        self._spec_check.toggled.connect(self._on_changed)
        row.addWidget(self._headlight_check)
        row.addWidget(self._spec_check)
        row.addStretch(1)
        vbox.addLayout(row)
        return group

    def _build_atoms_group(self) -> QGroupBox:
        group = QGroupBox(self.tr("Atoms & Bonds"))
        vbox = QVBoxLayout(group)
        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr("Style:")))
        self._style_btn = self._make_choice_button(_STYLE_ITEMS,
                                                   self._on_style_picked)
        row.addWidget(self._style_btn)
        row.addStretch(1)
        vbox.addLayout(row)
        self._sphere_slider = self._add_slider(
            vbox, self.tr("Sphere size"), 0.1, 1.0, 0.01, 0.60)
        self._bond_slider = self._add_slider(
            vbox, self.tr("Bond radius"), 0.02, 0.30, 0.01, 0.12, "{:.2f} Å")
        self._opacity_slider = self._add_slider(
            vbox, self.tr("Atom opacity"), 0.0, 1.0, 0.01, 1.0)
        self._opacity_slider.setToolTip(
            self.tr("Overall atom transparency; bonds fade together with "
                    "their atoms (0 = fully transparent)"))
        row2 = QHBoxLayout()
        self._bonds_elem_check = QCheckBox(self.tr("Bonds colored by element"))
        self._bonds_elem_check.toggled.connect(self._on_changed)
        row2.addWidget(self._bonds_elem_check)
        row2.addStretch(1)
        vbox.addLayout(row2)
        row3 = QHBoxLayout()
        self._show_cell_check = QCheckBox(self.tr("Show unit cell"))
        self._show_cell_check.toggled.connect(self._on_changed)
        row3.addWidget(self._show_cell_check)
        self._cell_swatch = self._make_swatch(self._on_cell_color_picked)
        row3.addWidget(self._cell_swatch)
        row3.addWidget(QLabel(self.tr("Cell line width:")))
        self._cell_width_slider = QSlider(Qt.Horizontal)
        self._cell_width_slider.setRange(1, 5)
        self._cell_width_slider.setToolTip(self.tr("Cell line width"))
        self._cell_width_slider.valueChanged.connect(self._on_changed)
        row3.addWidget(self._cell_width_slider)
        row3.addStretch(1)
        vbox.addLayout(row3)
        row4 = QHBoxLayout()
        self._show_labels_check = QCheckBox(self.tr("Show labels"))
        self._show_labels_check.toggled.connect(self._on_changed)
        row4.addWidget(self._show_labels_check)
        row4.addWidget(QLabel(self.tr("Label size:")))
        self._label_size_spin = QSpinBox()
        self._label_size_spin.setRange(8, 28)
        self._label_size_spin.setSuffix(" pt")
        self._label_size_spin.setToolTip(
            self.tr("Base label size; labels scale with the zoom"))
        self._label_size_spin.valueChanged.connect(self._on_changed)
        row4.addWidget(self._label_size_spin)
        row4.addStretch(1)
        vbox.addLayout(row4)
        return group

    def _build_colors_group(self) -> QGroupBox:
        group = QGroupBox(self.tr("Colors"))
        vbox = QVBoxLayout(group)
        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr("Scheme:")))
        self._scheme_btn = self._make_choice_button(_SCHEME_ITEMS,
                                                    self._on_scheme_picked)
        row.addWidget(self._scheme_btn)
        row.addStretch(1)
        vbox.addLayout(row)

        self._row_grid = QGridLayout()
        vbox.addLayout(self._row_grid)

        row2 = QHBoxLayout()
        self._add_element_btn = QToolButton()
        self._add_element_btn.setText(self.tr("Add element…"))
        self._add_element_btn.setPopupMode(QToolButton.InstantPopup)
        self._add_element_menu = QMenu(self)
        for sym in chemical_symbols[1:]:
            action = QAction(sym, self)
            action.triggered.connect(
                lambda checked=False, s=sym: self._on_add_element(s))
            self._add_element_menu.addAction(action)
        self._add_element_btn.setMenu(self._add_element_menu)
        row2.addWidget(self._add_element_btn)
        self._reset_all_btn = QPushButton(self.tr("Reset all overrides"))
        self._reset_all_btn.clicked.connect(self._on_reset_all_overrides)
        row2.addWidget(self._reset_all_btn)
        row2.addStretch(1)
        vbox.addLayout(row2)
        return group

    def _build_overlay_group(self) -> QGroupBox:
        group = QGroupBox(self.tr("Axes & Labels"))
        vbox = QVBoxLayout(group)
        row = QHBoxLayout()
        self._axes_check = QCheckBox(self.tr("Orientation axes indicator"))
        self._axes_check.toggled.connect(self._on_changed)
        self._cell_corners_check = QCheckBox(
            self.tr("Cell corner labels (O/A/B/C)"))
        self._cell_corners_check.toggled.connect(self._on_changed)
        row.addWidget(self._axes_check)
        row.addWidget(self._cell_corners_check)
        row.addStretch(1)
        vbox.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel(self.tr("Corner label size:")))
        self._corner_size_spin = QSpinBox()
        self._corner_size_spin.setRange(8, 28)
        self._corner_size_spin.setSuffix(" pt")
        self._corner_size_spin.setToolTip(
            self.tr("Size of the O/A/B/C cell corner labels"))
        self._corner_size_spin.valueChanged.connect(self._on_changed)
        row2.addWidget(self._corner_size_spin)
        row2.addStretch(1)
        vbox.addLayout(row2)
        return group

    def _build_effects_group(self) -> QGroupBox:
        group = QGroupBox(self.tr("Effects"))
        vbox = QVBoxLayout(group)
        self._gamma_slider = self._add_slider(
            vbox, self.tr("Gamma"), 1.0, 3.0, 0.05, 2.2)
        return group

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_swatch(self, handler) -> QPushButton:
        """Small color button (a square painted with the current color)."""
        btn = QPushButton()
        btn.setFixedSize(34, 24)
        btn.setToolTip(self.tr("Pick a color"))
        btn.clicked.connect(handler)
        return btn

    def _set_swatch_color(self, btn, rgb: tuple) -> None:
        qcolor = QColor.fromRgbF(*[max(0.0, min(1.0, c)) for c in rgb])
        btn.setStyleSheet(
            f"background-color: {qcolor.name()};"
            "border: 1px solid #888; border-radius: 2px;"
        )

    def _make_choice_button(
        self,
        items: list[tuple[str, str]],
        handler,
    ) -> QToolButton:
        """QToolButton + QMenu for a small choice list (avoids the
        QComboBox popup re-entrancy bug inside the scroll area)."""
        btn = QToolButton()
        btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(btn)
        group = QActionGroup(btn)
        group.setExclusive(True)
        for key, text in items:
            action = QAction(self.tr(text), btn)
            action.setCheckable(True)
            action.setData(key)
            action.triggered.connect(
                lambda checked=False, a=action, h=handler: h(a))
            group.addAction(action)
            menu.addAction(action)
        btn.setMenu(menu)
        btn._actions = {key: act for key, act in
                        [(a.data(), a) for a in group.actions()]}
        return btn

    def _add_slider(
        self,
        layout: QVBoxLayout,
        text: str,
        low: float,
        high: float,
        step: float,
        value: float,
        fmt: str = "{:.2f}",
    ) -> QSlider:
        """Slider row: [label] [slider] [live value label]."""
        slider = QSlider(Qt.Horizontal)
        slider.setRange(round(low / step), round(high / step))
        slider.setValue(round(value / step))
        value_label = QLabel(fmt.format(value))
        value_label.setMinimumWidth(52)
        slider.valueChanged.connect(
            lambda v, lab=value_label, st=step, f=fmt: lab.setText(f.format(v * st)))
        slider.valueChanged.connect(self._on_changed)
        row = QHBoxLayout()
        row.addWidget(QLabel(text))
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        layout.addLayout(row)
        return slider

    # ------------------------------------------------------------------
    # State: controls ↔ settings
    # ------------------------------------------------------------------

    def _load_settings_into_controls(self, rs: RenderSettings) -> None:
        """Fill every control from a settings object (used by reset)."""
        self._bg_color = rs.background_color
        if rs.background_color == BACKGROUND_DARK:
            self._bg_dark_radio.setChecked(True)
        elif rs.background_color == BACKGROUND_LIGHT:
            self._bg_light_radio.setChecked(True)
        else:
            self._bg_custom_radio.setChecked(True)
        self._set_swatch_color(self._bg_swatch, self._bg_color)

        self._grad_top = rs.gradient_top
        self._grad_bottom = rs.gradient_bottom
        self._grad_check.setChecked(rs.background_gradient)
        self._set_swatch_color(self._grad_top_swatch, self._grad_top)
        self._set_swatch_color(self._grad_bottom_swatch, self._grad_bottom)

        self._az_slider.setValue(round(rs.light_azimuth))
        self._el_slider.setValue(round(rs.light_elevation))
        self._amb_slider.setValue(round(rs.ambient / 0.01))
        self._diff_slider.setValue(round(rs.diffuse / 0.01))
        self._spec_slider.setValue(round(rs.specular / 0.01))
        self._shin_slider.setValue(round(rs.shininess))
        self._fill_slider.setValue(round(rs.fill_intensity / 0.01))
        self._headlight_check.setChecked(rs.headlight)
        self._spec_check.setChecked(rs.specular_enabled)

        self._set_choice(self._style_btn, rs.style)
        self._sphere_slider.setValue(round(rs.sphere_scale / 0.01))
        self._bond_slider.setValue(round(rs.bond_radius / 0.01))
        self._opacity_slider.setValue(round(rs.atom_opacity / 0.01))
        self._bonds_elem_check.setChecked(rs.bonds_by_element)
        self._show_cell_check.setChecked(rs.show_cell)
        self._cell_color = rs.cell_color
        self._set_swatch_color(self._cell_swatch, self._cell_color)
        self._cell_width_slider.setValue(round(rs.cell_line_width))
        self._show_labels_check.setChecked(rs.show_labels)
        self._label_size_spin.setValue(rs.label_size)

        self._set_choice(self._scheme_btn, rs.color_scheme)
        self._overrides = {
            sym: list(rgba) for sym, rgba in rs.atom_colors.items()
        }
        self._rebuild_element_rows()

        self._axes_check.setChecked(rs.show_axes)
        self._cell_corners_check.setChecked(rs.show_cell_corners)
        self._corner_size_spin.setValue(rs.corner_label_size)

        self._gamma_slider.setValue(round(rs.gamma / 0.05))

    def _set_choice(self, btn: QToolButton, key: str) -> None:
        """Set the button text + checked action for a choice button."""
        action = btn._actions.get(key)
        btn.setText(action.text() if action is not None else "")
        if action is not None:
            action.setChecked(True)

    def _choice_key(self, btn: QToolButton) -> str | None:
        for key, act in btn._actions.items():
            if act.isChecked():
                return key
        return None

    def _current_settings(self) -> RenderSettings:
        rs = RenderSettings()
        rs.background_color = self._bg_color
        rs.background_gradient = self._grad_check.isChecked()
        rs.gradient_top = self._grad_top
        rs.gradient_bottom = self._grad_bottom
        rs.light_azimuth = float(self._az_slider.value())
        rs.light_elevation = float(self._el_slider.value())
        rs.ambient = self._amb_slider.value() * 0.01
        rs.diffuse = self._diff_slider.value() * 0.01
        rs.specular = self._spec_slider.value() * 0.01
        rs.shininess = float(self._shin_slider.value())
        rs.fill_intensity = self._fill_slider.value() * 0.01
        rs.headlight = self._headlight_check.isChecked()
        rs.specular_enabled = self._spec_check.isChecked()
        style = self._choice_key(self._style_btn)
        rs.style = style if style in [k for k, _t in _STYLE_ITEMS] else "ball_stick"
        rs.sphere_scale = self._sphere_slider.value() * 0.01
        rs.bond_radius = self._bond_slider.value() * 0.01
        rs.atom_opacity = self._opacity_slider.value() * 0.01
        rs.bonds_by_element = self._bonds_elem_check.isChecked()
        rs.show_cell = self._show_cell_check.isChecked()
        rs.cell_color = self._cell_color
        rs.cell_line_width = float(self._cell_width_slider.value())
        rs.show_labels = self._show_labels_check.isChecked()
        rs.label_size = self._label_size_spin.value()
        scheme = self._choice_key(self._scheme_btn)
        rs.color_scheme = (
            scheme if scheme in PALETTES else "jmol")
        rs.atom_colors = {sym: list(rgba) for sym, rgba in self._overrides.items()}
        rs.show_axes = self._axes_check.isChecked()
        rs.show_cell_corners = self._cell_corners_check.isChecked()
        rs.corner_label_size = self._corner_size_spin.value()
        rs.gamma = self._gamma_slider.value() * 0.05
        return rs

    def _apply(self) -> None:
        """Push the current control state to the viewport (live preview)."""
        self._viewport.set_render_settings(self._current_settings())

    # ------------------------------------------------------------------
    # Element color rows
    # ------------------------------------------------------------------

    def _scheme_of(self, symbol: str) -> tuple[float, float, float]:
        """The color a symbol gets without its override (palette → Jmol)."""
        scheme = self._choice_key(self._scheme_btn) or "jmol"
        return PALETTES.get(scheme, {}).get(symbol, element_color(symbol))

    def _rebuild_element_rows(self) -> None:
        """Rebuild the per-element override rows from self._overrides."""
        grid = self._row_grid
        while grid.count():
            item = grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        # column headers (row 0) — the opacity slider needs a visible
        # caption, otherwise the per-element transparency control is
        # undiscoverable
        grid.addWidget(QLabel(self.tr("Element")), 0, 0)
        grid.addWidget(QLabel(self.tr("Color")), 0, 1)
        grid.addWidget(QLabel(self.tr("Opacity")), 0, 2)
        # every element of the current structure gets a row (plus any
        # override for elements NOT in the structure)
        symbols = sorted(
            set(self._viewport.structure_symbols()) | set(self._overrides))
        for row, sym in enumerate(symbols):
            self._insert_element_row(row + 1, sym)

    def _insert_element_row(self, row: int, sym: str) -> None:
        grid = self._row_grid
        label = QLabel(sym)
        label.setMinimumWidth(34)
        swatch = self._make_swatch(
            lambda checked=False, s=sym: self._on_element_color_picked(s))
        opacity = QSlider(Qt.Horizontal)
        opacity.setRange(0, 100)
        opacity.setToolTip(self.tr("Opacity"))
        opacity.valueChanged.connect(
            lambda v, s=sym: self._on_element_opacity(s, v))
        reset = QPushButton(self.tr("Reset"))
        reset.clicked.connect(lambda checked=False, s=sym: self._on_element_reset(s))
        grid.addWidget(label, row, 0)
        grid.addWidget(swatch, row, 1)
        grid.addWidget(opacity, row, 2)
        grid.addWidget(reset, row, 3)
        rgba = self._overrides.get(sym)
        if rgba is not None:
            self._set_swatch_color(swatch, rgba[:3])
        else:
            self._set_swatch_color(swatch, self._scheme_of(sym))
        # blockSignals: initializing the slider must NOT fire
        # valueChanged (that would auto-create an override for every
        # row element)
        opacity.blockSignals(True)
        opacity.setValue(round((rgba if rgba is not None else [0, 0, 0, 1.0])[3] * 100))
        opacity.blockSignals(False)

    def _row_swatch_for(self, sym: str):
        """The swatch button of an element row (or None)."""
        grid = self._row_grid
        for row in range(grid.rowCount()):
            label_item = grid.itemAtPosition(row, 0)
            if label_item is not None and label_item.widget().text() == sym:
                return grid.itemAtPosition(row, 1).widget()
        return None

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_changed(self, *_args) -> None:
        self._apply()

    def _on_style_picked(self, action) -> None:
        self._style_btn.setText(action.text())
        self._apply()

    def _on_scheme_picked(self, action) -> None:
        self._scheme_btn.setText(action.text())
        # rows without an override re-read the new scheme palette
        for sym in sorted(
                set(self._viewport.structure_symbols()) | set(self._overrides)):
            swatch = self._row_swatch_for(sym)
            if swatch is not None and sym not in self._overrides:
                self._set_swatch_color(swatch, self._scheme_of(sym))
        self._apply()

    def _on_bg_preset_changed(self, *_args) -> None:
        if self._bg_dark_radio.isChecked():
            if (tuple(self._bg_color) == BACKGROUND_LIGHT
                    and tuple(self._grad_top) == GRADIENT_LIGHT_TOP
                    and tuple(self._grad_bottom) == GRADIENT_LIGHT_BOTTOM):
                # untouched light preset → switch the gradient presets too
                self._grad_top, self._grad_bottom = (
                    GRADIENT_DARK_TOP, GRADIENT_DARK_BOTTOM)
            self._bg_color = BACKGROUND_DARK
        elif self._bg_light_radio.isChecked():
            if (tuple(self._bg_color) == BACKGROUND_DARK
                    and tuple(self._grad_top) == GRADIENT_DARK_TOP
                    and tuple(self._grad_bottom) == GRADIENT_DARK_BOTTOM):
                self._grad_top, self._grad_bottom = (
                    GRADIENT_LIGHT_TOP, GRADIENT_LIGHT_BOTTOM)
            self._bg_color = BACKGROUND_LIGHT
        self._set_swatch_color(self._bg_swatch, self._bg_color)
        self._set_swatch_color(self._grad_top_swatch, self._grad_top)
        self._set_swatch_color(self._grad_bottom_swatch, self._grad_bottom)
        self._apply()

    def _on_bg_color_picked(self) -> None:
        color = QColorDialog.getColor(QColor.fromRgbF(*self._bg_color), self)
        if not color.isValid():
            return
        self._bg_color = (color.redF(), color.greenF(), color.blueF())
        self._bg_custom_radio.setChecked(True)
        self._set_swatch_color(self._bg_swatch, self._bg_color)
        self._apply()

    def _on_cell_color_picked(self) -> None:
        color = QColorDialog.getColor(QColor.fromRgbF(*self._cell_color), self)
        if not color.isValid():
            return
        self._cell_color = (color.redF(), color.greenF(), color.blueF())
        self._set_swatch_color(self._cell_swatch, self._cell_color)
        self._apply()

    def _on_grad_top_picked(self) -> None:
        color = QColorDialog.getColor(QColor.fromRgbF(*self._grad_top), self)
        if color.isValid():
            self._grad_top = (color.redF(), color.greenF(), color.blueF())
            self._set_swatch_color(self._grad_top_swatch, self._grad_top)
            self._apply()

    def _on_grad_bottom_picked(self) -> None:
        color = QColorDialog.getColor(QColor.fromRgbF(*self._grad_bottom), self)
        if color.isValid():
            self._grad_bottom = (color.redF(), color.greenF(), color.blueF())
            self._set_swatch_color(self._grad_bottom_swatch, self._grad_bottom)
            self._apply()

    def _on_add_element(self, sym: str) -> None:
        if sym in self._overrides:
            return
        self._overrides[sym] = [*self._scheme_of(sym), 1.0]
        self._rebuild_element_rows()
        self._apply()

    def _on_element_color_picked(self, sym: str) -> None:
        rgba = self._overrides.get(sym, [*self._scheme_of(sym), 1.0])
        color = QColorDialog.getColor(QColor.fromRgbF(*rgba[:3]), self)
        if not color.isValid():
            return
        self._overrides[sym] = [color.redF(), color.greenF(), color.blueF(),
                                rgba[3]]
        swatch = self._row_swatch_for(sym)
        if swatch is not None:
            self._set_swatch_color(swatch, self._overrides[sym][:3])
        self._apply()

    def _on_element_opacity(self, sym: str, value: int) -> None:
        rgba = self._overrides.get(sym, [*self._scheme_of(sym), 1.0])
        self._overrides[sym] = [*rgba[:3], value / 100.0]
        self._apply()

    def _on_element_reset(self, sym: str) -> None:
        self._overrides.pop(sym, None)
        self._rebuild_element_rows()
        self._apply()

    def _on_reset_all_overrides(self) -> None:
        self._overrides.clear()
        self._rebuild_element_rows()
        self._apply()

    def _on_reset_defaults(self) -> None:
        self._load_settings_into_controls(RenderSettings.default())
        self._apply()

    # ------------------------------------------------------------------
    # Accept / reject
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        config = AppConfig()
        config.render_settings = self._current_settings()
        config.sync()
        self.accept()

    def reject(self) -> None:
        """Restore the original settings to the viewport before closing."""
        self._viewport.set_render_settings(self._original)
        super().reject()
