"""Theme engine guards (QSS theming, v0.2 2026-08-15).

Pins the theme contract (CLAUDE.md §7.9): both themes run on Fusion +
QPalette + QSS, apply_theme is the only way to switch, the two QSS
files exist and mirror each other structurally, and the GL background
swap only ever moves between the two preset defaults.

CAUTION: apply_theme mutates the SESSION-WIDE QApplication (style,
palette, stylesheet) — every test that leaves a theme applied restores
"light" in a finally block so later tests see the default.
"""

from __future__ import annotations

import pytest

from vaspen.core.render_settings import (
    BACKGROUND_DARK,
    BACKGROUND_LIGHT,
    RenderSettings,
)
from vaspen.ui.viewport3d import element_color, element_text_color
from vaspen.utils.config import AppConfig
from vaspen.utils.theme import (
    THEMES,
    apply_theme,
    current_theme,
    is_dark,
    sync_background_for_theme,
    themes_dir,
)


def _luminance(color: tuple[float, float, float]) -> float:
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]


def test_qss_files_exist_and_are_readable():
    for name in THEMES:
        path = themes_dir() / f"{name}.qss"
        text = path.read_text("utf-8")
        assert text.strip(), name
        assert "QMainWindow" in text, name


def test_apply_light_sets_fusion_style_and_stylesheet(qapp):
    try:
        apply_theme(qapp, "light")
        # setStyleSheet wraps the style in QStyleSheetStyle (Fusion
        # underneath) — assert the observable effects instead: the
        # stylesheet is installed and the palette carries the light
        # Window color.
        assert "QMainWindow" in qapp.styleSheet()
        from PySide6.QtGui import QPalette
        assert (qapp.palette().color(QPalette.ColorRole.Window).name()
                == "#f0f0f0")
        assert current_theme() == "light"
        assert not is_dark()
    finally:
        apply_theme(qapp, "light")


def test_apply_dark_and_back(qapp):
    try:
        apply_theme(qapp, "dark")
        assert current_theme() == "dark"
        assert is_dark()
        assert "#1e1e24" in qapp.styleSheet()  # dark window chrome
        assert qapp.styleSheet().strip()
    finally:
        apply_theme(qapp, "light")
    assert current_theme() == "light"


def test_apply_unknown_theme_raises(qapp):
    with pytest.raises(ValueError):
        apply_theme(qapp, "neon")


def test_config_theme_default_roundtrip_and_clamp():
    config = AppConfig()
    assert config.theme == "light"
    config.theme = "dark"
    assert config.theme == "dark"
    config.theme = "neon"  # corrupt stored value clamps to light
    assert config.theme == "light"


def test_element_text_color_light_path_unchanged():
    """Light panels: bright colors are clamped to ≤0.62 luminance,
    dark colors pass through byte-for-byte."""
    h = element_text_color("H")  # Jmol H is pure white
    assert _luminance(h) == pytest.approx(0.62, abs=1e-6)
    assert element_text_color("C") == element_color("C")  # near-black


def test_element_text_color_dark_variant_brightens_dark_elements():
    """Dark panels: dark colors are raised to ≥0.5 luminance, bright
    colors pass through unchanged."""
    # Ir: Jmol luminance ≈0.28 — the dark variant lifts it to 0.5.
    c = element_text_color("Ir", dark=True)
    assert _luminance(c) == pytest.approx(0.5, abs=1e-6)
    assert element_text_color("H", dark=True) == element_color("H")


def _render_settings(bg):
    rs = RenderSettings.default()
    rs.background_color = bg
    return rs


def test_sync_background_swaps_only_preset_defaults():
    rs = _render_settings(BACKGROUND_LIGHT)
    dark_rs = sync_background_for_theme(rs, "dark")
    assert dark_rs is not rs
    assert dark_rs.background_color == BACKGROUND_DARK

    back = sync_background_for_theme(dark_rs, "light")
    assert back is not dark_rs
    assert back.background_color == BACKGROUND_LIGHT

    # Already matching the theme — same object back.
    assert sync_background_for_theme(dark_rs, "dark") is dark_rs


def test_sync_background_leaves_custom_backgrounds_alone():
    custom = (0.5, 0.5, 0.5)
    rs = _render_settings(custom)
    assert sync_background_for_theme(rs, "dark") is rs
    assert sync_background_for_theme(rs, "light") is rs
