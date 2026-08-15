"""Application theming — light/dark via Fusion style + QPalette + QSS.

SETTLED POLICY (CLAUDE.md §7.9): BOTH themes use the Qt "Fusion" style
with a QPalette and an app-level stylesheet (vaspen/resources/themes/
{light,dark}.qss) — the light theme deliberately approximates the
previous native Windows look. Switching is live (no restart): Settings
persists ``AppConfig.theme`` and MainWindow calls :func:`apply_theme`
afterwards.

The themes directory is resolved from this module's own ``__file__``:
the package layout is intact in frozen builds (only the entry script
``main.py`` is flattened to ``_internal/main.py``), so a ``__file__``-
relative path is correct in dev AND frozen — do not move it into
main.py.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QPalette

from vaspen.utils.logger import logger

#: Supported theme names (mirrored by the AppConfig.theme setter).
THEMES = ("light", "dark")

#: Currently applied theme; set by :func:`apply_theme`.
_current: str = "light"


def themes_dir() -> Path:
    """Directory holding light.qss / dark.qss (dev + frozen safe)."""
    return Path(__file__).parent.parent / "resources" / "themes"


def load_qss(name: str) -> str:
    """Read ``{name}.qss`` (UTF-8). A missing file logs a warning and
    degrades to an empty stylesheet — the app must never fail startup
    over a theme file."""
    path = themes_dir() / f"{name}.qss"
    try:
        return path.read_text("utf-8")
    except OSError as exc:
        logger.warning("Could not read theme stylesheet %s: %s", path, exc)
        return ""


def current_theme() -> str:
    """Name of the theme currently applied ("light" until first apply)."""
    return _current


def is_dark() -> bool:
    """Whether the currently applied theme is dark."""
    return _current == "dark"


# ----------------------------------------------------------------------
# Palettes
# ----------------------------------------------------------------------

def _palette(
    window: str, base: str, text: str, button: str, button_text: str,
    highlight: str, border_mid: str, disabled: str, input_base: str,
    bright_text: str,
) -> QPalette:
    """Build a Fusion palette. All three ColorGroups are set (the
    Disabled group is mandatory — Fusion otherwise renders disabled
    text at full contrast)."""
    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: window,
        QPalette.ColorRole.WindowText: text,
        QPalette.ColorRole.Base: base,
        QPalette.ColorRole.AlternateBase: button,
        QPalette.ColorRole.Text: text,
        QPalette.ColorRole.Button: button,
        QPalette.ColorRole.ButtonText: button_text,
        QPalette.ColorRole.BrightText: bright_text,
        QPalette.ColorRole.Highlight: highlight,
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.Link: "#0066cc",
        QPalette.ColorRole.ToolTipBase: base,
        QPalette.ColorRole.ToolTipText: text,
        QPalette.ColorRole.PlaceholderText: disabled,
        QPalette.ColorRole.Mid: border_mid,
    }
    for role, color in roles.items():
        for group in (QPalette.ColorGroup.Active,
                      QPalette.ColorGroup.Inactive):
            pal.setColor(group, role, QColor(color))
    # Input fields sit slightly apart from the panel base in dark mode.
    pal.setColor(QPalette.ColorGroup.Active, QPalette.ColorRole.Base,
                 QColor(input_base))
    pal.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Base,
                 QColor(input_base))
    disabled_roles = {
        QPalette.ColorRole.Text: disabled,
        QPalette.ColorRole.ButtonText: disabled,
        QPalette.ColorRole.WindowText: disabled,
        QPalette.ColorRole.Highlight: button,
        QPalette.ColorRole.HighlightedText: disabled,
    }
    for role, color in disabled_roles.items():
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(color))
    return pal


_PALETTES = {
    "light": _palette(
        window="#f0f0f0", base="#ffffff", text="#1a1a1a",
        button="#e8e8e8", button_text="#1a1a1a", highlight="#0078d4",
        border_mid="#dcdcdc", disabled="#8a8a8a", input_base="#ffffff",
        bright_text="#c00000",
    ),
    "dark": _palette(
        # Window matches the GL dark background default (#1e1e24).
        window="#1e1e24", base="#26262e", text="#e8e8ee",
        button="#2e2e38", button_text="#e8e8ee", highlight="#3179b8",
        border_mid="#2a2a33", disabled="#6f6f7a", input_base="#16161c",
        bright_text="#ff5555",
    ),
}


def build_palette(name: str) -> QPalette:
    """QPalette for a theme name (validated by the caller)."""
    return QPalette(_PALETTES[name])


# ----------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------

def apply_theme(app, theme_name: str, themes_dir_: Path | None = None) -> None:
    """Apply a theme app-wide (Fusion style + palette + stylesheet).

    Args:
        app: The QApplication.
        theme_name: "light" or "dark".
        themes_dir_: Override for the QSS directory (tests only).

    Raises:
        ValueError: If theme_name is not in THEMES.
    """
    global _current
    if theme_name not in THEMES:
        raise ValueError(f"unknown theme {theme_name!r} (expected one of {THEMES})")
    # Order matters: setStyle resets the palette (Fusion initializes
    # its own), so style → palette → stylesheet.
    app.setStyle("Fusion")
    app.setPalette(build_palette(theme_name))
    if themes_dir_ is not None:
        app.setStyleSheet((themes_dir_ / f"{theme_name}.qss").read_text("utf-8"))
    else:
        app.setStyleSheet(load_qss(theme_name))
    _current = theme_name


def sync_background_for_theme(rs, theme_name: str):
    """Return render settings whose background matches the theme.

    When the background is still the OTHER theme's preset default
    (white ↔ #1e1e24 — i.e. the user never customized it), swap it to
    this theme's default so the GL viewport matches the chrome. A
    custom background is returned unchanged (same object).

    Lazy core import keeps utils free of a core dependency at module
    load time (same pattern as AppConfig.render_settings).
    """
    from vaspen.core.render_settings import (
        BACKGROUND_DARK,
        BACKGROUND_LIGHT,
        RenderSettings,
    )

    current = tuple(rs.background_color)
    if theme_name == "dark":
        if current == BACKGROUND_LIGHT:
            new = RenderSettings.from_dict(rs.to_dict())
            new.background_color = BACKGROUND_DARK
            return new
    elif theme_name == "light":
        if current == BACKGROUND_DARK:
            new = RenderSettings.from_dict(rs.to_dict())
            new.background_color = BACKGROUND_LIGHT
            return new
    return rs
