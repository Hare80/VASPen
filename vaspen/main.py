"""VASPen — Main entry point.

Launches the QApplication, main window, and event loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from vaspen.ui.main_window import MainWindow
from vaspen.utils.config import AppConfig
from vaspen.utils.logger import setup_logger


def _load_app_icon(icons_dir: Path) -> QIcon:
    """Build a multi-size icon so Windows picks the crispiest match.

    Uses the per-size square PNGs (16-256) generated from the source
    image; falls back to the full landscape app.png / app.svg.
    """
    icon = QIcon()
    for size in (16, 32, 48, 64, 128, 256):
        path = icons_dir / f"app_{size}.png"
        if path.exists():
            icon.addFile(str(path), QSize(size, size))
    if icon.isNull():
        for name in ("app.png", "app.svg"):
            path = icons_dir / name
            if path.exists():
                return QIcon(str(path))
    return icon


def _apply_windows_identity(app: QApplication, icons_dir: Path) -> None:
    """Give the taskbar button the VASPen icon in dev mode.

    Qt's setWindowIcon alone does not replace the python.exe icon on
    the taskbar; set the HICON on the main window's HWND directly via
    WM_SETICON (the frozen exe gets its icon from the embedded .ico).
    """
    try:
        import ctypes
        from ctypes import wintypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "VASPen.0.1"
        )
        ico_path = icons_dir / "app.ico"
        if not ico_path.exists():
            return

        def _set_hwnd_icon(window) -> None:
            hwnd = int(window.winId())
            h_icon_big = ctypes.windll.user32.LoadImageW(
                0, str(ico_path), 1, 32, 32, 0x10)  # IMAGE_ICON, LR_LOADFROMFILE
            h_icon_small = ctypes.windll.user32.LoadImageW(
                0, str(ico_path), 1, 16, 16, 0x10)
            if h_icon_big:
                ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 1, h_icon_big)  # WM_SETICON, ICON_BIG
            if h_icon_small:
                ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 0, h_icon_small)  # ICON_SMALL

        for widget in app.topLevelWidgets():
            _set_hwnd_icon(widget)
    except Exception:
        pass


def main() -> int:
    """Application entry point.

    Returns:
        0 on normal exit, non-zero on error.
    """
    # --- Application ---
    app = QApplication(sys.argv)
    app.setApplicationName("VASPen")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("VASPen")
    app.setOrganizationDomain("vaspen.dev")

    # --- Icon ---
    icons_dir = Path(__file__).parent / "resources" / "icons"
    app.setWindowIcon(_load_app_icon(icons_dir))

    # --- Logger ---
    logger = setup_logger()

    # --- Main Window (loads translations itself; switchable live) ---
    config = AppConfig()
    lang = config.language
    window = MainWindow()
    window.show()

    # --- Windows taskbar icon (dev mode otherwise shows python.exe) ---
    if sys.platform == "win32":
        _apply_windows_identity(app, icons_dir)

    # Open a file passed on the command line (e.g. double-click file association)
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if Path(filepath).exists():
            window._open_file(filepath)
            logger.info("Opened CLI file: %s", filepath)
        else:
            logger.warning("CLI file not found: %s", filepath)

    logger.info("VASPen started (lang=%s)", lang)

    # --- Event Loop ---
    exit_code = app.exec()

    # --- Cleanup ---
    logger.info("VASPen exiting (code=%d)", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
