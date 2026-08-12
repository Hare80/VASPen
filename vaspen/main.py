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

    # --- Windows taskbar identity (dev mode otherwise shows python.exe) ---
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "VASPen.0.1"
            )
        except Exception:
            pass

    # --- Logger ---
    logger = setup_logger()

    # --- Main Window (loads translations itself; switchable live) ---
    config = AppConfig()
    lang = config.language
    window = MainWindow()
    window.show()

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
