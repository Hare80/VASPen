"""VASPen — Main entry point.

Launches the QApplication, main window, and event loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from vaspen.ui.main_window import MainWindow
from vaspen.utils.config import AppConfig
from vaspen.utils.logger import setup_logger


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
    icon_path = Path(__file__).parent / "resources" / "icons" / "app.svg"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

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
