"""About dialog — large app logo + version information."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout


class AboutDialog(QDialog):
    """Modal about box showing the app icon at 160px and version info."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("About VASPen"))
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        # Large logo (the full icon art, scaled to 160 px)
        logo = QLabel()
        logo.setAlignment(Qt.AlignCenter)
        icon_path = Path(__file__).parent.parent / "resources" / "icons" / "app.png"
        if icon_path.exists():
            pixmap = QPixmap(str(icon_path))
            logo.setPixmap(pixmap.scaled(
                160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        layout.addWidget(logo)

        info = QLabel(self.tr(
            "<h2>VASPen v0.1.0</h2>"
            "<p>A cross-platform GUI for VASP first-principles calculations.</p>"
            "<p><b>Built with:</b> PySide6, ASE, pymatgen, Qt native OpenGL</p>"
            "<p>Free and open source (MIT License).</p>"
        ))
        info.setAlignment(Qt.AlignCenter)
        info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(info)
