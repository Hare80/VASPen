"""Read-only text viewer for project run files (§7.15).

Shows the raw content of a run's text files (INCAR, KPOINTS, OSZICAR,
OUTCAR, run.json, …). Large files are truncated to a preview window —
binary outputs (WAVECAR/CHGCAR) are meant to be listed, never loaded
wholesale. Nothing here interprets the content: VASP outputs are
displayed raw, exactly as on disk (settled boundary).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

#: Files larger than this are previewed head-only.
PREVIEW_LIMIT_BYTES = 512 * 1024


class TextViewerDialog(QDialog):
    """Read-only preview of one file inside a project run."""

    def __init__(self, path: str | Path,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path = Path(path)

        truncated = False
        data = b""
        try:
            size = self._path.stat().st_size
            with open(self._path, "rb") as handle:
                data = handle.read(PREVIEW_LIMIT_BYTES)
            truncated = size > len(data)
        except OSError as exc:
            data = f"[read error: {exc}]".encode("utf-8")

        self.setWindowTitle(
            f"{self._path.name} — {self._path.parent.name}")

        layout = QVBoxLayout(self)
        self._meta = QLabel(
            f"{self._path}  ({size:,} bytes)" if not truncated else
            self.tr("{}  (showing first {} of {} bytes)").format(
                self._path, PREVIEW_LIMIT_BYTES, f"{size:,}"))
        layout.addWidget(self._meta)

        self._text = QPlainTextEdit(self)
        self._text.setReadOnly(True)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._text.setFont(font)
        text = data.decode("utf-8", errors="replace")
        if truncated:
            text += "\n\n[…truncated — open the file externally for the rest]"
        self._text.setPlainText(text)
        layout.addWidget(self._text, 1)

        bottom = QHBoxLayout()
        note = QLabel(self.tr("Read-only — raw file content"))
        bottom.addWidget(note, 1)
        reveal = QPushButton(self.tr("Show in Explorer"))
        reveal.clicked.connect(self._reveal)
        bottom.addWidget(reveal)
        layout.addLayout(bottom)

        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        layout.addWidget(close)

    def _reveal(self) -> None:
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self._path.parent)))
