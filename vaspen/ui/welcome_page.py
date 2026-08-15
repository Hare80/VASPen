"""Welcome page — shown in the central stack when no structure is loaded.

Standard layout: title, tagline, a recent-files list (click selects,
double-click/Enter opens), New Structure / Open / Browse buttons, and
a drag-and-drop hint. All strings live in the "WelcomePage" i18n
context (enforced by tests/test_i18n.py); the widget is persistent
(never destroyed), so it implements changeEvent/_retranslate per
CLAUDE.md §11.2.

No stylesheet, no hardcoded colors — the theme's palette + QSS rules
style every widget here (CLAUDE.md §7.9); the title uses a code-level
QFont (the no-font-rules policy covers QSS only).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

#: Maximum height of the recent-files list (10 entries fit with room
#: to spare; the page stretches the rest as empty space).
_LIST_MAX_HEIGHT = 260


class WelcomePage(QWidget):
    """Empty-state landing page with recent files and quick actions."""

    open_file_requested = Signal(str)   # full path (double-click / Enter)
    new_requested = Signal()
    open_requested = Signal()           # open the SELECTED recent file
    browse_requested = Signal()         # file dialog

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 40, 48, 32)
        layout.setSpacing(10)

        title = QLabel(self.tr("VASPen"), self)
        title_font = QFont(self.font())
        title_font.setPointSize(title_font.pointSize() + 8)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(title)

        tagline = QLabel(
            self.tr("Visual structure modeling for VASP"), self)
        tagline.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(tagline)

        layout.addSpacing(18)

        recent_label = QLabel(self.tr("Recent Files"), self)
        recent_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(recent_label)

        self._recent_list = QListWidget(self)
        self._recent_list.setMaximumHeight(_LIST_MAX_HEIGHT)
        self._recent_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Click selects only; double-click / Enter open (itemActivated).
        self._recent_list.itemActivated.connect(self._on_item_activated)
        self._recent_list.itemSelectionChanged.connect(
            self._update_open_enabled)
        layout.addWidget(self._recent_list)

        dbl_hint = QLabel(self.tr("Double-click to open a file"), self)
        dbl_hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(dbl_hint)

        self._no_recent_label = QLabel(
            self.tr("(No recent files)"), self)
        self._no_recent_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._no_recent_label)
        self._no_recent_label.hide()  # default: list visible

        layout.addSpacing(14)

        button_row = QHBoxLayout()
        button_row.addStretch()
        self._new_btn = QPushButton(self.tr("New Structure..."), self)
        self._new_btn.clicked.connect(
            lambda checked=False: self.new_requested.emit())
        button_row.addWidget(self._new_btn)
        self._open_btn = QPushButton(self.tr("Open"), self)
        self._open_btn.setToolTip(self.tr("Open the selected file"))
        self._open_btn.setEnabled(False)  # enabled with a selection
        self._open_btn.clicked.connect(
            lambda checked=False: self.open_requested.emit())
        button_row.addWidget(self._open_btn)
        self._browse_btn = QPushButton(self.tr("Browse..."), self)
        self._browse_btn.clicked.connect(
            lambda checked=False: self.browse_requested.emit())
        button_row.addWidget(self._browse_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        layout.addStretch()

        hint = QLabel(self.tr(
            "Drag and drop a structure file anywhere in this window."),
            self)
        hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(hint)

        self._title = title
        self._tagline = tagline
        self._recent_label = recent_label
        self._dbl_hint = dbl_hint
        self._hint = hint

    # ------------------------------------------------------------------
    # Recent files
    # ------------------------------------------------------------------

    def set_recent_files(self, files: list[str]) -> None:
        """Fill the recent list (filename + full-path tooltip). An
        empty list hides the list and shows the hint label."""
        self._recent_list.clear()
        for fp in files:
            item = QListWidgetItem(Path(fp).name)
            item.setToolTip(fp)
            item.setData(Qt.ItemDataRole.UserRole, fp)
            self._recent_list.addItem(item)
        has_files = bool(files)
        self._recent_list.setVisible(has_files)
        self._no_recent_label.setVisible(not has_files)
        self._update_open_enabled()

    def selected_file(self) -> str | None:
        """Full path of the currently selected recent file (None if no
        selection — the Open button is disabled then anyway)."""
        item = self._recent_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _update_open_enabled(self) -> None:
        self._open_btn.setEnabled(self.selected_file() is not None)

    def _on_item_activated(self, item: QListWidgetItem) -> None:
        # itemActivated = double-click OR Enter.
        self.open_file_requested.emit(
            item.data(Qt.ItemDataRole.UserRole))

    # ------------------------------------------------------------------
    # i18n (persistent widget — re-applied on live language switch)
    # ------------------------------------------------------------------

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.LanguageChange:
            self._retranslate()
        super().changeEvent(event)

    def _retranslate(self) -> None:
        self._title.setText(self.tr("VASPen"))
        self._tagline.setText(self.tr("Visual structure modeling for VASP"))
        self._recent_label.setText(self.tr("Recent Files"))
        self._dbl_hint.setText(self.tr("Double-click to open a file"))
        self._no_recent_label.setText(self.tr("(No recent files)"))
        self._new_btn.setText(self.tr("New Structure..."))
        self._open_btn.setText(self.tr("Open"))
        self._open_btn.setToolTip(self.tr("Open the selected file"))
        self._browse_btn.setText(self.tr("Browse..."))
        self._hint.setText(self.tr(
            "Drag and drop a structure file anywhere in this window."))
