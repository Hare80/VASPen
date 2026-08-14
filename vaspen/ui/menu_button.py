"""Index-based text choice control: QToolButton + QMenu (no QComboBox)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu, QToolButton, QWidget


class MenuButton(QToolButton):
    """A stateful text choice presented as a menu (the "View button" pattern).

    Drop-in replacement for the QComboBox subset VASPen uses. QComboBox's
    popup lives inside its own state machine and its open flow re-enters on
    this platform (ghosted/doubled popup — see DisplayOptionsDialog's
    choice buttons); a QMenu popup has no such path. Selection is shown as
    checkable QActions in an exclusive group; the button text is the
    current item.

    Signal semantics match QComboBox for the used subset:
    - addItem/addItems/clear never emit
    - setCurrentIndex emits currentTextChanged + currentIndexChanged only
      when the index actually changes (programmatic changes emit too)
    - QObject.blockSignals suppresses emission (checked at emit time)
    """

    currentIndexChanged = Signal(int)
    currentTextChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._menu = QMenu(self)
        self._group = QActionGroup(self)
        self._group.setExclusive(True)
        self._actions: list[QAction] = []
        self._index: int = -1
        self.setMenu(self._menu)

    # ------------------------------------------------------------------
    # Items
    # ------------------------------------------------------------------

    def addItem(self, text: str) -> None:
        """Append a choice. Never emits (QComboBox subset behavior)."""
        action = QAction(text, self)
        action.setCheckable(True)
        action.triggered.connect(
            lambda checked=False, a=action: self._on_action_triggered(a))
        self._group.addAction(action)
        self._menu.addAction(action)
        self._actions.append(action)
        if self._index == -1:
            # First item becomes current (like QComboBox), silently.
            self._index = 0
            self.setText(text)
            action.setChecked(True)

    def addItems(self, texts: list[str] | tuple[str, ...]) -> None:
        for text in texts:
            self.addItem(text)

    def clear(self) -> None:
        """Remove all items; index -> -1. Never emits."""
        for action in self._actions:
            self._group.removeAction(action)
            self._menu.removeAction(action)
        self._actions.clear()
        self._index = -1
        self.setText("")

    def count(self) -> int:
        return len(self._actions)

    def itemText(self, index: int) -> str:
        return self._actions[index].text()

    def setItemText(self, index: int, text: str) -> None:
        """Update an item's label in place — never emits, keeps the menu
        structure, selection and popup state intact (no flicker)."""
        if 0 <= index < len(self._actions):
            self._actions[index].setText(text)
            if index == self._index:
                self.setText(text)

    # ------------------------------------------------------------------
    # Current selection
    # ------------------------------------------------------------------

    def currentIndex(self) -> int:
        return self._index

    def currentText(self) -> str:
        if 0 <= self._index < len(self._actions):
            return self._actions[self._index].text()
        return ""

    def findText(self, text: str) -> int:
        """Exact-match index of ``text``, or -1 (like QComboBox)."""
        for i, action in enumerate(self._actions):
            if action.text() == text:
                return i
        return -1

    def setCurrentText(self, text: str) -> None:
        self.setCurrentIndex(self.findText(text))

    def setCurrentIndex(self, index: int) -> None:
        """Select an item; -1 selects none. Emits only on a real change."""
        index = int(index)
        if index == self._index or not (-1 <= index < len(self._actions)):
            return
        self._index = index
        if index >= 0:
            self.setText(self._actions[index].text())
            self._actions[index].setChecked(True)
        else:
            self.setText("")
            for action in self._actions:
                action.setChecked(False)
        if self.signalsBlocked():
            return
        self.currentTextChanged.emit(self.currentText())
        self.currentIndexChanged.emit(index)

    # ------------------------------------------------------------------

    def _on_action_triggered(self, action: QAction) -> None:
        self.setCurrentIndex(self._actions.index(action))
