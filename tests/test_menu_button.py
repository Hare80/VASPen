"""MenuButton semantics: the QComboBox subset used across VASPen dialogs."""

from PySide6.QtWidgets import QComboBox

from vaspen.ui.menu_button import MenuButton


def _record(button):
    events = []
    button.currentIndexChanged.connect(lambda i: events.append(("index", i)))
    button.currentTextChanged.connect(lambda t: events.append(("text", t)))
    return events


def test_add_items_never_emits_and_first_item_is_current(qtbot):
    btn = MenuButton()
    qtbot.addWidget(btn)
    events = _record(btn)

    btn.addItems(["A", "B", "C"])

    assert events == []
    assert btn.currentIndex() == 0
    assert btn.currentText() == "A"
    assert btn.count() == 3
    assert [btn.itemText(i) for i in range(btn.count())] == ["A", "B", "C"]


def test_set_current_index_emits_only_on_change(qtbot):
    btn = MenuButton()
    qtbot.addWidget(btn)
    btn.addItems(["A", "B"])
    events = _record(btn)

    btn.setCurrentIndex(1)
    assert events == [("text", "B"), ("index", 1)]
    assert btn.currentText() == "B"
    assert btn._actions[1].isChecked() and not btn._actions[0].isChecked()

    btn.setCurrentIndex(1)  # no-op: no signals
    assert events == [("text", "B"), ("index", 1)]


def test_block_signals_suppresses_emission(qtbot):
    btn = MenuButton()
    qtbot.addWidget(btn)
    btn.addItems(["A", "B"])
    events = _record(btn)

    btn.blockSignals(True)
    btn.setCurrentIndex(1)
    btn.blockSignals(False)

    assert events == []
    # display still updated while blocked (QComboBox semantics)
    assert btn.currentIndex() == 1
    assert btn.currentText() == "B"


def test_clear_resets_without_emitting(qtbot):
    btn = MenuButton()
    qtbot.addWidget(btn)
    btn.addItems(["A", "B"])
    events = _record(btn)
    btn.setCurrentIndex(1)
    events.clear()

    btn.clear()

    assert events == []
    assert btn.count() == 0
    assert btn.currentIndex() == -1
    assert btn.currentText() == ""
    # re-populate after clear: first item becomes current again
    btn.addItem("X")
    assert btn.currentIndex() == 0 and btn.currentText() == "X"


def test_set_current_index_minus_one_is_legal(qtbot):
    btn = MenuButton()
    qtbot.addWidget(btn)
    btn.addItems(["A", "B"])
    events = _record(btn)

    btn.setCurrentIndex(-1)

    assert btn.currentIndex() == -1
    assert btn.currentText() == ""
    assert not any(a.isChecked() for a in btn._actions)
    assert events == [("text", ""), ("index", -1)]


def test_set_current_text_exact_match(qtbot):
    btn = MenuButton()
    qtbot.addWidget(btn)
    btn.addItems(["X", "Y", "Z"])

    btn.setCurrentText("Z")
    assert btn.currentIndex() == 2 and btn.currentText() == "Z"

    # miss → no selection (QComboBox setCurrentText semantics)
    btn.setCurrentText("nope")
    assert btn.currentIndex() == -1


def test_find_text(qtbot):
    btn = MenuButton()
    qtbot.addWidget(btn)
    btn.addItems(["Ga", "Ga_d"])
    assert btn.findText("Ga_d") == 1
    assert btn.findText("Ga_s") == -1


def test_action_trigger_selects_item(qtbot):
    """Clicking a menu item (action trigger) drives the selection."""
    btn = MenuButton()
    qtbot.addWidget(btn)
    btn.addItems(["A", "B"])
    events = _record(btn)

    btn._actions[1].trigger()

    assert btn.currentIndex() == 1
    assert events == [("text", "B"), ("index", 1)]


def test_is_not_a_qcombobox(qtbot):
    """The whole point: this is a QToolButton, not a QComboBox."""
    btn = MenuButton()
    qtbot.addWidget(btn)
    assert not isinstance(btn, QComboBox)


def test_set_item_text_updates_in_place():
    """setItemText never emits and keeps the menu structure intact."""
    from PySide6.QtWidgets import QApplication

    from vaspen.ui.menu_button import MenuButton

    btn = MenuButton()
    btn.addItems(["a", "b", "c"])
    actions = list(btn._actions)
    events = []
    btn.currentIndexChanged.connect(lambda i: events.append(i))
    btn.setCurrentIndex(1)
    btn.setItemText(1, "b2")
    assert btn.itemText(1) == "b2"
    assert btn.currentText() == "b2"  # current item's text follows
    assert list(btn._actions) == actions  # no structural change
    assert events == [1]  # setItemText never emitted
    assert btn.currentIndex() == 1  # selection untouched
