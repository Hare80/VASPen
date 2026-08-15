"""Welcome page widget tests (v0.2 2026-08-15).

The page is the empty-state landing panel in the central stack:
recent-files list (click selects, double-click/Enter opens), New
Structure / Open (opens the SELECTED file) / Browse (file dialog)
quick actions, and a drag-and-drop hint. No hardcoded colors, no own
stylesheet — it must compose purely from the theme palette/QSS.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QListWidgetItem

from vaspen.ui.welcome_page import WelcomePage


def _page_with_files(qtbot, files=None):
    page = WelcomePage()
    qtbot.addWidget(page)
    page.set_recent_files(files or [
        "D:/vasp/Cu_100_slab_new.vasp",
        "D:/vasp/benzene.xyz",
    ])
    return page


def test_empty_recents_show_hint_instead_of_list(qtbot):
    page = WelcomePage()
    qtbot.addWidget(page)
    page.set_recent_files([])
    assert page._recent_list.isHidden()
    assert not page._no_recent_label.isHidden()


def test_recents_fill_list_with_tooltips_and_data(qtbot):
    page = _page_with_files(qtbot)
    assert not page._recent_list.isHidden()
    assert page._no_recent_label.isHidden()
    assert page._recent_list.count() == 2
    assert page._recent_list.item(0).text() == "Cu_100_slab_new.vasp"
    assert page._recent_list.item(0).toolTip() == "D:/vasp/Cu_100_slab_new.vasp"
    assert (page._recent_list.item(0).data(Qt.ItemDataRole.UserRole)
            == "D:/vasp/Cu_100_slab_new.vasp")


def test_item_click_selects_without_opening(qtbot):
    """A plain click highlights the row only — opening happens on
    double-click/Enter (user decision: single-click open was
    unintuitive)."""
    page = _page_with_files(qtbot)
    opened = []
    page.open_file_requested.connect(opened.append)
    page._recent_list.setCurrentRow(0)  # selection, like a click
    assert opened == []
    assert page._recent_list.currentRow() == 0


def test_item_activated_emits_open_file_requested(qtbot):
    """itemActivated = double-click OR Enter."""
    page = _page_with_files(qtbot)
    opened = []
    page.open_file_requested.connect(opened.append)
    item: QListWidgetItem = page._recent_list.item(0)
    page._on_item_activated(item)
    assert opened == ["D:/vasp/Cu_100_slab_new.vasp"]


def test_selected_file_api(qtbot):
    page = WelcomePage()
    qtbot.addWidget(page)
    assert page.selected_file() is None
    page.set_recent_files(["D:/vasp/cu111.vasp"])
    page._recent_list.setCurrentRow(0)
    assert page.selected_file() == "D:/vasp/cu111.vasp"


def test_open_button_enabled_only_with_selection(qtbot):
    page = _page_with_files(qtbot)
    assert not page._open_btn.isEnabled()  # nothing selected yet
    page._recent_list.setCurrentRow(0)
    assert page._open_btn.isEnabled()
    page.set_recent_files([])  # clearing drops the selection
    assert not page._open_btn.isEnabled()


def test_buttons_emit_requests(qtbot):
    page = _page_with_files(qtbot)
    new_calls = []
    open_calls = []
    browse_calls = []
    page.new_requested.connect(lambda: new_calls.append(1))
    page.open_requested.connect(lambda: open_calls.append(1))
    page.browse_requested.connect(lambda: browse_calls.append(1))
    page._new_btn.click()
    page._recent_list.setCurrentRow(0)
    page._open_btn.click()
    page._browse_btn.click()
    assert new_calls and open_calls and browse_calls


def test_language_change_retranslates_without_crashing(qtbot):
    page = _page_with_files(qtbot)
    # The broadcast path of a live language switch: LanguageChange.
    event = QEvent(QEvent.Type.LanguageChange)
    page.changeEvent(event)  # must not raise; strings re-applied
    assert page._title.text() == "VASPen"
    assert page._dbl_hint.text() == "Double-click to open a file"
    assert page._browse_btn.text() == "Browse..."
