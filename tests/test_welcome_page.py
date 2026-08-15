"""Welcome page widget tests (v0.2 2026-08-15).

The page is the empty-state landing panel in the central stack:
recent-files list (single click opens), New/Open quick actions, a
drag-and-drop hint. No hardcoded colors, no own stylesheet — it must
compose purely from the theme palette/QSS.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QListWidgetItem

from vaspen.ui.welcome_page import WelcomePage


def test_empty_recents_show_hint_instead_of_list(qtbot):
    page = WelcomePage()
    qtbot.addWidget(page)
    page.set_recent_files([])
    assert page._recent_list.isHidden()
    assert not page._no_recent_label.isHidden()


def test_recents_fill_list_with_tooltips_and_data(qtbot):
    page = WelcomePage()
    qtbot.addWidget(page)
    page.set_recent_files([
        "D:/vasp/Cu_100_slab_new.vasp",
        "D:/vasp/benzene.xyz",
    ])
    assert not page._recent_list.isHidden()
    assert page._no_recent_label.isHidden()
    assert page._recent_list.count() == 2
    assert page._recent_list.item(0).text() == "Cu_100_slab_new.vasp"
    assert page._recent_list.item(0).toolTip() == "D:/vasp/Cu_100_slab_new.vasp"
    assert (page._recent_list.item(0).data(Qt.ItemDataRole.UserRole)
            == "D:/vasp/Cu_100_slab_new.vasp")


def test_item_click_emits_open_file_requested(qtbot):
    page = WelcomePage()
    qtbot.addWidget(page)
    page.set_recent_files(["D:/vasp/cu111.vasp"])
    opened = []
    page.open_file_requested.connect(opened.append)
    item: QListWidgetItem = page._recent_list.item(0)
    page._on_item_clicked(item)
    assert opened == ["D:/vasp/cu111.vasp"]


def test_buttons_emit_requests(qtbot):
    page = WelcomePage()
    qtbot.addWidget(page)
    new_calls = []
    open_calls = []
    page.new_requested.connect(lambda: new_calls.append(1))
    page.open_requested.connect(lambda: open_calls.append(1))
    page._new_btn.click()
    page._open_btn.click()
    assert new_calls and open_calls


def test_language_change_retranslates_without_crashing(qtbot):
    page = WelcomePage()
    qtbot.addWidget(page)
    # The broadcast path of a live language switch: LanguageChange.
    event = QEvent(QEvent.Type.LanguageChange)
    page.changeEvent(event)  # must not raise; strings re-applied
    assert page._title.text() == "VASPen"
