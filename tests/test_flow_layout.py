"""Tests for FlowLayout's max_rows cap (main-window resize fix).

The edit toolbar's wrapped heightForWidth, evaluated at the main
window's MINIMUM width, once reported a ~330px required height —
pinning the window's minimum height at ~871px, above short screens'
usable height, so the window could not be resized vertically.

The items are plain fakes (FlowLayout is our code — minimumSize /
heightForWidth / _do_layout only ever call sizeHint / minimumSize /
setGeometry): no Qt widgets are created, which keeps these tests out
of the offscreen-platform flakiness that long widget-heavy sessions
can hit. The real-window regression is pinned by
tests/test_ui.py::test_main_window_shrinkable_vertically.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize

from vaspen.ui.flow_layout import FlowLayout


class _FakeItem:
    """Duck-typed QLayoutItem: fixed sizeHint, records setGeometry."""

    def __init__(self, width: int, height: int) -> None:
        self._size = QSize(width, height)
        self.geometry: QRect | None = None

    def sizeHint(self) -> QSize:
        return self._size

    def minimumSize(self) -> QSize:
        return self._size

    def setGeometry(self, rect: QRect) -> None:
        self.geometry = QRect(rect)


def _flow(n: int, width: int = 24, height: int = 19,
          max_rows: int | None = 2) -> FlowLayout:
    lay = FlowLayout(margin=0, h_spacing=2, v_spacing=2, max_rows=max_rows)
    for _ in range(n):
        lay.addItem(_FakeItem(width, height))
    return lay


def test_height_for_width_capped():
    lay = _flow(24)
    cap = lay._capped_height()
    assert cap == 40  # 2 rows x 19 + 1 x 2 spacing
    # narrow widths would wrap into many rows — the cap bites
    assert lay.heightForWidth(60) == cap
    # one natural row stays under the 2-row cap
    assert lay.heightForWidth(4000) < cap


def test_uncapped_layout_still_wraps():
    assert _flow(24, max_rows=None).heightForWidth(60) > \
        _flow(24, max_rows=2).heightForWidth(60)


def test_minimum_size_stays_per_item():
    """The cap changes heightForWidth only — minimumSize remains the
    tallest/widest single item (a faithful FlowLayout port)."""
    lay = _flow(24)
    assert lay.minimumSize() == QSize(24, 19)


def test_do_layout_still_positions_items():
    lay = _flow(3, width=10, height=5, max_rows=1)
    rect = QRect(0, 0, 30, 100)  # two 10px items fit, third wraps (clipped)
    lay.setGeometry(rect)
    geoms = [lay.itemAt(i).geometry for i in range(lay.count())]
    assert geoms[0].topLeft() == rect.topLeft()
    assert geoms[1].x() == 12  # 10 + 2 spacing
    assert geoms[2].y() > geoms[0].y()  # wrapped row (beyond the cap)
