"""App icon asset guards (v0.2, 2026-08-15).

SETTLED POLICY — the icon artwork is NEVER cropped and NEVER zoomed:
square sizes are produced only by letterboxing the ENTIRE source
canvas (including its semi-transparent rounded corners) onto a
transparent square canvas. Content loss is a hard failure here.

These tests pin:
- the committed source art (structure, not hash — the art may be
  legitimately replaced),
- the letterbox = zero-crop property (pixel-exact),
- the generated PNG/ICO set (sizes, squareness, fully-opaque alpha,
  ICO entries) — by pixel-comparing the committed assets against a
  fresh render, not by re-deriving expected values,
- the runtime icon loader + AUMID ordering contract in main.py.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageChops

REPO = Path(__file__).parent.parent
ICON_DIR = REPO / "vaspen" / "resources" / "icons"
SOURCE = ICON_DIR / "vaspen_icon.png"


def _load_make_icon():
    """scripts/ is not a package — load make_icon.py by path so its
    pure functions are importable in tests."""
    spec = importlib.util.spec_from_file_location(
        "make_icon", REPO / "scripts" / "make_icon.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_icon = _load_make_icon()


def _ico_entries(path: Path) -> list[tuple[int, int]]:
    """(width, height) of every entry in an ICO file (width 0 in the
    header means 256, per the ICO spec)."""
    data = path.read_bytes()
    assert data[:4] == b"\x00\x00\x01\x00"
    count = struct.unpack_from("<H", data, 4)[0]
    sizes = []
    for i in range(count):
        w, h = data[6 + 16 * i], data[7 + 16 * i]
        sizes.append((256 if w == 0 else w, 256 if h == 0 else h))
    return sizes


def _ico_payload_magics(path: Path) -> list[bytes]:
    """First 4 bytes of every entry's payload (the encoding check)."""
    data = path.read_bytes()
    count = struct.unpack_from("<H", data, 4)[0]
    magics = []
    for i in range(count):
        entry = 6 + 16 * i
        payload_off = struct.unpack_from("<I", data, entry + 12)[0]
        magics.append(data[payload_off:payload_off + 4])
    return magics


def test_source_art_committed():
    """The source artwork lives in the repo (regeneration must never
    depend on a file outside it) and is structurally intact."""
    assert SOURCE.exists(), "source art missing — restore vaspen_icon.png"
    with Image.open(SOURCE) as im:
        assert im.size == (1536, 1024)
        assert im.mode == "RGBA"
        # Alpha bbox strictly inside the canvas: the art floats in a
        # transparent margin.
        bbox = im.getchannel("A").getbbox()
        assert bbox[0] > 0 and bbox[1] > 0
        assert bbox[2] < 1536 and bbox[3] < 1024


def test_letterbox_no_crop():
    """Letterboxing is a zero-crop operation: pixel-exact."""
    with Image.open(SOURCE) as src:
        src = src.convert("RGBA")
        canvas = make_icon.letterbox_square(src)
        assert canvas.size == (1536, 1536)
        dx, dy = (1536 - 1536) // 2, (1536 - 1024) // 2
        # The region the source was pasted into differs from the
        # source in NO pixel.
        assert ImageChops.difference(
            canvas.crop((dx, dy, dx + 1536, dy + 1024)), src
        ).getbbox() is None
        # The alpha bounding box shifts by exactly the paste offset.
        assert canvas.getchannel("A").getbbox() == tuple(
            b + d for b, d in zip(src.getchannel("A").getbbox(),
                                  (dx, dy, dx, dy)))


def test_render_icon_set_outputs(tmp_path):
    """The full set is written, all square, all exact sizes, with the
    alpha normalized to full opacity."""
    written = make_icon.render_icon_set(SOURCE, tmp_path)
    assert set(written) == {16, 24, 32, 48, 64, 128, 256}
    for size, path in written.items():
        with Image.open(path) as im:
            assert im.size == (size, size)
            assert im.mode == "RGBA"
    with Image.open(tmp_path / "app.png") as im:
        # Square letterboxed fallback — never the raw 3:2 art (Qt
        # would stretch it into the square icon slot).
        assert im.size == (256, 256)
    with Image.open(tmp_path / "app_256.png") as im:
        # The source PNG tops out at alpha 254; the render must be
        # fully opaque where the art is meant to be.
        assert im.getchannel("A").getextrema()[1] == 255
    assert (tmp_path / "app.ico").exists()


def test_committed_assets_match_fresh_render(tmp_path):
    """Every committed asset is pixel-identical to a fresh render
    from the source — no drift, no hand edits, no lost content."""
    make_icon.render_icon_set(SOURCE, tmp_path)
    names = ["app.png", *[f"app_{s}.png" for s in make_icon.PNG_SIZES]]
    for name in names:
        with Image.open(tmp_path / name) as fresh, \
                Image.open(ICON_DIR / name) as committed:
            diff = ImageChops.difference(
                fresh.convert("RGBA"), committed.convert("RGBA"))
            assert diff.getbbox() is None, name


def test_ico_entries():
    """The committed app.ico carries all 7 sizes as PNG entries."""
    assert set(_ico_entries(ICON_DIR / "app.ico")) == {
        (16, 16), (24, 24), (32, 32), (48, 48),
        (64, 64), (128, 128), (256, 256)}
    for magic in _ico_payload_magics(ICON_DIR / "app.ico"):
        assert magic == b"\x89PNG", magic


def test_load_app_icon_not_null(qapp):
    """The runtime loader finds the committed assets and carries all
    registered sizes."""
    from vaspen.main import _load_app_icon

    icon = _load_app_icon(ICON_DIR)
    assert not icon.isNull()
    sizes = {(s.width(), s.height()) for s in icon.availableSizes()}
    for size in (16, 24, 32, 48, 64, 128, 256):
        assert (size, size) in sizes, sizes


def test_aumid_constant():
    """The taskbar AppUserModelID is versionless — grouping stays
    stable across upgrades."""
    from vaspen.main import WINDOWS_APP_USER_MODEL_ID

    assert WINDOWS_APP_USER_MODEL_ID == "VASPen"


def test_icons_dir_dev_mode(monkeypatch):
    """Dev mode resolves icons next to the package, not via _MEIPASS."""
    from vaspen.main import _icons_dir

    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    assert _icons_dir() == ICON_DIR


def test_icons_dir_frozen(tmp_path):
    """Frozen (PyInstaller onedir) resolves via _MEIPASS + the package
    layout — the entry script is flattened to _internal/main.py while
    the data keeps vaspen/resources, so __file__-relative paths are
    wrong there. (The exe shipped with a NULL window icon until this
    was fixed.)"""
    from vaspen.main import _icons_dir

    setattr(sys, "_MEIPASS", str(tmp_path))  # monkeypatch needs it to exist
    try:
        assert _icons_dir() == tmp_path / "vaspen" / "resources" / "icons"
    finally:
        del sys._MEIPASS


def test_main_window_icon_smoke(qapp, qtbot):
    """Replicates main()'s icon order: app icon set before the window
    exists, window icon set explicitly — both non-null."""
    from vaspen.main import _load_app_icon
    from vaspen.ui.main_window import MainWindow

    icon = _load_app_icon(ICON_DIR)
    qapp.setWindowIcon(icon)
    window = MainWindow()
    qtbot.addWidget(window)
    window.setWindowIcon(qapp.windowIcon())
    assert not qapp.windowIcon().isNull()
    assert not window.windowIcon().isNull()


@pytest.mark.skipif(sys.platform != "win32",
                    reason="Windows-only shell32 API")
def test_set_windows_app_id_win32():
    from vaspen.main import _set_windows_app_id

    _set_windows_app_id()  # must not raise
