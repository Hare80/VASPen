"""Regenerate the VASPen icon set from the committed source artwork.

Usage:
    python scripts/make_icon.py [source.png]

Default source: vaspen/resources/icons/vaspen_icon.png (committed —
never depend on a file outside the repo).

SETTLED POLICY — do not re-litigate: the artwork is NEVER cropped and
NEVER zoomed. Square icons are produced by letterboxing the ENTIRE
source canvas — including its semi-transparent corners — centered on a
transparent square canvas. Content loss is a hard failure (pinned by
tests/test_icon_assets.py).
"""

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
ICON_DIR = ROOT / "vaspen" / "resources" / "icons"
SOURCE_NAME = "vaspen_icon.png"
#: 24 included for Windows small-icon contexts (e.g. taskbar, Alt-Tab)
PNG_SIZES = (16, 24, 32, 48, 64, 128, 256)
ICO_SIZES = PNG_SIZES


def letterbox_square(img: Image.Image) -> Image.Image:
    """Center the ENTIRE image on a transparent square canvas.

    No crop, no zoom — every pixel of the source (including the
    semi-transparent rounded corners) is preserved. A 3:2 source
    lands on a square canvas with transparent bands top/bottom.
    """
    img = img.convert("RGBA")
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def _normalize_alpha(img: Image.Image) -> Image.Image:
    """Stretch the alpha channel so the art tops out at 255.

    The source PNG has a slight global transparency (alpha max 254) —
    without this the icon never renders fully opaque. Only the alpha
    channel is scaled; spatial pixels and colors are untouched (this
    is NOT a crop). Returns the image unchanged if already normalized.
    """
    max_alpha = img.getchannel("A").getextrema()[1]
    if max_alpha <= 0 or max_alpha == 255:
        return img
    img = img.copy()
    alpha = img.getchannel("A").point(lambda p: p * 255 // max_alpha)
    img.putalpha(alpha)
    return img


def render_icon_set(source: Path, out_dir: Path) -> dict[int, Path]:
    """Write the multi-size PNG set + multi-size ICO from ``source``.

    Args:
        source: Path to the source artwork PNG.
        out_dir: Directory receiving app_{size}.png, app.png, app.ico.

    Returns:
        {size: written app_{size}.png path} — the sizes written.
    """
    img = _normalize_alpha(letterbox_square(Image.open(source)))
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, Path] = {}
    for size in PNG_SIZES:
        path = out_dir / f"app_{size}.png"
        img.resize((size, size), Image.LANCZOS).save(path)
        written[size] = path
    # Square fallback for the window icon — NEVER the raw 3:2 art
    # (Qt would stretch it into the square icon slot).
    img.resize((256, 256), Image.LANCZOS).save(out_dir / "app.png")
    # PNG-compressed ICO entries (Pillow default). PyInstaller
    # re-encodes whatever it is given into PNG entries for the exe
    # anyway (verified: BMP input produced a byte-identical exe icon),
    # so this is purely the size-optimal choice for the bundled file.
    img.resize((256, 256), Image.LANCZOS).save(
        out_dir / "app.ico",
        format="ICO",
        sizes=[(s, s) for s in ICO_SIZES],
    )
    return written


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        0 on success, 1 if the source image is missing.
    """
    args = sys.argv if argv is None else argv
    source = Path(args[1]) if len(args) > 1 else ICON_DIR / SOURCE_NAME
    if not source.exists():
        print(f"source image not found: {source}", file=sys.stderr)
        return 1
    written = render_icon_set(source, ICON_DIR)
    print(
        f"icons written to {ICON_DIR}: "
        + ", ".join(f"{size}px" for size in written)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
