"""Regenerate the VASPen icon set from a source image.

Usage:
    python scripts/make_icon.py <source.png>

Crops away fully transparent borders only (per user requirement — no
visible content is cropped), pads the result to a square canvas with
transparent margins, and writes the multi-size PNG set + multi-size
ICO into vaspen/resources/icons/.
"""

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
ICON_DIR = ROOT / "vaspen" / "resources" / "icons"
PNG_SIZES = [16, 32, 48, 64, 128, 256]
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    src = Path(sys.argv[1])
    img = Image.open(src).convert("RGBA")

    # Crop transparent borders only — keep all visible content
    bbox = img.getchannel("A").getbbox()
    if bbox is None:
        print("Image is fully transparent")
        return 1
    img = img.crop(bbox)

    # Pad to square with transparent margins (icons must be square)
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))

    ICON_DIR.mkdir(parents=True, exist_ok=True)
    for size in PNG_SIZES:
        canvas.resize((size, size), Image.LANCZOS).save(ICON_DIR / f"app_{size}.png")
    canvas.resize((256, 256), Image.LANCZOS).save(ICON_DIR / "app.png")
    canvas.resize((256, 256), Image.LANCZOS).save(
        ICON_DIR / "app.ico", sizes=[(s, s) for s in ICO_SIZES]
    )
    print(f"icons written to {ICON_DIR} (crop {bbox}, square {side}x{side})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
