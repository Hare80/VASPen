"""Regenerate the VASPen icon set from a source image.

Usage:
    python scripts/make_icon.py <source.png> [--zoom Z]

Crops away fully transparent borders, then zooms into the central
square region by factor Z (default 1 = no zoom, the whole content
fits the icon with transparent padding to make it square; Z=3 shows
the middle third). Writes the multi-size PNG set + multi-size ICO
into vaspen/resources/icons/.
"""

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
ICON_DIR = ROOT / "vaspen" / "resources" / "icons"
PNG_SIZES = [16, 32, 48, 64, 128, 256]
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]


def main() -> int:
    args = [a for a in sys.argv[1:]]
    zoom = 1.0
    if "--zoom" in args:
        i = args.index("--zoom")
        zoom = float(args.pop(i + 1))
        args.pop(i)
    if len(args) != 1:
        print(__doc__)
        return 1
    src = Path(args[0])
    img = Image.open(src).convert("RGBA")

    # Crop transparent borders only — keep all visible content
    bbox = img.getchannel("A").getbbox()
    if bbox is None:
        print("Image is fully transparent")
        return 1
    img = img.crop(bbox)

    # Zoom: keep only the central square of side min(w,h)/zoom
    w, h = img.size
    side = round(min(w, h) / zoom)
    left = (w - side) // 2
    top = (h - side) // 2
    canvas = img.crop((left, top, left + side, top + side))

    ICON_DIR.mkdir(parents=True, exist_ok=True)
    for size in PNG_SIZES:
        canvas.resize((size, size), Image.LANCZOS).save(ICON_DIR / f"app_{size}.png")
    canvas.resize((256, 256), Image.LANCZOS).save(ICON_DIR / "app.png")
    canvas.resize((256, 256), Image.LANCZOS).save(
        ICON_DIR / "app.ico", sizes=[(s, s) for s in ICO_SIZES]
    )
    print(f"icons written to {ICON_DIR} (crop {bbox}, zoom {zoom}, "
          f"region {side}x{side} at ({left},{top}))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
