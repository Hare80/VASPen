"""Render settings for the 3D viewport (pure Python — no Qt).

RenderSettings is the single source of truth for everything the
Viewport3D renders except the scene geometry itself: background,
lighting, atom/bond appearance, color schemes, overlays, effects.
The View → Display Options dialog edits one of these live; AppConfig
persists it as a JSON string (key "render_settings").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

from ase.data import chemical_symbols

# ----------------------------------------------------------------------
# Background presets
# ----------------------------------------------------------------------

BACKGROUND_DARK = (0.118, 0.118, 0.141)    # #1e1e24
BACKGROUND_LIGHT = (1.0, 1.0, 1.0)         # pure white (default)
GRADIENT_DARK_TOP = (0.20, 0.20, 0.24)
GRADIENT_DARK_BOTTOM = (0.08, 0.08, 0.10)
GRADIENT_LIGHT_TOP = (0.75, 0.80, 0.86)
GRADIENT_LIGHT_BOTTOM = (0.98, 0.99, 1.00)

# Fill light direction (world, fixed constant — not exposed in the UI)
_FILL = (0.6, -0.25, -0.7)
FILL_DIR = tuple(
    c / math.sqrt(sum(v * v for v in _FILL)) for c in _FILL
)


def light_direction(
    azimuth_deg: float,
    elevation_deg: float,
) -> tuple[float, float, float]:
    """Unit vector of the key light from azimuth/elevation (degrees).

    Uses the camera convention from viewport3d.py ``_camera_matrices``:
    azimuth about +y, elevation above the xz plane — direction =
    (cos(el)·sin(az), sin(el), cos(el)·cos(az)). The pre-upgrade light
    (0.35, 0.55, 0.75) is exactly azimuth 25.0° / elevation 33.6°.
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    return (
        math.cos(el) * math.sin(az),
        math.sin(el),
        math.cos(el) * math.cos(az),
    )


# ----------------------------------------------------------------------
# Color schemes and element tables
# ----------------------------------------------------------------------

# Metals for the metal/nonmetal scheme and per-element specular strength.
_METALS = frozenset({
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs",
    "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
    "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
})

_SYMBOLS = [s for s in chemical_symbols if s != "X"]

METAL_COLOR = (0.62, 0.70, 0.78)       # silver-blue
NONMETAL_COLOR = (0.40, 0.65, 0.40)    # green


def _build_block_palette() -> dict[str, tuple[float, float, float]]:
    """Periodic-table block palette: s/p/d/f block colors."""
    s_color = (0.95, 0.60, 0.60)
    p_color = (0.55, 0.75, 0.60)
    d_color = (0.55, 0.65, 0.90)
    f_color = (0.85, 0.70, 0.55)
    table: dict[str, tuple[float, float, float]] = {}
    for z, sym in enumerate(chemical_symbols):
        if sym == "X":
            continue
        if 57 <= z <= 71 or 89 <= z <= 103:
            table[sym] = f_color
        elif 21 <= z <= 30 or 39 <= z <= 48 or 72 <= z <= 80 or 104 <= z <= 112:
            table[sym] = d_color
        elif z in (1, 2, 3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88):
            table[sym] = s_color
        else:
            table[sym] = p_color
    return table


# Palette lookup order: per-element override > PALETTES[color_scheme] >
# Jmol (viewport3d.JMOL_COLORS). "jmol" is an empty table (pure fallback).
PALETTES: dict[str, dict[str, tuple[float, float, float]]] = {
    "jmol": {},
    "metal_nonmetal": {
        sym: (METAL_COLOR if sym in _METALS else NONMETAL_COLOR)
        for sym in _SYMBOLS
    },
    "block": _build_block_palette(),
}

STYLES = ("ball_stick", "cpk", "wireframe")


# ----------------------------------------------------------------------
# RenderSettings
# ----------------------------------------------------------------------

@dataclass
class RenderSettings:
    """All adjustable render parameters for the 3D viewport.

    Colors are RGB(A) tuples in 0-1 (display-referred sRGB). Persisted
    as JSON via to_dict()/from_dict() — unknown keys are ignored and
    missing keys default, so older/newer stored JSON never breaks.
    """

    background_color: tuple[float, float, float] = BACKGROUND_LIGHT
    background_gradient: bool = False
    gradient_top: tuple[float, float, float] = GRADIENT_DARK_TOP
    gradient_bottom: tuple[float, float, float] = GRADIENT_DARK_BOTTOM

    light_azimuth: float = 25.0
    light_elevation: float = 33.6
    headlight: bool = True  # light follows the camera (user decision 2026-08-13)
    # Brightened for the white background (user decision 2026-08-13):
    # ambient lifts the shadow side, diffuse the lit side. The sliders
    # go up to 2.0 so users can over-brighten if desired.
    ambient: float = 0.75
    ambient_sky: tuple[float, float, float] = (0.62, 0.64, 0.68)
    ambient_ground: tuple[float, float, float] = (0.42, 0.40, 0.38)
    diffuse: float = 0.80
    specular: float = 0.30
    shininess: float = 32.0
    # Specular highlights apply to ALL atoms uniformly (settled
    # 2026-08-13: the old metals-only distinction was removed).
    specular_enabled: bool = True
    fill_intensity: float = 0.15
    gamma: float = 2.2

    sphere_scale: float = 0.60
    bond_radius: float = 0.12
    style: str = "ball_stick"
    show_cell: bool = True
    cell_color: tuple[float, float, float] = (0.0, 0.0, 0.0)  # black frame on white
    cell_line_width: float = 1.0
    show_labels: bool = False
    bonds_by_element: bool = False
    # Global atom opacity (0-1, 0 = fully transparent). Multiplies the
    # per-element override alphas; bonds fade together with the atoms.
    atom_opacity: float = 1.0

    color_scheme: str = "jmol"
    atom_colors: dict[str, list[float]] = field(default_factory=dict)

    show_axes: bool = True
    show_cell_corners: bool = False  # O/A/B/C labels at the cell corners
    label_size: int = 12             # element labels
    corner_label_size: int = 12      # cell-corner labels (independent)

    @classmethod
    def default(cls) -> "RenderSettings":
        """A settings object with all default values."""
        return cls()

    def to_dict(self) -> dict:
        """JSON-serializable form (tuples → lists)."""
        out: dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, tuple):
                value = list(value)
            out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: dict | None) -> "RenderSettings":
        """Build from a persisted dict; unknown keys ignored, missing
        keys and unparsable values fall back to defaults."""
        rs = cls.default()
        if not data:
            return rs
        for f in fields(cls):
            if f.name == "atom_colors":
                continue  # handled separately below
            raw = data.get(f.name)
            if raw is None:
                continue
            current = getattr(rs, f.name)
            try:
                if isinstance(current, tuple):
                    value = tuple(float(v) for v in raw)
                elif isinstance(current, bool):
                    value = str(raw).lower() in ("true", "1")
                elif isinstance(current, int):
                    value = int(raw)
                elif isinstance(current, str):
                    value = str(raw)
                else:
                    value = float(raw)
            except (TypeError, ValueError):
                continue
            setattr(rs, f.name, value)
        rs.style = rs.style if rs.style in STYLES else "ball_stick"
        rs.color_scheme = (
            rs.color_scheme if rs.color_scheme in PALETTES else "jmol"
        )
        overrides = data.get("atom_colors")
        if isinstance(overrides, dict):
            cleaned: dict[str, list[float]] = {}
            for sym, rgba in overrides.items():
                try:
                    rgba4 = list(rgba)[:4]
                    if len(rgba4) < 4:
                        continue
                    cleaned[str(sym)] = [
                        max(0.0, min(1.0, float(c))) for c in rgba4
                    ]
                except (TypeError, ValueError):
                    continue
            rs.atom_colors = cleaned
        return rs
