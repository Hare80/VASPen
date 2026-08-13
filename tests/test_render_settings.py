"""RenderSettings: defaults, light direction, JSON roundtrip, palettes."""

import numpy as np
import pytest

from vaspen.core.render_settings import (
    BACKGROUND_DARK,
    BACKGROUND_LIGHT,
    FILL_DIR,
    PALETTES,
    RenderSettings,
    light_direction,
)
from vaspen.utils.config import AppConfig


# ----------------------------------------------------------------------
# Light direction
# ----------------------------------------------------------------------

def test_light_direction_reproduces_legacy_direction():
    """The default azimuth/elevation reproduces the pre-upgrade light."""
    d = np.asarray(light_direction(25.0, 33.6))
    legacy = np.array([0.35, 0.55, 0.75])
    legacy /= np.linalg.norm(legacy)
    assert np.allclose(d, legacy, atol=0.01)


def test_light_direction_is_unit():
    for az, el in [(0, 0), (45, 45), (-120, 60), (25, 33.6)]:
        d = np.asarray(light_direction(az, el))
        assert np.isclose(np.linalg.norm(d), 1.0)


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

def test_defaults():
    rs = RenderSettings.default()
    assert rs.background_color == BACKGROUND_LIGHT  # pure white default
    assert rs.cell_color == (0.0, 0.0, 0.0)         # black frame on white
    assert rs.style == "ball_stick"
    assert rs.show_cell is True
    assert rs.show_axes is True
    assert rs.headlight is True  # light follows the camera by default
    assert rs.gamma == pytest.approx(2.2)
    assert rs.ambient == pytest.approx(0.75)   # brightened for white bg
    assert rs.diffuse == pytest.approx(0.80)
    assert rs.atom_colors == {}


def test_label_size_fields_are_independent():
    rs = RenderSettings.default()
    assert rs.label_size == 12
    assert rs.corner_label_size == 12
    rs.label_size = 20
    rs2 = RenderSettings.from_dict(rs.to_dict())
    assert rs2.label_size == 20
    assert rs2.corner_label_size == 12


# ----------------------------------------------------------------------
# Dict roundtrip (persistence format)
# ----------------------------------------------------------------------

def test_dict_roundtrip():
    rs = RenderSettings.default()
    rs2 = RenderSettings.from_dict(rs.to_dict())
    assert rs2 == rs


def test_from_dict_ignores_unknown_and_fills_missing():
    rs = RenderSettings.from_dict({"ambient": 0.9, "not_a_field": 123})
    assert rs.ambient == pytest.approx(0.9)
    assert rs.gamma == pytest.approx(2.2)  # missing → default


def test_from_dict_invalid_values_fall_back():
    rs = RenderSettings.from_dict({
        "gamma": "banana",
        "style": "nonsense",
        "color_scheme": "crayons",
        "atom_colors": {"Fe": [1, 2]},  # too short
    })
    assert rs.gamma == pytest.approx(2.2)
    assert rs.style == "ball_stick"
    assert rs.color_scheme == "jmol"
    assert rs.atom_colors == {}


def test_atom_colors_survive_roundtrip():
    rs = RenderSettings.default()
    rs.atom_colors = {"Fe": [0.9, 0.1, 0.1, 0.5]}
    rs2 = RenderSettings.from_dict(rs.to_dict())
    assert rs2.atom_colors == {"Fe": [0.9, 0.1, 0.1, 0.5]}


# ----------------------------------------------------------------------
# AppConfig persistence
# ----------------------------------------------------------------------

def test_json_persistence_roundtrip():
    cfg = AppConfig()
    rs = RenderSettings.default()
    rs.background_color = BACKGROUND_LIGHT
    rs.gamma = 1.8
    rs.atom_colors = {"Fe": [1.0, 0.0, 0.0, 0.4]}
    cfg.render_settings = rs
    loaded = cfg.render_settings
    assert loaded == rs
    assert loaded.background_color == BACKGROUND_LIGHT
    assert loaded.gamma == pytest.approx(1.8)
    assert loaded.atom_colors == {"Fe": [1.0, 0.0, 0.0, 0.4]}


def test_corrupt_json_falls_back_to_defaults():
    cfg = AppConfig()
    cfg.set("render_settings", "{not json")
    assert cfg.render_settings == RenderSettings.default()


def test_fresh_config_gives_defaults():
    """A brand-new install (no stored settings) opens with the defaults —
    the user's expectation for first launch."""
    cfg = AppConfig()
    assert cfg.get("render_settings", "") == ""  # nothing stored
    assert cfg.render_settings == RenderSettings.default()
    assert cfg.render_settings.headlight is True


# ----------------------------------------------------------------------
# Palettes
# ----------------------------------------------------------------------

def test_metal_nonmetal_palette():
    p = PALETTES["metal_nonmetal"]
    assert p["Fe"] == p["Cu"] != p["O"]  # metals share, differ from non-metal
    assert p["O"] == p["C"]
    assert p["H"] != p["Na"]


def test_block_palette_assigns_blocks():
    p = PALETTES["block"]
    assert p["Fe"] == p["Ni"] == p["Pt"]  # d block
    assert p["H"] == p["Na"]              # s block
    assert p["C"] == p["O"]               # p block
    assert p["La"] == p["Ce"]             # f block
    assert p["Fe"] != p["C"] != p["La"]


def test_fill_dir_is_unit():
    assert np.isclose(np.linalg.norm(FILL_DIR), 1.0)


def test_specular_and_cell_defaults():
    rs = RenderSettings.default()
    assert rs.specular_enabled is True
    assert rs.cell_color == (0.0, 0.0, 0.0)
    assert rs.cell_line_width == pytest.approx(1.0)
    # removed fields must not exist anymore (round-3 decisions)
    assert not hasattr(rs, "metal_specular")
    assert not hasattr(rs, "show_outlines")
    assert not hasattr(rs, "depth_cue")
    assert not hasattr(rs, "depth_cue_strength")


def test_specular_and_cell_roundtrip():
    rs = RenderSettings.default()
    rs.specular_enabled = False
    rs.cell_color = (1.0, 0.2, 0.2)
    rs.cell_line_width = 3.0
    rs2 = RenderSettings.from_dict(rs.to_dict())
    assert rs2.specular_enabled is False
    assert rs2.cell_color == (1.0, 0.2, 0.2)
    assert rs2.cell_line_width == pytest.approx(3.0)


def test_atom_opacity_and_label_defaults():
    rs = RenderSettings.default()
    assert rs.atom_opacity == pytest.approx(1.0)
    assert rs.label_size == 12
    assert rs.show_cell_corners is False  # renamed from show_cell_labels


def test_atom_opacity_roundtrip():
    rs = RenderSettings.default()
    rs.atom_opacity = 0.35
    rs.label_size = 18
    rs.show_cell_corners = True
    rs2 = RenderSettings.from_dict(rs.to_dict())
    assert rs2.atom_opacity == pytest.approx(0.35)
    assert rs2.label_size == 18
    assert rs2.show_cell_corners is True
