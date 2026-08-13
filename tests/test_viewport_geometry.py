"""Pure-geometry tests for the viewport's cylinder baker (no GL)."""

import numpy as np
import pytest

from vaspen.ui.viewport3d import (
    _center_fade,
    _cylinder_verts,
    _effective_alphas,
    _look_at,
    _split_opaque_transparent,
    _view_depths,
)


def test_cylinder_vertex_layout_is_pos_normal_color():
    """Each vertex is 9 floats: pos(3) + normal(3) + color(3)."""
    verts = _cylinder_verts(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        radius=0.1,
    )
    assert verts.shape[1] == 9
    n_segments = 12
    assert len(verts) == n_segments * 2 * 3  # 2 triangles × 3 verts × 12


def test_cylinder_normals_point_radially_outward():
    """Normals point away from the axis and are unit length."""
    s = np.array([0.0, 0.0, 0.0])
    e = np.array([0.0, 0.0, 1.0])
    verts = _cylinder_verts(np.array([s]), np.array([e]), radius=0.25)
    for v in verts:
        pos, nrm = v[:3], v[3:6]
        radial = pos - np.array([0.0, 0.0, pos[2]])  # away from the axis
        assert np.isclose(np.linalg.norm(nrm), 1.0, atol=1e-6)
        assert float(nrm @ radial) > 0.0  # outward, never inward


def test_cylinder_winding_is_outward():
    """Front faces point away from the axis (no gl_FrontFacing reliance).

    Cross product of the first triangle's edges must point in the same
    hemisphere as the radial direction of its vertices.
    """
    s = np.array([0.0, 0.0, 0.0])
    e = np.array([0.0, 0.0, 1.0])
    verts = _cylinder_verts(np.array([s]), np.array([e]), radius=0.25)
    for k in range(0, len(verts), 3):
        a, b, c = verts[k, :3], verts[k + 1, :3], verts[k + 2, :3]
        face_normal = np.cross(b - a, c - a)
        mid = (a + b + c) / 3.0
        radial = mid - np.array([0.0, 0.0, mid[2]])
        assert float(face_normal @ radial) > 0.0


def test_cylinder_colors_split_ends():
    """color_i colors the start ring, color_j the end ring."""
    verts = _cylinder_verts(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        radius=0.1,
        color_i=(1.0, 0.0, 0.0),
        color_j=(0.0, 0.0, 1.0),
    )
    for v in verts:
        z = v[2]
        if np.isclose(z, 0.0):
            assert np.allclose(v[6:9], [1.0, 0.0, 0.0])
        else:
            assert np.allclose(v[6:9], [0.0, 0.0, 1.0])


def test_cylinder_lateral_offset_keeps_normals_radial():
    """The offset is a translation along u — normals stay axis-radial."""
    base = _cylinder_verts(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        radius=0.1,
    )
    offset = _cylinder_verts(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        radius=0.1,
        lateral_offset=0.12,
    )
    assert base.shape == offset.shape
    # the normal field is identical (pure translation)
    assert np.allclose(base[:, 3:6], offset[:, 3:6])


def test_cylinder_zero_length_returns_empty():
    verts = _cylinder_verts(
        np.array([[1.0, 1.0, 1.0]]),
        np.array([[1.0, 1.0, 1.0]]),
        radius=0.1,
    )
    assert verts.shape == (0, 9)


# ----------------------------------------------------------------------
# Transparency pass classification helpers (pure numpy, no GL)
# ----------------------------------------------------------------------

def test_view_depths_positive_in_front():
    """Depth = distance along the camera view axis (matches the shader's
    vViewDepth = -viewPos.z convention)."""
    view = _look_at(np.array([0.0, 0.0, 10.0]), np.zeros(3),
                    np.array([0.0, 1.0, 0.0]))
    depths = _view_depths(view, np.array([[0.0, 0.0, 0.0],
                                          [0.0, 0.0, 5.0]]))
    assert depths == pytest.approx([10.0, 5.0])


def test_center_fade_matches_glsl_smoothstep():
    assert _center_fade(np.array([2.0]))[0] == pytest.approx(1.0)
    assert _center_fade(np.array([0.25]))[0] == pytest.approx(0.0)
    # midpoint of the fade band (0.25, 0.8) → smoothstep = 0.5
    assert _center_fade(np.array([0.525]))[0] == pytest.approx(0.5)
    f = _center_fade(np.array([0.3, 0.4, 0.5]))
    assert (np.diff(f) > 0).all()  # monotonic


def test_effective_alpha_composition():
    """override alpha × global opacity × center fade, clamped 0-1."""
    eff = _effective_alphas(np.array([1.0, 0.5]), 0.5,
                            np.array([2.0, 2.0]))
    assert eff == pytest.approx([0.5, 0.25])
    eff = _effective_alphas(np.array([1.0]), 1.5, np.array([2.0]))
    assert eff[0] == pytest.approx(1.0)  # clamped


def test_split_opaque_transparent_sorts_far_to_near():
    depths = np.array([1.0, 3.0, 2.0, 5.0])
    eff = np.array([0.5, 1.0, 0.25, 1.0])
    opaque, trans = _split_opaque_transparent(depths, eff)
    assert opaque.tolist() == [1, 3]   # index order preserved
    assert trans.tolist() == [2, 0]    # depth 2.0 before 1.0


def test_split_all_opaque_returns_empty_transparent():
    """The default scene (opacity 1, no overrides, normal distance)
    stays on the old single-pass code path."""
    depths = np.array([1.0, 3.0, 2.0])
    eff = np.array([1.0, 1.0, 1.0])
    opaque, trans = _split_opaque_transparent(depths, eff)
    assert opaque.tolist() == [0, 1, 2]
    assert len(trans) == 0


def test_split_global_opacity_zero_all_transparent():
    depths = np.array([1.0, 3.0, 2.0])
    eff = _effective_alphas(np.ones(3), 0.0, depths)
    opaque, trans = _split_opaque_transparent(depths, eff)
    assert len(opaque) == 0
    assert trans.tolist() == [1, 2, 0]  # far → near
