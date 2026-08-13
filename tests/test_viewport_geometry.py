"""Pure-geometry tests for the viewport's cylinder baker (no GL)."""

import numpy as np

from vaspen.ui.viewport3d import _cylinder_verts


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
