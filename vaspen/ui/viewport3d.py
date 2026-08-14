"""3D structure viewport — hand-written Qt native OpenGL ball-and-stick.

Features:
- Ball-and-stick model with Jmol element colors and dark edge outlines
- Unit cell outline (12 edges, only when periodic)
- Mouse: left-drag rotate, right/middle-drag pan, wheel zoom,
  left-click (no drag) picks an atom
- Picking: ID-color framebuffer readback — every atom is temporarily
  rendered in a unique color into an offscreen FBO and the pixel under
  the cursor is read back. This bypasses all coordinate transforms, so
  picking always matches exactly what is visible on screen (including
  occlusion and same-element atoms).
- Signal-driven: set_structure() swaps the scene data; the widget
  re-renders on demand.

Design note: camera, projection and picking share ONE set of matrices
computed by this class — there is no library-side transform chain that
could drift out of sync with the rendered image (the vispy failure mode
that motivated this rewrite).
"""

from __future__ import annotations

import math

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii

from vaspen.core.bonds import find_bonds, mic_vector
from vaspen.core.transform import rotation_matrix
from vaspen.core.render_settings import (
    FILL_DIR,
    PALETTES,
    RenderSettings,
    light_direction,
)
from vaspen.utils.logger import logger

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QMatrix4x4,
    QPainter,
    QPainterPath,
    QPen,
    QSurfaceFormat,
    QVector3D,
)
from vaspen.ui.tools import ToolMode, make_tool
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from shiboken6 import VoidPtr

# ----------------------------------------------------------------------
# GL constants (fixed by the OpenGL spec — defined here instead of
# importing PySide6's enum wrappers, which vary between Qt versions).
# ----------------------------------------------------------------------

GL_DEPTH_TEST = 0x0B71
GL_BLEND = 0x0BE2
GL_DEPTH_CLAMP = 0x864F  # core in GL 3.3
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_COLOR_BUFFER_BIT = 0x4000
GL_DEPTH_BUFFER_BIT = 0x0100
GL_TRIANGLES = 0x0004
GL_LINES = 0x0001
GL_FLOAT = 0x1406
GL_UNSIGNED_INT = 0x1405
GL_UNSIGNED_BYTE = 0x1401
GL_ARRAY_BUFFER = 0x8892
GL_ELEMENT_ARRAY_BUFFER = 0x8893
GL_STATIC_DRAW = 0x88E4
GL_RGBA = 0x1908

# Request a modern core context with MSAA — must run before any
# QOpenGLWidget is created (this module is imported before QApplication
# exists in main.py).
_FMT = QSurfaceFormat()
_FMT.setVersion(3, 3)
_FMT.setProfile(QSurfaceFormat.CoreProfile)
_FMT.setSamples(4)          # MSAA 4x for smooth sphere edges
_FMT.setDepthBufferSize(24)
QSurfaceFormat.setDefaultFormat(_FMT)

# ----------------------------------------------------------------------
# Jmol element color scheme (RGB, 0-1). ~90 elements + default gray.
# ----------------------------------------------------------------------

JMOL_COLORS: dict[str, tuple[float, float, float]] = {
    "H":  (1.000, 1.000, 1.000),
    "He": (0.851, 1.000, 1.000),
    "Li": (0.800, 0.502, 1.000),
    "Be": (0.761, 1.000, 0.000),
    "B":  (1.000, 0.710, 0.710),
    "C":  (0.565, 0.565, 0.565),
    "N":  (0.188, 0.314, 0.973),
    "O":  (1.000, 0.051, 0.051),
    "F":  (0.565, 0.878, 0.314),
    "Ne": (0.702, 0.890, 0.961),
    "Na": (0.671, 0.361, 0.949),
    "Mg": (0.541, 1.000, 0.000),
    "Al": (0.749, 0.651, 0.651),
    "Si": (0.941, 0.784, 0.627),
    "P":  (1.000, 0.502, 0.000),
    "S":  (1.000, 1.000, 0.188),
    "Cl": (0.122, 0.941, 0.122),
    "Ar": (0.502, 0.820, 0.890),
    "K":  (0.561, 0.251, 0.831),
    "Ca": (0.239, 1.000, 0.000),
    "Ti": (0.749, 0.761, 0.780),
    "V":  (0.651, 0.651, 0.671),
    "Cr": (0.541, 0.600, 0.780),
    "Mn": (0.611, 0.478, 0.780),
    "Fe": (0.878, 0.400, 0.200),
    "Co": (0.941, 0.565, 0.627),
    "Ni": (0.314, 0.816, 0.314),
    "Cu": (0.784, 0.502, 0.200),
    "Zn": (0.490, 0.502, 0.690),
    "Ga": (0.761, 0.561, 0.561),
    "Ge": (0.400, 0.561, 0.561),
    "As": (0.741, 0.502, 0.890),
    "Se": (1.000, 0.631, 0.000),
    "Br": (0.651, 0.161, 0.161),
    "Kr": (0.361, 0.722, 0.820),
    "Rb": (0.439, 0.180, 0.690),
    "Sr": (0.000, 1.000, 0.000),
    "Y":  (0.580, 1.000, 1.000),
    "Zr": (0.580, 0.878, 0.878),
    "Nb": (0.451, 0.761, 0.788),
    "Mo": (0.329, 0.710, 0.710),
    "Tc": (0.231, 0.620, 0.620),
    "Ru": (0.141, 0.561, 0.561),
    "Rh": (0.039, 0.490, 0.549),
    "Pd": (0.000, 0.412, 0.522),
    "Ag": (0.753, 0.753, 0.753),
    "Cd": (1.000, 0.851, 0.561),
    "In": (0.651, 0.459, 0.451),
    "Sn": (0.400, 0.502, 0.502),
    "Sb": (0.620, 0.388, 0.710),
    "Te": (0.831, 0.478, 0.000),
    "I":  (0.580, 0.000, 0.580),
    "Xe": (0.259, 0.620, 0.690),
    "Cs": (0.341, 0.090, 0.561),
    "Ba": (0.000, 0.788, 0.000),
    "La": (0.439, 0.831, 1.000),
    "Ce": (1.000, 1.000, 0.780),
    "Pr": (0.851, 1.000, 0.780),
    "Nd": (0.780, 1.000, 0.780),
    "Sm": (0.561, 1.000, 0.780),
    "Eu": (0.380, 1.000, 0.780),
    "Gd": (0.271, 1.000, 0.780),
    "Tb": (0.190, 1.000, 0.780),
    "Dy": (0.122, 1.000, 0.780),
    "Ho": (0.000, 1.000, 0.612),
    "Er": (0.000, 0.902, 0.459),
    "Tm": (0.000, 0.831, 0.322),
    "Yb": (0.000, 0.749, 0.220),
    "Lu": (0.000, 0.671, 0.141),
    "Hf": (0.302, 0.761, 1.000),
    "Ta": (0.302, 0.651, 1.000),
    "W":  (0.129, 0.580, 0.839),
    "Re": (0.149, 0.490, 0.671),
    "Os": (0.149, 0.400, 0.588),
    "Ir": (0.090, 0.329, 0.529),
    "Pt": (0.816, 0.816, 0.878),
    "Au": (1.000, 0.820, 0.137),
    "Hg": (0.722, 0.722, 0.816),
    "Tl": (0.651, 0.329, 0.302),
    "Pb": (0.341, 0.349, 0.380),
    "Bi": (0.620, 0.310, 0.710),
    "Po": (0.671, 0.361, 0.000),
}

DEFAULT_ATOM_COLOR = (0.560, 0.560, 0.560)  # gray for unknown elements

# Highlight style for selected atom (color only — same size, per user
# preference: "点击高亮时候不要放大")
HIGHLIGHT_COLOR = (1.0, 0.85, 0.0)  # amber/yellow

# Near-distance fade (view-space Angstrom): atoms closer than
# NEAR_FADE_START begin fading and are fully transparent at NEAR_FADE_END,
# so the camera can zoom into a structure without seeing sphere
# cross-sections.
NEAR_FADE_START = 0.8
NEAR_FADE_END = 0.25

# Ball-and-stick scale factors — the ADJUSTABLE defaults (sphere scale,
# bond radius, background) live in RenderSettings (core/render_settings.py).
SPHERE_SCALE = 0.60   # default atom sphere radius = covalent radius × 0.60
EDGE_SCALE = 1.04     # dark outline sphere drawn slightly larger behind
BOND_RADIUS = 0.12    # default bond cylinder radius (Angstrom)

# Frozen atoms restore the old per-atom outline style: an edge sphere at
# EDGE_SCALE colored with _edge_color(body color) — the ORIGINAL outline
# look (user decision 2026-08-14, red was rejected). EDGE_DARKEN controls
# how dark the rim is (0.45 = the historical value; the user picked 0.60
# from generated Cu previews on 2026-08-14).
EDGE_DARKEN = 0.60

BACKGROUND_COLOR = (0.118, 0.118, 0.141)  # default background #1e1e24 dark

# Mouse-drag threshold: movement below this is a click, not a drag
CLICK_DRAG_PX = 4.0


def element_color(symbol: str) -> tuple[float, float, float]:
    """Return the Jmol RGB color (0-1) for an element symbol."""
    return JMOL_COLORS.get(symbol, DEFAULT_ATOM_COLOR)


def _edge_color(color: tuple[float, float, float]) -> tuple[float, float, float]:
    """Derive the darker outline color from a base atom color.

    The original renderer's edge style (user decision 2026-08-14:
    frozen atoms restore it with the ORIGINAL color, not red). The
    factor reads the module-level EDGE_DARKEN at draw time so preview
    scripts can sweep it without touching the render code.
    """
    return tuple(c * EDGE_DARKEN for c in color)


def element_text_color(symbol: str) -> tuple[float, float, float]:
    """Element color darkened for TEXT on light panels.

    The Jmol scheme has many near-white entries (H is pure white) that
    are invisible as foreground text on a light table background. Scale
    light colors down (preserving hue) so every element label has
    contrast; dark colors pass through unchanged.
    """
    base = element_color(symbol)
    luminance = 0.299 * base[0] + 0.587 * base[1] + 0.114 * base[2]
    if luminance > 0.62:
        factor = 0.62 / luminance
        base = tuple(c * factor for c in base)
    return base


def _cell_edges(cell: np.ndarray) -> np.ndarray:
    """Return the 12 edges of the unit cell as line segments (24×3)."""
    a, b, c = cell
    corners = np.array([
        [0, 0, 0], a, b, c, a + b, a + c, b + c, a + b + c
    ])
    edge_pairs = [
        (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4),
        (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
    ]
    lines = np.array(
        [corners[i] for pair in edge_pairs for i in pair],
        dtype=np.float32,
    )
    return lines


# ----------------------------------------------------------------------
# Geometry generation (pure numpy — no GL)
# ----------------------------------------------------------------------

def _unit_icosphere(subdivisions: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Unit icosphere centered at origin: (vertices N×3, faces M×3 uint32).

    Vertices are normalized, so positions double as normals.
    subdivisions=3 gives 642 vertices / 1280 faces — smooth enough for
    ball rendering with per-vertex normals and MSAA.
    """
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    for _ in range(subdivisions):
        new_verts = [v for v in verts]
        midpoint_cache: dict[tuple[int, int], int] = {}
        new_faces: list[list[int]] = []

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key not in midpoint_cache:
                m = (verts[a] + verts[b]) / 2.0
                m /= np.linalg.norm(m)
                new_verts.append(m)
                midpoint_cache[key] = len(new_verts) - 1
            return midpoint_cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces.extend([
                [a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca],
            ])
        verts = np.asarray(new_verts, dtype=np.float64)
        faces = new_faces

    return verts.astype(np.float32), np.asarray(faces, dtype=np.uint32)


def _cylinder_verts(
    starts: np.ndarray,
    ends: np.ndarray,
    radius: float,
    segments: int = 12,
    lateral_offset: float = 0.0,
    color_i: tuple[float, float, float] | None = None,
    color_j: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Bake bond cylinders into world-space triangle vertices (M×9).

    Each vertex is pos(3) + radial normal(3) + color(3), so cylinders
    are lit by the sphere shader like the atoms (previously unlit).
    Winding is OUTWARD and normals point radially away from the axis
    (the lateral_offset is a translation along u, so the radial normal
    is offset-independent). When color_i/color_j are given the i-end
    half of the cylinder is colored with color_i and the j-end half
    with color_j (bonds-by-element rendering); the per-vertex color is
    unused (zero) otherwise.

    Args:
        starts: Segment start points (K×3) — atom positions.
        ends: Segment end points (K×3) — atom + minimum-image vector.
        radius: Cylinder radius in Angstrom.
        lateral_offset: Shift the whole cylinder perpendicular to its
            axis (used to draw double/triple bond components side by
            side).

    Returns:
        Flat vertex array for glDrawArrays(GL_TRIANGLES). Ends are
        open (buried inside the atom spheres, so the joint is
        seamless).
    """
    verts: list[np.ndarray] = []
    for s, e in zip(starts, ends):
        axis = e - s
        length = float(np.linalg.norm(axis))
        if length < 1e-9:
            continue
        axis = axis / length
        # Orthonormal basis perpendicular to the axis
        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(axis @ ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(axis, ref)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        if lateral_offset:
            s = s + u * lateral_offset
            e = e + u * lateral_offset

        angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        ring_s = s + radius * (cos_a[:, None] * u + sin_a[:, None] * v)
        ring_e = e + radius * (cos_a[:, None] * u + sin_a[:, None] * v)
        # Radial normals (point from the cylinder axis to the ring vertex)
        normals = cos_a[:, None] * u + sin_a[:, None] * v
        ci = np.asarray(color_i, dtype=np.float32) if color_i is not None \
            else np.zeros(3, dtype=np.float32)
        cj = np.asarray(color_j, dtype=np.float32) if color_j is not None \
            else np.zeros(3, dtype=np.float32)
        for k in range(segments):
            k2 = (k + 1) % segments
            a0, a1 = ring_s[k], ring_s[k2]
            b0, b1 = ring_e[k], ring_e[k2]
            n0, n1 = normals[k], normals[k2]
            # OUTWARD winding [a0, b1, b0, a0, a1, b1] — face normals
            # point away from the axis, consistent with the sphere
            # geometry (no reliance on the gl_FrontFacing flip).
            verts.append(np.hstack([a0, n0, ci]))
            verts.append(np.hstack([b1, n1, cj]))
            verts.append(np.hstack([b0, n0, cj]))
            verts.append(np.hstack([a0, n0, ci]))
            verts.append(np.hstack([a1, n1, ci]))
            verts.append(np.hstack([b1, n1, cj]))
    if not verts:
        return np.zeros((0, 9), dtype=np.float32)
    return np.asarray(verts, dtype=np.float32)


def _look_at(
    eye: np.ndarray,
    center: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """View matrix (world → view space, GL convention)."""
    f = center - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    return np.array([
        [s[0], s[1], s[2], -float(s @ eye)],
        [u[0], u[1], u[2], -float(u @ eye)],
        [-f[0], -f[1], -f[2], float(f @ eye)],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def _ortho(
    left: float,
    right: float,
    bottom: float,
    top: float,
    near: float,
    far: float,
) -> np.ndarray:
    """Orthographic projection matrix (GL clip space).

    Args:
        near: Distance from the camera to the near plane (positive).
        far: Distance from the camera to the far plane (positive, > near).
    """
    return np.array([
        [2 / (right - left), 0, 0, -(right + left) / (right - left)],
        [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
        [0, 0, -2 / (far - near), -(far + near) / (far - near)],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def _sphere_model(pos: np.ndarray, scale: float) -> np.ndarray:
    """Model matrix: translate(pos) · scale(scale) for the unit sphere."""
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = m[1, 1] = m[2, 2] = scale
    m[:3, 3] = pos
    return m


def _to_qmatrix(m: np.ndarray) -> QMatrix4x4:
    """Convert a numpy 4×4 (C-order row-major) to QMatrix4x4.

    The QMatrix4x4 16-float constructor takes values in row-major
    order, matching numpy's C-order flatten directly (no transpose).
    """
    return QMatrix4x4(*np.ascontiguousarray(m, dtype=np.float64).reshape(16).tolist())


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """GLSL smoothstep for numpy arrays (edge0 < edge1)."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _view_depths(view: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """View-space depth (positive in front of the camera) for N world
    positions.

    Matches the shader's ``vViewDepth = -viewPos.z`` convention: the
    third row of the ``_look_at`` view matrix maps world → view z, so
    points in front of the camera come out negative and are negated.
    """
    z = positions @ view[2, :3] + view[2, 3]
    return -z


def _center_fade(depths: np.ndarray) -> np.ndarray:
    """Near-camera fade evaluated at the sphere CENTER view depth.

    Mirrors the shader's smoothstep(NEAR_FADE_END, NEAR_FADE_START,
    vViewDepth). Used ONLY for opaque/transparent pass classification —
    the shader still applies the true per-fragment fade.
    """
    return _smoothstep(NEAR_FADE_END, NEAR_FADE_START, depths)


def _effective_alphas(
    alphas: np.ndarray,
    global_opacity: float,
    depths: np.ndarray,
) -> np.ndarray:
    """Effective alpha = per-primitive alpha × global opacity × center
    fade, clamped to 0-1.

    Classification predictor only — the drawn alpha still comes from
    uColor.a × uOpacity × the per-fragment fade in the shader.
    """
    return np.clip(alphas * global_opacity * _center_fade(depths), 0.0, 1.0)


def _split_opaque_transparent(
    depths: np.ndarray,
    effective_alphas: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Split primitive indices into (opaque, transparent_sorted).

    Opaque indices keep ascending index order (np.nonzero guarantees
    it — under depth testing their order is irrelevant, and keeping it
    makes the all-opaque path identical to the pre-transparency code).
    Transparent indices are sorted by descending view depth = far to
    near, the correct blend order when depth writes are disabled.
    """
    opaque = np.nonzero(effective_alphas >= 1.0 - eps)[0]
    trans = np.nonzero(effective_alphas < 1.0 - eps)[0]
    trans = trans[np.argsort(-depths[trans], kind="stable")]
    return opaque, trans


# Identity model matrix — bonds are baked in world space already.
_IDENTITY = _to_qmatrix(np.eye(4, dtype=np.float32))


# ----------------------------------------------------------------------
# Shaders (GLSL 3.30 core)
# ----------------------------------------------------------------------

_SPHERE_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat4 uView;
out vec3 vNormal;
out vec3 vViewPos;
out float vViewDepth;
out vec3 vColor;
void main() {
    vNormal = aNormal;
    vColor = aColor;
    // View-space position MUST go through the model matrix: the raw
    // object-space aPos is on the UNIT sphere, and the depth error
    // (±atom radius, up to ~1 A) is the same order as the near-fade
    // thresholds — zooming close used to fade the wrong faces.
    vec4 viewPos = uView * uModel * vec4(aPos, 1.0);
    vViewPos = viewPos.xyz;
    vViewDepth = -viewPos.z;  // camera looks down -z in view space
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_SPHERE_FRAG = """
#version 330 core
in vec3 vNormal;        // world space
in vec3 vViewPos;       // view space, unnormalized
in float vViewDepth;
in vec3 vColor;
uniform vec4 uColor;
uniform mat4 uView;
uniform vec3 uLightDir;     // world, unit, direction TO the key light
uniform float uHeadlight;   // 0.0 world-fixed, 1.0 light from the camera
uniform float uAmbient;
uniform vec3 uAmbSky;       // hemisphere sky color (world up)
uniform vec3 uAmbGround;    // hemisphere ground color
uniform float uDiffuse;
uniform float uSpecular;
uniform float uSpecPerAtom; // per-element specular multiplier (0 = none)
uniform float uShininess;
uniform float uFillIntensity;
uniform vec3 uFillDir;      // world, unit
uniform float uGamma;       // display gamma (2.2)
uniform float uUseVtxColor; // 1.0 = use per-vertex color (bonds by element)
uniform float uOpacity;     // global atom/bond opacity (0-1)
uniform float uFadeStart;   // view depth (A) where fading begins
uniform float uFadeEnd;     // view depth (A) where the atom is fully faded
out vec4 fragColor;
void main() {
    vec3 n = normalize(vNormal);
    if (!gl_FrontFacing) n = -n;  // two-sided lighting: correct when viewed from inside

    // Headlight: the light comes from the camera itself — per-fragment
    // direction (a constant vec3(0,0,1) would skew off-center atoms).
    vec3 V = -normalize(vViewPos);
    vec3 Nv = mat3(uView) * n;
    vec3 Lv = (uHeadlight > 0.5) ? V : mat3(uView) * uLightDir;
    vec3 H = normalize(Lv + V);

    // Hemisphere ambient mixes on the WORLD up axis (not view up), so
    // orbiting the camera below the model does not change the shading.
    vec3 ambient = mix(uAmbGround, uAmbSky, 0.5 + 0.5 * n.y) * uAmbient;
    float diff = max(dot(Nv, Lv), 0.0);
    float spec = (diff > 0.0) ? pow(max(dot(Nv, H), 0.0), uShininess) : 0.0;
    float fill = uFillIntensity * max(dot(Nv, mat3(uView) * uFillDir), 0.0);

    // Element colors are display-referred sRGB: decode, light, re-encode.
    // A fully lit surface outputs the exact element color, and gamma
    // acts as a perceptual lift for the shadowed side.
    vec3 base = mix(uColor.rgb, vColor, uUseVtxColor);
    vec3 lin = pow(base, vec3(uGamma)) * (ambient + uDiffuse * diff + fill)
             + uSpecular * uSpecPerAtom * spec * vec3(1.0);
    vec3 enc = pow(max(lin, vec3(0.0)), vec3(1.0 / uGamma));

    // Near-distance fade (user decision 2026-08-13): atoms fade out as
    // the camera approaches them, so zooming in never shows a
    // cross-section disk through the sphere.
    float fade = smoothstep(uFadeEnd, uFadeStart, vViewDepth);
    fragColor = vec4(enc, uColor.a * uOpacity * fade);
}
"""

_FLAT_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_FLAT_FRAG = """
#version 330 core
uniform vec4 uColor;
out vec4 fragColor;
void main() {
    fragColor = uColor;
}
"""

_GRAD_VERT = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aColor;
out vec3 vColor;
void main() {
    vColor = aColor;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

_GRAD_FRAG = """
#version 330 core
in vec3 vColor;
out vec4 fragColor;
void main() {
    fragColor = vec4(vColor, 1.0);
}
"""

# ----------------------------------------------------------------------
# Label rendering: crisp vector TEXT drawn with QPainter (MS style) —
# background-colored outline + element-colored fill, no quads/textures.
# ----------------------------------------------------------------------

def _gradient_verts(
    top: tuple[float, float, float],
    bottom: tuple[float, float, float],
) -> np.ndarray:
    """Background-gradient fullscreen triangle (NDC, covers the viewport).

    Two bottom vertices carry the bottom color, the top vertex the top
    color — the rasterizer interpolates the vertical gradient.
    """
    return np.array([
        [-1.0, -1.0, *bottom],
        [3.0, -1.0, *bottom],
        [-1.0, 3.0, *top],
    ], dtype=np.float32)


class Viewport3D(QOpenGLWidget):
    """3D ball-and-stick structure viewport (Qt native OpenGL).

    Public API mirrors the previous VisPy implementation so the main
    window needs no changes.
    """

    # Signals
    atom_clicked = Signal(int)      # user clicked an atom (index)
    background_clicked = Signal()   # user clicked empty space
    bond_clicked = Signal(int)      # user clicked a bond (bond list index)
    atoms_selected = Signal(list, str)      # indices + "replace"|"add"|"toggle"
    atom_place_requested = Signal(object, object)  # world position, anchor index|None
    atoms_moved = Signal(list, object)      # indices + new positions (K×3)
    bond_created = Signal(int, int)         # atom indices
    delete_requested = Signal(str, int)     # "atom"|"bond" + index
    measurement_added = Signal(str, list)   # kind + atom indices
    mode_changed = Signal(object)           # new ToolMode
    frozen_drag_blocked = Signal()          # a drag tool grabbed a frozen atom

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Scene data (numpy, no GL — safe to set before initializeGL)
        self._atoms: Atoms | None = None
        self._atom_pos = np.zeros((0, 3), dtype=np.float32)
        self._atom_radius = np.zeros(0, dtype=np.float32)
        self._atom_color = np.zeros((0, 4), dtype=np.float32)
        self._symbols: list[str] = []
        self._bond_verts = np.zeros((0, 9), dtype=np.float32)
        self._cell_verts: np.ndarray | None = None
        # cell frame drawn as screen-space quads (glLineWidth is a no-op
        # on core profiles) — rebuilt per frame from _cell_verts
        self._cell_frame_verts = np.zeros((0, 3), dtype=np.float32)
        self._selected_indices: set[int] = set()
        self._selected_bonds: set[int] = set()  # bond list indices
        self._preview_indices: set[int] = set()
        # Per-atom fixed flags (VASP selective dynamics) — mirror of the
        # model's, fed via set_structure(fixed=...). Used by the drag
        # tools to reject frozen targets at press time, and by the
        # renderer for the frozen-atom edge outline.
        self._fixed_flags = np.zeros((0, 3), dtype=bool)
        self._fixed_mask = np.zeros(0, dtype=bool)  # any direction fixed
        self._bonds: list = []          # Bond objects, same order as render
        self._bond_ranges: list[tuple[int, int]] = []  # (first, count) vertex slices
        self._rubber_rect = None        # QRectF | None (box selection)
        self._ghost_pos = None          # np.ndarray | None (add-atom preview)
        self._ghost_bond_from = None    # world pos of the ghost bond's anchor
        self.current_element = "C"      # element of the atom being added
        self._measurements = []         # [(kind, indices, text), ...]
        self._meas_pairs: list[tuple[int, int]] = []  # per-measurement line endpoints (atom indices)
        self._meas_verts = np.zeros((0, 3), dtype=np.float32)
        self._preview_dirty = False     # bond geometry needs a rebake (drag preview)
        # Display style (parameter pack over the existing renderer)
        self._covalent_radii = np.zeros(0, dtype=np.float32)  # unscaled
        # All adjustable render parameters live in a RenderSettings
        # object (View → Display Options edits it live; AppConfig
        # persists it). _light_dir is the cached unit light vector.
        self._render_settings = RenderSettings.default()
        self._light_dir = np.asarray(
            light_direction(25.0, 33.6), dtype=np.float32)
        self._grad_verts = np.zeros((0, 5), dtype=np.float32)
        self._data_dirty = False

        # Interaction mode (tool dispatch — tools live in ui/tools.py)
        self._tool_instances: dict = {}
        self._mode = ToolMode.SELECT
        self._active_tool = make_tool(ToolMode.SELECT, self)
        self._tool_active = False       # True while the tool owns the press

        # Camera state (orbit camera, perspective projection)
        self._cam_center = np.zeros(3, dtype=np.float64)
        self._cam_distance = 10.0
        self._cam_azimuth = 45.0
        self._cam_elevation = 30.0
        self._cam_up = np.array([0.0, 1.0, 0.0])  # world up for the view
        self._fit_radius = 5.0
        self._fit_distance = 10.0  # camera distance after the last fit (label zoom reference)
        self._view_fitted = False  # True once _fit_camera has run
        self._has_cell = False  # set in set_structure; controls fit view
        self._cell = np.eye(3, dtype=np.float64)
        self._pbc: tuple[bool, bool, bool] = (False, False, False)

        # GL resources
        self._gl = None
        self._gl_ready = False
        self._gl_failed = False
        self._sphere_prog: QOpenGLShaderProgram | None = None
        self._flat_prog: QOpenGLShaderProgram | None = None
        self._grad_prog: QOpenGLShaderProgram | None = None
        self._unit_vao: QOpenGLVertexArrayObject | None = None
        self._unit_vbo: QOpenGLBuffer | None = None
        self._unit_ibo: QOpenGLBuffer | None = None
        self._unit_n_indices = 0
        self._bond_vao: QOpenGLVertexArrayObject | None = None
        self._bond_vbo: QOpenGLBuffer | None = None
        self._bond_n_verts = 0
        self._cell_vao: QOpenGLVertexArrayObject | None = None
        self._cell_vbo: QOpenGLBuffer | None = None
        self._grad_vao: QOpenGLVertexArrayObject | None = None
        self._grad_vbo: QOpenGLBuffer | None = None
        self._pick_fbo: QOpenGLFramebufferObject | None = None
        self._pick_fbo_size: tuple[int, int] = (0, 0)
        # Cached uniform locations (filled by initializeGL) — PySide6
        # has no (name, float) setUniformValue overload, so floats are
        # set by location.
        self._u_loc: dict[str, int] = {}
        self._far_plane = 100.0

        # Mouse interaction state
        self._press_button: Qt.MouseButton | None = None
        self._press_pos = None
        self._last_pos = None
        self._dragged = False

        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_structure(
        self,
        atoms: Atoms | None,
        reset_view: bool = True,
        bonds: list | None = None,
        fixed: np.ndarray | None = None,
    ) -> None:
        """Replace the displayed structure and re-render.

        Args:
            atoms: ASE Atoms to display, or None to clear the view.
            reset_view: If True, fit the camera to the structure
                (default, used when opening files). False preserves the
                current camera (used for in-place edits such as surface
                cleaving and supercells).
            bonds: Persistent bond list from the model (Bond objects);
                auto-computed from connectivity when None (standalone use).
            fixed: Per-atom per-direction fixed flags (N×3 bool, True =
                fixed). None keeps the current flags when the atom count
                matches, otherwise zeroes them (defensive fallback —
                stale flags of a different shape must never leak).
        """
        self._atoms = atoms
        self._selected_indices = set()
        self._selected_bonds = set()
        self._preview_indices = set()

        n = 0 if atoms is None else len(atoms)
        if fixed is not None:
            flags = np.asarray(fixed, dtype=bool)
            self._fixed_flags = (flags.copy()
                                 if flags.shape == (n, 3)
                                 else np.zeros((n, 3), dtype=bool))
        elif len(self._fixed_flags) != n:
            self._fixed_flags = np.zeros((n, 3), dtype=bool)
        self._fixed_mask = self._fixed_flags.any(axis=1)

        if atoms is None or len(atoms) == 0:
            self._atom_pos = np.zeros((0, 3), dtype=np.float32)
            self._atom_radius = np.zeros(0, dtype=np.float32)
            self._atom_color = np.zeros((0, 4), dtype=np.float32)
            self._bond_verts = np.zeros((0, 9), dtype=np.float32)
            self._bonds = []
            self._bond_ranges = []
            self._cell_verts = None
        else:
            positions = np.asarray(atoms.get_positions(), dtype=np.float32)
            symbols = list(atoms.get_chemical_symbols())
            self._symbols = symbols
            cell = atoms.get_cell().array if atoms.get_cell().rank == 3 else np.eye(3)
            pbc = tuple(atoms.get_pbc()) if atoms.get_pbc().any() else (False, False, False)

            radii = []
            colors = []
            for sym in symbols:
                z = atomic_numbers.get(sym, 0)
                r = float(covalent_radii[z] if 0 < z < len(covalent_radii) else 0.77)
                base = self._effective_atom_color(sym)
                radii.append(r)
                colors.append((base[0], base[1], base[2], base[3]))

            self._atom_pos = positions
            self._covalent_radii = np.asarray(radii, dtype=np.float32)
            self._apply_display()
            self._atom_color = np.asarray(colors, dtype=np.float32)

            # Cell/pbc must be updated BEFORE baking the bonds —
            # _bake_bond_verts uses them for the minimum-image vectors
            # (stale cell/pbc from a previously opened structure would
            # draw periodic bonds as sticks across the whole box).
            self._has_cell = any(pbc)
            self._cell = np.asarray(cell, dtype=np.float64)
            self._pbc = pbc
            self._cell_verts = _cell_edges(cell) if any(pbc) else None

            bond_list = (bonds if bonds is not None
                         else find_bonds(positions, symbols, cell, pbc))
            self._bonds = bond_list
            self._bake_bond_verts()

            if reset_view or not self._view_fitted:
                self._fit_camera()

        self._preview_dirty = False
        self._data_dirty = True
        self.update()

    @property
    def fixed_flags(self) -> np.ndarray:
        """Per-atom per-direction fixed flags (N×3 bool) of the scene."""
        return self._fixed_flags

    def _bake_bond_verts(self) -> None:
        """(Re)bake bond triangles from the CURRENT atom positions.

        Called from set_structure and — during a move drag — every frame
        the preview updates atom positions, so the sticks follow the
        atoms instead of lagging behind at their baked locations.

        Bake per bond so every bond keeps a vertex slice of its own
        (needed by the ID-color pick pass to draw individual bonds).
        Ranges stay aligned with bond indices even for zero-length
        bonds (drawing 0 verts is a no-op).
        Bond orders: single = one cylinder; double/triple = parallel
        cylinders offset laterally; aromatic (4) = dashed segments.
        """
        positions = self._atom_pos
        cell = self._cell
        pbc = self._pbc
        # Vertex colors for bonds-by-element rendering (i-end takes the
        # color of atom i, j-end of atom j) — baked per frame with the
        # geometry; unused when the setting is off (uUseVtxColor = 0).
        bake_colors = self._render_settings.bonds_by_element
        self._bond_ranges = []
        pieces = []
        offset = 0
        for b in self._bonds:
            s = np.asarray([positions[b.i]], dtype=np.float64)
            e = np.asarray([positions[b.i] + mic_vector(
                b.i, b.j, positions, cell, pbc)], dtype=np.float64)
            ci = tuple(float(c) for c in self._atom_color[b.i][:3]) if bake_colors else None
            cj = tuple(float(c) for c in self._atom_color[b.j][:3]) if bake_colors else None
            bond_pieces: list[np.ndarray] = []
            if b.order == 2:
                # thinner components so the parallel sticks stay
                # visually separate (they would merge at 0.12 Å)
                bond_pieces.append(_cylinder_verts(s, e, 0.05, lateral_offset=-0.09,
                                                   color_i=ci, color_j=cj))
                bond_pieces.append(_cylinder_verts(s, e, 0.05, lateral_offset=+0.09,
                                                   color_i=ci, color_j=cj))
            elif b.order == 3:
                bond_pieces.append(_cylinder_verts(s, e, 0.05,
                                                   color_i=ci, color_j=cj))
                bond_pieces.append(_cylinder_verts(s, e, 0.05, lateral_offset=-0.12,
                                                   color_i=ci, color_j=cj))
                bond_pieces.append(_cylinder_verts(s, e, 0.05, lateral_offset=+0.12,
                                                   color_i=ci, color_j=cj))
            elif b.order == 4:
                axis = e[0] - s[0]
                for d in range(6):
                    t0 = d / 6.0 + 0.015
                    t1 = min((d + 0.8) / 6.0, 1.0)
                    if t1 <= t0:
                        continue
                    bond_pieces.append(_cylinder_verts(
                        s + axis * t0, s + axis * t1, self._render_settings.bond_radius,
                        color_i=ci, color_j=cj))
            else:
                bond_pieces.append(_cylinder_verts(
                    s, e, self._render_settings.bond_radius,
                    color_i=ci, color_j=cj))
            count = sum(len(p) for p in bond_pieces)
            self._bond_ranges.append((offset, count))
            offset += count
            pieces.extend(p for p in bond_pieces if len(p))
        self._bond_verts = (np.concatenate(pieces) if pieces
                            else np.zeros((0, 9), dtype=np.float32))

    def highlight_atom(self, index: int | None) -> None:
        """Highlight the atom at index (None clears) — legacy single API."""
        self.set_highlight({index} if index is not None else None)

    def set_highlight(self, indices) -> None:
        """Set the selection highlight (iterable of atom indices)."""
        self._selected_indices = set(indices) if indices else set()
        self.update()

    def set_preview_highlight(self, indices) -> None:
        """Set a temporary highlight (e.g. CreateBond's first atom)."""
        self._preview_indices = set(indices) if indices else set()
        self.update()

    def set_bond_highlight(self, indices) -> None:
        """Set the selected-bond highlight (bond list indices)."""
        self._selected_bonds = set(indices) if indices else set()
        self.update()

    # ------------------------------------------------------------------
    # Display styles (parameter packs — no second renderer)
    # ------------------------------------------------------------------

    def _apply_display(self) -> None:
        """Apply the current style to the atom radii / bond visibility."""
        rs = self._render_settings
        if rs.style == "cpk":
            self._atom_radius = self._covalent_radii.copy()
        elif rs.style == "wireframe":
            self._atom_radius = self._covalent_radii * 0.25
        else:  # ball_stick
            self._atom_radius = self._covalent_radii * rs.sphere_scale
        self.update()

    @property
    def _show_bonds(self) -> bool:
        """Bonds are hidden in the CPK (space-filling) style."""
        return self._render_settings.style != "cpk"

    @property
    def _show_cell(self) -> bool:
        return self._render_settings.show_cell

    @property
    def _show_labels(self) -> bool:
        return self._render_settings.show_labels

    def set_render_settings(self, rs: RenderSettings) -> None:
        """Replace the whole render settings object and re-render.

        The existing granular setters below all funnel through this so
        the View menu wiring keeps working unchanged.
        """
        old = self._render_settings
        self._render_settings = RenderSettings.from_dict(rs.to_dict())
        self._light_dir = np.asarray(
            light_direction(rs.light_azimuth, rs.light_elevation),
            dtype=np.float32,
        )
        self._grad_verts = _gradient_verts(
            self._render_settings.gradient_top,
            self._render_settings.gradient_bottom,
        )
        self._apply_display()
        if self._bonds and old.bond_radius != self._render_settings.bond_radius:
            # the radius is baked into the bond triangles
            self._bake_bond_verts()
        if old.color_scheme != self._render_settings.color_scheme \
                or old.atom_colors != self._render_settings.atom_colors \
                or old.bonds_by_element != self._render_settings.bonds_by_element:
            # element colors feed the atom AND bond vertex arrays
            self._recolor_atoms()
            if self._bonds:
                self._bake_bond_verts()
        # uniform/gradient changes need a buffer (re)upload — cheap, and
        # simpler than diffing every field
        self._data_dirty = True
        self.update()

    def render_settings(self) -> RenderSettings:
        """The current render settings object."""
        return self._render_settings

    def _recolor_atoms(self) -> None:
        """Rebuild the per-atom color array from the settings."""
        colors = []
        for sym in self._symbols:
            base = self._effective_atom_color(sym)
            colors.append((base[0], base[1], base[2], base[3]))
        self._atom_color = np.asarray(colors, dtype=np.float32)

    def _effective_atom_color(self, symbol: str) -> tuple[float, float, float, float]:
        """Element RGBA honoring overrides → color scheme → Jmol.

        The alpha channel comes from the per-element override table
        (default 1.0 = opaque); the 3D view is the only consumer — the
        structure tree keeps its own readable text colors.
        """
        rs = self._render_settings
        override = rs.atom_colors.get(symbol)
        if override is not None:
            return (override[0], override[1], override[2], override[3])
        rgb = PALETTES.get(rs.color_scheme, {}).get(symbol, element_color(symbol))
        return (rgb[0], rgb[1], rgb[2], 1.0)

    def structure_symbols(self) -> list[str]:
        """Sorted unique element symbols of the displayed structure."""
        return sorted(set(self._symbols))

    def set_structure_style(self, style: str) -> None:
        """Switch the display style: "ball_stick" | "cpk" | "wireframe"."""
        if style not in ("ball_stick", "cpk", "wireframe"):
            raise ValueError(f"Unknown display style: {style}")
        self._render_settings.style = style
        self._apply_display()

    def structure_style(self) -> str:
        """The current display style."""
        return self._render_settings.style

    def set_show_cell(self, visible: bool) -> None:
        """Show/hide the unit cell frame."""
        self._render_settings.show_cell = bool(visible)
        self.update()

    def set_show_labels(self, visible: bool) -> None:
        """Show/hide element labels (only for structures ≤ 500 atoms)."""
        self._render_settings.show_labels = bool(visible)
        self.update()

    def set_background_color(self, color: tuple[float, float, float]) -> None:
        """Set the viewport background color (RGB 0-1)."""
        self._render_settings.background_color = tuple(float(c) for c in color)
        self.update()

    def reset_view(self) -> None:
        """Reset the camera to fit the current structure."""
        if self._atoms is not None and len(self._atoms) > 0:
            self._fit_camera()
            self.update()

    def set_view_direction(
        self,
        azimuth: float,
        elevation: float,
        up: tuple[float, float, float] | None = None,
    ) -> None:
        """Point the camera along a preset direction (View menu).

        Args:
            azimuth, elevation: Camera angles in degrees.
            up: Optional world-up override (needed for ±Y views where
                the default up would be parallel to the view direction).
        """
        self._cam_azimuth = float(azimuth)
        self._cam_elevation = float(elevation)
        if up is not None:
            self._cam_up = np.asarray(up, dtype=np.float64)
        self.update()

    @property
    def gl_available(self) -> bool:
        """True if OpenGL rendering was successfully initialized."""
        return self._gl_ready and not self._gl_failed

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _fit_camera(self) -> None:
        """Center the camera on the structure and reset orientation.

        Uses the bounding-sphere radius (max atom distance from the
        center) — the half-extent is NOT enough: corner atoms of a
        cubic cell sit at extent/2 × √3 from the center and would fall
        outside the view.

        Periodic structures anchor on the geometric cell center, not
        the atom centroid, and the fitted radius covers the cell
        corners as well: a padded vacuum box (or a slab with vacuum)
        is much larger than its atoms, so framing on atoms alone left
        the cell frame lopsided and spilling out of the viewport.

        Orientation: periodic structures default to a top-down view
        along the crystallographic c axis (c out of the screen, b up,
        a to the left — the VESTA convention); non-periodic molecules
        keep the generic isometric 45°/30° view.
        """
        positions = self._atom_pos
        if len(positions) == 0:
            return
        if self._has_cell:
            # Cell parallelepiped starts at the origin → center = (a+b+c)/2.
            self._cam_center = self._cell.sum(axis=0) / 2.0
            ref = np.concatenate(
                [positions - self._cam_center,
                 self._cell_verts - self._cam_center], axis=0)
        else:
            self._cam_center = np.nanmean(positions, axis=0)
            if not np.isfinite(self._cam_center).all():
                self._cam_center = np.zeros(3)
            ref = positions - self._cam_center
        radii = np.linalg.norm(ref, axis=1)
        # NaN coordinates in a corrupted file must not poison the
        # camera (max(nan, 1.0) returns nan, and reset_view could
        # never recover it) — ignore non-finite distances.
        radius = (float(np.nanmax(radii)) if np.isfinite(radii).any()
                  else 0.0)
        self._fit_radius = (
            max(radius, 1.0)
            + float(self._atom_radius.max())
            + 0.5
        )
        # Ortho framing: the visible half-height IS the camera distance,
        # so d = 1.35×R frames the bounding sphere with a 35% margin.
        self._cam_distance = max(self._fit_radius * 1.35, 1.0)
        self._fit_distance = self._cam_distance
        self._view_fitted = True

        if self._has_cell:
            c = np.asarray(self._cell[2], dtype=np.float64)
            norm = float(np.linalg.norm(c))
            if norm > 1e-9:
                forward = c / norm
                self._cam_azimuth = math.degrees(math.atan2(forward[0], forward[2]))
                self._cam_elevation = math.degrees(math.asin(forward[1]))
                # Screen-up = b projected onto the plane ⊥ c (so b points
                # up on screen); fall back to world up if b ∥ c.
                b = np.asarray(self._cell[1], dtype=np.float64)
                up = b - float(b @ forward) * forward
                if np.linalg.norm(up) > 1e-6:
                    self._cam_up = up / np.linalg.norm(up)
                else:
                    self._cam_up = np.array([0.0, 1.0, 0.0])
            return
        self._cam_azimuth = 45.0
        self._cam_elevation = 30.0
        self._cam_up = np.array([0.0, 1.0, 0.0])

    def _camera_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Projection and view matrices for the current camera + widget."""
        az = math.radians(self._cam_azimuth)
        el = math.radians(self._cam_elevation)
        direction = np.array([
            math.cos(el) * math.sin(az),
            math.sin(el),
            math.cos(el) * math.cos(az),
        ])
        eye = self._cam_center + self._cam_distance * direction
        forward = self._cam_center - eye
        forward /= np.linalg.norm(forward)
        # Screen-right from the up vector, re-orthogonalized against
        # forward. A ±Y view leaves _cam_up ∥ forward for a following
        # Front/Back view — the old z-vector fallback was parallel to
        # forward there too and produced a NaN view matrix (the render
        # went blank until Reset View).
        up = self._cam_up - float(self._cam_up @ forward) * forward
        if np.linalg.norm(up) < 1e-6:
            fallback = ([0.0, 0.0, 1.0] if abs(forward[2]) < 0.9
                        else [1.0, 0.0, 0.0])
            up = (np.asarray(fallback, dtype=np.float64)
                  - float(np.dot(fallback, forward)) * forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        view = _look_at(eye, self._cam_center, up)

        w = max(self.width(), 1)
        h = max(self.height(), 1)
        aspect = w / h
        half_h = self._cam_distance
        half_w = half_h * aspect
        # Orthographic projection (crystallography convention — the
        # user's preferred default; perspective distorts parallel
        # lattice edges). Near plane fixed at 0.01 so geometry only
        # clips once it is genuinely behind the camera.
        near = 0.01
        # Far plane must cover the CURRENT scene extent — after
        # in-place edits (supercell, add/delete atoms) the camera is
        # intentionally not re-fit, so _fit_radius is stale and a fixed
        # margin would clip far atoms into a cross-section. Zooming out
        # could never recover that, hence the per-frame computation.
        # The cell-frame corners are included too: for a molecule in a
        # padded vacuum box (or a slab with vacuum) they extend far
        # beyond the atoms, and omitting them clipped the frame's back
        # corners like an invisible wall.
        if len(self._atom_pos) > 0 or self._cell_verts is not None:
            centered = []
            if len(self._atom_pos) > 0:
                centered.append(self._atom_pos - self._cam_center)
            if self._cell_verts is not None:
                centered.append(self._cell_verts - self._cam_center)
            extent = float(
                np.linalg.norm(np.concatenate(centered, axis=0), axis=1).max()
            )
            if len(self._atom_pos) > 0:
                extent += float(self._atom_radius.max())
        else:
            extent = self._fit_radius
        far = self._cam_distance + extent + 5.0
        self._far_plane = float(far)  # depth-cue reference (read in paintGL)
        proj = _ortho(-half_w, half_w, -half_h, half_h, near, far)
        return proj, view

    # ------------------------------------------------------------------
    # GL lifecycle
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        """Compile shaders and create GL buffers (once per context)."""
        try:
            self._gl = self.context().functions()
            if self._gl is None:
                raise RuntimeError("no GL functions available")

            self._sphere_prog = self._build_program(_SPHERE_VERT, _SPHERE_FRAG)
            self._flat_prog = self._build_program(_FLAT_VERT, _FLAT_FRAG)
            self._grad_prog = self._build_program(_GRAD_VERT, _GRAD_FRAG)

            # Cached uniform locations (floats are set by location —
            # PySide6 has no (name, float) setUniformValue overload).
            for name in (
                "uHeadlight", "uAmbient", "uDiffuse", "uSpecular",
                "uSpecPerAtom", "uShininess", "uFillIntensity", "uGamma",
                "uUseVtxColor", "uOpacity", "uFadeStart", "uFadeEnd",
            ):
                self._u_loc[name] = self._sphere_prog.uniformLocation(name)

            # Unit icosphere (shared by all atoms — scaled per atom)
            verts, faces = _unit_icosphere()
            interleaved = np.hstack([verts, verts])  # pos + normal (same)
            self._unit_n_indices = faces.size
            self._unit_vao, self._unit_vbo, self._unit_ibo = self._make_mesh(
                interleaved, faces
            )
            self._unit_vao.bind()
            loc_pos = self._sphere_prog.attributeLocation("aPos")
            loc_nrm = self._sphere_prog.attributeLocation("aNormal")
            self._sphere_prog.enableAttributeArray(loc_pos)
            self._sphere_prog.enableAttributeArray(loc_nrm)
            self._sphere_prog.setAttributeBuffer(loc_pos, GL_FLOAT, 0, 3, 24)
            self._sphere_prog.setAttributeBuffer(loc_nrm, GL_FLOAT, 12, 3, 24)
            # aColor (location 2) stays DISABLED on the unit VAO — a
            # disabled array reads as (0,0,0) and uUseVtxColor = 0 makes
            # the shader ignore it.
            self._unit_vao.release()

            # Bond buffer (content uploaded on demand). Vertices are
            # pos(3) + normal(3) + color(3), stride 36. The VAO carries
            # pointers for BOTH programs: the lit sphere program reads
            # all three attributes, the flat program (used by the
            # ID-color pick pass) reads position only.
            self._bond_vao, self._bond_vbo, _ = self._make_mesh(
                np.zeros((0, 9), dtype=np.float32), None
            )
            self._bond_vao.bind()
            loc_b = self._flat_prog.attributeLocation("aPos")
            self._flat_prog.enableAttributeArray(loc_b)
            self._flat_prog.setAttributeBuffer(loc_b, GL_FLOAT, 0, 3, 36)
            self._sphere_prog.enableAttributeArray(loc_pos)
            self._sphere_prog.enableAttributeArray(loc_nrm)
            self._sphere_prog.enableAttributeArray(
                self._sphere_prog.attributeLocation("aColor"))
            self._sphere_prog.setAttributeBuffer(loc_pos, GL_FLOAT, 0, 3, 36)
            self._sphere_prog.setAttributeBuffer(loc_nrm, GL_FLOAT, 12, 3, 36)
            self._sphere_prog.setAttributeBuffer(
                self._sphere_prog.attributeLocation("aColor"),
                GL_FLOAT, 24, 3, 36)
            self._bond_vao.release()

            # Background-gradient fullscreen triangle (uploaded on demand)
            self._grad_vao, self._grad_vbo, _ = self._make_mesh(
                np.zeros((0, 5), dtype=np.float32), None
            )
            self._grad_vao.bind()
            loc_gp = self._grad_prog.attributeLocation("aPos")
            loc_gc = self._grad_prog.attributeLocation("aColor")
            self._grad_prog.enableAttributeArray(loc_gp)
            self._grad_prog.enableAttributeArray(loc_gc)
            self._grad_prog.setAttributeBuffer(loc_gp, GL_FLOAT, 0, 2, 20)
            self._grad_prog.setAttributeBuffer(loc_gc, GL_FLOAT, 8, 3, 20)
            self._grad_vao.release()

            self._cell_vao, self._cell_vbo, _ = self._make_mesh(
                np.zeros((0, 3), dtype=np.float32), None
            )
            self._cell_vao.bind()
            loc_c = self._flat_prog.attributeLocation("aPos")
            self._flat_prog.enableAttributeArray(loc_c)
            self._flat_prog.setAttributeBuffer(loc_c, GL_FLOAT, 0, 3, 12)
            self._cell_vao.release()

            # Measurement lines (GL_LINES, uploaded on demand)
            self._meas_vao, self._meas_vbo, _ = self._make_mesh(
                np.zeros((0, 3), dtype=np.float32), None
            )
            self._meas_vao.bind()
            loc_m = self._flat_prog.attributeLocation("aPos")
            self._flat_prog.enableAttributeArray(loc_m)
            self._flat_prog.setAttributeBuffer(loc_m, GL_FLOAT, 0, 3, 12)
            self._meas_vao.release()

            self._gl_ready = True
            self._data_dirty = True
        except Exception:
            self._gl_failed = True
            logger.exception("OpenGL initialization failed — 3D view degraded")

    def _build_program(self, vert_src: str, frag_src: str) -> QOpenGLShaderProgram:
        """Compile and link a shader program; raise on failure."""
        prog = QOpenGLShaderProgram(self)
        if not prog.addShaderFromSourceCode(QOpenGLShader.Vertex, vert_src):
            raise RuntimeError(f"vertex shader compile failed: {prog.log()}")
        if not prog.addShaderFromSourceCode(QOpenGLShader.Fragment, frag_src):
            raise RuntimeError(f"fragment shader compile failed: {prog.log()}")
        if not prog.link():
            raise RuntimeError(f"shader link failed: {prog.log()}")
        return prog

    def _make_mesh(
        self,
        verts: np.ndarray,
        indices: np.ndarray | None,
    ) -> tuple[QOpenGLVertexArrayObject, QOpenGLBuffer, QOpenGLBuffer | None]:
        """Create a VAO with a vertex VBO (and optional index buffer)."""
        vao = QOpenGLVertexArrayObject(self)
        vao.create()
        vao.bind()
        vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        vbo.create()
        vbo.bind()
        vbo.allocate(verts.tobytes(), verts.nbytes)
        ibo = None
        if indices is not None:
            ibo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
            ibo.create()
            ibo.bind()
            ibo.allocate(indices.tobytes(), indices.nbytes)
        vao.release()
        return vao, vbo, ibo

    def _upload_scene_buffers(self) -> None:
        """Upload current scene data into the GL buffers."""
        if self._gl is None:
            return
        self._bond_vao.bind()
        self._bond_vbo.bind()
        self._bond_vbo.allocate(self._bond_verts.tobytes(), self._bond_verts.nbytes)
        self._bond_n_verts = len(self._bond_verts)
        self._bond_vao.release()

        self._cell_vao.bind()
        self._cell_vbo.bind()
        self._cell_vbo.allocate(
            self._cell_frame_verts.tobytes(), self._cell_frame_verts.nbytes)
        self._cell_vao.release()

        self._meas_vao.bind()
        self._meas_vbo.bind()
        self._meas_vbo.allocate(self._meas_verts.tobytes(), self._meas_verts.nbytes)
        self._meas_vao.release()

        self._grad_vao.bind()
        self._grad_vbo.bind()
        self._grad_vbo.allocate(self._grad_verts.tobytes(), self._grad_verts.nbytes)
        self._grad_vao.release()
        self._data_dirty = False

    def paintGL(self) -> None:
        """Render the scene (called by Qt; must not be invoked directly)."""
        dpr = self.devicePixelRatioF()
        fb_w = max(int(self.width() * dpr), 1)
        fb_h = max(int(self.height() * dpr), 1)

        if self._gl is None or self._gl_failed:
            self._paint_text(self.tr(
                "3D rendering unavailable\n"
                "OpenGL could not be initialized on this system.\n"
                "Try updating your graphics driver or disable remote desktop."
            ))
            return

        gl = self._gl
        gl.glViewport(0, 0, fb_w, fb_h)
        if self._data_dirty:
            self._upload_scene_buffers()

        rs = self._render_settings
        if rs.background_gradient and len(self._grad_verts) > 0:
            # VESTA-style gradient background: a fullscreen triangle
            # (bottom color → top color) instead of a flat clear.
            gl.glClear(GL_DEPTH_BUFFER_BIT)
            self._grad_prog.bind()
            self._grad_vao.bind()
            gl.glDisable(GL_DEPTH_TEST)
            gl.glDrawArrays(GL_TRIANGLES, 0, len(self._grad_verts))
            self._grad_vao.release()
            self._grad_prog.release()
            gl.glEnable(GL_DEPTH_TEST)
        else:
            gl.glClearColor(*rs.background_color, 1.0)
            gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        gl.glEnable(GL_DEPTH_TEST)

        if len(self._atom_pos) == 0:
            self._paint_text(self.tr(
                "Open a structure file to begin\n"
                "(File → Open, or drag-and-drop)"
            ))
            return

        proj, view = self._camera_matrices()

        # Drag preview: re-bake the bonds from the preview atom positions
        # so the sticks follow the atoms instead of lagging behind.
        if self._preview_dirty and self._bonds:
            self._bake_bond_verts()
            self._upload_bond_buffer()
            self._preview_dirty = False

        # Measurement dashes are screen-space: rebuild + upload every
        # frame (zoom-proof dashes, lines track dragged atoms).
        if self._measurements:
            self._rebuild_meas_dashes()
            self._upload_meas_buffer()

        # Cell frame quads are screen-space too (width in px): rebuild
        # + upload every frame so the width is zoom-proof.
        if self._show_cell and self._cell_verts is not None \
                and len(self._cell_verts) > 0:
            self._rebuild_cell_frame_verts()
            self._upload_cell_buffer()

        # --- Atom spheres: opaque pass, then a transparent pass ---
        # Each primitive is classified by effective alpha (element
        # override × global opacity × center fade). Opaque primitives
        # draw first with depth writes ON — the old behavior, byte for
        # byte when everything is opaque. Transparent primitives draw
        # afterwards with depth writes OFF, sorted back-to-front, so
        # they blend over the cell frame (it stays visible through
        # them) and atoms behind show through the ones in front. (The
        # old fixed-order pass wrote depth for every sphere: a nearer
        # low-alpha atom hid the atom behind it, which read as an
        # opaque pale ball — and it depended on the camera angle.)
        atom_depths = _view_depths(view, self._atom_pos)
        atom_eff = _effective_alphas(
            self._atom_color[:, 3], rs.atom_opacity, atom_depths)
        opaque_atoms, trans_atoms = _split_opaque_transparent(
            atom_depths, atom_eff)

        self._sphere_prog.bind()
        self._sphere_prog.setUniformValue("uView", _to_qmatrix(view))
        u = self._u_loc
        self._sphere_prog.setUniformValue("uLightDir", QVector3D(*self._light_dir))
        self._sphere_prog.setUniformValue("uAmbSky", QVector3D(*rs.ambient_sky))
        self._sphere_prog.setUniformValue("uAmbGround", QVector3D(*rs.ambient_ground))
        self._sphere_prog.setUniformValue("uFillDir", QVector3D(*FILL_DIR))
        # Float uniforms MUST go through raw glUniform1f: PySide6's
        # setUniformValue(location, float) overload silently does
        # nothing in this build (verified by glGetUniformfv readback —
        # every location-set float stayed 0, rendering atoms black).
        gl.glUniform1f(u["uHeadlight"], 1.0 if rs.headlight else 0.0)
        gl.glUniform1f(u["uAmbient"], float(rs.ambient))
        gl.glUniform1f(u["uDiffuse"], float(rs.diffuse))
        gl.glUniform1f(u["uSpecular"], float(rs.specular))
        gl.glUniform1f(u["uShininess"], float(rs.shininess))
        gl.glUniform1f(u["uFillIntensity"], float(rs.fill_intensity))
        gl.glUniform1f(u["uGamma"], float(rs.gamma))
        gl.glUniform1f(u["uOpacity"], float(rs.atom_opacity))
        gl.glUniform1f(u["uFadeStart"], NEAR_FADE_START)
        gl.glUniform1f(u["uFadeEnd"], NEAR_FADE_END)
        # Specular highlights apply to all atoms uniformly (the old
        # metals-only distinction was removed by user decision).
        spec_on = 1.0 if rs.specular_enabled else 0.0

        def _draw_atom(i: int) -> None:
            """One atom sphere with the current program/uniforms."""
            scale = float(self._atom_radius[i])
            if i in self._selected_indices or i in self._preview_indices:
                # highlight keeps the element's opacity override
                color = (*HIGHLIGHT_COLOR, float(self._atom_color[i][3]))
            else:
                color = tuple(float(c) for c in self._atom_color[i])

            pos = self._atom_pos[i]
            if self._fixed_mask[i]:
                # Frozen atoms restore the old per-atom outline style:
                # a darker edge sphere at 1.04× drawn first — the body
                # sphere covers its center, leaving the dark rim visible.
                # The rim uses the ORIGINAL outline color (the displayed
                # body color × 0.45 — element color, or the highlight
                # color for a selected atom; user decision 2026-08-14,
                # red was rejected). Same alpha as the body so
                # transparent atoms keep a consistent rim; no specular
                # on the rim.
                edge = (*_edge_color(color[:3]), color[3])
                edge_model = _sphere_model(pos, scale * EDGE_SCALE)
                self._sphere_prog.setUniformValue(
                    "uMVP", _to_qmatrix(proj @ view @ edge_model))
                self._sphere_prog.setUniformValue(
                    "uModel", _to_qmatrix(edge_model))
                self._sphere_prog.setUniformValue("uColor", *edge)
                gl.glUniform1f(u["uSpecPerAtom"], 0.0)
                gl.glUniform1f(u["uUseVtxColor"], 0.0)
                gl.glDrawElements(GL_TRIANGLES, self._unit_n_indices,
                                  GL_UNSIGNED_INT, VoidPtr(0))

            model = _sphere_model(pos, scale)
            mvp = proj @ view @ model
            self._sphere_prog.setUniformValue("uMVP", _to_qmatrix(mvp))
            self._sphere_prog.setUniformValue("uModel", _to_qmatrix(model))
            self._sphere_prog.setUniformValue("uColor", *color)
            gl.glUniform1f(u["uSpecPerAtom"], spec_on)
            gl.glUniform1f(u["uUseVtxColor"], 0.0)
            gl.glDrawElements(GL_TRIANGLES, self._unit_n_indices,
                              GL_UNSIGNED_INT, VoidPtr(0))

        self._unit_vao.bind()
        gl.glEnable(GL_BLEND)
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        for i in opaque_atoms:
            _draw_atom(int(i))

        # --- Bonds (lit cylinders; selected bonds highlighted) ---
        # Opaque bonds first; transparent bonds join the transparent
        # pass below. Drawn with the sphere program: the radial normals
        # give them VESTA-style shading and the near fade applies to
        # them too. Bond vertices are world-space, so the model matrix
        # is the identity.
        trans_bonds = np.empty(0, dtype=np.int64)
        bond_depths = np.zeros(len(self._bond_ranges), dtype=np.float64)
        if self._bond_n_verts > 0 and self._show_bonds:
            # Bonds fade together with their atoms: per-element opacity
            # overrides produce per-bond alphas (mean of the two atoms);
            # a uniform scene stays a single draw call.
            bond_alphas = self._bond_alphas()
            base = (np.ones(len(self._bond_ranges), dtype=np.float64)
                    if bond_alphas is None else bond_alphas)
            mids = np.empty((len(self._bond_ranges), 3), dtype=np.float64)
            for k, (first, count) in enumerate(self._bond_ranges):
                if count == 0:
                    mids[k] = np.nan  # never drawn → classified opaque
                else:
                    # world-space midpoint (the lateral offsets of the
                    # cylinder rings cancel; minimum-image vectors are
                    # already baked into the vertices)
                    mids[k] = self._bond_verts[first:first + count, :3].mean(axis=0)
            bond_depths = _view_depths(view, mids)
            bond_depths[np.isnan(bond_depths)] = 1e9
            bond_eff = _effective_alphas(base, rs.atom_opacity, bond_depths)
            opaque_bonds, trans_bonds = _split_opaque_transparent(
                bond_depths, bond_eff)
            use_vtx = 1.0 if rs.bonds_by_element else 0.0

            def _draw_bond(k: int) -> None:
                """One bond cylinder with the current program/uniforms."""
                first, count = self._bond_ranges[k]
                if count == 0:
                    return
                self._sphere_prog.setUniformValue(
                    "uMVP", _to_qmatrix(proj @ view))
                self._sphere_prog.setUniformValue("uModel", _IDENTITY)
                gl.glUniform1f(u["uSpecPerAtom"], 0.5 * spec_on)
                if k in self._selected_bonds:
                    self._sphere_prog.setUniformValue(
                        "uColor", *HIGHLIGHT_COLOR, 1.0)
                    gl.glUniform1f(u["uUseVtxColor"], 0.0)
                else:
                    alpha = 1.0 if bond_alphas is None else float(bond_alphas[k])
                    self._sphere_prog.setUniformValue(
                        "uColor", 0.55, 0.55, 0.55, alpha)
                    gl.glUniform1f(u["uUseVtxColor"], use_vtx)
                gl.glDrawArrays(GL_TRIANGLES, first, count)

            self._bond_vao.bind()
            self._sphere_prog.setUniformValue("uMVP", _to_qmatrix(proj @ view))
            self._sphere_prog.setUniformValue("uModel", _IDENTITY)
            gl.glUniform1f(u["uSpecPerAtom"], 0.5 * spec_on)
            if len(opaque_bonds) == len(self._bond_ranges):
                # all bonds opaque — exactly the old code path
                if self._selected_bonds or bond_alphas is not None:
                    # per-bond colors: amber for selected, gray otherwise
                    for k, _range in enumerate(self._bond_ranges):
                        _draw_bond(k)
                else:
                    self._sphere_prog.setUniformValue(
                        "uColor", 0.55, 0.55, 0.55, 1.0)
                    gl.glUniform1f(u["uUseVtxColor"], use_vtx)
                    gl.glDrawArrays(GL_TRIANGLES, 0, self._bond_n_verts)
            else:
                for k in opaque_bonds:
                    _draw_bond(int(k))
            self._bond_vao.release()
        gl.glDisable(GL_BLEND)
        self._unit_vao.release()

        # --- Unit cell frame (only when periodic) ---
        # Drawn as screen-space quads of width cell_line_width px —
        # glLineWidth > 1 is silently ignored on GL core profiles, so
        # wide lines must be triangle geometry. Depth-TESTED like
        # normal geometry (atoms in front occlude the frame) AND
        # depth-CLAMPED: without the clamp, quad vertices passing
        # behind the near plane get clipped and the edges break into
        # dashed gaps whenever the camera is inside the cell or the
        # cell extends past the camera.
        if self._show_cell and self._cell_verts is not None and len(self._cell_verts) > 0:
            self._flat_prog.bind()
            self._cell_vao.bind()
            self._flat_prog.setUniformValue("uMVP", _to_qmatrix(proj @ view))
            self._flat_prog.setUniformValue("uColor", *rs.cell_color, 1.0)
            gl.glEnable(GL_DEPTH_CLAMP)
            gl.glEnable(GL_BLEND)
            gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            gl.glDrawArrays(GL_TRIANGLES, 0, len(self._cell_frame_verts))
            gl.glDisable(GL_DEPTH_CLAMP)
            gl.glDisable(GL_BLEND)
            self._cell_vao.release()

        # --- Transparent pass (atoms + bonds + ghost) ---
        # Depth-TESTED against the opaque scene (and the cell frame),
        # but depth WRITES OFF and sorted back-to-front: transparent
        # spheres blend over the frame instead of hiding it, and atoms
        # behind show through the ones in front.
        if len(trans_atoms) or len(trans_bonds) or self._ghost_pos is not None:
            prims: list[tuple[float, int, int]] = []  # (depth, kind, index)
            for i in trans_atoms:
                prims.append((float(atom_depths[i]), 0, int(i)))
            for k in trans_bonds:
                prims.append((float(bond_depths[k]), 1, int(k)))
            if self._ghost_pos is not None:
                gp = np.asarray(self._ghost_pos, dtype=np.float32)
                prims.append(
                    (float(_view_depths(view, gp.reshape(1, 3))[0]), 2, 0))
            prims.sort(key=lambda t: t[0], reverse=True)  # far → near

            self._sphere_prog.bind()
            gl.glEnable(GL_BLEND)
            gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            gl.glDepthMask(False)
            for _depth, kind, idx in prims:
                if kind == 0:
                    self._unit_vao.bind()
                    _draw_atom(idx)
                elif kind == 1:
                    self._bond_vao.bind()
                    _draw_bond(idx)
                else:
                    # Ghost atom (add-atom drag preview, semi-transparent)
                    z = atomic_numbers.get(self.current_element, 0)
                    ghost_radius = (
                        float(covalent_radii[z])
                        if 0 < z < len(covalent_radii) else 0.77
                    ) * rs.sphere_scale
                    ghost_color = self._effective_atom_color(
                        self.current_element)
                    model = _sphere_model(
                        np.asarray(self._ghost_pos, dtype=np.float32),
                        ghost_radius)
                    mvp = proj @ view @ model
                    self._unit_vao.bind()
                    self._sphere_prog.setUniformValue(
                        "uMVP", _to_qmatrix(mvp))
                    self._sphere_prog.setUniformValue(
                        "uModel", _to_qmatrix(model))
                    self._sphere_prog.setUniformValue(
                        "uColor", *ghost_color[:3], 0.55)
                    gl.glUniform1f(u["uSpecPerAtom"], spec_on)
                    gl.glUniform1f(u["uUseVtxColor"], 0.0)
                    gl.glDrawElements(GL_TRIANGLES, self._unit_n_indices,
                                      GL_UNSIGNED_INT, VoidPtr(0))
            gl.glDepthMask(True)
            gl.glDisable(GL_BLEND)
            self._bond_vao.release()
            self._unit_vao.release()

        # --- Measurement lines (dashed; drawn on TOP of the model
        # — depth test disabled so sticks never hide them) ---
        if len(self._meas_verts) > 0:
            self._flat_prog.bind()
            self._meas_vao.bind()
            self._flat_prog.setUniformValue("uMVP", _to_qmatrix(proj @ view))
            self._flat_prog.setUniformValue("uColor", *rs.measurement_color, 1.0)
            gl.glDisable(GL_DEPTH_TEST)
            gl.glEnable(GL_BLEND)
            gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            gl.glDrawArrays(GL_LINES, 0, len(self._meas_verts))
            gl.glDisable(GL_BLEND)
            gl.glEnable(GL_DEPTH_TEST)
            self._meas_vao.release()

        self._sphere_prog.release()
        self._flat_prog.release()

        # Measurement labels (screen space; projected from LIVE positions)
        for _kind, idx, text in self._measurements:
            if not idx:
                continue
            center = np.mean([self._atom_pos[i] for i in idx], axis=0)
            s = self.project_to_screen(center)
            if s is None:
                continue
            painter = QPainter(self)
            painter.setPen(self._measurement_pen())
            painter.setFont(self._measurement_font())
            self._draw_halo_text(painter, QPointF(s[0] + 8.0, s[1] - 6.0), text)
            painter.end()

        # Atom labels (element symbols; capped to keep the overlay fast).
        # MS-style vector text: background-colored outline + element
        # color fill, zoom-adaptive size. Drawn on top of the GL scene.
        if self._show_labels and 0 < len(self._atom_pos) <= 500:
            self._paint_atom_labels()

        # Cell corner labels O/A/B/C (VESTA convention, own font size)
        if (self._render_settings.show_cell_corners
                and self._cell_verts is not None):
            self._paint_cell_corner_labels()

        # Ghost bond preview (anchor → ghost atom, screen-space line)
        if self._ghost_bond_from is not None and self._ghost_pos is not None:
            a = self.project_to_screen(self._ghost_bond_from)
            b = self.project_to_screen(self._ghost_pos)
            if a is not None and b is not None:
                painter = QPainter(self)
                painter.setPen(QPen(QColor(255, 210, 90), 1.5, Qt.DashLine))
                painter.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))
                painter.end()

        # Rubber-band selection box (screen space, painted after GL)
        if self._rubber_rect is not None:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 210, 90), 1, Qt.DashLine))
            painter.drawRect(self._rubber_rect)
            painter.end()

        # Orientation axes indicator (bottom-right, rotates with the view)
        if self._render_settings.show_axes:
            self._paint_axes_indicator()

    def _overlay_text_color(self) -> QColor:
        """Adaptive overlay text color for dark/light backgrounds."""
        bg = self._render_settings.background_color
        luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        if luminance > 0.5:
            return QColor(30, 30, 30)
        return QColor(225, 225, 225)

    def _label_color(self, symbol: str) -> QColor:
        """Element color adjusted to keep the label readable against the
        background (paper-figure style labels).

        Elements whose luminance sits within 0.3 of the background are
        pushed away from it: dark elements on a dark background get
        brightened, light elements on a light background get darkened.
        """
        bg = self._render_settings.background_color
        bg_lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        # labels follow the atom's ACTUAL color (override → scheme → Jmol)
        base = list(self._effective_atom_color(symbol)[:3])
        lum = 0.299 * base[0] + 0.587 * base[1] + 0.114 * base[2]
        gap = 0.3  # minimum luminance distance from the background
        if lum > 1e-6:
            if lum - bg_lum < gap and bg_lum < 0.5:
                scale = (bg_lum + gap) / lum
                if scale > 1.0:
                    base = [min(1.0, c * scale) for c in base]
            elif bg_lum - lum < gap and bg_lum > 0.5:
                scale = (bg_lum - gap) / lum
                if scale < 1.0:
                    base = [max(0.0, c * scale) for c in base]
        return QColor.fromRgbF(*base)

    def _draw_halo_text(self, painter: QPainter, pos: QPointF, text: str) -> None:
        """Draw text with a background-colored halo (8 offsets).

        The halo is painted in the background color around the glyphs,
        so labels separate from the atoms behind them — the text never
        blends into a sphere or stick of a similar color.
        """
        halo = QColor.fromRgbF(*self._render_settings.background_color)
        fill = painter.pen().color()
        painter.setPen(halo)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    painter.drawText(pos + QPointF(dx, dy), text)
        painter.setPen(fill)
        painter.drawText(pos, text)

    def _measurement_pen(self) -> QPen:
        """QPen for measurement labels (user-adjustable color)."""
        return QPen(QColor.fromRgbF(*self._render_settings.measurement_color))

    def _measurement_font(self) -> QFont:
        """Font for measurement labels — fixed point size.

        Deliberately NOT zoom-scaled like the element labels: an
        annotation stays readable at any camera distance (matches the
        pre-options behavior; only the size itself is now adjustable).
        """
        font = QFont(self.font())
        font.setPointSize(self._render_settings.measurement_label_size)
        return font

    def _draw_outlined_text(
        self,
        painter: QPainter,
        pos: QPointF,
        text: str,
        fill_color: QColor,
    ) -> None:
        """MS-style label text: 2 px background-colored outline around
        the glyphs + element-colored fill (vector text, always crisp)."""
        path = QPainterPath()
        path.addText(pos.x(), pos.y(), painter.font(), text)
        halo = QColor.fromRgbF(*self._render_settings.background_color)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(halo, 2.0))
        painter.drawPath(path)
        painter.fillPath(path, fill_color)

    def _paint_atom_labels(self) -> None:
        """Element labels for the displayed atoms (≤ 500)."""
        painter = QPainter(self)
        font = QFont(self.font())
        font.setPointSizeF(
            self._render_settings.label_size * self._label_scale())
        font.setBold(True)
        painter.setFont(font)
        for i, pos in enumerate(self._atom_pos):
            s = self.project_to_screen(pos)
            if s is None:
                continue
            self._draw_outlined_text(
                painter, QPointF(s[0] + 6.0, s[1] - 6.0),
                self._symbols[i], self._label_color(self._symbols[i]))
        painter.end()

    def _paint_cell_corner_labels(self) -> None:
        """O/A/B/C labels at the four cell vertices (VESTA convention),
        with their own independent font size."""
        painter = QPainter(self)
        font = QFont(self.font())
        font.setPointSizeF(
            self._render_settings.corner_label_size * self._label_scale())
        font.setBold(True)
        painter.setFont(font)
        fill = self._overlay_text_color()
        for label, corner in (
            ("O", (0.0, 0.0, 0.0)),
            ("A", self._cell[0]),
            ("B", self._cell[1]),
            ("C", self._cell[2]),
        ):
            s = self.project_to_screen(np.asarray(corner, dtype=float))
            if s is None:
                continue
            self._draw_outlined_text(
                painter, QPointF(s[0] + 5.0, s[1] - 5.0), label, fill)
        painter.end()

    def _label_scale(self) -> float:
        """Zoom-adaptive label size factor.

        Labels grow when the camera zooms in and shrink when it zooms
        out, relative to the last fitted distance (square-root curve,
        clamped 0.6–2.5 so they stay readable but never overwhelm).
        """
        if self._fit_distance <= 0.0 or self._cam_distance <= 0.0:
            return 1.0
        return min(2.5, max(0.6, (self._fit_distance / self._cam_distance) ** 0.5))

    def _bond_alphas(self) -> np.ndarray | None:
        """Per-bond alpha values, or None when the scene is uniform.

        Bonds fade together with their atoms (VMD/ChimeraX convention):
        a bond's alpha is the mean of its two atoms' per-element
        opacities. None means every alpha is 1.0 and the bonds can be
        drawn in a single draw call. The global atom_opacity applies
        uniformly via the uOpacity shader uniform and needs no
        per-bond handling.
        """
        overrides = self._render_settings.atom_colors
        if not overrides:
            return None
        alphas = np.ones(len(self._bonds), dtype=np.float32)
        varying = False
        for k, b in enumerate(self._bonds):
            ai = overrides.get(self._symbols[b.i], (0.0, 0.0, 0.0, 1.0))[3]
            aj = overrides.get(self._symbols[b.j], (0.0, 0.0, 0.0, 1.0))[3]
            alphas[k] = 0.5 * (ai + aj)
            if ai < 1.0 or aj < 1.0:
                varying = True
        return alphas if varying else None

    def _paint_axes_indicator(self) -> None:
        """Small RGB orientation axes in the bottom-right corner."""
        _proj, view = self._camera_matrices()
        rot = view[:3, :3]
        origin = QPointF(self.width() - 52.0, self.height() - 52.0)
        length = 36.0
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont(self.font())
        font.setPointSize(8)
        font.setBold(True)
        for axis, color, label in (
            (0, QColor(224, 70, 70), "X"),
            (1, QColor(70, 180, 70), "Y"),
            (2, QColor(80, 110, 230), "Z"),
        ):
            world = np.zeros(3)
            world[axis] = 1.0
            # screen-space direction of the world axis (right/up rows of
            # the view rotation; screen y points down)
            sx = float(rot[0] @ world)
            sy = -float(rot[1] @ world)
            norm = math.hypot(sx, sy)
            if norm < 1e-6:
                continue  # pointing straight at/away from the camera
            dx = sx / norm * length
            dy = sy / norm * length
            pen = QPen(color, 2.0)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(origin, QPointF(origin.x() + dx, origin.y() + dy))
            painter.setPen(color)
            painter.setFont(font)
            painter.drawText(
                QPointF(origin.x() + dx * 1.25 - 4.0, origin.y() + dy * 1.25 + 3.0),
                label,
            )
        painter.end()

    def _paint_text(self, text: str) -> None:
        """Draw overlay text with QPainter (works inside paintGL)."""
        painter = QPainter(self)
        painter.setPen(self._overlay_text_color())
        font = QFont(self.font())
        font.setPointSize(11)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, text)
        painter.end()

    def resizeGL(self, w: int, h: int) -> None:
        """Viewport is recomputed in paintGL — nothing to do here."""
        pass

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if self._press_button is not None:
            # A drag/pan is already in progress — a second button must
            # NOT hijack the press state (it used to overwrite
            # _press_button/_press_pos/_dragged and leave stale overlays
            # like a stuck rubber band). The ongoing drag wins.
            event.accept()
            return
        self._press_button = event.button()
        self._press_pos = event.position()
        self._last_pos = event.position()
        self._dragged = False
        if (event.button() == Qt.LeftButton and self._active_tool is not None
                and self._active_tool.mouse_press(event, event.position())):
            self._tool_active = True
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._press_button is None or self._last_pos is None:
            return
        pos = event.position()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()

        moved = (pos - self._press_pos).manhattanLength()
        if moved > CLICK_DRAG_PX:
            self._dragged = True

        if self._tool_active and self._press_button == Qt.LeftButton:
            self._active_tool.mouse_move(event, pos)
        elif self._press_button == Qt.LeftButton:
            self.orbit(dx, dy)
        elif self._press_button in (Qt.RightButton, Qt.MiddleButton):
            self.pan(dx, dy)

        self._last_pos = pos
        if self._dragged:
            self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != self._press_button:
            # Release of a button that never started the gesture (e.g.
            # the right button pressed mid-drag) — must not clear the
            # drag state while the original button is still held.
            event.accept()
            return
        if self._tool_active and event.button() == Qt.LeftButton:
            self._active_tool.mouse_release(event, event.position())
            self._tool_active = False
        elif (event.button() == Qt.LeftButton and not self._dragged
                and self._atoms is not None and len(self._atoms) > 0):
            # Standalone fallback (no tool consumed the press): click
            # selects — legacy behavior preserved for bare viewports.
            index = self._pick_atom(event.position())
            if index is not None:
                self.set_highlight({index})
                self.atom_clicked.emit(index)
            else:
                self.set_highlight(None)
                self.background_clicked.emit()
        self._press_button = None
        self._last_pos = None
        event.accept()

    def keyPressEvent(self, event) -> None:
        """Forward keys to the active tool (Esc cancels, etc.)."""
        if (self._active_tool is not None
                and self._active_tool.key_press(event)):
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 0.9 ** (delta / 120.0)
        # Free zoom (VESTA-style — user decision 2026-08-13): no wall
        # or atom limits; the camera may pass through the cell frame
        # and atoms. The tiny floor only keeps the eye distinct from
        # the target point (the view math is singular at distance 0).
        self._cam_distance = max(self._cam_distance * factor, 1e-3)
        self.update()
        event.accept()

    # ------------------------------------------------------------------
    # Camera gestures (shared by the default navigation and tools)
    # ------------------------------------------------------------------

    def orbit(self, dx: float, dy: float) -> None:
        """Arcball orbit: drag right → around the camera up axis;
        drag up → around the camera right axis.

        The view direction itself is rotated (no elevation clamp), so a
        c-axis top-down view — where the camera sits near the
        elevation pole — rotates freely around axes perpendicular to c
        instead of stalling at the old ±89.9° clamp.
        """
        _proj, view = self._camera_matrices()
        rot = view[:3, :3]
        right = rot[0]
        up_axis = rot[1]
        forward = -rot[2]  # view direction (camera → center)

        R = (rotation_matrix(right, -dy * 0.5)
             @ rotation_matrix(up_axis, -dx * 0.5))
        v = R @ forward
        v /= np.linalg.norm(v)

        # The camera azimuth/elevation parametrize the EYE direction
        # (eye = center + d·direction), which is the OPPOSITE of the
        # rotated look direction.
        self._cam_azimuth = math.degrees(math.atan2(-v[0], -v[2]))
        self._cam_elevation = math.degrees(
            math.asin(float(np.clip(-v[1], -1.0, 1.0))))
        # The camera up rotates in the SAME frame as the view direction
        # (trackball-style) — recomputing it as "world up projected ⊥ v"
        # flips ~180° per frame near the pole (v ≈ ±y), which made
        # drags spin the model crazily.
        up = R @ up_axis
        up = up - float(up @ v) * v
        norm = float(np.linalg.norm(up))
        if norm < 1e-6:  # fully degenerate only
            up = np.cross(v, right)
            norm = float(np.linalg.norm(up))
        if norm > 1e-6:
            up = up / norm
        self._cam_up = up
        self.update()

    def pan(self, dx: float, dy: float) -> None:
        """Pan in the camera plane (MS-style right-drag pan).

        Camera right/up in world space are rows 0/1 of the view matrix
        rotation part. Ortho: world-units-per-pixel is constant at
        2·distance / logical height.
        """
        _proj, view = self._camera_matrices()
        rot = view[:3, :3]
        right = rot[0, :]
        up = rot[1, :]
        wpp = 2.0 * self._cam_distance / max(self.height(), 1)
        self._cam_center = self._cam_center + (-dx) * wpp * right + dy * wpp * up
        self.update()

    # ------------------------------------------------------------------
    # Interaction modes & tool API
    # ------------------------------------------------------------------

    def set_mode(self, mode: ToolMode) -> None:
        """Switch the active interaction mode (toolbar QActionGroup)."""
        self._mode = mode
        # Cancel the PREVIOUS tool first — switching mid-drag must not
        # leak its transient state (previews, rubber band, pending picks)
        if self._active_tool is not None:
            self._active_tool.cancel()
        self._tool_active = False
        if mode not in self._tool_instances:
            self._tool_instances[mode] = make_tool(mode, self)
        self._active_tool = self._tool_instances[mode]
        self.setCursor(self._active_tool.cursor())
        self.mode_changed.emit(mode)

    def mode(self) -> ToolMode:
        """The currently active interaction mode."""
        return self._mode

    def cancel_active_tool(self) -> None:
        """Reset the active tool's transient state.

        Called when the structure changes so no tool keeps stale atom
        indices or preview state across edits. Also clears screen-space
        overlays (rubber band) as a safety net.
        """
        self._tool_active = False
        self._rubber_rect = None
        if self._active_tool is not None:
            self._active_tool.cancel()
        self.update()

    def pick(self, pos):
        """Atom or bond under the cursor: ("atom", i) | ("bond", k) | None."""
        return self._pick(pos)

    def screen_to_world(self, pos) -> np.ndarray:
        """World point on the camera plane through _cam_center, under pos.

        Ortho unprojection from widget (logical) pixels: the visible
        half-height equals the camera distance, and the camera-plane
        basis is right/up (rows 0/1 of the view rotation).
        """
        _proj, view = self._camera_matrices()
        rot = view[:3, :3]
        right = rot[0, :]
        up = rot[1, :]
        wpp = 2.0 * self._cam_distance / max(self.height(), 1)
        px = (pos.x() - self.width() / 2.0) * wpp
        py = (pos.y() - self.height() / 2.0) * wpp
        return self._cam_center + px * right - py * up

    def project_to_screen(self, world) -> np.ndarray | None:
        """Project a world point to widget (logical) pixels; None if behind."""
        proj, view = self._camera_matrices()
        p = np.asarray(world, dtype=float)
        # Ortho: clip.w is always 1, so the perspective "behind" guard
        # never fires — test the point against the camera plane
        # explicitly. Free zoom lets the camera pass through atoms;
        # without this, labels/measurements of occluded atoms are drawn
        # at mirror positions.
        rot = view[:3, :3]
        eye = -(rot.T @ view[:3, 3])
        forward_w = -view[2, :3]
        if float((p - eye) @ forward_w) <= 0.0:
            return None
        clip = proj @ view @ np.append(p, 1.0)
        w = clip[3]
        if w <= 0.0:
            return None
        ndc = clip[:3] / w
        return np.array([
            (ndc[0] + 1.0) / 2.0 * self.width(),
            (1.0 - (ndc[1] + 1.0) / 2.0) * self.height(),
        ])

    def preview_atom_position(self, index: int, position: np.ndarray) -> None:
        """Drag fast path: move the displayed atom without rebuilding bonds.

        Atom spheres are drawn per-frame from _atom_pos with per-atom
        model matrices, so this needs no buffer upload and no bond
        recompute — the model is untouched until the drag commits.
        """
        if 0 <= index < len(self._atom_pos):
            self._atom_pos[index] = np.asarray(position, dtype=np.float32)
            self.update()

    def preview_atom_positions(self, indices, positions) -> None:
        """Drag fast path for several atoms (single repaint).

        Marks the bond geometry dirty so the sticks are re-baked from
        the preview positions on the next frame — bonds follow the
        atoms during the drag.
        """
        for idx, pos in zip(indices, positions):
            if 0 <= idx < len(self._atom_pos):
                self._atom_pos[idx] = np.asarray(pos, dtype=np.float32)
        self._preview_dirty = True
        self.update()

    def preview_ghost_atom(self, position: np.ndarray | None) -> None:
        """Show/hide the semi-transparent ghost atom (add-atom drag)."""
        self._ghost_pos = None if position is None else np.asarray(position, dtype=float)
        self.update()

    def preview_ghost_bond(self, anchor_position: np.ndarray | None) -> None:
        """Show/hide the ghost bond line from the anchor to the ghost atom."""
        self._ghost_bond_from = (
            None if anchor_position is None
            else np.asarray(anchor_position, dtype=float)
        )
        self.update()

    def set_measurements(self, payload) -> None:
        """Set measurement annotations: [(kind, indices, value text), ...].

        Stores the line endpoint pairs (atom indices). The dashed
        GL_LINES geometry is rebuilt EVERY FRAME from the live atom
        positions with a screen-space dash pattern (see
        _rebuild_meas_dashes) — dashes stay visible at any zoom and the
        lines track atom drags like the labels do.

        Settled with the user (2026-08-13): measurements are between
        the DISPLAYED atoms — direct Cartesian vectors, NO minimum
        image, so a periodic distance never crosses the cell boundary.
        """
        self._measurements = payload
        pairs: list[tuple[int, int]] = []
        for kind, idx, _text in payload:
            if kind == "distance":
                pairs.append((idx[0], idx[1]))
            elif kind == "angle":
                pairs.append((idx[1], idx[0]))
                pairs.append((idx[1], idx[2]))
            else:
                pairs.append((idx[0], idx[1]))
                pairs.append((idx[1], idx[2]))
                pairs.append((idx[2], idx[3]))
        self._meas_pairs = pairs
        self._rebuild_meas_dashes()
        self.update()

    def _rebuild_meas_dashes(self) -> None:
        """Rebuild the dashed measurement lines (screen-space dash length).

        Dash length = 10 px on screen (gap 5 px), converted to world
        units with the current world-units-per-pixel factor, so the
        dashes stay crisp and distinct at ANY zoom (fixed world-space
        dashes collapsed into solid-looking lines when zoomed out).
        """
        if not self._meas_pairs:
            self._meas_verts = np.zeros((0, 3), dtype=np.float32)
            return
        wpp = 2.0 * self._cam_distance / max(self.height(), 1)
        dash = 10.0 * wpp
        gap = 5.0 * wpp
        verts: list[np.ndarray] = []
        for a, b in self._meas_pairs:
            start = np.asarray(self._atom_pos[a], dtype=float)
            end = np.asarray(self._atom_pos[b], dtype=float)
            total = float(np.linalg.norm(end - start))
            if total < 1e-12:
                continue
            direction = (end - start) / total
            t = 0.0
            while t < total:
                t1 = min(t + dash, total)
                verts.append(start + direction * t)
                verts.append(start + direction * t1)
                t = t1 + gap
        self._meas_verts = (np.asarray(verts, dtype=np.float32)
                            if verts else np.zeros((0, 3), dtype=np.float32))

    def _upload_bond_buffer(self) -> None:
        """Upload the (re-baked) bond geometry into the GL buffer."""
        if self._gl is None:
            return
        self._bond_vao.bind()
        self._bond_vbo.bind()
        self._bond_vbo.allocate(self._bond_verts.tobytes(), self._bond_verts.nbytes)
        self._bond_n_verts = len(self._bond_verts)
        self._bond_vao.release()

    def _upload_meas_buffer(self) -> None:
        """Upload the (per-frame) measurement dash geometry."""
        if self._gl is None:
            return
        self._meas_vao.bind()
        self._meas_vbo.bind()
        self._meas_vbo.allocate(self._meas_verts.tobytes(), self._meas_verts.nbytes)
        self._meas_vao.release()

    def _rebuild_cell_frame_verts(self) -> None:
        """Rebuild the cell frame as camera-facing screen-space quads.

        Each of the 12 edges becomes a quad strip whose width is
        cell_line_width PIXELS (converted via world-units-per-pixel),
        offset along the edge's screen-space perpendicular so the
        on-screen width is constant at every orientation. glLineWidth
        is ignored on GL core profiles, so triangle quads are the only
        portable way to draw a wide cell frame.
        """
        wpp = 2.0 * self._cam_distance / max(self.height(), 1)
        # Minimum 1.5 px: a 1-px quad falls between pixel centers at
        # most angles and MSAA resolves it as an alternating
        # solid/faint pattern (the frame looked DASHED at some angles).
        width = max(float(self._render_settings.cell_line_width), 1.5) * wpp
        edges = self._cell_verts.reshape(-1, 2, 3)
        _proj, view = self._camera_matrices()
        rot = view[:3, :3]
        right = rot[0]
        up = rot[1]
        quads: list[np.ndarray] = []
        for p0, p1 in edges:
            edge = p1 - p0
            length = float(np.linalg.norm(edge))
            if length < 1e-9:
                continue
            e = edge / length
            rx = float(right @ e)
            ry = float(up @ e)
            # screen-projected length of the edge direction: when an
            # edge points at/away from the camera it collapses to a
            # point on screen — draw a small square dot instead of a
            # zero-area quad (otherwise the edge vanishes at some
            # viewing angles).
            screen_len = math.hypot(rx, ry)
            if screen_len < 0.01:
                mid = (p0 + p1) / 2.0
                half = width / 2.0
                quads.extend([
                    mid - right * half - up * half,
                    mid + right * half - up * half,
                    mid + right * half + up * half,
                    mid - right * half - up * half,
                    mid + right * half + up * half,
                    mid - right * half + up * half,
                ])
                continue
            # Offset along the edge's screen-space PERPENDICULAR
            # (-ry, rx)/screen_len: a unit vector whose screen
            # projection always has length 1, so the quad stays
            # exactly `width` px wide at every orientation. The old
            # offset (screen-right projected onto the plane ⊥ e)
            # thinned toward zero as the edge turned parallel to the
            # screen-right axis — the 4 parallel edges of one cell
            # direction vanished together, and near that angle the
            # sub-pixel width MSAA-resolved into the dashed pattern.
            off = (-ry * right + rx * up) / screen_len * (width / 2.0)
            a, b = p0 + off, p0 - off
            c, d = p1 + off, p1 - off
            quads.extend([a, b, c, b, d, c])
        self._cell_frame_verts = (np.asarray(quads, dtype=np.float32)
                                  if quads
                                  else np.zeros((0, 3), dtype=np.float32))

    def _upload_cell_buffer(self) -> None:
        """Upload the (per-frame) cell frame quad geometry."""
        if self._gl is None:
            return
        self._cell_vao.bind()
        self._cell_vbo.bind()
        self._cell_vbo.allocate(
            self._cell_frame_verts.tobytes(), self._cell_frame_verts.nbytes)
        self._cell_vao.release()

    def set_rubber_band(self, rect) -> None:
        """Show/hide the box-selection rubber band (QRectF or None)."""
        self._rubber_rect = rect
        self.update()

    # ------------------------------------------------------------------
    # Picking — ID-color framebuffer readback
    # ------------------------------------------------------------------

    def _pick(self, pos) -> tuple[str, int] | None:
        """Atom or bond under the cursor, or None.

        Renders the scene into an offscreen framebuffer with every atom
        and bond colored by a unique ID (cell omitted), then reads back
        the pixel under the cursor. Uses the same camera matrices as the
        on-screen pass, so the hit is exactly what the user sees. Bonds
        are drawn first and atoms last, so atoms win depth ties at the
        joints.

        Args:
            pos: Cursor position in logical widget pixels (y-down).

        Returns:
            ("atom", index) / ("bond", bond-list index), or None.
        """
        if (not self._gl_ready or self._gl_failed
                or self._atoms is None or len(self._atom_pos) == 0):
            return None

        dpr = self.devicePixelRatioF()
        fb_w = max(int(self.width() * dpr), 1)
        fb_h = max(int(self.height() * dpr), 1)
        if self._pick_fbo is None or self._pick_fbo_size != (fb_w, fb_h):
            fbo_fmt = QOpenGLFramebufferObjectFormat()
            fbo_fmt.setAttachment(QOpenGLFramebufferObject.Depth)
            self._pick_fbo = QOpenGLFramebufferObject(fb_w, fb_h, fbo_fmt)
            self._pick_fbo_size = (fb_w, fb_h)

        px = int(pos.x() * dpr)
        py = fb_h - 1 - int(pos.y() * dpr)  # GL origin is bottom-left
        if not (0 <= px < fb_w and 0 <= py < fb_h):
            return None

        gl = self._gl
        proj, view = self._camera_matrices()

        # GL work outside paintGL requires an explicit makeCurrent —
        # QOpenGLWidget's context is only current during paintGL.
        self.makeCurrent()
        if self._data_dirty:
            self._upload_scene_buffers()

        self._pick_fbo.bind()
        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        gl.glEnable(GL_DEPTH_TEST)

        # ID pass uses the UNLIT flat program: the lit sphere shader
        # multiplies colors by a lighting term < 1, which would truncate
        # the packed ID to zero in the 8-bit readback.
        n = len(self._atom_pos)
        self._flat_prog.bind()

        def _id_color(iid: int) -> tuple[float, float, float, float]:
            return (
                float(iid & 0xFF) / 255.0,
                float((iid >> 8) & 0xFF) / 255.0,
                float((iid >> 16) & 0xFF) / 255.0,
                1.0,
            )

        # Bonds first (ID = n + k + 1), atom spheres last — atoms win
        # depth ties at the joints.
        if self._bond_ranges:
            self._bond_vao.bind()
            self._flat_prog.setUniformValue("uMVP", _to_qmatrix(proj @ view))
            for k, (first, count) in enumerate(self._bond_ranges):
                if count == 0:
                    continue
                self._flat_prog.setUniformValue("uColor", *_id_color(n + k + 1))
                gl.glDrawArrays(GL_TRIANGLES, first, count)
            self._bond_vao.release()

        self._unit_vao.bind()
        for i in range(n):
            scale = float(self._atom_radius[i])
            # Draw at the edge-sphere scale (1.04×) so picking matches the
            # visible silhouette — clicks on the dark outline ring of an
            # atom must hit it, not fall through to the background.
            mvp = proj @ view @ _sphere_model(self._atom_pos[i], scale * EDGE_SCALE)
            self._flat_prog.setUniformValue("uMVP", _to_qmatrix(mvp))
            self._flat_prog.setUniformValue("uColor", *_id_color(i + 1))
            gl.glDrawElements(GL_TRIANGLES, self._unit_n_indices,
                              GL_UNSIGNED_INT, VoidPtr(0))
        self._unit_vao.release()
        self._flat_prog.release()

        buf = np.zeros(4, dtype=np.uint8)
        gl.glReadPixels(px, py, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, buf)
        self._pick_fbo.release()
        self.doneCurrent()

        iid = int(buf[0]) | (int(buf[1]) << 8) | (int(buf[2]) << 16)
        if iid == 0:
            return None
        if iid <= n:
            return ("atom", iid - 1)
        return ("bond", iid - n - 1)

    def _pick_atom(self, pos) -> int | None:
        """Atom index under the cursor, or None (legacy wrapper)."""
        hit = self._pick(pos)
        return hit[1] if hit is not None and hit[0] == "atom" else None
