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
from ase.geometry import get_distances

from vaspen.utils.logger import logger

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QMatrix4x4,
    QPainter,
    QSurfaceFormat,
)
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat,
    QOpenGLShader,
    QOpenGLShaderProgram,
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

# Ball-and-stick scale factors
SPHERE_SCALE = 0.60   # atom sphere radius = covalent radius × 0.60
EDGE_SCALE = 1.04     # dark outline sphere drawn slightly larger behind
BOND_RADIUS = 0.12    # bond cylinder radius (Angstrom)
BOND_TOLERANCE = 1.07 # covalent bond if d < (r1 + r2) × tolerance
BOND_MAX_LENGTH = 3.0 # global maximum bond / contact length
MAX_BOND_RADIUS = 1.35  # cap covalent radius used for bond detection
                        # (heavy metals like Ba=2.15 would otherwise bond
                        # to everything)

# Pauling electronegativity for non-metal detection (ionic-contact rule).
# Pairs metal↔non-metal within BOND_MAX_LENGTH are drawn as contacts
# (e.g. Ba–O 2.83 Å in perovskites), while metal–metal pairs are not.
ELECTRONEGATIVITY: dict[str, float] = {
    "H": 2.20, "He": 0.0, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44,
    "F": 3.98, "Ne": 0.0, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16,
    "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3.00,
    "Te": 2.10, "I": 2.66, "Xe": 2.60, "At": 2.20,
}
EN_NONMETAL_THRESHOLD = 2.0

BACKGROUND_COLOR = (0.118, 0.118, 0.141)  # #1e1e24 dark

# Mouse-drag threshold: movement below this is a click, not a drag
CLICK_DRAG_PX = 4.0


def _is_nonmetal(symbol: str) -> bool:
    """True if the element is a non-metal (Pauling EN ≥ 2.0).

    Unknown elements default to metallic behavior (EN < 2.0).
    """
    return ELECTRONEGATIVITY.get(symbol, 1.5) >= EN_NONMETAL_THRESHOLD


def element_color(symbol: str) -> tuple[float, float, float]:
    """Return the Jmol RGB color (0-1) for an element symbol."""
    return JMOL_COLORS.get(symbol, DEFAULT_ATOM_COLOR)


def _edge_color(color: tuple[float, float, float]) -> tuple[float, float, float]:
    """Derive a darker outline color from a base atom color."""
    return tuple(c * 0.45 for c in color)


def _mic_draw_vector(
    i: int,
    j: int,
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: tuple[bool, bool, bool],
) -> np.ndarray:
    """Minimum-image vector from atom i to atom j for bond DRAWING.

    Enumerates all 27 lattice translations and picks the shortest.
    On exact half-cell ties (e.g. Ti–O = 2.0 Å in a 4.0 Å cell), both
    images are equally short — prefer the one whose endpoint lies
    INSIDE the home cell, so the bond connects to the displayed atom
    instead of its periodic image outside the cell (which reads as a
    bond "to the boundary").
    """
    d = positions[j] - positions[i]
    inv_cell = np.linalg.inv(cell)

    best_v: np.ndarray | None = None
    best_len = float("inf")
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            for n3 in (-1, 0, 1):
                # Only allow shifts along periodic directions
                if (n1 != 0 and not pbc[0]) or \
                   (n2 != 0 and not pbc[1]) or \
                   (n3 != 0 and not pbc[2]):
                    continue
                shift = n1 * cell[0] + n2 * cell[1] + n3 * cell[2]
                v = d + shift
                length = float(np.linalg.norm(v))
                if length < best_len - 1e-9:
                    best_v, best_len = v, length
                elif abs(length - best_len) <= 1e-9:
                    # Tie — prefer endpoint inside the home cell
                    frac = (positions[i] + v) @ inv_cell
                    if np.all((frac >= 0.0) & (frac < 1.0)):
                        best_v, best_len = v, length
    return np.asarray(best_v, dtype=float)


def _find_bonds(
    positions: np.ndarray,
    symbols: list[str],
    cell: np.ndarray,
    pbc: tuple[bool, bool, bool],
) -> list[tuple[int, int, np.ndarray]]:
    """Find covalent bonds between atoms, respecting periodic boundaries.

    Args:
        positions: Cartesian positions (N×3).
        symbols: Element symbols (N).
        cell: 3×3 cell matrix.
        pbc: Periodic boundary flags.

    Returns:
        List of (i, j, vec) tuples where vec is the draw vector from
        atom i to atom j (minimum-image with in-cell tie-breaking).
    """
    n = len(positions)
    if n == 0:
        return []

    radii = np.array([
        min(covalent_radii[atomic_numbers[s]], MAX_BOND_RADIUS)
        for s in symbols
    ])
    nonmetals = [_is_nonmetal(s) for s in symbols]

    # Pairwise distances with minimum-image convention.
    # NOTE: ASE ≥3.29 returns (vectors (N,M,3), distances (N,M)) — order
    # swapped relative to older versions.
    _vecs, dists = get_distances(positions, positions, cell=cell, pbc=pbc)

    bonds: list[tuple[int, int, np.ndarray]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if d <= 0.3 or d > BOND_MAX_LENGTH:
                continue
            # Rule 1: covalent bond
            covalent = d < (radii[i] + radii[j]) * BOND_TOLERANCE
            # Rule 2: ionic contact between metal and non-metal
            ionic = nonmetals[i] != nonmetals[j]
            if covalent or ionic:
                vec = _mic_draw_vector(i, j, positions, cell, pbc)
                bonds.append((i, j, vec))
    return bonds


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
) -> np.ndarray:
    """Bake bond cylinders into world-space triangle vertices (M×3).

    Args:
        starts: Segment start points (K×3) — atom positions.
        ends: Segment end points (K×3) — atom + minimum-image vector.
        radius: Cylinder radius in Angstrom.

    Returns:
        Flat vertex array for glDrawArrays(GL_TRIANGLES). No normals —
        bonds are rendered unlit. Ends are open (buried inside the
        atom spheres, so the joint is seamless).
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

        angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        ring_s = s + radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
        ring_e = e + radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
        for k in range(segments):
            k2 = (k + 1) % segments
            a0, a1 = ring_s[k], ring_s[k2]
            b0, b1 = ring_e[k], ring_e[k2]
            verts.extend([a0, b0, b1, a0, b1, a1])
    if not verts:
        return np.zeros((0, 3), dtype=np.float32)
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


# ----------------------------------------------------------------------
# Shaders (GLSL 3.30 core)
# ----------------------------------------------------------------------

_SPHERE_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
uniform mat4 uMVP;
out vec3 vNormal;
void main() {
    vNormal = aNormal;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_SPHERE_FRAG = """
#version 330 core
in vec3 vNormal;
uniform vec4 uColor;
out vec4 fragColor;
void main() {
    vec3 n = normalize(vNormal);
    float diff = max(dot(n, normalize(vec3(0.35, 0.55, 0.75))), 0.0);
    float shade = 0.45 + 0.60 * diff;
    fragColor = vec4(uColor.rgb * shade, uColor.a);
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


class Viewport3D(QOpenGLWidget):
    """3D ball-and-stick structure viewport (Qt native OpenGL).

    Public API mirrors the previous VisPy implementation so the main
    window needs no changes.
    """

    # Signals
    atom_clicked = Signal(int)      # user clicked an atom (index)
    background_clicked = Signal()   # user clicked empty space

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Scene data (numpy, no GL — safe to set before initializeGL)
        self._atoms: Atoms | None = None
        self._atom_pos = np.zeros((0, 3), dtype=np.float32)
        self._atom_radius = np.zeros(0, dtype=np.float32)
        self._atom_color = np.zeros((0, 4), dtype=np.float32)
        self._atom_edge = np.zeros((0, 4), dtype=np.float32)
        self._bond_verts = np.zeros((0, 3), dtype=np.float32)
        self._cell_verts: np.ndarray | None = None
        self._selected_index: int | None = None
        self._data_dirty = False

        # Camera state (orbit camera, perspective projection)
        self._cam_center = np.zeros(3, dtype=np.float64)
        self._cam_distance = 10.0
        self._cam_azimuth = 45.0
        self._cam_elevation = 30.0
        self._cam_up = np.array([0.0, 1.0, 0.0])  # world up for the view
        self._fit_radius = 5.0
        self._view_fitted = False  # True once _fit_camera has run
        self._has_cell = False  # set in set_structure; controls fit view
        self._cell = np.eye(3, dtype=np.float64)

        # GL resources
        self._gl = None
        self._gl_ready = False
        self._gl_failed = False
        self._sphere_prog: QOpenGLShaderProgram | None = None
        self._flat_prog: QOpenGLShaderProgram | None = None
        self._unit_vao: QOpenGLVertexArrayObject | None = None
        self._unit_vbo: QOpenGLBuffer | None = None
        self._unit_ibo: QOpenGLBuffer | None = None
        self._unit_n_indices = 0
        self._bond_vao: QOpenGLVertexArrayObject | None = None
        self._bond_vbo: QOpenGLBuffer | None = None
        self._bond_n_verts = 0
        self._cell_vao: QOpenGLVertexArrayObject | None = None
        self._cell_vbo: QOpenGLBuffer | None = None
        self._pick_fbo: QOpenGLFramebufferObject | None = None
        self._pick_fbo_size: tuple[int, int] = (0, 0)

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

    def set_structure(self, atoms: Atoms | None, reset_view: bool = True) -> None:
        """Replace the displayed structure and re-render.

        Args:
            atoms: ASE Atoms to display, or None to clear the view.
            reset_view: If True, fit the camera to the structure
                (default, used when opening files). False preserves the
                current camera (used for in-place edits such as surface
                cuts and supercells).
        """
        self._atoms = atoms
        self._selected_index = None

        if atoms is None or len(atoms) == 0:
            self._atom_pos = np.zeros((0, 3), dtype=np.float32)
            self._atom_radius = np.zeros(0, dtype=np.float32)
            self._atom_color = np.zeros((0, 4), dtype=np.float32)
            self._atom_edge = np.zeros((0, 4), dtype=np.float32)
            self._bond_verts = np.zeros((0, 3), dtype=np.float32)
            self._cell_verts = None
        else:
            positions = np.asarray(atoms.get_positions(), dtype=np.float32)
            symbols = list(atoms.get_chemical_symbols())
            cell = atoms.get_cell().array if atoms.get_cell().rank == 3 else np.eye(3)
            pbc = tuple(atoms.get_pbc()) if atoms.get_pbc().any() else (False, False, False)

            radii = []
            colors = []
            edges = []
            for sym in symbols:
                z = atomic_numbers.get(sym, 0)
                r = float(covalent_radii[z] if 0 < z < len(covalent_radii) else 0.77)
                base = element_color(sym)
                radii.append(r * SPHERE_SCALE)
                colors.append((base[0], base[1], base[2], 1.0))
                e = _edge_color(base)
                edges.append((e[0], e[1], e[2], 1.0))

            self._atom_pos = positions
            self._atom_radius = np.asarray(radii, dtype=np.float32)
            self._atom_color = np.asarray(colors, dtype=np.float32)
            self._atom_edge = np.asarray(edges, dtype=np.float32)

            bonds = _find_bonds(positions, symbols, cell, pbc)
            if bonds:
                starts = np.asarray([positions[i] for i, _j, _v in bonds])
                ends = np.asarray([positions[i] + v for i, _j, v in bonds])
                self._bond_verts = _cylinder_verts(starts, ends, BOND_RADIUS)
            else:
                self._bond_verts = np.zeros((0, 3), dtype=np.float32)

            self._cell_verts = _cell_edges(cell) if any(pbc) else None

            self._has_cell = any(pbc)
            self._cell = np.asarray(cell, dtype=np.float64)
            if reset_view or not self._view_fitted:
                self._fit_camera()

        self._data_dirty = True
        self.update()

    def highlight_atom(self, index: int | None) -> None:
        """Highlight the atom at index (None clears the highlight)."""
        self._selected_index = index
        self.update()

    def reset_view(self) -> None:
        """Reset the camera to fit the current structure."""
        if self._atoms is not None and len(self._atoms) > 0:
            self._fit_camera()
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
        centroid) — the half-extent is NOT enough: corner atoms of a
        cubic cell sit at extent/2 × √3 from the center and would fall
        outside the view.

        Orientation: periodic structures default to a top-down view
        along the crystallographic c axis (c out of the screen, b up,
        a to the left — the VESTA convention); non-periodic molecules
        keep the generic isometric 45°/30° view.
        """
        positions = self._atom_pos
        if len(positions) == 0:
            return
        self._cam_center = positions.mean(axis=0).astype(np.float64)
        radii = np.linalg.norm(positions - self._cam_center, axis=1)
        self._fit_radius = (
            max(float(radii.max()), 1.0)
            + float(self._atom_radius.max())
            + 0.5
        )
        # Ortho framing: the visible half-height IS the camera distance,
        # so d = 1.35×R frames the bounding sphere with a 35% margin.
        self._cam_distance = max(self._fit_radius * 1.35, 1.0)
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
        right = np.cross(forward, self._cam_up)
        if np.linalg.norm(right) < 1e-9:
            right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        view = _look_at(eye, self._cam_center, up)

        w = max(self.width(), 1)
        h = max(self.height(), 1)
        aspect = w / h
        half_h = self._cam_distance
        half_w = half_h * aspect
        r = self._fit_radius
        # Orthographic projection (crystallography convention — the
        # user's preferred default; perspective distorts parallel
        # lattice edges). Near plane fixed at 0.01 so geometry only
        # clips once it is genuinely behind the camera.
        near = 0.01
        far = self._cam_distance + r + 5.0
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
            self._unit_vao.release()

            # Bond / cell buffers (content uploaded on demand)
            self._bond_vao, self._bond_vbo, _ = self._make_mesh(
                np.zeros((0, 3), dtype=np.float32), None
            )
            self._bond_vao.bind()
            loc_b = self._flat_prog.attributeLocation("aPos")
            self._flat_prog.enableAttributeArray(loc_b)
            self._flat_prog.setAttributeBuffer(loc_b, GL_FLOAT, 0, 3, 12)
            self._bond_vao.release()

            self._cell_vao, self._cell_vbo, _ = self._make_mesh(
                np.zeros((0, 3), dtype=np.float32), None
            )
            self._cell_vao.bind()
            loc_c = self._flat_prog.attributeLocation("aPos")
            self._flat_prog.enableAttributeArray(loc_c)
            self._flat_prog.setAttributeBuffer(loc_c, GL_FLOAT, 0, 3, 12)
            self._cell_vao.release()

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
        if self._cell_verts is not None:
            self._cell_vbo.allocate(self._cell_verts.tobytes(), self._cell_verts.nbytes)
        else:
            self._cell_vbo.allocate(b"", 0)
        self._cell_vao.release()
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
        gl.glClearColor(*BACKGROUND_COLOR, 1.0)
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        gl.glEnable(GL_DEPTH_TEST)

        if len(self._atom_pos) == 0:
            self._paint_text(self.tr(
                "Open a structure file to begin\n"
                "(File → Open, or drag-and-drop)"
            ))
            return

        if self._data_dirty:
            self._upload_scene_buffers()

        proj, view = self._camera_matrices()

        # --- Atom spheres (edge pass, then body pass) ---
        self._sphere_prog.bind()
        self._unit_vao.bind()
        for i in range(len(self._atom_pos)):
            scale = float(self._atom_radius[i])
            if i == self._selected_index:
                color = (*HIGHLIGHT_COLOR, 1.0)
                edge = _edge_color(HIGHLIGHT_COLOR)
                edge = (*edge, 1.0)
            else:
                color = tuple(float(c) for c in self._atom_color[i])
                edge = tuple(float(c) for c in self._atom_edge[i])

            pos = self._atom_pos[i]
            mvp = proj @ view @ _sphere_model(pos, scale * EDGE_SCALE)
            self._sphere_prog.setUniformValue("uMVP", _to_qmatrix(mvp))
            self._sphere_prog.setUniformValue("uColor", *edge)
            gl.glDrawElements(GL_TRIANGLES, self._unit_n_indices,
                              GL_UNSIGNED_INT, VoidPtr(0))

            mvp = proj @ view @ _sphere_model(pos, scale)
            self._sphere_prog.setUniformValue("uMVP", _to_qmatrix(mvp))
            self._sphere_prog.setUniformValue("uColor", *color)
            gl.glDrawElements(GL_TRIANGLES, self._unit_n_indices,
                              GL_UNSIGNED_INT, VoidPtr(0))
        self._unit_vao.release()

        # --- Bonds (unlit cylinders) ---
        if self._bond_n_verts > 0:
            self._flat_prog.bind()
            self._bond_vao.bind()
            self._flat_prog.setUniformValue("uMVP", _to_qmatrix(proj @ view))
            self._flat_prog.setUniformValue("uColor", 0.55, 0.55, 0.55, 1.0)
            gl.glDrawArrays(GL_TRIANGLES, 0, self._bond_n_verts)
            self._bond_vao.release()

        # --- Unit cell outline (faint blue, only when periodic) ---
        # Depth-tested like normal geometry: parts of the frame behind
        # atoms are occluded by the ball-and-stick model (user
        # preference — the frame must not float on top).
        if self._cell_verts is not None and len(self._cell_verts) > 0:
            self._flat_prog.bind()
            self._cell_vao.bind()
            self._flat_prog.setUniformValue("uMVP", _to_qmatrix(proj @ view))
            self._flat_prog.setUniformValue("uColor", 0.55, 0.65, 0.85, 0.35)
            gl.glEnable(GL_BLEND)
            gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            gl.glDrawArrays(GL_LINES, 0, len(self._cell_verts))
            gl.glDisable(GL_BLEND)
            self._cell_vao.release()

        self._sphere_prog.release()
        self._flat_prog.release()

    def _paint_text(self, text: str) -> None:
        """Draw overlay text with QPainter (works inside paintGL)."""
        painter = QPainter(self)
        painter.setPen(QColor(190, 190, 190))
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
        self._press_button = event.button()
        self._press_pos = event.position()
        self._last_pos = event.position()
        self._dragged = False
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

        dpr = self.devicePixelRatioF()
        fb_h = max(int(self.height() * dpr), 1)

        if self._press_button == Qt.LeftButton:
            # Orbit: drag right → azimuth up; drag up → elevation up
            self._cam_azimuth -= dx * 0.5
            self._cam_elevation = max(-89.9, min(89.9, self._cam_elevation + dy * 0.5))
        elif self._press_button in (Qt.RightButton, Qt.MiddleButton):
            # Pan in the camera plane (MS-style right-drag pan):
            # camera right/up in world space are rows 0/1 of the view
            # matrix rotation part. Ortho: world-units-per-pixel is
            # constant at 2·d / fb_h.
            _proj, view = self._camera_matrices()
            rot = view[:3, :3]
            right = rot[0, :]
            up = rot[1, :]
            wpp = 2.0 * self._cam_distance / fb_h
            self._cam_center = self._cam_center + (-dx) * wpp * right + dy * wpp * up

        self._last_pos = pos
        if self._dragged:
            self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if (event.button() == Qt.LeftButton and not self._dragged
                and self._atoms is not None and len(self._atoms) > 0):
            index = self._pick_atom(event.position())
            if index is not None:
                self.highlight_atom(index)
                self.atom_clicked.emit(index)
            else:
                self.highlight_atom(None)
                self.background_clicked.emit()
        self._press_button = None
        self._last_pos = None
        event.accept()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 0.9 ** (delta / 120.0)
        # Deep zoom with a direction-aware floor (VESTA-style): the
        # camera may approach the structure closely but never crosses
        # the cell wall (periodic) or an atom surface (molecules), so
        # the frame never partially disappears and the view never
        # passes through atoms.
        self._cam_distance = max(
            min(self._cam_distance * factor, 500.0),
            self._zoom_limit(),
        )
        self.update()
        event.accept()

    def _zoom_limit(self) -> float:
        """Minimum camera distance along the current view direction.

        Periodic structures: the front cell wall (center is inside the
        box) minus a small margin — the camera stops just outside the
        cell, so the frame stays fully visible and atoms are never
        entered. Molecules (no cell): the nearest atom surface
        intersecting the view ray, with a bounding-sphere fallback.
        """
        az = math.radians(self._cam_azimuth)
        el = math.radians(self._cam_elevation)
        d = np.array([
            math.cos(el) * math.sin(az),
            math.sin(el),
            math.cos(el) * math.cos(az),
        ])
        center = self._cam_center
        floor = 0.05

        if self._has_cell:
            a, b, c = self._cell
            corners = np.array([
                [0, 0, 0], a, b, c, a + b, a + c, b + c, a + b + c,
            ], dtype=np.float64)
            box_min = corners.min(axis=0)
            box_max = corners.max(axis=0)
            if np.all((center >= box_min) & (center <= box_max)):
                # Distance from center to the front wall along d.
                t_front = float("inf")
                for k in range(3):
                    if abs(d[k]) < 1e-9:
                        continue
                    t1 = (box_min[k] - center[k]) / d[k]
                    t2 = (box_max[k] - center[k]) / d[k]
                    t_front = min(t_front, max(t1, t2))
                return max(t_front - 0.02, floor)

        # Molecule: nearest atom surface intersecting the view ray
        # (the camera approaches from outside, so the LARGEST hit
        # distance is the first surface it would touch).
        limit = floor
        for i in range(len(self._atom_pos)):
            o = self._atom_pos[i] - center
            t = float(o @ d)
            lat2 = float(o @ o) - t * t
            r = float(self._atom_radius[i])
            if lat2 < r * r:
                hit = t + math.sqrt(r * r - lat2)
                if hit > limit:
                    limit = hit
        if limit == floor and len(self._atom_pos) > 0:
            limit = self._fit_radius * 0.3  # no atom on the ray — keep a sane floor
        return limit + 0.02

    # ------------------------------------------------------------------
    # Picking — ID-color framebuffer readback
    # ------------------------------------------------------------------

    def _pick_atom(self, pos) -> int | None:
        """Atom under the cursor, or None.

        Renders the scene into an offscreen framebuffer with every atom
        colored by its unique ID (bonds and cell omitted), then reads
        back the pixel under the cursor. Uses the same camera matrices
        as the on-screen pass, so the hit is exactly what the user sees.

        Args:
            pos: Cursor position in logical widget pixels (y-down).

        Returns:
            Atom index, or None for empty space / bond / cell.
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

        self._pick_fbo.bind()
        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        gl.glEnable(GL_DEPTH_TEST)

        # ID pass uses the UNLIT flat program: the lit sphere shader
        # multiplies colors by a lighting term < 1, which would truncate
        # the packed ID to zero in the 8-bit readback.
        self._flat_prog.bind()
        self._unit_vao.bind()
        for i in range(len(self._atom_pos)):
            scale = float(self._atom_radius[i])
            iid = i + 1
            color = (
                float(iid & 0xFF) / 255.0,
                float((iid >> 8) & 0xFF) / 255.0,
                float((iid >> 16) & 0xFF) / 255.0,
                1.0,
            )
            # Draw at the edge-sphere scale (1.04×) so picking matches the
            # visible silhouette — clicks on the dark outline ring of an
            # atom must hit it, not fall through to the background.
            mvp = proj @ view @ _sphere_model(self._atom_pos[i], scale * EDGE_SCALE)
            self._flat_prog.setUniformValue("uMVP", _to_qmatrix(mvp))
            self._flat_prog.setUniformValue("uColor", *color)
            gl.glDrawElements(GL_TRIANGLES, self._unit_n_indices,
                              GL_UNSIGNED_INT, VoidPtr(0))
        self._unit_vao.release()
        self._flat_prog.release()

        buf = np.zeros(4, dtype=np.uint8)
        gl.glReadPixels(px, py, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, buf)
        self._pick_fbo.release()
        self.doneCurrent()

        iid = int(buf[0]) | (int(buf[1]) << 8) | (int(buf[2]) << 16)
        return (iid - 1) if iid > 0 else None
