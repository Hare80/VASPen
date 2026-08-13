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
from vaspen.utils.logger import logger

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QMatrix4x4,
    QPainter,
    QPen,
    QSurfaceFormat,
)
from vaspen.ui.tools import ToolMode, make_tool
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

# Ball-and-stick scale factors
SPHERE_SCALE = 0.60   # atom sphere radius = covalent radius × 0.60
EDGE_SCALE = 1.04     # dark outline sphere drawn slightly larger behind
BOND_RADIUS = 0.12    # bond cylinder radius (Angstrom)

BACKGROUND_COLOR = (0.118, 0.118, 0.141)  # #1e1e24 dark

# Mouse-drag threshold: movement below this is a click, not a drag
CLICK_DRAG_PX = 4.0


def element_color(symbol: str) -> tuple[float, float, float]:
    """Return the Jmol RGB color (0-1) for an element symbol."""
    return JMOL_COLORS.get(symbol, DEFAULT_ATOM_COLOR)


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


def _edge_color(color: tuple[float, float, float]) -> tuple[float, float, float]:
    """Derive a darker outline color from a base atom color."""
    return tuple(c * 0.45 for c in color)


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
) -> np.ndarray:
    """Bake bond cylinders into world-space triangle vertices (M×3).

    Args:
        starts: Segment start points (K×3) — atom positions.
        ends: Segment end points (K×3) — atom + minimum-image vector.
        radius: Cylinder radius in Angstrom.
        lateral_offset: Shift the whole cylinder perpendicular to its
            axis (used to draw double/triple bond components side by
            side).

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
        if lateral_offset:
            s = s + u * lateral_offset
            e = e + u * lateral_offset

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
uniform mat4 uView;
out vec3 vNormal;
out float vViewDepth;
void main() {
    vNormal = aNormal;
    vec4 viewPos = uView * vec4(aPos, 1.0);
    vViewDepth = -viewPos.z;  // camera looks down -z in view space
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_SPHERE_FRAG = """
#version 330 core
in vec3 vNormal;
in float vViewDepth;
uniform vec4 uColor;
uniform float uFadeStart;  // view depth (A) where fading begins
uniform float uFadeEnd;    // view depth (A) where the atom is fully faded
out vec4 fragColor;
void main() {
    vec3 n = normalize(vNormal);
    if (!gl_FrontFacing) n = -n;  // two-sided lighting: correct when viewed from inside
    float diff = max(dot(n, normalize(vec3(0.35, 0.55, 0.75))), 0.0);
    float shade = 0.45 + 0.60 * diff;
    // Near-distance fade (user decision 2026-08-13): atoms fade out as
    // the camera approaches them, so zooming in never shows a
    // cross-section disk through the sphere.
    float fade = smoothstep(uFadeEnd, uFadeStart, vViewDepth);
    fragColor = vec4(uColor.rgb * shade, uColor.a * fade);
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
    bond_clicked = Signal(int)      # user clicked a bond (bond list index)
    atoms_selected = Signal(list, str)      # indices + "replace"|"add"|"toggle"
    atom_place_requested = Signal(object, object)  # world position, anchor index|None
    atoms_moved = Signal(list, object)      # indices + new positions (K×3)
    bond_created = Signal(int, int)         # atom indices
    bond_removed = Signal(int, int)         # atom indices
    delete_requested = Signal(str, int)     # "atom"|"bond" + index
    measurement_added = Signal(str, list)   # kind + atom indices
    mode_changed = Signal(object)           # new ToolMode

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Scene data (numpy, no GL — safe to set before initializeGL)
        self._atoms: Atoms | None = None
        self._atom_pos = np.zeros((0, 3), dtype=np.float32)
        self._atom_radius = np.zeros(0, dtype=np.float32)
        self._atom_color = np.zeros((0, 4), dtype=np.float32)
        self._atom_edge = np.zeros((0, 4), dtype=np.float32)
        self._symbols: list[str] = []
        self._bond_verts = np.zeros((0, 3), dtype=np.float32)
        self._cell_verts: np.ndarray | None = None
        self._selected_indices: set[int] = set()
        self._selected_bonds: set[int] = set()  # bond list indices
        self._preview_indices: set[int] = set()
        self._bonds: list = []          # Bond objects, same order as render
        self._bond_ranges: list[tuple[int, int]] = []  # (first, count) vertex slices
        self._rubber_rect = None        # QRectF | None (box selection)
        self._ghost_pos = None          # np.ndarray | None (add-atom preview)
        self._ghost_bond_from = None    # world pos of the ghost bond's anchor
        self.current_element = "C"      # element of the atom being added
        self._measurements = []         # [(kind, indices, text), ...]
        self._meas_verts = np.zeros((0, 3), dtype=np.float32)
        # Display style (parameter pack over the existing renderer)
        self._covalent_radii = np.zeros(0, dtype=np.float32)  # unscaled
        self._style = "ball_stick"      # "ball_stick" | "cpk" | "wireframe"
        self._show_bonds = True
        self._show_cell = True
        self._show_labels = False
        self._background_color = BACKGROUND_COLOR
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

    def set_structure(
        self,
        atoms: Atoms | None,
        reset_view: bool = True,
        bonds: list | None = None,
    ) -> None:
        """Replace the displayed structure and re-render.

        Args:
            atoms: ASE Atoms to display, or None to clear the view.
            reset_view: If True, fit the camera to the structure
                (default, used when opening files). False preserves the
                current camera (used for in-place edits such as surface
                cuts and supercells).
            bonds: Persistent bond list from the model (Bond objects);
                auto-computed from connectivity when None (standalone use).
        """
        self._atoms = atoms
        self._selected_indices = set()
        self._selected_bonds = set()
        self._preview_indices = set()

        if atoms is None or len(atoms) == 0:
            self._atom_pos = np.zeros((0, 3), dtype=np.float32)
            self._atom_radius = np.zeros(0, dtype=np.float32)
            self._atom_color = np.zeros((0, 4), dtype=np.float32)
            self._atom_edge = np.zeros((0, 4), dtype=np.float32)
            self._bond_verts = np.zeros((0, 3), dtype=np.float32)
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
            edges = []
            for sym in symbols:
                z = atomic_numbers.get(sym, 0)
                r = float(covalent_radii[z] if 0 < z < len(covalent_radii) else 0.77)
                base = element_color(sym)
                radii.append(r)
                colors.append((base[0], base[1], base[2], 1.0))
                e = _edge_color(base)
                edges.append((e[0], e[1], e[2], 1.0))

            self._atom_pos = positions
            self._covalent_radii = np.asarray(radii, dtype=np.float32)
            self._apply_display()
            self._atom_color = np.asarray(colors, dtype=np.float32)
            self._atom_edge = np.asarray(edges, dtype=np.float32)

            bond_list = (bonds if bonds is not None
                         else find_bonds(positions, symbols, cell, pbc))
            self._bonds = bond_list
            # Bake per bond so every bond keeps a vertex slice of its own
            # (needed by the ID-color pick pass to draw individual bonds).
            # Ranges stay aligned with bond indices even for zero-length
            # bonds (drawing 0 verts is a no-op).
            # Bond orders: single = one cylinder; double/triple = parallel
            # cylinders offset laterally; aromatic (4) = dashed segments.
            self._bond_ranges = []
            pieces = []
            offset = 0
            for b in bond_list:
                s = np.asarray([positions[b.i]], dtype=np.float64)
                e = np.asarray([positions[b.i] + mic_vector(
                    b.i, b.j, positions, cell, pbc)], dtype=np.float64)
                bond_pieces: list[np.ndarray] = []
                if b.order == 2:
                    # thinner components so the parallel sticks stay
                    # visually separate (they would merge at 0.12 Å)
                    bond_pieces.append(_cylinder_verts(s, e, 0.05, lateral_offset=-0.09))
                    bond_pieces.append(_cylinder_verts(s, e, 0.05, lateral_offset=+0.09))
                elif b.order == 3:
                    bond_pieces.append(_cylinder_verts(s, e, 0.05))
                    bond_pieces.append(_cylinder_verts(s, e, 0.05, lateral_offset=-0.12))
                    bond_pieces.append(_cylinder_verts(s, e, 0.05, lateral_offset=+0.12))
                elif b.order == 4:
                    axis = e[0] - s[0]
                    for d in range(6):
                        t0 = d / 6.0 + 0.015
                        t1 = min((d + 0.8) / 6.0, 1.0)
                        if t1 <= t0:
                            continue
                        bond_pieces.append(_cylinder_verts(
                            s + axis * t0, s + axis * t1, BOND_RADIUS))
                else:
                    bond_pieces.append(_cylinder_verts(s, e, BOND_RADIUS))
                count = sum(len(p) for p in bond_pieces)
                self._bond_ranges.append((offset, count))
                offset += count
                pieces.extend(p for p in bond_pieces if len(p))
            self._bond_verts = (np.concatenate(pieces) if pieces
                                else np.zeros((0, 3), dtype=np.float32))

            self._cell_verts = _cell_edges(cell) if any(pbc) else None

            self._has_cell = any(pbc)
            self._cell = np.asarray(cell, dtype=np.float64)
            if reset_view or not self._view_fitted:
                self._fit_camera()

        self._data_dirty = True
        self.update()

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
        if self._style == "cpk":
            self._atom_radius = self._covalent_radii.copy()
        elif self._style == "wireframe":
            self._atom_radius = self._covalent_radii * 0.25
        else:  # ball_stick
            self._atom_radius = self._covalent_radii * SPHERE_SCALE
        self._show_bonds = self._style != "cpk"
        self.update()

    def set_structure_style(self, style: str) -> None:
        """Switch the display style: "ball_stick" | "cpk" | "wireframe"."""
        if style not in ("ball_stick", "cpk", "wireframe"):
            raise ValueError(f"Unknown display style: {style}")
        self._style = style
        self._apply_display()

    def structure_style(self) -> str:
        """The current display style."""
        return self._style

    def set_show_cell(self, visible: bool) -> None:
        """Show/hide the unit cell frame."""
        self._show_cell = visible
        self.update()

    def set_show_labels(self, visible: bool) -> None:
        """Show/hide element labels (only for structures ≤ 500 atoms)."""
        self._show_labels = visible
        self.update()

    def set_background_color(self, color: tuple[float, float, float]) -> None:
        """Set the viewport background color (RGB 0-1)."""
        self._background_color = tuple(float(c) for c in color)
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
            self._cam_center = positions.mean(axis=0).astype(np.float64)
            ref = positions - self._cam_center
        radii = np.linalg.norm(ref, axis=1)
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
        if self._cell_verts is not None:
            self._cell_vbo.allocate(self._cell_verts.tobytes(), self._cell_verts.nbytes)
        else:
            self._cell_vbo.allocate(b"", 0)
        self._cell_vao.release()

        self._meas_vao.bind()
        self._meas_vbo.bind()
        self._meas_vbo.allocate(self._meas_verts.tobytes(), self._meas_verts.nbytes)
        self._meas_vao.release()
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
        gl.glClearColor(*self._background_color, 1.0)
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
        # Blending is on so the near-distance fade in the shader
        # actually fades instead of showing the raw cross-section.
        self._sphere_prog.bind()
        self._sphere_prog.setUniformValue("uView", _to_qmatrix(view))
        # PySide6 has no (name, float) overload for setUniformValue —
        # set by location instead.
        self._sphere_prog.setUniformValue(
            self._sphere_prog.uniformLocation("uFadeStart"), NEAR_FADE_START)
        self._sphere_prog.setUniformValue(
            self._sphere_prog.uniformLocation("uFadeEnd"), NEAR_FADE_END)
        self._unit_vao.bind()
        gl.glEnable(GL_BLEND)
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        for i in range(len(self._atom_pos)):
            scale = float(self._atom_radius[i])
            if i in self._selected_indices or i in self._preview_indices:
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

        # --- Ghost atom (add-atom drag preview, semi-transparent) ---
        if self._ghost_pos is not None:
            z = atomic_numbers.get(self.current_element, 0)
            ghost_radius = (
                float(covalent_radii[z]) if 0 < z < len(covalent_radii) else 0.77
            ) * SPHERE_SCALE
            ghost_color = element_color(self.current_element)
            mvp = proj @ view @ _sphere_model(
                np.asarray(self._ghost_pos, dtype=np.float32), ghost_radius)
            self._sphere_prog.setUniformValue("uMVP", _to_qmatrix(mvp))
            self._sphere_prog.setUniformValue("uColor", *ghost_color, 0.55)
            gl.glDrawElements(GL_TRIANGLES, self._unit_n_indices,
                              GL_UNSIGNED_INT, VoidPtr(0))
        gl.glDisable(GL_BLEND)
        self._unit_vao.release()

        # --- Bonds (unlit cylinders; selected bonds highlighted) ---
        if self._bond_n_verts > 0 and self._show_bonds:
            self._flat_prog.bind()
            self._bond_vao.bind()
            self._flat_prog.setUniformValue("uMVP", _to_qmatrix(proj @ view))
            if self._selected_bonds:
                # per-bond colors: amber for selected, gray otherwise
                for k, (first, count) in enumerate(self._bond_ranges):
                    if count == 0:
                        continue
                    if k in self._selected_bonds:
                        self._flat_prog.setUniformValue("uColor", *HIGHLIGHT_COLOR, 1.0)
                    else:
                        self._flat_prog.setUniformValue("uColor", 0.55, 0.55, 0.55, 1.0)
                    gl.glDrawArrays(GL_TRIANGLES, first, count)
            else:
                self._flat_prog.setUniformValue("uColor", 0.55, 0.55, 0.55, 1.0)
                gl.glDrawArrays(GL_TRIANGLES, 0, self._bond_n_verts)
            self._bond_vao.release()

        # --- Unit cell outline (faint blue, only when periodic) ---
        # Depth-tested like normal geometry: parts of the frame behind
        # atoms are occluded by the ball-and-stick model (user
        # preference — the frame must not float on top). GL_DEPTH_CLAMP
        # keeps the frame fully visible even when the camera is inside
        # the cell or past a wall: vertices behind the near plane are
        # clamped instead of clipped, so the frame never shows a
        # near-plane "cross-section".
        if self._show_cell and self._cell_verts is not None and len(self._cell_verts) > 0:
            self._flat_prog.bind()
            self._cell_vao.bind()
            self._flat_prog.setUniformValue("uMVP", _to_qmatrix(proj @ view))
            self._flat_prog.setUniformValue("uColor", 0.55, 0.65, 0.85, 0.35)
            gl.glEnable(GL_DEPTH_CLAMP)
            gl.glEnable(GL_BLEND)
            gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            gl.glDrawArrays(GL_LINES, 0, len(self._cell_verts))
            gl.glDisable(GL_DEPTH_CLAMP)
            gl.glDisable(GL_BLEND)
            self._cell_vao.release()

        # --- Measurement lines (dashed amber; drawn on TOP of the model
        # — depth test disabled so sticks never hide them) ---
        if len(self._meas_verts) > 0:
            self._flat_prog.bind()
            self._meas_vao.bind()
            self._flat_prog.setUniformValue("uMVP", _to_qmatrix(proj @ view))
            self._flat_prog.setUniformValue("uColor", 1.0, 0.85, 0.0, 0.95)
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
            painter.setPen(QColor(255, 214, 90))
            font = QFont(self.font())
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(QPointF(s[0] + 8.0, s[1] - 6.0), text)
            painter.end()

        # Atom labels (element symbols; capped to keep the overlay fast)
        if self._show_labels and 0 < len(self._atom_pos) <= 500:
            painter = QPainter(self)
            painter.setPen(QColor(225, 225, 225))
            font = QFont(self.font())
            font.setPointSize(8)
            painter.setFont(font)
            for i, pos in enumerate(self._atom_pos):
                s = self.project_to_screen(pos)
                if s is None:
                    continue
                painter.drawText(QPointF(s[0] + 4.0, s[1] - 4.0), self._symbols[i])
            painter.end()

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
        """Orbit the camera: drag right → azimuth up; drag up → elevation up."""
        self._cam_azimuth -= dx * 0.5
        self._cam_elevation = max(-89.9, min(89.9, self._cam_elevation + dy * 0.5))
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
        indices or preview state across edits.
        """
        self._tool_active = False
        if self._active_tool is not None:
            self._active_tool.cancel()

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
        clip = proj @ view @ np.append(np.asarray(world, dtype=float), 1.0)
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
        """Drag fast path for several atoms (single repaint)."""
        for idx, pos in zip(indices, positions):
            if 0 <= idx < len(self._atom_pos):
                self._atom_pos[idx] = np.asarray(pos, dtype=np.float32)
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

        Lines are baked into a GL_LINES buffer here (minimum-image for
        periodic cells); labels are projected every frame from the LIVE
        atom positions, so they track atom drags.
        """
        self._measurements = payload
        atoms = self._atoms
        cell = np.asarray(
            atoms.get_cell().array if atoms is not None and atoms.get_cell().rank == 3
            else np.eye(3), dtype=float)
        pbc = (tuple(atoms.get_pbc()) if atoms is not None and atoms.get_pbc().any()
               else (False, False, False))

        verts: list[np.ndarray] = []
        for kind, idx, _text in payload:
            if kind == "distance":
                pairs = [(idx[0], idx[1])]
            elif kind == "angle":
                pairs = [(idx[1], idx[0]), (idx[1], idx[2])]
            else:
                pairs = [(idx[0], idx[1]), (idx[1], idx[2]), (idx[2], idx[3])]
            for a, b in pairs:
                start = np.asarray(self._atom_pos[a], dtype=float)
                end = start + mic_vector(a, b, self._atom_pos, cell, pbc)
                # Dashed: 16 segments per line, 60% duty (world-space —
                # line stippling is not available in the GL core profile)
                for d in range(16):
                    t0 = d / 16.0
                    t1 = min((d + 0.6) / 16.0, 1.0)
                    verts.append(start + (end - start) * t0)
                    verts.append(start + (end - start) * t1)
        self._meas_verts = (np.asarray(verts, dtype=np.float32)
                            if verts else np.zeros((0, 3), dtype=np.float32))
        self._data_dirty = True
        self.update()

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
