"""Offscreen 2D structure rendering for the MCP ``render_preview`` tool.

ASE's matplotlib ball-and-stick plotter (``ase.visualize.plot``) with
the non-interactive Agg backend: deterministic, cross-platform (no GL
context needed — works the same on Windows and Linux) and testable.
The GUI's GL viewport is deliberately NOT used here — see §7.11.

matplotlib is a declared dependency of the ``mcp`` extra group (it is
also pulled in transitively by pymatgen, but this package imports it
directly).
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # MUST precede any pyplot import (headless, no display)

from ase import Atoms  # noqa: E402  (after matplotlib.use)
from ase.visualize.plot import plot_atoms  # noqa: E402

#: Valid ``view`` names for molecules (world axes).
_WORLD_VIEWS = ("x", "y", "z")


def _rotation_string(view: str, atoms: Atoms) -> str:
    """Euler-xyz rotation string that brings the view axis to +z.

    ``plot_atoms`` orthographically projects onto the xy plane (looking
    down z); rotating the structure first looks along the requested
    axis instead. World views (x/y/z) use fixed rotations; cell views
    (a/b/c) align the corresponding cell vector with z (computed, since
    the cell can be arbitrarily oriented).
    """
    if view in _WORLD_VIEWS:
        mapping = {"x": "0x,-90y", "y": "90x,0y", "z": "0x,0y"}
        return mapping[view]
    # Cell-vector view: rotate cell[axis] onto +z.
    axis = {"a": 0, "b": 1, "c": 2}[view]
    cell = atoms.get_cell().array
    v = cell[axis]
    norm = float((v * v).sum()) ** 0.5
    if norm == 0.0:
        raise ValueError(
            f"Cell vector {view} is degenerate — cannot render it.")
    v = v / norm
    from scipy.spatial.transform import Rotation as _Rotation

    # align_vectors maps the SECOND vector onto the first: z → v; we
    # need the inverse (v → z), which mirrors the rotation.
    rot = _Rotation.align_vectors([[0.0, 0.0, 1.0]], [v])[0]
    rx, ry, rz = rot.as_euler("xyz", degrees=True)
    return f"{rx:.3f}x,{ry:.3f}y,{rz:.3f}z"


def render_png(atoms: Atoms, view: str = "auto", max_px: int = 800) -> bytes:
    """Render ``atoms`` as a ball-and-stick PNG (orthographic).

    Args:
        atoms: Structure to render (not modified).
        view: Projection direction — x/y/z (world), a/b/c (cell
            vectors, periodic only) or "auto" (c for periodic
            structures, z for molecules).
        max_px: Maximum canvas edge in pixels (square figure).

    Returns:
        PNG image bytes.
    """
    if view == "auto":
        view = "c" if (atoms.get_cell().rank == 3 and atoms.pbc.any()) else "z"
    if view in ("a", "b", "c") and not (
            atoms.get_cell().rank == 3 and atoms.pbc.any()):
        raise ValueError(
            f"view={view!r} requires a periodic structure; use x/y/z "
            "for molecules.")
    import matplotlib.pyplot as plt

    rotation = _rotation_string(view, atoms)
    size_in = max(1, int(max_px)) / 100.0  # 100 dpi
    fig = plt.figure(figsize=(size_in, size_in), dpi=100)
    try:
        ax = fig.add_subplot(111)
        plot_atoms(atoms, ax=ax, rotation=rotation, show_unit_cell=2)
        ax.set_axis_off()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        return buf.getvalue()
    finally:
        plt.close(fig)
