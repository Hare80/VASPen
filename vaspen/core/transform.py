"""Geometric transforms for structures: translate / rotate / align.

Pure copy-first functions — the input Atoms object is never modified.
Positions are wrapped back into the cell along periodic axes after each
transform so atoms stay inside the simulation box.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms


def wrap_periodic(atoms: Atoms) -> Atoms:
    """Wrap positions into the cell along axes with pbc (in place on a copy).

    Also exported for callers that postpone wrapping (whole-structure
    transforms of periodic structures: wrap AFTER the cell is rotated —
    wrapping in the old cell first would scramble fractional positions
    whenever the rotation is not a lattice symmetry).
    """
    pbc = np.asarray(atoms.get_pbc())
    if atoms.get_cell().rank == 3 and pbc.any():
        scaled = atoms.get_scaled_positions()
        for ax in range(3):
            if pbc[ax]:
                scaled[:, ax] %= 1.0
        atoms.set_scaled_positions(scaled)
    return atoms


def _resolve_indices(atoms: Atoms, indices: list[int] | None) -> list[int]:
    idx = list(range(len(atoms))) if indices is None else list(indices)
    if not idx or not all(0 <= i < len(atoms) for i in idx):
        raise ValueError("Atom index out of bounds.")
    return idx


def rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """3×3 Rodrigues rotation matrix (right-hand rule about ``axis``).

    Raises:
        ValueError: If the axis has zero length.
    """
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("Rotation axis has zero length.")
    axis = axis / norm
    theta = np.deg2rad(float(angle_deg))
    kx, ky, kz = axis
    K = np.array([
        [0.0, -kz, ky],
        [kz, 0.0, -kx],
        [-ky, kx, 0.0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rotation_to_align(
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Rotation mapping direction ``a1`` onto ``b1`` (and ``a2`` onto
    ``b2`` as closely as possible).

    Raises:
        ValueError: If a direction pair is parallel.
    """
    A = _orthonormal_basis(np.asarray(a1, dtype=float),
                           np.asarray(a2, dtype=float))
    B = _orthonormal_basis(np.asarray(b1, dtype=float),
                           np.asarray(b2, dtype=float))
    return B @ A.T


def translate(
    atoms: Atoms,
    vector: np.ndarray,
    indices: list[int] | None = None,
    wrap: bool = True,
) -> Atoms:
    """Copy and translate ``indices`` (or all atoms) by a Cartesian vector.

    Args:
        atoms: Source structure (not modified).
        vector: (dx, dy, dz) in Angstrom.
        indices: Atom indices to move; None moves the whole structure.
        wrap: Wrap periodic axes after the transform (default True).

    Returns:
        Transformed copy (periodic axes wrapped unless ``wrap`` is False).
    """
    out = atoms.copy()
    idx = _resolve_indices(out, indices)
    positions = out.get_positions()
    positions[idx] += np.asarray(vector, dtype=float)
    out.set_positions(positions)
    return wrap_periodic(out) if wrap else out


def rotate(
    atoms: Atoms,
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    angle_deg: float,
    indices: list[int] | None = None,
    center: np.ndarray | None = None,
    wrap: bool = True,
) -> Atoms:
    """Copy and rotate ``indices`` (or all atoms) around an axis.

    Rodrigues rotation (right-hand rule around the axis pointing from
    ``axis_start`` to ``axis_end``).

    Args:
        atoms: Source structure (not modified).
        axis_start, axis_end: Two points defining the rotation axis.
        angle_deg: Rotation angle in degrees.
        indices: Atom indices to rotate; None rotates the whole structure.
        center: Point the rotation acts about. Default: centroid of the
            moved set.
        wrap: Wrap periodic axes after the transform (default True).

    Returns:
        Transformed copy (periodic axes wrapped unless ``wrap`` is False).

    Raises:
        ValueError: If the axis has zero length.
    """
    out = atoms.copy()
    idx = _resolve_indices(out, indices)
    start = np.asarray(axis_start, dtype=float)
    axis = np.asarray(axis_end, dtype=float) - start
    R = rotation_matrix(axis, angle_deg)
    if center is None:
        center = out.get_positions()[idx].mean(axis=0)
    else:
        center = np.asarray(center, dtype=float)
    positions = out.get_positions()
    positions[idx] = (positions[idx] - center) @ R.T + center
    out.set_positions(positions)
    return wrap_periodic(out) if wrap else out


def _orthonormal_basis(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Orthonormal basis (columns) from two non-parallel vectors."""
    e1 = u / np.linalg.norm(u)
    e2 = v - e1 * float(v @ e1)
    n2 = float(np.linalg.norm(e2))
    if n2 < 1e-12:
        raise ValueError("The two directions are parallel.")
    e2 = e2 / n2
    e3 = np.cross(e1, e2)
    return np.column_stack([e1, e2, e3])


def align(
    atoms: Atoms,
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    indices: list[int] | None = None,
    wrap: bool = True,
) -> Atoms:
    """Copy and rotate so direction ``a1`` maps onto ``b1`` (and ``a2``
    onto ``b2`` as closely as possible).

    The smallest rotation mapping the (a1, a2) frame onto the (b1, b2)
    frame; the rotation acts about the centroid of the moved set.

    Args:
        atoms: Source structure (not modified).
        a1, a2: Two non-parallel source directions (columns of the from-frame).
        b1, b2: Two non-parallel target directions (columns of the to-frame).
        indices: Atom indices to rotate; None rotates the whole structure.
        wrap: Wrap periodic axes after the transform (default True).

    Returns:
        Transformed copy (periodic axes wrapped unless ``wrap`` is False).

    Raises:
        ValueError: If a direction pair is parallel.
    """
    out = atoms.copy()
    idx = _resolve_indices(out, indices)
    R = rotation_to_align(a1, a2, b1, b2)
    center = out.get_positions()[idx].mean(axis=0)
    positions = out.get_positions()
    positions[idx] = (positions[idx] - center) @ R.T + center
    out.set_positions(positions)
    return wrap_periodic(out) if wrap else out
