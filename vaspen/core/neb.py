"""NEB path tools — linear interpolation and a distance metric.

The interpolation is the classic linear NEB setup: every atom moves in
a straight line between its initial and final positions, using the
periodic minimal-image displacement, with the lattice held constant
and equal to the initial cell. Endpoints and all intermediate images
share the same lattice.

The distance metric between two structures is the Euclidean norm of
the full 3N-atom displacement vector (in Angstrom), with each atomic
displacement wrapped to its Wigner-Seitz minimal image. Dividing it by
a per-image spacing (~0.8 Angstrom) gives the community-standard
estimate for the number of intermediate images.

Atom order is never changed: atom i of the initial structure always
interpolates to atom i of the final structure (strict file order).
``detect_order_mismatch`` only diagnoses whether a same-element
reordering would shorten the path considerably, so the caller can
warn the user.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from ase import Atoms
from PySide6.QtCore import QCoreApplication


def _tr(text: str) -> str:
    """Translate a user-visible Neb message (hand-maintained .ts)."""
    return QCoreApplication.translate("Neb", text)


#: Target path spacing used to estimate the number of images.
DEFAULT_IMAGE_SPACING = 0.8  # Angstrom per image
#: Upper bound for intermediate images (a NEB band of this size is
#: already very expensive; it also keeps the directory layout two-digit).
MAX_IMAGES = 98


def pbc_wrap(x):
    """Wrap coordinate differences to [-0.5, 0.5) with a single step.

    Works on scalars and arrays. Exactly -0.5 wraps to +0.5; +0.5 stays
    (asymmetric boundary by construction — matching the reference
    behaviour of the classic implementations).
    """
    x = np.asarray(x, dtype=float)
    return np.where(x <= -0.5, x + 1.0, np.where(x > 0.5, x - 1.0, x))


def ws_minimal_image(cell: np.ndarray, frac_delta: np.ndarray) -> np.ndarray:
    """Wigner-Seitz minimal image of a fractional displacement.

    Converts the fractional displacement to Cartesian, then hill-climbs
    over the 27 neighbouring lattice images until no single shift
    shortens the vector. Returns the shortest Cartesian displacement
    (Angstrom).

    Args:
        cell: 3×3 real-space cell matrix (Angstrom, rows = lattice
            vectors).
        frac_delta: Fractional displacement vector (any length).
    """
    cell = np.asarray(cell, dtype=float)
    v = frac_delta @ cell
    shifts = np.array(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=float,
    )
    improved = True
    while improved:
        improved = False
        candidates = v + shifts @ cell
        sq = np.einsum("ij,ij->i", candidates, candidates)
        best = int(np.argmin(sq))
        if sq[best] < float(np.dot(v, v)) - 1e-12:
            v = candidates[best]
            improved = True
    return v


def _validate_neb_pair(init: Atoms, final: Atoms) -> None:
    """Check that two structures can form a NEB pair (raises ValueError)."""
    if len(init) != len(final):
        raise ValueError(
            _tr("Initial and final structures must have the same number "
                "of atoms ({} vs {}).").format(len(init), len(final)))
    if Counter(init.get_chemical_symbols()) != Counter(final.get_chemical_symbols()):
        raise ValueError(
            _tr("Initial and final structures must contain the same "
                "elements with the same counts."))
    if init.get_cell().rank < 3 or final.get_cell().rank < 3:
        raise ValueError(
            _tr("NEB requires periodic structures (a full-rank cell on "
                "both sides)."))
    if not np.allclose(init.get_cell(), final.get_cell(), rtol=1e-4, atol=1e-4):
        raise ValueError(
            _tr("Initial and final cells differ — variable-cell NEB is "
                "not supported. Align the two structures to the same "
                "cell first."))


def neb_distance(init: Atoms, final: Atoms) -> float:
    """Distance between two structures for a NEB path (Angstrom).

    Each atom contributes its minimal-image Cartesian displacement
    (per-component wrap, then Wigner-Seitz refinement); the result is
    the Euclidean norm of the full 3N-atom displacement vector. Atoms
    are paired strictly in file order.

    Raises:
        ValueError: If the structures cannot form a NEB pair (different
            atoms/counts, missing cells, or differing cells).
    """
    _validate_neb_pair(init, final)
    cell = np.asarray(init.get_cell(), dtype=float)
    frac_i = init.get_scaled_positions(wrap=False)
    frac_f = final.get_scaled_positions(wrap=False)
    delta = pbc_wrap(frac_i - frac_f)
    total_sq = 0.0
    for d in delta:
        v = ws_minimal_image(cell, d)
        total_sq += float(np.dot(v, v))
    return float(np.sqrt(total_sq))


def detect_order_mismatch(init: Atoms, final: Atoms) -> float | None:
    """Diagnose a likely atom-order mismatch between the two structures.

    Computes the shortest achievable path length if atoms of the same
    element could be reordered (optimal assignment per element, cost =
    squared minimal-image pair distance). Returns that matched length
    (Angstrom) when reordering shortens the file-order distance by more
    than 25% AND more than 1 Angstrom — the caller should then warn the
    user. Returns None when the file order is fine.

    This is a pure diagnostic: interpolation always follows the strict
    file order.
    """
    try:
        file_norm = neb_distance(init, final)
    except ValueError:
        return None
    if file_norm < 1e-9:
        return None
    if Counter(init.get_chemical_symbols()) != Counter(final.get_chemical_symbols()):
        return None

    from scipy.optimize import linear_sum_assignment

    cell = np.asarray(init.get_cell(), dtype=float)
    frac_i = init.get_scaled_positions(wrap=False)
    frac_f = final.get_scaled_positions(wrap=False)
    symbols_i = init.get_chemical_symbols()
    symbols_f = final.get_chemical_symbols()

    total_sq = 0.0
    for element in set(symbols_i):
        idx_i = [i for i, s in enumerate(symbols_i) if s == element]
        idx_f = [j for j, s in enumerate(symbols_f) if s == element]
        cost = np.zeros((len(idx_i), len(idx_f)))
        for a, i in enumerate(idx_i):
            for b, j in enumerate(idx_f):
                v = ws_minimal_image(cell, pbc_wrap(frac_i[i] - frac_f[j]))
                cost[a, b] = float(np.dot(v, v))
        rows, cols = linear_sum_assignment(cost)
        total_sq += float(cost[rows, cols].sum())

    matched_norm = float(np.sqrt(total_sq))
    improvement = file_norm - matched_norm
    if improvement > 1.0 and improvement / file_norm > 0.25:
        return matched_norm
    return None


def suggest_n_images(
    distance: float,
    spacing: float = DEFAULT_IMAGE_SPACING,
) -> int:
    """Estimate the number of intermediate NEB images for a path.

    Community heuristic: one image per ``spacing`` Angstrom of path
    length (default 0.8). At least one image is always suggested.
    """
    return max(1, int(np.ceil(distance / spacing)))


def _frame_copy(atoms: Atoms) -> Atoms:
    """Copy an Atoms object for a NEB frame (no constraints/metadata)."""
    out = atoms.copy()
    out.constraints = []
    out.info = {}
    return out


def interpolate_neb(init: Atoms, final: Atoms, n_images: int) -> list[Atoms]:
    """Linearly interpolate a NEB path between two structures.

    The path has ``n_images`` intermediate images plus both endpoints
    (``n_images + 2`` frames in total). Interpolation is linear in
    fractional coordinates per atom: the minimal-image fractional
    displacement between the initial and final positions (per component)
    is split into ``n_images + 1`` equal steps, and each frame is
    wrapped back into the cell. The lattice of every frame equals the
    initial lattice. Strict file order is used for atom pairing.

    Args:
        init: Initial structure (periodic).
        final: Final structure (periodic, same cell and composition).
        n_images: Number of intermediate images (1..98).

    Returns:
        List of ``n_images + 2`` Atoms frames: the initial structure,
        the intermediates in order, then the final structure.

    Raises:
        ValueError: Pair mismatch, or n_images out of range.
    """
    _validate_neb_pair(init, final)
    if not (1 <= n_images <= MAX_IMAGES):
        raise ValueError(
            _tr("Number of images must be between 1 and {} (got {}).")
            .format(MAX_IMAGES, n_images))

    frac_i = init.get_scaled_positions(wrap=False)
    frac_f = final.get_scaled_positions(wrap=False)
    delta = pbc_wrap(frac_f - frac_i)

    frames: list[Atoms] = [_frame_copy(init)]
    for k in range(1, n_images + 1):
        frac = frac_i + delta * (k / (n_images + 1))
        # Keep every image inside the cell; snap float noise at the
        # boundary (values within 1e-12 of 0/1) to exactly 0 so wrapped
        # frames never come out as 1.0 where the boundary is meant.
        frac = np.where(np.abs(frac) < 1e-12, 0.0, frac)
        frac = np.mod(frac, 1.0)
        frac = np.where(frac > 1.0 - 1e-12, 0.0, frac)
        frame = _frame_copy(init)
        frame.set_scaled_positions(frac)
        frames.append(frame)
    frames.append(_frame_copy(final))
    return frames
