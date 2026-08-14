"""Surface / slab generation via Miller-index cutting.

Slab cutting and termination enumeration use pymatgen's
``SlabGenerator`` (unique terminations, symmetry-deduplicated);
``ase.build.cut`` remains for arbitrary-plane cuts. Provides a clean
API for the UI layer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.build import surface as ase_surface
from pymatgen.core import Element
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor

from vaspen.core.structure import StructureModel


@dataclass(frozen=True)
class SlabInfo:
    """One unique slab termination.

    Attributes:
        atoms: The slab as ASE Atoms (pbc=(T,T,T), full-rank cell).
        top_composition: Composition formula of the top surface layer
            (e.g. "SrO").
        bottom_composition: Same for the bottom surface layer.
        broken_bonds: Broken-bond count from SlabGenerator. Always 0
            with the default ``bonds=None``; kept for future
            bond-aware filtering.
        n_atoms: Number of atoms in the slab.
    """

    atoms: Atoms
    top_composition: str
    bottom_composition: str
    broken_bonds: int
    n_atoms: int


def _d_hkl(atoms: Atoms, miller: tuple[int, int, int]) -> float:
    """Interplanar spacing along the Miller normal (Angstrom).

    Raises:
        ValueError: If the Miller indices are all zero.
    """
    normal = np.array(miller, dtype=float)
    if not np.any(normal):
        raise ValueError("Miller indices must not all be zero.")
    return 1.0 / np.linalg.norm(np.linalg.solve(atoms.get_cell().T, normal))


def _reduce_in_plane(a: np.ndarray, b: np.ndarray,
                     n: np.ndarray, area: float) -> tuple[np.ndarray, np.ndarray]:
    """Shortest pair of in-plane lattice vectors spanning the same 2D lattice.

    Enumerates small integer combinations of (a, b) and keeps the pair
    with the smallest total length (ties: the one closest to
    orthogonal). pymatgen's (100) in-plane basis (1,0)/(−1,1)·s reduces
    to the (1,0)/(0,1)·s square cell this way; the (111) hex basis is
    already minimal and is kept. Pure lattice re-basis — the atom set
    is unchanged.
    """
    cands = [m * a + n * b for m in range(-2, 3) for n in range(-2, 3)]
    cands = [v for v in cands if np.linalg.norm(v) > 1e-9]
    best = None
    for i, w1 in enumerate(cands):
        for w2 in cands[i + 1:]:
            cr = np.cross(w1, w2)
            if abs(abs(float(np.dot(cr, n))) - area) > 1e-6 * area:
                continue  # must span the same 2D lattice
            score = (np.linalg.norm(w1) + np.linalg.norm(w2),
                     abs(float(np.dot(w1, w2))))
            if best is None or score < best[0]:
                best = (score, w1.copy(), w2.copy())
    _, v1, v2 = best
    return v1, v2


def _standard_slab_cell(atoms: Atoms, vacuum: float) -> Atoms:
    """Re-express a pymatgen slab in the conventional slab presentation.

    The raw pymatgen box is sheared for e.g. cubic (111) (the c axis is
    NOT perpendicular to a and b — the vacuum direction tilts relative
    to the surface normal). This rebuilds the cell so that:

    - the in-plane vectors are the shortest spanning pair (square cell
      for (100), hex cell for (111));
    - c lies exactly along the surface normal (vacuum ⊥ ab);
    - c length = slab thickness + ``vacuum`` (the requested vacuum is
      exact);
    - the slab is centered along c (half the vacuum on each side).

    All steps are lattice re-bases (unimodular integer transforms), a
    pure rotation and a translation — interatomic distances are
    preserved exactly.
    """
    # pymatgen's centering is skipped for performance (per-atom
    # neighbor searches, ~10× slower than the whole rest of the
    # pipeline) — its raw slabs may be split across the periodic
    # boundary, which would inflate the thickness measured below.
    atoms = _unwrap_layers(atoms)
    cell = np.asarray(atoms.get_cell().array, dtype=float)
    frac = np.asarray(atoms.get_scaled_positions(), dtype=float)
    a, b, c = cell
    n = np.cross(a, b)
    area = float(np.linalg.norm(n))
    n = n / area

    # 1. Reduce the in-plane basis; re-express the fractional x/y.
    v1, v2 = _reduce_in_plane(a, b, n, area)
    x_ref = v1 / np.linalg.norm(v1)
    y_ref = np.cross(n, x_ref)
    # canonical presentation: det along +n (kept), a into +x_ref, b
    # into +y_ref — all four variants are rotations of the same basis
    variants = [(v1, v2), (v2, -v1), (-v1, -v2), (-v2, v1)]
    v1, v2 = max(variants,
                 key=lambda p: (float(np.dot(p[0], x_ref))
                                + float(np.dot(p[1], y_ref))))
    basis = np.column_stack([a, b])
    c1, *_ = np.linalg.lstsq(basis, v1, rcond=None)
    c2, *_ = np.linalg.lstsq(basis, v2, rcond=None)
    m = np.column_stack([np.round(c1), np.round(c2)]).astype(int)
    if abs(float(np.linalg.det(m))) != 1:
        # Fallback: keep pymatgen's in-plane basis (never corrupt data).
        v1, v2 = a, b
        m = np.eye(2, dtype=int)
    frac[:, :2] = np.linalg.solve(m.T, frac[:, :2].T).T
    cell[0], cell[1] = v1, v2

    # 2. Pure rotation: n → z, a → +x (a, b land in the xy-plane).
    x_w = v1 / np.linalg.norm(v1)
    z_w = n
    y_w = np.cross(z_w, x_w)
    y_w /= np.linalg.norm(y_w)
    rot = np.vstack([x_w, y_w, z_w])  # rows = new-frame axes (det +1)
    cart = (frac @ cell) @ rot.T
    rcell = cell @ rot.T

    # 3. Exact vacuum + centering along the (now z) normal.
    zmin, zmax = cart[:, 2].min(), cart[:, 2].max()
    c_z = (zmax - zmin) + float(vacuum)
    cart[:, 2] += c_z / 2.0 - (zmin + zmax) / 2.0
    out_cell = np.array([rcell[0], rcell[1], [0.0, 0.0, c_z]])
    new_frac = np.linalg.solve(out_cell.T, cart.T).T
    new_frac -= np.floor(new_frac)

    out = atoms.copy()
    out.set_cell(out_cell)
    out.set_scaled_positions(new_frac)
    return out


def _surface_compositions(atoms: Atoms) -> tuple[str, str]:
    """(top, bottom) composition formulas of the surface layers.

    Deterministic pure-numpy projection: positions are projected on
    the c-vector; a layer is everything within ``tol`` of the min/max
    projected plane. The 0.49 factor keeps intra-double-layer planes
    of zincblende (111) (~0.8 Å apart) distinct while merging
    co-planar atoms.
    """
    c = np.asarray(atoms.get_cell()[2], dtype=float)
    c = c / np.linalg.norm(c)
    proj = np.asarray(atoms.get_positions()) @ c
    uniq = np.sort(np.unique(np.round(proj, 3)))
    gaps = np.diff(uniq)
    tol = min(1.0, 0.49 * float(gaps.min())) if len(gaps) > 0 else 1.0
    symbols = atoms.get_chemical_symbols()
    top = Counter(sym for sym, p in zip(symbols, proj) if p > proj.max() - tol)
    bot = Counter(sym for sym, p in zip(symbols, proj) if p < proj.min() + tol)
    return (_layer_formula(top), _layer_formula(bot))


def _layer_formula(counts: Counter) -> str:
    """Format a layer composition as "SrO" / "TiO2" / "O".

    Symbols are ordered by Pauling electronegativity (cation first),
    counts > 1 written as digits. Deliberately NOT pymatgen
    ``reduced_formula`` — it forces the standard-state diatomic
    convention (O -> O2, OH -> H2O2), which mislabels surfaces.
    """
    def _order(sym: str) -> tuple[float, str]:
        try:
            return (float(Element(sym).X), sym)
        except (TypeError, ValueError):
            return (0.0, sym)

    syms = sorted(counts, key=_order)
    return "".join(sym + (str(counts[sym]) if counts[sym] > 1 else "")
                   for sym in syms)


_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _subscript_formula(formula: str) -> str:
    """Render a composition formula with unicode subscripts: TiO2 -> TiO₂."""
    return formula.translate(_SUBSCRIPT)


def _unwrap_layers(atoms: Atoms) -> Atoms:
    """Fold layers split across the periodic c boundary into one block.

    Clusters the atoms by their fractional-c gaps (gap > 3× the smallest
    breaks a cluster — the vacuum gap dominates a wrapped slab), keeps
    the largest cluster as the reference window and folds every other
    atom back into it with integer ±c translations. The atom set and
    order are unchanged; the returned cell is untouched.
    """
    out = atoms.copy()
    frac = np.asarray(out.get_scaled_positions(), dtype=float)
    z = frac[:, 2]  # fractional coordinate along c

    order = np.argsort(z)
    z_sorted = z[order]
    gaps = np.diff(z_sorted)
    if len(gaps) == 0:
        return out  # a single atom has nothing to unwrap
    thresh = 3.0 * gaps.min()
    clusters: list[np.ndarray] = []
    start = 0
    for i, g in enumerate(gaps):
        if g > thresh:
            clusters.append(order[start:i + 1])
            start = i + 1
    clusters.append(order[start:])
    main = max(clusters, key=len)
    lo, hi = z[main].min(), z[main].max()
    # fold every atom into the contiguous window around the main
    # cluster (integer c translations — the atom set is unchanged)
    for i in range(len(out)):
        while z[i] < lo - 0.5:
            z[i] += 1.0
        while z[i] > hi + 0.5:
            z[i] -= 1.0
    frac[:, 2] = z
    out.set_scaled_positions(frac)
    return out


def rebox_slab(atoms: Atoms, vacuum: float) -> Atoms:
    """Re-box a slab: unwrap split layers, re-apply vacuum, center.

    The standalone "Re-box Slab" feature (Tools menu) for periodic
    structures that already carry vacuum (e.g. cut surfaces). Unwraps
    layers that periodic wrapping would split across the boundary
    (pure ±c translations per atom), then re-applies the requested
    vacuum along c with the slab centered. The in-plane cell (a, b)
    is untouched, the atom order and set are unchanged (fixed flags /
    magnetic moments map 1:1); only the c length and a rigid c-shift
    differ from the input.
    """
    out = _unwrap_layers(atoms)
    cell = np.asarray(out.get_cell().array, dtype=float)
    c = cell[2]
    c_len = float(np.linalg.norm(c))

    # exact vacuum + centering along c (works for any c orientation)
    proj = out.get_positions() @ (c / c_len)
    zmin, zmax = proj.min(), proj.max()
    new_c = (c / c_len) * ((zmax - zmin) + float(vacuum))
    out.translate((c / c_len) * ((zmax - zmin + float(vacuum)) / 2.0
                                 - (zmin + zmax) / 2.0))
    cell[2] = new_c
    out.set_cell(cell)
    frac = out.get_scaled_positions()
    frac -= np.floor(frac)
    out.set_scaled_positions(frac)
    return out


def supercell_in_plane(atoms: Atoms, a: int, b: int) -> Atoms:
    """Repeat the slab a×b in the surface plane (c untouched).

    A supercell along the vacuum direction is meaningless for a slab,
    so the c repeat factor is fixed at 1.
    """
    return atoms.repeat((a, b, 1))


class _ComputeOrder:
    """Compute-order/cancellation hint for :func:`iter_slabs` (no locks
    by design).

    The main thread writes ``priority`` (the index the user clicked)
    and ``cancelled`` (parameters changed — stop as soon as the current
    item finishes); the worker reads them once per item — single
    int/bool read/writes are atomic under the GIL. A priority that was
    already computed is simply not in ``pending`` anymore and is
    ignored, so it never has to be cleared explicitly.
    """

    __slots__ = ("priority", "cancelled")

    def __init__(self) -> None:
        self.priority: int = -1
        self.cancelled: bool = False


def _slab_info(slab, vacuum: float) -> SlabInfo:
    """One termination through the standard post-processing pipeline."""
    # get_orthogonal_c_slab makes c ⊥ a,b (pymatgen's raw box is
    # sheared — the vacuum tilted relative to the normal); then
    # re-express in the conventional presentation (standard
    # orientation, exact vacuum, centered).
    atoms = AseAtomsAdaptor.get_atoms(slab.get_orthogonal_c_slab())
    atoms = _standard_slab_cell(atoms, float(vacuum))
    top, bottom = _surface_compositions(atoms)
    return SlabInfo(
        atoms=atoms,
        top_composition=top,
        bottom_composition=bottom,
        broken_bonds=int(slab.energy or 0),
        n_atoms=len(atoms),
    )


def _slab_generator(atoms: Atoms, miller: tuple[int, int, int],
                    layers: int, vacuum: float) -> SlabGenerator:
    """The pymatgen SlabGenerator with the project's fixed settings.

    ``center_slab=False``: pymatgen's centering does per-atom neighbor
    searches (~10× slower than everything else combined, profiled) —
    our `_standard_slab_cell` centers exactly anyway. Single place so
    ``iter_slabs`` and ``slab_count`` stay in sync.
    """
    return SlabGenerator(
        AseAtomsAdaptor.get_structure(atoms),
        miller,
        min_slab_size=layers * _d_hkl(atoms, miller),
        min_vacuum_size=float(vacuum),
        center_slab=False,
    )


def _possible_terminations(gen: SlabGenerator) -> list[float]:
    """The termination shift values of a generator (instant — no slab).

    Ported from pymatgen ``SlabGenerator.get_slabs``' internal
    ``gen_possible_terminations`` (pymatgen is MIT-licensed): cluster
    the oriented-unit-cell z-coordinates with scipy's fcluster and take
    the cluster midpoints. pymatgen keeps this function local, so the
    port (public API only) makes incremental enumeration possible.
    """
    import itertools
    import math

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    frac_coords = gen.oriented_unit_cell.frac_coords
    n_atoms = len(frac_coords)
    if n_atoms == 1:
        # put the atom in the center
        termination = frac_coords[0][2] + 0.5
        return [termination - math.floor(termination)]
    # The z distances are scaled by pymatgen's _proj_height — the
    # projection of the ouc c vector onto the surface normal (the ouc
    # c is only "as normal as possible" and is NOT parallel for e.g.
    # cubic (111), where proj_height = |c|/√3). Prefer the private
    # value verbatim (exact match with pymatgen's own clustering);
    # fall back to the same formula pymatgen's __init__ uses.
    try:
        proj_height = float(gen._proj_height)  # noqa: SLF001
    except AttributeError:
        normal = gen.parent.lattice.reciprocal_lattice.get_cartesian_coords(
            gen.miller_index)
        normal = normal / np.linalg.norm(normal)
        proj_height = abs(float(
            np.dot(normal, gen.oriented_unit_cell.lattice.matrix[2])))
    dist_matrix = np.zeros((n_atoms, n_atoms), dtype=np.float64)
    for i, j in itertools.combinations(range(n_atoms), 2):
        z_dist = frac_coords[i][2] - frac_coords[j][2]
        z_dist = abs(z_dist - round(z_dist)) * proj_height
        dist_matrix[i, j] = z_dist
        dist_matrix[j, i] = z_dist
    clusters = fcluster(linkage(squareform(dist_matrix)), 0.1,
                        criterion="distance")
    clst_loc = {clst: frac_coords[idx][2]
                for idx, clst in enumerate(clusters)}
    possible = [coord - math.floor(coord)
                for coord in sorted(clst_loc.values())]
    n_terms = len(possible)
    terminations: list[float] = []
    for idx in range(n_terms):
        if idx == n_terms - 1:
            # first-last pair closes the periodic boundary
            termination = (possible[0] + 1 + possible[idx]) * 0.5
        else:
            termination = (possible[idx] + possible[idx + 1]) * 0.5
        terminations.append(termination - math.floor(termination))
    return sorted(terminations)


def slab_count(atoms: Atoms, miller: tuple[int, int, int],
               layers: int = 4, vacuum: float = 15.0) -> int | None:
    """Number of terminations — instant (no slab is built).

    Returns None when the shift enumeration is unavailable on the
    current pymatgen (the whole-list fallback path of ``iter_slabs``).
    """
    try:
        return len(_possible_terminations(
            _slab_generator(atoms, miller, layers, vacuum)))
    except Exception:  # noqa: BLE001 — unknown pymatgen internals
        return None


def iter_slabs(
    atoms: Atoms,
    miller: tuple[int, int, int],
    layers: int = 4,
    vacuum: float = 15.0,
    order: _ComputeOrder | None = None,
):
    """Yield the unique terminations one at a time as ``(index, SlabInfo)``.

    The incremental form of the cleave pipeline: the dialog previews
    the first termination as soon as it lands and fills the rest
    progressively (see also ``slab_count`` for the instant total).

    ``order`` (optional) reorders the COMPUTATION cooperatively: when
    ``order.priority`` names an index still pending, that one is
    computed next (the dialog sets it when the user clicks an unloaded
    item). The yielded index is always the canonical slot, so the UI
    order never changes.

    Fallback: if pymatgen exposes no shift enumeration hook, the whole
    list is computed at once and yielded in a burst (same output).
    """
    gen = _slab_generator(atoms, miller, layers, vacuum)
    # filter_out_sym_slabs=False: keep mirror-pair cleavages as
    # separate choices (e.g. SrTiO3 (100) offers SrO-top and TiO2-top
    # slabs). pymatgen's default symmetry dedup would collapse
    # translation-equivalent mirrors to a single entry, hiding the
    # top/bottom face choice from the user.
    try:
        shifts = _possible_terminations(gen)
    except Exception:  # noqa: BLE001 — unknown pymatgen internals
        # whole-list fallback (no incremental enumeration possible)
        for index, slab in enumerate(gen.get_slabs(filter_out_sym_slabs=False)):
            yield index, _slab_info(slab, vacuum)
        return
    total = len(shifts)
    pending = set(range(total))
    sequential = iter(range(total))
    while pending:
        # cooperative cancellation: parameters changed in the dialog —
        # stop before the next item (the caller's generation guard
        # discards everything we emitted so far)
        if order is not None and order.cancelled:
            return
        if order is not None and order.priority in pending:
            index = order.priority  # clicked item jumps the queue
        else:
            index = next(sequential)
        slab = gen.get_slab(shifts[index], tol=0.1)
        yield index, _slab_info(slab, vacuum)
        pending.discard(index)


class SurfaceCutter:
    """Generate slab structures from bulk crystals.

    Usage:
        cutter = SurfaceCutter(model)
        slab = cutter.cut(miller=(1, 1, 1), layers=4, vacuum=15.0)
    """

    def __init__(self, model: StructureModel) -> None:
        """Initialize with a bulk structure.

        Args:
            model: The bulk crystal StructureModel to cut from.
        """
        self._model = model

    def cut(
        self,
        miller: tuple[int, int, int],
        layers: int = 4,
        vacuum: float = 15.0,
    ) -> StructureModel:
        """Generate a slab from the bulk structure.

        Backward-compatible entry point: returns the first (lowest
        broken-bond count) unique termination. Uses pymatgen
        ``SlabGenerator``; ``layers`` is a minimum — the slab may
        contain one extra layer (pymatgen ceils to whole oriented
        unit cells).

        Args:
            miller: Miller indices (h, k, l) of the surface.
            layers: Number of atomic layers in the slab (minimum).
            vacuum: Vacuum spacing in Angstrom added above the slab.

        Returns:
            A new StructureModel containing the slab.
        """
        return self.cut_termination(miller, layers, vacuum, termination=0)

    def slabs(
        self,
        miller: tuple[int, int, int],
        layers: int = 4,
        vacuum: float = 15.0,
    ) -> list[SlabInfo]:
        """All unique terminations for the Miller index, via pymatgen.

        Thin wrapper around :func:`iter_slabs` (the incremental
        generator the dialog uses); see it for the pipeline. Sorted in
        pymatgen's canonical termination order (stable; broken-bond
        count when bonds are supplied). Initial magnetic moments
        round-trip through site properties automatically.

        Inputs are assumed to be dense bulks (per the pymatgen manual);
        structures that already carry vacuum belong to the separate
        Tools → Re-box Slab feature, not here.

        Args:
            miller: Miller indices (h, k, l) of the surface.
            layers: Minimum number of atomic layers in the slab.
            vacuum: Vacuum spacing in Angstrom (centered).

        Returns:
            List of SlabInfo, one per unique termination.

        Raises:
            ValueError: If the Miller indices are invalid or the cut
                cannot be performed.
        """
        return [info for _index, info in
                iter_slabs(self._model.atoms, miller, layers, vacuum)]

    def cut_termination(
        self,
        miller: tuple[int, int, int],
        layers: int = 4,
        vacuum: float = 15.0,
        termination: int = 0,
    ) -> StructureModel:
        """Build a StructureModel from the termination-th unique slab.

        Args:
            miller: Miller indices (h, k, l) of the surface.
            layers: Minimum number of atomic layers in the slab.
            vacuum: Vacuum spacing in Angstrom (centered).
            termination: Index into ``slabs()`` (0 = first).

        Returns:
            A new StructureModel containing the slab.

        Raises:
            IndexError: If ``termination`` is out of range.
        """
        return StructureModel(self.slabs(miller, layers, vacuum)[termination].atoms)

    def cut_with_thickness(
        self,
        miller: tuple[int, int, int],
        thickness: float = 10.0,
        vacuum: float = 15.0,
    ) -> StructureModel:
        """Generate a slab with a target thickness (Angstrom).

        Args:
            miller: Miller indices (h, k, l).
            thickness: Target slab thickness in Angstrom.
            vacuum: Vacuum spacing in Angstrom.

        Returns:
            A new StructureModel containing the slab.
        """
        atoms = self._model.atoms.copy()

        # Calculate interlayer spacing
        h, k, l = miller
        normal = np.array([h, k, l], dtype=float)
        d_spacing = 1.0 / np.linalg.norm(np.linalg.solve(atoms.get_cell().T, normal))

        # Estimate number of layers
        layers = max(1, int(thickness / d_spacing))

        slab_atoms = ase_surface(atoms, miller, layers=layers, vacuum=vacuum)
        return StructureModel(slab_atoms)
