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
from ase.build import cut as ase_cut
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
    """Interplanar spacing along the Miller normal (Angstrom)."""
    normal = np.array(miller, dtype=float)
    return 1.0 / np.linalg.norm(np.linalg.solve(atoms.get_cell().T, normal))


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


def supercell_in_plane(atoms: Atoms, a: int, b: int) -> Atoms:
    """Repeat the slab a×b in the surface plane (c untouched).

    A supercell along the vacuum direction is meaningless for a slab,
    so the c repeat factor is fixed at 1.
    """
    return atoms.repeat((a, b, 1))


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

        Pipeline: AseAtomsAdaptor.get_structure -> SlabGenerator(
        miller, min_slab_size=layers*d_hkl, min_vacuum_size=vacuum,
        center_slab=True).get_slabs() -> get_atoms per slab.
        ``min_slab_size`` is a MINIMUM (pymatgen ceils to whole
        oriented unit cells, and the ceil is float-sensitive), so the
        slab may contain one extra layer. Sorted as returned by
        get_slabs (stable order; broken-bond count). Initial magnetic
        moments round-trip through site properties automatically.

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
        struct = AseAtomsAdaptor.get_structure(self._model.atoms)
        d = _d_hkl(self._model.atoms, miller)
        gen = SlabGenerator(
            struct,
            miller,
            min_slab_size=layers * d,
            min_vacuum_size=float(vacuum),
            center_slab=True,
        )
        infos: list[SlabInfo] = []
        # filter_out_sym_slabs=False: keep mirror-pair cleavages as
        # separate choices (e.g. SrTiO3 (100) offers SrO-top and
        # TiO2-top slabs). pymatgen's default symmetry dedup would
        # collapse translation-equivalent mirrors to a single entry,
        # hiding the top/bottom face choice from the user.
        for slab in gen.get_slabs(filter_out_sym_slabs=False):
            atoms = AseAtomsAdaptor.get_atoms(slab)
            top, bottom = _surface_compositions(atoms)
            infos.append(SlabInfo(
                atoms=atoms,
                top_composition=top,
                bottom_composition=bottom,
                broken_bonds=int(slab.energy or 0),
                n_atoms=len(atoms),
            ))
        return infos

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

    @staticmethod
    def cut_arbitrary(
        model: StructureModel,
        origin: np.ndarray,
        normal: np.ndarray,
    ) -> StructureModel:
        """Cut a slab with arbitrary origin and normal vectors.

        Uses ASE's low-level ``cut`` function.

        Args:
            model: The bulk structure.
            origin: (x, y, z) origin of the cut plane.
            normal: (x, y, z) normal vector of the cut plane.

        Returns:
            New StructureModel with the cut structure.
        """
        atoms = model.atoms.copy()
        origin = np.asarray(origin, dtype=float)
        normal = np.asarray(normal, dtype=float)

        # ASE cut returns a new Atoms with the specified plane
        cut_atoms = ase_cut(atoms, a=normal, b=None, c=None, origo=origin, nlayers=None)
        # Re-wrap
        cut_atoms.center(vacuum=10.0)
        return StructureModel(cut_atoms)
