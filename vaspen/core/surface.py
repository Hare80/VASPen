"""Surface / slab generation via Miller-index cutting.

Uses ASE's ``surface`` and ``cut`` modules. Provides a clean API
for the UI layer.
"""

from __future__ import annotations

import numpy as np
from ase.build import cut as ase_cut
from ase.build import surface as ase_surface

from vaspen.core.structure import StructureModel


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

        Args:
            miller: Miller indices (h, k, l) of the surface.
            layers: Number of atomic layers in the slab.
            vacuum: Vacuum spacing in Angstrom added above the slab.

        Returns:
            A new StructureModel containing the slab.
        """
        atoms = self._model.atoms.copy()
        slab_atoms = ase_surface(atoms, miller, layers=layers, vacuum=vacuum)
        return StructureModel(slab_atoms)

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
