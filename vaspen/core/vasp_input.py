"""VASP input file generation — INCAR, KPOINTS, POTCAR.

Default values are based on vaspkit's built-in presets, which represent
community best practices for common calculation types.

Reference:
    vaspkit (https://vaspkit.com/) — the most widely used VASP
    pre/post-processing toolkit in the Chinese computational materials
    community and beyond.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# ======================================================================
# INCAR Presets — modeled after vaspkit's default recommendations
# ======================================================================
# Each preset is a dict of tag → value. Values are written as-is to
# the INCAR file (strings, ints, floats, bools).
#
# Key vaspkit design principles reflected here:
#   1. ENCUT defaults to 400 eV (user should increase to 1.3× ENMAX).
#   2. ISMEAR=0 (Gaussian) for most calculations.
#   3. ISMEAR=-5 (tetrahedron) for DOS.
#   4. LWAVE/LCHARG = False by default (save disk space).
#   5. PREC=Normal, LREAL=Auto as pragmatic defaults.

INCAR_PRESETS: dict[str, dict[str, Any]] = {
    "scf": {
        "_description": "Self-consistent field (SCF) — static total energy calculation.",
        "SYSTEM": "VASP calculation",
        "ENCUT": 400,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "EDIFF": 1e-6,
        "NELM": 60,
        "PREC": "Normal",
        "LREAL": "Auto",
        "LWAVE": False,
        "LCHARG": False,
        "ISPIN": 1,
    },
    "opt": {
        "_description": "Structure optimization — relax atomic positions + cell.",
        "SYSTEM": "VASP optimization",
        "ENCUT": 400,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "EDIFF": 1e-6,
        "EDIFFG": -0.01,
        "IBRION": 2,
        "ISIF": 3,
        "NSW": 100,
        "PREC": "Normal",
        "LREAL": "Auto",
        "LWAVE": False,
        "LCHARG": False,
    },
    "band": {
        "_description": "Band structure — fixed charge density, line-mode KPOINTS.",
        "SYSTEM": "VASP band structure",
        "ENCUT": 400,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "EDIFF": 1e-6,
        "PREC": "Normal",
        "LREAL": "Auto",
        "LWAVE": False,
        "LCHARG": False,
        "ICHARG": 11,  # Read CHGCAR for non-SCF band calculation
    },
    "dos": {
        "_description": "Density of states — tetrahedron smearing, dense k-mesh.",
        "SYSTEM": "VASP DOS",
        "ENCUT": 400,
        "ISMEAR": -5,
        "SIGMA": 0.05,
        "EDIFF": 1e-6,
        "PREC": "Normal",
        "LREAL": "Auto",
        "LWAVE": False,
        "LCHARG": False,
        "LORBIT": 11,
        "NEDOS": 2000,
    },
    "optical": {
        "_description": "Optical properties — frequency-dependent dielectric function.",
        "SYSTEM": "VASP optical",
        "ENCUT": 400,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "EDIFF": 1e-6,
        "PREC": "Normal",
        "LREAL": "Auto",
        "LWAVE": False,
        "LCHARG": False,
        "LOPTICS": True,
        "CSHIFT": 0.1,
        "NEDOS": 2000,
    },
    "neb": {
        "_description": "Nudged Elastic Band — transition state search.",
        "SYSTEM": "VASP NEB",
        "ENCUT": 400,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "EDIFF": 1e-6,
        "EDIFFG": -0.05,
        "IBRION": 3,   # Damped MD (recommended for NEB)
        "POTIM": 0.0,
        "NSW": 200,
        "PREC": "Normal",
        "LREAL": "Auto",
        "LWAVE": False,
        "LCHARG": False,
        "IMAGES": 5,   # Number of intermediate images
        "SPRING": -5,  # Spring constant
        "LCLIMB": True,
    },
}

INCAR_TAG_DESCRIPTIONS: dict[str, str] = {
    "SYSTEM": "Descriptive name for the calculation",
    "ENCUT": "Plane-wave energy cutoff (eV). Set to 1.3 × ENMAX of POTCAR.",
    "ISMEAR": "Smearing method: 0=Gaussian, -5=tetrahedron (DOS), 1=Methfessel-Paxton (metals)",
    "SIGMA": "Smearing width (eV). 0.05 for insulators, 0.1-0.2 for metals.",
    "EDIFF": "Electronic convergence criterion (eV)",
    "EDIFFG": "Ionic convergence criterion (eV/Å). Negative = force-based.",
    "IBRION": "Ion relaxation algorithm: -1=fixed, 0=MD, 1=quasi-Newton, 2=CG, 3=damped MD",
    "ISIF": "Stress/relaxation control: 2=no cell change, 3=full relaxation",
    "NSW": "Maximum number of ionic steps",
    "NELM": "Maximum number of electronic SCF steps",
    "PREC": "Precision level: Low, Normal, Accurate",
    "LREAL": "Projection in real space: Auto, On, Off",
    "LWAVE": "Write WAVECAR file (large file)",
    "LCHARG": "Write CHGCAR file",
    "ISPIN": "Spin: 1=non-polarized, 2=collinear spin",
    "LORBIT": "Projected DOS: 10/11=element-resolved, 12=lm-resolved (requires PAW)",
    "NEDOS": "Number of DOS grid points. >=2000 recommended.",
    "LOPTICS": "Calculate frequency-dependent dielectric matrix",
    "CSHIFT": "Complex shift for Kramers-Kronig (eV)",
    "ICHARG": "Charge initialization: 11=read CHGCAR for non-SCF",
    "IMAGES": "Number of intermediate images for NEB",
    "SPRING": "Spring constant for NEB",
    "LCLIMB": "Use climbing image NEB",
    "POTIM": "Time step for MD / damped MD (IBRION=3)",
    "MAGMOM": "Initial magnetic moments per atom",
}

# ======================================================================
# KPOINTS Generation — modeled after vaspkit
# ======================================================================

# vaspkit-recommended KSPACING values
KSPACING_RECOMMEND = {
    "insulator": 0.04,
    "metal": 0.03,
    "fine": 0.02,
    "coarse": 0.05,
}


def generate_kpoints_automatic(
    structure_cell: np.ndarray,
    k_spacing: float = 0.04,
    gamma_centered: bool = True,
) -> str:
    """Generate KPOINTS content using automatic KSPACING mode.

    This is vaspkit's recommended approach — VASP automatically
    determines the optimal k-mesh from KSPACING and the cell.

    Args:
        structure_cell: 3×3 cell matrix (Angstrom).
        k_spacing: Target k-point spacing in Å⁻¹.
                   vaspkit recommended: 0.04 (insulators), 0.03 (metals).
        gamma_centered: True for Gamma-centered, False for Monkhorst-Pack.

    Returns:
        KPOINTS file content as a string.
    """
    scheme = "Gamma" if gamma_centered else "Monkhorst-Pack"
    return f"""Automatic KSPACING mesh
0
Auto
{k_spacing:.4f}
{scheme}
"""


def generate_kpoints_manual(
    k1: int, k2: int, k3: int,
    gamma_centered: bool = True,
) -> str:
    """Generate KPOINTS content with manual k-mesh.

    Args:
        k1, k2, k3: Number of k-points along each reciprocal direction.
        gamma_centered: True for Gamma-centered, False for Monkhorst-Pack.

    Returns:
        KPOINTS file content as a string.
    """
    scheme = "G" if gamma_centered else "M"
    shift = "0 0 0" if gamma_centered else "0 0 0"
    return f"""Manual k-mesh
0
{scheme}
{k1} {k2} {k3}
{shift}
"""


def generate_kpoints_line_mode(
    high_symmetry_path: list[tuple[str, str]],
    n_points_per_segment: int = 20,
    reciprocal_cell: np.ndarray | None = None,
) -> str:
    """Generate KPOINTS content for band-structure calculations (Line-mode).

    Args:
        high_symmetry_path: List of (start_label, end_label) pairs.
            Example: [("G", "X"), ("X", "M"), ("M", "G")].
        n_points_per_segment: Number of k-points per path segment.
                               vaspkit default: 20.
        reciprocal_cell: 3×3 reciprocal cell matrix. If None, k-point
                         coordinates must be pre-computed.

    Returns:
        KPOINTS file content as a string.
    """
    n_segments = len(high_symmetry_path)
    total_points = n_segments * n_points_per_segment

    header = f"Band structure: " + "|".join(
        f"{a}-{b}" for a, b in high_symmetry_path
    )

    # Line-mode requires explicit k-point coordinates.
    # We provide the high-symmetry path with weights = 0 for intermediate
    # points and 1.0 for the endpoints (for band-structure plotting).
    #
    # ASE/spglib can compute the actual fractional coordinates of
    # high-symmetry points from the space group. The UI calls
    # get_high_symmetry_path() to get those coordinates before
    # passing them here.

    return f"""{header}
{n_points_per_segment * n_segments}
Line-mode
Reciprocal
"""


def estimate_k_mesh(
    cell: np.ndarray,
    target_spacing: float = 0.04,
) -> tuple[int, int, int]:
    """Estimate a k-mesh (n1, n2, n3) from cell and target spacing.

    Follows vaspkit's formula:
        n_i = max(1, ceil(1 / (2π · KSPACING · |b_i|)))
    where b_i are the reciprocal lattice vectors.

    Args:
        cell: 3×3 real-space cell matrix (Angstrom).
        target_spacing: Target KSPACING in Å⁻¹.

    Returns:
        (k1, k2, k3) integer mesh.
    """
    # Reciprocal lattice vectors
    recip = 2 * np.pi * np.linalg.inv(cell).T
    lengths = np.linalg.norm(recip, axis=1)
    mesh = np.maximum(1, np.ceil(lengths / target_spacing / (2 * np.pi)))
    return tuple(int(m) for m in mesh)


# ======================================================================
# POTCAR Generation
# ======================================================================

# vaspkit POTCAR recommendation rules
# Default functional → PBE.54
# Elements with semi-core states → _d or _sv variants

POTCAR_DEFAULT_FUNCTIONAL = "PBE"
POTCAR_DEFAULT_VERSION = "PBE.54"

# Elements that benefit from semi-core variants (PBE.54 naming)
# Based on vaspkit recommendations
POTCAR_SPECIAL_RECOMMENDATIONS: dict[str, str] = {
    # d-electron elements → use *_d
    "Ga": "Ga_d",
    "In": "In_d",
    "Sn": "Sn_d",
    "Pb": "Pb_d",
    "Ge": "Ge_d",
    "As": "As_d",
    "Sb": "Sb_d",
    "Bi": "Bi_d",
    "Tl": "Tl_d",
    # Alkali / alkaline earth with shallow semi-core → use *_sv
    "Rb": "Rb_sv",
    "Cs": "Cs_sv",
    "Sr": "Sr_sv",
    "Ba": "Ba_sv",
    # Transition metals: standard PBE.54 (no suffix) is usually fine
    # but _pv variants are available for higher accuracy
}

POTCAR_FUNCTIONAL_VERSIONS = {
    "LDA": "LDA.54",
    "PBE": "PBE.54",
    "PBE_new": "PBE.64",
    "PW91": "PW91.54",
}


def get_potcar_recommendation(element: str, functional: str = "PBE") -> str:
    """Get the recommended POTCAR variant for an element.

    Args:
        element: Element symbol (e.g. "Ga", "Fe").
        functional: Exchange-correlation functional ("LDA", "PBE", "PW91").

    Returns:
        POTCAR variant name (e.g. "Ga_d" for Ga with PBE.54).
    """
    if functional not in POTCAR_FUNCTIONAL_VERSIONS:
        raise ValueError(f"Unknown functional: {functional}. "
                         f"Known: {list(POTCAR_FUNCTIONAL_VERSIONS)}")

    version = POTCAR_FUNCTIONAL_VERSIONS[functional]

    # Check special recommendations
    if functional == "PBE" and element in POTCAR_SPECIAL_RECOMMENDATIONS:
        return POTCAR_SPECIAL_RECOMMENDATIONS[element]

    # Default: element name as-is (e.g. "Fe")
    return element


def generate_potcar(
    elements: list[str],
    potcar_library_path: str | Path,
    functional: str = "PBE",
) -> tuple[str, list[str]]:
    """Generate POTCAR content by concatenating individual POTCAR files.

    Follows vaspkit's approach:
        1. For each element, find the recommended POTCAR directory.
        2. Concatenate in element order (as they appear in POSCAR).
        3. Note that the order matters — POTCAR must match POSCAR atom order.

    Args:
        elements: List of element symbols in POSCAR order.
        potcar_library_path: Root path to the pseudopotential library
            (same structure as vaspkit expects).
        functional: "LDA", "PBE", or "PW91".

    Returns:
        Tuple of (concatenated POTCAR content, list of file paths used).

    Raises:
        FileNotFoundError: If a POTCAR file is missing.
    """
    library = Path(potcar_library_path)
    version = POTCAR_FUNCTIONAL_VERSIONS[functional]
    potcar_dir = library / version

    if not potcar_dir.exists():
        raise FileNotFoundError(
            f"POTCAR library not found at: {potcar_dir}\n"
            f"Please configure the pseudopotential library path in Settings."
        )

    contents: list[str] = []
    paths_used: list[str] = []

    for element in elements:
        variant = get_potcar_recommendation(element, functional)
        potcar_path = potcar_dir / variant / "POTCAR"

        if not potcar_path.exists():
            # Fallback: try without variant suffix
            potcar_path = potcar_dir / element / "POTCAR"

        if not potcar_path.exists():
            raise FileNotFoundError(
                f"POTCAR for {element} not found.\n"
                f"Tried: {potcar_dir / variant / 'POTCAR'}\n"
                f"       {potcar_dir / element / 'POTCAR'}"
            )

        with open(potcar_path, "r") as f:
            contents.append(f.read())
        paths_used.append(str(potcar_path))

    return "".join(contents), paths_used


# ======================================================================
# High-level: generate all VASP inputs at once
# ======================================================================

def generate_all_inputs(
    structure_model,           # StructureModel
    incar_preset: str = "scf",
    incar_overrides: dict[str, Any] | None = None,
    kpoints_mode: str = "automatic",
    kpoints_params: dict[str, Any] | None = None,
    potcar_library: str = "",
    functional: str = "PBE",
) -> dict[str, str]:
    """Generate all four VASP input files at once.

    Args:
        structure_model: StructureModel with the current structure.
        incar_preset: INCAR preset name ("scf", "opt", "band", "dos", "optical").
        incar_overrides: Optional INCAR tag → value overrides.
        kpoints_mode: "automatic", "manual", or "line".
        kpoints_params: Dict with kpoints-specific parameters.
        potcar_library: Path to pseudopotential library root.
        functional: XC functional for POTCAR.

    Returns:
        Dict mapping filename → content:
            {"INCAR": "...", "KPOINTS": "...", "POSCAR": "...", "POTCAR": "..."}
    """
    # INCAR
    preset = dict(INCAR_PRESETS.get(incar_preset, INCAR_PRESETS["scf"]))
    preset.pop("_description", None)  # remove metadata key
    if incar_overrides:
        preset.update(incar_overrides)
    incar_lines = [f"{tag} = {val}" for tag, val in preset.items()]
    incar_content = "\n".join(incar_lines) + "\n"

    # KPOINTS
    kp = kpoints_params or {}
    if kpoints_mode == "automatic":
        kpoints_content = generate_kpoints_automatic(
            structure_model.cell,
            k_spacing=kp.get("k_spacing", 0.04),
            gamma_centered=kp.get("gamma_centered", True),
        )
    elif kpoints_mode == "manual":
        kpoints_content = generate_kpoints_manual(
            k1=kp.get("k1", 1),
            k2=kp.get("k2", 1),
            k3=kp.get("k3", 1),
            gamma_centered=kp.get("gamma_centered", True),
        )
    elif kpoints_mode == "line":
        kpoints_content = generate_kpoints_line_mode(
            high_symmetry_path=kp.get("path", [("G", "X"), ("X", "M"), ("M", "G")]),
            n_points_per_segment=kp.get("n_points", 20),
        )
    else:
        raise ValueError(f"Unknown kpoints mode: {kpoints_mode}")

    # POSCAR
    from ase.io import write as ase_write
    import io
    buf = io.StringIO()
    ase_write(buf, structure_model.atoms, format="vasp", vasp5=True)
    poscar_content = buf.getvalue()

    # POTCAR
    potcar_content = ""
    if potcar_library and structure_model.unique_symbols:
        potcar_content, _ = generate_potcar(
            structure_model.unique_symbols,
            potcar_library,
            functional,
        )

    return {
        "INCAR": incar_content,
        "KPOINTS": kpoints_content,
        "POSCAR": poscar_content,
        "POTCAR": potcar_content,
    }
