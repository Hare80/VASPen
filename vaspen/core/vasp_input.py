"""VASP input file generation — INCAR, KPOINTS, POTCAR.

Default values follow community best practices for common VASP
calculation types.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from ase.dft.kpoints import get_special_points as ase_get_special_points

# ======================================================================
# INCAR Presets — community-standard defaults
# ======================================================================
# Each preset is a dict of tag → value. Values are written as-is to
# the INCAR file (strings, ints, floats, bools).
#
# Key design principles:
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
# KPOINTS Generation
# ======================================================================

# Recommended KSPACING values
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
    """Generate KPOINTS content from a KSPACING value.

    Computes an explicit regular k-mesh (Gamma-centered or
    Monkhorst-Pack) with N_i = max(1, ceil(|b_i| / KSPACING)), where
    b_i are the normalized reciprocal lattice vectors (b_i·a_j = δ_ij,
    rows of inv(cell)) and KSPACING follows the vaspkit convention
    (units of 2π/Å — equivalent to VASP's KSPACING tag with the value
    multiplied by 2π; see https://vasp.at/wiki/KSPACING).

    Args:
        structure_cell: 3×3 cell matrix (Angstrom).
        k_spacing: Target k-point spacing (2π/Å, vaspkit convention).
                   Recommended: 0.04 (insulators), 0.03 (metals).
        gamma_centered: True for Gamma-centered, False for Monkhorst-Pack.

    Returns:
        KPOINTS file content as a string.
    """
    mesh = estimate_k_mesh(np.asarray(structure_cell, dtype=float), k_spacing)
    scheme = "Gamma" if gamma_centered else "Monkhorst-Pack"
    return f"""Automatic k-point mesh (KSPACING = {k_spacing:.4f}, 2π/Å)
0
{scheme}
{mesh[0]} {mesh[1]} {mesh[2]}
0 0 0
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


def get_high_symmetry_points(cell) -> dict[str, np.ndarray]:
    """High-symmetry k-point labels → fractional coordinates for a cell.

    Uses spglib via ASE (labels of the conventional cell: G, X, M, R,
    K, L, W, ...).

    Args:
        cell: 3×3 cell matrix (Angstrom), array-like or ASE Cell.

    Returns:
        Dict mapping label strings to fractional coordinate arrays.
    """
    return ase_get_special_points(cell)


def generate_kpoints_line_mode(
    high_symmetry_path: list[tuple[str, str]],
    n_points_per_segment: int = 20,
    special_points: dict[str, np.ndarray] | None = None,
) -> str:
    """Generate KPOINTS content for band-structure calculations (Line-mode).

    Args:
        high_symmetry_path: List of (start_label, end_label) pairs.
            Example: [("G", "X"), ("X", "M"), ("M", "G")].
        n_points_per_segment: Number of k-points per path segment
            (default: 20). Each segment includes both endpoints;
            junction points appear once per adjacent segment.
        special_points: High-symmetry label → fractional coordinates,
            e.g. from get_high_symmetry_points(cell).

    Returns:
        KPOINTS file content as a string.

    Raises:
        ValueError: If special_points is missing or a path label is
            unknown for the cell.
    """
    if not special_points:
        raise ValueError(
            "Line-mode requires special_points (use get_high_symmetry_points "
            "with the structure cell)"
        )
    if n_points_per_segment < 2:
        raise ValueError("n_points_per_segment must be at least 2")

    n_segments = len(high_symmetry_path)
    header = "Band structure: " + "|".join(
        f"{a}-{b}" for a, b in high_symmetry_path
    )

    lines: list[str] = [
        header,
        str(n_points_per_segment * n_segments),
        "Line-mode",
        "Reciprocal",
    ]
    for start_label, end_label in high_symmetry_path:
        for label in (start_label, end_label):
            if label not in special_points:
                available = ", ".join(sorted(special_points))
                raise ValueError(
                    f"Unknown k-point label '{label}' for this cell. "
                    f"Available: {available}"
                )
        start = np.asarray(special_points[start_label], dtype=float)
        end = np.asarray(special_points[end_label], dtype=float)
        for i in range(n_points_per_segment):
            frac = i / (n_points_per_segment - 1)
            point = start + (end - start) * frac
            # weight 1.0 marks band-structure vertices (segment ends)
            weight = 1.0 if i in (0, n_points_per_segment - 1) else 0.0
            lines.append(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {weight:.1f}"
            )
    return "\n".join(lines) + "\n"


def estimate_k_mesh(
    cell: np.ndarray,
    target_spacing: float = 0.04,
) -> tuple[int, int, int]:
    """Estimate a k-mesh (n1, n2, n3) from cell and target spacing.

    Follows the vaspkit convention (KSPACING in units of 2π/Å):
        n_i = max(1, ceil(|b_i| / KSPACING))
    where b_i are the normalized reciprocal lattice vectors
    (b_i·a_j = δ_ij — rows of inv(cell), no 2π factor). This is
    equivalent to VASP's KSPACING tag formula on the wiki
    (https://vasp.at/wiki/KSPACING: n_i = max(1, ceil(2π·|b_i| /
    KSPACING_vasp))) with KSPACING_vasp = 2π · KSPACING_input.

    Args:
        cell: 3×3 real-space cell matrix (Angstrom).
        target_spacing: Target KSPACING (2π/Å, vaspkit convention).

    Returns:
        (k1, k2, k3) integer mesh.
    """
    # Normalized reciprocal lattice vectors (b_i·a_j = δ_ij)
    recip = np.linalg.inv(cell).T
    lengths = np.linalg.norm(recip, axis=1)
    mesh = np.maximum(1, np.ceil(lengths / target_spacing))
    return tuple(int(m) for m in mesh)


# ======================================================================
# POTCAR Generation
# ======================================================================
# Library directory names and per-element default variants follow the
# official VASP wiki: https://vasp.at/wiki/Available_pseudopotentials
# (each library's "Standard potentials" table marks the recommended
# default variant per element in bold).

POTCAR_DEFAULT_FUNCTIONAL = "PBE"

# Functional → candidate version-directory names (VASP distribution
# names, per the wiki). The first existing directory is used, so a
# library that only ships a newer release (e.g. only .64) still works;
# the bare short names are fallbacks for libraries without the
# potpaw_ prefix.
POTCAR_FUNCTIONAL_VERSIONS: dict[str, list[str]] = {
    "PBE": ["potpaw_PBE.54", "potpaw_PBE.64", "PBE.54", "PBE.64"],
    "PBE_new": ["potpaw_PBE.64", "PBE.64"],   # wiki: latest, recommended
    "LDA": ["potpaw_LDA.54", "potpaw_LDA.64", "LDA.54", "LDA.64"],
    "PW91": ["potpaw_GGA", "PW91.54", "GGA"],  # potpaw_GGA = PW91 (2006)
}

# Wiki-recommended default variant per element (the bold table entry).
# Elements not listed use the plain variant (e.g. "Fe").
_SV_FULL = {
    "Li": "Li_sv", "K": "K_sv", "Ca": "Ca_sv", "Sc": "Sc_sv", "Ti": "Ti_sv",
    "V": "V_sv", "Rb": "Rb_sv", "Sr": "Sr_sv", "Y": "Y_sv", "Zr": "Zr_sv",
    "Nb": "Nb_sv", "Mo": "Mo_sv", "Cs": "Cs_sv", "Ba": "Ba_sv",
    "W": "W_sv", "Fr": "Fr_sv", "Ra": "Ra_sv",
}
_PV = {
    "Na": "Na_pv", "Cr": "Cr_pv", "Mn": "Mn_pv", "Tc": "Tc_pv",
    "Ru": "Ru_pv", "Rh": "Rh_pv", "Hf": "Hf_pv", "Ta": "Ta_pv",
}
_D_FULL = {
    "Ga": "Ga_d", "Ge": "Ge_d", "In": "In_d", "Sn": "Sn_d",
    "Tl": "Tl_d", "Pb": "Pb_d", "Bi": "Bi_d", "Po": "Po_d",
}
_LANTHANIDES = {
    "Pr": "Pr_3", "Nd": "Nd_3", "Pm": "Pm_3", "Sm": "Sm_3",
    "Eu": "Eu_2", "Gd": "Gd_3", "Tb": "Tb_3", "Dy": "Dy_3",
    "Ho": "Ho_3", "Er": "Er_3", "Tm": "Tm_3", "Yb": "Yb_2", "Lu": "Lu_3",
}
# PW91 (2010) has no Mo_sv/W_sv/Fr_sv/Ra_sv and no Po_d
_SV_PW91 = {k: v for k, v in _SV_FULL.items() if k not in ("Mo", "W", "Fr", "Ra")}
_D_PW91 = {k: v for k, v in _D_FULL.items() if k != "Po"}

_PBE_SPECIAL = {**_SV_FULL, **_PV, **_D_FULL, **_LANTHANIDES}

POTCAR_SPECIAL_RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "PBE": _PBE_SPECIAL,
    "PBE_new": _PBE_SPECIAL,  # .54 and .64 share the same lists on the wiki
    "LDA": {**_SV_FULL, **_PV, **_D_FULL},  # no fixed-valence lanthanides
    "PW91": {**_SV_PW91, **_PV, **_D_PW91, **_LANTHANIDES},
}


def get_potcar_recommendation(element: str, functional: str = "PBE") -> str:
    """Get the wiki-recommended POTCAR variant for an element.

    Based on the bold (default) entries in the "Standard potentials"
    tables at https://vasp.at/wiki/Available_pseudopotentials.

    Args:
        element: Element symbol (e.g. "Ga", "Fe").
        functional: Exchange-correlation functional ("LDA", "PBE",
            "PBE_new", "PW91").

    Returns:
        POTCAR variant name (e.g. "Ga_d" for Ga; "Fe" when the plain
        potential is recommended).

    Raises:
        ValueError: If functional is not recognized.
    """
    if functional not in POTCAR_FUNCTIONAL_VERSIONS:
        raise ValueError(f"Unknown functional: {functional}. "
                         f"Known: {list(POTCAR_FUNCTIONAL_VERSIONS)}")
    return POTCAR_SPECIAL_RECOMMENDATIONS.get(functional, {}).get(element, element)


def resolve_potcar_dir(library: str | Path, functional: str) -> Path:
    """First existing version directory for the functional in a library.

    Falls back to the primary candidate name if none exists (the
    missing-library error is reported by generate_potcar).
    """
    names = POTCAR_FUNCTIONAL_VERSIONS[functional]
    lib = Path(library)
    for name in names:
        d = lib / name
        if d.exists():
            return d
    return lib / names[0]


def available_variants(
    library: str | Path,
    functional: str,
    element: str,
) -> list[str]:
    """POTCAR variant names available for an element in the library.

    Discovers variants from the library directory itself — any
    subdirectory of the version dir named ``{element}``,
    ``{element}_{suffix}`` (``_d``/``_sv``/``_pv``/``_s``/``_h``/
    ``_GW``/``_AE``/``_2``/``_3``/…) or fractional names like ``H.5`` —
    so every variant listed on the VASP wiki is recognized without a
    hardcoded list. The wiki-recommended default comes first.

    Returns:
        Ordered variant names (may be empty if the library is missing).
    """
    version_dir = resolve_potcar_dir(library, functional)
    variants: set[str] = set()
    if version_dir.exists():
        pattern = rf"^{re.escape(element)}(_[A-Za-z]+)?(\.\d+)?$"
        for d in version_dir.iterdir():
            if d.is_dir() and re.match(pattern, d.name):
                variants.add(d.name)
    recommended = get_potcar_recommendation(element, functional)
    ordered = [recommended] if recommended in variants else []
    ordered += sorted(variants - set(ordered))
    return ordered


def generate_potcar(
    elements: list[str],
    potcar_library_path: str | Path,
    functional: str = "PBE",
) -> tuple[str, list[str]]:
    """Generate POTCAR content by concatenating individual POTCAR files.

    Concatenates per-element POTCAR files in POSCAR order:
        1. For each element, find the recommended POTCAR directory.
        2. Concatenate in element order (as they appear in POSCAR).
        3. Note that the order matters — POTCAR must match POSCAR atom order.

    Args:
        elements: List of element symbols in POSCAR order.
        potcar_library_path: Root path to the pseudopotential library
            (standard VASP pseudopotential layout).
        functional: "LDA", "PBE", or "PW91".

    Returns:
        Tuple of (concatenated POTCAR content, list of file paths used).

    Raises:
        FileNotFoundError: If a POTCAR file is missing.
    """
    library = Path(potcar_library_path)
    potcar_dir = resolve_potcar_dir(library, functional)

    if not potcar_dir.exists():
        available = ""
        if library.exists():
            found = sorted(
                d.name for d in library.iterdir() if d.is_dir()
            )
            if found:
                available = f"\nFound in the library root: {', '.join(found)}"
        raise FileNotFoundError(
            f"POTCAR library for functional '{functional}' not found.\n"
            f"Tried: {potcar_dir}{available}\n"
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
    poscar_direct: bool = False,
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
        poscar_direct: Write POSCAR in fractional (Direct) coordinates
            instead of Cartesian.

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
        special = get_high_symmetry_points(structure_model.atoms.get_cell()[:])
        kpoints_content = generate_kpoints_line_mode(
            high_symmetry_path=kp.get("path", [("G", "X"), ("X", "M"), ("M", "G")]),
            n_points_per_segment=kp.get("n_points", 20),
            special_points=special,
        )
    else:
        raise ValueError(f"Unknown kpoints mode: {kpoints_mode}")

    # POSCAR
    from ase.io import write as ase_write
    import io

    from vaspen.core.file_io import vasp_write_atoms

    buf = io.StringIO()
    ase_write(buf, vasp_write_atoms(structure_model.atoms, poscar_direct),
              format="vasp", vasp5=True, direct=poscar_direct)
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
