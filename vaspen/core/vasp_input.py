"""VASP input file generation — INCAR, KPOINTS, POTCAR.

Default values follow community best practices for common VASP
calculation types.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from ase.dft.kpoints import get_special_points as ase_get_special_points
from PySide6.QtCore import QCoreApplication


def _tr(text: str) -> str:
    """Translate a user-visible VaspInput message (hand-maintained .ts)."""
    return QCoreApplication.translate("VaspInput", text)

# ======================================================================
# INCAR Presets — community-standard defaults
# ======================================================================
# Each preset is a dict of tag → value. Values are written as-is to
# the INCAR file (strings, ints, floats, bools).
#
# Key design principles (confirmed with the user 2026-08-14):
#   1. ENCUT defaults to 400 eV (user should increase to 1.3× ENMAX).
#   2. ISMEAR=0 (Gaussian) for most calculations.
#   3. ISMEAR=-5 (tetrahedron) for DOS.
#   4. LWAVE/LCHARG = False by default (save disk space); the band
#      preset writes CHGCAR because ICHARG=11 reads it back.
#   5. PREC=Normal, LREAL=Auto as pragmatic defaults.
#   6. Generated lines carry aligned (comments) — see
#      format_incar_content(); commented-out suggestion tags
#      (INCAR_SUGGESTIONS) follow after a blank line.

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
        "ISPIN": 1,
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
        "LCHARG": True,   # ICHARG=11 reads the CHGCAR written by the SCF run
        "ICHARG": 11,     # Read CHGCAR for non-SCF band calculation
        "ISPIN": 1,
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
        "NEDOS": 2001,   # odd: the Fermi level lands on a DOS grid point
        "ISPIN": 1,
    },
    "optical": {
        "_description": "Optical properties — frequency-dependent dielectric function.",
        "SYSTEM": "VASP optical",
        "ENCUT": 400,
        "ISMEAR": 0,
        "SIGMA": 0.01,
        "EDIFF": 1e-6,
        "PREC": "Normal",
        "LREAL": "Auto",
        "LWAVE": False,
        "LCHARG": False,
        "LOPTICS": True,
        "CSHIFT": 0.1,
        "NEDOS": 2000,
        "ISPIN": 1,
    },
    "neb": {
        "_description": "Nudged Elastic Band — transition state search.",
        "SYSTEM": "VASP NEB",
        "ENCUT": 400,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "EDIFF": 1e-6,
        "EDIFFG": -0.02,  # force criterion (eV/A)
        "IBRION": 3,   # damped MD — ion motion is taken over by IOPT
        "POTIM": 0.0,  # zero time step: VASP itself never moves the ions
        "IOPT": 1,     # 1-LBFGS (optimizes the whole band globally)
        "ICHAIN": 0,   # 0-NEB, 2-dimer, 3-Lanczos
        "NSW": 500,
        "PREC": "Normal",
        "LREAL": "Auto",
        "LWAVE": False,
        "LCHARG": False,
        "IMAGES": "",  # required: number of intermediate images — system-
                       # dependent, the user must fill it in (settled
                       # 2026-08-14; both generation paths block on empty)
        "SPRING": -5,  # Spring constant
        "LCLIMB": True,
        "ISPIN": 1,
    },
}

# Short trailing comments for generated INCAR lines (written in
# parentheses after the value; SYSTEM has none — VASP takes the whole
# line after '=' as the system name).
INCAR_TAG_COMMENTS: dict[str, str] = {
    "ENCUT": "plane-wave cutoff in eV; set to 1.3 x ENMAX of POTCAR",
    "ISMEAR": "smearing: 0-Gaussian, 1-MP (metals), -5-tetrahedron",
    "SIGMA": "smearing width in eV",
    "EDIFF": "electronic convergence in eV",
    "EDIFFG": "ionic convergence in eV/A; negative = force",
    "IBRION": "ions: -1-fixed, 0-MD, 1-quasi-Newton, 2-CG, 3-damped MD",
    "ISIF": "2-ions only, 3-ions + cell",
    "NSW": "max ionic steps",
    "NELM": "max electronic SCF steps",
    "PREC": "precision: Low, Normal, Accurate",
    "LREAL": "real-space projection: Auto, On, Off",
    "LWAVE": "write WAVECAR; very large file",
    "LCHARG": "write CHGCAR",
    "LORBIT": "projected DOS: 11-element resolved",
    "NEDOS": "DOS grid points",
    "LOPTICS": "frequency-dependent dielectric matrix",
    "CSHIFT": "complex shift for Kramers-Kronig in eV",
    "ICHARG": "charge init: 11-read CHGCAR (non-SCF)",
    "ISPIN": "spin: 1-non-polarized, 2-collinear",
    "IMAGES": "no. of intermediate images (required)",
    "IOPT": "1-LBFGS, 2-CG, 3-quick-min, 7-FIRE",
    "ICHAIN": "0-NEB, 2-dimer, 3-Lanczos",
    "LNEBCELL": "variable-cell NEB",
    "SPRING": "spring constant in eV/A^2",
    "LCLIMB": "climbing image NEB",
    "POTIM": "time step in fs (MD); 0 = VASP does not move the ions",
    "MAGMOM": "initial magnetic moments per atom; with ISPIN=2",
    "ISTART": "1-read existing WAVECAR if present",
    "IVDW": "11-DFT-D3 van der Waals correction",
}

# Commented-out suggestion lines appended after a blank line. They are
# inactive until the user removes the '# '; suggestions that are
# already active tags are skipped so a tag never appears twice.
_COMMON_SUGGESTIONS: dict[str, Any] = {
    "MAGMOM": "",   # initial magnetic moments; fill per atom
    "IVDW": 11,
    "ISTART": 1,
}

INCAR_SUGGESTIONS: dict[str, dict[str, Any]] = {
    preset: dict(_COMMON_SUGGESTIONS) for preset in INCAR_PRESETS
}
# NEB-specific extra suggestion (variable-cell NEB)
INCAR_SUGGESTIONS["neb"]["LNEBCELL"] = False

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
    "IOPT": "Optimizer for force-based methods: 1-LBFGS, 2-CG, 3-quick-min, 7-FIRE",
    "ICHAIN": "Method: 0-NEB, 2-dimer, 3-Lanczos",
    "LNEBCELL": "Variable-cell NEB",
    "SPRING": "Spring constant for NEB",
    "LCLIMB": "Use climbing image NEB",
    "POTIM": "Time step for MD / damped MD (IBRION=3); 0 = VASP does not move the ions",
    "MAGMOM": "Initial magnetic moments per atom",
}


# ======================================================================
# INCAR text formatting
# ======================================================================

def magmom_line(magmoms: np.ndarray) -> str:
    """Format per-atom moments as a MAGMOM value list.

    One value per atom in atom order (collinear); NaN (unset) atoms are
    written as 0.0 (settled 2026-08-14).
    """
    return " ".join(
        "0" if np.isnan(v) else f"{v:g}" for v in np.asarray(magmoms, dtype=float)
    )


def _format_incar_value(val: Any) -> str:
    """Format an INCAR value VASP-style (bools as .TRUE./.FALSE.)."""
    if isinstance(val, bool):
        return ".TRUE." if val else ".FALSE."
    if isinstance(val, float):
        return f"{val:g}".replace("e", "E")
    text = str(val).strip()
    lowered = text.lower()
    if lowered in ("true", ".true.", "t"):
        return ".TRUE."
    if lowered in ("false", ".false.", "f"):
        return ".FALSE."
    return text


def _format_incar_line(
    tag: str,
    value: Any,
    comment: str | None,
    commented: bool,
) -> str:
    """One INCAR line: aligned tag/value with the comment at column 26."""
    body = f"{tag:<7}=  {_format_incar_value(value):<13}"
    prefix = "  # " if commented else "  "
    if comment:
        return f"{prefix}{body}({comment})"
    return (prefix + body).rstrip()


def format_incar_content(
    tags: dict[str, Any],
    comments: dict[str, str] | None = None,
    suggestions: dict[str, Any] | None = None,
) -> str:
    """Render INCAR tags as aligned text with trailing (comments).

    Each tag is written exactly once, in dict order. Trailing comments
    come from `comments` (default: INCAR_TAG_COMMENTS); tags without an
    entry get no comment. Commented-out suggestion lines follow after
    a blank line. SYSTEM is written as a plain line because VASP treats
    everything after '=' as the system name.

    Args:
        tags: Tag → value mapping (insertion order is preserved).
        comments: Optional tag → comment table overriding the default.
        suggestions: Optional commented-out tag → value lines appended
            at the end; skipped when already present as active tags.

    Returns:
        INCAR file content as a string (LF endings, trailing newline).
    """
    table = INCAR_TAG_COMMENTS if comments is None else comments
    lines: list[str] = []
    for tag, value in tags.items():
        if tag == "SYSTEM":
            lines.append(f"SYSTEM = {_format_incar_value(value)}")
        else:
            lines.append(_format_incar_line(tag, value, table.get(tag), False))
    if suggestions:
        lines.append("")
        for tag, value in suggestions.items():
            if tag in tags:
                continue
            lines.append(_format_incar_line(tag, value, table.get(tag), True))
    return "\n".join(lines) + "\n"


_TAG_LINE_RE = re.compile(r"^\s*(\S+)\s*=\s*(.*)$")


def parse_incar_content(text: str) -> tuple[dict[str, str], list[tuple[int, str, str]]]:
    """Parse INCAR text back into ordered tags.

    Round-trips the output of format_incar_content and also accepts
    hand-written INCAR text.

    Args:
        text: INCAR content (generated or hand-edited).

    Returns:
        (tags, problems): tags preserves line order and the original tag
        casing; trailing (comments) are stripped from values. problems
        is a list of (line_no, kind, message) with kind "duplicate" or
        "malformed".

    Rules:
        - Blank lines and lines whose first non-space char is '#' or
          '!' are ignored (comment / commented-out suggestion lines).
        - SYSTEM (case-insensitive) keeps the whole text after '=' —
          VASP reads the full line as the system name.
        - A line not matching ``TAG = ...`` is reported as malformed.
        - A tag name that repeats (case-insensitive) is reported as a
          duplicate; the first occurrence wins.
    """
    tags: dict[str, str] = {}
    problems: list[tuple[int, str, str]] = []
    seen: dict[str, str] = {}
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith(("#", "!")):
            continue
        match = _TAG_LINE_RE.match(line)
        if not match:
            problems.append((line_no, "malformed", line.strip()))
            continue
        tag = match.group(1)
        value = match.group(2).strip()
        key = tag.casefold()
        if key in seen:
            problems.append((line_no, "duplicate", tag))
            continue
        seen[key] = tag
        if tag.upper() != "SYSTEM":
            value = re.sub(r"\s*\(.*\)\s*$", "", value).rstrip()
        tags[tag] = value
    return tags, problems

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


def suggest_band_path(
    atoms,
) -> tuple[list[tuple[str, str]], dict[str, np.ndarray]] | None:
    """Suggest a high-symmetry k-path for a periodic structure.

    Uses pymatgen's symmetry analysis to pick the lattice-appropriate
    path (Setyawan-Curtarolo convention) and transforms the symmetry
    points from the standardized primitive reciprocal basis into the
    reciprocal basis of the input cell, so the result feeds
    ``generate_kpoints_line_mode`` directly.

    Args:
        atoms: ASE Atoms with a full-rank periodic cell.

    Returns:
        (path_segments, special_points) where path_segments is a list
        of (start_label, end_label) pairs and special_points maps every
        label to fractional coordinates in the input cell's reciprocal
        basis. None when the structure is not periodic or no path can
        be determined (callers fall back to a default path).
    """
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.bandstructure import HighSymmKpath
    except Exception:
        return None

    try:
        if atoms.get_cell().rank < 3 or len(atoms) == 0:
            return None
        structure = Structure.from_ase_atoms(atoms)
        with warnings.catch_warnings():
            # pymatgen warns when the input cell is not the standard
            # primitive — we transform the k-points ourselves below.
            warnings.simplefilter("ignore")
            kpath = HighSymmKpath(structure)
        raw_path = kpath.kpath["path"]
        raw_points = kpath.kpath["kpoints"]
        # kpoints are fractional in the standardized primitive cell's
        # reciprocal basis; convert to the input cell's reciprocal basis:
        # f_in = f_prim @ inv(A_prim.T) @ A_in.T
        #   (cartesian k = f @ B with B = inv(A).T is basis-independent)
        prim_cell = kpath.prim.lattice.matrix
        in_cell = np.asarray(atoms.get_cell(), dtype=float)
        transform = np.linalg.inv(prim_cell.T) @ in_cell.T
    except Exception:
        return None

    try:
        special_points = {
            label: np.asarray(coord, dtype=float) @ transform
            for label, coord in raw_points.items()
        }
        # raw_path is a list of polylines (label lists); flatten each
        # polyline into consecutive (start, end) label pairs.
        path: list[tuple[str, str]] = []
        for polyline in raw_path:
            for i in range(len(polyline) - 1):
                start, end = polyline[i], polyline[i + 1]
                if start not in special_points or end not in special_points:
                    raise ValueError(start)
                path.append((start, end))
    except Exception:
        return None
    if not path:
        return None
    return path, special_points


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

def generate_poscar(structure_model, poscar_direct: bool = False) -> str:
    """Render the current structure as POSCAR text.

    Includes the fixed-atom constraints (Selective dynamics block) when
    any atom is frozen. Shared by generate_all_inputs and the unified
    input-file dialog.

    Args:
        structure_model: StructureModel with the current structure.
        poscar_direct: Write fractional (Direct) coordinates instead of
            Cartesian.

    Returns:
        POSCAR file content as a string.
    """
    import io

    from ase.io import write as ase_write

    from vaspen.core.file_io import (
        atoms_with_fixed_constraints,
        vasp_write_atoms,
    )

    poscar_atoms = atoms_with_fixed_constraints(
        vasp_write_atoms(structure_model.atoms, poscar_direct),
        structure_model.fixed_flags,
    )
    buf = io.StringIO()
    ase_write(buf, poscar_atoms, format="vasp", vasp5=True, direct=poscar_direct)
    return buf.getvalue()


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
    if structure_model.any_magmom:
        # GUI-set initial moments: one value per atom in atom order
        # (unset atoms are written as 0.0), with collinear spin enabled.
        # Overrides applied below still win over these injections.
        preset["MAGMOM"] = magmom_line(structure_model.magmoms)
        preset["ISPIN"] = 2
    if incar_overrides:
        preset.update(incar_overrides)
    empty = [tag for tag, val in preset.items() if str(val).strip() == ""]
    if empty:
        # e.g. NEB IMAGES — system-dependent and intentionally left blank
        # so the user has to fill it in (settled 2026-08-14).
        raise ValueError(
            _tr("INCAR tags without a value: {} — fill them in the INCAR editor.")
            .format(", ".join(empty))
        )
    incar_content = format_incar_content(
        preset, suggestions=INCAR_SUGGESTIONS.get(incar_preset)
    )

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
    poscar_content = generate_poscar(structure_model, poscar_direct)

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
