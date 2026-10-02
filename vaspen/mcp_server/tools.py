"""Transport-free MCP tool core for VASPen.

Every tool is a thin adapter over the existing core-layer API
(``vaspen.core.*``) — this module adds session state and output capping
only, and deliberately imports NO MCP SDK: it is executed both by the
stdio server (``server.py``, which owns the SDK and error mapping) and
inside the running GUI (``vaspen/ui/bridge_server.py``, the live
bridge), where the SDK is not bundled. Tool descriptions/responses are
English-only by design (AI-facing, not UI): this package never calls
``_tr()``/``tr()`` and is absent from the i18n parity tests.

Error convention: tools raise the core layer's native exceptions
(ValueError/FileNotFoundError/...). Each transport maps them for its
own protocol (ToolError for the SDK, an error field for the bridge).

Return convention: a dict result is returned as-is; a ``bytes`` result
is a PNG image (render_preview) — each transport wraps it (the SDK as
``Image`` content, the bridge as a base64 payload).
"""

from __future__ import annotations

import ast
import contextlib
import io
from pathlib import Path
from typing import Any

from ase import Atoms

# Aliases: several tool functions share a name with the core module or
# function they call (measure, estimate_k_mesh, rebox_slab,
# suggest_band_path); StructureBuilder is the class holding the model
# mutation statics (builder.make_supercell at module level is ASE's
# function, not the class method).
from vaspen.core import measure as _measure
from vaspen.core import neb, symmetry, transform
from vaspen.core.builder import StructureBuilder as _StructureBuilder
from vaspen.core.file_io import (
    PERIODIC_FORMATS,
    FileIO,
    resolve_format,
)
from vaspen.core.structure import StructureModel
from vaspen.core.surface import SurfaceCutter
from vaspen.core.surface import rebox_slab as _rebox_slab
from vaspen.core.vasp_input import (
    INCAR_PRESETS,
    INCAR_SUGGESTIONS,
    estimate_k_mesh as _estimate_k_mesh,
    format_incar_content,
    generate_all_inputs,
    generate_kpoints_automatic,
    generate_potcar,
    suggest_band_path as _suggest_band_path,
)
from vaspen.mcp_server.session import ServerSession
from vaspen.utils.config import AppConfig

SESSION = ServerSession()

# All registered tools, wrapped; build_server() hands them to the SDK.
_TOOLS: list = []

# Output caps (paginate or truncate instead of flooding the AI context).
MAX_LIST_ATOMS = 1000
MAX_BONDS = 1000
MAX_LISTED_SLABS = 50

# Persistent namespace for run_python (survives across calls within the
# session, like a console); "model" is refreshed on every call.
_RUN_NAMESPACE: dict | None = None


def tool(fn):
    """Register a raw tool function (see module docstring for the
    error/return conventions; transports wrap from there)."""
    _TOOLS.append(fn)
    return fn


# ----------------------------------------------------------------------
# Helpers (called under SESSION.lock)
# ----------------------------------------------------------------------


def _structure_summary(model: StructureModel) -> dict:
    """Compact machine-readable summary used by most tool responses."""
    info: dict[str, Any] = {
        "source": SESSION.filepath,
        "formula": model.chemical_formula,
        "n_atoms": model.n_atoms,
        "periodic": model.is_periodic,
        "pbc": [bool(p) for p in model.pbc],
        "has_disorder": model.has_disorder,
        "n_frozen_atoms": int(model.fixed_flags.any(axis=1).sum()),
        "has_magmoms": model.any_magmom,
        "dirty": model.is_dirty,
    }
    if model.is_periodic:
        lengths = model.cell_lengths
        angles = model.cell_angles
        info["cell"] = {
            "a": round(float(lengths[0]), 6),
            "b": round(float(lengths[1]), 6),
            "c": round(float(lengths[2]), 6),
            "alpha": round(float(angles[0]), 3),
            "beta": round(float(angles[1]), 3),
            "gamma": round(float(angles[2]), 3),
        }
        info["volume"] = round(float(model.atoms.get_volume()), 6)
    sym = symmetry.analyze(model.atoms)
    if sym is not None:
        if sym.kind == "space":
            info["space_group"] = f"{sym.international} (No. {sym.number})"
        else:
            info["point_group"] = sym.international
    return info


def _validate_indices(model: StructureModel, indices: list[int]) -> None:
    n = model.n_atoms
    for i in indices:
        if not 0 <= i < n:
            raise ValueError(
                f"Atom index {i} out of range (0..{n - 1}); "
                "indices are 0-based."
            )


def _check_disorder(model: StructureModel, allowed: bool,
                    operation: str) -> None:
    """Partial occupancy is display-only and lost on derived structures
    — the headless counterpart of the GUI's disorder confirm dialogs."""
    if model.has_disorder and not allowed:
        raise ValueError(
            f"{operation} would discard the partial occupancies of this "
            "disordered structure. Confirm by passing "
            "allow_disorder_loss=true, or use a fully ordered structure."
        )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _incar_presets_payload() -> dict:
    presets = {}
    for name, preset in INCAR_PRESETS.items():
        presets[name] = {
            "description": preset.get("_description", ""),
            "tags": {tag: str(value) for tag, value in preset.items()
                     if not tag.startswith("_")},
        }
    return presets


# ----------------------------------------------------------------------
# Structure session tools
# ----------------------------------------------------------------------


@tool
def open_structure(path: str) -> dict:
    """Load a structure file (CIF, XYZ, POSCAR/CONTCAR, XSF, PDB, ...)
    and make it the current session structure. Replaces any previously
    loaded structure.

    Args:
        path: Path to the structure file.

    Returns:
        Summary: formula, atom count, cell parameters, space group /
        point group, disorder / frozen-atom / magmom state.
    """
    with SESSION.lock:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")
        model = SESSION.load(path)
        return {"ok": True, **_structure_summary(model)}


@tool
def get_structure_info() -> dict:
    """Summary of the current structure: formula, atom count, cell
    parameters, space group / point group, disorder / frozen-atom /
    magmom state, and whether there are unsaved changes."""
    with SESSION.lock:
        model = SESSION.require_model()
        return {"ok": True, **_structure_summary(model)}


@tool
def list_atoms(start: int = 0, count: int = 100,
               fractional: bool = False) -> dict:
    """List atoms of the current structure with their positions.

    Indices are 0-based (ASE convention). Use start/count to page
    through large structures.

    Args:
        start: Index of the first atom to return.
        count: Maximum number of atoms to return (capped at 1000).
        fractional: Return fractional (cell) coordinates instead of
            Cartesian ones (periodic structures only).
    """
    with SESSION.lock:
        model = SESSION.require_model()
        n = model.n_atoms
        if not 0 <= start < max(n, 1):
            raise ValueError(f"start must be in 0..{max(n - 1, 0)}.")
        if fractional and not model.is_periodic:
            raise ValueError(
                "Fractional coordinates require a periodic structure.")
        count = max(1, min(int(count), MAX_LIST_ATOMS))
        stop = min(start + count, n)
        coords = (model.scaled_positions if fractional else model.positions)
        show_fixed = model.any_fixed
        show_magmom = model.any_magmom
        atoms_out = []
        for i in range(start, stop):
            entry: dict[str, Any] = {
                "index": i,
                "symbol": model.symbols[i],
                "position": [round(float(v), 8) for v in coords[i]],
            }
            if show_fixed:
                entry["fixed"] = [bool(v) for v in model.fixed_flags[i]]
            if show_magmom:
                entry["magmom"] = model.magmom(i)
            atoms_out.append(entry)
        return {
            "ok": True,
            "total": n,
            "start": start,
            "returned": len(atoms_out),
            "fractional": fractional,
            "atoms": atoms_out,
        }


@tool
def save_structure(path: str, fmt: str | None = None,
                   wrap_padding: float | None = None) -> dict:
    """Save the current structure to a file.

    Saving a non-periodic structure (a molecule) to a periodic format
    (vasp / cif) requires wrapping it into a periodic cell first: pass
    ``wrap_padding`` (vacuum in Angstrom added on every side of the
    atom bounding box, e.g. 5.0) to do that in the same call, or call
    make_periodic first. The wrap converts the current structure.

    Args:
        path: Destination path (required — the tool never implicitly
            overwrites the file the structure was opened from).
        fmt: ASE format name ('vasp', 'cif', 'xyz', ...). Auto-detected
            from the extension when omitted.
        wrap_padding: Vacuum padding in Angstrom for wrapping a
            molecule into a periodic cell (see above).
    """
    with SESSION.lock:
        model = SESSION.require_model()
        fmt_name = fmt or resolve_format(path)
        if fmt_name in PERIODIC_FORMATS and not model.is_periodic:
            if wrap_padding is None:
                raise ValueError(
                    "Saving a non-periodic structure to a periodic format "
                    "(vasp/cif) needs a periodic cell: pass wrap_padding "
                    "(vacuum in Angstrom, e.g. 5.0) or call make_periodic "
                    "first.")
            model.make_periodic(float(wrap_padding))
        model.save(path, fmt=fmt)
        return {"ok": True, "saved": str(Path(path)),
                **_structure_summary(model)}


@tool
def make_periodic(padding: float) -> dict:
    """Wrap the current molecule-like structure into a periodic
    (orthorhombic) vacuum cell: the cell is rebuilt from the atom
    bounding box with ``padding`` Angstrom of vacuum added on every
    side, and the box is centered on the atoms. Required before saving
    a molecule to POSCAR/CIF or generating a k-point mesh.

    Args:
        padding: Vacuum on each side of the bounding box, in Angstrom
            (the GUI default is 5.0).
    """
    with SESSION.lock:
        model = SESSION.require_model()
        if model.atoms.get_cell().rank == 3:
            raise ValueError(
                "The structure already has a full-rank cell — "
                "make_periodic is only for molecule-like structures."
            )
        model.make_periodic(float(padding))
        return {"ok": True, **_structure_summary(model)}


# ----------------------------------------------------------------------
# Transform / build tools
# ----------------------------------------------------------------------


@tool
def make_supercell(na: int, nb: int, nc: int) -> dict:
    """Tile the current periodic structure na×nb×nc along the cell
    vectors. Frozen-atom flags and magnetic moments are tiled along.
    Replaces the current structure with the supercell.

    Args:
        na, nb, nc: Repeat factors along a, b, c (each >= 1).
    """
    with SESSION.lock:
        model = SESSION.require_model()
        if not model.is_periodic:
            raise ValueError("Supercells require a periodic structure.")
        if min(int(na), int(nb), int(nc)) < 1:
            raise ValueError("Repeat factors must be >= 1.")
        _StructureBuilder.make_supercell(model, (int(na), int(nb), int(nc)))
        return {"ok": True, **_structure_summary(model)}


@tool
def set_fixed_atoms(indices: list[int],
                    fixed: bool | list[bool] = True) -> dict:
    """Set VASP selective-dynamics (frozen) flags on atoms.

    Args:
        indices: 0-based atom indices.
        fixed: True/False applies to all three directions, or a
            3-element list [fix_a, fix_b, fix_c] applied to every
            listed atom.
    """
    with SESSION.lock:
        model = SESSION.require_model()
        _validate_indices(model, list(indices))
        mask: bool | list[bool]
        if isinstance(fixed, list):
            if len(fixed) != 3:
                raise ValueError("fixed must be a bool or a 3-element list.")
            mask = [bool(v) for v in fixed]
        else:
            mask = bool(fixed)
        model.set_fixed(list(indices), mask)
        return {"ok": True, **_structure_summary(model)}


@tool
def set_magmoms(indices: list[int], value: float) -> dict:
    """Set initial magnetic moments (VASP MAGMOM, collinear) on atoms.

    Args:
        indices: 0-based atom indices.
        value: Moment in Bohr magnetons applied to every listed atom.
    """
    with SESSION.lock:
        model = SESSION.require_model()
        _validate_indices(model, list(indices))
        model.set_magmom(list(indices), float(value))
        return {"ok": True, **_structure_summary(model)}


@tool
def translate_atoms(vector: list[float],
                    indices: list[int] | None = None) -> dict:
    """Translate atoms by a Cartesian vector (Angstrom).

    Args:
        vector: (dx, dy, dz) in Angstrom.
        indices: 0-based atom indices to move; omit to move the whole
            structure. Periodic axes are wrapped after the move.
    """
    with SESSION.lock:
        model = SESSION.require_model()
        if indices:
            _validate_indices(model, list(indices))
        new_atoms = transform.translate(model.atoms,
                                        [float(v) for v in vector],
                                        indices)
        SESSION.replace_atoms(new_atoms, fixed_flags=model.fixed_flags,
                              magmoms=model.magmoms)
        return {"ok": True, **_structure_summary(model)}


@tool
def rotate_atoms(axis_start: list[float], axis_end: list[float],
                 angle_deg: float, indices: list[int] | None = None,
                 center: list[float] | None = None) -> dict:
    """Rotate atoms around an axis (right-hand rule). Rotates atom
    positions only — the cell is unchanged; periodic axes are wrapped
    afterwards.

    Args:
        axis_start, axis_end: Two Cartesian points (Angstrom) defining
            the rotation axis; read atom positions from list_atoms to
            build an axis through specific atoms.
        angle_deg: Rotation angle in degrees.
        indices: 0-based atom indices to rotate; omit to rotate the
            whole structure.
        center: Point the rotation acts about; defaults to the centroid
            of the moved set.
    """
    with SESSION.lock:
        model = SESSION.require_model()
        if indices:
            _validate_indices(model, list(indices))
        new_atoms = transform.rotate(
            model.atoms,
            [float(v) for v in axis_start],
            [float(v) for v in axis_end],
            float(angle_deg),
            indices,
            None if center is None else [float(v) for v in center],
        )
        SESSION.replace_atoms(new_atoms, fixed_flags=model.fixed_flags,
                              magmoms=model.magmoms)
        return {"ok": True, **_structure_summary(model)}


@tool
def measure(kind: str, i: int, j: int, k: int | None = None,
            l: int | None = None) -> dict:
    """Measure a geometric quantity between atoms (0-based indices,
    direct Cartesian geometry — no minimum-image folding, so use
    neighboring images explicitly when atoms are in different cells).

    Args:
        kind: "distance" (i-j), "angle" (i-j-k, vertex at j) or
            "dihedral" (i-j-k-l).
        i, j, k, l: Atom indices; k/l required per kind.
    """
    with SESSION.lock:
        model = SESSION.require_model()
        if kind == "distance":
            needed = [i, j]
        elif kind == "angle":
            if k is None:
                raise ValueError("angle requires indices i, j, k.")
            needed = [i, j, k]
        elif kind == "dihedral":
            if k is None or l is None:
                raise ValueError("dihedral requires indices i, j, k, l.")
            needed = [i, j, k, l]
        else:
            raise ValueError(
                'kind must be "distance", "angle" or "dihedral".')
        _validate_indices(model, needed)
        if kind == "distance":
            value, unit = _measure.distance(model.atoms, i, j), "Angstrom"
        elif kind == "angle":
            value, unit = _measure.angle(model.atoms, i, j, k), "degrees"
        else:
            value, unit = (_measure.dihedral(model.atoms, i, j, k, l),
                           "degrees")
        return {"ok": True, "kind": kind, "indices": needed,
                "value": round(float(value), 6), "unit": unit}


@tool
def find_bonds() -> dict:
    """Covalent/ionic bonds of the current structure (periodic-aware;
    auto-detection is skipped above 1000 atoms, as in the GUI)."""
    with SESSION.lock:
        model = SESSION.require_model()
        bonds = model.bonds
        listed = [{"i": b.i, "j": b.j, "order": b.order}
                  for b in bonds[:MAX_BONDS]]
        return {
            "ok": True,
            "bond_mode": model.bond_mode,
            "total": len(bonds),
            "returned": len(listed),
            "bonds": listed,
        }


# ----------------------------------------------------------------------
# Symmetry / surface tools
# ----------------------------------------------------------------------


@tool
def analyze_symmetry(symprec: float = 1e-5) -> dict:
    """Symmetry of the current structure: spglib space group for
    periodic structures, pymatgen Schoenflies point group for
    molecules.

    Args:
        symprec: Position tolerance in Angstrom (spglib).
    """
    with SESSION.lock:
        model = SESSION.require_model()
        sym = symmetry.analyze(model.atoms, float(symprec))
        if sym is None:
            raise ValueError(
                "Symmetry analysis failed — try a larger symprec.")
        return {
            "ok": True,
            "kind": sym.kind,
            "number": sym.number,
            "symbol": sym.international,
            "hall": sym.hall,
            "pointgroup": sym.pointgroup,
            "choice": sym.choice,
        }


@tool
def symmetrize_cell(cell_type: str = "conventional",
                    symprec: float = 1e-5,
                    allow_disorder_loss: bool = False) -> dict:
    """Convert the current periodic structure to its spglib
    standardized conventional or primitive cell (pure rotation, atom
    distances preserved). Replaces the current structure.

    Args:
        cell_type: "conventional" (default) or "primitive".
        symprec: Position tolerance in Angstrom (spglib).
        allow_disorder_loss: Confirm discarding partial occupancies.
    """
    with SESSION.lock:
        model = SESSION.require_model()
        if cell_type not in ("conventional", "primitive"):
            raise ValueError(
                'cell_type must be "conventional" or "primitive".')
        _check_disorder(model, allow_disorder_loss, "Symmetrize")
        result = symmetry.symmetrize(model.atoms, float(symprec), cell_type)
        if result is None:
            raise ValueError(
                "Symmetrization failed — the structure may be too far "
                "from its ideal symmetry; try a larger symprec.")
        SESSION.replace_atoms(result)
        return {"ok": True, "cell_type": cell_type,
                **_structure_summary(model)}


@tool
def list_slab_terminations(h: int, k: int, l: int, layers: int = 4,
                           vacuum: float = 15.0) -> dict:
    """Enumerate the unique surface terminations of the current bulk
    structure for a Miller index (every termination is built with
    pymatgen, so this can take seconds each on complex cells).

    Args:
        h, k, l: Miller indices of the surface (not all zero).
        layers: Minimum number of atomic layers in the slab.
        vacuum: Vacuum spacing in Angstrom (centered).

    Returns:
        One entry per termination: termination index (for cut_surface),
        top/bottom surface composition, broken bonds, atom count.
    """
    with SESSION.lock:
        model = SESSION.require_model()
        infos = SurfaceCutter(model).slabs((int(h), int(k), int(l)),
                                           int(layers), float(vacuum))
        out = [{
            "termination": idx,
            "top": info.top_composition,
            "bottom": info.bottom_composition,
            "broken_bonds": info.broken_bonds,
            "n_atoms": info.n_atoms,
        } for idx, info in enumerate(infos[:MAX_LISTED_SLABS])]
        return {"ok": True, "total": len(infos), "terminations": out}


@tool
def cut_surface(h: int, k: int, l: int, layers: int = 4,
                vacuum: float = 15.0, termination: int = 0,
                allow_disorder_loss: bool = False) -> dict:
    """Cut a surface slab from the current bulk structure (pymatgen;
    the c axis is re-expressed exactly along the surface normal with
    the requested vacuum). Replaces the current structure with the
    slab — reopen the file to get the bulk back. Use
    list_slab_terminations to pick a termination.

    Args:
        h, k, l: Miller indices of the surface.
        layers: Minimum number of atomic layers.
        vacuum: Vacuum spacing in Angstrom (centered).
        termination: Termination index from list_slab_terminations.
        allow_disorder_loss: Confirm discarding partial occupancies.
    """
    with SESSION.lock:
        model = SESSION.require_model()
        _check_disorder(model, allow_disorder_loss, "Surface cutting")
        slab = SurfaceCutter(model).cut_termination(
            (int(h), int(k), int(l)), int(layers), float(vacuum),
            int(termination))
        SESSION.replace_atoms(slab.atoms)
        return {"ok": True, "miller": [int(h), int(k), int(l)],
                "termination": int(termination),
                **_structure_summary(model)}


@tool
def rebox_slab(vacuum: float, allow_disorder_loss: bool = False) -> dict:
    """Re-box the current slab: unwrap layers split across the periodic
    boundary, re-apply the requested vacuum along c and center the slab.
    The in-plane cell, atom set and atom order are unchanged (frozen
    flags and magmoms carry over 1:1). For structures that already
    carry vacuum — use cut_surface to cleave a bulk.

    Args:
        vacuum: Vacuum spacing along c in Angstrom (slabs may use
            larger values than molecules — up to 100 in the GUI).
        allow_disorder_loss: Confirm discarding partial occupancies.
    """
    with SESSION.lock:
        model = SESSION.require_model()
        _check_disorder(model, allow_disorder_loss, "Re-boxing")
        result = _rebox_slab(model.atoms, float(vacuum))
        SESSION.replace_atoms(result, fixed_flags=model.fixed_flags,
                              magmoms=model.magmoms)
        return {"ok": True, **_structure_summary(model)}


# ----------------------------------------------------------------------
# VASP input tools
# ----------------------------------------------------------------------


@tool
def suggest_band_path() -> dict:
    """Suggested high-symmetry k-path for band-structure calculations
    of the current periodic structure (pymatgen Setyawan-Curtarolo
    convention, transformed into the input cell's basis)."""
    with SESSION.lock:
        model = SESSION.require_model()
        if not model.is_periodic:
            raise ValueError("Band paths require a periodic structure.")
        suggested = _suggest_band_path(model.atoms)
        if suggested is None:
            raise ValueError(
                "No band path could be suggested for this structure.")
        segments, special_points = suggested
        return {
            "ok": True,
            "segments": [[str(a), str(b)] for a, b in segments],
            "special_points": {label: [float(v) for v in pt]
                               for label, pt in special_points.items()},
        }


@tool
def estimate_k_mesh(spacing: float = 0.04) -> dict:
    """k-point mesh for the current periodic structure at a target
    KSPACING (units of 2π/Å — the vaspkit convention, equivalent to
    VASP's KSPACING tag times 2π). Recommended: 0.04 insulators,
    0.03 metals.

    Args:
        spacing: Target k-point spacing in 2π/Å.
    """
    with SESSION.lock:
        model = SESSION.require_model()
        if not model.is_periodic:
            raise ValueError("k-point meshes require a periodic structure.")
        mesh = _estimate_k_mesh(model.cell, float(spacing))
        return {"ok": True, "spacing_2pi_over_A": float(spacing),
                "mesh": [int(v) for v in mesh]}


@tool
def list_incar_presets() -> dict:
    """INCAR presets for the calculation task selector (scf, opt, band,
    dos, optical, neb) with their community-standard default tags."""
    return {"ok": True, "presets": _incar_presets_payload()}


@tool
def generate_inputs(out_dir: str, preset: str = "scf",
                    incar_overrides: dict[str, Any] | None = None,
                    kpoints_mode: str = "automatic",
                    kpoints_params: dict[str, Any] | None = None,
                    potcar_library: str | None = None,
                    functional: str = "PBE",
                    poscar_direct: bool = False,
                    allow_disorder_loss: bool = False) -> dict:
    """Generate the four VASP input files (INCAR, KPOINTS, POSCAR,
    POTCAR) for the current structure into a directory.

    POTCAR needs a local pseudopotential library: pass potcar_library
    (standard VASP potcar layout, e.g. .../PBE.54/Fe/POTCAR) or
    configure it once in the VASPen GUI preferences — the setting is
    shared. Without a library the other three files are still written.

    kpoints_params keys per mode: automatic → k_spacing (2π/Å),
    gamma_centered; manual → k1, k2, k3, gamma_centered;
    line → path (list of [start, end] label pairs — see
    suggest_band_path), n_points. Note: a "band" preset expects
    line-mode KPOINTS (ICHARG=11 workflow); for NEB prefer neb_setup.

    Args:
        out_dir: Destination directory (created if missing).
        preset: INCAR preset name (see list_incar_presets).
        incar_overrides: INCAR tag → value overrides applied last.
        kpoints_mode: "automatic" | "manual" | "line".
        kpoints_params: Parameters for the chosen mode (see above).
        potcar_library: Pseudopotential library root; defaults to the
            GUI-configured path.
        functional: XC functional for POTCAR ("PBE", "PBE_new", "LDA",
            "PW91").
        poscar_direct: Write POSCAR in fractional (Direct) coordinates.
        allow_disorder_loss: Confirm that POSCAR cannot represent the
            partial occupancies of a disordered structure.
    """
    with SESSION.lock:
        model = SESSION.require_model()
        _check_disorder(model, allow_disorder_loss, "POSCAR generation")
        library = potcar_library or AppConfig().potcar_library_path
        contents = generate_all_inputs(
            model,
            incar_preset=preset,
            incar_overrides=incar_overrides,
            kpoints_mode=kpoints_mode,
            kpoints_params=kpoints_params,
            potcar_library=library,
            functional=functional,
            poscar_direct=poscar_direct,
        )
        out = Path(out_dir)
        files: dict[str, str] = {}
        for name, content in contents.items():
            if not content:
                continue
            target = out / name
            _write_text(target, content)
            files[name] = str(target)
        response: dict[str, Any] = {
            "ok": True,
            "out_dir": str(out),
            "files": files,
            "INCAR": contents["INCAR"],
            "KPOINTS": contents["KPOINTS"],
        }
        if "POTCAR" not in files and not library:
            response["potcar_note"] = (
                "POTCAR not written — no pseudopotential library "
                "configured (pass potcar_library or set it in the VASPen "
                "GUI preferences).")
        return response


# ----------------------------------------------------------------------
# NEB tools
# ----------------------------------------------------------------------


def _load_neb_pair(initial_path: str, final_path: str) -> tuple[Atoms, Atoms]:
    for p in (initial_path, final_path):
        if not Path(p).exists():
            raise FileNotFoundError(f"File not found: {p}")
    init = FileIO.read(initial_path)
    final = FileIO.read(final_path)
    return init, final


@tool
def neb_check(initial_path: str, final_path: str) -> dict:
    """Validate a NEB endpoint pair without writing anything: atom
    pairing, path distance, suggested image count, atom-order
    diagnostic and frozen-atom consistency.

    Args:
        initial_path, final_path: Structure files of the initial and
            final states (equal cells and per-element counts required).
    """
    with SESSION.lock:
        init, final = _load_neb_pair(initial_path, final_path)
        distance = neb.neb_distance(init, final)
        suggested = neb.suggest_n_images(distance)
        response: dict[str, Any] = {
            "ok": True,
            "distance": round(float(distance), 6),
            "suggested_n_images": int(suggested),
            "n_atoms": len(init),
        }
        mismatch = neb.detect_order_mismatch(init, final)
        if mismatch is not None:
            response["order_warning"] = (
                "The atom order of the two files would shorten the path "
                f"by {mismatch * 100:.0f}% — the elements are likely "
                "listed in a different order. Check before interpolating.")
        frozen_init = neb.constraints_to_fixed_flags(init)
        frozen_final = neb.constraints_to_fixed_flags(final)
        response["n_frozen_init"] = int(frozen_init.all(axis=1).sum())
        response["n_frozen_final"] = int(frozen_final.all(axis=1).sum())
        try:
            neb.interpolate_neb(init, final, 1,
                                frozen_mask=frozen_init.all(axis=1))
        except ValueError as exc:
            response["frozen_error"] = str(exc)
        return response


@tool
def neb_setup(initial_path: str, final_path: str, out_dir: str,
              n_images: int | None = None, method: str = "linear",
              force: bool = False, potcar_library: str | None = None,
              functional: str = "PBE") -> dict:
    """Prepare a complete standard-layout NEB run directory:
    00/POSCAR ... 0N/POSCAR (endpoints included, Selective dynamics
    carried) plus INCAR (neb preset, IMAGES filled), KPOINTS (automatic
    mesh) and POTCAR in the root.

    Frozen atoms of the initial structure keep their initial position
    in every frame; a frozen atom at different positions in the two
    files blocks interpolation. A detected atom-order mismatch refuses
    to write unless force=true.

    Args:
        initial_path, final_path: Endpoint structure files.
        out_dir: Destination directory (created if missing).
        n_images: Intermediate images (1..98); default = ceil(distance
            / 0.8).
        method: "linear" or "idpp" (image-dependent pair potential —
            smoother paths when a linear path would collide atoms).
        force: Proceed despite the atom-order warning.
        potcar_library: Pseudopotential library root; defaults to the
            GUI-configured path.
        functional: XC functional for POTCAR.
    """
    with SESSION.lock:
        if method not in ("linear", "idpp"):
            raise ValueError('method must be "linear" or "idpp".')
        init, final = _load_neb_pair(initial_path, final_path)
        distance = neb.neb_distance(init, final)
        images = (int(n_images) if n_images is not None
                  else neb.suggest_n_images(distance))
        mismatch = None if force else neb.detect_order_mismatch(init, final)
        if mismatch is not None:
            raise ValueError(
                f"Atom-order mismatch: reordering would shorten the path "
                f"by {mismatch * 100:.0f}%. Verify the element order of "
                "the two files, then pass force=true to continue in file "
                "order.")
        # Fully-frozen atoms (all three directions) keep their initial
        # position in every frame; partially frozen ones interpolate
        # normally (VASP enforces their FixScaled at run time).
        frozen = neb.constraints_to_fixed_flags(init).all(axis=1)
        if method == "idpp":
            frames = neb.interpolate_idpp(init, final, images,
                                          frozen_mask=frozen)
        else:
            frames = neb.interpolate_neb(init, final, images,
                                         frozen_mask=frozen)


        out = Path(out_dir)
        written: list[str] = []
        for idx, frame in enumerate(frames):
            target = out / f"{idx:02d}" / "POSCAR"
            target.parent.mkdir(parents=True, exist_ok=True)
            FileIO.write(str(target), frame,
                         fixed_flags=neb.constraints_to_fixed_flags(frame))
            written.append(str(target))

        preset = dict(INCAR_PRESETS["neb"])
        preset.pop("_description", None)
        preset["IMAGES"] = str(images)
        _write_text(out / "INCAR",
                    format_incar_content(
                        preset, suggestions=INCAR_SUGGESTIONS.get("neb")))
        written.append(str(out / "INCAR"))

        _write_text(out / "KPOINTS",
                    generate_kpoints_automatic(init.get_cell().array))
        written.append(str(out / "KPOINTS"))

        library = potcar_library or AppConfig().potcar_library_path
        potcar_note = None
        if library:
            elements = list(dict.fromkeys(init.get_chemical_symbols()))
            potcar_content, _paths = generate_potcar(elements, library,
                                                     functional)
            _write_text(out / "POTCAR", potcar_content)
            written.append(str(out / "POTCAR"))
        else:
            potcar_note = (
                "POTCAR not written — no pseudopotential library "
                "configured (pass potcar_library or set it in the VASPen "
                "GUI preferences).")

        response: dict[str, Any] = {
            "ok": True,
            "out_dir": str(out),
            "method": method,
            "distance": round(float(distance), 6),
            "n_images": images,
            "n_frames": len(frames),
            "files_written": written,
            "incar": str(out / "INCAR"),
        }
        if potcar_note:
            response["potcar_note"] = potcar_note
        return response


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------


@tool
def render_preview(view: str = "auto", max_px: int = 800):
    """Render the current structure as a ball-and-stick image (2D
    orthographic projection) so you can visually verify geometry after
    edits, surface cuts or supercells.

    Args:
        view: Projection direction — "x", "y" or "z" (world axes;
            molecules and crystals), "a", "b" or "c" (look along a cell
            vector, periodic structures only) or "auto" (c for
            periodic, z for molecules).
        max_px: Maximum canvas edge in pixels (square, 100 dpi).
    """
    with SESSION.lock:
        model = SESSION.require_model()
        if view not in ("auto", "x", "y", "z", "a", "b", "c"):
            raise ValueError(
                'view must be one of "auto", "x", "y", "z", "a", "b", "c".')
        # Lazy import: keeps matplotlib out of the tool-listing path.
        from vaspen.mcp_server.render import render_png

        # bytes = PNG (see module docstring); the transport wraps it
        return render_png(model.atoms, view=view, max_px=int(max_px))


# ----------------------------------------------------------------------
# Escape hatch
# ----------------------------------------------------------------------


def _run_namespace() -> dict:
    """The run_python namespace: heavy names built once, per-call state
    (model/session) refreshed every call."""
    global _RUN_NAMESPACE
    if _RUN_NAMESPACE is None:
        import ase as _ase
        import numpy as np

        from vaspen.core import neb as _neb
        from vaspen.core import symmetry as _symmetry
        from vaspen.core import transform as _transform
        from vaspen.core import vasp_input as _vasp_input
        from vaspen.core import measure as _measure_mod
        from vaspen.core.builder import StructureBuilder as _StructureBuilder
        from vaspen.core.file_io import FileIO as _FileIO
        from vaspen.core.surface import SurfaceCutter as _SurfaceCutter
        from vaspen.core.surface import rebox_slab as _rebox_slab_fn

        _RUN_NAMESPACE = {
            "np": np,
            "numpy": np,
            "ase": _ase,
            "Atoms": _ase.Atoms,
            "FileIO": _FileIO,
            "builder": _StructureBuilder,
            "StructureBuilder": _StructureBuilder,
            "SurfaceCutter": _SurfaceCutter,
            "rebox_slab": _rebox_slab_fn,
            "symmetry": _symmetry,
            "transform": _transform,
            "neb": _neb,
            "measure": _measure_mod,
            "vasp_input": _vasp_input,
        }
    ns = dict(_RUN_NAMESPACE)
    ns["session"] = SESSION
    ns["model"] = SESSION.model
    return ns


@tool
def run_python(code: str) -> dict:
    """Execute Python against the current session — the escape hatch
    for anything the dedicated tools do not cover (the full vaspen.core
    API and numpy/ASE are available).

    Predefined names: session (the ServerSession), model (the current
    StructureModel — refreshed on every call; mutate it through its
    methods and every other tool sees the change), np/numpy, ase, Atoms,
    FileIO, StructureBuilder, SurfaceCutter, rebox_slab, symmetry,
    transform, neb, measure, vasp_input. print() output is captured;
    the value of the LAST expression is returned as "result" (e.g. end
    with `model.n_atoms` to read a value).

    Args:
        code: Python source to execute.
    """
    if not AppConfig().mcp_allow_run_python:
        raise ValueError(
            "run_python is disabled — enable 'Allow code execution "
            "(run_python)' in the VASPen Settings dialog.")
    with SESSION.lock:
        ns = _run_namespace()
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                tree = ast.parse(code, mode="exec")
                result: Any = None
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last = tree.body.pop()
                    exec(compile(tree, "<run_python>", "exec"), ns)  # noqa: S102
                    result = eval(  # noqa: S307
                        compile(ast.Expression(last.value),
                                "<run_python>", "eval"), ns)
                else:
                    exec(compile(tree, "<run_python>", "exec"), ns)  # noqa: S102
        except SyntaxError as exc:
            raise ValueError(
                f"SyntaxError: {exc.msg} (line {exc.lineno})") from exc
        except Exception as exc:
            # User-code errors are expected outcomes here — re-raise as
            # ValueError with the original type name so both transports
            # map them and the AI can self-correct.
            raise ValueError(f"{type(exc).__name__}: {exc}") from exc
        response: dict[str, Any] = {
            "ok": True,
            "stdout": buf.getvalue(),
            "result": result if isinstance(
                result, (bool, int, float, str, list, dict, type(None)))
            else repr(result),
        }
        if SESSION.model is not None and SESSION.model.n_atoms:
            response["summary"] = _structure_summary(SESSION.model)
        return response
