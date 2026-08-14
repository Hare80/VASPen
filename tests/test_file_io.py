"""Tests for the FileIO registry."""

import numpy as np
import pytest
from ase import Atoms
from ase.io import read as ase_read

from vaspen.core.file_io import EXTENSION_DISPLAY_NAMES, FileIO, resolve_format


def test_write_read_roundtrip_xyz(tmp_path):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10])
    path = tmp_path / "h2.xyz"
    FileIO.write(str(path), atoms)
    loaded = FileIO.read(str(path))
    assert loaded.get_chemical_symbols() == ["H", "H"]
    assert np.allclose(loaded.positions, atoms.positions)


def test_unsupported_extension_raises(tmp_path):
    with pytest.raises(ValueError, match="Unsupported file extension"):
        FileIO.read(str(tmp_path / "x.nope"))


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        FileIO.read("no-such-file.xyz")


def test_file_filter_contains_main_entries():
    f = FileIO.file_filter()
    assert f.startswith("Structure files (")
    assert "*.cif" in f
    assert "POSCAR CONTCAR" in f
    assert f.endswith("All files (*)")


def test_display_names_have_unique_keys():
    assert len(EXTENSION_DISPLAY_NAMES) == len(set(EXTENSION_DISPLAY_NAMES))


# ----------------------------------------------------------------------
# Format resolution
# ----------------------------------------------------------------------

def test_resolve_format_vasp_family():
    for name in ("x.POSCAR", "x.poscar", "x.CONTCAR", "x.contcar", "x.vasp",
                 "POSCAR", "CONTCAR", "poscar"):
        assert resolve_format(name) == "vasp", name
    assert resolve_format("x.cif") == "cif"
    assert resolve_format("x.xyz") == "xyz"
    assert resolve_format("x.extxyz") == "extxyz"
    assert resolve_format("x.nope") is None
    assert resolve_format("README") is None


# ----------------------------------------------------------------------
# Round-trips (periodic structures)
# ----------------------------------------------------------------------

def _assert_roundtrip(loaded, original):
    assert np.allclose(loaded.get_cell()[:], original.get_cell()[:], atol=1e-4)
    assert loaded.get_chemical_symbols() == original.get_chemical_symbols()
    assert tuple(loaded.pbc) == (True, True, True)
    # POSCAR stores 8 digits; compare fractional coordinates, wrapped to
    # [0,1) and sorted row-wise (lexicographic, atoms may be reordered)
    scaled = loaded.get_scaled_positions() % 1.0
    original_scaled = original.get_scaled_positions() % 1.0
    scaled = scaled[np.lexsort(scaled.T[::-1])]
    original_scaled = original_scaled[np.lexsort(original_scaled.T[::-1])]
    assert np.allclose(scaled, original_scaled, atol=1e-4)


@pytest.mark.parametrize("ext", [".vasp", ".poscar", ".contcar", ".POSCAR"])
def test_roundtrip_vasp_suffixes(si_bulk, tmp_path, ext):
    path = tmp_path / f"bulk{ext}"
    FileIO.write(str(path), si_bulk)
    loaded = FileIO.read(str(path))
    _assert_roundtrip(loaded, si_bulk)


@pytest.mark.parametrize("name", ["POSCAR", "CONTCAR", "poscar"])
def test_roundtrip_extensionless_poscar_contcar(si_bulk, tmp_path, name):
    path = tmp_path / name
    FileIO.write(str(path), si_bulk)
    loaded = FileIO.read(str(path))
    _assert_roundtrip(loaded, si_bulk)


def test_roundtrip_cif_sets_pbc(si_bulk, tmp_path):
    """Regression: ASE's CIF reader never sets pbc — VASPen must.

    CIF stores a/b/c/α/β/γ, so the cell matrix comes back reoriented to
    the standard crystallographic setting — compare lengths and angles.
    """
    path = tmp_path / "bulk.cif"
    FileIO.write(str(path), si_bulk)
    loaded = FileIO.read(str(path))
    assert tuple(loaded.pbc) == (True, True, True)
    assert np.allclose(loaded.get_cell().lengths(),
                       si_bulk.get_cell().lengths(), atol=1e-4)
    assert np.allclose(loaded.get_cell().angles(),
                       si_bulk.get_cell().angles(), atol=1e-4)
    assert loaded.get_chemical_symbols() == si_bulk.get_chemical_symbols()


def test_write_explicit_fmt_vasp_any_suffix(si_bulk, tmp_path):
    path = tmp_path / "out.dat"
    FileIO.write(str(path), si_bulk, fmt="vasp")
    loaded = ase_read(str(path), format="vasp")
    assert np.allclose(loaded.get_cell()[:], si_bulk.get_cell()[:], atol=1e-4)


def test_slab_partial_pbc_writes_to_vasp(tmp_path):
    """Surface slabs (partial pbc, full cell) must never be blocked."""
    slab = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]],
                 cell=[10, 10, 20], pbc=(True, True, False))
    path = tmp_path / "slab.vasp"
    FileIO.write(str(path), slab)
    assert path.exists()


# ----------------------------------------------------------------------
# Non-periodic structures
# ----------------------------------------------------------------------

@pytest.mark.parametrize("ext", [".vasp", ".cif"])
def test_write_nonperiodic_to_periodic_raises(water_molecule, tmp_path, ext):
    """Core layer refuses to write molecules to periodic formats (no file left)."""
    path = tmp_path / f"mol{ext}"
    with pytest.raises(ValueError, match="non-periodic"):
        FileIO.write(str(path), water_molecule)
    assert not path.exists()


def test_write_molecule_xyz_ok(water_molecule, tmp_path):
    """xyz is a non-periodic format — molecules round-trip unchanged."""
    path = tmp_path / "mol.xyz"
    FileIO.write(str(path), water_molecule)
    loaded = FileIO.read(str(path))
    assert np.allclose(loaded.positions, water_molecule.positions)
    assert not loaded.pbc.any()
    assert loaded.get_cell().rank == 0


def test_read_extensionless_unknown_raises(tmp_path):
    with pytest.raises(ValueError, match="Unsupported file extension"):
        FileIO.read(str(tmp_path / "README"))
    with pytest.raises(ValueError, match="Unsupported file extension"):
        FileIO.write(str(tmp_path / "out.nope"), Atoms("H"))


# ----------------------------------------------------------------------
# Partial occupancy (disordered structures)
# ----------------------------------------------------------------------

from tests.conftest import DISORDERED_CIF  # noqa: E402


def test_sanitize_cif_occupancy():
    """'?'/'.' occupancy tokens become 1.0; '0.5(2)' keeps its prefix;
    other loops and columns are untouched."""
    from vaspen.core.file_io import _sanitize_cif_occupancy

    cif = """data_x
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_occupancy
Fe1 Fe 0.0 ?
Ni1 Ni 0.5 0.5(2)
O1 O 0.25 .
loop_
_other_tag
_other2
v1 v2
"""
    out = _sanitize_cif_occupancy(cif)
    assert "Fe1 Fe 0.0 1.0" in out
    assert "Ni1 Ni 0.5 0.5" in out
    assert "O1 O 0.25 1.0" in out
    assert "v1 v2" in out  # other loops untouched


def test_cif_unknown_occupancy_reads(tmp_path):
    """Regression: ASE crashed on '?' occupancy with a str/float
    comparison (ase/spacegroup/xtal.py). FileIO sanitizes it to 1.0."""
    from vaspen.core.file_io import extract_occupancy

    cif = DISORDERED_CIF.replace("Fe1 Fe 0.0 0.0 0.0 0.5",
                                 "Fe1 Fe 0.0 0.0 0.0 ?")
    path = tmp_path / "unknown_occ.cif"
    path.write_text(cif, encoding="utf-8")
    atoms = FileIO.read(str(path))
    occ = extract_occupancy(atoms)
    assert occ is not None
    assert occ[0]["Fe"] == 1.0  # '?' normalized to full occupation


def test_cif_mixed_occupancy_preserves_info(tmp_path, disordered_atoms):
    """Co-located species merge into one atom per site; the full
    composition survives in info['occupancy'] + spacegroup_kinds."""
    from vaspen.core.file_io import extract_occupancy

    path = tmp_path / "disordered.cif"
    FileIO.write(str(path), disordered_atoms)
    loaded = FileIO.read(str(path))
    occ = extract_occupancy(loaded)
    assert occ is not None
    totals = {}
    for comp in occ:
        for sym, o in comp.items():
            totals[sym] = totals.get(sym, 0.0) + o
    assert totals == {"Fe": 2.05, "Ni": 0.5, "Co": 0.2}


def test_xyz_roundtrip_preserves_occupancy(tmp_path, disordered_atoms):
    """ASE 3.29 routes .xyz through extxyz — the occupancy info is
    serialized as a _JSON comment and must survive the round trip."""
    from vaspen.core.file_io import extract_occupancy

    path = tmp_path / "disordered.xyz"
    FileIO.write(str(path), disordered_atoms)
    loaded = FileIO.read(str(path))
    occ = extract_occupancy(loaded)
    assert occ == extract_occupancy(disordered_atoms)


def test_extract_occupancy_uniform_is_none():
    """An occupancy column of all 1.0 is not disorder."""
    import io as _io

    from vaspen.core.file_io import extract_occupancy

    cif = """data_x
_cell_length_a 3.60
_cell_length_b 3.60
_cell_length_c 3.60
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Fe1 Fe 0.0 0.0 0.0 1.0
O1 O 0.5 0.5 0.5 1.0
"""
    atoms = ase_read(_io.StringIO(cif), format="cif")
    assert extract_occupancy(atoms) is None


def test_extract_occupancy_vacancy():
    """A single partially-occupied species is a vacancy, not None."""
    import io as _io

    from vaspen.core.file_io import extract_occupancy

    cif = DISORDERED_CIF.replace(
        "Fe1 Fe 0.0 0.0 0.0 0.5\nNi1 Ni 0.0 0.0 0.0 0.5\n",
        "Fe1 Fe 0.0 0.0 0.0 0.5\n")
    atoms = ase_read(_io.StringIO(cif), format="cif")
    occ = extract_occupancy(atoms)
    assert occ is not None
    assert occ[0] == {"Fe": 0.5}


def test_write_filter_is_categorized():
    """Save filter lists formats per type (VESTA-style) with no combined
    "Structure files" entry — the read filter keeps the combined entry."""
    read_filter = FileIO.file_filter(for_writing=False)
    assert read_filter.startswith("Structure files (")

    write_filter = FileIO.file_filter(for_writing=True)
    assert "Structure files (" not in write_filter
    assert write_filter.startswith("CIF — ")
    assert write_filter.endswith("All files (*)")
    assert "POSCAR / CONTCAR (VASP) (*.vasp" in write_filter
    assert "*.cif" in write_filter and "*.xyz" in write_filter


# ----------------------------------------------------------------------
# VASP coordinate mode (fractional Direct vs Cartesian)
# ----------------------------------------------------------------------

def _poscar_keyword(path) -> str:
    """Coordinate keyword line ("Direct" or "Cartesian") of a POSCAR."""
    for line in path.read_text().splitlines():
        if line.strip() in ("Direct", "Cartesian"):
            return line.strip()
    raise AssertionError(f"No coordinate keyword found in {path}")


def test_vasp_write_default_is_cartesian(si_bulk, tmp_path):
    path = tmp_path / "POSCAR"
    FileIO.write(str(path), si_bulk)
    assert _poscar_keyword(path) == "Cartesian"


def test_vasp_write_direct_writes_fractional(si_bulk, tmp_path):
    path = tmp_path / "POSCAR"
    FileIO.write(str(path), si_bulk, direct=True)
    lines = path.read_text().splitlines()
    kw = lines.index("Direct")
    for line in lines[kw + 1:]:
        frac = [float(t) for t in line.split()[:3]]
        assert all(0.0 <= f < 1.0 for f in frac)
    # fractional mode round-trips to the same structure
    loaded = ase_read(str(path), format="vasp")
    key = lambda r: tuple(np.round(r, 6))
    assert np.allclose(sorted(loaded.get_scaled_positions() % 1.0, key=key),
                       sorted(si_bulk.get_scaled_positions() % 1.0, key=key),
                       atol=1e-4)


def test_vasp_direct_wraps_atoms_outside_cell(si_bulk, tmp_path):
    """Direct mode wraps atoms into [0,1); the caller's atoms are untouched."""
    outside = si_bulk.copy()
    outside.positions[0] += outside.get_cell()[0]
    original = outside.positions.copy()

    path = tmp_path / "POSCAR"
    FileIO.write(str(path), outside, direct=True)

    assert np.allclose(outside.positions, original)
    assert _poscar_keyword(path) == "Direct"
    loaded = ase_read(str(path), format="vasp")
    frac = loaded.get_scaled_positions()
    assert ((frac >= 0.0) & (frac < 1.0)).all()
    # wrapping is a periodic identity: fractions match modulo 1
    key = lambda r: tuple(np.round(r, 6))
    assert np.allclose(sorted(frac % 1.0, key=key),
                       sorted(outside.get_scaled_positions() % 1.0, key=key),
                       atol=1e-4)


def test_vasp_cartesian_keeps_atoms_outside_cell(si_bulk, tmp_path):
    """Cartesian mode writes actual positions — no wrapping applied."""
    outside = si_bulk.copy()
    outside.positions[0] += outside.get_cell()[0]

    path = tmp_path / "POSCAR"
    FileIO.write(str(path), outside, direct=False)

    assert _poscar_keyword(path) == "Cartesian"
    loaded = ase_read(str(path), format="vasp")
    assert np.allclose(loaded.positions[0], outside.positions[0], atol=1e-4)
