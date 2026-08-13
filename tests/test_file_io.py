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
