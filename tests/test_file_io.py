"""Tests for the FileIO registry."""

import numpy as np
import pytest
from ase import Atoms

from vaspen.core.file_io import EXTENSION_DISPLAY_NAMES, FileIO


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
    assert f.endswith("All files (*)")


def test_display_names_have_unique_keys():
    assert len(EXTENSION_DISPLAY_NAMES) == len(set(EXTENSION_DISPLAY_NAMES))
