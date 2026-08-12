"""Tests for VASP input generation (INCAR/KPOINTS/POTCAR)."""

import numpy as np
import pytest
from ase import Atoms

from vaspen.core.structure import StructureModel
from vaspen.core.vasp_input import (
    available_variants,
    estimate_k_mesh,
    generate_all_inputs,
    generate_kpoints_automatic,
    generate_kpoints_line_mode,
    generate_kpoints_manual,
    generate_potcar,
    get_high_symmetry_points,
    get_potcar_recommendation,
    resolve_potcar_dir,
)


# ----------------------------------------------------------------------
# k-mesh estimate
# ----------------------------------------------------------------------

def test_estimate_k_mesh_matches_vasp_formula():
    # Si: a = 5.43 Å → |b| = 2π/5.43 ≈ 1.1569 → ceil(1.1569 / 0.04) = 29
    cell = np.eye(3) * 5.43
    assert estimate_k_mesh(cell, 0.04) == (29, 29, 29)


def test_estimate_k_mesh_floor_is_one():
    cell = np.eye(3) * 50.0  # very large cell → coarse mesh
    mesh = estimate_k_mesh(cell, 0.05)
    assert all(m >= 1 for m in mesh)


# ----------------------------------------------------------------------
# Line-mode KPOINTS
# ----------------------------------------------------------------------

SIMPLE_SPECIAL = {
    "G": np.array([0.0, 0.0, 0.0]),
    "X": np.array([0.5, 0.0, 0.0]),
    "M": np.array([0.5, 0.5, 0.0]),
}


def test_line_mode_requires_special_points():
    with pytest.raises(ValueError, match="special_points"):
        generate_kpoints_line_mode([("G", "X")], 20)


def test_line_mode_emits_valid_coordinate_lines():
    content = generate_kpoints_line_mode(
        [("G", "X"), ("X", "M")], n_points_per_segment=20,
        special_points=SIMPLE_SPECIAL,
    )
    lines = content.splitlines()
    assert lines[0].startswith("Band structure: G-X|X-M")
    assert lines[1] == "40"        # 2 segments × 20 points
    assert lines[2] == "Line-mode"
    assert lines[3] == "Reciprocal"
    assert len(lines) == 4 + 40    # one coordinate line per declared point

    # segment 1 start = G, weight 1
    parts = lines[4].split()
    assert len(parts) == 4
    assert [float(x) for x in parts[:3]] == [0.0, 0.0, 0.0]
    assert parts[3] == "1.0"
    # interior point has weight 0
    assert lines[5].split()[3] == "0.0"
    # segment 2 end = M, weight 1
    parts = lines[-1].split()
    assert [round(float(x), 8) for x in parts[:3]] == [0.5, 0.5, 0.0]
    assert parts[3] == "1.0"


def test_line_mode_unknown_label_raises():
    with pytest.raises(ValueError, match="Unknown k-point label 'X'"):
        generate_kpoints_line_mode(
            [("G", "X")], 20, special_points={"G": np.zeros(3)}
        )


def test_high_symmetry_points_for_si():
    special = get_high_symmetry_points(np.eye(3) * 5.43)
    assert "G" in special and "X" in special
    assert np.allclose(special["G"], [0, 0, 0], atol=1e-6)


# ----------------------------------------------------------------------
# POTCAR
# ----------------------------------------------------------------------

def _fake_library(tmp_path):
    lib = tmp_path / "PBE.54"
    for el in ("O", "Fe"):
        d = lib / el
        d.mkdir(parents=True)
        (d / "POTCAR").write_text(f"POTCAR-{el}\n", encoding="utf-8")
    return tmp_path


def test_generate_potcar_concatenates_in_element_order(tmp_path):
    lib = _fake_library(tmp_path)
    content, paths = generate_potcar(["O", "Fe"], lib, "PBE")
    assert content == "POTCAR-O\nPOTCAR-Fe\n"
    assert len(paths) == 2


def test_potcar_recommendation_uses_d_variant():
    assert get_potcar_recommendation("Ga", "PBE") == "Ga_d"
    assert get_potcar_recommendation("Fe", "PBE") == "Fe"


def test_wiki_recommendations_per_functional():
    """Defaults follow the VASP wiki 'Available pseudopotentials' tables."""
    # PBE.54/64 semi-core rules
    assert get_potcar_recommendation("Li", "PBE") == "Li_sv"
    assert get_potcar_recommendation("Na", "PBE") == "Na_pv"
    assert get_potcar_recommendation("W", "PBE") == "W_sv"
    assert get_potcar_recommendation("Ta", "PBE") == "Ta_pv"
    assert get_potcar_recommendation("Bi", "PBE") == "Bi_d"
    # As/Sb are PLAIN on the wiki (no _d)
    assert get_potcar_recommendation("As", "PBE") == "As"
    assert get_potcar_recommendation("Sb", "PBE") == "Sb"
    # lanthanides with fixed valence
    assert get_potcar_recommendation("Pr", "PBE") == "Pr_3"
    assert get_potcar_recommendation("Eu", "PBE") == "Eu_2"
    assert get_potcar_recommendation("Yb", "PBE") == "Yb_2"
    # LDA has no fixed-valence lanthanides
    assert get_potcar_recommendation("Pr", "LDA") == "Pr"
    assert get_potcar_recommendation("Ga", "LDA") == "Ga_d"
    # PW91 (2010) lacks W_sv / Po_d
    assert get_potcar_recommendation("W", "PW91") == "W"
    assert get_potcar_recommendation("Po", "PW91") == "Po"
    assert get_potcar_recommendation("Ga", "PW91") == "Ga_d"
    # PBE_new shares the PBE lists
    assert get_potcar_recommendation("W", "PBE_new") == "W_sv"


def test_resolve_potcar_dir_prefers_potpaw_name(tmp_path):
    (tmp_path / "potpaw_PBE.54").mkdir()
    (tmp_path / "PBE.54").mkdir()
    assert resolve_potcar_dir(tmp_path, "PBE").name == "potpaw_PBE.54"


def test_resolve_potcar_dir_falls_back_to_newer_release(tmp_path):
    """A library that only ships .64 still works for PBE/LDA."""
    (tmp_path / "potpaw_PBE.64").mkdir()
    (tmp_path / "potpaw_LDA.64").mkdir()
    assert resolve_potcar_dir(tmp_path, "PBE").name == "potpaw_PBE.64"
    assert resolve_potcar_dir(tmp_path, "LDA").name == "potpaw_LDA.64"


def test_generate_potcar_error_lists_available_dirs(tmp_path):
    (tmp_path / "potpaw_PBE.64").mkdir()
    with pytest.raises(FileNotFoundError, match="potpaw_PBE.64"):
        generate_potcar(["Fe"], tmp_path, "PW91")


def test_available_variants_discovers_wiki_variants(tmp_path):
    lib = tmp_path / "potpaw_PBE.54"
    for name in ("Ga", "Ga_d", "Ga_h", "Fe", "H.5", "junk"):
        (lib / name).mkdir(parents=True)
    variants = available_variants(tmp_path, "PBE", "Ga")
    assert variants[0] == "Ga_d"  # wiki recommendation first
    assert set(variants) == {"Ga_d", "Ga", "Ga_h"}
    # fractional hydrogen variants are recognized
    assert "H.5" in available_variants(tmp_path, "PBE", "H")
    # non-matching dirs are ignored
    assert "junk" not in variants


def test_available_variants_empty_without_library(tmp_path):
    assert available_variants(tmp_path, "PBE", "Fe") == []


def test_unknown_functional_raises():
    with pytest.raises(ValueError, match="Unknown functional"):
        get_potcar_recommendation("Fe", "HSE")


# ----------------------------------------------------------------------
# generate_all_inputs — POSCAR/POTCAR element order must agree
# ----------------------------------------------------------------------

def test_generate_all_inputs_element_order_consistent(tmp_path):
    lib = _fake_library(tmp_path)
    model = StructureModel()
    model.load_atoms(Atoms(
        "OFe2", positions=np.eye(3) * 1.5, cell=[3, 3, 3], pbc=True,
    ))
    files = generate_all_inputs(
        model, incar_preset="scf", kpoints_mode="automatic",
        potcar_library=str(lib),
    )
    # both use first-appearance order: O first, then Fe
    assert files["POSCAR"].splitlines()[0].split() == ["O", "Fe"]
    assert files["POTCAR"] == "POTCAR-O\nPOTCAR-Fe\n"
    # INCAR/KPOINTS sanity
    assert "ENCUT = 400" in files["INCAR"]
    assert "Auto" in files["KPOINTS"]
