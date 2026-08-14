"""Tests for VASP input generation (INCAR/KPOINTS/POTCAR)."""

import re

import numpy as np
import pytest
from ase import Atoms

from vaspen.core.structure import StructureModel
from vaspen.core.vasp_input import (
    INCAR_PRESETS,
    INCAR_SUGGESTIONS,
    _format_incar_value,
    available_variants,
    estimate_k_mesh,
    format_incar_content,
    generate_all_inputs,
    generate_kpoints_automatic,
    generate_kpoints_line_mode,
    generate_kpoints_manual,
    generate_potcar,
    get_high_symmetry_points,
    get_potcar_recommendation,
    parse_incar_content,
    resolve_potcar_dir,
)


# ----------------------------------------------------------------------
# k-mesh estimate
# ----------------------------------------------------------------------

def test_estimate_k_mesh_vaspkit_convention():
    # vaspkit convention (KSPACING in 2π/Å): N_i = max(1, ceil(|b_i|/K))
    # with normalized reciprocal vectors (b_i·a_j = δ_ij).
    # Si cubic: |b| = 1/5.43 = 0.18414 → ceil(0.18414/0.04) = 5
    cell = np.eye(3) * 5.43
    assert estimate_k_mesh(cell, 0.04) == (5, 5, 5)
    # ceiling semantics: ceil(0.18414/0.05) = ceil(3.6828) = 4
    assert estimate_k_mesh(cell, 0.05) == (4, 4, 4)


def test_estimate_k_mesh_gaas_fcc_primitive():
    """User example: GaAs primitive cell (FCC basis), a = 5.6537.
    |b| = √3/a = 0.30636."""
    a = 5.6537
    cell = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * a
    assert estimate_k_mesh(cell, 0.030) == (11, 11, 11)  # ceil(10.212)
    assert estimate_k_mesh(cell, 0.020) == (16, 16, 16)  # ceil(15.318)
    assert estimate_k_mesh(cell, 0.040) == (8, 8, 8)     # ceil(7.659)


def test_estimate_k_mesh_zno_hexagonal():
    """User example: ZnO hexagonal, a = 3.289, c = 5.307.
    |b1| = |b2| = 1/(a·sin60°) = 0.35108, |b3| = 1/c = 0.18843."""
    a, c = 3.289, 5.307
    cell = np.array([
        [a, 0.0, 0.0],
        [-0.5 * a, a * np.sqrt(3) / 2, 0.0],
        [0.0, 0.0, c],
    ])
    assert estimate_k_mesh(cell, 0.040) == (9, 9, 5)


def test_estimate_k_mesh_floor_is_one():
    cell = np.eye(3) * 50.0  # very large cell → coarse mesh
    mesh = estimate_k_mesh(cell, 0.05)
    assert all(m >= 1 for m in mesh)


def test_generate_kpoints_automatic_writes_explicit_mesh():
    cell = np.eye(3) * 5.43
    content = generate_kpoints_automatic(cell, k_spacing=0.04, gamma_centered=True)
    lines = content.splitlines()
    assert lines[1] == "0"
    assert lines[2] == "Gamma"
    assert lines[3] == "5 5 5"
    assert lines[4] == "0 0 0"

    content_mp = generate_kpoints_automatic(cell, k_spacing=0.04, gamma_centered=False)
    assert content_mp.splitlines()[2] == "Monkhorst-Pack"


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
    assert re.search(r"^\s*ENCUT\s*=\s*400", files["INCAR"], re.M)
    assert "Gamma" in files["KPOINTS"]
    assert "9 9 9" in files["KPOINTS"]  # cell [3,3,3], KSPACING 0.04 → ceil(8.33)


# ----------------------------------------------------------------------
# Non-periodic structures: wrap first, then generate
# ----------------------------------------------------------------------

def test_generate_all_poscar_direct_mode(water_molecule):
    """poscar_direct=True writes fractional (Direct) coordinates."""
    model = StructureModel()
    model.load_atoms(water_molecule)
    model.make_periodic(10.0)

    cartesian = generate_all_inputs(model)
    direct = generate_all_inputs(model, poscar_direct=True)

    assert "Cartesian\n" in cartesian["POSCAR"]
    assert "Direct\n" in direct["POSCAR"]
    frac = [float(t) for t in direct["POSCAR"].splitlines()[-3].split()[:3]]
    assert all(0.0 <= f < 1.0 for f in frac)


def test_generate_all_after_make_periodic(water_molecule):
    """After make_periodic, POSCAR/KPOINTS generation must not crash."""
    model = StructureModel()
    model.load_atoms(water_molecule)
    model.make_periodic(10.0)

    files = generate_all_inputs(model, incar_preset="scf", kpoints_mode="automatic")

    poscar_lines = files["POSCAR"].splitlines()
    assert len(poscar_lines) >= 6  # comment + scaling + 3 lattice vectors
    # the 3 lattice-vector lines are numeric
    assert all(
        len(line.split()) == 3 for line in poscar_lines[2:5]
    )
    mesh_line = [line for line in files["KPOINTS"].splitlines()
                 if len(line.split()) == 3 and line.split()[0].isdigit()]
    assert mesh_line, files["KPOINTS"]
    n1, n2, n3 = (int(x) for x in mesh_line[0].split())
    assert n1 >= 1 and n2 >= 1 and n3 >= 1


# ----------------------------------------------------------------------
# generate_all_inputs — fixed atoms (Selective dynamics)
# ----------------------------------------------------------------------

def test_generate_all_poscar_has_selective_dynamics_when_fixed():
    model = StructureModel()
    model.load_atoms(Atoms(
        "OFe2", positions=np.eye(3) * 1.5, cell=[3, 3, 3], pbc=True,
    ))
    model.set_fixed(0, True)

    files = generate_all_inputs(model)

    lines = files["POSCAR"].splitlines()
    assert "Selective dynamics" in lines
    coord_line = lines.index("Selective dynamics") + 1
    first_coords = lines[coord_line + 1].split()
    assert first_coords[3:6] == ["F", "F", "F"]
    second_coords = lines[coord_line + 2].split()
    assert second_coords[3:6] == ["T", "T", "T"]
    # other files unaffected
    assert re.search(r"^\s*ENCUT\s*=\s*400", files["INCAR"], re.M)
    assert "Gamma" in files["KPOINTS"]


def test_generate_all_poscar_plain_when_all_free():
    model = StructureModel()
    model.load_atoms(Atoms(
        "OFe2", positions=np.eye(3) * 1.5, cell=[3, 3, 3], pbc=True,
    ))
    files = generate_all_inputs(model)
    assert "Selective dynamics" not in files["POSCAR"]


# ----------------------------------------------------------------------
# INCAR formatting — aligned comments, no duplicate tags, .TRUE. bools
# ----------------------------------------------------------------------

def _preset_incar(preset: str) -> str:
    """Render one preset's tags with the production formatter."""
    tags = dict(INCAR_PRESETS[preset])
    tags.pop("_description", None)
    return format_incar_content(tags, suggestions=INCAR_SUGGESTIONS[preset])


def _active_tags(incar: str) -> list[str]:
    """Tag names of active (non-commented) lines in INCAR content."""
    tags = []
    for line in incar.splitlines():
        if line.startswith("  #"):
            continue
        m = re.match(r"^\s*(\w+)\s*=", line)
        if m:
            tags.append(m.group(1))
    return tags


def test_incar_no_duplicate_tags():
    model = StructureModel()
    model.load_atoms(Atoms(
        "OFe2", positions=np.eye(3) * 1.5, cell=[3, 3, 3], pbc=True,
    ))
    for preset in INCAR_PRESETS:
        content = generate_all_inputs(model, incar_preset=preset)["INCAR"]
        tags = _active_tags(content)
        assert len(tags) == len(set(tags)), f"{preset}: {tags}"
    # overrides replace an existing tag instead of duplicating it
    content = generate_all_inputs(model, incar_overrides={"ENCUT": 500})["INCAR"]
    assert _active_tags(content).count("ENCUT") == 1


def test_incar_comments_aligned_at_column_26():
    for preset in INCAR_PRESETS:
        for line in _preset_incar(preset).splitlines():
            if "(" in line and not line.startswith("  #"):
                assert line[25] == "(", repr(line)


def test_incar_line_format_matches_reference():
    """'(' at column 26, e.g. '  NSW    =  300          (number of...)'."""
    content = format_incar_content(
        {"NSW": 300}, comments={"NSW": "number of ionic steps"}
    )
    assert content.splitlines()[0] == "  NSW    =  300          (number of ionic steps)"


def test_incar_bools_as_dot_true():
    content = _preset_incar("neb")
    assert ".TRUE." in content
    assert "True" not in content


def test_band_preset_lcharg_true():
    content = _preset_incar("band")
    assert re.search(r"^\s*LCHARG\s*=\s*\.TRUE\.", content, re.M)
    assert re.search(r"^\s*ICHARG\s*=\s*11", content, re.M)


def test_dos_nedos_2001():
    assert re.search(r"^\s*NEDOS\s*=\s*2001", _preset_incar("dos"), re.M)


def test_optical_sigma_001():
    assert re.search(r"^\s*SIGMA\s*=\s*0\.01\b", _preset_incar("optical"), re.M)


def test_neb_nsw_500():
    assert re.search(r"^\s*NSW\s*=\s*500\b", _preset_incar("neb"), re.M)


def test_ediﬀ_stays_1e_6():
    content = _preset_incar("scf")
    assert re.search(r"^\s*EDIFF\s*=\s*1E-06\b", content, re.M)


def test_ispin_active_default_1():
    for preset in INCAR_PRESETS:
        content = _preset_incar(preset)
        assert re.search(r"^\s*ISPIN\s*=\s*1\b", content, re.M), preset
        assert not any(
            l.startswith("  # ISPIN") for l in content.splitlines()
        ), preset


def test_system_line_has_no_comment():
    for preset in INCAR_PRESETS:
        content = _preset_incar(preset)
        line = next(l for l in content.splitlines() if l.startswith("SYSTEM"))
        assert "(" not in line, line


def test_suggestions_commented_out():
    content = _preset_incar("scf")
    lines = content.splitlines()
    assert any(l.startswith("  # MAGMOM") for l in lines)
    assert "MAGMOM" not in _active_tags(content)
    for preset in INCAR_PRESETS:
        content = _preset_incar(preset)
        assert any(l.startswith("  # IVDW") for l in content.splitlines()), preset
        assert any(l.startswith("  # ISTART") for l in content.splitlines()), preset


def test_format_incar_value():
    assert _format_incar_value(True) == ".TRUE."
    assert _format_incar_value(False) == ".FALSE."
    assert _format_incar_value(1e-8) == "1E-08"
    assert _format_incar_value(0.05) == "0.05"
    assert _format_incar_value(-0.02) == "-0.02"
    assert _format_incar_value(400) == "400"
    assert _format_incar_value("Auto") == "Auto"
    assert _format_incar_value("True") == ".TRUE."
    assert _format_incar_value("false") == ".FALSE."
    assert _format_incar_value("") == ""


def test_format_incar_content_unknown_tag_no_comment():
    content = format_incar_content({"CUSTOM": 3, "SYSTEM": "x"})
    lines = content.splitlines()
    assert lines[0].startswith("  CUSTOM ")
    assert lines[1] == "SYSTEM = x"
    assert "(" not in lines[0]
    assert lines[0].rstrip() == lines[0]  # no trailing spaces


def test_format_incar_content_suggestion_skipped_when_active():
    content = format_incar_content(
        {"ISPIN": 2}, suggestions={"ISPIN": 2, "IVDW": 11}
    )
    assert _active_tags(content).count("ISPIN") == 1
    assert not any(l.startswith("  # ISPIN") for l in content.splitlines())
    assert any(l.startswith("  # IVDW") for l in content.splitlines())


# ----------------------------------------------------------------------
# parse_incar_content — recognition of generated / hand-edited INCAR
# ----------------------------------------------------------------------

def test_parse_roundtrips_presets():
    """parse(format(preset)) recovers every tag and normalized value."""
    for preset in INCAR_PRESETS:
        tags = dict(INCAR_PRESETS[preset])
        tags.pop("_description", None)
        parsed, problems = parse_incar_content(
            format_incar_content(tags, suggestions=INCAR_SUGGESTIONS[preset])
        )
        assert problems == [], preset
        assert list(parsed) == list(tags), preset  # same order
        for tag, value in tags.items():
            expected = _format_incar_value(value)
            assert parsed[tag] == expected, f"{preset}: {tag}"


def test_parse_skips_comment_and_blank_lines():
    text = "! a comment\n\n  ENCUT = 400\n  # IVDW = 11\n   \n!LDAU = .TRUE.\n"
    tags, problems = parse_incar_content(text)
    assert problems == []
    assert list(tags) == ["ENCUT"]
    assert tags["ENCUT"] == "400"


def test_parse_system_keeps_full_line():
    tags, _ = parse_incar_content("SYSTEM = My job (2026-08-14)\nENCUT = 400\n")
    assert tags["SYSTEM"] == "My job (2026-08-14)"


def test_parse_strips_trailing_comment():
    tags, problems = parse_incar_content(
        "  ENCUT  =  400          (plane-wave cutoff in eV)\n"
    )
    assert problems == []
    assert tags["ENCUT"] == "400"


def test_parse_reports_malformed_lines():
    tags, problems = parse_incar_content("ENCUT = 400\nthis is not a tag\n")
    assert list(tags) == ["ENCUT"]
    assert problems == [(2, "malformed", "this is not a tag")]


def test_parse_reports_case_insensitive_duplicates():
    tags, problems = parse_incar_content("ENCUT = 400\nencut = 500\n")
    assert tags["ENCUT"] == "400"  # first occurrence wins
    assert problems == [(2, "duplicate", "encut")]


def test_parse_empty_value():
    tags, problems = parse_incar_content("MAGMOM =\n")
    assert problems == []
    assert tags["MAGMOM"] == ""
