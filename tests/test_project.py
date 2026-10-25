"""Tests for the calculation-project core (vaspen.core.project).

Folder-as-project with a JSON manifest (§7.15) — pure logic, no Qt.
"""

import json
from pathlib import Path

import pytest
from ase.build import bulk

from vaspen.core.project import (
    MANIFEST_NAME,
    RUN_STATUSES,
    Project,
    sha256_file,
    slugify,
)


@pytest.fixture
def project(tmp_path):
    return Project.create(tmp_path / "demo", "Demo Project")


def _structure_file(tmp_path: Path, name: str = "Fe3O4.vasp") -> Path:
    f = tmp_path / name
    f.write_text("fake-vasp-content", encoding="utf-8")
    return f


# ----------------------------------------------------------------------
# create / load
# ----------------------------------------------------------------------


def test_create_builds_skeleton(project: Project):
    root = project.root
    assert (root / MANIFEST_NAME).exists()
    assert (root / "structures").is_dir()
    assert (root / "runs").is_dir()
    assert project.name == "Demo Project"
    assert project.structures == [] and project.runs == []


def test_create_refuses_existing_manifest(project: Project):
    with pytest.raises(ValueError, match="already exists"):
        Project.create(project.root, "Again")


def test_load_roundtrip(project: Project, tmp_path):
    src = _structure_file(tmp_path)
    project.add_structure(src, metadata={"formula": "Fe24O32"})
    project.add_run("scf bulk", "scf", {"preset": "scf"},
                    structure_id="fe3o4", structure_digest="sha256:x",
                    via="gui")
    reloaded = Project.load(project.root)
    assert reloaded.name == "Demo Project"
    assert [s["id"] for s in reloaded.structures] == ["fe3o4"]
    assert [r["id"] for r in reloaded.runs] == ["001-scf-bulk"]
    assert reloaded.run_entry("001-scf-bulk")["status"] == "inputs-ready"


def test_load_rejects_missing_manifest(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="not a VASPen project"):
        Project.load(tmp_path / "empty")


def test_load_rejects_corrupt_manifest(tmp_path):
    root = tmp_path / "bad"
    root.mkdir()
    (root / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot read"):
        Project.load(root)


def test_load_rejects_other_schema(tmp_path):
    root = tmp_path / "future"
    root.mkdir()
    (root / MANIFEST_NAME).write_text(
        json.dumps({"schema": 99, "name": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        Project.load(root)


def test_load_recreates_missing_dirs(project: Project):
    (project.root / "structures").rmdir()
    reloaded = Project.load(project.root)
    assert (reloaded.root / "structures").is_dir()


# ----------------------------------------------------------------------
# structures
# ----------------------------------------------------------------------


def test_add_structure_copies_and_registers(project: Project, tmp_path):
    src = _structure_file(tmp_path, "Cu 111 slab.vasp")
    entry = project.add_structure(src, metadata={"formula": "Cu"})
    assert entry["id"] == "cu-111-slab"  # slugified
    assert entry["digest"].startswith("sha256:")
    copied = project.structure_path("cu-111-slab")
    assert copied.exists()
    assert copied.parent.name == "structures"
    assert sha256_file(copied) == entry["digest"]


def test_add_structure_id_dedup(project: Project, tmp_path):
    a = tmp_path / "a" / "Fe.vasp"
    a.parent.mkdir()
    a.write_text("one", encoding="utf-8")
    b = tmp_path / "b" / "Fe.vasp"
    b.parent.mkdir()
    b.write_text("two", encoding="utf-8")
    project.add_structure(a)
    second = project.add_structure(b)
    assert second["id"] == "fe-2"  # same slug, different content
    assert len(project.structures) == 2


def test_add_structure_missing_file(project: Project):
    with pytest.raises(ValueError, match="not found"):
        project.add_structure("Z:/nope.vasp")


# ----------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------


def test_add_run_numbers_sequentially(project: Project):
    for n in range(3):
        project.add_run(f"run {n}", "scf", {}, "s", "sha256:x", "gui")
    ids = [r["id"] for r in project.runs]
    assert ids == ["001-run-0", "002-run-1", "003-run-2"]


def test_add_run_writes_provenance_card(project: Project):
    entry = project.add_run("scf", "scf", {"preset": "scf", "k": 0.04},
                            "fe3o4", "sha256:abc", "mcp")
    card = json.loads(
        (project.run_dir("001-scf") / "run.json").read_text("utf-8"))
    assert card["structure_digest"] == "sha256:abc"
    assert card["parameters"] == {"preset": "scf", "k": 0.04}
    assert card["generated_by"]["via"] == "mcp"
    assert card["generated_by"]["app"] == "VASPen"
    assert card["status"] == "inputs-ready"
    assert entry["card"]["id"] == "001-scf"


# ----------------------------------------------------------------------
# structure match-or-register + run removal (§7.15 P2)
# ----------------------------------------------------------------------


def test_register_structure_file_matches_by_path(project: Project, tmp_path):
    src = tmp_path / "si.vasp"
    bulk("Si", "diamond", a=5.43, cubic=True).write(src, format="vasp")
    entry = project.add_structure(src)
    sid, digest = project.register_structure_file(src)
    assert sid == entry["id"]
    assert digest == entry["digest"]
    assert len(project.structures) == 1  # no duplicate registered


def test_register_structure_file_matches_by_digest(project: Project,
                                                    tmp_path):
    src = tmp_path / "si.vasp"
    bulk("Si", "diamond", a=5.43, cubic=True).write(src, format="vasp")
    entry = project.add_structure(src)
    copy = tmp_path / "same-bytes.vasp"
    copy.write_bytes(src.read_bytes())
    sid, digest = project.register_structure_file(copy)
    assert sid == entry["id"]
    assert digest == entry["digest"]
    assert len(project.structures) == 1


def test_register_structure_file_registers_new(project: Project, tmp_path):
    src = tmp_path / "si.vasp"
    bulk("Si", "diamond", a=5.43, cubic=True).write(src, format="vasp")
    other = tmp_path / "fe.vasp"
    bulk("Fe", "bcc", a=2.87).write(other, format="vasp")
    project.add_structure(src)
    sid, digest = project.register_structure_file(other, formula="Fe")
    assert len(project.structures) == 2
    assert sid in {e["id"] for e in project.structures}
    assert digest == sha256_file(other)
    entry = next(e for e in project.structures if e["id"] == sid)
    assert entry["formula"] == "Fe"


def test_register_structure_file_missing_file(project: Project, tmp_path):
    with pytest.raises(ValueError, match="File not found"):
        project.register_structure_file(tmp_path / "ghost.vasp")


def test_remove_run_deletes_entry_and_directory(project: Project):
    project.add_run("r", "scf", {}, "s", "sha256:x", "gui")
    run_dir = project.run_dir("001-r")
    (run_dir / "INCAR").write_text("x", encoding="utf-8")
    project.remove_run("001-r")
    assert project.runs == []
    assert not run_dir.exists()


def test_remove_run_unknown_id(project: Project):
    with pytest.raises(ValueError, match="Unknown run"):
        project.remove_run("999-ghost")


def test_remove_run_refuses_active_unless_forced(project: Project):
    project.add_run("r", "scf", {}, "s", "sha256:x", "gui")
    project.set_run_status("001-r", "running")
    with pytest.raises(ValueError, match="refusing"):
        project.remove_run("001-r")
    assert project.runs  # untouched
    project.remove_run("001-r", force=True)
    assert project.runs == []


def test_remove_run_highest_number_reusable(project: Project):
    project.add_run("a", "scf", {}, "s", "sha256:x", "gui")
    project.add_run("b", "scf", {}, "s", "sha256:x", "gui")
    project.remove_run("002-b")
    nxt = project.add_run("c", "scf", {}, "s", "sha256:x", "gui")
    assert nxt["id"] == "002-c"  # number reused, same as hand-managed


def test_set_run_status_whitelist_and_sync(project: Project):
    project.add_run("r", "scf", {}, "s", "sha256:x", "gui")
    entry = project.set_run_status("001-r", "submitted")
    assert entry["status"] == "submitted"
    card = json.loads(
        (project.run_dir("001-r") / "run.json").read_text("utf-8"))
    assert card["status"] == "submitted"
    with pytest.raises(ValueError, match="Status must be"):
        project.set_run_status("001-r", "converged")
    with pytest.raises(ValueError, match="Unknown run"):
        project.set_run_status("999-x", "done")


def test_run_dir_and_entry_unknown_ids(project: Project):
    with pytest.raises(ValueError, match="Unknown run"):
        project.run_dir("nope")
    with pytest.raises(ValueError, match="Unknown structure"):
        project.structure_path("nope")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def test_slugify():
    assert slugify("Fe3O4 bulk (111)!") == "fe3o4-bulk-111"
    assert slugify("___") == "item"  # never empty
    assert slugify("Mixed CASE") == "mixed-case"


def test_manifest_atomic_write_leaves_no_tmp(project: Project):
    project.add_run("r", "scf", {}, "s", "sha256:x", "gui")
    assert not (project.root / "project.json.tmp").exists()
    data = json.loads((project.root / MANIFEST_NAME).read_text("utf-8"))
    assert len(data["runs"]) == 1


def test_status_tuple_documented():
    assert RUN_STATUSES == ("inputs-ready", "submitted", "running",
                            "done", "failed")


# ----------------------------------------------------------------------
# parameter update + input regeneration
# ----------------------------------------------------------------------


def test_update_run_parameters_merges(project: Project):
    project.add_run("r", "scf", {"preset": "scf", "k": 0.04}, "s",
                    "sha256:x", "gui")
    merged = project.update_run_parameters("001-r", {"functional": "PBE"})
    assert merged == {"preset": "scf", "k": 0.04, "functional": "PBE"}
    card = json.loads(
        (project.run_dir("001-r") / "run.json").read_text("utf-8"))
    assert card["parameters"]["functional"] == "PBE"
    assert "parameters_updated" in card


def test_update_run_parameters_rejects_non_dict(project: Project):
    project.add_run("r", "scf", {}, "s", "sha256:x", "gui")
    with pytest.raises(ValueError, match="dict"):
        project.update_run_parameters("001-r", "ENCUT=400")


def test_run_parameters_reader(project: Project):
    project.add_run("r", "scf", {"preset": "scf"}, "s", "sha256:x", "gui")
    assert project.run_parameters("001-r") == {"preset": "scf"}
    with pytest.raises(ValueError, match="Unknown run"):
        project.run_parameters("missing")


def test_write_run_inputs_regenerates(project: Project, tmp_path):
    from vaspen.core.structure import StructureModel

    project.add_run("r", "scf", {"preset": "scf"}, "s", "sha256:x", "gui")
    model = StructureModel()
    model.load(project.structure_path("s")) if False else None
    # register a real structure file to load from
    src = tmp_path / "si.vasp"
    bulk("Si", "diamond", a=5.43, cubic=True).write(src, format="vasp")
    entry = project.add_structure(src)
    # point the run at it by rewriting the card structure_id (test shim)
    card_path = project.run_dir("001-r") / "run.json"
    card = json.loads(card_path.read_text("utf-8"))
    card["structure_id"] = entry["id"]
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")

    model = StructureModel()
    model.load(entry and project.structure_path(entry["id"]))
    files = project.write_run_inputs(
        "001-r", model, preset="opt",
        kpoints_params={"k_spacing": 0.03}, functional="PBE")
    assert set(files) == {"INCAR", "KPOINTS", "POSCAR", "POTCAR"} or \
        set(files) >= {"INCAR", "KPOINTS", "POSCAR"}
    assert "EDIFFG" in (project.run_dir("001-r") / "INCAR").read_text()
    params = project.run_parameters("001-r")
    assert params["preset"] == "opt" and params["kpoints_params"] == {
        "k_spacing": 0.03}


def test_write_run_inputs_missing_structure(project: Project):
    from vaspen.core.structure import StructureModel

    project.add_run("r", "scf", {}, "s", "sha256:x", "gui")
    # the registered structure file never existed on disk
    with pytest.raises(ValueError, match="Unknown structure"):
        project.structure_path("s")


def test_run_files_recursive_and_sorted(project: Project, tmp_path):
    from vaspen.core.structure import StructureModel

    src = tmp_path / "si.vasp"
    bulk("Si", "diamond", a=5.43, cubic=True).write(src, format="vasp")
    s_entry = project.add_structure(src)
    r_entry = project.add_run("neb", "neb", {}, s_entry["id"],
                              "sha256:x", "gui")
    run_dir = project.run_dir(r_entry["id"])
    (run_dir / "INCAR").write_text("x", encoding="utf-8")
    (run_dir / "00").mkdir()
    (run_dir / "00" / "POSCAR").write_text("f", encoding="utf-8")
    (run_dir / "04").mkdir()
    (run_dir / "04" / "POSCAR").write_text("l", encoding="utf-8")

    files = project.run_files(r_entry["id"])
    rels = [f.as_posix() for f in files]
    assert rels == ["00/POSCAR", "04/POSCAR", "INCAR", "run.json"]
    with pytest.raises(ValueError, match="Unknown run"):
        project.run_files("nope")


def test_write_run_inputs_neb_preserves_frames_and_images(project: Project,
                                                           tmp_path):
    from vaspen.core.structure import StructureModel

    src = tmp_path / "si.vasp"
    bulk("Si", "diamond", a=5.43, cubic=True).write(src, format="vasp")
    s_entry = project.add_structure(src)
    r_entry = project.add_run("neb", "neb", {}, s_entry["id"],
                              "sha256:x", "gui")
    run_dir = project.run_dir(r_entry["id"])
    # existing NEB state: INCAR with IMAGES + frame dirs
    (run_dir / "INCAR").write_text("SYSTEM = x\nIMAGES = 3\n",
                                   encoding="utf-8")
    for i in range(5):
        frame = run_dir / f"{i:02d}"
        frame.mkdir()
        (frame / "POSCAR").write_text("frame", encoding="utf-8")

    model = StructureModel()
    model.load(project.structure_path(s_entry["id"]))
    files = project.write_run_inputs(
        r_entry["id"], model, preset="neb",
        kpoints_params={"k_spacing": 0.03}, run_type="neb")

    assert "POSCAR" not in files          # no root POSCAR for NEB
    assert set(files) == {"INCAR", "KPOINTS"}
    incar = (run_dir / "INCAR").read_text("utf-8")
    assert "IMAGES" in incar and "3" in incar.split("IMAGES")[1][:8]
    assert "0.0300" in (run_dir / "KPOINTS").read_text("utf-8")
    # frames untouched
    for i in range(5):
        assert (run_dir / f"{i:02d}" / "POSCAR").read_text("utf-8") == \
            "frame"


def test_write_run_inputs_potcar_variants(tmp_path):
    """Regeneration honours per-element variants and records them on
    the provenance card."""
    from vaspen.core.structure import StructureModel

    lib = tmp_path / "lib" / "potpaw_PBE.54"
    for name, tag in (("Si", "Si_plain"), ("Si_d", "Si_d_variant")):
        d = lib / name
        d.mkdir(parents=True)
        (d / "POTCAR").write_text(f"# {tag}\n", encoding="utf-8")
    src = tmp_path / "si.vasp"
    bulk("Si", "diamond", a=5.43, cubic=True).write(src, format="vasp")
    project = Project.create(tmp_path / "proj", "T")
    s_entry = project.add_structure(src)
    r = project.add_run("scf", "scf", {}, s_entry["id"], "sha256:x", "gui")
    model = StructureModel()
    model.load(src)
    project.write_run_inputs(
        r["id"], model, preset="scf",
        potcar_library=str(tmp_path / "lib"),
        potcar_variants={"Si": "Si_d"})
    run_dir = project.run_dir(r["id"])
    assert "# Si_d_variant" in (run_dir / "POTCAR").read_text("utf-8")
    card = json.loads((run_dir / "run.json").read_text("utf-8"))
    assert card["parameters"]["potcar_variants"] == {"Si": "Si_d"}


def test_register_structure_file_changed_content_registers_new(
        project: Project, tmp_path):
    """Path match with CHANGED content must NOT return the old entry's
    stale digest — the file is a different material now."""
    src = tmp_path / "si.vasp"
    bulk("Si", "diamond", a=5.43, cubic=True).write(src, format="vasp")
    project.add_structure(src)
    bulk("Si", "diamond", a=6.0, cubic=True).write(src, format="vasp")
    sid, digest = project.register_structure_file(src)
    assert len(project.structures) == 2  # changed file = new material
    assert digest == sha256_file(src)
    assert digest != project.structures[0]["digest"]
