"""Tests for the headless MCP server (vaspen.mcp_server).

Tools are exercised as plain functions through the module surface; one
end-to-end test speaks the real stdio protocol to a subprocess
(newline-delimited JSON-RPC) so the SDK wiring stays pinned without
depending on SDK test helpers. Structures are built in code — the test
suite never reads examples/ (CLAUDE.md §9).
"""

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from vaspen.core.file_io import FileIO
from vaspen.core.project import Project
from vaspen.mcp_server import server as mcp_server
from vaspen.mcp_server import tools as mcp_tools
from vaspen.mcp_server.session import ServerSession


@pytest.fixture(autouse=True)
def fresh_session(monkeypatch):
    """Every test starts with an empty session and no open project
    (both are process-global and would otherwise leak across tests)."""
    monkeypatch.setattr(mcp_tools, "SESSION", ServerSession())
    monkeypatch.setattr(mcp_tools, "PROJECT_REF", mcp_tools._ProjectRef())


def _open(atoms, tmp_path, name="struct.vasp"):
    path = tmp_path / name
    atoms.write(path, format="vasp")
    return mcp_server.open_structure(str(path))


def _si_diamond_cubic():
    """8-atom conventional diamond Si (a=5.43) — the KSPACING worked
    example (0.04 → 5×5×5)."""
    from ase.build import bulk
    return bulk("Si", "diamond", a=5.43, cubic=True)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def test_all_tools_registered_with_unique_names():
    names = [fn.__name__ for fn in mcp_server._TOOLS]
    assert len(names) == len(set(names))
    for expected in ("open_structure", "get_structure_info", "list_atoms",
                     "save_structure", "make_periodic", "make_supercell",
                     "set_fixed_atoms", "set_magmoms", "translate_atoms",
                     "rotate_atoms", "measure", "find_bonds",
                     "analyze_symmetry", "symmetrize_cell",
                     "list_slab_terminations", "cut_surface", "rebox_slab",
                     "suggest_band_path", "estimate_k_mesh",
                     "list_incar_presets", "generate_inputs", "neb_check",
                     "neb_setup", "render_preview", "run_python"):
        assert expected in names


def test_build_server_registers_tools_and_resources():
    srv = mcp_server.build_server()
    assert srv.name == "VASPen"


# ----------------------------------------------------------------------
# Session / structure tools
# ----------------------------------------------------------------------


def test_open_structure_returns_summary(tmp_path, si_bulk):
    result = _open(si_bulk, tmp_path)
    assert result["ok"] is True
    assert result["formula"] == "Si2"
    assert result["n_atoms"] == 2
    assert result["periodic"] is True
    assert "cell" in result and result["cell"]["alpha"] == pytest.approx(60.0)
    assert "No." in result["space_group"]
    assert result["source"].endswith("struct.vasp")


def test_open_structure_missing_file():
    with pytest.raises(ToolError, match="not found"):
        mcp_server.open_structure("Z:/no/such/file.vasp")


def test_tools_require_a_loaded_structure():
    with pytest.raises(ToolError, match="No structure loaded"):
        mcp_server.get_structure_info()
    with pytest.raises(ToolError, match="No structure loaded"):
        mcp_server.list_atoms()


def test_list_atoms_pagination_and_guards(tmp_path, si_bulk):
    _open(si_bulk, tmp_path)
    page = mcp_server.list_atoms(start=0, count=1)
    assert page["total"] == 2 and page["returned"] == 1
    assert page["atoms"][0]["index"] == 0
    last = mcp_server.list_atoms(start=1, count=1)
    assert last["atoms"][0]["index"] == 1
    frac = mcp_server.list_atoms(fractional=True)
    assert all(0.0 <= v <= 1.0 for v in frac["atoms"][0]["position"])
    with pytest.raises(ToolError, match="out of range|start"):
        mcp_server.list_atoms(start=5)


def test_list_atoms_fractional_requires_periodic(tmp_path, water_molecule):
    path = tmp_path / "h2o.xyz"
    water_molecule.write(path)
    mcp_server.open_structure(str(path))
    with pytest.raises(ToolError, match="[Ff]ractional"):
        mcp_server.list_atoms(fractional=True)


def test_save_structure_molecule_needs_wrap(tmp_path, water_molecule):
    path = tmp_path / "h2o.xyz"
    water_molecule.write(path)
    mcp_server.open_structure(str(path))
    target = tmp_path / "h2o.vasp"
    with pytest.raises(ToolError, match="wrap_padding"):
        mcp_server.save_structure(str(target))
    mcp_server.save_structure(str(target), wrap_padding=5.0)
    assert target.exists()
    info = mcp_server.get_structure_info()
    assert info["periodic"] is True
    # molecule saved to xyz needs no wrap at all
    mcp_server.save_structure(str(tmp_path / "back.xyz"))


def test_make_periodic_roundtrip(tmp_path, water_molecule, si_bulk):
    water_path = tmp_path / "h2o.xyz"
    water_molecule.write(water_path)
    mcp_server.open_structure(str(water_path))
    result = mcp_server.make_periodic(5.0)
    assert result["periodic"] is True
    assert result["n_atoms"] == 3
    with pytest.raises(ToolError, match="full-rank"):
        _open(si_bulk, tmp_path)
        mcp_server.make_periodic(5.0)


# ----------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------


def test_make_supercell(tmp_path, si_bulk):
    _open(si_bulk, tmp_path)
    result = mcp_server.make_supercell(2, 2, 2)
    assert result["n_atoms"] == 16
    assert result["cell"]["a"] == pytest.approx(si_bulk.cell.cellpar()[0] * 2)
    with pytest.raises(ToolError, match=">= 1"):
        mcp_server.make_supercell(0, 1, 1)


def test_make_supercell_requires_periodic(tmp_path, water_molecule):
    path = tmp_path / "h2o.xyz"
    water_molecule.write(path)
    mcp_server.open_structure(str(path))
    with pytest.raises(ToolError, match="[Pp]eriodic"):
        mcp_server.make_supercell(2, 2, 2)


def test_set_fixed_atoms_and_magmoms(tmp_path, si_bulk):
    _open(si_bulk, tmp_path)
    result = mcp_server.set_fixed_atoms([0], fixed=True)
    assert result["n_frozen_atoms"] == 1
    result = mcp_server.set_fixed_atoms([1], fixed=[True, False, True])
    assert result["n_frozen_atoms"] == 2
    result = mcp_server.set_magmoms([0], 2.0)
    assert result["has_magmoms"] is True
    with pytest.raises(ToolError, match="out of range"):
        mcp_server.set_fixed_atoms([99])


def test_translate_and_rotate(tmp_path, si_bulk, water_molecule):
    _open(si_bulk, tmp_path)
    moved = mcp_server.translate_atoms([0.25, 0.0, 0.0], indices=[0])
    assert moved["n_atoms"] == 2
    with pytest.raises(ToolError, match="out of range"):
        mcp_server.translate_atoms([0.1, 0, 0], indices=[99])

    path = tmp_path / "h2o.xyz"
    water_molecule.write(path)
    mcp_server.open_structure(str(path))
    before = np.array([a["position"] for a in
                       mcp_server.list_atoms()["atoms"]])
    mcp_server.rotate_atoms([0, 0, 0], [0, 0, 1], 90.0)
    after = np.array([a["position"] for a in
                      mcp_server.list_atoms()["atoms"]])
    assert not np.allclose(before, after)
    assert mcp_server.get_structure_info()["n_atoms"] == 3


def test_measure_known_geometry(tmp_path, water_molecule, benzene_molecule):
    path = tmp_path / "h2o.xyz"
    water_molecule.write(path)
    mcp_server.open_structure(str(path))
    d_oh = mcp_server.measure("distance", 0, 1)
    assert d_oh["unit"] == "Angstrom"
    assert d_oh["value"] == pytest.approx(water_molecule.get_distance(0, 1),
                                          abs=1e-6)
    a_hoh = mcp_server.measure("angle", 1, 0, 2)
    assert a_hoh["value"] == pytest.approx(water_molecule.get_angle(1, 0, 2),
                                           abs=1e-6)
    ring = tmp_path / "benzene.xyz"
    benzene_molecule.write(ring)
    mcp_server.open_structure(str(ring))
    dih = mcp_server.measure("dihedral", 0, 1, 2, 3)
    assert -180.0 <= dih["value"] <= 180.0
    with pytest.raises(ToolError, match="angle requires"):
        mcp_server.measure("angle", 0, 1)
    with pytest.raises(ToolError, match="kind must be"):
        mcp_server.measure("bond", 0, 1)
    with pytest.raises(ToolError, match="out of range"):
        mcp_server.measure("distance", 0, 99)


def test_find_bonds(tmp_path, si_bulk):
    _open(si_bulk, tmp_path)
    result = mcp_server.find_bonds()
    assert result["total"] > 0
    assert all(b["i"] < b["j"] for b in result["bonds"])


# ----------------------------------------------------------------------
# Symmetry / surfaces
# ----------------------------------------------------------------------


def test_analyze_symmetry(tmp_path, si_bulk, water_molecule):
    _open(si_bulk, tmp_path)
    sym = mcp_server.analyze_symmetry()
    assert sym["kind"] == "space"
    assert sym["number"] == 227  # Fd-3m
    path = tmp_path / "h2o.xyz"
    water_molecule.write(path)
    mcp_server.open_structure(str(path))
    sym = mcp_server.analyze_symmetry()
    assert sym["kind"] == "point"
    assert sym["symbol"] == "C2v"


def test_symmetrize_cell(tmp_path, si_bulk):
    _open(si_bulk, tmp_path)
    result = mcp_server.symmetrize_cell("primitive")
    assert result["ok"] is True
    assert result["n_atoms"] == 2
    assert "No. 227" in result["space_group"]
    with pytest.raises(ToolError, match="cell_type"):
        mcp_server.symmetrize_cell("bogus")


def test_symmetrize_disorder_guard(tmp_path, disordered_atoms):
    path = tmp_path / "alloy.cif"
    FileIO.write(str(path), disordered_atoms)
    mcp_server.open_structure(str(path))
    with pytest.raises(ToolError, match="partial occupancies"):
        mcp_server.symmetrize_cell("conventional")


def test_estimate_k_mesh_and_band_path(tmp_path):
    path = tmp_path / "si.vasp"
    _si_diamond_cubic().write(path, format="vasp")
    mcp_server.open_structure(str(path))
    mesh = mcp_server.estimate_k_mesh(0.04)
    assert mesh["mesh"] == [5, 5, 5]
    path_result = mcp_server.suggest_band_path()
    assert path_result["segments"]
    assert all(len(pt) == 3 for pt in
               path_result["special_points"].values())


def test_list_incar_presets():
    result = mcp_server.list_incar_presets()
    assert set(result["presets"]) >= {
        "scf", "opt", "band", "dos", "optical", "neb"}
    # the NEB preset leaves IMAGES blank on purpose (blocks generation
    # until filled — settled 2026-08-14); neb_setup fills it.
    assert result["presets"]["neb"]["tags"]["IMAGES"] == ""


# ----------------------------------------------------------------------
# VASP input generation
# ----------------------------------------------------------------------


def test_generate_inputs_without_library(tmp_path, si_bulk):
    _open(si_bulk, tmp_path)
    result = mcp_server.generate_inputs(str(tmp_path / "inputs"))
    files = result["files"]
    assert Path(files["INCAR"]).exists()
    assert Path(files["KPOINTS"]).exists()
    assert Path(files["POSCAR"]).exists()
    assert "POTCAR" not in files
    assert "POTCAR" in result["potcar_note"]
    assert "ENCUT" in result["INCAR"]


def test_generate_inputs_with_fake_library(tmp_path, si_bulk):
    lib = tmp_path / "potcar_lib" / "potpaw_PBE.54" / "Si"
    lib.mkdir(parents=True)
    (lib / "POTCAR").write_text("fake-si-potcar")
    _open(si_bulk, tmp_path)
    result = mcp_server.generate_inputs(
        str(tmp_path / "inputs"), potcar_library=str(tmp_path / "potcar_lib"))
    assert "POTCAR" in result["files"]
    assert "fake-si-potcar" in Path(result["files"]["POTCAR"]).read_text()


def test_generate_inputs_disorder_guard(tmp_path, disordered_atoms):
    path = tmp_path / "alloy.cif"
    FileIO.write(str(path), disordered_atoms)
    mcp_server.open_structure(str(path))
    with pytest.raises(ToolError, match="partial occupancies"):
        mcp_server.generate_inputs(str(tmp_path / "out"))


# ----------------------------------------------------------------------
# NEB
# ----------------------------------------------------------------------


def _write_pair(tmp_path, init, final):
    pi = tmp_path / "initial.vasp"
    pf = tmp_path / "final.vasp"
    init.write(pi, format="vasp")
    final.write(pf, format="vasp")
    return str(pi), str(pf)


def test_neb_check(tmp_path, vacancy_hop_pair, frozen_pass_block_pair):
    pi, pf = vacancy_hop_pair
    init_p, final_p = _write_pair(tmp_path, pi, pf)
    result = mcp_server.neb_check(init_p, final_p)
    assert result["n_atoms"] == 7
    assert result["distance"] == pytest.approx(3.615 / np.sqrt(2), abs=1e-3)
    assert result["suggested_n_images"] == 4
    assert "frozen_error" not in result

    fpi, fpf, bip, bfp = frozen_pass_block_pair
    init_p, final_p = _write_pair(tmp_path, fpi, fpf, )
    result = mcp_server.neb_check(init_p, final_p)
    assert result["n_frozen_init"] == 6
    assert "frozen_error" not in result
    init_p, final_p = _write_pair(tmp_path, bip, bfp)
    result = mcp_server.neb_check(init_p, final_p)
    assert "frozen_error" in result


def test_neb_setup_writes_standard_layout(tmp_path, frozen_pass_block_pair):
    fpi, fpf, _bi, _bf = frozen_pass_block_pair
    init_p, final_p = _write_pair(tmp_path, fpi, fpf)
    out = tmp_path / "neb"
    result = mcp_server.neb_setup(init_p, final_p, str(out), n_images=2)
    assert result["n_frames"] == 4
    for idx in range(4):
        assert (out / f"{idx:02d}" / "POSCAR").exists()
    incar = (out / "INCAR").read_text()
    assert "IMAGES" in incar and "2" in incar
    assert (out / "KPOINTS").exists()
    assert "POTCAR" in result["potcar_note"]
    # frozen atoms keep their initial position in every frame
    from vaspen.core.neb import constraints_to_fixed_flags
    from vaspen.core.file_io import FileIO
    frozen = constraints_to_fixed_flags(fpi)
    init_pos = fpi.get_positions()
    for idx in range(4):
        frame = FileIO.read(str(out / f"{idx:02d}" / "POSCAR"))
        frame_frozen = constraints_to_fixed_flags(frame)
        assert np.array_equal(frame_frozen, frozen)
        assert np.allclose(frame.get_positions()[frozen],
                           init_pos[frozen], atol=1e-6)


def test_neb_setup_default_images(tmp_path, vacancy_hop_pair):
    pi, pf = vacancy_hop_pair
    init_p, final_p = _write_pair(tmp_path, pi, pf)
    result = mcp_server.neb_setup(init_p, final_p, str(tmp_path / "neb"))
    assert result["n_images"] == 4  # ceil(2.5562 / 0.8)


def _swapped_order_pair(tmp_path):
    """Same-element pair whose file order pairs atoms the wrong way
    (file-order path ~6.4 A, optimal ~0.7 A) — fires the order
    mismatch diagnostic."""
    from ase import Atoms
    cell = np.eye(3) * 10.0
    init = Atoms("Si2", cell=cell, pbc=True,
                 positions=[[0, 0, 0], [0, 0, 5.0]])
    final = Atoms("Si2", cell=cell, pbc=True,
                  positions=[[0, 0, 4.5], [0, 0, 0.5]])
    return _write_pair(tmp_path, init, final)


def test_neb_setup_order_mismatch_gate(tmp_path):
    init_p, final_p = _swapped_order_pair(tmp_path)
    with pytest.raises(ToolError, match="[Oo]rder"):
        mcp_server.neb_setup(init_p, final_p, str(tmp_path / "neb"))
    result = mcp_server.neb_setup(init_p, final_p, str(tmp_path / "neb"),
                                  force=True)
    assert result["ok"] is True
    assert result["n_frames"] == result["n_images"] + 2


def test_neb_setup_method_guard(tmp_path, vacancy_hop_pair):
    pi, pf = vacancy_hop_pair
    init_p, final_p = _write_pair(tmp_path, pi, pf)
    with pytest.raises(ToolError, match="method"):
        mcp_server.neb_setup(init_p, final_p, str(tmp_path / "neb"),
                             method="spiral")


# ----------------------------------------------------------------------
# Visualization / escape hatch
# ----------------------------------------------------------------------


def test_render_preview_returns_png_for_all_views(tmp_path):
    path = tmp_path / "si.vasp"
    _si_diamond_cubic().write(path, format="vasp")
    mcp_server.open_structure(str(path))
    for view in ("auto", "x", "y", "z", "a", "b", "c"):
        image = mcp_server.render_preview(view=view, max_px=300)
        assert image.data[:4] == b"\x89PNG"
        content = image.to_image_content()
        assert content.type == "image"
        assert content.mime_type == "image/png"


def test_render_preview_molecule_and_guards(tmp_path, water_molecule):
    path = tmp_path / "h2o.xyz"
    water_molecule.write(path)
    mcp_server.open_structure(str(path))
    image = mcp_server.render_preview()  # auto → z for molecules
    assert image.data[:4] == b"\x89PNG"
    with pytest.raises(ToolError, match="[Pp]eriodic"):
        mcp_server.render_preview(view="a")
    with pytest.raises(ToolError, match="view must be"):
        mcp_server.render_preview(view="diagonal")


def test_run_python_evaluates_and_reports(tmp_path, si_bulk):
    _open(si_bulk, tmp_path)
    assert mcp_server.run_python("1 + 1")["result"] == 2
    assert mcp_server.run_python("model.n_atoms")["result"] == 2
    captured = mcp_server.run_python("print('hi'); model.n_atoms * 2")
    assert captured["stdout"] == "hi\n"
    assert captured["result"] == 4
    # mutations through the model are visible to the summary
    result = mcp_server.run_python("model.set_magmom([0], 2.0)\n"
                                   "model.any_magmom")
    assert result["result"] == True  # noqa: E712
    assert result["summary"]["has_magmoms"] is True
    # numpy types degrade to repr
    result = mcp_server.run_python("np.zeros(3)")
    assert isinstance(result["result"], str)


def test_run_python_error_mapping(tmp_path, si_bulk, monkeypatch):
    _open(si_bulk, tmp_path)
    with pytest.raises(ToolError, match="NameError"):
        mcp_server.run_python("undefined_name_xyz")
    with pytest.raises(ToolError, match="SyntaxError"):
        mcp_server.run_python("def broken(:")
    # model is None without a structure — the escape hatch surfaces
    # the raw exception type by design (the AI can self-correct).
    monkeypatch.setattr(mcp_tools, "SESSION", ServerSession())
    with pytest.raises(ToolError, match="AttributeError"):
        mcp_server.run_python("model.n_atoms")


def test_run_python_can_be_disabled(tmp_path, si_bulk):
    from vaspen.utils.config import AppConfig

    _open(si_bulk, tmp_path)
    AppConfig().mcp_allow_run_python = False  # isolated_config scope
    with pytest.raises(ToolError, match="disabled"):
        mcp_server.run_python("1")


# ----------------------------------------------------------------------
# End-to-end over real stdio (protocol wiring, not tool logic)
# ----------------------------------------------------------------------


def test_stdio_subprocess_protocol():
    repo_root = str(Path(__file__).resolve().parent.parent)
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen",
           "VASPEN_HEADLESS": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "vaspen.mcp_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8",
        cwd=repo_root, env=env,
    )
    err_queue: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [err_queue.put(line) for line in proc.stderr],
                     daemon=True).start()
    out_queue: queue.Queue = queue.Queue()
    threading.Thread(
        target=lambda: [out_queue.put(line) for line in proc.stdout],
        daemon=True).start()

    def send(obj):
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    def recv(timeout=60.0):
        line = out_queue.get(timeout=timeout)
        return json.loads(line)

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                         "clientInfo": {"name": "vaspen-tests", "version": "0"}}})
        init = recv()
        assert init["result"]["serverInfo"]["name"] == "VASPen"
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        listing = recv()
        tools = {t["name"] for t in listing["result"]["tools"]}
        assert "open_structure" in tools and "neb_setup" in tools
        assert len(tools) == len(mcp_server._TOOLS)

        send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
              "params": {"name": "list_incar_presets", "arguments": {}}})
        call = recv()
        assert call["result"].get("isError") in (False, None)
        text = call["result"]["content"][0]["text"]
        assert '"scf"' in text

        send({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
              "params": {"name": "get_structure_info", "arguments": {}}})
        failed = recv()
        assert failed["result"].get("isError") is True
        assert "No structure loaded" in failed["result"]["content"][0]["text"]

        send({"jsonrpc": "2.0", "id": 5, "method": "resources/list"})
        resources = recv()
        uris = {str(r["uri"]) for r in resources["result"]["resources"]}
        assert "vaspen://formats" in uris
        assert "vaspen://incar-presets" in uris
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_stdio_subprocess_image_content(tmp_path):
    """Over the wire, render_preview must deliver native image content
    (not base64-in-text) after open_structure."""
    repo_root = str(Path(__file__).resolve().parent.parent)
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen",
           "VASPEN_HEADLESS": "1"}
    si = tmp_path / "si.vasp"
    _si_diamond_cubic().write(si, format="vasp")
    proc = subprocess.Popen(
        [sys.executable, "-m", "vaspen.mcp_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
        cwd=repo_root, env=env,
    )

    def send(obj):
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    def recv(timeout=90.0):
        return json.loads(proc.stdout.readline())

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                         "clientInfo": {"name": "vaspen-tests", "version": "0"}}})
        recv()
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
              "params": {"name": "open_structure",
                         "arguments": {"path": str(si)}}})
        assert recv()["result"]["content"][0]["type"] == "text"
        send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
              "params": {"name": "render_preview",
                         "arguments": {"view": "z", "max_px": 300}}})
        content = recv()["result"]["content"][0]
        assert content["type"] == "image"
        assert content["mimeType"] == "image/png"
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


# ----------------------------------------------------------------------
# Calculation projects (§7.15)
# ----------------------------------------------------------------------


def test_project_lifecycle_roundtrip(tmp_path):
    project = Project.create(tmp_path / "proj", "Test")
    mcp_tools.PROJECT_REF.set(project)

    assert mcp_tools.get_project()["ok"] is True
    assert mcp_tools.get_project()["name"] == "Test"

    # structure added from the session (needs a source file)
    f = tmp_path / "si.vasp"
    _si_diamond_cubic().write(f, format="vasp")
    mcp_server.open_structure(str(f))
    r = mcp_server.add_structure_to_project()
    assert r["structure"]["id"] == "si"
    assert r["structure"]["formula"] == "Si8"

    # run created from the current structure
    run = mcp_server.create_run("scf bulk", preset="scf")
    assert run["run"]["id"] == "001-scf-bulk"
    run_dir = Path(run["dir"])
    assert (run_dir / "INCAR").exists()
    assert (run_dir / "KPOINTS").exists()
    assert (run_dir / "POSCAR").exists()
    assert (run_dir / "run.json").exists()
    card = json.loads((run_dir / "run.json").read_text("utf-8"))
    assert card["structure_digest"].startswith("sha256:")
    assert card["generated_by"]["via"] == "mcp"

    # manifest registered everything
    manifest = json.loads(
        (project.root / "project.json").read_text("utf-8"))
    assert [s["id"] for s in manifest["structures"]] == ["si"]
    assert manifest["runs"][0]["status"] == "inputs-ready"

    # status flow + run structure reload
    assert mcp_server.set_run_status("001-scf-bulk",
                                     "submitted")["run"]["status"] == "submitted"
    loaded = mcp_server.open_run_structure("001-scf-bulk")
    assert loaded["ok"] is True
    assert loaded["formula"] == "Si8"


def test_project_tools_require_open_project(tmp_path):
    with pytest.raises(ToolError, match="No project open"):
        mcp_server.get_project()
    with pytest.raises(ToolError, match="No project open"):
        mcp_server.set_run_status("001-x", "done")
    with pytest.raises(ToolError, match="not a VASPen project|Folder not found"):
        mcp_server.open_project(str(tmp_path / "empty"))


def test_get_run_and_update_run(tmp_path):
    project = Project.create(tmp_path / "proj", "T")
    mcp_tools.PROJECT_REF.set(project)
    f = tmp_path / "si.vasp"
    _si_diamond_cubic().write(f, format="vasp")
    mcp_server.open_structure(str(f))
    mcp_server.add_structure_to_project()
    run = mcp_server.create_run("scf", preset="scf")
    run_id = run["run"]["id"]

    detail = mcp_server.get_run(run_id)
    assert detail["run"]["type"] == "scf"
    assert "INCAR" in detail["files"]
    assert detail["has_outputs"] is False

    updated = mcp_server.update_run(run_id, preset="opt",
                                    kpoints_params={"k_spacing": 0.03})
    assert updated["parameters"]["preset"] == "opt"
    assert updated["parameters"]["kpoints_params"] == {"k_spacing": 0.03}
    assert "EDIFFG" in Path(updated["files"]["INCAR"]).read_text()

    with pytest.raises(ToolError, match="Unknown run"):
        mcp_server.get_run("999-nope")


def test_read_run_file(tmp_path):
    project = Project.create(tmp_path / "proj", "T")
    mcp_tools.PROJECT_REF.set(project)
    f = tmp_path / "si.vasp"
    _si_diamond_cubic().write(f, format="vasp")
    mcp_server.open_structure(str(f))
    mcp_server.add_structure_to_project()
    run = mcp_server.create_run("scf", preset="scf")
    run_id = run["run"]["id"]
    run_dir = Path(run["dir"])
    (run_dir / "OSZICAR").write_text("1 F= -.123\n2 F= -.456\n" * 20,
                                     encoding="utf-8")

    r = mcp_server.read_run_file(run_id, "INCAR")
    assert r["ok"] is True
    assert "ENCUT" in r["content"]
    r = mcp_server.read_run_file(run_id, "OSZICAR")
    assert "F=" in r["content"]
    # NEB-style subdirectory path
    r = mcp_server.read_run_file(run_id, "POSCAR")
    assert "Direct" in r["content"] or "Cartesian" in r["content"] or \
        r["content"]
    with pytest.raises(ToolError, match="inside the run directory"):
        mcp_server.read_run_file(run_id, "../../secrets.txt")
    with pytest.raises(ToolError, match="No file"):
        mcp_server.read_run_file(run_id, "CONTCAR")  # not run yet


def test_update_run_neb_aware(tmp_path):
    project = Project.create(tmp_path / "proj", "T")
    mcp_tools.PROJECT_REF.set(project)
    f = tmp_path / "si.vasp"
    _si_diamond_cubic().write(f, format="vasp")
    mcp_server.open_structure(str(f))
    mcp_server.add_structure_to_project()
    run = mcp_server.create_run("nebish", preset="neb",
                                incar_overrides={"IMAGES": "2"})
    run_id = run["run"]["id"]
    run_dir = Path(run["dir"])
    for i in range(4):  # 2 images + endpoints
        frame = run_dir / f"{i:02d}"
        frame.mkdir()
        (frame / "POSCAR").write_text("frame", encoding="utf-8")

    r = mcp_server.update_run(run_id,
                              kpoints_params={"k_spacing": 0.03})
    assert r["ok"] is True
    assert "POSCAR" not in r["files"]
    incar = (run_dir / "INCAR").read_text("utf-8")
    assert "IMAGES" in incar and "2" in incar.split("IMAGES")[1][:8]
    for i in range(4):
        assert (run_dir / f"{i:02d}" / "POSCAR").read_text("utf-8") == \
            "frame"
