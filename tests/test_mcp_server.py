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
from vaspen.mcp_server import server as mcp_server
from vaspen.mcp_server.session import ServerSession


@pytest.fixture(autouse=True)
def fresh_session(monkeypatch):
    """Every test starts with an empty session (the module-level one
    is process-global and would otherwise leak structures across
    tests)."""
    monkeypatch.setattr(mcp_server, "SESSION", ServerSession())


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
                     "neb_setup"):
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
# End-to-end over real stdio (protocol wiring, not tool logic)
# ----------------------------------------------------------------------


def test_stdio_subprocess_protocol():
    repo_root = str(Path(__file__).resolve().parent.parent)
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
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
