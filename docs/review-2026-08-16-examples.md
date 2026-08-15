# Code Review 2026-08-16 (examples round) — Findings & Dispositions

Targeted review of the three commits since the last reviewed baseline
(3a23dc8, pushed through 9f51ebf): `b52e2e6` (examples data curation),
`9c85041` (test decoupling from examples/), `3d7127f` (NEB case reorg).

**Method:** 3 focused review agents (test-decoupling correctness,
examples data integrity, CLAUDE.md consistency), all findings
adversarially verified against the code before fixing. This was a
proportionate follow-up, not a full layered re-review — the old code
already went through two full rounds.

## MEDIUM (fixed)

| # | Finding | Fix |
|---|---------|-----|
| E1 | `example_model`/`ethane_model` fixtures passed the shared pair atoms straight into `StructureModel.load_atoms`, which aliases (`self._atoms = atoms`) and strips constraints in place (`set_constraint([])`) — a model mutation would silently corrupt the fixture pair, and `load_atoms(pi)` destroyed the fixture's `FixAtoms` before the test could write it out. Benign today only because every consumer copies. | The model fixtures and the frozen-flow tests pass `.copy()`; conftest comment documents the aliasing. |
| E2 | `examples/neb_au111_vacancy_hop/README.md` claimed only the hopping atom is free — actually 18 atoms carry `F F F` (frozen) and 17 carry `T T T` (free); only one of the free atoms hops (2.84 Å). | README corrected to the real flag counts. |
| E3 | `examples/Cu_bulk.vasp` (4-atom fcc conventional) survived the curation as an orphan: not in the README index, referenced nowhere, duplicating `Cu_bulk_primitive.vasp`'s demo role. | Deleted — the curated set is exactly the 14 indexed structures. |

## LOW (fixed)

- **T1**: one `FixAtoms` instance was shared between `pi` and `pf`
  (`Atoms.constraints` does not copy on assignment) — a future
  in-place mutation of one would re-freeze the other. Two separate
  instances now.
- **T2**: `test_symmetrize_primitive_matches_reference_file` compared
  the symmetrize output against ASE's own `cellpar_to_cell` — both
  sides from the same library, so the "reference" pinned nothing.
  The numeric reference cell (exact vectors of the fcc Cu primitive)
  is now hardcoded in the test, independent of ASE.
- **D1**: CLAUDE.md v0.2 roadmap said "single-click open" for the
  welcome page — contradicts the settled §7.10 contract (click
  selects; double-click/Enter opens). Corrected.
- **D2**: v0.3 still listed "[ ] Test suite" — marked done.
- **D3**: §3 project tree omitted conftest.py, 9 test files and the
  examples/ directory — completed.
- **D4**: examples/README.md naming convention had three silent
  deviations (benzene.xyz, graphene_3x3.xyz, MFI_zeolite.vasp) —
  documented exceptions (molecules / framework codes).
- **E4**: migration README called the cell "hexagonal" — γ is 117.1°,
  not 120°. Wording fixed.
- **E5**: ethane NEB POSCARs had bare `C  H` comment lines — headers
  now carry neutral descriptions like the other trees.
- **T3**: unused `tmp_path` parameter dropped from
  `test_symmetrize_bcc_supercell_primitive_vs_conventional`.

## Acknowledged (no change)

- **T4**: the deleted `test_ethane_example_files_regression` also lost
  the cross-check that the committed ethane example files still show
  the collision contrast. This is the deliberate cost of the §9
  decoupling policy (tests never read examples/) — the files were
  verified by one-off scripts during this round instead.
- **E6**: opening `PZT_disordered.cif` emits an ASE `UserWarning`
  (crystal-system vs space-group mismatch in the 2010 COD file) —
  inherent to the verbatim copy; parse result is correct (Zr0.65/Ti0.35
  occupancy on the B site). The CIF is the COD 2010 revision (header
  untouched, as required); the 2016 reload differs only in metadata.
- **T5**: `StructureModel.load_atoms` aliasing + in-place constraint
  stripping is pre-existing core behavior (not introduced this round);
  the fixture-side copies in E1 neutralize the exposure for the suite.

## Verified clean

- Zero-dependency invariant: `vaspen/`, `scripts/`, `VASPen.spec`,
  `pyproject.toml` contain no examples/ references; tests/ only the
  policy comment in conftest.py.
- All 14 top-level structure files + 6 NEB POSCARs parse via
  `FileIO.read`; both new NEB pairs interpolate (Linear + IDPP),
  Au111 frozen atoms never move, endpoints exact.
- Pinned values reproduce: vacancy-hop distance 2.55619101398937
  (rel 3.5e-16 vs the pin), IDPP collision thresholds, "6 atoms
  frozen" flow, written `F   F   F` rows.
- No third-party tool names anywhere in examples/ (headers and
  READMEs); PZT CIF keeps its full COD header verbatim.
- No fixture-name collisions, no leftover dead imports, all 3 NEB
  trees contain exactly initial/final + README.
- Full suite: 663 passed after the fixes.
