# VASPen Examples

A small curated set of structure files for exploring VASPen's features.
File naming convention: `<Formula>_<structure>.<ext>`.

## Structure files

| File | Structure | Demonstrates |
|------|-----------|--------------|
| `Cu_bulk_primitive.vasp` | fcc Cu, primitive cell | Symmetry dialog — primitive ↔ conventional conversion |
| `Fe_bcc_2x2x2.vasp` | bcc Fe, 2×2×2 supercell | Spin-polarized INCAR (ISPIN / MAGMOM); supercell editing |
| `Mg_hcp.vasp` | hcp Mg | Hexagonal cell rendering |
| `C_diamond.vasp` | Diamond, conventional cell | Covalent insulator; band / DOS INCAR presets |
| `NaCl_rocksalt.vasp` | NaCl rocksalt | Two-element POTCAR; non-polar (100) cleave |
| `GaAs_zincblende.vasp` | GaAs zincblende | Semiconductor; line-mode k-path for band tasks |
| `ZnO_wurtzite.vasp` | ZnO wurtzite | Hexagonal k-mesh (automatic KPOINTS mode) |
| `TiO2_rutile.vasp` | TiO2 rutile | Tetragonal cell |
| `BaTiO3_perovskite.vasp` | BaTiO3 perovskite | Three-element POTCAR; polar (001) terminations |
| `MFI_zeolite.vasp` | Pure-silica MFI zeolite (288 atoms) | Large-cell rendering; auto bond detection |
| `Cu_111_slab.vasp` | Cu (111) slab with vacuum | Slab viewing |
| `graphene_3x3.xyz` | Graphene 3×3 sheet | 2D structure with partial periodicity |
| `benzene.xyz` | Benzene molecule | Molecule editing; saving to a periodic format triggers the wrap dialog |
| `PZT_disordered.cif` | Pb(Zr0.65Ti0.35)O3 perovskite | Partial site occupancy from a CIF; disorder handling |

## NEB examples

Each tree contains an `initial/POSCAR` and `final/POSCAR` pair (plus a
README) for the NEB tab of Generate All Input Files:

- `neb_vacancy_hop/` — vacancy hop in fcc Cu (linear interpolation)
- `neb_ethane_rotation/` — methyl-group rotation (IDPP vs linear)
- `neb_frozen/` — frozen-atom pass / block case pairs

## Provenance

Structure data comes from standard crystallographic references (public
structure databases); lattice parameters and coordinates are the
published values. File headers are short VASPen-style comments; the
PZT CIF keeps the original database file header.
