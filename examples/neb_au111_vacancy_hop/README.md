# NEB Example: Au(111) Surface Vacancy Hop

A 3×3 four-layer Au(111) slab with one surface vacancy (35 atoms). In
the final structure a neighboring surface atom hops into the vacancy
site (path length 2.84 Å).

- `initial/POSCAR` — vacancy at a hollow site
- `final/POSCAR` — a neighbor hopped into the vacancy

The slab atoms carry `F F F` selective-dynamics flags (18 atoms fully
frozen) and the hopping atom is free (`T T T`): interpolation keeps the
frozen atoms exactly in place while the free atom travels.

## How to test in VASPen

1. Open `initial/POSCAR`.
2. Calculate → Generate All Input Files… → task **NEB**.
3. POSCAR tab → **Browse…** next to "Final structure" → `final/POSCAR`.
4. Click **Interpolate** (Linear or IDPP both work) — click the middle
   frames to preview the hop; the frozen slab never moves.
5. Generate as usual (`00/POSCAR`…`0N/POSCAR` layout).
