# NEB Example: Ethane Methyl Rotation (IDPP demo)

Ethane (C2H6) in a 10 Å cubic box: one methyl group is rotated by 120°
around the C–C axis (staggered → staggered, passing through the
eclipsed transition state).

- `initial/POSCAR` — staggered conformation
- `final/POSCAR` — one CH3 group rotated by 120°

This is the classic case where **linear interpolation fails**: the
hydrogens of the two methyl groups rotate into each other, and the
middle frames have unphysical H–H distances (~0.65 Å). **IDPP** spreads
the rotation evenly and keeps every interatomic distance physical
(≥ ~1.09 Å).

## How to test in VASPen

1. Open `initial/POSCAR`.
2. Calculate → Generate All Input Files… → task **NEB**.
3. POSCAR tab → **Browse…** next to "Final structure" → `final/POSCAR`.
4. Keep **Algorithm: Linear** and click **Interpolate** — click the
   middle frames (e.g. 03) in the list to preview them: the CH3
   hydrogens overlap.
5. Switch **Algorithm: IDPP** and interpolate again — the rotation is
   now spread evenly and no atoms collide. `IMAGES` is still filled
   into the INCAR tab automatically.
6. Generate as usual (`00/POSCAR`…`0N/POSCAR` layout).
