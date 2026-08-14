# NEB Example: Frozen Atoms (pass / block cases)

Based on the Cu vacancy-hop pair (7 atoms), with **Selective dynamics**
in the POSCARs. Only the hopping atom is free; the other six are
frozen (`F F F`) in the initial structure.

- `pass/` — the frozen atoms have identical positions in the initial
  and final structures → **interpolation succeeds**; every image keeps
  the frozen atoms at their positions and carries the `F F F` rows.
- `block/` — the final structure additionally moves one of the frozen
  atoms → **interpolation is blocked** with a message listing the
  conflicting atom (contradiction: a frozen atom cannot move during
  the NEB run).

## How to test in VASPen

1. Open `pass/initial/POSCAR` (the frozen flags load with the file).
2. Calculate → Generate All Input Files… → task **NEB** →
   Browse `pass/final/POSCAR`. The diagnostics show
   "6 atoms frozen — kept fixed in all images".
3. Interpolate (Linear or IDPP) → click frames in the list: only the
   hopping atom moves; the rest stay put. Generate → every
   `0N/POSCAR` contains a `Selective dynamics` block with 6 `F F F`
   rows.
4. Repeat with `block/initial/POSCAR` + `block/final/POSCAR` →
   interpolation is refused with an error listing the moved frozen
   atom.
