# NEB Example: Vacancy Hop in fcc Cu

A minimal NEB test case: a nearest-neighbor Cu atom hops into an
adjacent vacancy in a 2×2×2 fcc Cu primitive cell (7 atoms).

- `initial/POSCAR` — Cu with one vacancy
- `final/POSCAR` — the nearest neighbor moved into the vacancy site

Path length (3N displacement norm): **2.5562 Å** → suggested number of
intermediate images: **ceil(2.5562 / 0.8) = 4**.

## How to test in VASPen

1. Open `initial/POSCAR`.
2. Calculate → Generate All Input Files…
3. Select the **NEB** task. The POSCAR tab switches to NEB mode —
   the initial structure is the opened one.
4. Click **Browse…** next to "Final structure" and choose
   `final/POSCAR`. The path distance and the suggested image count
   (4) appear; adjust if you like.
5. Click **Interpolate** — image frames `00`…`04` are generated.
   Click any frame in the list to preview it in the 3D viewport.
   `IMAGES = 4` is filled into the INCAR tab automatically.
6. Pick an output directory and click **Generate**. You get
   `00/POSCAR`…`04/POSCAR` plus `INCAR`, `KPOINTS`, `POTCAR` in the
   output directory — ready for a VASP NEB run.

Expected result: frames are linear interpolations of the hop, the
lattice is identical in every frame, and the final frame equals
`final/POSCAR`.
