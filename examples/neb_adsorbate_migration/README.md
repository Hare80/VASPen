# NEB Example: Adsorbate Migration on Carbon

A P–O–H functional group migrating between two surface sites on a
carbon framework (C16 P O H, 19 atoms; ~7.3 × 6.5 Å in-plane cell,
γ ≈ 117°, with vacuum along c). The oxygen atom moves 1.31 Å between
the endpoints.

- `initial/POSCAR` — the group bound at one site
- `final/POSCAR` — the group at the neighboring site

Structure data taken from a published NEB demonstration dataset.

## How to test in VASPen

1. Open `initial/POSCAR`.
2. Calculate → Generate All Input Files… → task **NEB**.
3. POSCAR tab → **Browse…** next to "Final structure" → `final/POSCAR`.
4. Click **Interpolate** (Linear or IDPP both work; the suggested
   image count is 2–3 for this path) — click the middle frames to
   preview the migration.
5. Generate as usual (`00/POSCAR`…`0N/POSCAR` layout).
