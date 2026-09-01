## `eps0`: epipolar-band tolerance — semantics differ between openptv2/liboptv and 3dptv.exe

`eps0` is a correspondence-matching parameter (the `criteria.eps0` YAML field,
`VolumePar.eps0` in code) that controls how far off the epipolar line a
candidate point in another camera is still accepted as a possible match. Its
**name and legacy `.par` file slot are identical** between openptv2 and
3dptv.exe, but its **units and meaning are not** — copying a 3dptv `criteria.par`
`eps0` value verbatim into an openptv2 YAML silently changes matching
behavior. This was found and root-caused during the wp1 dataset investigation
(see `docs/plans/2026-08-27-track3d-beat-gt-plan.md`) and is a specific
instance of a general risk: **legacy `.par` parameters with the same name as
an openptv2 YAML field are not guaranteed to have the same definition.**

### What openptv2 does

`openptv2.algorithms.epi.find_candidate` and the production matching kernel
`openptv2.algorithms.correspondences._build_adjacency_for_pair` both use
`eps0` as a **flat millimeter tolerance**, applied identically to every
candidate regardless of particle size:

```python
tol_band_width = vpar.eps0  # mm, constant for the whole frame
```

This is liboptv's mainline algorithm (`find_candidate` in liboptv's classic
`correspondences.c`), correctly ported.

### What 3dptv.exe does

3dptv.exe does not run mainline liboptv's `find_candidate`. It runs a locally
patched variant, `find_candidate_plus` (`3dptv/src_c/epi.c`, patch by Beat,
April 2010), where `eps0` is a **dimensionless multiplier** scaled per
candidate by pixel size and the *source particle's own detected blob size*:

```c
// 3dptv/src_c/epi.c, find_candidate_plus
if (nx > ny) particle_size = nx; else particle_size = ny;   // px, source blob's own nx/ny
tol_band_width = eps0 * 0.5*(pix_x + pix_y) * particle_size;
if (tol_band_width < 0.06) tol_band_width = 0.06;            // mm, hard floor
```

So the actual mm tolerance 3dptv used varies per particle, and is **never
smaller than 0.06mm** regardless of how small `eps0` or the particle are.

### Converting a legacy `eps0` to openptv2's flat mm value

openptv2 has no per-particle dynamic scaling (see "Should we port the dynamic
formula?" below), so converting a legacy value means picking one
representative flat mm number that approximates what 3dptv's per-particle
formula would have produced across the dataset:

```
flat_eps0_mm = max(0.06, eps0_raw * 0.5*(pix_x + pix_y) * P)
```

where:
- `eps0_raw` — the value read straight from the legacy `criteria.par`'s
  `eps0` line (last field, dimensionless).
- `pix_x`, `pix_y` — mm/pixel, from `ptv.par` / YAML `ptv.pix_x`/`pix_y`.
- `P` — a **representative particle size in pixels** (`max(nx, ny)` of a
  typical detected blob in *this* dataset). Use the median of `max(nx, ny)`
  over the dataset's own detected targets (already available from a `RunStore`
  via `read_targets`), not a guess — particle size is dataset- and
  camera-lens-dependent.

**Before trusting the result, check whether the flat approximation is even
valid for this dataset** — compute what fraction of detected particles would
have their *true* per-particle tolerance dominated by the 0.06mm floor:

```python
tol = eps0_raw * 0.5 * (pix_x + pix_y) * particle_sizes  # array, one per target
floor_fraction = (tol < 0.06).mean()
```

- **`floor_fraction` ≈ 1.0** (small, uniform tracers — this was true for wp1,
  where every particle measured `max(nx,ny)` in the 2–8px range): the flat
  approximation is essentially exact. Just set `flat_eps0_mm = 0.06`; the
  nominal `eps0_raw` and `P` barely matter because the floor absorbs them.
- **`floor_fraction` well below 1.0** (larger or size-varied particles —
  dumbbells, big tracers, mixed populations): the flat approximation is
  **lossy**. A single flat value cannot reproduce a formula that scales
  per-particle; large particles get an unfairly tight window (if `P` was
  picked from the smaller population) or small particles get an unfairly
  loose one (if `P` was picked from the larger population) — you have to pick
  which class of match error you're willing to accept. This is the case where
  porting the real per-particle formula (see below) stops being optional
  polish and starts being the correct fix.

### Should openptv2 port the dynamic per-particle formula?

Not done yet — tracked as a follow-up
(`docs/plans/2026-08-27-eps0-dynamic-band-study-plan.md`). Short version:

- **Why the flat approximation was acceptable for wp1**: this dataset's
  particles are small and size-uniform enough that `floor_fraction = 1.0` —
  the dynamic formula and a flat `0.06mm` are numerically the same thing here.
- **Why porting the real formula anyway is worth studying**: the inputs
  (`nx`, `ny` of the source particle, `pix_x`/`pix_y`) are already threaded
  through both `find_candidate` and `_build_adjacency_for_pair` — this isn't
  a new API, and it removes the entire class of "guess a flat mm value and
  hope" bugs this doc exists to warn about, for every *other* legacy 3dptv
  dataset with larger or mixed particle sizes where `floor_fraction < 1.0`.
- **Why it's not a trivial drop-in**: it changes matching behavior for every
  existing openptv2 dataset unless gated behind an explicit opt-in mode
  (liboptv-parity flat behavior must stay the default), it adds a
  multiply+branch to the innermost O(N₁×N₂) matching loop in a compiled
  Cython kernel (needs a perf check), and it needs a rebuild plus a run of
  the existing correspondence/parity test suite before it can be trusted.

### Practical recommendation

1. **Never copy a legacy `criteria.par` `eps0` value into an openptv2 YAML
   verbatim.** Convert it using the formula above.
2. **Always compute `floor_fraction` for the target dataset first.** If it's
   ≈1.0, use `flat_eps0_mm = 0.06` and stop — no further tuning needed. If
   it's well below 1.0, treat the flat value as an approximation, validate
   its effect (candidate/point counts vs. any available ground truth) before
   trusting it, and consider requesting the dynamic-scaling mode once it
   exists.
3. **When in doubt, check the effect empirically, not just the formula**:
   rerun the correspondence stage and compare per-frame point counts against
   a trusted reference (ground truth, or a known-good prior run) — a flat
   `eps0` guessed too high inflates false 3D points (more candidates pass the
   epipolar-distance check); guessed too low starves recall (real matches
   fall outside the band). See `wp1_10_images/scripts/gate_sweep_gt_cloud.py`
   and `classify_by_level.py` in the wp1 case study for the pattern.
