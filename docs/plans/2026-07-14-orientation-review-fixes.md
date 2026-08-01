# Senior review fixes: orientation.py (calibration) — detailed implementation plan

> Order: implement this plan FIRST, then `2026-07-14-mmlut-multimedia-strategy.md`.

## Context

Yesterday's work (commit `e1a4a80` + uncommitted changes) fixed the orient residual
bias, added radial-distortion fold detection with a staged k3/k2 fallback, normalized
the k1/k2/k3 design columns, typed the correspondence memoryviews, and hoisted
inner-loop slices. A senior review of that diff confirmed the core algebra and hot-path
changes are correct (memoryview dtypes are `int32`/`float64` matching
`cython.int`/`cython.double`; hoists are loop-invariant; the normalized-units scaling
of X columns / prior rows / `beta_raw` / `sigmabeta` is self-consistent; GUI `interf`
wiring is complete; `populate_runs(active_yaml=...)` exists).

Four findings remain to fix, **all in `src/openptv2/algorithms/orientation.py`**,
plus one new test. The user approved fixing Finding 1 even though it deviates from
liboptv C parity.

IMPORTANT for the implementer: this file compiles under Cython Pure Python mode.
After editing you MUST rebuild before running tests (see Verification), otherwise the
stale compiled extension is imported and your edits have no effect.

---

## Fix 1 — Glass-interface updates are wiped every iteration (the real bug)

**Where:** `src/openptv2/algorithms/orientation.py`, function `orient()` (starts line 603).

**Problem:** `safety_x/y/z` are captured ONCE, before the `while` iteration loop
(currently lines 673–675):

```python
    safety_x = cal.glass_par.vec_x
    safety_y = cal.glass_par.vec_y
    safety_z = cal.glass_par.vec_z
```

Inside the per-observation loop, the numeric glass-derivative blocks perturb
`cal.glass_par.vec_*` and then restore them with `cal.glass_par.vec_x = safety_x`
(etc.) — this restore appears **three times** (after the `al`, `be`, and `ga`
derivative blocks, lines ~795–797, ~807–809, ~819–821). But at the END of each
iteration, when `flags.interfflag` is set, the glass vector is updated:

```python
        if flags.interfflag:
            cal.glass_par.vec_x += e1[0] * nGl * beta_raw[16]
            ...
```

On the next iteration, the first derivative block's restore resets the glass vector
back to the ORIGINAL (pre-fit) values, silently discarding the update. Interface
fitting therefore never accumulates across iterations. (Inherited verbatim from
liboptv `orient.c`; it matters now because the uncommitted change in
`full_calibration` enables `interfflag` from the GUI via the `"interf"` flag.)

**Edit:** Move the three `safety_*` assignments from their current location
(lines 673–675, just before the `# Column scaling ...` comment block) to inside the
`while` loop body, immediately after the loop-reset block, i.e.:

```python
    while stopflag == 0 and itnum < NUM_ITER:
        itnum += 1

        X[:] = 0.0
        y[:] = 0.0
        P[:] = 1.0
        n = 0

        # Re-capture per iteration: the numeric glass-derivative blocks below
        # restore the glass vector to these values, so they must track the
        # interface updates applied at the end of the previous iteration.
        safety_x = cal.glass_par.vec_x
        safety_y = cal.glass_par.vec_y
        safety_z = cal.glass_par.vec_z
```

Delete the original three lines at 673–675. Do NOT touch `e1`, `e2`, `nGl` — they
stay computed once from the initial glass vector (the end-of-iteration update step
uses the same basis, so they must remain consistent).

**Do NOT change** the odd-looking `e1_z = 1 * cal.glass_par.vec_y - 2 * cal.glass_par.vec_y`
lines or the unused X column 18 (`ga` derivative) — deliberate liboptv parity.

---

## Fix 2 — Duplicate `r_max` computation

**Where:** same file, inside `orient()`, in the `if stopflag:` block (lines ~995–997):

```python
r_max = 0.5 * float(np.hypot(cpar.imx * cpar.pix_x, cpar.imy * cpar.pix_y))
```

This is identical to `r_max_norm` computed at line 684. Delete these lines and use
`r_max_norm` instead: `r_fold = radial_distortion_folds(cal.added_par, r_max_norm)`,
and replace `{r_max:.2f}` with `{r_max_norm:.2f}` in the two warning f-strings just
below (the "Refitting with ... disabled" warning and the "even with k1 alone" warning).

---

## Fix 3 — Dead import in `raw_orient`

**Where:** same file, line 533 inside `raw_orient()`:

```python
    from .trafo import pixel_to_metric, correct_brown_affin
```

`correct_brown_affin` is never called in `raw_orient` (it zeroes the distortion
parameters instead). Change to:

```python
    from .trafo import pixel_to_metric
```

---

## Fix 4 — Stale `full_calibration` docstring

**Where:** same file, `full_calibration()` docstring (lines ~1141–1143):

```
        flags: list of flag name strings to enable. Recognized:
            'cc', 'xh', 'yh', 'k1', 'k2', 'k3', 'p1', 'p2',
            'scale', 'shear'. If None, no flags enabled (raw-like).
```

Add `'interf'` (enables glass-interface fitting) to the recognized list, e.g.:

```
        flags: list of flag name strings to enable. Recognized:
            'cc', 'xh', 'yh', 'k1', 'k2', 'k3', 'p1', 'p2',
            'scale', 'shear', 'interf'. If None, no flags enabled (raw-like).
```

---

## New regression test for Fix 1

**Where:** append to `tests/unit/test_orientation.py`.

**Template:** copy the structure of the existing `test_orient()` (line 108 of that
file) — it already shows the exact fixture paths and the synthesize-observations
pattern. New test:

```python
def test_orient_interface_updates_accumulate():
    """interfflag fit must accumulate glass-vector updates across iterations.

    Regression: safety_x/y/z used to be captured once before the iteration
    loop, so the numeric-derivative restores wiped each iteration's interface
    update and the glass vector never moved.
    """
    fix = np.zeros((64, 3))
    pt_id = 0
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                fix[pt_id] = np.array([(ix * 10) - 60, iy * 5, iz * 5])
                pt_id += 1

    ori_file = "test_data/calibration/sym_cam1.tif.ori"
    add_file = "test_data/calibration/cam1.tif.addpar"

    cal = Calibration.from_file(ori_file, add_file)
    cpar = ControlPar.from_yaml("test_data/parameters.yaml")

    # Truth: the unmodified calibration. Synthesize exact observations from it.
    pix = [Target() for _ in range(64)]
    for i in range(64):
        xp, yp = img_coord(fix[i], cal, cpar.mm)
        x_pix, y_pix = metric_to_pixel(xp, yp, cpar)
        pix[i].x = x_pix
        pix[i].y = y_pix
        pix[i].pnr = i

    true_glass = np.array(
        [cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z]
    )

    # Start from a glass vector perturbed in the e1/e2 plane that the
    # interface fit spans (any small tilt off the true direction works).
    nGl = np.linalg.norm(true_glass)
    cal.glass_par.vec_x += 0.02 * nGl
    cal.glass_par.vec_y -= 0.01 * nGl
    start_err = np.linalg.norm(
        np.array([cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z])
        - true_glass
    )

    opar = OrientPar.from_file("test_data/parameters/orient.par")
    opar.interfflag = 1

    sigmabeta = np.zeros(20)
    resi = orient(cal, cpar, 64, fix, pix, opar, sigmabeta)
    assert resi is not None

    end_err = np.linalg.norm(
        np.array([cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z])
        - true_glass
    )
    # The fit must move the glass vector toward the truth. Before the fix the
    # updates were wiped every iteration, so end_err stayed ~= start_err.
    assert end_err < 0.5 * start_err
```

Notes for the implementer:
- All names used (`Calibration`, `ControlPar`, `Target`, `OrientPar`, `orient`,
  `img_coord`, `metric_to_pixel`, `np`) are already imported at the top of
  `tests/unit/test_orientation.py` — check and reuse; do not add duplicate imports.
- Confirm `OrientPar` exposes `interfflag` (it does — see
  `src/openptv2/gui/parameter_defaults.py` `"interfflag": 0` and
  `OrientPar` in `src/openptv2/algorithms/parameters.py`).
- Run the test BEFORE applying Fix 1 (after a rebuild) to confirm it fails
  (`end_err ≈ start_err`), then apply Fix 1, rebuild, and confirm it passes.
  If the exterior parameters absorb part of the tilt and the 0.5 factor is
  flaky, tightening the perturbation (e.g. 0.01/-0.005) or asserting
  `end_err < start_err * 0.8` is acceptable — but first try as written.

---

## Deliberately NOT changed (do not "fix" these)

- `correspondences.py` `_saturation_warned` module-global warn-once: silent on
  re-runs within one session, but keeps 1000-frame batch runs from spamming. Leave.
- liboptv parity oddities in `orient()`: `e1_z`/`e2_z` formulas using `vec_y` twice,
  unused Jacobian column 18.
- `notebooks/scripts/standalone_calibration.py` and
  `standalone_dumbbell_calibration.py`: modified in the worktree but unreadable
  (EACCES / "Function not implemented" from both git and direct reads — cloud-sync
  placeholder or file lock). Out of scope; the user must hydrate/unlock them.

## Verification (run in this order)

```bash
# 1. Rebuild Cython extensions (REQUIRED after editing algorithms/*.py)
uv run python setup.py build_ext --inplace

# 2. New + existing orientation tests
uv run pytest tests/unit/test_orientation.py tests/unit/test_orientation_wrappers.py tests/unit/test_orientation_coverage.py -v

# 3. Hot-path smoke test (correspondences untouched, but confirm)
uv run pytest tests/unit/test_track.py tests/unit/test_track3d.py tests/unit/test_correspondences.py -v --tb=short

# 4. Lint
uv run ruff check src/openptv2/algorithms/orientation.py tests/unit/test_orientation.py
```

All four steps must pass. If a parity test (`-m parity`, needs the C `optv` package)
fails on the glass-interface path, that is the expected divergence from the liboptv
bug approved by the user — report it, don't revert.
