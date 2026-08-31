# 16-bit images: make the grey scale explicit

**Status (2026-08-31):** problem characterised, three steps proposed. Steps 1–2
are small and self-contained; step 3 touches a Cython hot path.

---

## The problem

openptv2's detection thresholds — `targ_rec.gvthres`, `targ_rec.sumg_min`,
`detect_plate.gvth_*`, `detect_plate.sum_grey` — are all written on a 0–255
scale. The Illmenau cameras deliver **16-bit** images (`uint16`, measured range
2112–65520 on both the `Kalibrierung_*` and `Messung_*` sets). Something must
map one onto the other, and today **three different things do, inconsistently**:

| path | conversion | effect on a 2112–65520 image |
|---|---|---|
| `algorithms/segmentation.py:_load_image_array` | `skimage.util.img_as_ubyte` | **fixed** full-range map, 65535→255. Median lands at ~21 |
| `gui/calibration_gui.py` (all load sites) | `img_as_ubyte` | same fixed map — consistent with the above |
| `detect_plate.py:205` | `(x − p1) / (p99.5 − p1) × 255` | **per-image adaptive stretch**. Median lands near mid-grey |

So the same number means two different things depending on which detector reads
it. `gvth_1: 20` under the stretch is a sensible dark threshold; `gvthres: 20`
under the fixed map sits *at the image median*, because the fixed map compresses
a 2112–65520 image into 8–255 and leaves the median at 21.

### What it cost

On the Illmenau cameras 5–8 measurement frame, `targ_rec` with the configured
`gvthres: 3` found **550 / 416 / 1044 / 504** blobs per camera — mostly noise,
because 3 is far below the noise floor of the fixed-mapped image. That
saturated the epipolar candidate lists:

> `2 epipolar candidate lists hit MAXCAND=200; correspondence matching will be
> slow and unreliable`

and correspondences collapsed to a handful of quadruplets. The calibration was
perfect throughout — the same `.ori` yields 40 quadruplets on the plate. Hours
went into suspecting the geometry.

### Two aggravating details

* `targ_rec`'s compiled signature is `img: cython.uchar[:, ::1]`, so a `uint16`
  array is **rejected at the boundary** with `Buffer dtype mismatch`. Good — it
  fails loud. But the pure-Python fallback at `segmentation.py:156` does
  `np.ascontiguousarray(img, dtype=np.uint8)`, an unsafe cast that **wraps mod
  256**: 2112 → 64, 65520 → 240, 32768 → 0. Compiled and interpreted builds
  would disagree catastrophically on the same input.
* Nothing in `parameters_Run1.yaml` records the bit depth or the scaling rule,
  so a parameter set is not portable between an 8-bit and a 16-bit dataset, and
  a threshold cannot be interpreted without knowing which loader produced the
  image.

---

## Step 1 — one normalisation, named and documented

A single `openptv2.image_scaling` module owning every 16→8 conversion:

```python
to_uint8(img, mode="fixed", lo=None, hi=None, percentiles=(1.0, 99.5))
```

* `fixed` — full-range map (`img_as_ubyte` semantics). Absolute: the same grey
  value always maps to the same 8-bit value, so thresholds are comparable
  across frames and cameras. The right default for a sequence.
* `stretch` — per-image percentile stretch, today's `detect_plate` behaviour.
  Adaptive, so it copes with drifting illumination, but a threshold means
  something slightly different in every frame.
* `range` — explicit `lo`/`hi` in the source dtype's units. Absolute *and*
  tuned to the sensor's real range; the honest choice once you know it.

Every existing call site routes through it. `detect_plate` keeps `stretch` so
its behaviour is unchanged; `_load_image_array` and the GUI keep `fixed`. The
change here is that the choice becomes visible and named, not that any default
moves.

Also fixes the wrapping cast in the pure-Python `targ_rec` path so compiled and
interpreted builds cannot disagree.

## Step 2 — record the choice in the parameter file

```yaml
ptv:
  bit_depth: 16          # informational; what the camera writes
  grey_scaling: fixed    # fixed | stretch | range
  grey_range: [2112, 65520]   # only for mode 'range'
```

Read by `ControlPar`, defaulted so every existing parameter file keeps working
untouched. This is what makes a parameter set portable and a threshold
interpretable.

## Step 3 — let `targ_rec` accept `uint16`

Two options:

1. **A Python-level wrapper** (`targ_rec_scaled`) that normalises through step 1
   and calls the compiled kernel. Cheap, no Cython change, no risk to the hot
   path — but the thresholds stay on the 8-bit scale.
2. **A fused-type kernel** accepting `uchar` or `ushort`, with thresholds in the
   source dtype's units. Correct, and lets a 16-bit dataset use its full
   dynamic range, but it changes the meaning of every threshold and needs the
   parity tests run against it.

Start with (1) — it closes the crash and the compiled/interpreted divergence.
(2) is a separate piece of work, gated on the parity suite.

---

## Verification

* `to_uint8` round-trips and matches `img_as_ubyte` exactly in `fixed` mode, and
  reproduces the current `detect_plate` output byte-for-byte in `stretch` mode.
* Compiled and pure-Python `targ_rec` agree on a `uint16` input.
* The Illmenau cameras 5–8 plate frame still yields **40 quadruplets**.
* `uv run pytest tests/unit -q` stays green.

## Out of scope

The GUI's own display normalisation (`astype(np.uint8)` at
`calibration_gui.py:232`, `:1326`) is a *display* path, not a detection one. It
wraps for 16-bit input, which is a real display bug, but it is separate from the
threshold-semantics problem and is not touched here.
