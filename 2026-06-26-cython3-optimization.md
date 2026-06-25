# Cython 3 Optimization Plan — `algorithms/`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Python-level overhead in the compiled `algorithms/` Cython 3 Pure Python modules, guided by Cython annotation scores extracted from `algorithms/*.html`.

**Architecture:** Seven targeted improvements in priority order: replace Python `math.*` calls with C `libc.math` cimports (11 modules, 168 calls), add `@cython.inline` to scalar helpers, type untyped parameters in the three hottest `track_kernels` functions, replace the `segmentation.peak_fit` BFS deque with a typed array queue, and replace `lsqadj.matinv`'s handwritten Gauss-Jordan loop with `np.linalg.inv`.

**Tech Stack:** Python 3.11+, Cython 3.2.5, NumPy 2.x, uv, pytest, setuptools build backend

---

## Context — scores from `cython -a` annotation HTML

| Module | Hot lines (≥5) | Max score | Avg score |
|--------|---------------|-----------|-----------|
| track_kernels | 452 | 1267 | 42.0 |
| segmentation | 156 | 360 | 27.5 |
| ray_tracing | 73 | 323 | 24.9 |
| trafo | 105 | 241 | 35.9 |
| multimed | 161 | 236 | 30.2 |
| epi | 111 | 232 | 31.1 |
| imgcoord | 178 | 211 | 23.6 |
| track | 489 | 193 | 23.6 |
| orientation | 423 | 131 | 23.1 |
| vec_utils | 108 | 119 | 38.2 |
| lsqadj | 29 | 109 | 38.7 |

Score = number of Python API calls Cython generates for that line. Lower is faster in compiled mode.

---

## File map

Every change is a modification of an existing `algorithms/*.py` file. No new files needed.

| File | Change |
|------|--------|
| `algorithms/lsqadj.py` | Replace `matinv` triple loop with `np.linalg.inv` |
| `algorithms/vec_utils.py` | `math.sqrt` → `c_sqrt`; add `@cython.inline` to all scalar helpers |
| `algorithms/epi.py` | `math.sqrt` → `c_sqrt` |
| `algorithms/sortgrid.py` | `math.sqrt` → `c_sqrt` |
| `algorithms/track3d.py` | `math.sqrt` → `c_sqrt` |
| `algorithms/ray_tracing.py` | `math.sqrt` → `c_sqrt` (16 calls) |
| `algorithms/trafo.py` | `math.*` → `c_*` (19 calls: sqrt×9, sin×5, cos×5) |
| `algorithms/multimed.py` | `math.*` → `c_*` (15 calls: sqrt×8, tan×3, asin×2, atan×1) |
| `algorithms/orientation.py` | `math.sqrt` → `c_sqrt` (10 calls) |
| `algorithms/imgcoord.py` | `math.*` → `c_*` (35 calls: sqrt×10, tan×9, asin×6, sin×5, ...) |
| `algorithms/track.py` | `math.*` → `c_*` (13 calls: sqrt×10, sin×1, cos×1, acos×1) |
| `algorithms/track_kernels.py` | `math.*` → `c_*` (55 calls); type 8 untyped params in 3 hot functions |
| `algorithms/segmentation.py` | Replace `deque` BFS with typed array queue in `peak_fit` |

---

## Task 1: `lsqadj.matinv` — triple loop → `np.linalg.inv`

**Files:** Modify `algorithms/lsqadj.py`, Test `algorithms/tests/test_lsqadj.py`

The handwritten Gauss-Jordan loop (O(n³) Python iterations) reimplements LAPACK that NumPy already wraps.

- [ ] **Step 1: Run existing tests — establish baseline green**

```bash
uv run pytest algorithms/tests/test_lsqadj.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 2: Replace `matinv` body**

In `algorithms/lsqadj.py`, replace the entire body of `matinv` (lines 65–115 approximately). Remove the `@cython.boundscheck(False)` and `@cython.wraparound(False)` decorators too — they only matter for C loops.

**Before:**
```python
@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def matinv(a: np.ndarray, n: int) -> np.ndarray:
    """Invert a square matrix via Gauss-Jordan elimination.
    ...
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, n)[:n, :n].copy()
    n_large: cython.int = n  # in our simplified API, n_large == n
    n_int: cython.int = n

    ipiv: cython.Py_ssize_t
    irow: cython.Py_ssize_t
    icol: cython.Py_ssize_t
    pivot: cython.double
    npivot: cython.double

    for ipiv in range(n_int):
        pivot = 1.0 / a[ipiv, ipiv]
        npivot = -pivot

        # Update off-pivot elements
        for irow in range(n_int):
            for icol in range(n_int):
                if irow != ipiv and icol != ipiv:
                    a[irow, icol] -= a[ipiv, icol] * a[irow, ipiv] * pivot

        # Scale pivot row (excluding pivot element)
        for icol in range(n_int):
            if ipiv != icol:
                a[ipiv, icol] *= npivot

        # Scale pivot column (excluding pivot element)
        for irow in range(n_int):
            if ipiv != irow:
                a[irow, ipiv] *= pivot

        # Set pivot element
        a[ipiv, ipiv] = pivot

    return a
```

**After:**
```python
@cython.ccall
def matinv(a: np.ndarray, n: int) -> np.ndarray:
    """Invert a square matrix.

    Args:
        a: matrix of shape (n, n) or (n_large, n_large).
        n: size of the sub-matrix to invert.

    Returns:
        Inverse of a[:n, :n] as ndarray of shape (n, n).

    Raises:
        np.linalg.LinAlgError: if a is singular.
    """
    return np.linalg.inv(np.asarray(a, dtype=np.float64).reshape(-1, n)[:n, :n])
```

- [ ] **Step 3: Run tests — verify same results**

```bash
uv run pytest algorithms/tests/test_lsqadj.py -v
```

Expected: all 4 tests pass. If `test_matinv` compares against a known-singular matrix expecting `ZeroDivisionError`, update to expect `np.linalg.LinAlgError` instead.

- [ ] **Step 4: Run broader test suite to catch orientation/calibration regressions**

`matinv` is called by `orientation.py`. Run:

```bash
uv run pytest algorithms/tests/test_orientation.py algorithms/tests/test_calibration.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add algorithms/lsqadj.py
git commit -m "perf: replace matinv Gauss-Jordan loop with np.linalg.inv"
```

---

## Task 2: `math.*` → `cython.cimports.libc.math` — vec_utils (pilot)

**Files:** Modify `algorithms/vec_utils.py`, Test `algorithms/tests/test_vec_utils.py`

Use `vec_utils` as the pilot module to validate the import pattern before applying it to 10 more modules.

- [ ] **Step 1: Run vec_utils tests — baseline green**

```bash
uv run pytest algorithms/tests/test_vec_utils.py -v
```

Expected: all pass.

- [ ] **Step 2: Add libc.math cimport guard**

In `algorithms/vec_utils.py`, replace the top-of-file `import math` block:

**Before:**
```python
import math
import cython
import numpy as np
```

**After:**
```python
import cython
import numpy as np

if cython.compiled:
    from cython.cimports.libc.math import sqrt as c_sqrt, isnan as c_isnan
else:
    from math import sqrt as c_sqrt, isnan as c_isnan
```

- [ ] **Step 3: Replace `math.sqrt` and `math.isnan` calls**

In `algorithms/vec_utils.py`:

Line ~24: `return math.isnan(x)` → `return c_isnan(x)`
Line ~130: `return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)` → `return c_sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)`
Line ~148: `return math.sqrt(dx * dx + dy * dy + dz * dz)` → `return c_sqrt(dx * dx + dy * dy + dz * dz)`

- [ ] **Step 4: Run tests**

```bash
uv run pytest algorithms/tests/test_vec_utils.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add algorithms/vec_utils.py
git commit -m "perf: replace math.sqrt/isnan with libc.math cimports in vec_utils"
```

---

## Task 3: `math.sqrt` → `c_sqrt` — epi, sortgrid, track3d

**Files:** Modify `algorithms/epi.py`, `algorithms/sortgrid.py`, `algorithms/track3d.py`

Each has exactly 1 `math.sqrt` call. Mechanical, low-risk.

- [ ] **Step 1: Run baseline tests**

```bash
uv run pytest algorithms/tests/test_epi.py algorithms/tests/test_sortgrid.py algorithms/tests/test_track3d.py -v
```

Expected: all pass.

- [ ] **Step 2: Patch `epi.py`**

In `algorithms/epi.py`, replace `import math` block at the top:

**Before:**
```python
import math
import cython
```

**After:**
```python
import cython

if cython.compiled:
    from cython.cimports.libc.math import sqrt as c_sqrt
else:
    from math import sqrt as c_sqrt
```

Find line with `math.sqrt` (line ~279): `abs((crd[j].y - m * crd[j].x - b) / math.sqrt(m * m + 1))` → replace `math.sqrt` with `c_sqrt`.

- [ ] **Step 3: Patch `sortgrid.py`**

In `algorithms/sortgrid.py`, add after `import cython`:

```python
if cython.compiled:
    from cython.cimports.libc.math import sqrt as c_sqrt
else:
    from math import sqrt as c_sqrt
```

Remove `import math` if it was the only use. Find line ~42: `d = math.sqrt((x - p.x)**2 + (y - p.y)**2)` → `d = c_sqrt((x - p.x)**2 + (y - p.y)**2)`

- [ ] **Step 4: Patch `track3d.py`**

In `algorithms/track3d.py`, the `import math` is inside a block. Find it and add the guard pattern. Line ~34: `d = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)` → `d = c_sqrt(ddx * ddx + ddy * ddy + ddz * ddz)`

If `import math` is inside a try/HAS_NUMBA guard, put the `c_sqrt` import at module level:

```python
if cython.compiled:
    from cython.cimports.libc.math import sqrt as c_sqrt
else:
    from math import sqrt as c_sqrt
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest algorithms/tests/test_epi.py algorithms/tests/test_sortgrid.py algorithms/tests/test_track3d.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add algorithms/epi.py algorithms/sortgrid.py algorithms/track3d.py
git commit -m "perf: replace math.sqrt with libc.math cimports in epi, sortgrid, track3d"
```

---

## Task 4: `math.*` → `c_*` — ray_tracing and trafo

**Files:** Modify `algorithms/ray_tracing.py` (16 calls: sqrt×16), `algorithms/trafo.py` (19 calls: sqrt×9, sin×5, cos×5)

- [ ] **Step 1: Run baseline tests**

```bash
uv run pytest algorithms/tests/test_ray_tracing.py algorithms/tests/test_trafo.py -v
```

Expected: all pass.

- [ ] **Step 2: Patch `ray_tracing.py`**

Replace `import math` at top:

```python
import cython
import numpy as np

if cython.compiled:
    from cython.cimports.libc.math import sqrt as c_sqrt
else:
    from math import sqrt as c_sqrt
```

Then replace all 16 occurrences of `math.sqrt(` with `c_sqrt(` throughout the file. Use:

```bash
grep -n 'math\.sqrt' algorithms/ray_tracing.py
```

to get exact line numbers, then edit each.

- [ ] **Step 3: Patch `trafo.py`**

Replace `import math` at top:

```python
import cython
import numpy as np

if cython.compiled:
    from cython.cimports.libc.math import (
        sqrt as c_sqrt, sin as c_sin, cos as c_cos,
    )
else:
    from math import sqrt as c_sqrt, sin as c_sin, cos as c_cos
```

Replace all `math.sqrt(` → `c_sqrt(`, `math.sin(` → `c_sin(`, `math.cos(` → `c_cos(` throughout `trafo.py`.

Verify count before and after:

```bash
grep -c 'math\.' algorithms/trafo.py  # should be 0 after patch
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest algorithms/tests/test_ray_tracing.py algorithms/tests/test_trafo.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add algorithms/ray_tracing.py algorithms/trafo.py
git commit -m "perf: replace math.* with libc.math cimports in ray_tracing and trafo"
```

---

## Task 5: `math.*` → `c_*` — multimed and orientation

**Files:** Modify `algorithms/multimed.py` (15 calls: sqrt×8, tan×3, asin×2, atan×1), `algorithms/orientation.py` (10 calls: sqrt×10)

- [ ] **Step 1: Run baseline tests**

```bash
uv run pytest algorithms/tests/test_multimed.py algorithms/tests/test_orientation.py -v
```

Expected: all pass.

- [ ] **Step 2: Patch `multimed.py`**

Replace `import math` at top:

```python
import cython
import numpy as np

if cython.compiled:
    from cython.cimports.libc.math import (
        sqrt as c_sqrt, tan as c_tan, asin as c_asin, atan as c_atan,
    )
else:
    from math import sqrt as c_sqrt, tan as c_tan, asin as c_asin, atan as c_atan
```

Replace all `math.sqrt(` → `c_sqrt(`, `math.tan(` → `c_tan(`, `math.asin(` → `c_asin(`, `math.atan(` → `c_atan(`.

Verify:
```bash
grep -c 'math\.' algorithms/multimed.py  # should be 0
```

- [ ] **Step 3: Patch `orientation.py`**

Replace `import math` at top:

```python
if cython.compiled:
    from cython.cimports.libc.math import sqrt as c_sqrt
else:
    from math import sqrt as c_sqrt
```

Replace all 10 `math.sqrt(` → `c_sqrt(`.

Verify:
```bash
grep -c 'math\.sqrt' algorithms/orientation.py  # should be 0
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest algorithms/tests/test_multimed.py algorithms/tests/test_orientation.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add algorithms/multimed.py algorithms/orientation.py
git commit -m "perf: replace math.* with libc.math cimports in multimed and orientation"
```

---

## Task 6: `math.*` → `c_*` — imgcoord (35 calls)

**Files:** Modify `algorithms/imgcoord.py`

Highest call count of any single module (35). Functions: sqrt×10, tan×9, asin×6, sin×5, cos×3, atan2×1, others×1.

- [ ] **Step 1: Run baseline tests**

```bash
uv run pytest algorithms/tests/test_imgcoord.py algorithms/tests/test_validation_imgcoord.py -v
```

Expected: all pass.

- [ ] **Step 2: Audit exact math functions used**

```bash
grep -oP 'math\.\w+' algorithms/imgcoord.py | sort | uniq -c | sort -rn
```

This gives the exact list of functions to cimport. Add the guard with exactly those functions:

```python
if cython.compiled:
    from cython.cimports.libc.math import (
        sqrt as c_sqrt, tan as c_tan, asin as c_asin,
        sin as c_sin, cos as c_cos, atan2 as c_atan2,
        # add others found by the grep above
    )
else:
    from math import (
        sqrt as c_sqrt, tan as c_tan, asin as c_asin,
        sin as c_sin, cos as c_cos, atan2 as c_atan2,
    )
```

Remove `import math`.

- [ ] **Step 3: Replace all `math.*` calls**

```bash
# Verify nothing remains:
grep -n 'math\.' algorithms/imgcoord.py
```

Expected: zero matches (or only comments/strings if any).

- [ ] **Step 4: Run tests**

```bash
uv run pytest algorithms/tests/test_imgcoord.py algorithms/tests/test_validation_imgcoord.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add algorithms/imgcoord.py
git commit -m "perf: replace math.* with libc.math cimports in imgcoord (35 calls)"
```

---

## Task 7: `math.*` → `c_*` — track and track_kernels (68 total calls)

**Files:** Modify `algorithms/track.py` (13 calls: sqrt×10, sin/cos/acos×1 each), `algorithms/track_kernels.py` (55 calls: sqrt×41, sin×4, tan×3, cos×3, acos×2, atan×1, atan2×1)

These are the hottest files in the codebase (track_kernels max score 1267, track max score 193).

- [ ] **Step 1: Run baseline tests**

```bash
uv run pytest algorithms/tests/test_track.py algorithms/tests/test_batch_fast.py -v
```

Expected: all pass (some may be slow — that's OK).

- [ ] **Step 2: Audit exact functions in each module**

```bash
grep -oP 'math\.\w+' algorithms/track.py | sort | uniq -c | sort -rn
grep -oP 'math\.\w+' algorithms/track_kernels.py | sort | uniq -c | sort -rn
```

- [ ] **Step 3: Patch `track.py`**

After existing `import cython` / `import numpy as np`, add:

```python
if cython.compiled:
    from cython.cimports.libc.math import (
        sqrt as c_sqrt, sin as c_sin, cos as c_cos, acos as c_acos,
    )
else:
    from math import sqrt as c_sqrt, sin as c_sin, cos as c_cos, acos as c_acos
```

Remove `import math`. Replace all `math.sqrt(` → `c_sqrt(`, `math.sin(` → `c_sin(`, `math.cos(` → `c_cos(`, `math.acos(` → `c_acos(`.

- [ ] **Step 4: Patch `track_kernels.py`**

Same pattern with the full function set (sqrt, sin, cos, tan, acos, atan, atan2):

```python
if cython.compiled:
    from cython.cimports.libc.math import (
        sqrt as c_sqrt, sin as c_sin, cos as c_cos,
        tan as c_tan, acos as c_acos, atan as c_atan, atan2 as c_atan2,
    )
else:
    from math import (
        sqrt as c_sqrt, sin as c_sin, cos as c_cos,
        tan as c_tan, acos as c_acos, atan as c_atan, atan2 as c_atan2,
    )
```

Replace all 55 `math.*` calls. Verify:

```bash
grep -c 'math\.' algorithms/track_kernels.py  # should be 0
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest algorithms/tests/test_track.py algorithms/tests/test_batch_fast.py algorithms/tests/test_synthetic_tracking.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add algorithms/track.py algorithms/track_kernels.py
git commit -m "perf: replace math.* with libc.math cimports in track and track_kernels (68 calls)"
```

---

## Task 8: `@cython.inline` on `vec_utils` scalar helpers

**Files:** Modify `algorithms/vec_utils.py`

All 14 `@cython.ccall` functions in `vec_utils` do scalar/3-vector math. In compiled mode, adding `@cython.inline` eliminates the function call overhead entirely for call sites within the same module. For cross-module callers (multimed, orientation, lsqadj call `vec_cross`, `unit_vector`, `vec_norm`), inlining only helps same-module calls — but the reduction in function prologue overhead helps all callers.

- [ ] **Step 1: Add `@cython.inline` to all scalar ccall functions in vec_utils**

For every function in `algorithms/vec_utils.py` that has `@cython.ccall` and does pure scalar math (not file I/O, not Python object construction), add `@cython.inline` immediately after `@cython.ccall`:

Functions to add `@cython.inline` to:
- `is_empty` (L22)
- `vec_norm` (L121)
- `vec_diff_norm` (L134)
- `vec_dot` (L152)
- `vec_cmp` (L182)
- `vec_approx_cmp` (L195)

Do NOT add to functions that return `np.ndarray` (allocation makes inlining less useful and may cause issues): `vec_init`, `vec_set`, `vec_copy`, `vec_subt`, `vec_add`, `vec_scalar_mul`, `vec_cross`, `unit_vector`.

Pattern:
```python
@cython.ccall
@cython.inline  # ← add this line
def vec_norm(vec: cython.double[:]) -> cython.double:
    ...
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest algorithms/tests/test_vec_utils.py -v
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add algorithms/vec_utils.py
git commit -m "perf: add @cython.inline to vec_utils scalar helpers"
```

---

## Task 9: Type untyped parameters in `track_kernels` hot functions

**Files:** Modify `algorithms/track_kernels.py`

Three functions have parameters that pass Python sequences (tuples/lists) without type annotations. In compiled mode every element access unpacks a Python object. Score impact: `trackcorr_loop_fast` (1267), `trackback_loop_fast` (1006), `sorted_candidates_fast` (421).

- [ ] **Step 1: Run baseline tests**

```bash
uv run pytest algorithms/tests/test_track.py algorithms/tests/test_batch_fast.py -v
```

Expected: all pass.

- [ ] **Step 2: Audit callers to confirm array types passed**

In `algorithms/track.py`, find where `trackcorr_loop_fast` and `trackback_loop_fast` are called and check what types are passed for the currently untyped params (`targ_x_1`, `targ_y_1`, `targ_tnr_2`, etc.):

```bash
grep -n 'trackcorr_loop_fast\|trackback_loop_fast\|sorted_candidates_fast' algorithms/track.py | head -20
```

The callers in `track.py` use `_targ_x_1, _targ_y_1` etc. from `fb.targ_x` which are numpy `float64` arrays, and `fb.targ_tnr` which are numpy `int32` arrays.

- [ ] **Step 3: Add type annotations to `trackcorr_loop_fast` untyped params**

In `algorithms/track_kernels.py`, find `trackcorr_loop_fast` (~L1148). The current signature has:

```python
def trackcorr_loop_fast(
    ...
    targ_x_1, targ_y_1,           # ← untyped
    ...
    targ_x_2, targ_y_2, targ_tnr_2, ...   # ← untyped
    targ_x_3, targ_y_3, targ_tnr_3, ...   # ← untyped
```

Change untyped parameters to typed memoryviews:

```python
def trackcorr_loop_fast(
    ...
    targ_x_1: cython.double[:],   # was: targ_x_1
    targ_y_1: cython.double[:],   # was: targ_y_1
    ...
    targ_x_2: cython.double[:],   # was: targ_x_2
    targ_y_2: cython.double[:],   # was: targ_y_2
    targ_tnr_2: cython.int[:],    # was: targ_tnr_2
    ...
    targ_x_3: cython.double[:],   # was: targ_x_3
    targ_y_3: cython.double[:],   # was: targ_y_3
    targ_tnr_3: cython.int[:],    # was: targ_tnr_3
```

Do the same for `trackback_loop_fast` (~L1623) and `sorted_candidates_fast` (~L613).

The exact parameter names — use `grep -A 50 'def trackcorr_loop_fast' algorithms/track_kernels.py | head -60` to confirm before editing.

- [ ] **Step 4: Verify callers pass correct dtypes**

In `algorithms/track.py`, confirm callers convert to correct dtype before calling. If not, add conversion at call sites:

```python
# Example: ensure float64 for targ_x arrays
targ_x_1 = np.ascontiguousarray(fb.targ_x[1], dtype=np.float64)
targ_tnr_2 = np.ascontiguousarray(fb.targ_tnr[2], dtype=np.int32)
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest algorithms/tests/test_track.py algorithms/tests/test_batch_fast.py algorithms/tests/test_synthetic_tracking.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add algorithms/track_kernels.py algorithms/track.py
git commit -m "perf: type untyped memoryview params in trackcorr_loop_fast, trackback_loop_fast, sorted_candidates_fast"
```

---

## Task 10: `segmentation.peak_fit` — deque BFS → typed array queue

**Files:** Modify `algorithms/segmentation.py`, Test `algorithms/tests/test_segmentation.py`

Current code: `waitlist: deque[tuple[int, int]] = deque([(j, i)])` then `wx, wy = waitlist.popleft()`. Python `deque` and tuple-iteration are slow in compiled mode.

- [ ] **Step 1: Run baseline segmentation tests**

```bash
uv run pytest algorithms/tests/test_segmentation.py -v
```

Expected: all pass.

- [ ] **Step 2: Add typed queue arrays at the start of the `peak_fit` BFS block**

In `algorithms/segmentation.py`, inside `peak_fit`, where each peak's BFS starts (the block that currently does `waitlist = deque([(j, i)])`), replace with:

**Before:**
```python
# BFS region growing
waitlist: deque[tuple[int, int]] = deque([(j, i)])
label_img[i, j] = n_peaks

while waitlist:
    wx, wy = waitlist.popleft()
    gvref = int(img[wy, wx])

    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx_pos, ny_pos = wx + dx, wy + dy
        ...
        waitlist.append((nx_pos, ny_pos))
```

**After:**
```python
# BFS region growing — typed array queue (faster than deque in compiled mode)
_qx = np.empty(imy * imx, dtype=np.int32)
_qy = np.empty(imy * imx, dtype=np.int32)
qx: cython.int[:] = _qx
qy: cython.int[:] = _qy
qhead: cython.int = 0
qtail: cython.int = 0
qx[qtail] = j
qy[qtail] = i
qtail += 1
label_img[i, j] = n_peaks

# 4-neighbor offsets (unrolled — avoids Python tuple creation per iteration)
_ddx: cython.int; _ddy: cython.int
nx_pos: cython.int; ny_pos: cython.int; wx: cython.int; wy: cython.int

while qhead < qtail:
    wx = qx[qhead]
    wy = qy[qhead]
    qhead += 1
    gvref = int(img[wy, wx])

    for _ddx, _ddy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx_pos = wx + _ddx
        ny_pos = wy + _ddy
        ...
        qx[qtail] = nx_pos
        qy[qtail] = ny_pos
        qtail += 1
```

Note: `imy * imx` as queue size is a safe upper bound (every pixel enqueued at most once). Move `_qx`/`_qy` allocation outside the outer `for i in range(ymin, ymax)` loop and reset `qhead = qtail = 0` for each new peak to avoid repeated allocation.

- [ ] **Step 3: Remove deque import if no longer used**

```python
# Remove this line from imports if deque is gone:
from collections import deque
```

Verify:
```bash
grep -n 'deque\|waitlist' algorithms/segmentation.py
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest algorithms/tests/test_segmentation.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add algorithms/segmentation.py
git commit -m "perf: replace deque BFS with typed array queue in segmentation.peak_fit"
```

---

## Task 11: Rebuild and verify annotation improvement

**Goal:** Recompile all changed modules and regenerate HTML annotations to confirm score reductions.

- [ ] **Step 1: Rebuild compiled modules**

```bash
uv pip install -e . --no-build-isolation
```

This triggers Cython 3 to recompile all `algorithms/*.py` to `.so` files.

- [ ] **Step 2: Regenerate annotation HTML for changed modules**

```bash
cd algorithms
for mod in vec_utils lsqadj ray_tracing trafo multimed orientation imgcoord track track_kernels segmentation epi sortgrid track3d; do
    uv run cython -3 -a ${mod}.py 2>/dev/null || echo "SKIP $mod"
done
cd ..
```

This overwrites the `.html` files with fresh annotations.

- [ ] **Step 3: Check score improvements**

```bash
python3 - << 'EOF'
import re
from pathlib import Path
from html import unescape

html_dir = Path("algorithms")
LINE_RE = re.compile(
    r'<pre[^>]*class="cython line score-(\d+)"[^>]*>.*?<span[^>]*>(\d+)</span>:\s*(.*?)</pre>',
    re.DOTALL
)
STRIP_TAG = re.compile(r'<[^>]+>')

for html_file in sorted(html_dir.glob("*.html")):
    if html_file.stem == '__init__':
        continue
    content = html_file.read_text(errors='replace')
    hot = [int(m.group(1)) for m in LINE_RE.finditer(content) if int(m.group(1)) >= 5]
    if hot:
        print(f"{html_file.stem:<25} hot={len(hot):4d}  max={max(hot):6d}  avg={sum(hot)/len(hot):6.1f}")
EOF
```

Compare output against the baseline table at the top of this plan. Modules with `math.*` replacements should show reduced average scores.

- [ ] **Step 4: Run full test suite to confirm nothing broken**

```bash
uv run pytest algorithms/tests/ -x -q
```

Expected: all pass. `-x` stops on first failure so regressions surface immediately.

- [ ] **Step 5: Commit final state if everything is green**

```bash
git add algorithms/*.html
git commit -m "chore: regenerate Cython annotation HTML after P1-P4 optimizations"
```

---

## Deferred — Task 12: `tracking_frame_buf` numpy structured arrays (P7)

Replacing `[Target() for _ in range(max_targets)]` with a numpy structured dtype touches all callers across `track.py`, `correspondences.py`, `tracking_run.py`. Estimated 3–5 days of refactoring. Defer until Tasks 1–10 are validated.

When ready, the dtype would be:

```python
TARGET_DTYPE = np.dtype([
    ('x', np.float64), ('y', np.float64), ('n', np.int32),
    ('nx', np.int32), ('ny', np.int32), ('sumg', np.int32),
    ('pnr', np.int32), ('tnr', np.int32),
])
```

And `Frame.targets` would become `np.empty((num_cams, max_targets), dtype=TARGET_DTYPE)`.

---

## Quick-reference: test commands by module

```bash
uv run pytest algorithms/tests/test_lsqadj.py -v
uv run pytest algorithms/tests/test_vec_utils.py -v
uv run pytest algorithms/tests/test_epi.py -v
uv run pytest algorithms/tests/test_sortgrid.py -v
uv run pytest algorithms/tests/test_track3d.py -v
uv run pytest algorithms/tests/test_ray_tracing.py -v
uv run pytest algorithms/tests/test_trafo.py -v
uv run pytest algorithms/tests/test_multimed.py -v
uv run pytest algorithms/tests/test_orientation.py -v
uv run pytest algorithms/tests/test_imgcoord.py -v
uv run pytest algorithms/tests/test_validation_imgcoord.py -v
uv run pytest algorithms/tests/test_track.py -v
uv run pytest algorithms/tests/test_batch_fast.py -v
uv run pytest algorithms/tests/test_synthetic_tracking.py -v
uv run pytest algorithms/tests/test_segmentation.py -v
uv run pytest algorithms/tests/ -x -q  # full suite
```

## Quick-reference: math functions per module

| Module | Functions needed |
|--------|-----------------|
| vec_utils | sqrt, isnan |
| epi | sqrt |
| sortgrid | sqrt |
| track3d | sqrt |
| ray_tracing | sqrt |
| trafo | sqrt, sin, cos |
| multimed | sqrt, tan, asin, atan |
| orientation | sqrt |
| imgcoord | sqrt, tan, asin, sin, cos, atan2 (verify with grep) |
| track | sqrt, sin, cos, acos |
| track_kernels | sqrt, sin, cos, tan, acos, atan, atan2 |
