# Plan: fix v0.5.5 CI failures + store-fed tracking parity

Date: 2026-08-25 (session start ~18:00 UTC) — continuation planned 2026-08-26
Branch: `ci-debug/zero-links` (local, uncommitted work in tree)

## 0. RESOLVED 2026-08-26 — parity bug root causes

The store-fed tracking divergence (113/591 links vs ASCII 532) had THREE
stacked causes; none involved parallel zarr I/O:

1. **Contaminated fixture store.** `test_cavity/run.zarr` was ingested from a
   working-copy `res/` that already held a previous tracking pass output:
   `ptv_is`+`added` linkage with 642/520/586 seeded links and prio columns,
   which `res_orig` does NOT contain (it ships rt_is only). Store-fed runs
   started from already-linked state.
2. **`read_links=False` ignored by the store branch of `read_path_frame`.**
   With empty `linkage_file_base`, the fallback ``else "ptv_is"`` still read
   stale linkage back as prev/next state, while the ascii branch correctly
   seeded -1/-2. Fixed: link_name defaults to "" (no linkage read) unless an
   explicit base is given; prio now falls back to the "added" group so
   legacy-ingested stores match pipeline-written ones
   (algorithms/tracking_frame_buf.py).
3. **Stale compiled extension.** The `.pyd` predated the fix, masking the
   first re-measurement. Always `uv run python setup.py build_ext --inplace`
   after touching algorithms/.

Also done on 2026-08-26:
- Regenerated `test_cavity/run.zarr` cleanly (targets + correspondences only;
  verified bit-exact vs res_orig; zero linkage groups). Deleted unused,
  contaminated `test_data/track/run.zarr` (no consumer; track tests are
  pure-ASCII over res_orig/img_orig).
- Bumped cython 3.2.9 -> 3.3.0 via uv lock (constraint unchanged >=3,<4);
  rebuilt extensions on py3.11 and py3.13.
- Verification: A/B harness (same params/cal) gives npart=2082 nlinks=1596
  for ascii AND store at BOTH 1 and 4 threads; post-priming buffer state
  bit-identical across modes; affected unit tests green on py3.11+3.13;
  full `-m "not slow"` suite green modulo env-only issues (flowtracks/matplotlib
  extras) fixed with `uv sync --all-extras`.


## 1. Where we are

### SOLVED — root cause of all 10 CI failures identified

Every CI failure has ONE cause: fixture data that exists only in developers'
working copies. `.gitignore` excludes `*_targets`, `res/`,
`test_data/**/res/` — so fresh CI checkouts lack files the tests read.
Locally everything passes; on CI everything fails, deterministically, on all
4 matrix jobs (ubuntu/windows × py3.11/3.13).

| failing test | missing on CI | fix |
|---|---|---|
| `test_parallel_tracking.*` (2) | `test_cavity/img/*_targets` | zarr migration (below, step B) |
| `test_parallel_correspondences` (3 incl. parity/zarr variants) | same | zarr migration (step B) |
| `test_trackcorr_store_only` ×2 (`adapt_proptv_*`) | `synthetic_turbulent/res` rmtree'd unconditionally | DONE: guarded rmtree |
| `test_tracker_3d_compat.*` (2) | entire `test_data/track` untracked | commit zarr store + minimal params/cal |
| `test_standalone_dumbbell_calibration_cycle` | `cal/camN.tif_targets` | commit 4 small ASCII cal-targets (calibration tooling is ASCII by design) |

Also done: lint job fixed (F841 in new script), `scripts/run_regression_tests.py`
added (`--tier1`, `--py 3.13`).

### OPEN — store-fed tracking kernel parity

Migrating `test_parallel_tracking.py` to Zarr fixtures works mechanically
(data flows, linking happens) but results diverge from the ASCII baseline:

| input source | npart | nlinks @1 thread | nlinks @4 threads |
|---|---|---|---|
| ASCII `res_orig` (reference) | 2082 | **532** | **532** (deterministic ✓) |
| Zarr store-fed `TrackingRun(store=…)` | 2082 | **113** | **591** ⚠ |

Both wrong AND thread-dependent. This is a real openptv2 bug (or ingestion
bug), not a test artifact.

## 2. Investigation plan for the parity bug (tomorrow)

Working repro: rewritten `tests/unit/test_parallel_tracking.py` on the branch
+ `uv run --python 3.13 -m pytest tests/unit/test_parallel_tracking.py -q`.
ASCII reference: `git show HEAD:tests/unit/test_parallel_tracking.py`.

Hypotheses, ordered by likelihood:

**H1 — targets buffer not fed from the store during tracking-only runs.**
The epipolar search needs per-frame 2D TargetArrays. In the ASCII flow the
kernel loads them from `img/camN.<frame>_targets`. With `store=` attached,
check which code path populates target buffers:
- grep compiled-side calls: does `trackcorr_c_loop`/`track_forward_start` go
  through `tracking_frame_buf.read_targets(..., store=run.store)`?
- Experiment: point `spar.img_base_name` at a directory WITH the ASCII
  targets while feeding correspondences from the store. If links jump to
  ~532 → H1 confirmed: targets silently missing in pure-store mode.

**H2 — cam_target_ids round-trip damage at ingestion.**
`convert_ascii_to_zarr` → `write_correspondences(frame, pos_3d, cam_ids)`.
Verify dtype/sentinels survive:
```python
from openptv2.storage import RunStore
from openptv2.algorithms.tracking_frame_buf import read_path_frame
import numpy as np
store = RunStore('test_data/test_cavity/run.zarr', mode='r')
cor_s, path_s = read_path_frame('res/rt_is','res/ptv_is','res/added',10001,store=store)
# compare cor_s[i].p against ASCII rt_is columns 5..8 for frame 10001
```
Check: int32 vs int64, -1 sentinels preserved, ordering (rt_is row order vs
sorted-by-y), pnr/tnr remapping. NOTE: standalone `read_path_frame(...,
store=None)` returns 0 rows even against a populated `res/` — understand WHY
(likely linkage/prio file absence handling) before trusting it as oracle;
the old test's flow primes buffers inside `track_forward_start`.

**H3 — float precision shift flips gate decisions.**
Positions pass through zarr round-trip; cavity gates are tight. Diff
store-read vs ascii-read Pathinfo.x bit-exactly (`np.array_equal`, not
allclose). If tiny diffs exist, check whether ingestion writes float32.

**H4 — lazy per-frame store reads race with OpenMP search (explains 113≠591).**
If H1/H2/H3 check out but 1T≠4T persists with identical inputs, the
num_threads>1 path races on lazy buffer population. Look at
`trackcorr_c_loop`'s buffer advance: is `read_path_frame` called inside the
parallel region? Fix would be eager priming under lock or before the region.

**H5 — prev/next linkage seeding differs.**
Store without linkage seeds `prev=-1,next=-1`; ASCII res_orig ptv_is may or
may not exist pre-tracking. Verify both paths start from identical
prev/next/prio state.

Suggested order: H2 (pure data diff, fast) → H3 → H1 → H5 → H4.
Decision rule: if any hypothesis shows data damage, fix ingestion/read path
first and re-measure the table above; only touch kernel threading last.

## 3. Landing checklist (after parity resolved)

1. Regenerate fixture stores cleanly:
   `convert_ascii_to_zarr('test_data/test_cavity', store_path='test_data/test_cavity/run.zarr')`
   (same for track). Current ones were built from local working copies —
   verify against `res_orig` content before committing (track's came from
   generated `res/`, not `res_orig`!).
2. Commit: both `run.zarr` fixture stores; 4 × `cal/camN.tif_targets`;
   minimal `test_data/track` non-generated files (parameters/, cal/,
   conf.yaml); rewritten tests; rmtree guards; regression script.
   Do NOT commit: generated `res/`, `img/` working copies, ASCII run data.
3. Rewrite remaining consumers:
   - `tests/unit/test_parallel_correspondences.py`: source targets from
     store (`RunStore.read_targets`), keep in-memory + zarr_store variants.
   - `tests/unit/test_tracker_3d_compat.py`: copy `run.zarr` instead of
     `res_orig`/`img_orig` dirs; keep legacy `.par` parameter reading (that
     IS the compat surface under test).
4. Full suite green locally on py3.11 AND 3.13
   (`scripts/run_regression_tests.py`, then `pytest -m "not slow"`).
5. Push branch → CI must be green on fresh checkout → merge to main.
6. Then resume the TT13_aorta cloud benchmark (wp1 tracking phase relaunch
   needs this fixed image if we want store-fed parallel tracking in-cloud;
   sequential fallback already works).

## 4. Open questions for Alex

- Cal-image `_targets`: stay ASCII (calibration GUI tools read them), or
  migrate into RunStore too? (I kept them committed as ASCII.)
- Should `tr_new()` gain an explicit `store=` passthrough (currently only
  `TrackingRun(...)` ctor takes it)? The asymmetry bit us.
- Is 4-thread-vs-1-thread link-count divergence expected to be EXACTLY zero
  on well-formed input? (The determinism test assumes yes.)
