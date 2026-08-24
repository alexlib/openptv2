# Remove `prange`/OpenMP from `track_kernels_corr` and `track.py`

**Status:** proposed, not started. Written for a follow-up implementation
session (possibly a different/cheaper model — self-contained for handoff).

## Scope

Remove the `cython.parallel.prange`/`nogil` parallel section from
`src/openptv2/algorithms/track_kernels_corr.py` (the trackback pass), the
`num_threads`/`OPENPTV_NUM_THREADS` plumbing that feeds it from
`src/openptv2/algorithms/track.py`, and the per-platform `-fopenmp`/`/openmp`
compiler+linker flags in `setup.py` that exist only to build it. Nothing else
about the tracking algorithm's output changes — this is a build/perf
simplification, not an algorithm change.

## Why: evidence, not a hunch

`ADR-001-cython-optimization-plan.md`'s Phase 4 section proposed this
parallelization with an **estimated** 2-4× speedup (table: "Est. Speedup"),
cumulative with phases 1-3 into a projected 7-17× vs. a pure-Python baseline.
Those were pre-implementation estimates, never a measured benchmark of the
parallel section in isolation.

Two things found during real-workload testing from `openptv-cloud`
(2026-08-24) argue for removing it rather than trying to make it pay off:

1. **It is never actually multi-threaded in production.** The `prange` call
   (`track_kernels_corr.py:1302-1303`) takes `num_threads` from
   `trackcorr_c_loop()` (`track.py:1049-1053`):

   ```python
   def trackcorr_c_loop(run_info, step, num_threads=None):
       if num_threads is None:
           num_threads = int(os.environ.get("OPENPTV_NUM_THREADS", "1"))
   ```

   `OPENPTV_NUM_THREADS` is read in exactly this one place in the whole repo
   and is never set anywhere else — not by any CLI flag, any GUI control, any
   test default, any doc. `openptv-cloud` (the only real caller of this code
   in production) never sets it either. So `num_threads=1` unconditionally,
   in every real deployment today. The parallel section runs, but always with
   one thread — all the `nogil`/lock-free/thread-local-buffer machinery
   is active for a loop that never has more than one worker.

   (Cython's `prange(num_threads=...)` kwarg — not the standard OpenMP
   `OMP_NUM_THREADS` env var — controls this, which is an easy trap: setting
   `OMP_NUM_THREADS` has **no effect** here, since the explicit kwarg
   overrides it.)

2. **Even forcing it on measured no benefit.** `openptv-cloud`'s `wp4`/`wp5`
   example dataset (`examples/lv-300`, ~700-750 particles/frame, 50 frames)
   was reconfigured to actually exercise the backward pass
   (`plugins.selected_tracking: default` + `track.run_backward: true`,
   confirmed via the reversed-order `step: 8 → step: 2` log lines) and timed
   end to end with `OMP_NUM_THREADS=1` vs. unset:

   | Run | `OMP_NUM_THREADS=1` | unset |
   |---|---|---|
   | wp4 | 185.05s | 200.04s |
   | wp5 | 222.20s | 224.09s |

   No speedup — wp4 was *slower* with the env var unset, wp5 was a wash. (As
   point 1 explains, this env var doesn't even reach the kernel; the
   real-world number that matters is that both conditions ran at
   `num_threads=1` regardless, and thread-spawn/sync overhead for a ~700-750
   particle problem size may not be worth paying even if a caller did wire
   `OPENPTV_NUM_THREADS` up.)

**Conclusion:** this is real, paid-for complexity (build flags on every
platform, a Homebrew `libomp` runtime dependency on macOS, `libgomp1` as an
explicit apt dependency in every consumer's Docker image) for a code path
that has never actually run multi-threaded in production and shows no
measured benefit even when forced to.

## What to remove

### `src/openptv2/algorithms/track_kernels_corr.py`
- `from cython.parallel import prange, threadid` (line 40) and the `prange`
  loop (~line 1302-1303) → replace with a plain `for h in range(orig_parts_1)`
  serial loop.
- The `num_threads` parameter (line 1120) and its validation (lines
  1180-1182) become dead — either drop it from the signature (breaking
  change for direct callers) or keep it accepted-but-ignored for one release
  (see Decisions below).
- The thread-local/pre-allocated-per-thread buffer bookkeeping that exists
  only to make the parallel section safe (comments at lines ~1174, ~1214,
  ~1243) — once serial, these collapse to single buffers, which is a real
  simplification, not just flag removal. Expect this file to shrink
  meaningfully, not just lose two lines.

### `src/openptv2/algorithms/track.py`
- `trackcorr_c_loop()` (line 1049): drop the `num_threads` parameter and the
  `OPENPTV_NUM_THREADS` env read (line 1053), or keep `num_threads` accepted
  but unused for compatibility (see Decisions).

### `setup.py`
- `_windows_compile_args`: drop `/openmp` (line 258).
- `_darwin_compile_args` (lines 259-274): drop `-Xpreprocessor -fopenmp`,
  the `libomp` include/link flags, and — check `_libomp_prefix()` — if
  nothing else needs Homebrew `libomp`, delete that helper and the
  build-time Homebrew dependency entirely. This is the biggest packaging win:
  no more relying on `libomp` being present via Homebrew on macOS CI/dev
  machines, no more `delocate` bundling `libomp.dylib` into the wheel.
  Double-check `delocate` config (`pyproject.toml` / CI workflow) for
  anything that references `libomp` specifically.
- `_posix_compile_args` (lines 276-288): drop `-fopenmp` from both
  `extra_compile_args` and `extra_link_args`.
- Since the comment at `_platform_compile_args` says "same for every
  module," confirm no other extension module accidentally *needs*
  `-fopenmp`/`/openmp` to link (grep for `prange`/`parallel` repo-wide,
  found only in `track_kernels_corr.py` as of this writing — but re-check
  before deleting, since this flag currently applies unconditionally to
  every extension in `get_extensions()`).

### Docs
- `docs/decisions/ADR-001-cython-optimization-plan.md`: Phase 4's numbers
  are now known-estimates that didn't pan out in production; add a note
  (don't rewrite history) pointing at this plan and its measurements.
- `docs/algorithms/tracking.md`, `docs/cloud-batch.md`: grep for `OpenMP`/
  `prange`/`OPENPTV_NUM_THREADS` and remove or correct any user-facing claims
  about multi-core tracking.

### Downstream: `openptv-cloud`
- `Dockerfile.job` installs `libgomp1` specifically because "openptv2's
  compiled kernels are built with `-fopenmp`". Once this lands and a new
  `openptv2` release is pulled in, that `apt-get install libgomp1` line (and
  its comment) can be deleted — small further image-size/build-step win on
  top of the size fixes already done there.

## Testing strategy

- `tests/unit/test_parallel_tracking.py`: currently asserts *determinism*
  across `num_threads in [1, 2, 4, 8]` — the whole point of the test was to
  prove the parallel section doesn't change results. After removal there is
  only one code path, so this test either collapses to a single-threaded
  regression check (delete the multi-`num_threads` parametrization, keep the
  reference-output assertions) or gets deleted if fully redundant with
  existing tracking correctness tests. Don't delete without confirming no
  unique reference-output coverage lives only here.
- `tests/unit/test_track_kernels_tracking_coverage.py`: same audit — check
  what it's actually covering before touching.
- Full test suite (242+ tests per ADR-001) must still pass, particularly
  `test_cavity` nlinks / trajectory-count assertions (the existing bar for
  "did the algorithm change").
- Rebuild on all three platforms (Windows/macOS/Linux CI) to confirm the
  compiler-flag removal doesn't break the build in either direction —
  removing `-fopenmp` should make Linux/macOS builds *less* fragile (no
  `libomp` availability dependency on macOS), but verify.

## Decisions needed before implementing

1. **`num_threads` parameter: delete or deprecate-in-place?**
   - **(a) Delete outright.** Cleanest, but a breaking change for anyone
     calling `trackcorr_c_loop(..., num_threads=N)` or
     `trackback_loop_fast(..., num_threads=N)` directly (not through the
     `default` tracking plugin). Grep for external callers before choosing
     this.
   - **(b) Keep the parameter, ignore it, warn once if set >1.** Softer
     migration, avoids a breaking API change, costs a few lines of
     dead-parameter noise. Given `OPENPTV_NUM_THREADS` was never wired to
     anything reachable from a released CLI/GUI path, (a) is likely safe —
     but confirm via a repo-wide grep for `num_threads=` call sites before
     committing to it.
2. **Keep `OPENPTV_NUM_THREADS` as a no-op env var for one release** (to
   avoid a silent behavior change for anyone who *did* set it expecting
   multi-threading), or remove it immediately since it was never documented
   as a public interface? Given no `docs/` file mentions it (confirmed by
   grep), immediate removal is likely fine.
