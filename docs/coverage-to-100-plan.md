# Plan: Trustworthy Coverage → 100% (algorithms + GUI)

## Context / why this exists

While fixing the high-pass-filter-size bug (a wrong variable, `get_cross_size()`,
passed as the filter kernel — invisible for a long time), we asked "how did no
test catch this?" Investigating coverage revealed a bigger problem:

**The coverage setup measured 0% of the source.** `[tool.coverage.run]` had
`source = ["openptv2", "algorithms", "gui", "batch"]`. The bare names
`algorithms`/`gui`/`batch` are not importable top-level modules (they live under
`openptv2.*`), so coverage emitted `Module algorithms was never imported` and
silently measured **only the test files**. The "64.8%" number the suite printed
was coverage *of the tests*, not the product. Nobody could see an untested source
line, which is exactly how a wrong-variable bug survives.

Second problem: every module under `openptv2/algorithms/` is compiled to a
`.so` via Cython. `coverage.py` traces Python bytecode line events and **cannot
see into compiled extensions** — so even heavily-tested modules (e.g. `trafo.py`,
exercised by 9 passing tests) report 0%. Measuring them needs Cython line
tracing, not more tests.

So "get to 100%" is really three problems: (1) measure the right thing,
(2) make the compiled algorithms measurable, (3) close the real gaps.

## Step 0 — Fix the coverage config (DONE)

`[tool.coverage.run].source` is now `["src/openptv2"]` with `omit` for
notebooks/demos, and `[tool.coverage.report].show_missing = true`. A normal
`uv run pytest --cov` now measures the pure-Python source (gui, batch,
`transforms.py`, `correspondences.py`, `autocalibration.py`, `cli.py`, …).
Compiled `algorithms/*` still report 0% until Step 1.

## Step 1 — Make the Cython algorithms measurable (line tracing)

Without this, ~8,000 statements of `algorithms/*` are permanently 0% and "100%"
is impossible. `Cython.Coverage` is already installed. Enable an **opt-in**
line-tracing build (never on by default — line tracing is much slower):

1. **`setup.py` `_cythonize_all()`** — gate two additions on an env var so
   normal/production builds are unaffected:
   ```python
   trace = os.environ.get("OPENPTV_CYTHON_TRACE") == "1"
   directives = { ...existing..., }
   if trace:
       directives["linetrace"] = True
       directives["profile"] = True
   cythonize(targets, ..., compiler_directives=directives)
   ```
2. **`setup.py` `get_extensions()`** — when tracing, add the C macro so the
   generated C actually emits trace calls:
   ```python
   if os.environ.get("OPENPTV_CYTHON_TRACE") == "1":
       for ext in extensions:
           ext.define_macros += [("CYTHON_TRACE_NOGIL", "1")]
   ```
3. **Coverage plugin** — measure with the Cython plugin enabled *only* in the
   tracing workflow (keep it out of the default `[tool.coverage.run]` so normal
   runs don't need `.c` files present). Use a dedicated config:
   `.coveragerc.cython` with `plugins = Cython.Coverage` + the same
   `source`/`omit`.
4. **Workflow** (documented in a `scripts/coverage.sh`):
   ```bash
   OPENPTV_CYTHON_TRACE=1 uv run python setup.py build_ext --inplace   # slow, traced build
   uv run pytest --cov --cov-config=.coveragerc.cython --cov-report=term-missing
   # then rebuild WITHOUT the env var to restore the fast production .so
   uv run python setup.py build_ext --inplace
   ```

Deliverable of Step 1: a real per-line coverage number for every
`algorithms/*.py`. Expectation: many are already high (rich unit tests exist);
the gaps will be error branches, `if cython.compiled` fallbacks, and rarely-hit
kernels.

## Step 2 — Establish the real baseline

Run both modes once and record per-module coverage:
- pure-Python: `uv run pytest --cov` (Step 0 config)
- algorithms: the Step 1 traced workflow

`<!-- BASELINE TABLE: filled from /tmp/cov_fixed.json after the corrected
full-suite run completes; per-module stmts / missing / % for gui, batch, and
top-level pure-Python modules. -->`

## Step 3 — Close gaps by area

### 3a. algorithms/ (target: 100% of reachable lines)
Strong unit tests already exist. After Step 1 shows the misses, add focused
tests for: error/validation branches, the `if not cython.compiled:` interpreted
fallbacks, degenerate inputs (0 targets, 1 camera, empty frames), and any kernel
paths the fixtures never hit. Interpreted-only fallback blocks that can't run
compiled get `# pragma: no cover` with a one-line reason.

### 3b. GUI (the hard part — be honest about "100%")
`gui/` is TraitsUI/Chaco/PySide6. Three tiers:
1. **Pure logic → unit-test headless** (Agg backend). Already started:
   `plot_3d_positions.build_3d_figure`/`compute_fov_bounds` are pure and tested.
   Extend this pattern: pull computation out of handlers into pure functions and
   test those (rt_is parsing, path building, parameter mapping, frame-number
   extraction, `image_split`, `simple_highpass` wiring).
2. **Handlers → test with a fake `info`** whose `.object` is a lightweight stand-in
   exposing the attributes the handler reads (`exp_path`, `vpar`, `cpar`, `cals`,
   `get_parameter`). Assert side effects (files written, dialogs chosen) without
   opening a window. This is where the high-pass class of bug should be caught.
3. **Event-loop / rendering glue** (`configure_traits`, editor `init`, Qt canvas)
   — not unit-testable without a display. Mark with `# pragma: no cover` and a
   reason, OR cover via a smoke test under `xvfb`/`QT_QPA_PLATFORM=offscreen` in CI.

Realistic target: **100% of tiers 1–2**, tier 3 explicitly excluded with
justifications (so "100%" means "100% of testable lines", enforced — not a
misleading raw number).

### 3c. batch/ (target: 100%)
`run_batch`/`main` are drivable end-to-end (already partly tested). Add tests for
the error/branch paths (missing files, mode routing both/sequence/tracking,
plugin-not-found) using the small fixtures.

## Step 4 — Regression test for the bug class that started this

Add `tests/batch/test_sequence_singleframe_parity.py` (already stubbed): run the
**same frame** through the single-frame pipeline (`py_pre_processing_c` →
`py_detection_proc_c` → `correspondences`) and through `py_sequence_loop`, and
assert **identical** correspondence counts / rt_is. This fails on ANY
single-frame-vs-sequence divergence (high-pass size, dtype, param object),
regardless of dataset SNR — the guard the suite was missing.

## Step 5 — Enforce in CI so it can't rot

Once baselines are green, add `--cov-fail-under=100` (against the measurable set)
to the CI test job, plus the traced-algorithms job. Ratchet: set the threshold to
the current number and raise it as gaps close, so coverage can only go up.

## Honest expectation

- algorithms + all pure logic: **100% is achievable.**
- GUI event-loop/rendering: **100% of raw lines is not practical** without a
  headless display harness; the plan reaches 100% of testable lines with
  documented `# pragma: no cover` (or optional offscreen-Qt CI to cover even
  those). Recommend the pragma approach first; add offscreen-Qt only if you want
  the literal number.

Effort: Step 1 (~half day, infra), Steps 3a/3c (~1–2 days), Step 3b GUI (the
bulk — several days, and where the refactor-for-testability pays off long-term).
