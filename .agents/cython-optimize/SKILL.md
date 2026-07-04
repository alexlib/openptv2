---
name: cython-pure-mode
description: Guides converting and speeding up existing Python code using Cython 3's "pure Python mode" — adding static types with PEP 484/526 annotations, cython.declare, @cython.cfunc/@cython.ccall/@cython.cclass, and typed memoryviews — so a .py file compiles to a fast C extension while remaining importable and runnable as plain Python. Use this whenever the user wants to speed up slow Python code, compile Python with Cython, add static typing for performance, optimize numeric or array-heavy loops (NumPy, image processing, particle tracking, signal/video processing, simulations), or mentions Cython "pure mode", cythonize, .pyx vs .py performance, or turning a hot function into a compiled extension without giving up testability. Also trigger when the user describes a slow nested loop, a profiled hot path, or "how do I make this Python function C-fast" even if they never say the word "Cython".
---

# Cython 3 Pure Python Mode: Optimization Workflow

Pure mode lets you add Cython's static typing directly to a `.py` file (via
type annotations, or via decorators/calls from the `import cython` module)
instead of writing a separate `.pyx` file. The huge advantage: the file stays
valid, importable, testable Python. The tradeoff: you're restricted to what
can be expressed (or emulated) in Python syntax — genuinely exotic C/C++
features still require a real `.pyx` file.

Read `references/pure_mode_reference.md` for the full syntax cheat sheet
(declare, cfunc/ccall/cclass, C types, GIL control, structs, cimports) and
`references/memoryviews_numpy.md` when the code touches NumPy arrays or
raw buffers — this is the part that matters most for numeric/scientific and
image-processing code, since plain pure-mode typing alone rarely fixes
array-heavy hot loops.

## Why order matters here

Typing every variable in a module is tempting but usually wrong: it makes the
code harder to read, doesn't move the needle where the interpreter wasn't
spending time anyway, and increases the chance of subtly changing behavior
(C integer overflow instead of a Python exception, a `None` where a typed
object was expected, etc). Cython's own docs note that a pure-mode compile
with zero type annotations already buys ~20-50% just from cutting interpreter
overhead — the static typing is what gets you to 10-100x, but only on the
lines that are actually the bottleneck. Profile, then type surgically.

## Workflow

1. **Profile first.** Use `cProfile` (whole-program) or `line_profiler`
   (line-by-line) to find which function(s) actually dominate runtime. If the
   user hasn't profiled yet, do this before touching any code — it's common
   for the "obviously slow" loop to not be the real bottleneck.

2. **Sanity-check that pure mode is the right tool.** If the hot code needs
   C++ templates, complex operator overloading, or other features that don't
   have a Python-expressible equivalent, pure mode can't do it — fall back to
   a real `.pyx` file for that piece. For everything else (the large
   majority of numeric/scientific hot loops), pure mode is the better
   starting point because the file keeps running under the plain interpreter
   the whole time you're iterating.

3. **Add `import cython`** at the top of the target module. This import is a
   no-op at interpreted runtime (Cython ships a pure-Python shadow
   implementation) and only does something when the file is compiled.

4. **Type incrementally, starting at function boundaries:**
   - Add PEP 484 parameter/return annotations, or use `@cython.locals(...)`
     and `@cython.returns(...)` when you don't want annotations to be
     visible to other tooling (e.g. if the project already uses type hints
     for something else, like mypy contracts that shouldn't change meaning).
   - Watch the `int`/`float` gotcha: a bare `int` annotation means "exact
     Python int object", not a C int — you must write `cython.int` to get a
     genuine machine integer with wraparound-on-overflow semantics. `float`
     annotations, by contrast, do map to C `double`. See the reference
     table in `references/pure_mode_reference.md`.
   - For internal helper functions never called from outside the module,
     promote them with `@cython.cfunc` (C-level call, no Python call
     overhead) or `@cython.ccall` if Python code still needs to call them.
   - For classes whose attribute access dominates a hot loop, convert with
     `@cython.cclass` and type the attributes via `cython.declare(...)` —
     this replaces per-attribute dict lookups with C struct field access.

5. **For NumPy/array-heavy code, use typed memoryviews, not scalar typing.**
   Typing loop indices as `cython.int` helps some, but the dominant cost in
   array code is almost always element access through the buffer protocol.
   Read `references/memoryviews_numpy.md` before touching any code that
   indexes into `np.ndarray` in a loop — the syntax and contiguity flags
   (`[:, ::1]` etc.) matter a lot for the resulting speed, and getting them
   wrong silently falls back to slow generic indexing.

6. **Only disable safety nets deliberately, and only on functions you're
   confident about.** `@cython.boundscheck(False)` and
   `@cython.wraparound(False)` remove real safety checks — out-of-bounds
   access becomes a silent crash or memory corruption instead of a Python
   exception. Apply these only after the typed version is already correct
   and tested, as a final tightening pass on a proven hot function.

7. **Decide if you need to touch the GIL.** If the goal is just single-
   threaded speed, skip this. If the goal is true parallelism (e.g. running
   independent particle-tracking chunks across cores), typed/`nogil`
   functions plus `cython.parallel.prange` are the relevant tools — covered
   in `references/memoryviews_numpy.md`.

8. **Generate the build files.** Run
   `scripts/build_template.py <path/to/module.py>` to emit a `setup.py` and
   `pyproject.toml` using `cythonize(..., language_level="3")`. Pass
   `--numpy` if the module uses typed memoryviews/NumPy buffers, and
   `--openmp` if it uses `prange`. Then build with:
   ```
   pip install -e .
   ```
   or
   ```
   python setup.py build_ext --inplace
   ```

9. **Verify correctness before celebrating a speedup.** Typed code can
   change behavior silently (integer overflow, exceptions swallowed by
   `nogil`/`noexcept` paths, `None` handling on typed attributes). Run the
   project's existing test suite against the compiled module, or at minimum
   compare outputs between the interpreted and compiled versions on the same
   inputs.

10. **Benchmark before vs. after.** Use `scripts/benchmark.py` to time the
    original pure-Python path against the compiled extension on the same
    inputs and report the speedup factor. Re-profile the compiled module
    afterward — the next-hottest function is now a different one, and the
    same loop (profile → type → verify → benchmark) repeats.

## Common pitfalls worth flagging to the user

- **Vectorized NumPy code doesn't benefit from element-wise typing.** If the
  hot code already calls NumPy ufuncs/whole-array operations (`a + b`,
  `np.sum(a)`), Cython typing won't speed that up — the work already happens
  in C inside NumPy. Pure-mode typing pays off for explicit Python-level
  loops that NumPy can't vectorize (irregular access patterns, data-
  dependent branching per element, custom kernels) — which is common in
  particle-tracking/correlation code but not universal, so profile to
  confirm the loop is actually where time goes.
- **Setting a typed object/memoryview to `None`** is legal but any further
  attribute access or indexing on it can segfault instead of raising — don't
  disable bounds checking on code paths that might still see `None`.
- **`int` vs `cython.int`** — the single most common silent-behavior-change
  mistake in ported pure-mode code.
- **`@cython.cfunc` is invisible from Python once compiled.** It only exists
  at the C level, so a test suite, a benchmark script, or any other module
  trying to `import` and call it directly will get an `AttributeError` after
  compilation even though it worked fine interpreted. If nothing else in the
  same compiled module calls it, the C compiler will even warn that the
  function is unused. Use `@cython.ccall` for anything that needs to remain
  externally callable (including your own verification/benchmark step) —
  reserve `@cython.cfunc` for true internal helpers only called from other
  code in the same file.
