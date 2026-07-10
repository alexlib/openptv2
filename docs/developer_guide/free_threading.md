# Free-Threaded Python (no-GIL) in OpenPTV2

Python 3.13 introduced an experimental no-GIL build (PEP 703), and Python 3.14
ships it as stable. These builds are identified by a `t` suffix — `cp313t`,
`cp314t` — and distributed as separate wheels. This page explains what
free-threading means for OpenPTV2, what performance gains are realistic, how to
set it up locally and in the cloud, and what to watch out for.

---

## 1. Two kinds of parallelism — don't confuse them

OpenPTV2 has **two independent parallelism axes**. It is important to keep them
separate:

| Axis | Mechanism | Works on GIL Python? | Gains from free-threading? |
|------|-----------|----------------------|----------------------------|
| **Intra-loop SIMD** | OpenMP `prange` inside a Cython kernel | Yes — OpenMP bypasses the GIL | No additional gain |
| **Inter-task Python-thread fan-out** | `ThreadPoolExecutor` dispatching multiple independent kernel calls | No — threads serialize under the GIL | **Yes — this is the free-threading win** |

`trackcorr_c_loop(..., num_threads=N)` uses **OpenMP** (axis 1). It parallelises
the inner per-particle loop and already works well on standard GIL Python.
Free-threading does not change that path.

The gain from free-threading comes from dispatching **multiple independent
frames / cameras / batch chunks** to separate Python threads simultaneously
(axis 2). Each thread enters a compiled `@cython.nogil` kernel and releases the
GIL; on a free-threaded interpreter those threads actually run in parallel.

---

## 2. GIL vs no-GIL — what changes

### Runtime marker

```python
import sys, sysconfig

# Build-time flag (True only for *t wheels)
is_ft_build = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))

# Runtime state (can be re-enabled by an imported C extension)
gil_active = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
```

### Comparison table

| Property | GIL Python (`cp313`, `cp314`) | Free-threaded (`cp313t`, `cp314t`) |
|----------|-------------------------------|------------------------------------|
| `Py_GIL_DISABLED` | `0` | `1` |
| `sys._is_gil_enabled()` | `True` | `False` (if all imports cooperate) |
| Single-thread compute speed | baseline | ~5–15 % slower (atomic refcounting) |
| CPU-bound Python threads | serialize — wall-time grows with N | run in parallel — wall-time flat |
| OpenMP kernels | unaffected | unaffected |
| Wheel filename | `...-cp314-cp314-linux_x86_64.whl` | `...-cp314t-cp314t-linux_x86_64.whl` |

### Measured benchmark numbers

Measured on an 8-core x86\_64 desktop (single sample; treat as indicative, not
a guarantee). Image: 1280 × 1024 pixels, `fast_box_blur` span=3.

**Single-thread compute** (`bench_compute.py`):

| Interpreter | `lowpass_3` | `fast_box_blur` |
|-------------|-------------|-----------------|
| Python 3.13.9 GIL | 2.1 ms | 24.0 ms |
| Python 3.14.0 GIL | 1.8 ms (–14 %) | 24.0 ms |
| Python 3.14.3t (no-GIL) | 2.0 ms (+10 % vs 3.14) | 26.5 ms (+10 %) |

**Thread scaling** — N tasks dispatched to N threads (`bench_threads.py`):

| Threads | GIL speedup | Free-threaded speedup |
|---------|-------------|----------------------|
| 1 | 1.00× | 1.00× |
| 2 | 1.18× | 1.85× |
| 4 | 1.21× | 2.43× |
| 8 | 1.22× | 2.61× |

The GIL build caps at ~1.2× regardless of thread count. The free-threaded build
scales to ~2.6× on 8 cores (limited by allocator contention and hyperthreading,
not by code correctness).

---

## 3. The `freethreading_compatible` directive and the stale-`.c` gotcha

All openptv2 Cython modules are built with:

```python
# setup.py
directives = {
    ...
    "freethreading_compatible": True,
}
```

This causes Cython to emit `Py_MOD_GIL_NOT_USED` in the generated C, which
tells the free-threaded interpreter not to re-enable the GIL on import.

**The critical pitfall:** cythonize skips `.c` regeneration when the `.py` file
is older than the existing `.c` file. If you add `freethreading_compatible` to
`setup.py` but forget to touch the source files, the old `.c` files are reused
and the directive never reaches the compiled code. The symptom is
`sys._is_gil_enabled()` returning `True` even in a `cp314t` process.

**Fix — force regeneration:**

```bash
touch src/openptv2/algorithms/*.py
.venv/bin/python setup.py build_ext --inplace
```

CI builds are always clean (no stale `.c` files) so this is a local-only trap.

---

## 4. Local installation

### Install a free-threaded Python

```bash
# uv knows about *t variants
uv python install 3.14t

# Verify
uv run --python 3.14t python -c "import sysconfig; print(sysconfig.get_config_var('Py_GIL_DISABLED'))"
# → 1
```

### Build openptv2 against it

```bash
# Create a dedicated venv
uv venv --python 3.14t .venv314t
source .venv314t/bin/activate

# Install build dependencies
pip install "numpy>=2.0,<3" "cython>=3.0.10,<4" "setuptools>=77" "packaging>=24.2"

# Force .c regeneration (needed if you have stale files from a previous build)
touch src/openptv2/algorithms/*.py
python setup.py build_ext --inplace

# Verify GIL is off
python -c "import sys; print('GIL enabled:', sys._is_gil_enabled())"
# → GIL enabled: False

# Run tests
pytest tests/unit/ -m "not slow" -v
```

### Verify the build is actually no-GIL

```python
import sys, sysconfig
from openptv2.algorithms import track_kernels

assert sysconfig.get_config_var("Py_GIL_DISABLED"), "Not a free-threaded build"
assert not sys._is_gil_enabled(), "GIL was re-enabled by an import — check for stale .c files"
print("All good — running no-GIL")
```

---

## 5. Cloud deployment

### When no-GIL wins

Free-threading pays off when work is **embarrassingly parallel at the Python
level** and each unit of work is CPU-bound compiled code:

- Processing multiple frames in parallel (one thread per frame)
- Processing multiple cameras simultaneously
- Batch-mode tracking over a large sequence with independent sub-ranges

A typical cloud pattern:

```python
from concurrent.futures import ThreadPoolExecutor
from openptv2.algorithms.image_processing import fast_box_blur

frames = [load_frame(i) for i in range(N)]

with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
    results = list(pool.map(lambda f: fast_box_blur(f, span=3, nx=W, ny=H), frames))
```

On GIL Python, the threads serialize and you get ~1.2× at best. On
free-threaded Python you approach linear scaling up to allocator contention
(typically 4–8 cores).

### Container / wheel notes

- Pull the `*t` wheel for the matching Python version. PyPI hosts them
  separately from the standard wheels.
- In a Dockerfile: `uv python install 3.14t` or pin the base image to a
  free-threaded variant.
- The openptv2 CI builds `cp313t` and `cp314t` wheels automatically via the
  `enable = ["free-threaded", "cpython-prerelease"]` setting in
  `pyproject.toml` and the matching `CIBW_ENABLE` env var.
- OpenMP still works inside compiled kernels — free-threading and OpenMP are
  orthogonal. Use OpenMP for inner loops, Python threads for outer fan-out.

### When to stay on GIL Python

- Single-threaded workloads: the ~5–15 % single-thread overhead makes no-GIL
  slower.
- NumPy/SciPy intensive paths that call into their own C layer — those already
  release the GIL internally; no extra benefit from a free-threaded build.
- Any C extension in your dependency tree that does not declare
  `Py_MOD_GIL_NOT_USED` will re-enable the GIL silently. Check with
  `sys._is_gil_enabled()` after imports.

---

## 6. Related

- [Building from Source](building.md) — editable installs, forced `.c` regen
- [Cython & Pure Python Modes](cython_and_pure_python.md) — compiled vs interpreted
- `scripts/bench_compute.py` — single-thread microbenchmark (CPU-only)
- `scripts/bench_threads.py` — thread-scaling benchmark (GIL vs no-GIL)
- `tests/unit/test_free_threading.py` — runtime GIL-state and concurrency tests
