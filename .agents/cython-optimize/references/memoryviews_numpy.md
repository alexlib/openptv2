# Typed Memoryviews for NumPy / Array-Heavy Code (Pure Mode)

Source: https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html
and https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html

Read this whenever the hot loop indexes into a NumPy array (or any
buffer-protocol object — C arrays, `array.array`, etc.) element by element.
This is the single highest-leverage technique for numeric/scientific and
image-processing code: it is what actually removes per-element Python
overhead, whereas scalar `cython.int` typing on loop indices alone gets you
much less.

**Don't try to type NumPy arrays with `cython.declare(x=np.ndarray)` or
similar** — pure mode has no useful way to express the old
`cnp.ndarray[dtype, ndim=n]` buffer syntax. Use typed memoryviews instead;
they're strictly better (cleaner syntax, usually no GIL needed, faster).

## Basic syntax

A memoryview type is written as `cython.<ctype>[<slices>]`, same shape as
NumPy slicing:

```python
import cython

view1D: cython.int[:] = exporting_object          # 1D view
view3D: cython.int[:, :, :] = exporting_object     # 3D view

def process_3d_buffer(view: cython.int[:, :, :]):
    ...
```

Any object exposing the PEP 3118 buffer protocol works as `exporting_object`
— a NumPy array, a `cython.view.array`, a C array, `array.array`, etc. The
same compiled function transparently accepts any of them.

## None handling

In pure mode, a bare memoryview-typed argument **rejects `None` by default**
(the opposite of the old `.pyx` behavior, where memoryviews accept `None`
unless declared `not None`). If the function should accept an optional
buffer, use `typing.Optional`:

```python
import typing

def process_buffer(
    input_view: cython.int[:, :],
    output_view: typing.Optional[cython.int[:, :]] = None,
):
    if output_view is None:
        output_view = np.empty_like(input_view)
    ...
```

## Indexing

Indexing is translated straight to a memory address (no Python-object
creation) when every dimension gets an integer index:

```python
def add_one(buf: cython.int[:, :]):
    for x in range(buf.shape[0]):
        for y in range(buf.shape[1]):
            buf[x, y] += 1
```

`buf.shape[0]` is a cheap C-level read; prefer it over `arr.shape[0]` on the
original NumPy object inside a hot loop. Negative indices and `...` slicing
work like NumPy. Partial indexing (fewer indices than dimensions) returns a
new view rather than a scalar.

## Contiguity — this is the part that actually matters for speed

Left unspecified, a dimension is assumed *direct + strided* — correct for
any layout, but not the fastest possible code path. If you know (or can
guarantee via `np.ascontiguousarray`) that the array is C-contiguous, say so
with `::1` on the **last** dimension:

```python
c_contiguous: cython.int[:, :, ::1] = c_contig_array
```

Fortran-contiguous: `::1` on the **first** dimension instead:

```python
f_contiguous: cython.int[::1, :, :] = f_contig_array
```

Passing a buffer that doesn't actually match the declared contiguity raises
`ValueError` at the point of assignment — so validate/copy
(`np.ascontiguousarray(arr)`) before calling into a function that declares
`::1`, rather than letting it fail deep in a loop.

Transposing a memoryview (`view.T`) works like NumPy but requires direct
(non-indirect) access in all dimensions.

## GIL

Memoryview indexing, slicing, and transposing generally **do not need the
GIL** — this is a major advantage over the old NumPy buffer syntax, and it's
what makes `@cython.nogil` functions and `prange` parallelism practical over
array data:

```python
@cython.nogil
@cython.ccall
def sum3d(arr: cython.int[:, :, :]) -> cython.int:
    i: cython.size_t
    total: cython.int = 0
    for i in range(arr.shape[0]):
        total += arr[i, 0, 0]
    return total
```

The GIL is still needed for `.copy()`/`.copy_fortran()` and for any element
of dtype `object`.

## Parallel loops with `prange`

For embarrassingly-parallel work over independent array elements/rows
(e.g. per-particle or per-pixel computation), `cython.parallel.prange`
distributes iterations across OpenMP threads. It requires the GIL to be
released for the loop:

```python
from cython.parallel import prange

def func(x: cython.double[:], alpha: cython.double):
    i: cython.Py_ssize_t
    for i in prange(x.shape[0], nogil=True):
        x[i] = alpha * x[i]
```

Assigning to a variable inside the loop body makes it `lastprivate`; using
an in-place operator (`+=` etc.) makes it a reduction across threads
automatically — no manual locking needed for simple accumulation patterns.
`use_threads_if=<condition>` lets you fall back to sequential execution for
small inputs, where thread spawn overhead would dominate:

```python
for i in prange(n, nogil=True, use_threads_if=n > 1000):
    ...
```

**Building with OpenMP** requires extra compiler flags — pass `--openmp` to
`scripts/build_template.py` so the generated `setup.py` includes
`extra_compile_args=['-fopenmp']` and `extra_link_args=['-fopenmp']` (use
`/openmp` instead on MSVC, without a link-arg equivalent).

## Passing a memoryview's raw pointer to an external C function

If you need to call into an existing C routine that takes a raw `double*`,
get the address of the first element (this does need contiguity):

```python
def multiply_by_10(arr):
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    arr_memview: cython.double[::1] = arr
    multiply_by_10_in_C(cython.address(arr_memview[0]), arr_memview.shape[0])
    return arr
```

## Performance knobs, roughly in order of how often they're worth it

1. Contiguous (`::1`) typed memoryview instead of untyped indexing — usually
   the single biggest win for element-wise loops.
2. `@cython.boundscheck(False)` / `@cython.wraparound(False)` on a function
   you've already validated — apply last, after correctness is confirmed.
3. `@cython.initializedcheck(False)` — memoryviews normally check they've
   been initialized on every access; safe to disable once you're sure the
   view is always assigned before use.
4. `nogil` + `prange` — only pays off once the per-iteration work is
   large enough to amortize thread overhead; for small arrays,
   `use_threads_if` avoids regressions.

## What this does NOT speed up

If the code already does whole-array NumPy operations (`a + b`, `a.sum()`,
`np.correlate(...)`), those already run in optimized C inside NumPy —
wrapping them in a typed function changes nothing. Memoryviews only help
where you have an explicit per-element Python loop that NumPy can't express
as a vectorized op (custom kernels, data-dependent branching, neighbor
lookups like particle-linking or correlation-window search).
