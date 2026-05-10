# openptv2 (optv bindings) -- Generated-Code Orientation

**Project**: openptv2 -- Particle Tracking Velocimetry with C core + Cython bindings
**SHA**: 5a8e937cf5e27df489aa89c541c9821072238010
**Generators detected**: Cython

## 1. Generator inventory

| Generator | Version | Source files | Output files | LOC: source / output |
|---|---|---|---|---|
| Cython | 3.2.4 | 11 .pyx + 12 .pxd | 11 .c | 2841+402 / 206,156 |
| Hand-written C (liboptv) | n/a | 18 files in lib/src/ | linked via cmake | (separate library) |

## 2. Source-of-truth map

- **Bugs originate in**: `bindings/optv/*.pyx` (11 files, 2841 LOC total)
- **Generated artefacts (review only via cross-reference)**: `bindings/optv/*.c` (11 files, 206K LOC)
- **C library (separate audit scope)**: `lib/src/*.c` (18 files) -- the underlying C engine
- **Vendored deps**: none

## 3. Cross-reference recipe

Cython 3.2.4 embeds source-line markers in generated `.c`:
```
grep '"calibration.pyx":35' bindings/optv/calibration.c
```

## 4. Idioms to treat as ACCEPTABLE noise

| Pattern | grep regex | Reason |
|---|---|---|
| Refcount macros | `__Pyx_X?(INC\|DEC)REF\(` | Cython runtime |
| Ownership tracking | `__Pyx_X?(GIVE\|GOT)REF\(` | Cython runtime |
| Error indicator checks | `__Pyx_PyErr_` | Cython error-path boilerplate |
| Buffer protocol setup | `__Pyx_GetBuffer\|__Pyx_ReleaseBuffer` | Cython typed-memoryview internals |
| Type-ready macros | `__Pyx_PyType_Ready` | Cython module init |
| Import machinery | `__Pyx_Import\|__Pyx_ImportFrom` | Cython module init |

## 5. Cython-specific bug patterns (no AST scripts run -- manual analysis)

**No AST script outputs were found** in `reports/optv_v1/preflight/q*.json`. The following is manual identification.

### Pattern A: `__init__` used instead of `__cinit__` for C allocation (ALL 11 cdef classes)

Every `cdef class` in this project allocates C memory in `__init__`, never `__cinit__`. This means:
- If `__init__` raises mid-way, partially-allocated fields may leak (no `__cinit__` guarantee of `__dealloc__` being called)
- Subclass `__init__` chains could re-allocate without freeing

**Affected classes** (verified): `Calibration` (calibration.pyx:35), `MatchedCoords` (correspondences.pyx:63), `MultimediaParams` (parameters.pyx:57), `TrackingParams` (parameters.pyx:154), `SequenceParams` (parameters.pyx:327), `VolumeParams` (parameters.pyx:420), `ControlParams` (parameters.pyx:561), `TargetParams` (parameters.pyx:761), `Tracker` (tracker.pyx:40), `Target` (tracking_framebuf.pyx:31), `TargetArray` (tracking_framebuf.pyx:132), `Frame` (tracking_framebuf.pyx:241).

**Triage**: CONSIDER MEDIUM -- no subclassing observed in codebase; risk is theoretical but architecturally fragile.

### Pattern B: `read_*` methods reassign C pointer -- inconsistent free-before-reassign

| Method | File:line | Frees old pointer? | Risk |
|---|---|---|---|
| `Calibration.from_file` | calibration.pyx:77 | YES (`free()`) | OK |
| `TrackingParams.read_track_par` | parameters.pyx:210 | YES (`free()`) | OK |
| `SequenceParams.read_sequence_par` | parameters.pyx:378-379 | YES (NULL-checked `free_sequence_par`) | OK |
| `VolumeParams.read_volume_par` | parameters.pyx:533-534 | YES (NULL-checked `free()`) | OK |
| `ControlParams.read_control_par` | parameters.pyx:695-697 | YES (complex: nulls mm, frees, reassigns) | OK but fragile |
| `TargetParams.read_target_par` | parameters.pyx:900 | YES (`free()`) | OK |

All `read_*` methods do free before reassign. **No active memory leaks here.**

### Pattern C: `cdef` function with no return-type exception spec

`wrap_1d_c_arr_as_ndarray` at parameters.pyx:17 is `cdef` with implicit `object` return -- exceptions propagate correctly for `object`-returning cdefs. **Not a bug.**

The `cdef int` local variables (parameters.pyx lines 115-737) are **variable declarations**, not function signatures. **No Q1-type bugs present.**

### Pattern D: No `nogil` blocks, no buffer protocol `__getbuffer__`

Grep confirms: no `with nogil:` blocks, no `__getbuffer__`/`__releasebuffer__` implementations. Q2/Q5 patterns are not applicable.

## 6. Project-specific structural patterns

| Pattern | Where | Description | Why it matters |
|---|---|---|---|
| Ownership-flag pattern | `Target.__init__` (tracking_framebuf.pyx:48-51) | `_owns_data` flag controls whether `__dealloc__` frees memory | Prevents double-free when Target wraps borrowed C pointer; downstream agents must not flag the conditional free as a bug |
| keepalive reference | `Tracker.__init__` (tracker.pyx:57) | `self._keepalive = (cpar, vpar, tpar, spar, cals)` | Prevents GC of parameter objects whose C pointers are borrowed by `run_info`; correct RAII pattern |
| `wrap_1d_c_arr_as_ndarray` | parameters.pyx:17-46 | Creates ndarray view over C data with `Py_INCREF(base_obj)` + `PyArray_SetBaseObject` | Correct base-object pattern; prevents use-after-free of C arrays returned as numpy views |
| `ControlParams` owns `MultimediaParams` | parameters.pyx:695-701 | `read_control_par` nulls `_mm_np` before free to prevent double-free of shared `mm` pointer | Fragile ownership coupling -- a real area to watch |

## 7. Build / configuration

- **Cython version**: 3.2.4 (from generated .c header); `pyproject.toml` requires `cython>=3.0.0` -- **no upper bound, Cython 4 risk**
- **language_level**: `3` set in parameters.pyx, tracker.pyx, and all .pxd files; **missing from** calibration.pyx, correspondences.pyx, epipolar.pyx, image_processing.pyx, imgcoord.pyx, orientation.pyx, segmentation.pyx, tracking_framebuf.pyx, transforms.pyx (9 of 11 .pyx files)
- **Safety directives**: no `boundscheck`, `wraparound`, `cdivision`, `initializedcheck`, or `nonecheck` overrides found -- all use Cython defaults (safe)
- **abi3 / limited API**: not used (`Py_LIMITED_API` not defined)
- **Free-threading**: no `freethreading_compatible` directive; no `Py_MOD_GIL` macros
- **Subinterpreters**: default Cython single-interpreter check applies
- **Module-state**: no `CYTHON_USE_MODULE_STATE` or `CYTHON_USE_TYPE_SPECS` overrides

## 8. What this orientation does NOT cover

- **AST scripts were not run** -- no Q1-Q5 JSON outputs exist. Manual analysis substituted but is less systematic.
- **No playbook file found** at expected location -- orientation produced from direct source analysis.
- **lib/src/*.c audit** -- the underlying C library is a separate scope; bugs there affect optv but require C-level review.
- **Refcount dataflow across .pyx module boundaries** -- e.g., `MatchedCoords` takes `TargetArray` and `Calibration` by reference; lifetime correctness depends on Python GC, not verified end-to-end.
- **Cython 3.2 vs 3.3+ idiom differences** -- not yet documented.

## 9. Triage hints for downstream agents

This is a medium-complexity Cython binding layer (2841 LOC across 11 modules) wrapping a C library. The generated C is 98.6% Cython runtime -- filter aggressively. Real bugs concentrate in **parameters.pyx** (907 LOC, 6 cdef classes, complex ownership) and **tracking_framebuf.pyx** (367 LOC, ownership-flag pattern). The universal use of `__init__` instead of `__cinit__` for C allocation is the main architectural concern but is low-risk given no subclassing. The `ControlParams`/`MultimediaParams` shared-pointer coupling (parameters.pyx:695-701) is the most fragile pattern. Nine .pyx files lack explicit `language_level=3` directives -- not a runtime bug with Cython 3.x defaults but a forward-compatibility risk.
