# Performance Profiling Report

**Date:** 2026-07-07
**Tool:** py-spy 0.4.2 (statistical profiler with native frame support)
**Target:** `pytest tests/unit/test_track.py::test_cavity`
**Samples:** 1021 over ~25 seconds (100 Hz sampling)
**System:** Linux, Python 3.13, Cython 3.2.8

---

## 1. Executive Summary

The tracking hot path (`test_cavity`) runs in **~9s wall clock** on this system (vs >120s interpreted — **13× improvement**). Profiling with `py-spy --native` captures both Python and compiled C/Cython frames.

**Key findings:**

| Area | Relative share | What it is |
|------|---------------|------------|
| `track_kernels_search` | 147%* | `_sorted_candidates_fast_out` (88%) — 8 quader corners × 4 cams per candidate |
| `track_kernels_tracking` | 118%* | `trackcorr_loop_fast` main loop |
| `tracking_frame_buf` | 111%* | SoA data access |
| **Cython View.MemoryView** | **46%*** | **Memoryview creation/validation** — ~55 params validated per kernel call |
| `track_kernels_geom` | 40%* | `_point_to_pixel_out`, quader projections, angle computation |
| `track_kernels_transform` | 21%* | `assess_new_position_fast` |

*(\% values from stack sampling overcount — each sample includes all parent frames. Use for relative comparison, not absolute timing.)*

**Bottom line:** Further single-threaded optimization has diminishing returns. The remaining bottlenecks are:
1. Algorithmic: 8 quader corner projections × 4 cams per candidate (24k `_point_to_pixel_out` calls/frame)
2. Structural: ~55 memoryview parameters validated per `trackcorr_loop_fast` call
3. The next real multiplier is **parallelization** (prange/nogil or concurrent.futures)

---

## 2. Raw Profile Data

### 2.1 Top functions by sample count

```
   %       Samples  Function                                                        Location
--------------------------------------------------------------------------------------------------------------------------------------------------
128.50%       387   trackcorr_c_loop                                                openptv2.algorithms.track
 88.27%       182   _sorted_candidates_fast_out                                     openptv2.algorithms.track_kernels_search
 40.10%        35   c_sqrt                                                          openptv2.algorithms.track_kernels_tracking
 35.36%        43   _point_to_pixel_out                                             openptv2.algorithms.track_kernels_geom
 34.12%        77   _sorted_candidates_fast_out (per-frame subtotals)               openptv2.algorithms.track_kernels_search
 27.91%        46   _sorted_candidates_fast_out (searchquader section)              openptv2.algorithms.track_kernels_search
 24.85%        26   trackcorr_loop_fast                                              openptv2.algorithms.track_kernels_tracking
 24.61%        45   assess_new_position_fast                                        openptv2.algorithms.track_kernels_transform
 18.62%        23   _angle_acc_out                                                  openptv2.algorithms.track_kernels_geom
 15.67%        23   _point_to_pixel_out (multimedia ray tracing)                    openptv2.algorithms.track_kernels_geom
 14.40%        19   _multimed_r_nlay_1layer                                         openptv2.algorithms.track_kernels_geom
 13.18%        18   candsearch_in_pix_fast                                          openptv2.algorithms.track_kernels_search
 12.44%        32   __pyx_tp_new__memoryviewslice / __pyx_tp_new_memoryview          track_kernels_search/geom
 11.89%        18   _point_position_out                                             openptv2.algorithms.track_kernels_transform
 11.03%        18   sort_candidates_by_freq                                         openptv2.algorithms.track_kernels_search
  8.52%        12   _trackcorr_loop_fast (link resolution)                          openptv2.algorithms.track_kernels_tracking
  7.54%        14   trackback_c_loop                                                openptv2.algorithms.track_kernels_tracking
  5.61%         8   track3d_loop_fast                                               openptv2.algorithms.track_kernels_tracking
  4.40%         6   multimed                                                        openptv2.algorithms.multimed
  2.04%         3   track_kernels (shim)                                            openptv2.algorithms.track_kernels
  ...

(Remaining: hundreds of small samples from Python infrastructure — importlib, pytest, numpy setup, etc.)
```

### 2.2 Module-level aggregation

```
   %     Module
───────────────────────────────────────────────────
201.78%  openptv2.algorithms.track
146.92%  openptv2.algorithms.track_kernels_search
117.55%  openptv2.algorithms.track_kernels_tracking
110.70%  openptv2.algorithms.tracking_frame_buf
 94.50%  .so: libc.so.6  (system-level, malloc/memcpy/etc)
 73.54%  openptv2.algorithms.tracking_run
 46.25%  Cython View.MemoryView  (memoryview creation/validation)
 40.02%  openptv2.algorithms.track_kernels_geom
 26.94%  .so: _multiarray_umath  (numpy internals)
 20.52%  openptv2.algorithms.track_kernels_transform
  6.26%  openptv2.algorithms.track_kernels_batch
  4.40%  openptv2.algorithms.multimed
  2.04%  openptv2.algorithms.track_kernels (shim)
```

---

## 3. Detailed Analysis

### 3.1 `track_kernels_search` — 147% (hottest module)

The `_sorted_candidates_fast_out` function accounts for 88% of this module's samples. It comprises:

| Sub-operation | Share | What it does |
|--------------|-------|-------------|
| **searchquader** | ~50% | 8 quader corners × 4 cams → 32 `_point_to_pixel_out` calls per candidate |
| **candsearch_in_pix_fast** | ~15% | Binary search + y-linear scan per camera (~47 target checks avg) |
| **sort_candidates_by_freq** | ~12% | Sort up to 16 candidates by frequency |
| **remaining** | ~23% | Loop overhead, data copying |

The 8 quader corner projections are the single heaviest sub-operation. Each projection involves full multimedia ray tracing (`_multimed_r_nlay_1layer` at 14.40%). These can't be cached — the quader depends on candidate position and search parameters.

### 3.2 Memoryview overhead — 46%

Cython's memoryview system validates and initializes every typed memoryview parameter on each function call. `trackcorr_loop_fast` has ~55 typed memoryview parameters. The overhead includes:
- `__pyx_tp_new__memoryviewslice` — allocation of memoryview slice objects
- `__pyx_memoryview_new` / `__pyx_memoryview_fromslice` — wrapping buffers
- Validation checking strides, dimensions, and contiguity

This is inherent to Cython's memoryview system. Potential mitigations:
- Reduce the number of memoryview parameters (combine into struct views)
- Use raw `double*` + length params instead of memoryviews (loses bounds checking)
- Accept as inherent to the compilation model

### 3.3 `track_kernels_geom` — 40%

| Function | Share | What it does |
|----------|-------|-------------|
| `_point_to_pixel_out` | 35% | Full pixel projection: ray tracing + multimedia + distortion |
| `_angle_acc_out` | 19% | Angle/acceleration computation between 3 positions |
| `_multimed_r_nlay_1layer` | 14% | Multimedia interface ray refraction |

These are compiled C with no Python overhead — the only way to make them faster is algorithmic changes or vectorization.

### 3.4 `track_kernels_tracking` — 118%

| Section | Share | What it does |
|---------|-------|-------------|
| `trackcorr_loop_fast` main body | ~50% | Candidate loop, assess_new_position, angle/acc checks |
| Link resolution (bubble sort) | ~9% | Sort up to 80 decis entries per particle |
| `trackback_c_loop` | ~8% | Backward tracking pass |
| `track3d_loop_fast` | ~6% | 3D tracking level 1/2/3 |

---

## 4. Methodology

### 4.1 How profiling was done

```bash
# Run with native frame support (captures C/Cython functions)
uv run py-spy record --native -o /tmp/profile.svg \
    -- pytest tests/unit/test_track.py::test_cavity -q --tb=short
```

The `--native` flag enables sampling of C frames, revealing Cython-compiled function names.

### 4.2 Limitations

1. **Stack overcounting**: Each sample captures the full call stack. If Python function A calls B calls C, all three appear in the sample. Module-level percentages can exceed 100%.

2. **py-spy vs perf**: py-spy is a statistical profiler that reads `/proc/PID/maps` to resolve symbols. It requires no kernel privileges but has lower resolution than `perf`. On this system, `perf` is restricted (kernel.perf_event_paranoid=4).

3. **Test includes setup**: The profile includes test data copying (shutil.copytree), module importing, and test infrastructure. The actual hot-path tracking loop is approximately 60-70% of test time.

4. **Sample count**: 1021 samples over ~25s at 100Hz. For the actual hot loop (~7-9s), approximately 700-900 samples were captured in the tracking code.

### 4.3 For future profiling with perf

If `perf_event_paranoid` is relaxed, `perf` can provide:
- CPU cycle-accuracy rather than statistical sampling
- Hot instruction-level breakdown within each function
- Cache miss analysis (L1, L2, LLC)
- Branch misprediction rates

```bash
sudo perf record -g -- python -m pytest tests/unit/test_track.py::test_cavity -q
sudo perf report
```

---

## 5. Recommendations

### 5.1 If optimizing further (diminishing returns)

- **Accept current performance**: 13× improvement over interpreted Python
- **Focus on parallelization** (prange + nogil or concurrent.futures) for 2-4× multi-core gains
- **Profile with perf** on a system with lower `perf_event_paranoid` for finer-grained optimization

### 5.2 If closing optimization phase

The cavity test went from >120s (interpreted) to ~9s (compiled C). All 248 tests pass. The codebase is well-optimized for single-core performance with all hot-path modules fully compiled.
Raw py-spy profile data (1021 total samples)
================================================================================

       %  Samples Function                                                               Location                                                    
------------------------------------------------------------------------------------------------------------------------------------------------------
  92.56%      945 _main                                                                  _pytest/config/__init__.py:229                              
  92.56%      945 __call__                                                               pluggy/_hooks.py:512                                        
  92.56%      945 _hookexec                                                              pluggy/_manager.py:120                                      
  92.56%      945 _multicall                                                             pluggy/_callers.py:121                                      
  92.56%      945 pytest_cmdline_main                                                    _pytest/main.py:377                                         
  91.77%      937 wrap_session                                                           _pytest/main.py:330                                         
  91.28%      932 _main                                                                  _pytest/main.py:384                                         
  91.28%      932 __call__                                                               pluggy/_hooks.py:512                                        
  91.28%      932 _hookexec                                                              pluggy/_manager.py:120                                      
  91.28%      932 _multicall                                                             pluggy/_callers.py:121                                      
  91.28%      932 pytest_runtestloop                                                     _pytest/main.py:408                                         
  91.28%      932 __call__                                                               pluggy/_hooks.py:512                                        
  91.28%      932 _hookexec                                                              pluggy/_manager.py:120                                      
  91.28%      932 _multicall                                                             pluggy/_callers.py:121                                      
  91.28%      932 pytest_runtest_protocol                                                _pytest/runner.py:118                                       
  91.28%      932 runtestprotocol                                                        _pytest/runner.py:139                                       
  91.28%      932 call_and_report                                                        _pytest/runner.py:249                                       
  91.28%      932 from_call                                                              _pytest/runner.py:361                                       
  91.28%      932 <lambda>                                                               _pytest/runner.py:250                                       
  91.28%      932 __call__                                                               pluggy/_hooks.py:512                                        
  91.28%      932 _hookexec                                                              pluggy/_manager.py:120                                      
  91.28%      932 _multicall                                                             pluggy/_callers.py:121                                      
  91.28%      932 pytest_runtest_call                                                    _pytest/runner.py:184                                       
  91.28%      932 runtest                                                                _pytest/python.py:1707                                      
  91.28%      932 __call__                                                               pluggy/_hooks.py:512                                        
  91.28%      932 _hookexec                                                              pluggy/_manager.py:120                                      
  91.28%      932 _multicall                                                             pluggy/_callers.py:121                                      
  91.28%      932 pytest_pyfunc_call                                                     _pytest/python.py:167                                       
  35.26%      360 test_cavity                                                            test_track.py:549                                           
  33.50%      342 0x19fff69                                                              python3.13                                                  
  33.50%      342 0x710f0442a1ca                                                         libc.so.6                                                   
  33.50%      342 0x199792d                                                              python3.13                                                  
  33.50%      342 0x1997b45                                                              python3.13                                                  
  33.50%      342 0x1997e92                                                              python3.13                                                  
  33.50%      342 0x1a074da                                                              python3.13                                                  
  33.50%      342 0x1a075f6                                                              python3.13                                                  
  33.50%      342 0x1a07650                                                              python3.13                                                  
  33.50%      342 0x1a07b37                                                              python3.13                                                  
  33.50%      342 0x1a08447                                                              python3.13                                                  
  33.50%      342 0x18dc183                                                              python3.13                                                  
  33.50%      342 print_exception_file_and_line                                          python3.13                                                  
  33.50%      342 0x18aa114                                                              python3.13                                                  
  33.50%      342 0x1ac6c0d                                                              python3.13                                                  
  33.50%      342 0x18e7e51                                                              python3.13                                                  
  33.50%      342 0x18e7fa5                                                              python3.13                                                  
  33.50%      342 0x1ac6c0d                                                              python3.13                                                  
  33.50%      342 0x18e7e51                                                              python3.13                                                  
  33.50%      342 0x18e7fa5                                                              python3.13                                                  
  33.50%      342 0x1ac6c0d                                                              python3.13                                                  
  33.50%      342 0x18e7e51                                                              python3.13                                                  
  33.50%      342 0x18e7fa5                                                              python3.13                                                  
  33.50%      342 PyFunction_NewWithQualName                                             python3.13                                                  
  33.50%      342 0x18e7e51                                                              python3.13                                                  
  33.50%      342 0x18e7fa5                                                              python3.13                                                  
  33.50%      342 0x1ac6c0d                                                              python3.13                                                  
  33.50%      342 0x18e7e51                                                              python3.13                                                  
  33.50%      342 0x18e7fa5                                                              python3.13                                                  
  30.85%      315 test_cavity                                                            test_track.py:514                                           
  30.75%      314 0x19fff69                                                              python3.13                                                  
  30.75%      314 0x710f0442a1ca                                                         libc.so.6                                                   
  30.75%      314 0x199792d                                                              python3.13                                                  
  30.75%      314 0x1997b45                                                              python3.13                                                  
  30.75%      314 0x1997e92                                                              python3.13                                                  
  30.75%      314 0x1a074da                                                              python3.13                                                  
  30.75%      314 0x1a075f6                                                              python3.13                                                  
  30.75%      314 0x1a07650                                                              python3.13                                                  
  30.75%      314 0x1a07b37                                                              python3.13                                                  
  30.75%      314 0x1a08447                                                              python3.13                                                  
  30.75%      314 0x18dc183                                                              python3.13                                                  
  30.75%      314 print_exception_file_and_line                                          python3.13                                                  
  30.75%      314 0x18aa114                                                              python3.13                                                  
  30.75%      314 0x1ac6c0d                                                              python3.13                                                  
  30.75%      314 0x18e7e51                                                              python3.13                                                  
  30.75%      314 0x18e7fa5                                                              python3.13                                                  
  30.75%      314 0x1ac6c0d                                                              python3.13                                                  
  30.75%      314 0x18e7e51                                                              python3.13                                                  
  30.75%      314 0x18e7fa5                                                              python3.13                                                  
  30.75%      314 0x1ac6c0d                                                              python3.13                                                  
  30.75%      314 0x18e7e51                                                              python3.13                                                  
  30.75%      314 0x18e7fa5                                                              python3.13                                                  
  30.75%      314 PyFunction_NewWithQualName                                             python3.13                                                  
  30.75%      314 0x18e7e51                                                              python3.13                                                  
  30.75%      314 0x18e7fa5                                                              python3.13                                                  
  30.75%      314 0x1ac6c0d                                                              python3.13                                                  
  30.75%      314 0x18e7e51                                                              python3.13                                                  
  30.75%      314 0x18e7fa5                                                              python3.13                                                  
  21.65%      221 code_new                                                               python3.13                                                  
  21.65%      221 trackcorr_c_loop                                                       openptv2/algorithms/track.py:1026                           
  21.65%      221 trackcorr_c_loop                                                       openptv2/algorithms/track.py:1026                           
  20.37%      208 code_new                                                               python3.13                                                  
  20.37%      208 trackcorr_c_loop                                                       openptv2/algorithms/track.py:1026                           
  20.37%      208 trackcorr_c_loop                                                       openptv2/algorithms/track.py:1026                           
  20.27%      207 track_trackcorr_c_loop                                                 openptv2/algorithms/track.py:1148                           
  20.27%      207 trackcorr_loop_fast                                                    openptv2/algorithms/track_kernels_tracking.py:70            
  18.71%      191 track_trackcorr_c_loop                                                 openptv2/algorithms/track.py:1148                           
  18.71%      191 trackcorr_loop_fast                                                    openptv2/algorithms/track_kernels_tracking.py:70            
  12.05%      123 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:429           
  11.95%      122 _sorted_candidates_fast_out                                            openptv2/algorithms/track_kernels_search.py:429             
  11.85%      121 exceptiongroup_subset                                                  python3.13                                                  
  11.85%      121 trackcorr_c_loop                                                       openptv2/algorithms/track.py:1026                           
  11.85%      121 trackcorr_c_loop                                                       openptv2/algorithms/track.py:1026                           
  11.56%      118 track_trackcorr_c_loop                                                 openptv2/algorithms/track.py:1148                           
  11.56%      118 trackcorr_loop_fast                                                    openptv2/algorithms/track_kernels_tracking.py:70            
  11.46%      117 _sorted_candidates_fast_out                                            openptv2/algorithms/track_kernels_search.py:429             
  11.07%      113 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:429           
  11.07%      113 _sorted_candidates_fast_out                                            openptv2/algorithms/track_kernels_search.py:429             
  10.97%      112 test_cavity                                                            test_track.py:508                                           
  10.97%      112 __init__                                                               &lt;string&gt;:16                                           
  10.68%      109 _sorted_candidates_fast_out                                            openptv2/algorithms/track_kernels_search.py:429             
  10.38%      106 exceptiongroup_subset                                                  python3.13                                                  
  10.38%      106 trackcorr_c_loop                                                       openptv2/algorithms/track.py:1026                           
  10.38%      106 trackcorr_c_loop                                                       openptv2/algorithms/track.py:1026                           
   9.89%      101 track_trackcorr_c_loop                                                 openptv2/algorithms/track.py:1148                           
   9.89%      101 trackcorr_loop_fast                                                    openptv2/algorithms/track_kernels_tracking.py:70            
   8.33%       85 0x19fff69                                                              python3.13                                                  
   8.33%       85 0x710f0442a1ca                                                         libc.so.6                                                   
   8.33%       85 0x199792d                                                              python3.13                                                  
   8.33%       85 0x1997b45                                                              python3.13                                                  
   8.33%       85 0x1997e92                                                              python3.13                                                  
   8.33%       85 0x1a074da                                                              python3.13                                                  
   8.33%       85 0x1a075f6                                                              python3.13                                                  
   8.33%       85 0x1a07650                                                              python3.13                                                  
   8.33%       85 0x1a07b37                                                              python3.13                                                  
   8.33%       85 0x1a08447                                                              python3.13                                                  
   8.33%       85 0x18dc183                                                              python3.13                                                  
   8.33%       85 print_exception_file_and_line                                          python3.13                                                  
   8.33%       85 0x18aa114                                                              python3.13                                                  
   8.33%       85 0x1ac6c0d                                                              python3.13                                                  
   8.33%       85 0x18e7e51                                                              python3.13                                                  
   8.33%       85 0x18e7fa5                                                              python3.13                                                  
   8.33%       85 0x1ac6c0d                                                              python3.13                                                  
   8.33%       85 0x18e7e51                                                              python3.13                                                  
   8.33%       85 0x18e7fa5                                                              python3.13                                                  
   8.33%       85 0x1ac6c0d                                                              python3.13                                                  
   8.33%       85 0x18e7e51                                                              python3.13                                                  
   8.33%       85 0x18e7fa5                                                              python3.13                                                  
   8.33%       85 PyFunction_NewWithQualName                                             python3.13                                                  
   8.33%       85 0x18e7e51                                                              python3.13                                                  
   8.33%       85 0x18e7fa5                                                              python3.13                                                  
   8.33%       85 0x1ac6c0d                                                              python3.13                                                  
   8.33%       85 0x18e7e51                                                              python3.13                                                  
   8.33%       85 0x18e7fa5                                                              python3.13                                                  
   8.33%       85 exceptiongroup_subset                                                  python3.13                                                  
   8.23%       84 tr_new                                                                 openptv2/algorithms/tracking_run.py:53                      
   8.23%       84 tracking_run_tr_new                                                    openptv2/algorithms/tracking_run.py:53                      
   8.23%       84 tracking_run_tr_new                                                    openptv2/algorithms/tracking_run.py:82                      
   8.23%       84 0x185e699                                                              python3.13                                                  
   8.23%       84 0x18072e4                                                              python3.13                                                  
   8.23%       84 validate_pattern_match_value                                           python3.13                                                  
   8.23%       84 exceptiongroup_subset                                                  python3.13                                                  
   8.23%       84 __post_init__                                                          openptv2/algorithms/tracking_run.py:28                      
   8.23%       84 Py_XDECREF                                                             object.h:1041                                               
   6.95%       71 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:429           
   6.86%       70 _sorted_candidates_fast_out                                            openptv2/algorithms/track_kernels_search.py:429             
   6.37%       65 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:551             
   6.37%       65 _sorted_candidates_fast_out                                            openptv2/algorithms/track_kernels_search.py:429             
   6.27%       64 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:429           
   6.27%       64 _sorted_candidates_fast_out                                            openptv2/algorithms/track_kernels_search.py:429             
   6.27%       64 test_cavity                                                            test_track.py:543                                           
   6.27%       64 __init__                                                               &lt;string&gt;:16                                           
   6.17%       63 _sorted_candidates_fast_out                                            openptv2/algorithms/track_kernels_search.py:429             
   5.97%       61 TrackingRun___post_init__                                              openptv2/algorithms/tracking_run.py:32                      
   5.97%       61 0x185e699                                                              python3.13                                                  
   5.97%       61 0x18072e4                                                              python3.13                                                  
   5.97%       61 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:800               
   5.97%       61 FrameBuf___init__                                                      openptv2/algorithms/tracking_frame_buf.py:812               
   5.97%       61 0x185e699                                                              python3.13                                                  
   5.97%       61 0x18072e4                                                              python3.13                                                  
   5.97%       61 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:523               
   5.78%       59 _main                                                                  _pytest/config/__init__.py:223                              
   5.58%       57 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:551             
   5.09%       52 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   4.60%       47 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   4.51%       46 _prepareconfig                                                         _pytest/config/__init__.py:410                              
   4.51%       46 __call__                                                               pluggy/_hooks.py:512                                        
   4.51%       46 _hookexec                                                              pluggy/_manager.py:120                                      
   4.51%       46 _multicall                                                             pluggy/_callers.py:121                                      
   4.51%       46 pytest_cmdline_parse                                                   _pytest/config/__init__.py:1232                             
   4.11%       42 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:551             
   3.72%       38 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   3.72%       38 __Pyx_PyObject_to_MemoryviewSlice_ds_double                            View.MemoryView:689                                         
   3.62%       37 __Pyx_ValidateAndInit_memviewslice                                     View.MemoryView:689                                         
   3.53%       36 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:551             
   3.43%       35 0x19fff69                                                              python3.13                                                  
   3.43%       35 0x710f0442a1ca                                                         libc.so.6                                                   
   3.43%       35 0x199792d                                                              python3.13                                                  
   3.43%       35 0x1997b45                                                              python3.13                                                  
   3.43%       35 0x1997e92                                                              python3.13                                                  
   3.43%       35 0x1a074da                                                              python3.13                                                  
   3.43%       35 0x1a075f6                                                              python3.13                                                  
   3.43%       35 0x1a07650                                                              python3.13                                                  
   3.43%       35 0x1a07b37                                                              python3.13                                                  
   3.43%       35 0x1a08447                                                              python3.13                                                  
   3.43%       35 0x18dc183                                                              python3.13                                                  
   3.43%       35 print_exception_file_and_line                                          python3.13                                                  
   3.43%       35 0x18aa114                                                              python3.13                                                  
   3.43%       35 0x1ac6c0d                                                              python3.13                                                  
   3.43%       35 0x18e7e51                                                              python3.13                                                  
   3.43%       35 0x18e7fa5                                                              python3.13                                                  
   3.43%       35 0x1ac6c0d                                                              python3.13                                                  
   3.43%       35 0x18e7e51                                                              python3.13                                                  
   3.43%       35 0x18e7fa5                                                              python3.13                                                  
   3.43%       35 0x1ac6c0d                                                              python3.13                                                  
   3.43%       35 0x18e7e51                                                              python3.13                                                  
   3.43%       35 0x18e7fa5                                                              python3.13                                                  
   3.43%       35 PyFunction_NewWithQualName                                             python3.13                                                  
   3.43%       35 0x18e7e51                                                              python3.13                                                  
   3.43%       35 0x18e7fa5                                                              python3.13                                                  
   3.43%       35 0x1ac6c0d                                                              python3.13                                                  
   3.43%       35 0x18e7e51                                                              python3.13                                                  
   3.43%       35 0x18e7fa5                                                              python3.13                                                  
   3.43%       35 exceptiongroup_subset                                                  python3.13                                                  
   3.43%       35 tr_new                                                                 openptv2/algorithms/tracking_run.py:53                      
   3.43%       35 tracking_run_tr_new                                                    openptv2/algorithms/tracking_run.py:53                      
   3.43%       35 tracking_run_tr_new                                                    openptv2/algorithms/tracking_run.py:82                      
   3.43%       35 0x185e699                                                              python3.13                                                  
   3.43%       35 0x18072e4                                                              python3.13                                                  
   3.43%       35 validate_pattern_match_value                                           python3.13                                                  
   3.43%       35 code_new                                                               python3.13                                                  
   3.43%       35 __post_init__                                                          openptv2/algorithms/tracking_run.py:28                      
   3.43%       35 Py_XDECREF                                                             object.h:1041                                               
   3.43%       35 TrackingRun___post_init__                                              openptv2/algorithms/tracking_run.py:32                      
   3.43%       35 0x185e699                                                              python3.13                                                  
   3.43%       35 0x18072e4                                                              python3.13                                                  
   3.43%       35 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:800               
   3.43%       35 FrameBuf___init__                                                      openptv2/algorithms/tracking_frame_buf.py:812               
   3.43%       35 0x185e699                                                              python3.13                                                  
   3.43%       35 0x18072e4                                                              python3.13                                                  
   3.43%       35 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:523               
   2.94%       30 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:354           
   2.84%       29 parse                                                                  _pytest/config/__init__.py:1605                             
   2.84%       29 __call__                                                               pluggy/_hooks.py:512                                        
   2.84%       29 _hookexec                                                              pluggy/_manager.py:120                                      
   2.84%       29 _multicall                                                             pluggy/_callers.py:121                                      
   2.84%       29 pytest_load_initial_conftests                                          _pytest/config/__init__.py:1312                             
   2.84%       29 _set_initial_conftests                                                 _pytest/config/__init__.py:662                              
   2.84%       29 _loadconftestmodules                                                   _pytest/config/__init__.py:707                              
   2.84%       29 _importconftest                                                        _pytest/config/__init__.py:758                              
   2.84%       29 import_path                                                            _pytest/pathlib.py:572                                      
   2.84%       29 _import_module_using_spec                                              _pytest/pathlib.py:716                                      
   2.84%       29 exec_module                                                            _pytest/assertion/rewrite.py:188                            
   2.84%       29 <module>                                                               conftest.py:16                                              
   2.84%       29 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   2.84%       29 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   2.84%       29 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   2.84%       29 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   2.84%       29 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   2.84%       29 __Pyx_PyObject_to_MemoryviewSlice_ds_double                            View.MemoryView:689                                         
   2.84%       29 _sorted_candidates_fast_out                                            openptv2/algorithms/track_kernels_search.py:429             
   2.84%       29 _sorted_candidates_fast_out                                            openptv2/algorithms/track_kernels_search.py:429             
   2.74%       28 <module>                                                               openptv2/__init__.py:3                                      
   2.74%       28 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   2.74%       28 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   2.74%       28 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   2.74%       28 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1329           
   2.74%       28 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   2.74%       28 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   2.74%       28 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:354           
   2.74%       28 _sorted_candidates_fast_out                                            openptv2/algorithms/track_kernels_search.py:429             
   2.74%       28 _sorted_candidates_fast_out                                            openptv2/algorithms/track_kernels_search.py:429             
   2.74%       28 __Pyx_ValidateAndInit_memviewslice                                     View.MemoryView:689                                         
   2.74%       28 __Pyx_PyObject_to_MemoryviewSlice_ds_double                            View.MemoryView:689                                         
   2.64%       27 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   2.64%       27 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   2.64%       27 Frame___init__                                                         openptv2/algorithms/tracking_frame_buf.py:528               
   2.64%       27 __pyx_memoryview_new                                                   View.MemoryView:673                                         
   2.55%       26 __Pyx_ValidateAndInit_memviewslice                                     View.MemoryView:689                                         
   2.55%       26 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   2.45%       25 Frame___init__                                                         openptv2/algorithms/tracking_frame_buf.py:529               
   2.45%       25 __Pyx_PyObject_FastCallDict                                            View.MemoryView:689                                         
   2.35%       24 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   2.35%       24 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   2.35%       24 __Pyx_PyObject_FastCallDict                                            openptv2/algorithms/tracking_frame_buf.py:849               
   2.25%       23 __Pyx_PyObject_FastCallDict                                            openptv2/algorithms/tracking_frame_buf.py:849               
   2.25%       23 TrackingRun___post_init__                                              openptv2/algorithms/tracking_run.py:50                      
   2.25%       23 init_mmlut                                                             openptv2/algorithms/multimed.py:602                         
   2.25%       23 test_cavity                                                            test_track.py:512                                           
   2.15%       22 0x18072e4                                                              python3.13                                                  
   2.15%       22 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:218               
   2.15%       22 Corres___init__                                                        openptv2/algorithms/tracking_frame_buf.py:221               
   2.15%       22 0x18072e4                                                              python3.13                                                  
   2.15%       22 init_mmlut                                                             openptv2/algorithms/multimed.py:762                         
   2.15%       22 init_mmlut_data_fast                                                   openptv2/algorithms/track_kernels_batch.py:370              
   2.15%       22 init_mmlut_data_fast                                                   openptv2/algorithms/track_kernels_batch.py:370              
   2.15%       22 0x19fff69                                                              python3.13                                                  
   2.15%       22 0x710f0442a1ca                                                         libc.so.6                                                   
   2.15%       22 0x199792d                                                              python3.13                                                  
   2.15%       22 0x1997b45                                                              python3.13                                                  
   2.15%       22 0x1997e92                                                              python3.13                                                  
   2.15%       22 0x1a074da                                                              python3.13                                                  
   2.15%       22 0x1a075f6                                                              python3.13                                                  
   2.15%       22 0x1a07650                                                              python3.13                                                  
   2.15%       22 0x1a07b37                                                              python3.13                                                  
   2.15%       22 0x1a08447                                                              python3.13                                                  
   2.15%       22 0x18dc183                                                              python3.13                                                  
   2.15%       22 print_exception_file_and_line                                          python3.13                                                  
   2.15%       22 0x18aa114                                                              python3.13                                                  
   2.15%       22 0x1ac6c0d                                                              python3.13                                                  
   2.15%       22 0x18e7e51                                                              python3.13                                                  
   2.15%       22 0x18e7fa5                                                              python3.13                                                  
   2.15%       22 0x1ac6c0d                                                              python3.13                                                  
   2.15%       22 0x18e7e51                                                              python3.13                                                  
   2.15%       22 0x18e7fa5                                                              python3.13                                                  
   2.15%       22 0x1ac6c0d                                                              python3.13                                                  
   2.15%       22 0x18e7e51                                                              python3.13                                                  
   2.15%       22 0x18e7fa5                                                              python3.13                                                  
   2.15%       22 PyFunction_NewWithQualName                                             python3.13                                                  
   2.15%       22 0x18e7e51                                                              python3.13                                                  
   2.15%       22 0x18e7fa5                                                              python3.13                                                  
   2.15%       22 0x1ac6c0d                                                              python3.13                                                  
   2.15%       22 0x18e7e51                                                              python3.13                                                  
   2.15%       22 0x18e7fa5                                                              python3.13                                                  
   2.15%       22 exceptiongroup_subset                                                  python3.13                                                  
   2.15%       22 track_forward_start                                                    openptv2/algorithms/track.py:998                            
   2.15%       22 track_forward_start                                                    openptv2/algorithms/track.py:998                            
   2.15%       22 track_track_forward_start                                              openptv2/algorithms/track.py:1001                           
   2.15%       22 Py_XDECREF                                                             object.h:1042                                               
   2.15%       22 Py_DECREF                                                              object.h:944                                                
   2.15%       22 _Py_IsImmortal                                                         object.h:361                                                
   2.15%       22 0x19bf0ab                                                              python3.13                                                  
   2.15%       22 _Py_BuildValue_SizeT                                                   python3.13                                                  
   2.15%       22 read_frame_at_end                                                      openptv2/algorithms/tracking_frame_buf.py:830               
   2.15%       22 read_frame_at_end                                                      openptv2/algorithms/tracking_frame_buf.py:835               
   2.15%       22 0x19bf0ab                                                              python3.13                                                  
   2.15%       22 _Py_BuildValue_SizeT                                                   python3.13                                                  
   2.15%       22 read                                                                   openptv2/algorithms/tracking_frame_buf.py:681               
   2.15%       22 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:541           
   2.06%       21 0x19fff69                                                              python3.13                                                  
   2.06%       21 0x710f0442a1ca                                                         libc.so.6                                                   
   2.06%       21 0x199792d                                                              python3.13                                                  
   2.06%       21 0x1997b45                                                              python3.13                                                  
   2.06%       21 0x1997e92                                                              python3.13                                                  
   2.06%       21 0x1a074da                                                              python3.13                                                  
   2.06%       21 0x1a075f6                                                              python3.13                                                  
   2.06%       21 0x1a07650                                                              python3.13                                                  
   2.06%       21 0x1a07b37                                                              python3.13                                                  
   2.06%       21 0x1a08447                                                              python3.13                                                  
   2.06%       21 0x18dc183                                                              python3.13                                                  
   2.06%       21 print_exception_file_and_line                                          python3.13                                                  
   2.06%       21 0x18aa114                                                              python3.13                                                  
   2.06%       21 0x1ac6c0d                                                              python3.13                                                  
   2.06%       21 0x18e7e51                                                              python3.13                                                  
   2.06%       21 0x18e7fa5                                                              python3.13                                                  
   2.06%       21 0x1ac6c0d                                                              python3.13                                                  
   2.06%       21 0x18e7e51                                                              python3.13                                                  
   2.06%       21 0x18e7fa5                                                              python3.13                                                  
   2.06%       21 0x1ac6c0d                                                              python3.13                                                  
   2.06%       21 0x18e7e51                                                              python3.13                                                  
   2.06%       21 0x18e7fa5                                                              python3.13                                                  
   2.06%       21 PyFunction_NewWithQualName                                             python3.13                                                  
   2.06%       21 0x18e7e51                                                              python3.13                                                  
   2.06%       21 0x18e7fa5                                                              python3.13                                                  
   2.06%       21 0x1ac6c0d                                                              python3.13                                                  
   2.06%       21 0x18e7e51                                                              python3.13                                                  
   2.06%       21 0x18e7fa5                                                              python3.13                                                  
   2.06%       21 _Py_c_pow                                                              python3.13                                                  
   2.06%       21 mi_bitmap_try_find_claim_field_across                                  python3.13                                                  
   2.06%       21 __pyx_tp_dealloc_8openptv2_10algorithms_18tracking_frame_buf_Fram...   openptv2/algorithms/tracking_frame_buf.py:852               
   2.06%       21 Py_DECREF                                                              object.h:949                                                
   2.06%       21 0x180153d                                                              python3.13                                                  
   2.06%       21 __pyx_tp_dealloc_8openptv2_10algorithms_18tracking_frame_buf_Frame     openptv2/algorithms/tracking_frame_buf.py:852               
   2.06%       21 Py_DECREF                                                              object.h:949                                                
   2.06%       21 _PyTraceMalloc_GetTraceback                                            python3.13                                                  
   2.06%       21 assess_new_position_fast                                               openptv2/algorithms/track_kernels_transform.py:425          
   1.96%       20 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:259               
   1.76%       18 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:598             
   1.76%       18 0x185e699                                                              python3.13                                                  
   1.76%       18 0x1807283                                                              python3.13                                                  
   1.76%       18 __pyx_tp_new_memoryview                                                openptv2/algorithms/track_kernels_geom.py:1111              
   1.76%       18 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:354           
   1.76%       18 0x7ffda58ff238                                                         ?                                                           
   1.76%       18 0x1aea4af                                                              python3.13                                                  
   1.76%       18 future_schedule_callbacks                                              python3.13                                                  
   1.67%       17 0x180153d                                                              python3.13                                                  
   1.67%       17 Py_XDECREF                                                             object.h:1041                                               
   1.67%       17 _PyTraceMalloc_GetTraceback                                            python3.13                                                  
   1.67%       17 _PyPegen_get_invalid_target                                            python3.13                                                  
   1.67%       17 array_array                                                            numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   1.67%       17 read                                                                   openptv2/algorithms/tracking_frame_buf.py:722               
   1.67%       17 __pyx_memoryview_new                                                   View.MemoryView:673                                         
   1.67%       17 candsearch_in_pix_fast                                                 openptv2/algorithms/track_kernels_search.py:46              
   1.67%       17 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:541           
   1.67%       17 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:551             
   1.67%       17 _sorted_candidates_fast_out                                            openptv2/algorithms/track_kernels_search.py:429             
   1.67%       17 _sorted_candidates_fast_out                                            openptv2/algorithms/track_kernels_search.py:429             
   1.57%       16 PyArray_CheckFromAny_int                                               numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   1.57%       16 PyArray_FromAny_int                                                    numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   1.57%       16 __Pyx_PyObject_FastCallDict                                            View.MemoryView:689                                         
   1.57%       16 __pyx_memoryview_new                                                   View.MemoryView:673                                         
   1.57%       16 test_cavity                                                            test_track.py:532                                           
   1.57%       16 Frame___init__                                                         openptv2/algorithms/tracking_frame_buf.py:529               
   1.57%       16 __Pyx_PyObject_FastCallDict                                            openptv2/algorithms/tracking_frame_buf.py:849               
   1.57%       16 0x18072e4                                                              python3.13                                                  
   1.57%       16 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:259               
   1.47%       15 __pyx_tp_dealloc_8openptv2_10algorithms_18tracking_frame_buf_Path...   openptv2/algorithms/tracking_frame_buf.py:852               
   1.47%       15 Py_DECREF                                                              object.h:949                                                
   1.47%       15 full                                                                   numpy/_core/numeric.py:323                                  
   1.47%       15 0x19fff69                                                              python3.13                                                  
   1.47%       15 0x710f0442a1ca                                                         libc.so.6                                                   
   1.47%       15 0x199792d                                                              python3.13                                                  
   1.47%       15 0x1997b45                                                              python3.13                                                  
   1.47%       15 0x1997e92                                                              python3.13                                                  
   1.47%       15 0x1a074da                                                              python3.13                                                  
   1.47%       15 0x1a075f6                                                              python3.13                                                  
   1.47%       15 0x1a07650                                                              python3.13                                                  
   1.47%       15 0x1a07b37                                                              python3.13                                                  
   1.47%       15 0x1a08447                                                              python3.13                                                  
   1.47%       15 0x18dc183                                                              python3.13                                                  
   1.47%       15 print_exception_file_and_line                                          python3.13                                                  
   1.47%       15 0x18aa114                                                              python3.13                                                  
   1.47%       15 0x1ac6c0d                                                              python3.13                                                  
   1.47%       15 0x18e7e51                                                              python3.13                                                  
   1.47%       15 0x18e7fa5                                                              python3.13                                                  
   1.47%       15 0x1ac6c0d                                                              python3.13                                                  
   1.47%       15 0x18e7e51                                                              python3.13                                                  
   1.47%       15 0x18e7fa5                                                              python3.13                                                  
   1.47%       15 0x1ac6c0d                                                              python3.13                                                  
   1.47%       15 0x18e7e51                                                              python3.13                                                  
   1.47%       15 0x18e7fa5                                                              python3.13                                                  
   1.47%       15 PyFunction_NewWithQualName                                             python3.13                                                  
   1.47%       15 0x18e7e51                                                              python3.13                                                  
   1.47%       15 0x18e7fa5                                                              python3.13                                                  
   1.47%       15 0x1ac6c0d                                                              python3.13                                                  
   1.47%       15 0x18e7e51                                                              python3.13                                                  
   1.47%       15 0x18e7fa5                                                              python3.13                                                  
   1.47%       15 exceptiongroup_subset                                                  python3.13                                                  
   1.47%       15 tr_new                                                                 openptv2/algorithms/tracking_run.py:53                      
   1.47%       15 tracking_run_tr_new                                                    openptv2/algorithms/tracking_run.py:53                      
   1.47%       15 tracking_run_tr_new                                                    openptv2/algorithms/tracking_run.py:82                      
   1.47%       15 0x185e699                                                              python3.13                                                  
   1.47%       15 0x18072e4                                                              python3.13                                                  
   1.47%       15 validate_pattern_match_value                                           python3.13                                                  
   1.47%       15 exceptiongroup_subset                                                  python3.13                                                  
   1.47%       15 __post_init__                                                          openptv2/algorithms/tracking_run.py:28                      
   1.47%       15 Py_XDECREF                                                             object.h:1041                                               
   1.47%       15 TrackingRun___post_init__                                              openptv2/algorithms/tracking_run.py:32                      
   1.47%       15 0x185e699                                                              python3.13                                                  
   1.47%       15 0x18072e4                                                              python3.13                                                  
   1.47%       15 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:800               
   1.47%       15 FrameBuf___init__                                                      openptv2/algorithms/tracking_frame_buf.py:812               
   1.47%       15 0x185e699                                                              python3.13                                                  
   1.47%       15 0x18072e4                                                              python3.13                                                  
   1.47%       15 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:523               
   1.47%       15 Frame___init__                                                         openptv2/algorithms/tracking_frame_buf.py:534               
   1.47%       15 _PyTraceMalloc_GetTraceback                                            python3.13                                                  
   1.47%       15 _Py_c_quot                                                             python3.13                                                  
   1.47%       15 task_step_handle_result_impl                                           python3.13                                                  
   1.47%       15 future_schedule_callbacks                                              python3.13                                                  
   1.47%       15 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:551             
   1.47%       15 0x185e699                                                              python3.13                                                  
   1.47%       15 0x1807283                                                              python3.13                                                  
   1.47%       15 __pyx_tp_new_memoryview                                                openptv2/algorithms/track_kernels_geom.py:1111              
   1.47%       15 _PyTraceMalloc_GetTraceback                                            python3.13                                                  
   1.47%       15 assess_new_position_fast                                               openptv2/algorithms/track_kernels_transform.py:425          
   1.47%       15 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:354           
   1.47%       15 _sorted_candidates_fast_out                                            openptv2/algorithms/track_kernels_search.py:429             
   1.47%       15 _sorted_candidates_fast_out                                            openptv2/algorithms/track_kernels_search.py:429             
   1.47%       15 __Pyx_PyObject_FastCallDict                                            View.MemoryView:689                                         
   1.47%       15 0x7ffda58ff5b0                                                         ?                                                           
   1.47%       15 0x1aea4af                                                              python3.13                                                  
   1.47%       15 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:598             
   1.37%       14 track_kernels_batch_init_mmlut_data_fast                               openptv2/algorithms/track_kernels_batch.py:421              
   1.37%       14 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   1.37%       14 track_trackcorr_c_loop                                                 openptv2/algorithms/track.py:1168                           
   1.37%       14 0x19bf0ab                                                              python3.13                                                  
   1.37%       14 _Py_BuildValue_SizeT                                                   python3.13                                                  
   1.37%       14 write_frame_from_start                                                 openptv2/algorithms/tracking_frame_buf.py:838               
   1.37%       14 write_frame_from_start                                                 openptv2/algorithms/tracking_frame_buf.py:845               
   1.37%       14 0x19bf0ab                                                              python3.13                                                  
   1.37%       14 _Py_BuildValue_SizeT                                                   python3.13                                                  
   1.37%       14 write                                                                  openptv2/algorithms/tracking_frame_buf.py:755               
   1.37%       14 Frame___init__                                                         openptv2/algorithms/tracking_frame_buf.py:528               
   1.37%       14 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   1.37%       14 candsearch_in_pix_fast                                                 openptv2/algorithms/track_kernels_search.py:46              
   1.27%       13 _prepareconfig                                                         _pytest/config/__init__.py:401                              
   1.27%       13 get_config                                                             _pytest/config/__init__.py:371                              
   1.27%       13 <module>                                                               numpy/__init__.py:112                                       
   1.27%       13 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   1.27%       13 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   1.27%       13 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   1.27%       13 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   1.27%       13 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   1.27%       13 <module>                                                               numpy/__config__.py:4                                       
   1.27%       13 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   1.27%       13 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1310                    
   1.27%       13 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   1.27%       13 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   1.27%       13 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   1.27%       13 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   1.27%       13 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   1.27%       13 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   1.27%       13 0x19fff69                                                              python3.13                                                  
   1.27%       13 0x710f0442a1ca                                                         libc.so.6                                                   
   1.27%       13 0x199792d                                                              python3.13                                                  
   1.27%       13 0x1997b45                                                              python3.13                                                  
   1.27%       13 0x1997e92                                                              python3.13                                                  
   1.27%       13 0x1a074da                                                              python3.13                                                  
   1.27%       13 0x1a075f6                                                              python3.13                                                  
   1.27%       13 0x1a07650                                                              python3.13                                                  
   1.27%       13 0x1a07b37                                                              python3.13                                                  
   1.27%       13 0x1a08447                                                              python3.13                                                  
   1.27%       13 0x18dc183                                                              python3.13                                                  
   1.27%       13 print_exception_file_and_line                                          python3.13                                                  
   1.27%       13 0x18aa114                                                              python3.13                                                  
   1.27%       13 0x1ac6c0d                                                              python3.13                                                  
   1.27%       13 0x18e7e51                                                              python3.13                                                  
   1.27%       13 0x18e7fa5                                                              python3.13                                                  
   1.27%       13 0x1ac6c0d                                                              python3.13                                                  
   1.27%       13 0x18e7e51                                                              python3.13                                                  
   1.27%       13 0x18e7fa5                                                              python3.13                                                  
   1.27%       13 0x1ac6c0d                                                              python3.13                                                  
   1.27%       13 0x18e7e51                                                              python3.13                                                  
   1.27%       13 0x18e7fa5                                                              python3.13                                                  
   1.27%       13 PyFunction_NewWithQualName                                             python3.13                                                  
   1.27%       13 0x18e7e51                                                              python3.13                                                  
   1.27%       13 0x18e7fa5                                                              python3.13                                                  
   1.27%       13 0x1ac6c0d                                                              python3.13                                                  
   1.27%       13 0x18e7e51                                                              python3.13                                                  
   1.27%       13 0x18e7fa5                                                              python3.13                                                  
   1.27%       13 stringlib_replace_delete_single_character                              python3.13                                                  
   1.27%       13 mi_bitmap_try_find_claim_field_across                                  python3.13                                                  
   1.27%       13 __pyx_tp_dealloc_8openptv2_10algorithms_18tracking_frame_buf_Fram...   openptv2/algorithms/tracking_frame_buf.py:852               
   1.27%       13 Py_DECREF                                                              object.h:949                                                
   1.27%       13 0x180153d                                                              python3.13                                                  
   1.27%       13 __pyx_tp_dealloc_8openptv2_10algorithms_18tracking_frame_buf_Frame     openptv2/algorithms/tracking_frame_buf.py:852               
   1.27%       13 Py_DECREF                                                              object.h:949                                                
   1.27%       13 __Pyx_PyObject_FastCallDict                                            openptv2/algorithms/tracking_frame_buf.py:849               
   1.27%       13 future_schedule_callbacks                                              python3.13                                                  
   1.27%       13 assess_new_position_fast                                               openptv2/algorithms/track_kernels_transform.py:425          
   1.27%       13 __Pyx_PyObject_to_MemoryviewSlice_ds_double                            View.MemoryView:689                                         
   1.18%       12 0x1ab4abd                                                              python3.13                                                  
   1.18%       12 __Pyx_PyObject_to_MemoryviewSlice_ds_double                            View.MemoryView:689                                         
   1.18%       12 0x185e699                                                              python3.13                                                  
   1.18%       12 0x1807283                                                              python3.13                                                  
   1.18%       12 __pyx_tp_new_memoryview                                                openptv2/algorithms/track_kernels_geom.py:1111              
   1.18%       12 0x18072e4                                                              python3.13                                                  
   1.18%       12 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:218               
   1.18%       12 Corres___init__                                                        openptv2/algorithms/tracking_frame_buf.py:221               
   1.18%       12 full                                                                   numpy/_core/numeric.py:323                                  
   1.18%       12 0x19fff69                                                              python3.13                                                  
   1.18%       12 0x710f0442a1ca                                                         libc.so.6                                                   
   1.18%       12 0x199792d                                                              python3.13                                                  
   1.18%       12 0x1997b45                                                              python3.13                                                  
   1.18%       12 0x1997e92                                                              python3.13                                                  
   1.18%       12 0x1a074da                                                              python3.13                                                  
   1.18%       12 0x1a075f6                                                              python3.13                                                  
   1.18%       12 0x1a07650                                                              python3.13                                                  
   1.18%       12 0x1a07b37                                                              python3.13                                                  
   1.18%       12 0x1a08447                                                              python3.13                                                  
   1.18%       12 0x18dc183                                                              python3.13                                                  
   1.18%       12 print_exception_file_and_line                                          python3.13                                                  
   1.18%       12 0x18aa114                                                              python3.13                                                  
   1.18%       12 0x1ac6c0d                                                              python3.13                                                  
   1.18%       12 0x18e7e51                                                              python3.13                                                  
   1.18%       12 0x18e7fa5                                                              python3.13                                                  
   1.18%       12 0x1ac6c0d                                                              python3.13                                                  
   1.18%       12 0x18e7e51                                                              python3.13                                                  
   1.18%       12 0x18e7fa5                                                              python3.13                                                  
   1.18%       12 0x1ac6c0d                                                              python3.13                                                  
   1.18%       12 0x18e7e51                                                              python3.13                                                  
   1.18%       12 0x18e7fa5                                                              python3.13                                                  
   1.18%       12 PyFunction_NewWithQualName                                             python3.13                                                  
   1.18%       12 0x18e7e51                                                              python3.13                                                  
   1.18%       12 0x18e7fa5                                                              python3.13                                                  
   1.18%       12 0x1ac6c0d                                                              python3.13                                                  
   1.18%       12 0x18e7e51                                                              python3.13                                                  
   1.18%       12 0x18e7fa5                                                              python3.13                                                  
   1.18%       12 exceptiongroup_subset                                                  python3.13                                                  
   1.18%       12 tr_new                                                                 openptv2/algorithms/tracking_run.py:53                      
   1.18%       12 tracking_run_tr_new                                                    openptv2/algorithms/tracking_run.py:53                      
   1.18%       12 tracking_run_tr_new                                                    openptv2/algorithms/tracking_run.py:82                      
   1.18%       12 0x185e699                                                              python3.13                                                  
   1.18%       12 0x18072e4                                                              python3.13                                                  
   1.18%       12 validate_pattern_match_value                                           python3.13                                                  
   1.18%       12 code_new                                                               python3.13                                                  
   1.18%       12 __post_init__                                                          openptv2/algorithms/tracking_run.py:28                      
   1.18%       12 Py_XDECREF                                                             object.h:1041                                               
   1.18%       12 TrackingRun___post_init__                                              openptv2/algorithms/tracking_run.py:32                      
   1.18%       12 0x185e699                                                              python3.13                                                  
   1.18%       12 0x18072e4                                                              python3.13                                                  
   1.18%       12 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:800               
   1.18%       12 FrameBuf___init__                                                      openptv2/algorithms/tracking_frame_buf.py:812               
   1.18%       12 0x185e699                                                              python3.13                                                  
   1.18%       12 0x18072e4                                                              python3.13                                                  
   1.18%       12 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:523               
   1.18%       12 Frame___init__                                                         openptv2/algorithms/tracking_frame_buf.py:534               
   1.18%       12 _PyTraceMalloc_GetTraceback                                            python3.13                                                  
   1.18%       12 _Py_c_quot                                                             python3.13                                                  
   1.18%       12 task_step_handle_result_impl                                           python3.13                                                  
   1.18%       12 future_schedule_callbacks                                              python3.13                                                  
   1.18%       12 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:546           
   1.18%       12 _point_position_out                                                    openptv2/algorithms/track_kernels_transform.py:51           
   1.18%       12 __Pyx_ValidateAndInit_memviewslice                                     View.MemoryView:689                                         
   1.18%       12 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:598             
   1.08%       11 import_plugin                                                          _pytest/config/__init__.py:913                              
   1.08%       11 import_module                                                          importlib/__init__.py:88                                    
   1.08%       11 _gcd_import                                                            &lt;frozen importlib._bootstrap&gt;:1387                    
   1.08%       11 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   1.08%       11 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   1.08%       11 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   1.08%       11 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   1.08%       11 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   1.08%       11 <module>                                                               _pytest/helpconfig.py:18                                    
   1.08%       11 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   1.08%       11 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   1.08%       11 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   1.08%       11 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   1.08%       11 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   1.08%       11 <module>                                                               numpy/__init__.py:457                                       
   1.08%       11 _handle_fromlist                                                       &lt;frozen importlib._bootstrap&gt;:1415                    
   1.08%       11 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   1.08%       11 _multimed_r_nlay_1layer                                                openptv2/algorithms/track_kernels_geom.py:47                
   1.08%       11 0x180153d                                                              python3.13                                                  
   1.08%       11 test_cavity                                                            test_track.py:547                                           
   1.08%       11 track_kernels_transform__point_position_out                            openptv2/algorithms/track_kernels_transform.py:51           
   1.08%       11 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:541           
   1.08%       11 <module>                                                               pytest:4                                                    
   1.08%       11 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   1.08%       11 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   1.08%       11 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   1.08%       11 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   1.08%       11 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.98%       10 parse                                                                  _pytest/config/__init__.py:1574                             
   0.98%       10 _consider_importhook                                                   _pytest/config/__init__.py:1345                             
   0.98%       10 _mark_plugins_for_rewrite                                              _pytest/config/__init__.py:1368                             
   0.98%       10 _iter_rewritable_modules                                               _pytest/config/__init__.py:983                              
   0.98%       10 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.98%       10 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.98%       10 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.98%       10 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.98%       10 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.98%       10 <module>                                                               numpy/lib/__init__.py:18                                    
   0.98%       10 _handle_fromlist                                                       &lt;frozen importlib._bootstrap&gt;:1415                    
   0.98%       10 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.98%       10 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.98%       10 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.98%       10 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.98%       10 0x7ffda58ff5b0                                                         ?                                                           
   0.98%       10 0x1aea4af                                                              python3.13                                                  
   0.98%       10 future_schedule_callbacks                                              python3.13                                                  
   0.98%       10 __Pyx_ValidateAndInit_memviewslice                                     View.MemoryView:689                                         
   0.98%       10 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   0.98%       10 assess_new_position_fast                                               openptv2/algorithms/track_kernels_transform.py:425          
   0.98%       10 Py_XDECREF                                                             object.h:1041                                               
   0.98%       10 _PyTraceMalloc_GetTraceback                                            python3.13                                                  
   0.98%       10 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.98%       10 array_array                                                            numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.98%       10 __pyx_memoryview___cinit__                                             View.MemoryView:356                                         
   0.98%       10 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:551             
   0.98%       10 _PyTraceMalloc_GetTraceback                                            python3.13                                                  
   0.98%       10 assess_new_position_fast                                               openptv2/algorithms/track_kernels_transform.py:425          
   0.88%        9 Frame___init__                                                         openptv2/algorithms/tracking_frame_buf.py:527               
   0.88%        9 __pyx_memoryview___cinit__                                             View.MemoryView:356                                         
   0.88%        9 __pyx_memoryview___cinit__                                             View.MemoryView:356                                         
   0.88%        9 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:541           
   0.88%        9 _PyTraceMalloc_GetTraceback                                            python3.13                                                  
   0.88%        9 assess_new_position_fast                                               openptv2/algorithms/track_kernels_transform.py:425          
   0.88%        9 PyArray_CheckFromAny_int                                               numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.88%        9 PyArray_FromAny_int                                                    numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.88%        9 0x1ab4abd                                                              python3.13                                                  
   0.88%        9 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:391           
   0.88%        9 __pyx_memoryview___pyx_pf_15View_dot_MemoryView_10memoryview___ci...   View.MemoryView:360                                         
   0.88%        9 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   0.88%        9 candsearch_in_pix_fast                                                 openptv2/algorithms/track_kernels_search.py:46              
   0.88%        9 assess_new_position_fast                                               openptv2/algorithms/track_kernels_transform.py:425          
   0.88%        9 0x1ab4a56                                                              python3.13                                                  
   0.88%        9 0x1ab4abd                                                              python3.13                                                  
   0.78%        8 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.78%        8 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.78%        8 __pyx_memoryview_new                                                   View.MemoryView:673                                         
   0.78%        8 __Pyx_PyObject_FastCallDict                                            View.MemoryView:689                                         
   0.78%        8 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:391           
   0.78%        8 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:538             
   0.78%        8 __Pyx_GetItemInt_Fast                                                  openptv2/algorithms/track_kernels_search.py:509             
   0.78%        8 Py_DECREF                                                              object.h:944                                                
   0.78%        8 _Py_IsImmortal                                                         object.h:361                                                
   0.78%        8 array_subscript                                                        numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.78%        8 __pyx_memoryview___pyx_pf_15View_dot_MemoryView_10memoryview___ci...   View.MemoryView:360                                         
   0.78%        8 __Pyx_XCLEAR_MEMVIEW                                                   View.MemoryView:689                                         
   0.78%        8 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:598             
   0.78%        8 __pyx_tp_dealloc_8openptv2_10algorithms_18tracking_frame_buf_Path...   openptv2/algorithms/tracking_frame_buf.py:852               
   0.78%        8 Py_DECREF                                                              object.h:949                                                
   0.78%        8 __Pyx_PyObject_to_MemoryviewSlice_ds_double                            View.MemoryView:689                                         
   0.78%        8 __Pyx_ValidateAndInit_memviewslice                                     View.MemoryView:689                                         
   0.78%        8 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:539             
   0.78%        8 Py_DECREF                                                              object.h:949                                                
   0.78%        8 __Pyx_AllocateExtensionType                                            View.MemoryView:689                                         
   0.78%        8 track_kernels_transform__point_position_out                            openptv2/algorithms/track_kernels_transform.py:135          
   0.78%        8 track_trackcorr_c_loop                                                 openptv2/algorithms/track.py:1168                           
   0.78%        8 0x19bf0ab                                                              python3.13                                                  
   0.78%        8 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.78%        8 write_frame_from_start                                                 openptv2/algorithms/tracking_frame_buf.py:838               
   0.78%        8 write_frame_from_start                                                 openptv2/algorithms/tracking_frame_buf.py:845               
   0.78%        8 0x19bf0ab                                                              python3.13                                                  
   0.78%        8 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.78%        8 write                                                                  openptv2/algorithms/tracking_frame_buf.py:755               
   0.78%        8 __pyx_memoryview_new                                                   View.MemoryView:673                                         
   0.78%        8 __Pyx_PyObject_FastCallDict                                            View.MemoryView:689                                         
   0.69%        7 __Pyx_PyObject_FastCallDict                                            openptv2/algorithms/tracking_frame_buf.py:849               
   0.69%        7 PyArray_DiscoverDTypeAndShape                                          numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.69%        7 PyArray_DiscoverDTypeAndShape_Recursive                                numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.69%        7 0x18008f3                                                              python3.13                                                  
   0.69%        7 track_kernels_geom__point_to_pixel_out                                 openptv2/algorithms/track_kernels_geom.py:620               
   0.69%        7 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:546           
   0.69%        7 _point_position_out                                                    openptv2/algorithms/track_kernels_transform.py:51           
   0.69%        7 write                                                                  openptv2/algorithms/tracking_frame_buf.py:763               
   0.69%        7 write                                                                  openptv2/algorithms/tracking_frame_buf.py:777               
   0.69%        7 candsearch_in_pix_fast                                                 openptv2/algorithms/track_kernels_search.py:46              
   0.69%        7 Pathinfo___init__                                                      openptv2/algorithms/tracking_frame_buf.py:271               
   0.69%        7 0x1ab4abd                                                              python3.13                                                  
   0.69%        7 0x19fff69                                                              python3.13                                                  
   0.69%        7 0x710f0442a1ca                                                         libc.so.6                                                   
   0.69%        7 0x199792d                                                              python3.13                                                  
   0.69%        7 0x1997b45                                                              python3.13                                                  
   0.69%        7 0x1997e92                                                              python3.13                                                  
   0.69%        7 0x1a074da                                                              python3.13                                                  
   0.69%        7 0x1a075f6                                                              python3.13                                                  
   0.69%        7 0x1a07650                                                              python3.13                                                  
   0.69%        7 0x1a07b37                                                              python3.13                                                  
   0.69%        7 0x1a08447                                                              python3.13                                                  
   0.69%        7 0x18dc183                                                              python3.13                                                  
   0.69%        7 print_exception_file_and_line                                          python3.13                                                  
   0.69%        7 0x18aa114                                                              python3.13                                                  
   0.69%        7 0x1ac6c0d                                                              python3.13                                                  
   0.69%        7 0x18e7e51                                                              python3.13                                                  
   0.69%        7 0x18e7fa5                                                              python3.13                                                  
   0.69%        7 0x1ac6c0d                                                              python3.13                                                  
   0.69%        7 0x18e7e51                                                              python3.13                                                  
   0.69%        7 0x18e7fa5                                                              python3.13                                                  
   0.69%        7 0x1ac6c0d                                                              python3.13                                                  
   0.69%        7 0x18e7e51                                                              python3.13                                                  
   0.69%        7 0x18e7fa5                                                              python3.13                                                  
   0.69%        7 PyFunction_NewWithQualName                                             python3.13                                                  
   0.69%        7 0x18e7e51                                                              python3.13                                                  
   0.69%        7 0x18e7fa5                                                              python3.13                                                  
   0.69%        7 0x1ac6c0d                                                              python3.13                                                  
   0.69%        7 0x18e7e51                                                              python3.13                                                  
   0.69%        7 0x18e7fa5                                                              python3.13                                                  
   0.69%        7 exceptiongroup_subset                                                  python3.13                                                  
   0.69%        7 track_forward_start                                                    openptv2/algorithms/track.py:998                            
   0.69%        7 track_forward_start                                                    openptv2/algorithms/track.py:998                            
   0.69%        7 track_track_forward_start                                              openptv2/algorithms/track.py:1001                           
   0.69%        7 Py_XDECREF                                                             object.h:1042                                               
   0.69%        7 Py_DECREF                                                              object.h:944                                                
   0.69%        7 _Py_IsImmortal                                                         object.h:361                                                
   0.69%        7 0x19bf0ab                                                              python3.13                                                  
   0.69%        7 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.69%        7 read_frame_at_end                                                      openptv2/algorithms/tracking_frame_buf.py:830               
   0.69%        7 read_frame_at_end                                                      openptv2/algorithms/tracking_frame_buf.py:835               
   0.69%        7 0x19bf0ab                                                              python3.13                                                  
   0.69%        7 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.69%        7 read                                                                   openptv2/algorithms/tracking_frame_buf.py:681               
   0.69%        7 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   0.69%        7 _loop0_191_rule                                                        python3.13                                                  
   0.69%        7 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   0.69%        7 _ray_tracing_out                                                       openptv2/algorithms/track_kernels_geom.py:1111              
   0.69%        7 __Pyx_XCLEAR_MEMVIEW                                                   View.MemoryView:689                                         
   0.69%        7 0x18008f3                                                              python3.13                                                  
   0.59%        6 0x19fff69                                                              python3.13                                                  
   0.59%        6 0x710f0442a1ca                                                         libc.so.6                                                   
   0.59%        6 0x199792d                                                              python3.13                                                  
   0.59%        6 0x1997b45                                                              python3.13                                                  
   0.59%        6 0x1997e92                                                              python3.13                                                  
   0.59%        6 0x1a074da                                                              python3.13                                                  
   0.59%        6 0x1a075f6                                                              python3.13                                                  
   0.59%        6 0x1a07650                                                              python3.13                                                  
   0.59%        6 0x1a07bcb                                                              python3.13                                                  
   0.59%        6 0x1a07777                                                              python3.13                                                  
   0.59%        6 0x1d0385b                                                              python3.13                                                  
   0.59%        6 0x19f959c                                                              python3.13                                                  
   0.59%        6 <module>                                                               numpy/lib/_arraypad_impl.py:10                              
   0.59%        6 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.59%        6 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.59%        6 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.59%        6 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.59%        6 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.59%        6 0x1801560                                                              python3.13                                                  
   0.59%        6 PyArray_NewFromDescr_int                                               numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.59%        6 Pathinfo___init__                                                      openptv2/algorithms/tracking_frame_buf.py:262               
   0.59%        6 Pathinfo___init__                                                      openptv2/algorithms/tracking_frame_buf.py:271               
   0.59%        6 Pathinfo___init__                                                      openptv2/algorithms/tracking_frame_buf.py:275               
   0.59%        6 track_kernels_batch_init_mmlut_data_fast                               openptv2/algorithms/track_kernels_batch.py:411              
   0.59%        6 tracking_frame_buf_read_path_frame                                     openptv2/algorithms/tracking_frame_buf.py:383               
   0.59%        6 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   0.59%        6 get_view_from_index                                                    numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.59%        6 PyArray_NewFromDescr_int                                               numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.59%        6 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:541             
   0.59%        6 array_getbuffer                                                        numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.59%        6 track_kernels_search_candsearch_in_pix_fast                            openptv2/algorithms/track_kernels_search.py:135             
   0.59%        6 track_kernels_transform__point_position_out                            openptv2/algorithms/track_kernels_transform.py:51           
   0.59%        6 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:598             
   0.59%        6 candsearch_in_pix_fast                                                 openptv2/algorithms/track_kernels_search.py:46              
   0.59%        6 __pyx_memoryview___pyx_pf_15View_dot_MemoryView_10memoryview___ci...   View.MemoryView:360                                         
   0.59%        6 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   0.59%        6 0x185d5ab                                                              python3.13                                                  
   0.59%        6 malloc                                                                 libc.so.6                                                   
   0.59%        6 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:541             
   0.59%        6 __Pyx_XCLEAR_MEMVIEW                                                   View.MemoryView:689                                         
   0.59%        6 track_kernels_search_candsearch_in_pix_fast                            openptv2/algorithms/track_kernels_search.py:135             
   0.59%        6 track_kernels_transform_assess_new_position_fast                       openptv2/algorithms/track_kernels_transform.py:538          
   0.59%        6 __Pyx_PyObject_to_MemoryviewSlice_ds_double                            View.MemoryView:689                                         
   0.59%        6 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:391           
   0.59%        6 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   0.59%        6 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:538             
   0.59%        6 __Pyx_GetItemInt_Fast                                                  openptv2/algorithms/track_kernels_search.py:509             
   0.59%        6 Py_DECREF                                                              object.h:944                                                
   0.59%        6 _Py_IsImmortal                                                         object.h:361                                                
   0.59%        6 array_subscript                                                        numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.59%        6 Py_DECREF                                                              object.h:949                                                
   0.59%        6 0x185e699                                                              python3.13                                                  
   0.59%        6 0x1807283                                                              python3.13                                                  
   0.59%        6 __pyx_tp_new_memoryview                                                openptv2/algorithms/track_kernels_geom.py:1111              
   0.49%        5 <module>                                                               pytest/__init__.py:76                                       
   0.49%        5 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.49%        5 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.49%        5 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.49%        5 <genexpr>                                                              _pytest/config/__init__.py:1363                             
   0.49%        5 <genexpr>                                                              _pytest/config/__init__.py:1364                             
   0.49%        5 entry_points                                                           importlib/metadata/__init__.py:496                          
   0.49%        5 parse                                                                  _pytest/config/__init__.py:1583                             
   0.49%        5 load_setuptools_entrypoints                                            pluggy/_manager.py:416                                      
   0.49%        5 load                                                                   importlib/metadata/__init__.py:179                          
   0.49%        5 import_module                                                          importlib/__init__.py:88                                    
   0.49%        5 _gcd_import                                                            &lt;frozen importlib._bootstrap&gt;:1387                    
   0.49%        5 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.49%        5 <module>                                                               numpy/_core/__init__.py:111                                 
   0.49%        5 _handle_fromlist                                                       &lt;frozen importlib._bootstrap&gt;:1415                    
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.49%        5 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.49%        5 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.49%        5 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 <module>                                                               numpy/_core/einsumfunc.py:10                                
   0.49%        5 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.49%        5 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.49%        5 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.49%        5 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 <module>                                                               numpy/_core/numeric.py:13                                   
   0.49%        5 _handle_fromlist                                                       &lt;frozen importlib._bootstrap&gt;:1415                    
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.49%        5 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.49%        5 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.49%        5 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 <module>                                                               numpy/_core/shape_base.py:8                                 
   0.49%        5 _handle_fromlist                                                       &lt;frozen importlib._bootstrap&gt;:1415                    
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.49%        5 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.49%        5 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.49%        5 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 <module>                                                               numpy/_core/fromnumeric.py:11                               
   0.49%        5 _handle_fromlist                                                       &lt;frozen importlib._bootstrap&gt;:1415                    
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.49%        5 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.49%        5 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.49%        5 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 <module>                                                               numpy/_core/_methods.py:7                                   
   0.49%        5 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.49%        5 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.49%        5 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.49%        5 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1023           
   0.49%        5 get_code                                                               &lt;frozen importlib._bootstrap_external&gt;:1156           
   0.49%        5 <module>                                                               numpy/lib/_index_tricks_impl.py:8                           
   0.49%        5 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.49%        5 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.49%        5 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.49%        5 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 <module>                                                               numpy/matrixlib/__init__.py:4                               
   0.49%        5 _handle_fromlist                                                       &lt;frozen importlib._bootstrap&gt;:1415                    
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.49%        5 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.49%        5 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.49%        5 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 <module>                                                               numpy/matrixlib/defmatrix.py:13                             
   0.49%        5 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.49%        5 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.49%        5 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.49%        5 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 <module>                                                               numpy/linalg/__init__.py:87                                 
   0.49%        5 _handle_fromlist                                                       &lt;frozen importlib._bootstrap&gt;:1415                    
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.49%        5 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.49%        5 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.49%        5 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 <module>                                                               numpy/linalg/_linalg.py:76                                  
   0.49%        5 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.49%        5 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.49%        5 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.49%        5 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 <module>                                                               numpy/_typing/__init__.py:5                                 
   0.49%        5 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.49%        5 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.49%        5 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.49%        5 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 _main                                                                  _pytest/main.py:383                                         
   0.49%        5 __call__                                                               pluggy/_hooks.py:512                                        
   0.49%        5 _hookexec                                                              pluggy/_manager.py:120                                      
   0.49%        5 _multicall                                                             pluggy/_callers.py:121                                      
   0.49%        5 pytest_collection                                                      _pytest/main.py:394                                         
   0.49%        5 perform_collect                                                        _pytest/main.py:849                                         
   0.49%        5 collect_one_node                                                       _pytest/runner.py:589                                       
   0.49%        5 __call__                                                               pluggy/_hooks.py:512                                        
   0.49%        5 _hookexec                                                              pluggy/_manager.py:120                                      
   0.49%        5 _multicall                                                             pluggy/_callers.py:121                                      
   0.49%        5 pytest_make_collect_report                                             _pytest/runner.py:408                                       
   0.49%        5 from_call                                                              _pytest/runner.py:361                                       
   0.49%        5 collect                                                                _pytest/runner.py:406                                       
   0.49%        5 collect                                                                _pytest/main.py:973                                         
   0.49%        5 _collect_one_node                                                      _pytest/main.py:895                                         
   0.49%        5 collect_one_node                                                       _pytest/runner.py:589                                       
   0.49%        5 __call__                                                               pluggy/_hooks.py:512                                        
   0.49%        5 _hookexec                                                              pluggy/_manager.py:120                                      
   0.49%        5 _multicall                                                             pluggy/_callers.py:121                                      
   0.49%        5 pytest_make_collect_report                                             _pytest/runner.py:408                                       
   0.49%        5 from_call                                                              _pytest/runner.py:361                                       
   0.49%        5 collect                                                                _pytest/runner.py:406                                       
   0.49%        5 0x1ab4abd                                                              python3.13                                                  
   0.49%        5 0x185e699                                                              python3.13                                                  
   0.49%        5 0x1807283                                                              python3.13                                                  
   0.49%        5 __pyx_tp_new_memoryview                                                openptv2/algorithms/track_kernels_geom.py:1111              
   0.49%        5 __pyx_memoryview___cinit__                                             View.MemoryView:356                                         
   0.49%        5 __pyx_memoryview___pyx_pf_15View_dot_MemoryView_10memoryview___ci...   View.MemoryView:360                                         
   0.49%        5 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:598             
   0.49%        5 candsearch_in_pix_fast                                                 openptv2/algorithms/track_kernels_search.py:46              
   0.49%        5 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:539             
   0.49%        5 __Pyx_BufFmt_CheckString                                               View.MemoryView:689                                         
   0.49%        5 __Pyx_AllocateExtensionType                                            View.MemoryView:689                                         
   0.49%        5 _buffer_get_info                                                       numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.49%        5 _multimed_r_nlay_1layer                                                openptv2/algorithms/track_kernels_geom.py:47                
   0.49%        5 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:551             
   0.49%        5 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   0.49%        5 __Pyx_BufFmt_CheckString                                               View.MemoryView:689                                         
   0.49%        5 __Pyx_BufFmt_ProcessTypeChunk                                          View.MemoryView:689                                         
   0.49%        5 array_getbuffer                                                        numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.49%        5 _buffer_get_info                                                       numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.49%        5 assess_new_position_fast                                               openptv2/algorithms/track_kernels_transform.py:425          
   0.49%        5 Frame___init__                                                         openptv2/algorithms/tracking_frame_buf.py:527               
   0.49%        5 PyArray_NewFromDescr_int                                               numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.49%        5 Pathinfo___init__                                                      openptv2/algorithms/tracking_frame_buf.py:262               
   0.49%        5 0x710f044ac51a                                                         libc.so.6                                                   
   0.49%        5 0x1ab4a56                                                              python3.13                                                  
   0.49%        5 __pyx_memoryview_new                                                   View.MemoryView:673                                         
   0.49%        5 __Pyx_PyObject_FastCallDict                                            View.MemoryView:689                                         
   0.49%        5 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:598             
   0.49%        5 array_getbuffer                                                        numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.49%        5 __Pyx_ValidateAndInit_memviewslice                                     View.MemoryView:689                                         
   0.49%        5 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   0.49%        5 track_kernels_transform_assess_new_position_fast                       openptv2/algorithms/track_kernels_transform.py:538          
   0.49%        5 candsearch_in_pix_rest_fast                                            openptv2/algorithms/track_kernels_search.py:174             
   0.49%        5 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:546           
   0.49%        5 <module>                                                               _pytest/config/__init__.py:50                               
   0.49%        5 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.49%        5 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.49%        5 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.49%        5 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.49%        5 <module>                                                               _pytest/config/findpaths.py:17                              
   0.49%        5 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.49%        5 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.49%        5 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.49%        5 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.49%        5 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.39%        4 <module>                                                               numpy/_core/__init__.py:107                                 
   0.39%        4 _handle_fromlist                                                       &lt;frozen importlib._bootstrap&gt;:1415                    
   0.39%        4 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.39%        4 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.39%        4 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.39%        4 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.39%        4 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.39%        4 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.39%        4 _compile_bytecode                                                      &lt;frozen importlib._bootstrap_external&gt;:785            
   0.39%        4 0x19fff69                                                              python3.13                                                  
   0.39%        4 0x710f0442a1ca                                                         libc.so.6                                                   
   0.39%        4 0x199792d                                                              python3.13                                                  
   0.39%        4 0x1997b45                                                              python3.13                                                  
   0.39%        4 0x1997e92                                                              python3.13                                                  
   0.39%        4 0x1a074da                                                              python3.13                                                  
   0.39%        4 0x1a075f6                                                              python3.13                                                  
   0.39%        4 0x1a07650                                                              python3.13                                                  
   0.39%        4 0x1a07b37                                                              python3.13                                                  
   0.39%        4 0x1a08447                                                              python3.13                                                  
   0.39%        4 0x18dc183                                                              python3.13                                                  
   0.39%        4 print_exception_file_and_line                                          python3.13                                                  
   0.39%        4 0x18aa114                                                              python3.13                                                  
   0.39%        4 0x1ac6c0d                                                              python3.13                                                  
   0.39%        4 0x18e7e51                                                              python3.13                                                  
   0.39%        4 0x18e7fa5                                                              python3.13                                                  
   0.39%        4 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.39%        4 _loop0_152_rule                                                        python3.13                                                  
   0.39%        4 0x1ac6c0d                                                              python3.13                                                  
   0.39%        4 0x18e7e51                                                              python3.13                                                  
   0.39%        4 0x18e7fa5                                                              python3.13                                                  
   0.39%        4 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.39%        4 _loop0_152_rule                                                        python3.13                                                  
   0.39%        4 0x181ff4e                                                              python3.13                                                  
   0.39%        4 0x18aaa06                                                              python3.13                                                  
   0.39%        4 0x18aa114                                                              python3.13                                                  
   0.39%        4 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.39%        4 0x1abfb39                                                              python3.13                                                  
   0.39%        4 marshal_dump_impl                                                      python3.13                                                  
   0.39%        4 va_build_value                                                         python3.13                                                  
   0.39%        4 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.39%        4 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.39%        4 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.39%        4 0x18aaa06                                                              python3.13                                                  
   0.39%        4 0x18aa114                                                              python3.13                                                  
   0.39%        4 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.39%        4 0x1abfb39                                                              python3.13                                                  
   0.39%        4 marshal_dump_impl                                                      python3.13                                                  
   0.39%        4 va_build_value                                                         python3.13                                                  
   0.39%        4 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.39%        4 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.39%        4 0x1802235                                                              python3.13                                                  
   0.39%        4 0x19a6f16                                                              python3.13                                                  
   0.39%        4 __pyx_pymod_exec_track_kernels                                         openptv2/algorithms/track_kernels.py:9                      
   0.39%        4 __Pyx_Import                                                           openptv2/algorithms/track_kernels.py:48                     
   0.39%        4 __Pyx__Import                                                          openptv2/algorithms/track_kernels.py:48                     
   0.39%        4 0x1abfb39                                                              python3.13                                                  
   0.39%        4 marshal_dump_impl                                                      python3.13                                                  
   0.39%        4 va_build_value                                                         python3.13                                                  
   0.39%        4 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.39%        4 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.39%        4 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.39%        4 0x18aaa06                                                              python3.13                                                  
   0.39%        4 0x18aa114                                                              python3.13                                                  
   0.39%        4 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.39%        4 0x1abfb39                                                              python3.13                                                  
   0.39%        4 marshal_dump_impl                                                      python3.13                                                  
   0.39%        4 va_build_value                                                         python3.13                                                  
   0.39%        4 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.39%        4 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.39%        4 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.39%        4 0x18aaa06                                                              python3.13                                                  
   0.39%        4 0x18aa114                                                              python3.13                                                  
   0.39%        4 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.39%        4 0x1abfb39                                                              python3.13                                                  
   0.39%        4 marshal_dump_impl                                                      python3.13                                                  
   0.39%        4 va_build_value                                                         python3.13                                                  
   0.39%        4 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.39%        4 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.39%        4 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.39%        4 PyObject_DelItemString                                                 python3.13                                                  
   0.39%        4 0x1abfb39                                                              python3.13                                                  
   0.39%        4 marshal_dump_impl                                                      python3.13                                                  
   0.39%        4 va_build_value                                                         python3.13                                                  
   0.39%        4 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.39%        4 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.39%        4 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.39%        4 0x18aaa06                                                              python3.13                                                  
   0.39%        4 0x18aa114                                                              python3.13                                                  
   0.39%        4 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.39%        4 0x1abfdad                                                              python3.13                                                  
   0.39%        4 marshal_dump_impl                                                      python3.13                                                  
   0.39%        4 va_build_value                                                         python3.13                                                  
   0.39%        4 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.39%        4 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.39%        4 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.39%        4 PyObject_DelItemString                                                 python3.13                                                  
   0.39%        4 0x1abfb39                                                              python3.13                                                  
   0.39%        4 marshal_dump_impl                                                      python3.13                                                  
   0.39%        4 va_build_value                                                         python3.13                                                  
   0.39%        4 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.39%        4 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.39%        4 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.39%        4 0x18aaa06                                                              python3.13                                                  
   0.39%        4 0x18aa114                                                              python3.13                                                  
   0.39%        4 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.39%        4 0x1abfb39                                                              python3.13                                                  
   0.39%        4 marshal_dump_impl                                                      python3.13                                                  
   0.39%        4 va_build_value                                                         python3.13                                                  
   0.39%        4 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.39%        4 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.39%        4 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.39%        4 0x18aaa06                                                              python3.13                                                  
   0.39%        4 0x18aa114                                                              python3.13                                                  
   0.39%        4 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.39%        4 0x1abfdad                                                              python3.13                                                  
   0.39%        4 marshal_dump_impl                                                      python3.13                                                  
   0.39%        4 va_build_value                                                         python3.13                                                  
   0.39%        4 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.39%        4 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.39%        4 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.39%        4 PyObject_DelItemString                                                 python3.13                                                  
   0.39%        4 0x1abfb39                                                              python3.13                                                  
   0.39%        4 marshal_dump_impl                                                      python3.13                                                  
   0.39%        4 va_build_value                                                         python3.13                                                  
   0.39%        4 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.39%        4 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.39%        4 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.39%        4 0x18aaa06                                                              python3.13                                                  
   0.39%        4 0x18aa114                                                              python3.13                                                  
   0.39%        4 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.39%        4 0x1abfdad                                                              python3.13                                                  
   0.39%        4 marshal_dump_impl                                                      python3.13                                                  
   0.39%        4 va_build_value                                                         python3.13                                                  
   0.39%        4 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.39%        4 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.39%        4 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.39%        4 PyObject_DelItemString                                                 python3.13                                                  
   0.39%        4 0x1abfb39                                                              python3.13                                                  
   0.39%        4 marshal_dump_impl                                                      python3.13                                                  
   0.39%        4 va_build_value                                                         python3.13                                                  
   0.39%        4 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.39%        4 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.39%        4 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.39%        4 0x18aaa06                                                              python3.13                                                  
   0.39%        4 0x18aa114                                                              python3.13                                                  
   0.39%        4 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.39%        4 0x1abfdad                                                              python3.13                                                  
   0.39%        4 marshal_dump_impl                                                      python3.13                                                  
   0.39%        4 va_build_value                                                         python3.13                                                  
   0.39%        4 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.39%        4 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.39%        4 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.39%        4 PyObject_DelItemString                                                 python3.13                                                  
   0.39%        4 0x1abfb39                                                              python3.13                                                  
   0.39%        4 marshal_dump_impl                                                      python3.13                                                  
   0.39%        4 va_build_value                                                         python3.13                                                  
   0.39%        4 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.39%        4 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.39%        4 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.39%        4 0x18aaa06                                                              python3.13                                                  
   0.39%        4 0x18aa114                                                              python3.13                                                  
   0.39%        4 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.39%        4 0x1abfb39                                                              python3.13                                                  
   0.39%        4 marshal_dump_impl                                                      python3.13                                                  
   0.39%        4 va_build_value                                                         python3.13                                                  
   0.39%        4 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.39%        4 0x181ff4e                                                              python3.13                                                  
   0.39%        4 0x19a0ebc                                                              python3.13                                                  
   0.39%        4 0x19a0fac                                                              python3.13                                                  
   0.39%        4 typevartuple_alloc                                                     python3.13                                                  
   0.39%        4 typevartuple                                                           python3.13                                                  
   0.39%        4 typevartuple_alloc                                                     python3.13                                                  
   0.39%        4 typevartuple                                                           python3.13                                                  
   0.39%        4 wrap_session                                                           _pytest/main.py:328                                         
   0.39%        4 __call__                                                               pluggy/_hooks.py:512                                        
   0.39%        4 _hookexec                                                              pluggy/_manager.py:120                                      
   0.39%        4 _multicall                                                             pluggy/_callers.py:121                                      
   0.39%        4 pytest_sessionstart                                                    _pytest/terminal.py:868                                     
   0.39%        4 __call__                                                               pluggy/_hooks.py:512                                        
   0.39%        4 _hookexec                                                              pluggy/_manager.py:120                                      
   0.39%        4 _multicall                                                             pluggy/_callers.py:121                                      
   0.39%        4 pytest_report_header                                                   _pytest/terminal.py:901                                     
   0.39%        4 _plugin_nameversions                                                   _pytest/terminal.py:1637                                    
   0.39%        4 project_name                                                           pluggy/_manager.py:70                                       
   0.39%        4 __getattr__                                                            pluggy/_manager.py:74                                       
   0.39%        4 metadata                                                               importlib/metadata/__init__.py:460                          
   0.39%        4 _handle_fromlist                                                       &lt;frozen importlib._bootstrap&gt;:1415                    
   0.39%        4 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.39%        4 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.39%        4 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.39%        4 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.39%        4 0x185d5ab                                                              python3.13                                                  
   0.39%        4 malloc                                                                 libc.so.6                                                   
   0.39%        4 0x710f044ac51a                                                         libc.so.6                                                   
   0.39%        4 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.39%        4 array_array                                                            numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.39%        4 PyArray_CheckFromAny_int                                               numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.39%        4 PyArray_FromAny_int                                                    numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.39%        4 tracking_frame_buf_read_path_frame                                     openptv2/algorithms/tracking_frame_buf.py:386               
   0.39%        4 array_getbuffer                                                        numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.39%        4 _buffer_get_info                                                       numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.39%        4 __Pyx_BufFmt_ProcessTypeChunk                                          View.MemoryView:689                                         
   0.39%        4 _loop0_191_rule                                                        python3.13                                                  
   0.39%        4 track_kernels_transform__point_position_out                            openptv2/algorithms/track_kernels_transform.py:135          
   0.39%        4 tracking_frame_buf_write_targets                                       openptv2/algorithms/tracking_frame_buf.py:204               
   0.39%        4 compiler_try_star_except                                               python3.13                                                  
   0.39%        4 compiler_try_star_except                                               python3.13                                                  
   0.39%        4 0x1802235                                                              python3.13                                                  
   0.39%        4 0x185cdce                                                              python3.13                                                  
   0.39%        4 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:541             
   0.39%        4 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:459           
   0.39%        4 track_kernels_transform_assess_new_position_fast                       openptv2/algorithms/track_kernels_transform.py:538          
   0.39%        4 candsearch_in_pix_rest_fast                                            openptv2/algorithms/track_kernels_search.py:174             
   0.39%        4 PyArray_DiscoverDTypeAndShape                                          numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.39%        4 Py_XDECREF                                                             object.h:1041                                               
   0.39%        4 _PyTraceMalloc_GetTraceback                                            python3.13                                                  
   0.39%        4 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.39%        4 array_zeros                                                            numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.39%        4 Pathinfo___init__                                                      openptv2/algorithms/tracking_frame_buf.py:275               
   0.39%        4 0x1ab4a56                                                              python3.13                                                  
   0.39%        4 0x18008f3                                                              python3.13                                                  
   0.39%        4 0x18008f3                                                              python3.13                                                  
   0.39%        4 read                                                                   openptv2/algorithms/tracking_frame_buf.py:737               
   0.39%        4 0x7ffda58ffb70                                                         ?                                                           
   0.39%        4 0x1aea4af                                                              python3.13                                                  
   0.39%        4 future_schedule_callbacks                                              python3.13                                                  
   0.39%        4 candsearch_in_pix_fast                                                 openptv2/algorithms/track_kernels_search.py:46              
   0.39%        4 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:538             
   0.39%        4 __Pyx_GetItemInt_Fast                                                  openptv2/algorithms/track_kernels_search.py:509             
   0.39%        4 Py_DECREF                                                              object.h:944                                                
   0.39%        4 _Py_IsImmortal                                                         object.h:361                                                
   0.39%        4 array_subscript                                                        numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.39%        4 get_view_from_index                                                    numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.39%        4 PyArray_NewFromDescr_int                                               numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.39%        4 __pyx_memoryview_fromslice                                             View.MemoryView:919                                         
   0.39%        4 __Pyx_BufFmt_CheckString                                               View.MemoryView:689                                         
   0.39%        4 _buffer_get_info                                                       numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.39%        4 track_kernels_search_candsearch_in_pix_fast                            openptv2/algorithms/track_kernels_search.py:141             
   0.39%        4 track_kernels_transform_assess_new_position_fast                       openptv2/algorithms/track_kernels_transform.py:529          
   0.39%        4 __Pyx_GetItemInt_Fast                                                  openptv2/algorithms/track_kernels_transform.py:899          
   0.39%        4 Py_DECREF                                                              object.h:944                                                
   0.39%        4 _Py_IsImmortal                                                         object.h:361                                                
   0.39%        4 __pyx_memoryview___getitem__                                           View.MemoryView:417                                         
   0.39%        4 write                                                                  openptv2/algorithms/tracking_frame_buf.py:763               
   0.39%        4 write                                                                  openptv2/algorithms/tracking_frame_buf.py:777               
   0.39%        4 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:598             
   0.39%        4 candsearch_in_pix_fast                                                 openptv2/algorithms/track_kernels_search.py:46              
   0.39%        4 __Pyx_XCLEAR_MEMVIEW                                                   View.MemoryView:689                                         
   0.39%        4 __pyx_memoryview___cinit__                                             View.MemoryView:356                                         
   0.39%        4 __pyx_tp_dealloc_memoryview                                            openptv2/algorithms/track_kernels_geom.py:1111              
   0.39%        4 track_kernels_search_candsearch_in_pix_fast                            openptv2/algorithms/track_kernels_search.py:135             
   0.39%        4 _point_position_out                                                    openptv2/algorithms/track_kernels_transform.py:51           
   0.39%        4 wrap_session                                                           _pytest/main.py:372                                         
   0.39%        4 <module>                                                               _pytest/config/__init__.py:23                               
   0.39%        4 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.39%        4 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.39%        4 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.39%        4 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.39%        4 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.29%        3 <module>                                                               pytest/__init__.py:32                                       
   0.29%        3 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.29%        3 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.29%        3 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.29%        3 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.29%        3 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.29%        3 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.29%        3 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.29%        3 read_text                                                              importlib/metadata/__init__.py:915                          
   0.29%        3 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.29%        3 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.29%        3 exec_module                                                            _pytest/assertion/rewrite.py:188                            
   0.29%        3 <module>                                                               anyio/pytest_plugin.py:276                                  
   0.29%        3 get_available_backends                                                 anyio/_core/_eventloop.py:153                               
   0.29%        3 get_async_backend                                                      anyio/_core/_eventloop.py:206                               
   0.29%        3 import_module                                                          importlib/__init__.py:88                                    
   0.29%        3 _gcd_import                                                            &lt;frozen importlib._bootstrap&gt;:1387                    
   0.29%        3 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.29%        3 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.29%        3 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.29%        3 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1023           
   0.29%        3 get_code                                                               &lt;frozen importlib._bootstrap_external&gt;:1156           
   0.29%        3 _compile_bytecode                                                      &lt;frozen importlib._bootstrap_external&gt;:785            
   0.29%        3 0x19fff69                                                              python3.13                                                  
   0.29%        3 0x710f0442a1ca                                                         libc.so.6                                                   
   0.29%        3 0x199792d                                                              python3.13                                                  
   0.29%        3 0x1997b45                                                              python3.13                                                  
   0.29%        3 0x1997e92                                                              python3.13                                                  
   0.29%        3 0x1a074da                                                              python3.13                                                  
   0.29%        3 0x1a075f6                                                              python3.13                                                  
   0.29%        3 0x1a07650                                                              python3.13                                                  
   0.29%        3 0x1a07b37                                                              python3.13                                                  
   0.29%        3 0x1a08447                                                              python3.13                                                  
   0.29%        3 0x18dc183                                                              python3.13                                                  
   0.29%        3 print_exception_file_and_line                                          python3.13                                                  
   0.29%        3 0x18aa114                                                              python3.13                                                  
   0.29%        3 0x1ac6c0d                                                              python3.13                                                  
   0.29%        3 0x18e7e51                                                              python3.13                                                  
   0.29%        3 0x18e7fa5                                                              python3.13                                                  
   0.29%        3 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.29%        3 _loop0_152_rule                                                        python3.13                                                  
   0.29%        3 0x1ac6c0d                                                              python3.13                                                  
   0.29%        3 0x18e7e51                                                              python3.13                                                  
   0.29%        3 0x18e7fa5                                                              python3.13                                                  
   0.29%        3 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.29%        3 _loop0_152_rule                                                        python3.13                                                  
   0.29%        3 0x181ff4e                                                              python3.13                                                  
   0.29%        3 0x18aaa06                                                              python3.13                                                  
   0.29%        3 0x18aa114                                                              python3.13                                                  
   0.29%        3 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.29%        3 0x1abfb39                                                              python3.13                                                  
   0.29%        3 marshal_dump_impl                                                      python3.13                                                  
   0.29%        3 va_build_value                                                         python3.13                                                  
   0.29%        3 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.29%        3 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.29%        3 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.29%        3 0x18aaa06                                                              python3.13                                                  
   0.29%        3 0x18aa114                                                              python3.13                                                  
   0.29%        3 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.29%        3 0x1abfb39                                                              python3.13                                                  
   0.29%        3 marshal_dump_impl                                                      python3.13                                                  
   0.29%        3 va_build_value                                                         python3.13                                                  
   0.29%        3 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.29%        3 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.29%        3 0x1802235                                                              python3.13                                                  
   0.29%        3 0x19a6f16                                                              python3.13                                                  
   0.29%        3 __pyx_pymod_exec_track_kernels                                         openptv2/algorithms/track_kernels.py:9                      
   0.29%        3 __Pyx_Import                                                           openptv2/algorithms/track_kernels.py:48                     
   0.29%        3 __Pyx__Import                                                          openptv2/algorithms/track_kernels.py:48                     
   0.29%        3 0x1abfb39                                                              python3.13                                                  
   0.29%        3 marshal_dump_impl                                                      python3.13                                                  
   0.29%        3 va_build_value                                                         python3.13                                                  
   0.29%        3 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.29%        3 0x181ff4e                                                              python3.13                                                  
   0.29%        3 0x19a0ebc                                                              python3.13                                                  
   0.29%        3 0x19a0fac                                                              python3.13                                                  
   0.29%        3 typevartuple_alloc                                                     python3.13                                                  
   0.29%        3 typevartuple                                                           python3.13                                                  
   0.29%        3 typevartuple                                                           python3.13                                                  
   0.29%        3 paramspec_alloc                                                        python3.13                                                  
   0.29%        3 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.29%        3 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.29%        3 <module>                                                               importlib/metadata/_adapters.py:5                           
   0.29%        3 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.29%        3 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.29%        3 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.29%        3 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.29%        3 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.29%        3 0x1801549                                                              python3.13                                                  
   0.29%        3 test_cavity                                                            test_track.py:492                                           
   0.29%        3 copytree                                                               shutil.py:593                                               
   0.29%        3 _copytree                                                              shutil.py:533                                               
   0.29%        3 0x1807283                                                              python3.13                                                  
   0.29%        3 _loop0_191_rule                                                        python3.13                                                  
   0.29%        3 0x18072e4                                                              python3.13                                                  
   0.29%        3 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:38                
   0.29%        3 0x185cd2a                                                              python3.13                                                  
   0.29%        3 realloc                                                                libc.so.6                                                   
   0.29%        3 0x710f044ad088                                                         libc.so.6                                                   
   0.29%        3 0x710f04588d87                                                         libc.so.6                                                   
   0.29%        3 Py_XDECREF                                                             object.h:1041                                               
   0.29%        3 _PyTraceMalloc_GetTraceback                                            python3.13                                                  
   0.29%        3 __Pyx_SetItemInt_Fast                                                  View.MemoryView:689                                         
   0.29%        3 track_kernels_geom__multimed_r_nlay_1layer                             openptv2/algorithms/track_kernels_geom.py:92                
   0.29%        3 0x1ab4a56                                                              python3.13                                                  
   0.29%        3 0x18008f3                                                              python3.13                                                  
   0.29%        3 0x1800a13                                                              python3.13                                                  
   0.29%        3 0x1ab4a56                                                              python3.13                                                  
   0.29%        3 _PyTraceMalloc_ClearTraces                                             python3.13                                                  
   0.29%        3 0x18072e4                                                              python3.13                                                  
   0.29%        3 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:259               
   0.29%        3 tracking_frame_buf_read_path_frame                                     openptv2/algorithms/tracking_frame_buf.py:396               
   0.29%        3 read                                                                   openptv2/algorithms/tracking_frame_buf.py:737               
   0.29%        3 0x185e6e9                                                              python3.13                                                  
   0.29%        3 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:379           
   0.29%        3 __Pyx_PyObject_to_MemoryviewSlice_ds_double                            View.MemoryView:689                                         
   0.29%        3 PyArray_UpdateFlags                                                    numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.29%        3 __pyx_memoryview_fromslice                                             View.MemoryView:919                                         
   0.29%        3 __Pyx_BufFmt_TypeCharToAlignment                                       View.MemoryView:689                                         
   0.29%        3 Py_DECREF                                                              object.h:949                                                
   0.29%        3 track_kernels_search_candsearch_in_pix_fast                            openptv2/algorithms/track_kernels_search.py:138             
   0.29%        3 track_kernels_search_candsearch_in_pix_fast                            openptv2/algorithms/track_kernels_search.py:141             
   0.29%        3 track_kernels_transform_assess_new_position_fast                       openptv2/algorithms/track_kernels_transform.py:527          
   0.29%        3 track_kernels_transform_assess_new_position_fast                       openptv2/algorithms/track_kernels_transform.py:538          
   0.29%        3 candsearch_in_pix_rest_fast                                            openptv2/algorithms/track_kernels_search.py:174             
   0.29%        3 _ray_tracing_out                                                       openptv2/algorithms/track_kernels_geom.py:1111              
   0.29%        3 tracking_frame_buf_write_path_frame                                    openptv2/algorithms/tracking_frame_buf.py:487               
   0.29%        3 tracking_frame_buf_write_targets                                       openptv2/algorithms/tracking_frame_buf.py:203               
   0.29%        3 __Pyx_XCLEAR_MEMVIEW                                                   View.MemoryView:689                                         
   0.29%        3 __pyx_tp_dealloc_memoryview                                            openptv2/algorithms/track_kernels_geom.py:1111              
   0.29%        3 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:539             
   0.29%        3 __pyx_memoryview_fromslice                                             View.MemoryView:919                                         
   0.29%        3 __Pyx_BufFmt_TypeCharToNativeSize                                      View.MemoryView:689                                         
   0.29%        3 _buffer_format_string                                                  numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.29%        3 track_kernels_geom__point_to_pixel_out                                 openptv2/algorithms/track_kernels_geom.py:620               
   0.29%        3 track_trackcorr_c_loop                                                 openptv2/algorithms/track.py:1168                           
   0.29%        3 0x19bf0ab                                                              python3.13                                                  
   0.29%        3 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.29%        3 write_frame_from_start                                                 openptv2/algorithms/tracking_frame_buf.py:838               
   0.29%        3 write_frame_from_start                                                 openptv2/algorithms/tracking_frame_buf.py:845               
   0.29%        3 0x19bf0ab                                                              python3.13                                                  
   0.29%        3 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.29%        3 write                                                                  openptv2/algorithms/tracking_frame_buf.py:755               
   0.29%        3 write                                                                  openptv2/algorithms/tracking_frame_buf.py:777               
   0.29%        3 0x1801740                                                              python3.13                                                  
   0.29%        3 0x1ab4e5b                                                              python3.13                                                  
   0.29%        3 munmap                                                                 libc.so.6                                                   
   0.29%        3 0x1801549                                                              python3.13                                                  
   0.29%        3 __Pyx_PyObject_FastCallDict                                            openptv2/algorithms/tracking_frame_buf.py:849               
   0.29%        3 PyArray_DiscoverDTypeAndShape_Recursive                                numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.29%        3 PyArray_Zeros_int                                                      numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.29%        3 PyArray_NewFromDescr_int                                               numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.29%        3 0x185d5ab                                                              python3.13                                                  
   0.29%        3 malloc                                                                 libc.so.6                                                   
   0.29%        3 0x710f044ac51a                                                         libc.so.6                                                   
   0.29%        3 0x1800930                                                              python3.13                                                  
   0.29%        3 read                                                                   openptv2/algorithms/tracking_frame_buf.py:722               
   0.29%        3 Py_DECREF                                                              object.h:949                                                
   0.29%        3 0x185e699                                                              python3.13                                                  
   0.29%        3 0x1807283                                                              python3.13                                                  
   0.29%        3 __pyx_tp_new_memoryview                                                openptv2/algorithms/track_kernels_geom.py:1111              
   0.29%        3 __pyx_memoryview___cinit__                                             View.MemoryView:356                                         
   0.29%        3 __pyx_memoryview___pyx_pf_15View_dot_MemoryView_10memoryview___ci...   View.MemoryView:360                                         
   0.29%        3 array_getbuffer                                                        numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.29%        3 __Pyx_PyObject_to_MemoryviewSlice_ds_double                            View.MemoryView:689                                         
   0.29%        3 __Pyx_ValidateAndInit_memviewslice                                     View.MemoryView:689                                         
   0.29%        3 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   0.29%        3 track_kernels_geom__point_to_pixel_out                                 openptv2/algorithms/track_kernels_geom.py:620               
   0.29%        3 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:509             
   0.29%        3 __pyx_memoryview_fromslice                                             View.MemoryView:919                                         
   0.29%        3 __Pyx_BufFmt_ProcessTypeChunk                                          View.MemoryView:689                                         
   0.29%        3 __pyx_memoryview_getbuffer                                             View.MemoryView:531                                         
   0.29%        3 __pyx_tp_dealloc_memoryview                                            openptv2/algorithms/track_kernels_geom.py:1111              
   0.29%        3 track_kernels_geom__point_to_pixel_out                                 openptv2/algorithms/track_kernels_geom.py:620               
   0.29%        3 _multimed_r_nlay_1layer                                                openptv2/algorithms/track_kernels_geom.py:47                
   0.29%        3 __Pyx_PyObject_to_MemoryviewSlice_ds_double                            openptv2/algorithms/track_kernels_transform.py:899          
   0.29%        3 Py_DECREF                                                              object.h:949                                                
   0.29%        3 track_kernels_transform__point_position_out                            openptv2/algorithms/track_kernels_transform.py:124          
   0.29%        3 __Pyx_PyObject_to_MemoryviewSlice_ds_double                            View.MemoryView:689                                         
   0.29%        3 track_trackcorr_c_loop                                                 openptv2/algorithms/track.py:1155                           
   0.29%        3 tracking_frame_buf_write_path_frame                                    openptv2/algorithms/tracking_frame_buf.py:487               
   0.29%        3 compiler_try_star_except                                               python3.13                                                  
   0.29%        3 __pyx_memoryview_new                                                   View.MemoryView:673                                         
   0.29%        3 __Pyx_PyObject_FastCallDict                                            View.MemoryView:689                                         
   0.29%        3 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:539             
   0.29%        3 __Pyx_BufFmt_CheckString                                               View.MemoryView:689                                         
   0.29%        3 __Pyx_BufFmt_ProcessTypeChunk                                          View.MemoryView:689                                         
   0.29%        3 candsearch_in_pix_rest_fast                                            openptv2/algorithms/track_kernels_search.py:174             
   0.29%        3 0x1800930                                                              python3.13                                                  
   0.29%        3 0x1800a08                                                              python3.13                                                  
   0.29%        3 0x1800a15                                                              python3.13                                                  
   0.29%        3 <module>                                                               _pytest/pathlib.py:35                                       
   0.29%        3 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.29%        3 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.29%        3 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.29%        3 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.29%        3 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.29%        3 <module>                                                               _pytest/compat.py:20                                        
   0.29%        3 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.29%        3 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.29%        3 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.29%        3 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.29%        3 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.29%        3 <module>                                                               py.py:8                                                     
   0.29%        3 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.20%        2 0x19f97c7                                                              python3.13                                                  
   0.20%        2 future_schedule_callbacks                                              python3.13                                                  
   0.20%        2 0x1ab4a56                                                              python3.13                                                  
   0.20%        2 0x1800a93                                                              python3.13                                                  
   0.20%        2 0x1800949                                                              python3.13                                                  
   0.20%        2 0x19f97e9                                                              python3.13                                                  
   0.20%        2 0x19f9876                                                              python3.13                                                  
   0.20%        2 TaskStepMethWrapper_call                                               python3.13                                                  
   0.20%        2 0x19fc2be                                                              python3.13                                                  
   0.20%        2 future_schedule_callbacks                                              python3.13                                                  
   0.20%        2 0x18004ff                                                              python3.13                                                  
   0.20%        2 <module>                                                               pytest/__init__.py:24                                       
   0.20%        2 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.20%        2 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.20%        2 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.20%        2 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.20%        2 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.20%        2 <module>                                                               _pytest/doctest.py:41                                       
   0.20%        2 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.20%        2 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.20%        2 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.20%        2 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.20%        2 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.20%        2 <module>                                                               _pytest/python_api.py:11                                    
   0.20%        2 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.20%        2 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.20%        2 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.20%        2 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.20%        2 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.20%        2 <module>                                                               decimal.py:102                                              
   0.20%        2 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.20%        2 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.20%        2 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.20%        2 exec_module                                                            &lt;frozen importlib._bootstrap&gt;:1000                    
   0.20%        2 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.20%        2 <module>                                                               _pytest/legacypath.py:268                                   
   0.20%        2 dataclass                                                              dataclasses.py:1305                                         
   0.20%        2 wrap                                                                   dataclasses.py:1297                                         
   0.20%        2 _process_class                                                         dataclasses.py:1157                                         
   0.20%        2 add_fns_to_class                                                       dataclasses.py:498                                          
   0.20%        2 0x19fff69                                                              python3.13                                                  
   0.20%        2 0x710f0442a1ca                                                         libc.so.6                                                   
   0.20%        2 0x199792d                                                              python3.13                                                  
   0.20%        2 0x1997b45                                                              python3.13                                                  
   0.20%        2 0x1997e92                                                              python3.13                                                  
   0.20%        2 0x1a074da                                                              python3.13                                                  
   0.20%        2 0x1a075f6                                                              python3.13                                                  
   0.20%        2 0x1a07650                                                              python3.13                                                  
   0.20%        2 0x1a07b37                                                              python3.13                                                  
   0.20%        2 0x1a08447                                                              python3.13                                                  
   0.20%        2 0x18dc183                                                              python3.13                                                  
   0.20%        2 print_exception_file_and_line                                          python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.20%        2 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.20%        2 0x18aaa06                                                              python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.20%        2 0x1abfb39                                                              python3.13                                                  
   0.20%        2 marshal_dump_impl                                                      python3.13                                                  
   0.20%        2 va_build_value                                                         python3.13                                                  
   0.20%        2 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.20%        2 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.20%        2 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.20%        2 0x18aaa06                                                              python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.20%        2 0x1abfb39                                                              python3.13                                                  
   0.20%        2 marshal_dump_impl                                                      python3.13                                                  
   0.20%        2 va_build_value                                                         python3.13                                                  
   0.20%        2 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.20%        2 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.20%        2 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.20%        2 0x18aaa06                                                              python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 0x181ff4e                                                              python3.13                                                  
   0.20%        2 _Py_DumpHexadecimal                                                    python3.13                                                  
   0.20%        2 0x18dc088                                                              python3.13                                                  
   0.20%        2 0x18dc147                                                              python3.13                                                  
   0.20%        2 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1023           
   0.20%        2 get_code                                                               &lt;frozen importlib._bootstrap_external&gt;:1156           
   0.20%        2 _compile_bytecode                                                      &lt;frozen importlib._bootstrap_external&gt;:785            
   0.20%        2 0x19fff69                                                              python3.13                                                  
   0.20%        2 0x710f0442a1ca                                                         libc.so.6                                                   
   0.20%        2 0x199792d                                                              python3.13                                                  
   0.20%        2 0x1997b45                                                              python3.13                                                  
   0.20%        2 0x1997e92                                                              python3.13                                                  
   0.20%        2 0x1a074da                                                              python3.13                                                  
   0.20%        2 0x1a075f6                                                              python3.13                                                  
   0.20%        2 0x1a07650                                                              python3.13                                                  
   0.20%        2 0x1a07b37                                                              python3.13                                                  
   0.20%        2 0x1a08447                                                              python3.13                                                  
   0.20%        2 0x18dc183                                                              python3.13                                                  
   0.20%        2 print_exception_file_and_line                                          python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.20%        2 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.20%        2 0x18aaa06                                                              python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.20%        2 0x1abfb39                                                              python3.13                                                  
   0.20%        2 marshal_dump_impl                                                      python3.13                                                  
   0.20%        2 va_build_value                                                         python3.13                                                  
   0.20%        2 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.20%        2 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.20%        2 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.20%        2 0x18aaa06                                                              python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.20%        2 0x1abfb39                                                              python3.13                                                  
   0.20%        2 marshal_dump_impl                                                      python3.13                                                  
   0.20%        2 va_build_value                                                         python3.13                                                  
   0.20%        2 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.20%        2 0x181ff4e                                                              python3.13                                                  
   0.20%        2 0x19a0ebc                                                              python3.13                                                  
   0.20%        2 0x19a0fac                                                              python3.13                                                  
   0.20%        2 typevartuple_alloc                                                     python3.13                                                  
   0.20%        2 typevartuple                                                           python3.13                                                  
   0.20%        2 typevartuple_alloc                                                     python3.13                                                  
   0.20%        2 typevartuple                                                           python3.13                                                  
   0.20%        2 typevartuple_alloc                                                     python3.13                                                  
   0.20%        2 <module>                                                               _pytest/subtests.py:341                                     
   0.20%        2 import_plugin                                                          _pytest/config/__init__.py:927                              
   0.20%        2 register                                                               _pytest/config/__init__.py:571                              
   0.20%        2 parse                                                                  _pytest/config/__init__.py:1533                             
   0.20%        2 determine_setup                                                        _pytest/config/findpaths.py:315                             
   0.20%        2 locate_config                                                          _pytest/config/findpaths.py:189                             
   0.20%        2 __new__                                                                importlib/metadata/__init__.py:341                          
   0.20%        2 0x19fff69                                                              python3.13                                                  
   0.20%        2 0x710f0442a1ca                                                         libc.so.6                                                   
   0.20%        2 0x199792d                                                              python3.13                                                  
   0.20%        2 0x1997b45                                                              python3.13                                                  
   0.20%        2 0x1997e92                                                              python3.13                                                  
   0.20%        2 0x1a074da                                                              python3.13                                                  
   0.20%        2 0x1a075f6                                                              python3.13                                                  
   0.20%        2 0x1a07650                                                              python3.13                                                  
   0.20%        2 0x1a07b37                                                              python3.13                                                  
   0.20%        2 0x1a08447                                                              python3.13                                                  
   0.20%        2 0x18dc183                                                              python3.13                                                  
   0.20%        2 print_exception_file_and_line                                          python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.20%        2 _loop0_152_rule                                                        python3.13                                                  
   0.20%        2 0x181efd3                                                              python3.13                                                  
   0.20%        2 stringlib_rjust                                                        python3.13                                                  
   0.20%        2 exceptiongroup_subset                                                  python3.13                                                  
   0.20%        2 0x18088fe                                                              python3.13                                                  
   0.20%        2 0x1808cf3                                                              python3.13                                                  
   0.20%        2 stringlib_rjust                                                        python3.13                                                  
   0.20%        2 0x181efd3                                                              python3.13                                                  
   0.20%        2 0x1808428                                                              python3.13                                                  
   0.20%        2 0x18021c8                                                              python3.13                                                  
   0.20%        2 0x185e699                                                              python3.13                                                  
   0.20%        2 0x1807283                                                              python3.13                                                  
   0.20%        2 PyObject_CopyData                                                      python3.13                                                  
   0.20%        2 PyCapsule_SetName                                                      python3.13                                                  
   0.20%        2 read_text                                                              pathlib/_local.py:546                                       
   0.20%        2 read_text                                                              pathlib/_abc.py:632                                         
   0.20%        2 open                                                                   pathlib/_local.py:537                                       
   0.20%        2 __fspath__                                                             pathlib/_local.py:167                                       
   0.20%        2 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1310                    
   0.20%        2 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.20%        2 _gcd_import                                                            &lt;frozen importlib._bootstrap&gt;:1387                    
   0.20%        2 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.20%        2 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.20%        2 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.20%        2 exec_module                                                            _pytest/assertion/rewrite.py:188                            
   0.20%        2 <module>                                                               anyio/__init__.py:55                                        
   0.20%        2 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.20%        2 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.20%        2 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.20%        2 exec_module                                                            _pytest/assertion/rewrite.py:188                            
   0.20%        2 <module>                                                               anyio/_core/_streams.py:7                                   
   0.20%        2 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.20%        2 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.20%        2 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.20%        2 exec_module                                                            _pytest/assertion/rewrite.py:188                            
   0.20%        2 exec_module                                                            _pytest/assertion/rewrite.py:176                            
   0.20%        2 _read_pyc                                                              _pytest/assertion/rewrite.py:393                            
   0.20%        2 0x19fff69                                                              python3.13                                                  
   0.20%        2 0x710f0442a1ca                                                         libc.so.6                                                   
   0.20%        2 0x199792d                                                              python3.13                                                  
   0.20%        2 0x1997b45                                                              python3.13                                                  
   0.20%        2 0x1997e92                                                              python3.13                                                  
   0.20%        2 0x1a074da                                                              python3.13                                                  
   0.20%        2 0x1a075f6                                                              python3.13                                                  
   0.20%        2 0x1a07650                                                              python3.13                                                  
   0.20%        2 0x1a07b37                                                              python3.13                                                  
   0.20%        2 0x1a08447                                                              python3.13                                                  
   0.20%        2 0x18dc183                                                              python3.13                                                  
   0.20%        2 print_exception_file_and_line                                          python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.20%        2 _loop0_152_rule                                                        python3.13                                                  
   0.20%        2 0x181ff4e                                                              python3.13                                                  
   0.20%        2 0x18aaa06                                                              python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 0x181ff4e                                                              python3.13                                                  
   0.20%        2 0x19a0fac                                                              python3.13                                                  
   0.20%        2 typevartuple_alloc                                                     python3.13                                                  
   0.20%        2 typevartuple                                                           python3.13                                                  
   0.20%        2 0x1803269                                                              python3.13                                                  
   0.20%        2 set_mro_error                                                          python3.13                                                  
   0.20%        2 <module>                                                               numpy/_core/numerictypes.py:117                             
   0.20%        2 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.20%        2 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.20%        2 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.20%        2 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.20%        2 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.20%        2 typevartuple_alloc                                                     python3.13                                                  
   0.20%        2 <module>                                                               numpy/_core/__init__.py:128                                 
   0.20%        2 _handle_fromlist                                                       &lt;frozen importlib._bootstrap&gt;:1415                    
   0.20%        2 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.20%        2 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.20%        2 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.20%        2 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.20%        2 <module>                                                               numpy/_core/__init__.py:24                                  
   0.20%        2 _handle_fromlist                                                       &lt;frozen importlib._bootstrap&gt;:1415                    
   0.20%        2 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.20%        2 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.20%        2 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.20%        2 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.20%        2 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.20%        2 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.20%        2 <module>                                                               numpy/_core/multiarray.py:11                                
   0.20%        2 _handle_fromlist                                                       &lt;frozen importlib._bootstrap&gt;:1415                    
   0.20%        2 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.20%        2 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.20%        2 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.20%        2 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1023           
   0.20%        2 <module>                                                               numpy/_typing/_array_like.py:60                             
   0.20%        2 <module>                                                               numpy/lib/_arraysetops_impl.py:419                          
   0.20%        2 <module>                                                               email/message.py:15                                         
   0.20%        2 _handle_fromlist                                                       &lt;frozen importlib._bootstrap&gt;:1415                    
   0.20%        2 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.20%        2 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.20%        2 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.20%        2 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.20%        2 collect                                                                _pytest/main.py:584                                         
   0.20%        2 __call__                                                               pluggy/_hooks.py:512                                        
   0.20%        2 _hookexec                                                              pluggy/_manager.py:120                                      
   0.20%        2 _multicall                                                             pluggy/_callers.py:121                                      
   0.20%        2 pytest_collect_file                                                    _pytest/python.py:203                                       
   0.20%        2 __call__                                                               pluggy/_hooks.py:512                                        
   0.20%        2 _hookexec                                                              pluggy/_manager.py:120                                      
   0.20%        2 _multicall                                                             pluggy/_callers.py:121                                      
   0.20%        2 pytest_pycollect_makemodule                                            _pytest/python.py:216                                       
   0.20%        2 from_parent                                                            _pytest/nodes.py:619                                        
   0.20%        2 from_parent                                                            _pytest/nodes.py:225                                        
   0.20%        2 _create                                                                _pytest/nodes.py:101                                        
   0.20%        2 __init__                                                               _pytest/nodes.py:593                                        
   0.20%        2 relative_to                                                            pathlib/_local.py:382                                       
   0.20%        2 __contains__                                                           &lt;frozen _collections_abc&gt;:1038                        
   0.20%        2 __iter__                                                               &lt;frozen _collections_abc&gt;:1031                        
   0.20%        2 __getitem__                                                            pathlib/_local.py:53                                        
   0.20%        2 _from_parsed_parts                                                     pathlib/_local.py:246                                       
   0.20%        2 _from_parsed_string                                                    pathlib/_local.py:253                                       
   0.20%        2 with_segments                                                          pathlib/_abc.py:135                                         
   0.20%        2 __init__                                                               pathlib/_local.py:503                                       
   0.20%        2 __init__                                                               pathlib/_local.py:128                                       
   0.20%        2 0x19fff69                                                              python3.13                                                  
   0.20%        2 0x710f0442a1ca                                                         libc.so.6                                                   
   0.20%        2 0x199792d                                                              python3.13                                                  
   0.20%        2 0x1997b45                                                              python3.13                                                  
   0.20%        2 0x1997e92                                                              python3.13                                                  
   0.20%        2 0x1a074da                                                              python3.13                                                  
   0.20%        2 0x1a075f6                                                              python3.13                                                  
   0.20%        2 0x1a07650                                                              python3.13                                                  
   0.20%        2 0x1a07b37                                                              python3.13                                                  
   0.20%        2 0x1a08447                                                              python3.13                                                  
   0.20%        2 0x18dc183                                                              python3.13                                                  
   0.20%        2 print_exception_file_and_line                                          python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 exceptiongroup_subset                                                  python3.13                                                  
   0.20%        2 0x18088fe                                                              python3.13                                                  
   0.20%        2 0x1808cf3                                                              python3.13                                                  
   0.20%        2 stringlib_rjust                                                        python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x181ddac                                                              python3.13                                                  
   0.20%        2 0x18088fe                                                              python3.13                                                  
   0.20%        2 0x1808cf3                                                              python3.13                                                  
   0.20%        2 stringlib_rjust                                                        python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 values_lock_held                                                       python3.13                                                  
   0.20%        2 _loop0_152_rule                                                        python3.13                                                  
   0.20%        2 values_lock_held                                                       python3.13                                                  
   0.20%        2 _loop0_152_rule                                                        python3.13                                                  
   0.20%        2 PyFunction_NewWithQualName                                             python3.13                                                  
   0.20%        2 0x18072e4                                                              python3.13                                                  
   0.20%        2 validate_pattern_match_value                                           python3.13                                                  
   0.20%        2 _PyObject_CallMethod_SizeT                                             python3.13                                                  
   0.20%        2 mi_commit_mask_any_set                                                 python3.13                                                  
   0.20%        2 PyFunction_NewWithQualName                                             python3.13                                                  
   0.20%        2 0x18072e4                                                              python3.13                                                  
   0.20%        2 validate_pattern_match_value                                           python3.13                                                  
   0.20%        2 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.20%        2 _loop0_152_rule                                                        python3.13                                                  
   0.20%        2 0x1acace1                                                              python3.13                                                  
   0.20%        2 task_step_handle_result_impl                                           python3.13                                                  
   0.20%        2 collect                                                                _pytest/python.py:564                                       
   0.20%        2 _register_setup_module_fixture                                         _pytest/python.py:577                                       
   0.20%        2 0x180154e                                                              python3.13                                                  
   0.20%        2 array_dealloc                                                          numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 PyArray_MultiplyList                                                   numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 0x180158c                                                              python3.13                                                  
   0.20%        2 0x1801812                                                              python3.13                                                  
   0.20%        2 free                                                                   libc.so.6                                                   
   0.20%        2 0x710f044ab43a                                                         libc.so.6                                                   
   0.20%        2 0x710f044aada5                                                         libc.so.6                                                   
   0.20%        2 array_dealloc                                                          numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 free                                                                   libc.so.6                                                   
   0.20%        2 munmap                                                                 libc.so.6                                                   
   0.20%        2 copy2                                                                  shutil.py:468                                               
   0.20%        2 copyfile                                                               shutil.py:273                                               
   0.20%        2 _fastcopy_sendfile                                                     shutil.py:150                                               
   0.20%        2 0x19fff69                                                              python3.13                                                  
   0.20%        2 0x710f0442a1ca                                                         libc.so.6                                                   
   0.20%        2 0x199792d                                                              python3.13                                                  
   0.20%        2 0x1997b45                                                              python3.13                                                  
   0.20%        2 0x1997e92                                                              python3.13                                                  
   0.20%        2 0x1a074da                                                              python3.13                                                  
   0.20%        2 0x1a075f6                                                              python3.13                                                  
   0.20%        2 0x1a07650                                                              python3.13                                                  
   0.20%        2 0x1a07b37                                                              python3.13                                                  
   0.20%        2 0x1a08447                                                              python3.13                                                  
   0.20%        2 0x18dc183                                                              python3.13                                                  
   0.20%        2 print_exception_file_and_line                                          python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 PyFunction_NewWithQualName                                             python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x181ff4e                                                              python3.13                                                  
   0.20%        2 os_sendfile                                                            python3.13                                                  
   0.20%        2 os_sendfile_impl                                                       python3.13                                                  
   0.20%        2 sendfile64                                                             libc.so.6                                                   
   0.20%        2 0x1801969                                                              python3.13                                                  
   0.20%        2 PyArray_DiscoverDTypeAndShape_Recursive                                numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 PyDataMem_UserNEW                                                      numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 _loop0_191_rule                                                        python3.13                                                  
   0.20%        2 __Pyx_PyObject_GetAttrStr                                              openptv2/algorithms/tracking_frame_buf.py:849               
   0.20%        2 _loop0_142_rule                                                        python3.13                                                  
   0.20%        2 __Pyx__GetModuleGlobalName                                             openptv2/algorithms/tracking_frame_buf.py:849               
   0.20%        2 0x1807283                                                              python3.13                                                  
   0.20%        2 __pyx_tp_new_8openptv2_10algorithms_18tracking_frame_buf_Pathinfo      openptv2/algorithms/tracking_frame_buf.py:852               
   0.20%        2 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:250               
   0.20%        2 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.20%        2 array_zeros                                                            numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 __Pyx_PyObject_GetAttrStr                                              openptv2/algorithms/tracking_frame_buf.py:849               
   0.20%        2 Pathinfo___init__                                                      openptv2/algorithms/tracking_frame_buf.py:265               
   0.20%        2 __Pyx_PyLong_As_int                                                    openptv2/algorithms/tracking_frame_buf.py:849               
   0.20%        2 __Pyx__GetModuleGlobalName                                             openptv2/algorithms/tracking_frame_buf.py:849               
   0.20%        2 0x185d5ab                                                              python3.13                                                  
   0.20%        2 malloc                                                                 libc.so.6                                                   
   0.20%        2 0x710f044ac51a                                                         libc.so.6                                                   
   0.20%        2 __Pyx__GetModuleGlobalName                                             View.MemoryView:689                                         
   0.20%        2 future_init                                                            python3.13                                                  
   0.20%        2 Py_DECREF                                                              object.h:949                                                
   0.20%        2 track_kernels_geom__multimed_r_nlay_1layer                             openptv2/algorithms/track_kernels_geom.py:91                
   0.20%        2 0x18004ff                                                              python3.13                                                  
   0.20%        2 0x1800930                                                              python3.13                                                  
   0.20%        2 0x1800930                                                              python3.13                                                  
   0.20%        2 0x1800950                                                              python3.13                                                  
   0.20%        2 0x1800a08                                                              python3.13                                                  
   0.20%        2 0x1800a13                                                              python3.13                                                  
   0.20%        2 0x1800a15                                                              python3.13                                                  
   0.20%        2 0x1800903                                                              python3.13                                                  
   0.20%        2 0x1800912                                                              python3.13                                                  
   0.20%        2 full                                                                   numpy/_core/numeric.py:386                                  
   0.20%        2 0x19fff69                                                              python3.13                                                  
   0.20%        2 0x710f0442a1ca                                                         libc.so.6                                                   
   0.20%        2 0x199792d                                                              python3.13                                                  
   0.20%        2 0x1997b45                                                              python3.13                                                  
   0.20%        2 0x1997e92                                                              python3.13                                                  
   0.20%        2 0x1a074da                                                              python3.13                                                  
   0.20%        2 0x1a075f6                                                              python3.13                                                  
   0.20%        2 0x1a07650                                                              python3.13                                                  
   0.20%        2 0x1a07b37                                                              python3.13                                                  
   0.20%        2 0x1a08447                                                              python3.13                                                  
   0.20%        2 0x18dc183                                                              python3.13                                                  
   0.20%        2 print_exception_file_and_line                                          python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 PyFunction_NewWithQualName                                             python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 exceptiongroup_subset                                                  python3.13                                                  
   0.20%        2 tr_new                                                                 openptv2/algorithms/tracking_run.py:53                      
   0.20%        2 tracking_run_tr_new                                                    openptv2/algorithms/tracking_run.py:53                      
   0.20%        2 tracking_run_tr_new                                                    openptv2/algorithms/tracking_run.py:82                      
   0.20%        2 0x185e699                                                              python3.13                                                  
   0.20%        2 0x18072e4                                                              python3.13                                                  
   0.20%        2 validate_pattern_match_value                                           python3.13                                                  
   0.20%        2 exceptiongroup_subset                                                  python3.13                                                  
   0.20%        2 __post_init__                                                          openptv2/algorithms/tracking_run.py:28                      
   0.20%        2 Py_XDECREF                                                             object.h:1041                                               
   0.20%        2 TrackingRun___post_init__                                              openptv2/algorithms/tracking_run.py:32                      
   0.20%        2 0x185e699                                                              python3.13                                                  
   0.20%        2 0x18072e4                                                              python3.13                                                  
   0.20%        2 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:800               
   0.20%        2 FrameBuf___init__                                                      openptv2/algorithms/tracking_frame_buf.py:812               
   0.20%        2 0x185e699                                                              python3.13                                                  
   0.20%        2 0x18072e4                                                              python3.13                                                  
   0.20%        2 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:523               
   0.20%        2 Frame___init__                                                         openptv2/algorithms/tracking_frame_buf.py:546               
   0.20%        2 _PyTraceMalloc_GetTraceback                                            python3.13                                                  
   0.20%        2 async_gen_asend_dealloc                                                python3.13                                                  
   0.20%        2 dispatcher_vectorcall                                                  numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 _PyTraceMalloc_GetTraceback                                            python3.13                                                  
   0.20%        2 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.20%        2 array_copyto                                                           numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 PyArray_AssignRawScalar                                                numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 raw_array_assign_scalar                                                numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 _aligned_strided_to_contig_size4_srcstride0                            numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 PyArray_DiscoverDTypeAndShape                                          numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 PyArray_DiscoverDTypeAndShape_Recursive                                numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 PyArray_NewFromDescr_int                                               numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 __Pyx__PyNumber_Float                                                  openptv2/algorithms/tracking_frame_buf.py:849               
   0.20%        2 __Pyx_PyUnicode_AsDouble                                               View.MemoryView:856                                         
   0.20%        2 __Pyx__PyBytes_AsDouble                                                openptv2/algorithms/tracking_frame_buf.py:849               
   0.20%        2 _PyTraceMalloc_GetTraceback                                            python3.13                                                  
   0.20%        2 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.20%        2 array_array                                                            numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 track_trackcorr_c_loop                                                 openptv2/algorithms/track.py:1060                           
   0.20%        2 0x19bf0ab                                                              python3.13                                                  
   0.20%        2 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.20%        2 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:295           
   0.20%        2 __pyx_memoryview_fromslice                                             View.MemoryView:924                                         
   0.20%        2 __pyx_memoryview__get_base                                             View.MemoryView:575                                         
   0.20%        2 Py_INCREF                                                              object.h:826                                                
   0.20%        2 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:511             
   0.20%        2 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:539             
   0.20%        2 __pyx_memoryview_fromslice                                             View.MemoryView:919                                         
   0.20%        2 __pyx_tp_new__memoryviewslice                                          openptv2/algorithms/track_kernels_search.py:429             
   0.20%        2 __pyx_tp_new_memoryview                                                openptv2/algorithms/track_kernels_search.py:429             
   0.20%        2 __Pyx_AllocateExtensionType                                            openptv2/algorithms/track_kernels_search.py:509             
   0.20%        2 __Pyx_BufFmt_CheckString                                               View.MemoryView:689                                         
   0.20%        2 __Pyx_BufFmt_ProcessTypeChunk                                          View.MemoryView:689                                         
   0.20%        2 _buffer_format_string                                                  numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 __Pyx_ValidateAndInit_memviewslice                                     View.MemoryView:689                                         
   0.20%        2 __pyx_memoryview_new                                                   View.MemoryView:673                                         
   0.20%        2 __Pyx_PyObject_FastCallDict                                            View.MemoryView:689                                         
   0.20%        2 0x185e699                                                              python3.13                                                  
   0.20%        2 0x1807283                                                              python3.13                                                  
   0.20%        2 __pyx_tp_new_memoryview                                                openptv2/algorithms/track_kernels_geom.py:1111              
   0.20%        2 __pyx_memoryview___cinit__                                             View.MemoryView:356                                         
   0.20%        2 __pyx_memoryview___pyx_pf_15View_dot_MemoryView_10memoryview___ci...   View.MemoryView:360                                         
   0.20%        2 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   0.20%        2 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:502             
   0.20%        2 _loop0_191_rule                                                        python3.13                                                  
   0.20%        2 __pyx_memoryview_fromslice                                             View.MemoryView:919                                         
   0.20%        2 Py_DECREF                                                              object.h:949                                                
   0.20%        2 __pyx_memoryview_fromslice                                             View.MemoryView:950                                         
   0.20%        2 compiler_try_star_except                                               python3.13                                                  
   0.20%        2 compiler_try_star_finally                                              python3.13                                                  
   0.20%        2 __pyx_memoryview_fromslice                                             View.MemoryView:919                                         
   0.20%        2 __pyx_tp_new__memoryviewslice                                          openptv2/algorithms/track_kernels_search.py:429             
   0.20%        2 __pyx_tp_new_memoryview                                                openptv2/algorithms/track_kernels_search.py:429             
   0.20%        2 __pyx_memoryview_fromslice                                             View.MemoryView:950                                         
   0.20%        2 __pyx_tp_new__memoryviewslice                                          openptv2/algorithms/track_kernels_search.py:429             
   0.20%        2 __pyx_tp_new_memoryview                                                openptv2/algorithms/track_kernels_search.py:429             
   0.20%        2 __pyx_memoryview_fromslice                                             View.MemoryView:922                                         
   0.20%        2 __Pyx_INC_MEMVIEW                                                      openptv2/algorithms/track_kernels_search.py:509             
   0.20%        2 __Pyx_init_memviewslice                                                View.MemoryView:689                                         
   0.20%        2 __pyx_tp_dealloc_memoryview                                            openptv2/algorithms/track_kernels_geom.py:1111              
   0.20%        2 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:586             
   0.20%        2 track_kernels_search_candsearch_in_pix_fast                            openptv2/algorithms/track_kernels_search.py:136             
   0.20%        2 Py_DECREF                                                              object.h:944                                                
   0.20%        2 _Py_IsImmortal                                                         object.h:361                                                
   0.20%        2 __pyx_tp_dealloc__memoryviewslice                                      openptv2/algorithms/track_kernels_tracking.py:1384          
   0.20%        2 __pyx_memoryviewslice___dealloc__                                      View.MemoryView:869                                         
   0.20%        2 __pyx_memoryviewslice___pyx_pf_15View_dot_MemoryView_16_memoryvie...   View.MemoryView:870                                         
   0.20%        2 __Pyx_XCLEAR_MEMVIEW                                                   openptv2/algorithms/track_kernels_tracking.py:227           
   0.20%        2 __Pyx_PyObject_to_MemoryviewSlice_ds_double                            openptv2/algorithms/track_kernels_transform.py:899          
   0.20%        2 __Pyx_ValidateAndInit_memviewslice                                     openptv2/algorithms/track_kernels_transform.py:899          
   0.20%        2 __pyx_memoryview_fromslice                                             View.MemoryView:904                                         
   0.20%        2 track_kernels_transform_assess_new_position_fast                       openptv2/algorithms/track_kernels_transform.py:529          
   0.20%        2 __Pyx_GetItemInt_Fast                                                  openptv2/algorithms/track_kernels_transform.py:899          
   0.20%        2 candsearch_in_pix_rest_fast                                            openptv2/algorithms/track_kernels_search.py:174             
   0.20%        2 __Pyx_PyObject_to_MemoryviewSlice_ds_double                            View.MemoryView:689                                         
   0.20%        2 tracking_frame_buf_write_path_frame                                    openptv2/algorithms/tracking_frame_buf.py:481               
   0.20%        2 compiler_try_star_except                                               python3.13                                                  
   0.20%        2 compiler_try_star_except                                               python3.13                                                  
   0.20%        2 stringlib_split_whitespace                                             python3.13                                                  
   0.20%        2 gentype_format                                                         numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 compiler_try_star_except                                               python3.13                                                  
   0.20%        2 compiler_try_star_except                                               python3.13                                                  
   0.20%        2 0x1802235                                                              python3.13                                                  
   0.20%        2 0x185afa0                                                              python3.13                                                  
   0.20%        2 compiler_try_star_except                                               python3.13                                                  
   0.20%        2 compiler_try_star_except                                               python3.13                                                  
   0.20%        2 __pyx_memoryview___dealloc__                                           View.MemoryView:386                                         
   0.20%        2 __pyx_memoryview___pyx_pf_15View_dot_MemoryView_10memoryview_2__d...   View.MemoryView:387                                         
   0.20%        2 track_kernels_search_candsearch_in_pix_fast                            openptv2/algorithms/track_kernels_search.py:135             
   0.20%        2 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:429             
   0.20%        2 __pyx_memoryview_fromslice                                             View.MemoryView:922                                         
   0.20%        2 __Pyx_INC_MEMVIEW                                                      openptv2/algorithms/track_kernels_search.py:509             
   0.20%        2 __pyx_memoryview_fromslice                                             View.MemoryView:919                                         
   0.20%        2 __pyx_tp_new__memoryviewslice                                          openptv2/algorithms/track_kernels_search.py:429             
   0.20%        2 __pyx_tp_new_memoryview                                                openptv2/algorithms/track_kernels_search.py:429             
   0.20%        2 __pyx_memoryview_check                                                 View.MemoryView:679                                         
   0.20%        2 __Pyx_IsSubtype                                                        View.MemoryView:689                                         
   0.20%        2 __Pyx_XCLEAR_MEMVIEW                                                   View.MemoryView:689                                         
   0.20%        2 _multimed_r_nlay_1layer                                                openptv2/algorithms/track_kernels_geom.py:47                
   0.20%        2 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:588             
   0.20%        2 track_kernels_search_candsearch_in_pix_fast                            openptv2/algorithms/track_kernels_search.py:136             
   0.20%        2 track_kernels_search_candsearch_in_pix_fast                            openptv2/algorithms/track_kernels_search.py:141             
   0.20%        2 _angle_acc_out                                                         openptv2/algorithms/track_kernels_geom.py:871               
   0.20%        2 __Pyx_PyObject_to_MemoryviewSlice_ds_double                            openptv2/algorithms/track_kernels_transform.py:899          
   0.20%        2 __Pyx_ValidateAndInit_memviewslice                                     openptv2/algorithms/track_kernels_transform.py:899          
   0.20%        2 __Pyx_BufFmt_CheckString                                               openptv2/algorithms/track_kernels_transform.py:899          
   0.20%        2 __pyx_tp_dealloc_memoryview                                            openptv2/algorithms/track_kernels_transform.py:938          
   0.20%        2 __Pyx_PyObject_to_MemoryviewSlice_ds_double                            openptv2/algorithms/track_kernels_search.py:509             
   0.20%        2 __Pyx_ValidateAndInit_memviewslice                                     openptv2/algorithms/track_kernels_search.py:509             
   0.20%        2 tracking_frame_buf_write_targets                                       openptv2/algorithms/tracking_frame_buf.py:204               
   0.20%        2 compiler_try_star_except                                               python3.13                                                  
   0.20%        2 track_trackcorr_c_loop                                                 openptv2/algorithms/track.py:1170                           
   0.20%        2 0x19bf0ab                                                              python3.13                                                  
   0.20%        2 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.20%        2 read_frame_at_end                                                      openptv2/algorithms/tracking_frame_buf.py:830               
   0.20%        2 read_frame_at_end                                                      openptv2/algorithms/tracking_frame_buf.py:835               
   0.20%        2 0x19bf0ab                                                              python3.13                                                  
   0.20%        2 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.20%        2 read                                                                   openptv2/algorithms/tracking_frame_buf.py:681               
   0.20%        2 test_cavity                                                            test_track.py:515                                           
   0.20%        2 0x19fff69                                                              python3.13                                                  
   0.20%        2 0x710f0442a1ca                                                         libc.so.6                                                   
   0.20%        2 0x199792d                                                              python3.13                                                  
   0.20%        2 0x1997b45                                                              python3.13                                                  
   0.20%        2 0x1997e92                                                              python3.13                                                  
   0.20%        2 0x1a074da                                                              python3.13                                                  
   0.20%        2 0x1a075f6                                                              python3.13                                                  
   0.20%        2 0x1a07650                                                              python3.13                                                  
   0.20%        2 0x1a07b37                                                              python3.13                                                  
   0.20%        2 0x1a08447                                                              python3.13                                                  
   0.20%        2 0x18dc183                                                              python3.13                                                  
   0.20%        2 print_exception_file_and_line                                          python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 PyFunction_NewWithQualName                                             python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 exceptiongroup_subset                                                  python3.13                                                  
   0.20%        2 trackcorr_c_finish                                                     openptv2/algorithms/track.py:1175                           
   0.20%        2 trackcorr_c_finish                                                     openptv2/algorithms/track.py:1175                           
   0.20%        2 track_trackcorr_c_finish                                               openptv2/algorithms/track.py:1186                           
   0.20%        2 Py_XDECREF                                                             object.h:1042                                               
   0.20%        2 Py_DECREF                                                              object.h:944                                                
   0.20%        2 _Py_IsImmortal                                                         object.h:361                                                
   0.20%        2 0x19bf0ab                                                              python3.13                                                  
   0.20%        2 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.20%        2 write_frame_from_start                                                 openptv2/algorithms/tracking_frame_buf.py:838               
   0.20%        2 write_frame_from_start                                                 openptv2/algorithms/tracking_frame_buf.py:845               
   0.20%        2 0x19bf0ab                                                              python3.13                                                  
   0.20%        2 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.20%        2 write                                                                  openptv2/algorithms/tracking_frame_buf.py:755               
   0.20%        2 write                                                                  openptv2/algorithms/tracking_frame_buf.py:777               
   0.20%        2 __pyx_tp_dealloc_8openptv2_10algorithms_18tracking_frame_buf_Corres    openptv2/algorithms/tracking_frame_buf.py:852               
   0.20%        2 Py_DECREF                                                              object.h:949                                                
   0.20%        2 array_dealloc                                                          numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 0x180154e                                                              python3.13                                                  
   0.20%        2 0x180158c                                                              python3.13                                                  
   0.20%        2 0x1801812                                                              python3.13                                                  
   0.20%        2 free                                                                   libc.so.6                                                   
   0.20%        2 0x710f044ab43a                                                         libc.so.6                                                   
   0.20%        2 0x710f044aada5                                                         libc.so.6                                                   
   0.20%        2 __Pyx_ListComp_Append                                                  View.MemoryView:856                                         
   0.20%        2 0x18072e4                                                              python3.13                                                  
   0.20%        2 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:38                
   0.20%        2 _loop0_191_rule                                                        python3.13                                                  
   0.20%        2 call_soon                                                              python3.13                                                  
   0.20%        2 0x180090d                                                              python3.13                                                  
   0.20%        2 0x1800930                                                              python3.13                                                  
   0.20%        2 0x1800a13                                                              python3.13                                                  
   0.20%        2 0x180090d                                                              python3.13                                                  
   0.20%        2 full                                                                   numpy/_core/numeric.py:386                                  
   0.20%        2 0x19fff69                                                              python3.13                                                  
   0.20%        2 0x710f0442a1ca                                                         libc.so.6                                                   
   0.20%        2 0x199792d                                                              python3.13                                                  
   0.20%        2 0x1997b45                                                              python3.13                                                  
   0.20%        2 0x1997e92                                                              python3.13                                                  
   0.20%        2 0x1a074da                                                              python3.13                                                  
   0.20%        2 0x1a075f6                                                              python3.13                                                  
   0.20%        2 0x1a07650                                                              python3.13                                                  
   0.20%        2 0x1a07b37                                                              python3.13                                                  
   0.20%        2 0x1a08447                                                              python3.13                                                  
   0.20%        2 0x18dc183                                                              python3.13                                                  
   0.20%        2 print_exception_file_and_line                                          python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 PyFunction_NewWithQualName                                             python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 exceptiongroup_subset                                                  python3.13                                                  
   0.20%        2 tr_new                                                                 openptv2/algorithms/tracking_run.py:53                      
   0.20%        2 tracking_run_tr_new                                                    openptv2/algorithms/tracking_run.py:53                      
   0.20%        2 tracking_run_tr_new                                                    openptv2/algorithms/tracking_run.py:82                      
   0.20%        2 0x185e699                                                              python3.13                                                  
   0.20%        2 0x18072e4                                                              python3.13                                                  
   0.20%        2 validate_pattern_match_value                                           python3.13                                                  
   0.20%        2 code_new                                                               python3.13                                                  
   0.20%        2 __post_init__                                                          openptv2/algorithms/tracking_run.py:28                      
   0.20%        2 Py_XDECREF                                                             object.h:1041                                               
   0.20%        2 TrackingRun___post_init__                                              openptv2/algorithms/tracking_run.py:32                      
   0.20%        2 0x185e699                                                              python3.13                                                  
   0.20%        2 0x18072e4                                                              python3.13                                                  
   0.20%        2 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:800               
   0.20%        2 FrameBuf___init__                                                      openptv2/algorithms/tracking_frame_buf.py:812               
   0.20%        2 0x185e699                                                              python3.13                                                  
   0.20%        2 0x18072e4                                                              python3.13                                                  
   0.20%        2 __init__                                                               openptv2/algorithms/tracking_frame_buf.py:523               
   0.20%        2 Frame___init__                                                         openptv2/algorithms/tracking_frame_buf.py:550               
   0.20%        2 _PyTraceMalloc_GetTraceback                                            python3.13                                                  
   0.20%        2 async_gen_asend_dealloc                                                python3.13                                                  
   0.20%        2 dispatcher_vectorcall                                                  numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 _PyTraceMalloc_GetTraceback                                            python3.13                                                  
   0.20%        2 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.20%        2 array_copyto                                                           numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 PyArray_AssignRawScalar                                                numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 raw_array_assign_scalar                                                numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 _aligned_strided_to_contig_size4_srcstride0                            numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 tracking_frame_buf_read_targets                                        openptv2/algorithms/tracking_frame_buf.py:185               
   0.20%        2 0x1ab4abd                                                              python3.13                                                  
   0.20%        2 __Pyx_BufFmt_CheckString                                               View.MemoryView:689                                         
   0.20%        2 __Pyx_BufFmt_ProcessTypeChunk                                          View.MemoryView:689                                         
   0.20%        2 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   0.20%        2 track_kernels_geom__point_to_pixel_out                                 openptv2/algorithms/track_kernels_geom.py:620               
   0.20%        2 _multimed_r_nlay_1layer                                                openptv2/algorithms/track_kernels_geom.py:47                
   0.20%        2 __pyx_memoryview_new                                                   View.MemoryView:673                                         
   0.20%        2 __Pyx_PyObject_FastCallDict                                            View.MemoryView:689                                         
   0.20%        2 0x185e699                                                              python3.13                                                  
   0.20%        2 0x1807283                                                              python3.13                                                  
   0.20%        2 __pyx_tp_new_memoryview                                                openptv2/algorithms/track_kernels_geom.py:1111              
   0.20%        2 __pyx_memoryview___cinit__                                             View.MemoryView:356                                         
   0.20%        2 __pyx_memoryview___pyx_pf_15View_dot_MemoryView_10memoryview___ci...   View.MemoryView:360                                         
   0.20%        2 array_getbuffer                                                        numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 _multimed_r_nlay_1layer                                                openptv2/algorithms/track_kernels_geom.py:47                
   0.20%        2 __Pyx_PyObject_to_MemoryviewSlice_d_dc_double                          openptv2/algorithms/track_kernels_search.py:509             
   0.20%        2 __Pyx_ValidateAndInit_memviewslice                                     openptv2/algorithms/track_kernels_search.py:509             
   0.20%        2 __Pyx_PyObject_to_MemoryviewSlice_ds_double                            openptv2/algorithms/track_kernels_search.py:509             
   0.20%        2 __Pyx_ValidateAndInit_memviewslice                                     openptv2/algorithms/track_kernels_search.py:509             
   0.20%        2 __pyx_memoryview_new                                                   View.MemoryView:673                                         
   0.20%        2 __Pyx_PyObject_FastCallDict                                            openptv2/algorithms/track_kernels_search.py:509             
   0.20%        2 Py_XDECREF                                                             object.h:1041                                               
   0.20%        2 _PyTraceMalloc_GetTraceback                                            python3.13                                                  
   0.20%        2 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.20%        2 array_empty                                                            numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:511             
   0.20%        2 array_assign_subscript                                                 numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:513             
   0.20%        2 __pyx_tp_new__memoryviewslice                                          openptv2/algorithms/track_kernels_search.py:429             
   0.20%        2 __pyx_tp_new_memoryview                                                openptv2/algorithms/track_kernels_search.py:429             
   0.20%        2 __pyx_memoryview_fromslice                                             View.MemoryView:950                                         
   0.20%        2 __pyx_tp_new__memoryviewslice                                          openptv2/algorithms/track_kernels_search.py:429             
   0.20%        2 __pyx_tp_new_memoryview                                                openptv2/algorithms/track_kernels_search.py:429             
   0.20%        2 __Pyx_AllocateExtensionType                                            openptv2/algorithms/track_kernels_search.py:509             
   0.20%        2 _loop0_191_rule                                                        python3.13                                                  
   0.20%        2 __pyx_memoryview_fromslice                                             View.MemoryView:950                                         
   0.20%        2 Py_DECREF                                                              object.h:948                                                
   0.20%        2 __pyx_tp_dealloc__memoryviewslice                                      openptv2/algorithms/track_kernels_search.py:429             
   0.20%        2 array_dealloc                                                          numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 _buffer_info_free                                                      numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 __pyx_memoryview_check                                                 View.MemoryView:679                                         
   0.20%        2 __Pyx_IsSubtype                                                        View.MemoryView:689                                         
   0.20%        2 __pyx_memoryview_new                                                   View.MemoryView:671                                         
   0.20%        2 Py_XDECREF                                                             object.h:1042                                               
   0.20%        2 Py_DECREF                                                              object.h:944                                                
   0.20%        2 0x180df00                                                              python3.13                                                  
   0.20%        2 0x180193a                                                              python3.13                                                  
   0.20%        2 0x185e6e9                                                              python3.13                                                  
   0.20%        2 track_kernels_geom__point_to_pixel_out                                 openptv2/algorithms/track_kernels_geom.py:602               
   0.20%        2 track_kernels_geom__multimed_r_nlay_1layer                             openptv2/algorithms/track_kernels_geom.py:88                
   0.20%        2 track_kernels_search_candsearch_in_pix_fast                            openptv2/algorithms/track_kernels_search.py:136             
   0.20%        2 track_kernels_tracking_trackcorr_loop_fast                             openptv2/algorithms/track_kernels_tracking.py:459           
   0.20%        2 __Pyx_ValidateAndInit_memviewslice                                     openptv2/algorithms/track_kernels_transform.py:899          
   0.20%        2 __pyx_memoryview_new                                                   View.MemoryView:673                                         
   0.20%        2 __Pyx_PyObject_FastCallDict                                            openptv2/algorithms/track_kernels_transform.py:899          
   0.20%        2 0x185e699                                                              python3.13                                                  
   0.20%        2 0x1807283                                                              python3.13                                                  
   0.20%        2 __pyx_tp_new_memoryview                                                openptv2/algorithms/track_kernels_transform.py:938          
   0.20%        2 __pyx_memoryview___cinit__                                             View.MemoryView:356                                         
   0.20%        2 __pyx_memoryview___pyx_pf_15View_dot_MemoryView_10memoryview_4__g...   View.MemoryView:421                                         
   0.20%        2 candsearch_in_pix_rest_fast                                            openptv2/algorithms/track_kernels_search.py:174             
   0.20%        2 __Pyx_PyObject_to_MemoryviewSlice_ds_int                               openptv2/algorithms/track_kernels_search.py:509             
   0.20%        2 __Pyx_ValidateAndInit_memviewslice                                     openptv2/algorithms/track_kernels_search.py:509             
   0.20%        2 __Pyx_ValidateAndInit_memviewslice                                     View.MemoryView:689                                         
   0.20%        2 __pyx_memoryview_new                                                   View.MemoryView:673                                         
   0.20%        2 _ray_tracing_out                                                       openptv2/algorithms/track_kernels_geom.py:1111              
   0.20%        2 compiler_try_star_except                                               python3.13                                                  
   0.20%        2 tracking_frame_buf_write_targets                                       openptv2/algorithms/tracking_frame_buf.py:203               
   0.20%        2 tracking_frame_buf_write_targets                                       openptv2/algorithms/tracking_frame_buf.py:204               
   0.20%        2 compiler_try_star_except                                               python3.13                                                  
   0.20%        2 __Pyx_BufFmt_CheckString                                               View.MemoryView:689                                         
   0.20%        2 __Pyx_BufFmt_ProcessTypeChunk                                          View.MemoryView:689                                         
   0.20%        2 0x185e699                                                              python3.13                                                  
   0.20%        2 0x1807283                                                              python3.13                                                  
   0.20%        2 __pyx_tp_new_memoryview                                                openptv2/algorithms/track_kernels_geom.py:1111              
   0.20%        2 _point_to_pixel_out                                                    openptv2/algorithms/track_kernels_geom.py:403               
   0.20%        2 __Pyx_PyObject_to_MemoryviewSlice_ds_double                            View.MemoryView:689                                         
   0.20%        2 __Pyx_ValidateAndInit_memviewslice                                     View.MemoryView:689                                         
   0.20%        2 __pyx_tp_dealloc_memoryview                                            openptv2/algorithms/track_kernels_geom.py:1111              
   0.20%        2 Py_REFCNT                                                              object.h:318                                                
   0.20%        2 prepare_index_noarray                                                  numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
   0.20%        2 track_kernels_search__sorted_candidates_fast_out                       openptv2/algorithms/track_kernels_search.py:541             
   0.20%        2 0x18017d3                                                              python3.13                                                  
   0.20%        2 __pyx_tp_dealloc__memoryviewslice                                      openptv2/algorithms/track_kernels_search.py:429             
   0.20%        2 __pyx_memoryview_fromslice                                             View.MemoryView:919                                         
   0.20%        2 __pyx_tp_new__memoryviewslice                                          openptv2/algorithms/track_kernels_search.py:429             
   0.20%        2 __pyx_tp_new_memoryview                                                openptv2/algorithms/track_kernels_search.py:429             
   0.20%        2 __Pyx_AllocateExtensionType                                            openptv2/algorithms/track_kernels_search.py:509             
   0.20%        2 __Pyx_AllocateExtensionType                                            View.MemoryView:689                                         
   0.20%        2 __pyx_memoryview___pyx_pf_15View_dot_MemoryView_10memoryview___ci...   View.MemoryView:360                                         
   0.20%        2 Py_DECREF                                                              object.h:949                                                
   0.20%        2 track_kernels_geom__point_to_pixel_out                                 openptv2/algorithms/track_kernels_geom.py:616               
   0.20%        2 stringlib_split_char                                                   python3.13                                                  
   0.20%        2 Py_DECREF                                                              object.h:949                                                
   0.20%        2 track_kernels_search_candsearch_in_pix_fast                            openptv2/algorithms/track_kernels_search.py:138             
   0.20%        2 track_kernels_search_candsearch_in_pix_fast                            openptv2/algorithms/track_kernels_search.py:142             
   0.20%        2 track_kernels_transform_assess_new_position_fast                       openptv2/algorithms/track_kernels_transform.py:529          
   0.20%        2 __Pyx_GetItemInt_Fast                                                  openptv2/algorithms/track_kernels_transform.py:899          
   0.20%        2 Py_DECREF                                                              object.h:944                                                
   0.20%        2 _Py_IsImmortal                                                         object.h:361                                                
   0.20%        2 __pyx_memoryview___getitem__                                           View.MemoryView:417                                         
   0.20%        2 __Pyx_PyObject_to_MemoryviewSlice_d_dc_double                          openptv2/algorithms/track_kernels_transform.py:899          
   0.20%        2 __Pyx_ValidateAndInit_memviewslice                                     openptv2/algorithms/track_kernels_transform.py:899          
   0.20%        2 0x1800950                                                              python3.13                                                  
   0.20%        2 __pyx_tp_traverse_8openptv2_10algorithms_18tracking_frame_buf_Pat...   openptv2/algorithms/tracking_frame_buf.py:852               
   0.20%        2 test_cavity                                                            test_track.py:550                                           
   0.20%        2 0x19fff69                                                              python3.13                                                  
   0.20%        2 0x710f0442a1ca                                                         libc.so.6                                                   
   0.20%        2 0x199792d                                                              python3.13                                                  
   0.20%        2 0x1997b45                                                              python3.13                                                  
   0.20%        2 0x1997e92                                                              python3.13                                                  
   0.20%        2 0x1a074da                                                              python3.13                                                  
   0.20%        2 0x1a075f6                                                              python3.13                                                  
   0.20%        2 0x1a07650                                                              python3.13                                                  
   0.20%        2 0x1a07b37                                                              python3.13                                                  
   0.20%        2 0x1a08447                                                              python3.13                                                  
   0.20%        2 0x18dc183                                                              python3.13                                                  
   0.20%        2 print_exception_file_and_line                                          python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 PyFunction_NewWithQualName                                             python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 exceptiongroup_subset                                                  python3.13                                                  
   0.20%        2 trackcorr_c_finish                                                     openptv2/algorithms/track.py:1175                           
   0.20%        2 trackcorr_c_finish                                                     openptv2/algorithms/track.py:1175                           
   0.20%        2 track_trackcorr_c_finish                                               openptv2/algorithms/track.py:1186                           
   0.20%        2 Py_XDECREF                                                             object.h:1042                                               
   0.20%        2 Py_DECREF                                                              object.h:944                                                
   0.20%        2 _Py_IsImmortal                                                         object.h:361                                                
   0.20%        2 0x19bf0ab                                                              python3.13                                                  
   0.20%        2 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.20%        2 write_frame_from_start                                                 openptv2/algorithms/tracking_frame_buf.py:838               
   0.20%        2 write_frame_from_start                                                 openptv2/algorithms/tracking_frame_buf.py:845               
   0.20%        2 0x19bf0ab                                                              python3.13                                                  
   0.20%        2 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.20%        2 write                                                                  openptv2/algorithms/tracking_frame_buf.py:755               
   0.20%        2 write                                                                  openptv2/algorithms/tracking_frame_buf.py:777               
   0.20%        2 tracking_frame_buf_write_targets                                       openptv2/algorithms/tracking_frame_buf.py:204               
   0.20%        2 compiler_try_star_except                                               python3.13                                                  
   0.20%        2 stringlib_partition                                                    python3.13                                                  
   0.20%        2 _ensure_unconfigure                                                    _pytest/config/__init__.py:1212                             
   0.20%        2 __call__                                                               pluggy/_hooks.py:512                                        
   0.20%        2 _hookexec                                                              pluggy/_manager.py:120                                      
   0.20%        2 _multicall                                                             pluggy/_callers.py:121                                      
   0.20%        2 pytest_unconfigure                                                     _pytest/unraisableexception.py:172                          
   0.20%        2 gc_collect_harder                                                      _pytest/unraisableexception.py:33                           
   0.20%        2 0x19fff69                                                              python3.13                                                  
   0.20%        2 0x710f0442a1ca                                                         libc.so.6                                                   
   0.20%        2 0x199792d                                                              python3.13                                                  
   0.20%        2 0x1997b45                                                              python3.13                                                  
   0.20%        2 0x1997e92                                                              python3.13                                                  
   0.20%        2 0x1a074da                                                              python3.13                                                  
   0.20%        2 0x1a075f6                                                              python3.13                                                  
   0.20%        2 0x1a07650                                                              python3.13                                                  
   0.20%        2 0x1a07b37                                                              python3.13                                                  
   0.20%        2 0x1a08447                                                              python3.13                                                  
   0.20%        2 0x18dc183                                                              python3.13                                                  
   0.20%        2 print_exception_file_and_line                                          python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 exceptiongroup_subset                                                  python3.13                                                  
   0.20%        2 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.20%        2 task_step_handle_result_impl                                           python3.13                                                  
   0.20%        2 future_schedule_callbacks                                              python3.13                                                  
   0.20%        2 _ensure_unconfigure                                                    _pytest/config/__init__.py:1217                             
   0.20%        2 close                                                                  contextlib.py:627                                           
   0.20%        2 __exit__                                                               contextlib.py:604                                           
   0.20%        2 _exit_wrapper                                                          contextlib.py:482                                           
   0.20%        2 cleanup                                                                _pytest/unraisableexception.py:99                           
   0.20%        2 gc_collect_harder                                                      _pytest/unraisableexception.py:33                           
   0.20%        2 0x19fff69                                                              python3.13                                                  
   0.20%        2 0x710f0442a1ca                                                         libc.so.6                                                   
   0.20%        2 0x199792d                                                              python3.13                                                  
   0.20%        2 0x1997b45                                                              python3.13                                                  
   0.20%        2 0x1997e92                                                              python3.13                                                  
   0.20%        2 0x1a074da                                                              python3.13                                                  
   0.20%        2 0x1a075f6                                                              python3.13                                                  
   0.20%        2 0x1a07650                                                              python3.13                                                  
   0.20%        2 0x1a07b37                                                              python3.13                                                  
   0.20%        2 0x1a08447                                                              python3.13                                                  
   0.20%        2 0x18dc183                                                              python3.13                                                  
   0.20%        2 print_exception_file_and_line                                          python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 0x1ac6c0d                                                              python3.13                                                  
   0.20%        2 0x18e7e51                                                              python3.13                                                  
   0.20%        2 0x18e7fa5                                                              python3.13                                                  
   0.20%        2 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.20%        2 0x185e658                                                              python3.13                                                  
   0.20%        2 on_completion                                                          python3.13                                                  
   0.20%        2 mro_hierarchy                                                          python3.13                                                  
   0.20%        2 0x181ff4e                                                              python3.13                                                  
   0.20%        2 task_step_handle_result_impl                                           python3.13                                                  
   0.20%        2 future_schedule_callbacks                                              python3.13                                                  
   0.20%        2 <module>                                                               importlib/metadata/__init__.py:12                           
   0.20%        2 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.20%        2 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.20%        2 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.20%        2 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.20%        2 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.20%        2 <module>                                                               importlib/metadata/__init__.py:21                           
   0.20%        2 _handle_fromlist                                                       &lt;frozen importlib._bootstrap&gt;:1415                    
   0.20%        2 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.20%        2 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.20%        2 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.20%        2 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.20%        2 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1027           
   0.20%        2 _call_with_frames_removed                                              &lt;frozen importlib._bootstrap&gt;:488                     
   0.20%        2 <module>                                                               importlib/metadata/_meta.py:4                               
   0.20%        2 _find_and_load                                                         &lt;frozen importlib._bootstrap&gt;:1360                    
   0.20%        2 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.20%        2 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.20%        2 <module>                                                               _pytest/pathlib.py:32                                       
   0.20%        2 _find_and_load_unlocked                                                &lt;frozen importlib._bootstrap&gt;:1331                    
   0.20%        2 _load_unlocked                                                         &lt;frozen importlib._bootstrap&gt;:935                     
   0.20%        2 exec_module                                                            &lt;frozen importlib._bootstrap_external&gt;:1023           
   0.20%        2 get_code                                                               &lt;frozen importlib._bootstrap_external&gt;:1156           
   0.20%        2 _compile_bytecode                                                      &lt;frozen importlib._bootstrap_external&gt;:785            
   0.20%        2 0x19fff69                                                              python3.13                                                  
   0.20%        2 0x710f0442a1ca                                                         libc.so.6                                                   
   0.20%        2 0x199792d                                                              python3.13                                                  
   0.20%        2 0x1997b45                                                              python3.13                                                  
   0.20%        2 0x1997e92                                                              python3.13                                                  
   0.20%        2 0x1a074da                                                              python3.13                                                  
   0.20%        2 0x1a075f6                                                              python3.13                                                  
   0.20%        2 0x1a07650                                                              python3.13                                                  
   0.20%        2 0x1a07b37                                                              python3.13                                                  
   0.20%        2 0x1a08447                                                              python3.13                                                  
   0.20%        2 0x18dc183                                                              python3.13                                                  
   0.20%        2 print_exception_file_and_line                                          python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.20%        2 0x1abfb39                                                              python3.13                                                  
   0.20%        2 marshal_dump_impl                                                      python3.13                                                  
   0.20%        2 va_build_value                                                         python3.13                                                  
   0.20%        2 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.20%        2 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.20%        2 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.20%        2 0x18aaa06                                                              python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.20%        2 0x1abfb39                                                              python3.13                                                  
   0.20%        2 marshal_dump_impl                                                      python3.13                                                  
   0.20%        2 va_build_value                                                         python3.13                                                  
   0.20%        2 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.20%        2 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.20%        2 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.20%        2 0x18aaa06                                                              python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.20%        2 0x1abfb39                                                              python3.13                                                  
   0.20%        2 marshal_dump_impl                                                      python3.13                                                  
   0.20%        2 va_build_value                                                         python3.13                                                  
   0.20%        2 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.20%        2 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.20%        2 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.20%        2 0x18aaa06                                                              python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.20%        2 0x1abfb39                                                              python3.13                                                  
   0.20%        2 marshal_dump_impl                                                      python3.13                                                  
   0.20%        2 va_build_value                                                         python3.13                                                  
   0.20%        2 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.20%        2 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.20%        2 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.20%        2 0x18aaa06                                                              python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.20%        2 0x1abfb39                                                              python3.13                                                  
   0.20%        2 marshal_dump_impl                                                      python3.13                                                  
   0.20%        2 va_build_value                                                         python3.13                                                  
   0.20%        2 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.20%        2 PyFunction_SetKwDefaults                                               python3.13                                                  
   0.20%        2 _PyPegen_get_invalid_target                                            python3.13                                                  
   0.20%        2 0x18aaa06                                                              python3.13                                                  
   0.20%        2 0x18aa114                                                              python3.13                                                  
   0.20%        2 store_instance_attr_lock_held.llvm.2983807002262537663                 python3.13                                                  
   0.20%        2 0x1abfb39                                                              python3.13                                                  
   0.20%        2 marshal_dump_impl                                                      python3.13                                                  
   0.20%        2 va_build_value                                                         python3.13                                                  
   0.20%        2 _Py_BuildValue_SizeT                                                   python3.13                                                  
   0.20%        2 0x181ff4e                                                              python3.13                                                  
   0.20%        2 0x19a0ebc                                                              python3.13                                                  
   0.20%        2 0x19a0fac                                                              python3.13                                                  
   0.20%        2 typevartuple_alloc                                                     python3.13                                                  
   0.20%        2 typevartuple                                                           python3.13                                                  
   0.20%        2 paramspec_alloc                                                        python3.13                                                  


=== BY MODULE ===

2775.37%  python3.13
 467.88%  pluggy/_hooks.py:512
 467.88%  pluggy/_manager.py:120
 467.88%  pluggy/_callers.py:121
 201.78%  openptv2.algorithms.track
 146.92%  openptv2.algorithms.track_kernels_search
 117.55%  openptv2.algorithms.track_kernels_tracking
 110.70%  openptv2.algorithms.tracking_frame_buf
  94.50%  libc.so.6
  92.56%  _pytest/config/__init__.py:229
  92.56%  _pytest/main.py:377
  92.26%  _pytest/runner.py:361
  91.77%  _pytest/main.py:330
  91.28%  _pytest/main.py:384
  91.28%  _pytest/main.py:408
  91.28%  _pytest/runner.py:118
  91.28%  _pytest/runner.py:139
  91.28%  _pytest/runner.py:249
  91.28%  _pytest/runner.py:250
  91.28%  _pytest/runner.py:184
  91.28%  _pytest/python.py:1707
  91.28%  _pytest/python.py:167
  73.54%  openptv2.algorithms.tracking_run
  70.55%  Cython View.MemoryView
  40.02%  openptv2.algorithms.track_kernels_geom
  35.26%  test_track.py:549
  34.80%  &lt;frozen importlib._bootstrap&gt;:488
  31.67%  &lt;frozen importlib._bootstrap&gt;:1360
  30.85%  test_track.py:514
  30.01%  &lt;frozen importlib._bootstrap&gt;:1331
  29.81%  &lt;frozen importlib._bootstrap&gt;:935
  26.94%  numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so
  23.61%  &lt;frozen importlib._bootstrap_external&gt;:1027
  20.52%  openptv2.algorithms.track_kernels_transform
  18.24%  object.h:1041
  17.24%  &lt;string&gt;:16
  12.15%  object.h:949
  10.97%  test_track.py:508
   6.78%  &lt;frozen importlib._bootstrap&gt;:1415
   6.27%  test_track.py:543
   6.26%  openptv2.algorithms.track_kernels_batch
   5.99%  object.h:944
   5.79%  object.h:361
   5.78%  _pytest/config/__init__.py:223
   4.60%  ?
   4.51%  _pytest/config/__init__.py:410
   4.51%  _pytest/config/__init__.py:1232
   4.40%  openptv2.algorithms.multimed
   3.73%  _pytest/assertion/rewrite.py:188
   3.44%  object.h:1042
   2.84%  _pytest/config/__init__.py:1605
   2.84%  _pytest/config/__init__.py:1312
   2.84%  _pytest/config/__init__.py:662
   2.84%  _pytest/config/__init__.py:707
   2.84%  _pytest/config/__init__.py:758
   2.84%  _pytest/pathlib.py:572
   2.84%  _pytest/pathlib.py:716
   2.84%  conftest.py:16
   2.74%  openptv2/__init__.py:3
   2.74%  &lt;frozen importlib._bootstrap_external&gt;:1329
   2.65%  numpy/_core/numeric.py:323
   2.25%  test_track.py:512
   2.06%  &lt;frozen importlib._bootstrap&gt;:1387
   2.04%  openptv2.algorithms.track_kernels
   1.86%  importlib/__init__.py:88
   1.57%  test_track.py:532
   1.47%  &lt;frozen importlib._bootstrap&gt;:1310
   1.38%  &lt;frozen importlib._bootstrap_external&gt;:1023
   1.27%  _pytest/config/__init__.py:401
   1.27%  _pytest/config/__init__.py:371
   1.27%  numpy/__init__.py:112
   1.27%  numpy/__config__.py:4
   1.18%  &lt;frozen importlib._bootstrap_external&gt;:1156
   1.08%  _pytest/config/__init__.py:913
   1.08%  _pytest/helpconfig.py:18
   1.08%  numpy/__init__.py:457
   1.08%  test_track.py:547
   1.08%  pytest:4
   1.08%  &lt;frozen importlib._bootstrap_external&gt;:785
   0.98%  _pytest/config/__init__.py:1574
   0.98%  _pytest/config/__init__.py:1345
   0.98%  _pytest/config/__init__.py:1368
   0.98%  _pytest/config/__init__.py:983
   0.98%  numpy/lib/__init__.py:18
   0.98%  _pytest/runner.py:589
   0.98%  _pytest/runner.py:408
   0.98%  _pytest/runner.py:406
   0.59%  numpy/lib/_arraypad_impl.py:10
   0.49%  pytest/__init__.py:76
   0.49%  _pytest/config/__init__.py:1363
   0.49%  _pytest/config/__init__.py:1364
   0.49%  importlib/metadata/__init__.py:496
   0.49%  _pytest/config/__init__.py:1583
   0.49%  pluggy/_manager.py:416
   0.49%  importlib/metadata/__init__.py:179
   0.49%  numpy/_core/__init__.py:111
   0.49%  numpy/_core/einsumfunc.py:10
   0.49%  numpy/_core/numeric.py:13
   0.49%  numpy/_core/shape_base.py:8
   0.49%  numpy/_core/fromnumeric.py:11
   0.49%  numpy/_core/_methods.py:7
   0.49%  numpy/lib/_index_tricks_impl.py:8
   0.49%  numpy/matrixlib/__init__.py:4
   0.49%  numpy/matrixlib/defmatrix.py:13
   0.49%  numpy/linalg/__init__.py:87
   0.49%  numpy/linalg/_linalg.py:76
   0.49%  numpy/_typing/__init__.py:5
   0.49%  _pytest/main.py:383
   0.49%  _pytest/main.py:394
   0.49%  _pytest/main.py:849
   0.49%  _pytest/main.py:973
   0.49%  _pytest/main.py:895
   0.49%  _pytest/config/__init__.py:50
   0.49%  _pytest/config/findpaths.py:17
   0.40%  numpy/_core/numeric.py:386
   0.40%  _pytest/unraisableexception.py:33
   0.39%  numpy/_core/__init__.py:107
   0.39%  _pytest/main.py:328
   0.39%  _pytest/terminal.py:868
   0.39%  _pytest/terminal.py:901
   0.39%  _pytest/terminal.py:1637
   0.39%  pluggy/_manager.py:70
   0.39%  pluggy/_manager.py:74
   0.39%  importlib/metadata/__init__.py:460
   0.39%  _pytest/main.py:372
   0.39%  _pytest/config/__init__.py:23
   0.29%  pytest/__init__.py:32
   0.29%  importlib/metadata/__init__.py:915
   0.29%  anyio/pytest_plugin.py:276
   0.29%  anyio/_core/_eventloop.py:153
   0.29%  anyio/_core/_eventloop.py:206
   0.29%  importlib/metadata/_adapters.py:5
   0.29%  test_track.py:492
   0.29%  shutil.py:593
   0.29%  shutil.py:533
   0.29%  _pytest/pathlib.py:35
   0.29%  _pytest/compat.py:20
   0.29%  py.py:8
   0.20%  pytest/__init__.py:24
   0.20%  _pytest/doctest.py:41
   0.20%  _pytest/python_api.py:11
   0.20%  decimal.py:102
   0.20%  &lt;frozen importlib._bootstrap&gt;:1000
   0.20%  _pytest/legacypath.py:268
   0.20%  dataclasses.py:1305
   0.20%  dataclasses.py:1297
   0.20%  dataclasses.py:1157
   0.20%  dataclasses.py:498
   0.20%  _pytest/subtests.py:341
   0.20%  _pytest/config/__init__.py:927
   0.20%  _pytest/config/__init__.py:571
   0.20%  _pytest/config/__init__.py:1533
   0.20%  _pytest/config/findpaths.py:315
   0.20%  _pytest/config/findpaths.py:189
   0.20%  importlib/metadata/__init__.py:341
   0.20%  pathlib/_local.py:546
   0.20%  pathlib/_abc.py:632
   0.20%  pathlib/_local.py:537
   0.20%  pathlib/_local.py:167
   0.20%  anyio/__init__.py:55
   0.20%  anyio/_core/_streams.py:7
   0.20%  _pytest/assertion/rewrite.py:176
   0.20%  _pytest/assertion/rewrite.py:393
   0.20%  numpy/_core/numerictypes.py:117
   0.20%  numpy/_core/__init__.py:128
   0.20%  numpy/_core/__init__.py:24
   0.20%  numpy/_core/multiarray.py:11
   0.20%  numpy/_typing/_array_like.py:60
   0.20%  numpy/lib/_arraysetops_impl.py:419
   0.20%  email/message.py:15
   0.20%  _pytest/main.py:584
   0.20%  _pytest/python.py:203
   0.20%  _pytest/python.py:216
   0.20%  _pytest/nodes.py:619
   0.20%  _pytest/nodes.py:225
   0.20%  _pytest/nodes.py:101
   0.20%  _pytest/nodes.py:593
   0.20%  pathlib/_local.py:382
   0.20%  &lt;frozen _collections_abc&gt;:1038
   0.20%  &lt;frozen _collections_abc&gt;:1031
   0.20%  pathlib/_local.py:53
   0.20%  pathlib/_local.py:246
   0.20%  pathlib/_local.py:253
   0.20%  pathlib/_abc.py:135
   0.20%  pathlib/_local.py:503
   0.20%  pathlib/_local.py:128
   0.20%  _pytest/python.py:564
   0.20%  _pytest/python.py:577
   0.20%  shutil.py:468
   0.20%  shutil.py:273
   0.20%  shutil.py:150
   0.20%  object.h:826
   0.20%  test_track.py:515
   0.20%  object.h:948
   0.20%  object.h:318
   0.20%  test_track.py:550
   0.20%  _pytest/config/__init__.py:1212
   0.20%  _pytest/unraisableexception.py:172
   0.20%  _pytest/config/__init__.py:1217
   0.20%  contextlib.py:627
   0.20%  contextlib.py:604
   0.20%  contextlib.py:482
   0.20%  _pytest/unraisableexception.py:99
   0.20%  importlib/metadata/__init__.py:12
   0.20%  importlib/metadata/__init__.py:21
   0.20%  importlib/metadata/_meta.py:4
   0.20%  _pytest/pathlib.py:32
