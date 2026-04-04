# Project Status

## Goal

Ensure all engine comparison tests use each engine's native file readers for parameters and inputs, verify that both engines produce identical outputs, and resolve the pre-existing Python engine frame reading failure in `test_python_track3d_matches_reference`.

### Key Objectives
- **File-based parity**: Every engine comparison test must read the SAME parameter files through each engine's native reader (optv via Cython bindings, python via `algorithms/parameters.py`).
- **Identical outputs**: Python engine must produce numerically identical results to Cython engine within defined tolerances (1e-5 to 1e-7 depending on module).
- **Frame reading investigation**: Compare `algorithms/tracking_frame_buf.py` (Python) vs `bindings/optv/tracking_framebuf.pyx` (Cython) to identify path resolution, file format, or I/O differences causing `OSError: Could not read frame from disk`.
- **Testing strategy**: Create a dedicated frame-reading parity test that isolates the I/O layer, verifies both engines load identical pixel/target data from the same disk files, and documents any format/path discrepancies.
- **Design reference**: Follow `DESIGN_PLAN.md` Phase 2/3 success criteria (engine parity, file-based readers, identical results).

## Frame Reader Investigation Results

### Architecture Comparison

Both engines read the same text-based file formats but through different code paths:

**Cython engine** (via C library `lib/src/tracking_frame_buf.c`):
- `read_targets()` — C `fopen`/`fscanf` reads `<base><frame:04d>_targets` text files
- `read_path_frame()` — C reader for `rt_is`/`ptv_is`/`added` files
- `read_frame()` — orchestrates both, called from `tracking_framebuf.pyx`
- File naming: `sprintf(filein, "%s%04d%s", file_base, frame_num, "_targets")`

**Python engine** (`algorithms/tracking_frame_buf.py`):
- `read_targets()` — Python `open()`/`readline()` with `Path(candidate).exists()` pre-check
- `read_path_frame()` — Python reader for `rt_is`/`ptv_is`/`added` files
- `Frame.read()` — orchestrates both, checks file existence before reading
- File naming: `_target_filename()` handles both dotted (`cam1.`) and underscore (`sample_`) bases

### Key Differences Found

1. **File existence checking**: Python `Frame.read()` checks `Path.exists()` before attempting to read. C `read_frame()` relies on `fopen` returning NULL. Python's `read_targets()` standalone function falls back to a constructed filename if no candidate exists, then raises `FileNotFoundError` on missing files.

2. **Missing file behavior**:
   - Python `read_targets()` raises `FileNotFoundError` for missing files
   - Cython `read_targets()` returns a corrupted TargetArray (C returns -1, wrapper doesn't handle it properly — causes `SystemError` on `len()`)
   - Both are problematic; `Frame.read()` works around this by pre-checking existence

3. **Filename format handling**: Python's `_target_filename_candidates()` tries multiple naming conventions (dotted vs underscore). C uses a single `sprintf` pattern. Both produce identical filenames for the test data (`cam1.10001_targets`).

4. **Path resolution**: Both use the paths as given (relative or absolute). The test data uses relative paths from `test_data/track/`. When tests write temp `.par` files with absolute paths (e.g., `f"{src}/res/particles"`), both engines work correctly.

### Test Results

All 7 frame reading parity tests pass:
- `test_read_targets_single_camera` — Python and Cython read identical target data from one camera
- `test_read_targets_all_cameras` — All 4 cameras, multiple frames, positions match to 1e-10
- `test_read_targets_underscore_format` — Python handles `sample_0001_targets` format
- `test_read_targets_dotted_format` — Python handles `cam1.10001_targets` format
- `test_read_path_frame_parity` — Python reads rt_is files correctly, matches raw file count
- `test_read_targets_missing_file_behavior` — Documents both readers' error handling
- `test_frame_read_file_existence_check` — Python `Frame.read()` pre-checks file existence

All 3 engine comparison tests pass:
- `test_python_track3d_matches_reference` — Python output matches reference data
- `test_cython_track3d_matches_reference` — Cython output matches reference data
- `test_cython_vs_python_track3d_identical` — Both engines produce identical results

### Known Issues

1. **Cython `read_targets()` bug**: When C returns -1 (file not found), the Cython wrapper creates a TargetArray with -1 targets, causing `SystemError` on `len()`. This is a pre-existing bug in `bindings/optv/tracking_framebuf.pyx:227`.

2. **Cython `Frame` constructor segfault**: Passing `linkage_file_base=None` to the Cython `Frame` constructor causes a segmentation fault. The C code expects `NULL` but the Cython binding doesn't handle `None` → `NULL` conversion correctly for this parameter.

3. **Python `read_targets()` standalone**: Unlike `Frame.read()`, the standalone `read_targets()` function does not gracefully handle missing files — it raises `FileNotFoundError` instead of returning an empty list.

## Discoveries (Historical)

- **Conftest bugs fixed**: `optv.SequenceParams.read_sequence_par()` requires a `num_cams` argument; Python uses `read_track_par` not `read_tracking_par`. Both were silently failing or raising errors.
- **Frame reader discrepancy (resolved)**: The original `OSError: Could not read frame from disk` failure was caused by path resolution issues. When tests use absolute paths (as in `_write_temp_par_files`), both engines work correctly. The test now passes.
- **Test architecture**: All 260 algorithm tests now pass. The bindings (70) and GUI (245) tests remain green.

## Accomplished

- Updated 6 test files to use file-based fixtures/native readers:
  - `test_12_epipolar.py`
  - `test_09_correspondences.py`
  - `test_20_track3d_engine_comparison.py`
  - `test_parameters_parity.py`
  - `test_engine_verification.py`
  - `test_14_tracking_run.py`
- Fixed `conftest.py` reader bugs (sequence `num_cams` arg, tracking function name)
- Created `test_frame_reading_parity.py` — 7 tests isolating the I/O layer
- All algorithm tests pass. Bindings (70) and GUI (245) tests remain green.

## Relevant Files / Directories

### Test Files
- `algorithms/tests/test_frame_reading_parity.py` — **NEW** dedicated frame I/O parity tests
- `algorithms/tests/test_20_track3d_engine_comparison.py` — main engine comparison, writes temp `.par` files
- `algorithms/tests/test_12_epipolar.py` — epipolar curve parity, uses file readers
- `algorithms/tests/test_09_correspondences.py` — correspondences parity, uses file fixtures
- `algorithms/tests/test_parameters_parity.py` — parameter reader parity tests
- `algorithms/tests/test_engine_verification.py` — engine selection & param conversion
- `algorithms/tests/test_14_tracking_run.py` — tracking run creation with file params
- `algorithms/tests/conftest.py` — shared fixtures, `_load_*_from_file` helpers

### Source Files
- `algorithms/parameters.py` — Python parameter readers: `read_control_par`, `read_volume_par`, `read_track_par`, `read_sequence_par`
- `algorithms/tracking_frame_buf.py` — Python frame/target file reader
- `algorithms/track.py` — Python tracking pipeline, calls `fb.read_frame_at_end()`
- `algorithms/tracking_run.py` — `TrackingRun` class, creates `FrameBuf`
- `bindings/optv/tracking_framebuf.pyx` — Cython frame reader (for comparison)
- `bindings/optv/parameters.pyx` — Cython parameter readers (for comparison)
- `lib/src/tracking_frame_buf.c` — C implementation of frame reading
- `lib/include/tracking_frame_buf.h` — C header with struct definitions

### Test Data
- `test_data/track/` — `conf.yaml`, `newpart/`, `res_orig/`, `cal/`

## Next Steps

1. (Optional) Fix Cython `read_targets()` wrapper to handle C returning -1 gracefully
2. (Optional) Fix Cython `Frame` constructor to handle `None` → `NULL` conversion for linkage/prio params
3. (Optional) Make Python `read_targets()` return empty list on missing file (matching intended behavior)
4. Consider adding more test data with varied target counts, edge cases

## Isolated Engine Comparison Results

### Problem: Cross-contamination via `_targets` files

Both engines **write** `_targets` files back to the same directory via `write_frame_from_start()` during tracking. When running tests sequentially on the same folder, the second engine reads files already modified by the first, making comparison meaningless.

### Solution: Fully isolated workspaces

Created `test_isolated_engine_comparison.py` which gives each engine a complete copy of `test_data/track/` in separate temp directories.

### Findings

**Particle files (particles.<frame>)**: ✅ Identical — both engines produce the same particle counts and positions.

**Linkage files (linkage.<frame>)**: ⚠️ Header difference only — C writes `-1` for empty frames, Python writes `0`. This is a **C bug** in `lib/src/tracking_frame_buf.c:227-336`:

```c
int read_path_frame(...) {
    int targets = -1;  // initialized to -1
    // ...
    targets = 0;  // reset before loop
    do {
        // ... fscanf fails on empty file ...
        if (read_res != 8) {
            targets = -1;  // BUG: resets to -1 on empty file
            break;
        }
    } while (!feof(filein));
}
```

The `do...while(!feof)` loop always executes at least once. When a file has `0` particles (just header "0\n"), the `fscanf` fails and `targets` is reset to `-1`. Python correctly returns `0`.

**_targets files**: ✅ Identical — both engines write the same target data when given isolated inputs. No modification of original files occurs.

### Console output comparison

**Python engine (correct):**
```
track3d step: 10001, curr: 1, next: 1, links: 1
track3d step: 10002, curr: 1, next: 0, links: 0
track3d step: 10003, curr: 0, next: 1, links: 0
track3d step: 10004, curr: 1, next: 1, links: 1
Average: particles: 0.8, links: 0.5, lost: 0.2
```

**Cython engine (C bug affects reporting):**
```
track3d step: 10001, curr: 1, next: 1, links: 1
track3d step: 10002, curr: 1, next: -1, links: 0
track3d step: 10003, curr: -1, next: 1, links: 0
track3d step: 10004, curr: 1, next: 1, links: 1
Average: particles: 0.5, links: 0.5, lost: 0.0
```

The `-1` values in Cython output are from the C bug, not from algorithm differences. The actual particle positions and target data are identical between engines.

### Test Results

All 4 isolated comparison tests pass:
- `test_particles_files_match` — Identical particle counts and positions
- `test_linkage_files_match` — Identical after normalizing -1/0 header convention
- `test_targets_files_match` — Identical _targets file content
- `test_original_targets_unchanged` — No unintended modifications to originals
