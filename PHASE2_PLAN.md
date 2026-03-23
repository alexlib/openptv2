# Phase 2: Python Engine Alignment Plan

**Branch**: `phase2-python-engine-alignment`
**Date**: March 23, 2026
**Approach**: Minimal refactoring - align Python algorithms to match Cython API

---

## Guiding Principles

1. **GUI stays unchanged** - Currently works with optv, should continue to work
2. **Python engine adapts to Cython API** - Not the other way around
3. **Add visualization as optional feature** - Separate `track_with_viz()` method
4. **One workflow first** - Complete tracking pipeline before expanding

---

## Cython API to Match

### 1. `Tracker` Class (from `bindings/optv/tracker.pyx`)

```python
class Tracker:
    def __init__(self, ControlParams cpar, VolumeParams vpar,
                 TrackingParams tpar, SequenceParams spar, 
                 list cals, dict naming=None, flatten_tol=0.0001)
    
    def restart(self)
    def step_forward(self) -> bool
    def finalize(self)
    def full_forward(self)
    def step_forward_3d(self) -> bool  # optional
    def full_forward_3d(self)  # optional
    def full_backward(self)  # optional
    def current_step(self) -> int
```

### 2. `Target` Class (from `bindings/optv/tracking_framebuf.pyx`)

```python
class Target:
    def __init__(self, **kwd)  # pnr, x, y, n, nx, ny, sumg, tnr
    def tnr(self) -> int
    def set_tnr(tnr: int)
    def pnr(self) -> int
    def set_pnr(pnr: int)
    def pos(self) -> Tuple[float, float]  # (x, y)
    def set_pos(pos: Tuple[float, float])
    def count_pixels(self) -> Tuple[int, int, int]  # (n, nx, ny)
    def set_pixel_counts(n, nx, ny)
    def sum_grey_value(self) -> int
    def set_sum_grey_value(sumg: int)
```

### 3. `TargetArray` Class

```python
class TargetArray:
    def __init__(self, int size=0)
    def sort_y()  # in-place sort by Y coordinate
    def write(char *file_base, int frame_num)
    def __getitem__(self, int ix) -> Target
    def __len__(self) -> int
```

### 4. `Frame` Class

```python
class Frame:
    def __init__(self, num_cams, corres_file_base=None,
                 linkage_file_base=None, prio_file_base=None, 
                 target_file_base=None, frame_num=None)
    
    def read(corres_file_base, linkage_file_base,
             target_file_base, frame_num, prio_file_base)
    
    def positions() -> np.ndarray  # (n, 3) array of 3D positions
    def target_positions_for_camera(int cam) -> np.ndarray  # (n, 2) array
```

---

## Current Python Implementation Status

### ✅ Already Aligned (mostly)

- `algorithms/tracking_frame_buf.py`:
  - `Target` dataclass - has all fields and methods ✓
  - `TargetArray` class - exists but may need API alignment
  - `Frame` class - needs verification

### ⚠️ Needs Work

1. **`algorithms/track.py`**:
   - `Tracker` class is **commented out** (lines 1343-1432)
   - Need to uncomment and align with Cython API
   - Add `track_with_viz()` method for visualization

2. **Data structure alignment**:
   - Verify `Target` methods match Cython signatures exactly
   - Verify `TargetArray` API matches
   - Verify `Frame` API matches

3. **Visualization hooks**:
   - Add callback mechanism to tracking loop
   - Create `track_with_viz()` method that yields intermediate states

---

## Implementation Tasks

### Task 1: Align Tracker Class (HIGH PRIORITY)

**File**: `algorithms/track.py`

**Actions**:
1. Uncomment `Tracker` class
2. Match `__init__` signature to Cython (use tuple types for params)
3. Implement methods: `restart()`, `step_forward()`, `finalize()`, `full_forward()`
4. Add `current_step()` method
5. Ensure `run_info` uses `TrackingRun` from `algorithms/tracking_run.py`

**API**:
```python
class Tracker:
    def __init__(self, cpar, vpar, tpar, spar, cals, naming=None, flatten_tol=0.0001):
        # Match Cython signature exactly
        pass
    
    def restart(self):
        # Initialize tracking run
        pass
    
    def step_forward(self) -> bool:
        # Step forward, return True if more frames
        pass
    
    def finalize(self):
        # Finish tracking run
        pass
    
    def full_forward(self):
        # Complete forward tracking
        pass
    
    def current_step(self) -> int:
        # Return current step number
        pass
    
    # NEW: Visualization method
    def track_with_viz(self, callback, on_particle=None, on_algorithm_step=None):
        """
        Track with visualization callbacks.
        
        Args:
            callback: Called after each frame: callback(frame_num, state_dict)
            on_particle: Called for each particle (optional)
            on_algorithm_step: Called during algorithm steps (optional)
        """
        self.restart()
        while self.step_forward():
            state = self._get_current_state()
            callback(self.current_step(), state)
            yield state
        self.finalize()
```

---

### Task 2: Align Target Class (MEDIUM PRIORITY)

**File**: `algorithms/tracking_frame_buf.py`

**Actions**:
1. Verify all methods match Cython signatures
2. Ensure `__init__` accepts keyword arguments like Cython version
3. Add type hints for clarity

**Current status**: Already very close, just needs verification

---

### Task 3: Align TargetArray Class (MEDIUM PRIORITY)

**File**: `algorithms/tracking_frame_buf.py`

**Actions**:
1. Uncomment `TargetArray` class if needed
2. Ensure `sort_y()` works in-place and renumbers `pnr`
3. Verify `__getitem__` and `__len__` work correctly

---

### Task 4: Align Frame Class (MEDIUM PRIORITY)

**File**: `algorithms/tracking_frame_buf.py`

**Actions**:
1. Verify `positions()` returns (n, 3) numpy array
2. Verify `target_positions_for_camera(cam)` returns (n, 2) array
3. Ensure file I/O matches Cython behavior

---

### Task 5: Add Visualization Support (LOW PRIORITY - after basic alignment)

**File**: `algorithms/track.py` (new module: `algorithms/viz_hooks.py`)

**Actions**:
1. Create callback mechanism in tracking loop
2. Define state dictionary format for visualization
3. Create example visualization callback

**State Dictionary Format**:
```python
state = {
    'frame_number': int,
    'particles': np.ndarray,  # (N, 3) - 3D positions
    'correspondences': np.ndarray,  # (N, 5) - [nr, p0, p1, p2, p3]
    'search_volumes': list,  # Optional - for debugging
    'added_count': int,
    'lost_count': int
}
```

---

### Task 6: Engine Comparison Test (MEDIUM PRIORITY)

**File**: `tests/engine_comparison/test_tracker_parity.py`

**Actions**:
1. Create test that runs both optv and python trackers
2. Compare results with tolerance 1e-10
3. Verify both produce identical output files

---

### Task 7: GUI Debug Mode (LOW PRIORITY - after everything else)

**File**: `gui/pyptv/pyptv_gui.py`

**Actions**:
1. Add `--debug-mode` command line flag
2. When debug mode is on, use Python engine instead of optv
3. Add simple visualization panel (can be matplotlib-based)

---

## Testing Strategy

### Phase 2A: Basic Alignment
```bash
# Test that Python Tracker can be instantiated
python -c "from algorithms.track import Tracker; print('OK')"

# Test that methods exist and have correct signatures
python -c "
from algorithms.track import Tracker
import inspect
sig = inspect.signature(Tracker.__init__)
print(sig)
"
```

### Phase 2B: Parity Testing
```bash
# Run same data through both engines
pytest tests/engine_comparison/test_tracker_parity.py -v
```

### Phase 2C: Visualization Testing
```bash
# Test visualization callbacks
python examples/track_with_viz.py
```

---

## Success Criteria

- [ ] Python `Tracker` class has identical API to Cython version
- [ ] Python `Target`, `TargetArray`, `Frame` match Cython APIs
- [ ] GUI works unchanged with optv engine
- [ ] Python engine can be used as drop-in replacement
- [ ] `track_with_viz()` method works with callbacks
- [ ] Engine comparison tests pass (tolerance 1e-10)

---

## Next Steps

1. **Start with Task 1** - Align `Tracker` class in `algorithms/track.py`
2. **Verify Task 2-4** - Check data structures match
3. **Test basic workflow** - Run tracking with Python engine
4. **Add visualization** - Implement `track_with_viz()` method
5. **Create tests** - Engine comparison tests
6. **GUI integration** - Add debug mode flag (optional)

---

## Notes

- **Keep it simple**: Don't over-engineer the adapter layer
- **GUI first**: Ensure GUI continues to work with optv
- **Incremental**: One workflow at a time (tracking → correspondence → calibration)
- **Test early**: Run both engines on same data frequently
