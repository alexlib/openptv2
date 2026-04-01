# Plan: Incorporate track3d into openptv2

**Created**: 2026-04-02
**Status**: Ready for implementation

## Overview

| Step | Component | Files | Status |
|------|-----------|-------|--------|
| 1 | C unit tests | `lib/tests/check_track3d.c` (modify) | Pending |
| 2 | Cython validation tests | `bindings/tests/test_tracker.py` (modify) | Pending |
| 3 | Python algorithm | `algorithms/track.py` (modify) | Pending |
| 4 | Python tests | `algorithms/tests/test_19_track3d.py` (create) | Pending |
| 5 | Engine comparison | `algorithms/tests/test_20_track3d_engine_comparison.py` (create) | Pending |
| 6 | Tolerance config | `algorithms/tests/conftest.py` (modify) | Pending |

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| C implementation | ✅ Exists | `lib/src/track3d.c` (203 lines) |
| C integration tests | ✅ Exists | `lib/tests/check_track3d.c` (3 tests) |
| C unit tests | ❌ Missing | No tests for `find_candidates_in_3d()` |
| CMake integration | ✅ Exists | Lines 64-65, 126-127 |
| Cython bindings | ✅ Partial | `step_forward_3d()`, `full_forward_3d()` exist |
| Cython .pxd | ✅ Exists | `track3d_loop` declared |
| Cython tests | ⚠️ Minimal | Only "runs without error" checks |
| Python algorithm | ❌ Missing | `step_forward_3d()` raises NotImplementedError |
| Python tests | ❌ Missing | Skipped in `test_15_tracker.py` |

## Tolerance

- Engine comparison: **1e-5** (matches C test `EPS = 1E-5`)
- Python unit tests: **1e-7** (matches existing `tracker` tolerance)

## Reference Data

- `test_data/track/res_orig/particles.10001-10005` — 4 cameras, 5 frames
- `test_data/test_cavity/res_orig/rt_is.10001-10004` — 4 cameras, 4 frames
- `test_data/track/conf.yaml` — YAML configuration for track dataset

---

## Step 1: Add C Unit Tests for `find_candidates_in_3d()`

**File**: `lib/tests/check_track3d.c`

**What to add** — insert before `test_track3d_no_add`:

### 1.1 `test_find_candidates_in_3d_empty_frame`
```c
START_TEST(test_find_candidates_in_3d_empty_frame) {
    frame frm;
    frm.num_parts = 0;
    vec3d pos = {5.0, 5.0, 5.0};
    int indices[MAX_CANDS];
    int count = find_candidates_in_3d(&frm, pos, 1.0, 1.0, 1.0, indices, MAX_CANDS);
    ck_assert_msg(count == 0, "Expected 0 candidates, got %d", count);
}
END_TEST
```

### 1.2 `test_find_candidates_in_3d_single_match`
```c
START_TEST(test_find_candidates_in_3d_single_match) {
    frame frm;
    P path_info[MAX_TARGETS];
    frm.path_info = path_info;
    frm.num_parts = 1;
    frm.path_info[0].x[0] = 5.0;
    frm.path_info[0].x[1] = 5.0;
    frm.path_info[0].x[2] = 5.0;
    
    vec3d pos = {5.0, 5.0, 5.0};
    int indices[MAX_CANDS];
    int count = find_candidates_in_3d(&frm, pos, 1.0, 1.0, 1.0, indices, MAX_CANDS);
    ck_assert_msg(count == 1, "Expected 1 candidate, got %d", count);
    ck_assert_msg(indices[0] == 0, "Expected index 0, got %d", indices[0]);
}
END_TEST
```

### 1.3 `test_find_candidates_in_3d_no_match_outside_box`
```c
START_TEST(test_find_candidates_in_3d_no_match_outside_box) {
    frame frm;
    P path_info[MAX_TARGETS];
    frm.path_info = path_info;
    frm.num_parts = 1;
    frm.path_info[0].x[0] = 5.0;
    frm.path_info[0].x[1] = 5.0;
    frm.path_info[0].x[2] = 5.0;
    
    vec3d pos = {10.0, 10.0, 10.0};
    int indices[MAX_CANDS];
    int count = find_candidates_in_3d(&frm, pos, 1.0, 1.0, 1.0, indices, MAX_CANDS);
    ck_assert_msg(count == 0, "Expected 0 candidates, got %d", count);
}
END_TEST
```

### 1.4 `test_find_candidates_in_3d_multiple_matches`
```c
START_TEST(test_find_candidates_in_3d_multiple_matches) {
    frame frm;
    P path_info[MAX_TARGETS];
    frm.path_info = path_info;
    frm.num_parts = 5;
    
    // Particles at (0,0,0), (1,1,1), (5,5,5), (6,6,6), (10,10,10)
    double positions[5][3] = {{0,0,0}, {1,1,1}, {5,5,5}, {6,6,6}, {10,10,10}};
    for (int i = 0; i < 5; i++)
        for (int d = 0; d < 3; d++)
            frm.path_info[i].x[d] = positions[i][d];
    
    // Search near (5,5,5) with box size 2.0 — should find particles 2,3
    vec3d pos = {5.0, 5.0, 5.0};
    int indices[MAX_CANDS];
    int count = find_candidates_in_3d(&frm, pos, 2.0, 2.0, 2.0, indices, MAX_CANDS);
    ck_assert_msg(count == 2, "Expected 2 candidates, got %d", count);
}
END_TEST
```

### 1.5 `test_find_candidates_in_3d_max_cands_limit`
```c
START_TEST(test_find_candidates_in_3d_max_cands_limit) {
    frame frm;
    P path_info[MAX_TARGETS];
    frm.path_info = path_info;
    frm.num_parts = 10;
    
    // All particles within small volume
    for (int i = 0; i < 10; i++) {
        frm.path_info[i].x[0] = 5.0 + i * 0.01;
        frm.path_info[i].x[1] = 5.0;
        frm.path_info[i].x[2] = 5.0;
    }
    
    vec3d pos = {5.0, 5.0, 5.0};
    int indices[3];  // Only room for 3
    int count = find_candidates_in_3d(&frm, pos, 1.0, 1.0, 1.0, indices, 3);
    ck_assert_msg(count == 3, "Expected 3 candidates (max), got %d", count);
}
END_TEST
```

### 1.6 `test_find_candidates_in_3d_boundary`
```c
START_TEST(test_find_candidates_in_3d_boundary) {
    frame frm;
    P path_info[MAX_TARGETS];
    frm.path_info = path_info;
    frm.num_parts = 1;
    frm.path_info[0].x[0] = 6.0;  // Exactly pos + dx
    frm.path_info[0].x[1] = 5.0;
    frm.path_info[0].x[2] = 5.0;
    
    vec3d pos = {5.0, 5.0, 5.0};
    int indices[MAX_CANDS];
    // Uses < not <=, so boundary point should be excluded
    int count = find_candidates_in_3d(&frm, pos, 1.0, 1.0, 1.0, indices, MAX_CANDS);
    ck_assert_msg(count == 0, "Expected 0 (boundary excluded), got %d", count);
}
END_TEST
```

### Register in suite
Add to `track3d_suite()`:
```c
tc_core = tcase_create("find_candidates_in_3d");
tcase_add_test(tc_core, test_find_candidates_in_3d_empty_frame);
tcase_add_test(tc_core, test_find_candidates_in_3d_single_match);
tcase_add_test(tc_core, test_find_candidates_in_3d_no_match_outside_box);
tcase_add_test(tc_core, test_find_candidates_in_3d_multiple_matches);
tcase_add_test(tc_core, test_find_candidates_in_3d_max_cands_limit);
tcase_add_test(tc_core, test_find_candidates_in_3d_boundary);
suite_add_tcase(s, tc_core);
```

---

## Step 2: Add Cython Output Validation Tests

**File**: `bindings/tests/test_tracker.py`

**What to add** — inside `TestTracker` class:

### 2.1 `test_forward_3d_output_matches_reference`
```python
def test_forward_3d_output_matches_reference(self):
    """Verify track3d output matches reference res_orig/ files."""
    import numpy as np
    
    shutil.copytree("test_data/track/res_orig/", "test_data/track/res/")
    self.tracker.full_forward_3d()
    
    for step in range(10001, 10005):
        out_file = f"test_data/track/res/particles.{step}"
        ref_file = f"test_data/track/res_orig/particles.{step}"
        
        self.assertTrue(os.path.exists(out_file), f"Missing output: {out_file}")
        
        with open(out_file) as f_out, open(ref_file) as f_ref:
            out_lines = f_out.readlines()
            ref_lines = f_ref.readlines()
            
            # First line is particle count
            self.assertEqual(out_lines[0].strip(), ref_lines[0].strip(),
                           f"Particle count mismatch at step {step}")
            
            # Compare positions
            out_parts = [list(map(float, l.split())) for l in out_lines[1:]]
            ref_parts = [list(map(float, l.split())) for l in ref_lines[1:]]
            
            for i, (o, r) in enumerate(zip(out_parts, ref_parts)):
                np.testing.assert_allclose(o, r, atol=1e-5,
                    err_msg=f"Position mismatch at step {step}, particle {i}")
```

### 2.2 `test_forward_3d_step_by_step_output`
```python
def test_forward_3d_step_by_step_output(self):
    """Verify step_forward_3d produces correct per-step output."""
    shutil.copytree("test_data/track/res_orig/", "test_data/track/res/")
    
    self.tracker.restart()
    last_step = 10001
    while self.tracker.step_forward_3d():
        self.assertGreater(self.tracker.current_step(), last_step)
        
        # Verify output file exists for completed step
        out_file = f"test_data/track/res/particles.{last_step}"
        self.assertTrue(os.path.exists(out_file),
                       f"Missing output after step {last_step}")
        
        last_step += 1
    self.tracker.finalize()
```

---

## Step 3: Implement Python `track3d_loop`

**File**: `algorithms/track.py`

### 3.1 Add `find_candidates_in_3d` function

Insert after `sort()` function (line ~590):

```python
def find_candidates_in_3d(frm, pos, dx, dy, dz, max_cands=MAX_CANDS):
    """
    Find particles within a 3D box centered at pos.
    
    Arguments:
        frm - Frame object with path_info list
        pos - (3,) array-like, center position
        dx, dy, dz - box half-sizes in each dimension
        max_cands - maximum candidates to return
    
    Returns:
        list of particle indices within the box
    """
    indices = []
    for i in range(frm.num_parts):
        pi = frm.path_info[i]
        if (abs(pi.x[0] - pos[0]) < dx and
            abs(pi.x[1] - pos[1]) < dy and
            abs(pi.x[2] - pos[2]) < dz):
            if len(indices) < max_cands:
                indices.append(i)
    return indices
```

### 3.2 Add `track3d_loop` function

Insert after `trackcorr_c_finish()` function:

```python
def track3d_loop(run_info, step):
    """
    3D tracking loop - links particles in 3D space without camera projection.
    
    Three-level linking strategy:
    1. Particles with previous links: predict = 2*curr - prev
    2. No prev link, neighbors have links: predict = curr + avg_neighbor_velocity
    3. No prev link, no neighbor links: predict = curr
    
    Arguments:
        run_info - TrackingRun object
        step - current frame number
    """
    import math
    
    fb = run_info.fb
    tpar = run_info.tpar
    
    prev = fb.buf[0]
    curr = fb.buf[1]
    next_buf = fb.buf[2]
    
    orig_parts = curr.num_parts
    count1 = 0
    
    dx = tpar.dvxmax
    dy = tpar.dvymax
    dz = tpar.dvzmax
    
    # Level 1: Particles with previous links
    for i in range(orig_parts):
        curr_pi = curr.path_info[i]
        if curr_pi.prev_frame < 0:
            continue
        prev_idx = curr_pi.prev_frame
        if prev_idx < 0 or prev_idx >= prev.num_parts:
            continue
        prev_pi = prev.path_info[prev_idx]
        
        # Predict: 2*curr - prev
        predicted = 2 * curr_pi.x - prev_pi.x
        
        cand_indices = find_candidates_in_3d(next_buf, predicted, dx, dy, dz)
        
        decis = [0.0] * len(cand_indices)
        linkdecis = [0] * len(cand_indices)
        
        for k, cidx in enumerate(cand_indices):
            acc = 0.0
            for d in range(3):
                diff = curr_pi.x[d] - 2 * next_buf.path_info[cidx].x[d] + prev_pi.x[d]
                acc += diff * diff
            decis[k] = math.sqrt(acc)
            linkdecis[k] = cidx
        
        if len(cand_indices) > 1:
            sort(len(decis), decis, linkdecis)
        
        if cand_indices and next_buf.path_info[linkdecis[0]].prev_frame < 0:
            curr_pi.next_frame = linkdecis[0]
            next_buf.path_info[linkdecis[0]].prev_frame = i
            count1 += 1
        else:
            curr_pi.next_frame = -1
    
    # Level 2: No previous link, but neighbors have previous links
    for i in range(orig_parts):
        curr_pi = curr.path_info[i]
        if curr_pi.prev_frame >= 0 or curr_pi.next_frame >= 0:
            continue
        
        vel = np.zeros(3)
        nvel = 0
        for j in range(orig_parts):
            if j == i:
                continue
            nbr = curr.path_info[j]
            if (abs(nbr.x[0] - curr_pi.x[0]) < dx and
                abs(nbr.x[1] - curr_pi.x[1]) < dy and
                abs(nbr.x[2] - curr_pi.x[2]) < dz and
                nbr.prev_frame >= 0):
                vel += nbr.x - prev.path_info[nbr.prev_frame].x
                nvel += 1
        
        if nvel == 0:
            continue
        vel /= nvel
        predicted = curr_pi.x + vel
        
        cand_indices = find_candidates_in_3d(next_buf, predicted, dx, dy, dz)
        
        decis = [0.0] * len(cand_indices)
        linkdecis = [0] * len(cand_indices)
        
        for k, cidx in enumerate(cand_indices):
            acc = 0.0
            for d in range(3):
                diff = curr_pi.x[d] - 2 * next_buf.path_info[cidx].x[d] + predicted[d]
                acc += diff * diff
            decis[k] = math.sqrt(acc)
            linkdecis[k] = cidx
        
        if len(cand_indices) > 1:
            sort(len(decis), decis, linkdecis)
        
        if cand_indices and next_buf.path_info[linkdecis[0]].prev_frame < 0:
            curr_pi.next_frame = linkdecis[0]
            next_buf.path_info[linkdecis[0]].prev_frame = i
            count1 += 1
        else:
            curr_pi.next_frame = -1
    
    # Level 3: No previous link, no neighbors with previous links
    for i in range(orig_parts):
        curr_pi = curr.path_info[i]
        if curr_pi.prev_frame >= 0 or curr_pi.next_frame >= 0:
            continue
        
        predicted = curr_pi.x.copy()
        
        cand_indices = find_candidates_in_3d(next_buf, predicted, dx, dy, dz)
        
        decis = [0.0] * len(cand_indices)
        linkdecis = [0] * len(cand_indices)
        
        for k, cidx in enumerate(cand_indices):
            acc = 0.0
            for d in range(3):
                diff = curr_pi.x[d] - 2 * next_buf.path_info[cidx].x[d] + predicted[d]
                acc += diff * diff
            decis[k] = math.sqrt(acc)
            linkdecis[k] = cidx
        
        if len(cand_indices) > 1:
            sort(len(decis), decis, linkdecis)
        
        if cand_indices and next_buf.path_info[linkdecis[0]].prev_frame < 0:
            curr_pi.next_frame = linkdecis[0]
            next_buf.path_info[linkdecis[0]].prev_frame = i
            count1 += 1
        else:
            curr_pi.next_frame = -1
    
    print(f"track3d step: {step}, curr: {fb.buf[1].num_parts}, "
          f"next: {fb.buf[2].num_parts}, links: {count1}")
    
    run_info.npart += fb.buf[1].num_parts
    run_info.nlinks += count1
    
    fb.fb_next()
    fb.write_frame_from_start(step)
    if step < run_info.seq_par.last - 2:
        fb.read_frame_at_end(step + 3, 0)
```

### 3.3 Update `Tracker` class methods

Replace `step_forward_3d()` (line ~1453):
```python
def step_forward_3d(self):
    """
    Perform one 3D tracking step for the current frame of iteration.
    
    Returns:
        bool: True if more frames to process, False if done.
    """
    if self.step >= self.run_info.seq_par.last:
        return False
    
    track3d_loop(self.run_info, self.step)
    self.step += 1
    return True
```

Replace `full_forward_3d()` (line ~1461):
```python
def full_forward_3d(self):
    """Do a full 3D tracking run from restart to finalize."""
    track_forward_start(self.run_info)
    for step in range(self.run_info.seq_par.first, self.run_info.seq_par.last):
        track3d_loop(self.run_info, step)
    trackcorr_c_finish(self.run_info, self.run_info.seq_par.last)
    self.step = 0
```

---

## Step 4: Create Python Tests for track3d

**File**: `algorithms/tests/test_19_track3d.py` (create new)

```python
"""
Tests for Python track3d_loop implementation.

Mirrors the C tests in lib/tests/check_track3d.c and the Cython tests
in bindings/tests/test_tracker.py to ensure identical behavior.

Tolerance: 1e-7 (full tracking pipeline)
"""

import os
import shutil
import math
import yaml
import numpy as np
import pytest

from .conftest import get_tolerance

TOLERANCE = get_tolerance("tracker")

TRACK_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "test_data", "track"
)

framebuf_naming = {
    "corres": "test_data/track/res/particles",
    "linkage": "test_data/track/res/linkage",
    "prio": "test_data/track/res/whatever",
}


def _load_cals_from_yaml(yaml_conf):
    from algorithms.calibration import Calibration
    cals = []
    for cam_spec in yaml_conf["cameras"]:
        cal = Calibration()
        cal.from_file(cam_spec["ori_file"], cam_spec.get("addpar_file", None))
        cals.append(cal)
    return cals


def _build_python_tracker(yaml_conf):
    from algorithms.parameters import ControlPar, VolumePar, TrackParTuple, SequencePar
    from algorithms.track import Tracker

    seq_cfg = yaml_conf["sequence"]
    scene = yaml_conf["scene"]
    corresp = yaml_conf["correspondences"]
    tracking = yaml_conf["tracking"]

    cals = _load_cals_from_yaml(yaml_conf)

    cpar = ControlPar(num_cams=len(yaml_conf["cameras"]))
    cpar.imx = scene["image_size"][0]
    cpar.imy = scene["image_size"][1]
    cpar.pix_x = scene["pixel_size"][0]
    cpar.pix_y = scene["pixel_size"][1]

    vpar = VolumePar(
        x_lay=corresp["x_span"],
        z_min_lay=corresp["z_spans"][0],
        z_max_lay=corresp["z_spans"][1],
    )

    vel = tracking["velocity_lims"]
    tpar = TrackParTuple(
        dvxmin=vel[0][0], dvxmax=vel[0][1],
        dvymin=vel[1][0], dvymax=vel[1][1],
        dvzmin=vel[2][0], dvzmax=vel[2][1],
        dangle=tracking["angle_lim"],
        dacc=tracking["accel_lim"],
        add=tracking["add_particle"],
        dsumg=0.0, dn=0.0, dnx=0.0, dny=0.0,
    )

    img_base = [
        seq_cfg["targets_template"].format(cam=cix + 1)
        for cix in range(len(yaml_conf["cameras"]))
    ]
    spar = SequencePar(
        img_base_name=img_base,
        first=seq_cfg["first"],
        last=seq_cfg["last"],
    )

    return Tracker(cpar, vpar, tpar, spar, cals, framebuf_naming)


@pytest.fixture
def track_test_dir(tmp_path):
    """Set up temporary copy of track test data."""
    src = TRACK_DATA_DIR
    res_orig = os.path.join(src, "res_orig")
    res_dst = os.path.join(src, "res")
    if os.path.exists(res_dst):
        shutil.rmtree(res_dst)
    shutil.copytree(res_orig, res_dst)
    newpart_dir = os.path.join(src, "newpart")
    backup_dir = str(tmp_path / "newpart_backup")
    shutil.copytree(newpart_dir, backup_dir)
    yield src
    if os.path.exists(res_dst):
        shutil.rmtree(res_dst)
    if os.path.exists(newpart_dir):
        shutil.rmtree(newpart_dir)
    shutil.copytree(backup_dir, newpart_dir)


class TestFindCandidatesIn3D:
    """Unit tests for find_candidates_in_3d function."""
    
    def test_empty_frame(self):
        """Empty frame returns no candidates."""
        from algorithms.track import find_candidates_in_3d
        from algorithms.tracking_frame_buf import Frame
        
        frm = Frame(num_cams=1, max_targets=10)
        frm.num_parts = 0
        
        indices = find_candidates_in_3d(frm, np.array([5.0, 5.0, 5.0]), 1.0, 1.0, 1.0)
        assert len(indices) == 0
    
    def test_single_match(self):
        """Single particle within box is found."""
        from algorithms.track import find_candidates_in_3d
        from algorithms.tracking_frame_buf import Frame, Pathinfo
        
        frm = Frame(num_cams=1, max_targets=10)
        frm.num_parts = 1
        frm.path_info[0].x = np.array([5.0, 5.0, 5.0])
        
        indices = find_candidates_in_3d(frm, np.array([5.0, 5.0, 5.0]), 1.0, 1.0, 1.0)
        assert len(indices) == 1
        assert indices[0] == 0
    
    def test_no_match_outside_box(self):
        """Particle outside box is not found."""
        from algorithms.track import find_candidates_in_3d
        from algorithms.tracking_frame_buf import Frame
        
        frm = Frame(num_cams=1, max_targets=10)
        frm.num_parts = 1
        frm.path_info[0].x = np.array([5.0, 5.0, 5.0])
        
        indices = find_candidates_in_3d(frm, np.array([10.0, 10.0, 10.0]), 1.0, 1.0, 1.0)
        assert len(indices) == 0
    
    def test_multiple_matches(self):
        """Multiple particles within box are found."""
        from algorithms.track import find_candidates_in_3d
        from algorithms.tracking_frame_buf import Frame
        
        frm = Frame(num_cams=1, max_targets=10)
        frm.num_parts = 5
        positions = [[0, 0, 0], [1, 1, 1], [5, 5, 5], [6, 6, 6], [10, 10, 10]]
        for i, pos in enumerate(positions):
            frm.path_info[i].x = np.array(pos, dtype=float)
        
        indices = find_candidates_in_3d(frm, np.array([5.0, 5.0, 5.0]), 2.0, 2.0, 2.0)
        assert len(indices) == 2
        assert 2 in indices  # particle at (5,5,5)
        assert 3 in indices  # particle at (6,6,6)
    
    def test_max_cands_limit(self):
        """Result count is limited by max_cands."""
        from algorithms.track import find_candidates_in_3d
        from algorithms.tracking_frame_buf import Frame
        
        frm = Frame(num_cams=1, max_targets=20)
        frm.num_parts = 10
        for i in range(10):
            frm.path_info[i].x = np.array([5.0 + i * 0.01, 5.0, 5.0])
        
        indices = find_candidates_in_3d(frm, np.array([5.0, 5.0, 5.0]), 1.0, 1.0, 1.0, max_cands=3)
        assert len(indices) == 3
    
    def test_boundary_excluded(self):
        """Particle exactly on boundary is excluded (< not <=)."""
        from algorithms.track import find_candidates_in_3d
        from algorithms.tracking_frame_buf import Frame
        
        frm = Frame(num_cams=1, max_targets=10)
        frm.num_parts = 1
        frm.path_info[0].x = np.array([6.0, 5.0, 5.0])  # pos + dx
        
        indices = find_candidates_in_3d(frm, np.array([5.0, 5.0, 5.0]), 1.0, 1.0, 1.0)
        assert len(indices) == 0


class TestTrack3DLoop:
    """Integration tests for track3d_loop via Tracker class."""
    
    def _make_tracker(self):
        with open(os.path.join(TRACK_DATA_DIR, "conf.yaml")) as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
        return _build_python_tracker(yaml_conf)
    
    def test_step_forward_3d(self, track_test_dir):
        """Manual step_forward_3d run."""
        tracker = self._make_tracker()
        tracker.restart()
        last_step = 10001
        while tracker.step_forward_3d():
            assert tracker.current_step() > last_step
            last_step += 1
        tracker.finalize()
    
    def test_full_forward_3d(self, track_test_dir):
        """Automatic full_forward_3d run."""
        tracker = self._make_tracker()
        tracker.full_forward_3d()
    
    def test_full_forward_3d_produces_output(self, track_test_dir):
        """Verify output files are created."""
        tracker = self._make_tracker()
        tracker.full_forward_3d()
        for step in range(10001, 10005):
            path = f"test_data/track/res/particles.{step}"
            assert os.path.exists(path), f"Missing output: {path}"
    
    def test_forward_3d_no_not_implemented_error(self, track_test_dir):
        """Verify NotImplementedError is no longer raised."""
        tracker = self._make_tracker()
        tracker.restart()
        # Should not raise
        result = tracker.step_forward_3d()
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Step 5: Create Engine Comparison Tests

**File**: `algorithms/tests/test_20_track3d_engine_comparison.py` (create new)

```python
"""
Engine comparison tests for track3d: Cython vs Python.

Runs both engines on the same data and compares outputs frame-by-frame.

Tolerance: 1e-5 (matches C test EPS)
"""

import os
import shutil
import yaml
import numpy as np
import pytest

TOLERANCE = 1e-5

TRACK_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "test_data", "track"
)
CAVITY_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "test_data", "test_cavity"
)


def _read_particles_file(filepath):
    """Read a particles or rt_is file and return (count, positions array)."""
    with open(filepath) as f:
        lines = f.readlines()
    count = int(lines[0].strip())
    positions = []
    for line in lines[1:count + 1]:
        parts = list(map(float, line.split()))
        positions.append(parts[:3])  # x, y, z
    return count, np.array(positions) if positions else np.empty((0, 3))


@pytest.fixture(params=["track", "test_cavity"])
def dataset(request, tmp_path):
    """Parametrized fixture for both datasets."""
    if request.param == "track":
        src = TRACK_DATA_DIR
        yaml_file = "conf.yaml"
        ref_prefix = "particles"
        frame_range = range(10001, 10005)
    else:
        src = CAVITY_DATA_DIR
        yaml_file = "parameters_Run1.yaml"
        ref_prefix = "rt_is"
        frame_range = range(10001, 10004)
    
    res_orig = os.path.join(src, "res_orig")
    res_dst = os.path.join(src, "res")
    if os.path.exists(res_dst):
        shutil.rmtree(res_dst)
    shutil.copytree(res_orig, res_dst)
    
    # Also backup newpart/targets if track dataset
    if request.param == "track":
        newpart_dir = os.path.join(src, "newpart")
        backup_dir = str(tmp_path / "newpart_backup")
        shutil.copytree(newpart_dir, backup_dir)
    
    yield src, yaml_file, ref_prefix, frame_range
    
    if os.path.exists(res_dst):
        shutil.rmtree(res_dst)
    if request.param == "track":
        newpart_dir = os.path.join(src, "newpart")
        if os.path.exists(newpart_dir):
            shutil.rmtree(newpart_dir)
        shutil.copytree(backup_dir, newpart_dir)


class TestTrack3DEngineComparison:
    """Compare Cython and Python track3d implementations."""
    
    def test_python_track3d_matches_reference(self, dataset):
        """Python track3d output matches reference data."""
        src, yaml_file, ref_prefix, frame_range = dataset
        
        with open(os.path.join(src, yaml_file)) as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
        
        naming = {
            "corres": f"{src}/res/rt_is",
            "linkage": f"{src}/res/ptv_is",
            "prio": f"{src}/res/added",
        }
        
        # Build Python tracker
        from algorithms.parameters import ControlPar, VolumePar, TrackParTuple, SequencePar
        from algorithms.track import Tracker
        from algorithms.calibration import Calibration
        
        cals = []
        for cam_spec in yaml_conf["cameras"]:
            cal = Calibration()
            cal.from_file(cam_spec["ori_file"], cam_spec.get("addpar_file", None))
            cals.append(cal)
        
        scene = yaml_conf["scene"]
        seq_cfg = yaml_conf["sequence"]
        corresp = yaml_conf["correspondences"]
        tracking = yaml_conf["tracking"]
        
        cpar = ControlPar(num_cams=len(yaml_conf["cameras"]))
        cpar.imx = scene["image_size"][0]
        cpar.imy = scene["image_size"][1]
        cpar.pix_x = scene["pixel_size"][0]
        cpar.pix_y = scene["pixel_size"][1]
        
        vpar = VolumePar(
            x_lay=corresp["x_span"],
            z_min_lay=corresp["z_spans"][0],
            z_max_lay=corresp["z_spans"][1],
        )
        
        vel = tracking["velocity_lims"]
        tpar = TrackParTuple(
            dvxmin=vel[0][0], dvxmax=vel[0][1],
            dvymin=vel[1][0], dvymax=vel[1][1],
            dvzmin=vel[2][0], dvzmax=vel[2][1],
            dangle=tracking["angle_lim"],
            dacc=tracking["accel_lim"],
            add=tracking["add_particle"],
            dsumg=0.0, dn=0.0, dnx=0.0, dny=0.0,
        )
        
        img_base = [seq_cfg["targets_template"].format(cam=cix + 1) 
                    for cix in range(len(yaml_conf["cameras"]))]
        spar = SequencePar(
            img_base_name=img_base,
            first=seq_cfg["first"],
            last=seq_cfg["last"],
        )
        
        tracker = Tracker(cpar, vpar, tpar, spar, cals, naming)
        tracker.full_forward_3d()
        
        for step in frame_range:
            out_file = os.path.join(src, "res", f"{ref_prefix}.{step}")
            ref_file = os.path.join(src, "res_orig", f"{ref_prefix}.{step}")
            
            assert os.path.exists(out_file), f"Missing output: {out_file}"
            
            out_count, out_pos = _read_particles_file(out_file)
            ref_count, ref_pos = _read_particles_file(ref_file)
            
            assert out_count == ref_count, \
                f"Step {step}: particle count {out_count} != {ref_count}"
            
            if ref_count > 0:
                np.testing.assert_allclose(out_pos, ref_pos, atol=TOLERANCE,
                    err_msg=f"Step {step}: positions differ")
    
    def test_cython_track3d_matches_reference(self, dataset):
        """Cython track3d output matches reference data."""
        src, yaml_file, ref_prefix, frame_range = dataset
        
        with open(os.path.join(src, yaml_file)) as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
        
        naming = {
            "corres": f"{src}/res/rt_is".encode(),
            "linkage": f"{src}/res/ptv_is".encode(),
            "prio": f"{src}/res/added".encode(),
        }
        
        from optv.tracker import Tracker as CythonTracker
        from optv.calibration import Calibration as CythonCal
        from optv.parameters import ControlParams, VolumeParams, TrackingParams, SequenceParams
        
        cals = []
        for cam_spec in yaml_conf["cameras"]:
            cal = CythonCal()
            ori = cam_spec["ori_file"]
            addpar = cam_spec.get("addpar_file")
            if addpar:
                cal.from_file(ori.encode(), addpar.encode())
            else:
                cal.from_file(ori.encode(), b"")
            cals.append(cal)
        
        scene = yaml_conf["scene"]
        cpar = ControlParams(len(yaml_conf["cameras"]), **scene)
        vpar = VolumeParams(**yaml_conf["correspondences"])
        tpar = TrackingParams(**yaml_conf["tracking"])
        
        seq_cfg = yaml_conf["sequence"]
        img_base = []
        for cix in range(len(yaml_conf["cameras"])):
            img_base.append(seq_cfg["targets_template"].format(cam=cix + 1).encode())
        spar = SequenceParams(
            image_base=img_base,
            frame_range=(seq_cfg["first"], seq_cfg["last"]),
        )
        
        tracker = CythonTracker(cpar, vpar, tpar, spar, cals, naming)
        tracker.full_forward_3d()
        
        for step in frame_range:
            out_file = os.path.join(src, "res", f"{ref_prefix}.{step}")
            ref_file = os.path.join(src, "res_orig", f"{ref_prefix}.{step}")
            
            assert os.path.exists(out_file), f"Missing output: {out_file}"
            
            out_count, out_pos = _read_particles_file(out_file)
            ref_count, ref_pos = _read_particles_file(ref_file)
            
            assert out_count == ref_count, \
                f"Step {step}: particle count {out_count} != {ref_count}"
            
            if ref_count > 0:
                np.testing.assert_allclose(out_pos, ref_pos, atol=TOLERANCE,
                    err_msg=f"Step {step}: positions differ")
```

---

## Step 6: Update conftest.py

**File**: `algorithms/tests/conftest.py`

Add to `TOLERANCES` dict (line ~35):
```python
"track3d": 1e-5,
```

---

## Execution Order

1. **Step 1** — C unit tests (independent)
2. **Step 3** — Python algorithm (must precede Step 4)
3. **Step 4** — Python tests (depends on Step 3)
4. **Step 2** — Cython validation tests (can parallel with Step 3-4)
5. **Step 5** — Engine comparison (depends on Steps 2-4)
6. **Step 6** — conftest update (trivial, do with Step 4)

## Verification

```bash
# C tests
cd lib/build && cmake .. && make && ctest -R check_track3d -V

# Python unit tests
pytest algorithms/tests/test_19_track3d.py -v

# Cython validation tests
pytest bindings/tests/test_tracker.py::TestTracker::test_forward_3d_output_matches_reference -v

# Engine comparison
pytest algorithms/tests/test_20_track3d_engine_comparison.py -v

# All tests
pytest -v
```
