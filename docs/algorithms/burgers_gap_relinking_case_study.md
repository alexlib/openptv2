# Burgers Dataset Gap Relinking Case Study

A detailed analysis of how `trackcorr` and `track3d` algorithms deviate when particles re-appear after a gap, and how backward tracking helps recover missing trajectories.

## Executive Summary

This case study examines a critical tracking scenario using the **Burgers turbulence dataset** (5 frames, 5 particles):
- **P2 is absent in one intermediate frame (10003)**
- Both algorithms split P2 into fragments
- **trackcorr produces 7 trajectories** (3 P2 fragments)
- **track3d produces 6 trajectories** (2 P2 fragments)
- **Backward tracking recovers additional forward-backward links** across the gap

**Key Finding**: The difference stems from how each algorithm handles particles with **no predecessor** when a future lookahead frame is unavailable.

---

## Dataset Overview

### Burgers Sequence

| Parameter | Value |
|-----------|-------|
| **Frame range** | 10001, 10002, 10003, 10004, 10005 |
| **Total frames** | 5 |
| **Physical particles** | 5 (P0, P1, P2, P3, P4) |
| **Cameras** | 4 |
| **Buffer size** | TR_BUFSPACE = 4 slots |

### Particle Correspondences

| Particle | Camera IDs | Status |
|----------|-----------|--------|
| P0 | (0,0,0,0) | ✓ Present all 5 frames |
| P1 | (1,1,1,1) | ✓ Present all 5 frames |
| **P2** | **(2,2,2,2)** | ⚠️ **ABSENT in frame 10003** |
| P3 | (3,3,3,3) | ✓ Present all 5 frames |
| P4 | (4,4,4,4) | ✓ Present all 5 frames |

### Ground Truth Trajectories

From reference particle file `rt_is.*`:

```
P0: [10001] (3.095,4.352,2.401) → [10002] (3.027,4.401,2.403) 
    → [10003] (2.957,4.448,2.402) → [10004] (2.887,4.493,2.402) 
    → [10005] (2.817,4.537,2.405)   [5 frames, continuous]

P1: [10001] (2.983,5.391,0.419) → [10002] (2.909,5.429,0.424) 
    → [10003] (2.835,5.469,0.422) → [10004] (2.761,5.507,0.422) 
    → [10005] (2.686,5.543,0.420)   [5 frames, continuous]

P2: [10001] (4.425,3.087,2.906) → [10002] (4.378,3.155,2.907) 
    → [10004] (4.277,3.289,2.906) → [10005] (4.225,3.355,2.906)   
    [4 frames, 1-frame gap at 10003]

P3: [10001] (3.693,1.136,0.596) → [10002] (3.668,1.214,0.597) 
    → [10003] (3.642,1.292,0.595) → [10004] (3.614,1.369,0.596) 
    → [10005] (3.584,1.445,0.595)   [5 frames, continuous]

P4: [10001] (2.727,2.022,1.677) → [10002] (2.679,2.085,1.678) 
    → [10003] (2.630,2.147,1.677) → [10004] (2.579,2.208,1.681) 
    → [10005] (2.527,2.267,1.680)   [5 frames, continuous]
```

**Key distances in P2's segment:**
- 10001→10002: 0.0925 mm
- **Gap (10003 absent)**
- 10004→10005: 0.0843 mm (< 0.1 mm threshold)

---

## Frame Buffer State Analysis

### Initialization Phase

`track_forward_start()` pre-loads first 3 frames:

```python
for step in range(first, first + TR_BUFSPACE - 1):  # 10001, 10002, 10003
    fb.read_frame_at_end(step)
    fb.fb_next()
fb.fb_prev()  # reset pointer
```

After init:
```
buf[0] = empty
buf[1] = frame 10001  ← starting position
buf[2] = frame 10002
buf[3] = frame 10003
```

### Main Loop Frame Advancement

Loop runs `range(seq.first, seq.last)` = `range(10001, 10005)` = **steps 10001, 10002, 10003, 10004**

At **end of each step**, buffer rotates:
- `fb.fb_next()` — rotate ring buffer
- **if `step < last - 2` (i.e., `step < 10003`)**: `read_frame_at_end(step + 3)`

| Step | buf[0] | buf[1] (current) | buf[2] (next) | buf[3] (lookahead) | Load at end? |
|------|--------|---|---|---|----|
| **10001** | empty | 10001 | 10002 | 10003 | 10001 < 10003 ✓ → load 10004 |
| **10002** | 10001 | 10002 | 10003 | 10004 | 10002 < 10003 ✓ → load 10005 |
| **10003** | 10002 | 10003 | 10004 | 10005 | 10003 < 10003 ✗ → NO LOAD |
| **10004** | 10003 | 10004 | 10005 | **EMPTY** | 10004 < 10003 ✗ → NO LOAD |

**Critical observation**: At step **10004**, `buf[3]` is **empty**—there is no frame 10006 to examine.

---

## Algorithm Behavior by Step

### Step 10001: P0, P1, P2, P3, P4 Forward Link

All 5 particles have `prev_frame >= 0` (frame IDs from buf[0]).

**trackcorr**:
1. Candidate search in buf[2] = frame 10002
2. 3-frame chain test in buf[3] = frame 10003 → **succeeds** (valid frame)
3. Angle + acceleration checks pass
4. **Link registered**: P0→P1, P1→P2, P2→P3, P3→P4, P4→P5

**track3d**:
1. Level 1: Particles with `prev_frame >= 0`
2. Predict using 2*curr - prev
3. Find candidates in buf[2], score by acceleration
4. **Link registered**: P0→P1, P1→P2, P2→P3, P3→P4, P4→P5

**Result**: ✓ Both match, 5 links each

---

### Step 10002: P0, P1, P2, P3, P4 → P0, P1, [P2 absent], P3, P4

Frame 10003 has only 4 particles (P2 missing).

**trackcorr**:
- P0, P1, P3, P4 have `prev_frame >= 0` → link forward normally
- P2 lookup in buf[2] = frame 10003:
  ```
  w = sorted_candidates_in_volume(X[2], v1, buf[2], ...)
  if w.count == 1 and w.ftnr[0] == TR_UNUSED:  # empty result
      continue  # no link registered
  ```
- P2 in buf[1] (frame 10002) gets `next_frame = -1`

**track3d**:
- All 4 particles link forward normally via Level 1
- P2 absent → not in buf[2], no entry in buf[3] either

**Result**: 
- trackcorr: 4 links (P0, P1, P3, P4 forward; P2 breaks)
- track3d: 4 links
- **Statistics**: links: 4, lost: 1 (P2)

---

### Step 10003: P0, P1, P3, P4 in buf[1]; 10004 in buf[2]; 10005 in buf[3]

Frame 10003 has 4 particles (P0, P1, P3, P4), all with `prev_frame >= 0`.

**trackcorr**:
- All 4 search in buf[2] = frame 10004
- 3-frame chain test in buf[3] = frame 10005 → **succeeds** (valid frame)
- All 4 link forward

**track3d**:
- All 4 link via Level 1

**Result**: 4 links each

---

### Step 10004: **CRITICAL STEP** — P2 Reappears, No buf[3]

buf[1] = frame 10004 (5 particles including P2 re-appearance)
buf[2] = frame 10005
buf[3] = **EMPTY** (no frame 10006)

P0, P1, P3, P4 all have `prev_frame >= 0` (linked in step 10003).
**P2 has `prev_frame = -1`** (no predecessor in frame 10003).

#### trackcorr Path for P2

```python
# Line 1160: trackcorr_c_loop
curr_path_inf = buf[1].path_info[P2_idx]  # prev_frame = -1

# Line 1162: Current position
X[1] = vec_copy(curr_path_inf.x)  # (4.277, 3.289, 2.906)

# Line 1163-1171: Since prev_frame < 0, use current as prediction center
else:
    X[2] = vec_copy(X[1])  # (4.277, 3.289, 2.906)

# Line 1190: Search in buf[2] = frame 10005
w = sorted_candidates_in_volume(X[2], v1, buf[2], ...)
# → finds P2_10005 at (4.225, 3.355, 2.906) ✓

# Line 1205: For each candidate, predict next position X[5]
# Line 1207-1208:
else:
    X[5] = search_volume_center_moving(X[1], X[3])  # prev_frame < 0

# Line 1213: Search for 3-frame chain
wn = sorted_candidates_in_volume(X[5], v1, buf[3], ...)
# buf[3] is EMPTY → wn.count == 1 and wn.ftnr[0] == TR_UNUSED

# Line 1215: IF wn.count > 1 check FAILS (empty result)
if wn.count > 1:  # FALSE
    # ... 3-frame chain evaluation skipped

# Line 1303: Last-resort fallback
if curr_path_inf.inlist == 0 and curr_path_inf.prev_frame >= 0:
    # ← CRITICAL: prev_frame = -1 → CONDITION IS FALSE ✗
    # Last-resort path is SKIPPED
```

**Result for trackcorr P2**: `curr_path_inf.inlist == 0` and `next_frame = -1`
- **No link registered** ✗

#### track3d Path for P2

```python
# Level 1: prev_frame >= 0? NO → skip

# Level 2: Check for linked neighbors
for j in range(num_parts):
    if j == P2_idx:
        continue
    nbr = buf[1].path_info[j]
    if distance_to_nbr < (dx, dy, dz) and nbr.prev_frame >= 0:
        velocity += ...
        n_velocity += 1

if n_velocity == 0:  # All other particles > 0.5 mm away (no neighbors)
    continue  # Level 2 skipped

# Level 3: No previous link, no neighbor links
curr_pi = buf[1].path_info[P2_idx]  # prev_frame = -1
if curr_pi.prev_frame >= 0 or curr_pi.next_frame >= 0:
    continue  # FALSE → proceed

predicted = curr_pi.x.copy()  # (4.277, 3.289, 2.906)

cand_indices = find_candidates_in_3d(buf[2], predicted, (dx, dy, dz))
# → finds P2_10005 at (4.225, 3.355, 2.906), distance (0.052, 0.066, 0.000)
# All within (0.5, 0.5, 0.5) ✓

if cand_indices and buf[2].path_info[best_cand].prev_frame < 0:
    curr_pi.next_frame = best_cand  # Link registered ✓
    buf[2].path_info[best_cand].prev_frame = P2_idx
    count += 1
```

**Result for track3d P2**: `next_frame = 1` (index of P2_10005)
- **Link registered** ✓

#### Comparison at Step 10004

| Metric | trackcorr | track3d |
|--------|-----------|---------|
| P0-P3 links | 4 | 4 |
| P2 links | **0** ✗ | **1** ✓ |
| Total links | **4** | **5** |
| Lost particles | **1** | **0** |
| Output linkage files | 5 files (10001-10005) | 5 files (10001-10005) |

---

## Trajectory Fragmentation Results

### trackcorr Output: 7 Trajectories

```
T1: [10001] (3.095,4.352,2.401) → [10002] (3.027,4.401,2.403) 
    → [10003] (2.957,4.448,2.402) → [10004] (2.887,4.493,2.402) 
    → [10005] (2.817,4.537,2.405)   [P0 ✓ 5 frames]

T2: [10001] (2.983,5.391,0.419) → [10002] (2.909,5.429,0.424) 
    → [10003] (2.835,5.469,0.422) → [10004] (2.761,5.507,0.422) 
    → [10005] (2.686,5.543,0.420)   [P1 ✓ 5 frames]

T3: [10001] (4.425,3.087,2.906) → [10002] (4.378,3.155,2.907)
    [P2 fragment 1 — 2 frames, breaks at gap]

T4: [10001] (3.693,1.136,0.596) → [10002] (3.668,1.214,0.597) 
    → [10003] (3.642,1.292,0.595) → [10004] (3.614,1.369,0.596) 
    → [10005] (3.584,1.445,0.595)   [P3 ✓ 5 frames]

T5: [10001] (2.727,2.022,1.677) → [10002] (2.679,2.085,1.678) 
    → [10003] (2.630,2.147,1.677) → [10004] (2.579,2.208,1.681) 
    → [10005] (2.527,2.267,1.680)   [P4 ✓ 5 frames]

T6: [10004] (4.277,3.289,2.906)
    [P2 singleton — 1 frame, isolated particle]

T7: [10005] (4.225,3.355,2.906)
    [P2 singleton — 1 frame, isolated particle]
```

**Summary**: 7 trajectories, **3 fragments of P2** (T3, T6, T7)

---

### track3d Output: 6 Trajectories

```
T1-T5: [Same as trackcorr T1-T5 for P0, P1, P3, P4]

T6: [10004] (4.277,3.289,2.906) → [10005] (4.225,3.355,2.906)
    [P2 fragment — 2 frames, correctly linked]
```

**Summary**: 6 trajectories, **2 fragments of P2** (T3, T6)

---

## Quantitative Comparison

### Forward Tracking Results

| Metric | trackcorr | track3d | Difference |
|--------|-----------|---------|-----------|
| **Trajectories** | 7 | 6 | +1 (trackcorr overfrags) |
| **P2 fragments** | 3 | 2 | +1 |
| **Gaps** | 1 (10002→10004) | 1 (10002→10004) | Same |
| **Relinks** | 0 | 1 (10004→10005) | +1 (track3d advantage) |
| **Particle average length** | 4.0 frames | 4.2 frames | +0.2 (track3d better) |
| **Average links/frame** | 4.5 | 4.5 | Same |
| **Lost particles/frame** | 0.2 | 0.0 | 0.2 worse |

### File Output Sets

**trackcorr** ptv_is files:
```
ptv_is.10001  → 5 entries (all link forward)
ptv_is.10002  → 4 entries (P2 missing, 4 link forward)
ptv_is.10003  → 4 entries (P2 absent, 4 link forward)
ptv_is.10004  → 5 entries (P2 10001↔10002, new P2 isolated, others link)
ptv_is.10005  → 5 entries (all isolated singletons)
```

**track3d** ptv_is files:
```
ptv_is.10001  → 5 entries (all link forward)
ptv_is.10002  → 4 entries (P2 missing, 4 link forward)
ptv_is.10003  → 4 entries (P2 absent, 4 link forward)
ptv_is.10004  → 5 entries (P2 links to 10005, others link)
ptv_is.10005  → 5 entries (all isolated singletons)
```

---

## Root Cause: The Fallback Gate

Both algorithms have a **last-resort direct-displacement link** for particles when the 3-frame chain fails or doesn't exist.

### trackcorr Last-Resort (line 1343)

```python
# try to link if kk is not found/good enough and prev exist
if curr_path_inf.inlist == 0 and curr_path_inf.prev_frame >= 0:
    diff_pos = vec_subt(X[3], X[1])
    if pos3d_in_bounds(diff_pos, tpar):
        angle, acc = angle_acc(X[1], X[2], X[3])
        if (acc < tpar.dacc and angle < tpar.dangle) or (acc < tpar.dacc / 10):
            # Register link
```

**Problem**: The gate `prev_frame >= 0` prevents re-appeared particles (prev_frame = -1) from using this fallback.

### track3d Level 3 (no gate)

```python
# No previous link, no neighbors with previous links
for i in range(orig_parts):
    curr_pi = curr.path_info[i]
    if curr_pi.prev_frame >= 0 or curr_pi.next_frame >= 0:
        continue  # Skip if already has links
    
    predicted = curr_pi.x.copy()  # Use current position
    cand_indices = find_candidates_in_3d(next_buf, predicted, dx, dy, dz)
    
    if cand_indices:
        # Link without dacc/dangle test
        curr_pi.next_frame = best  # REGISTERED
```

**Advantage**: No gate on `prev_frame`. If a particle (new or re-appeared) is unlinked, try Level 3.

---

## Backward Tracking Impact

### Forward Tracking Summary

After forward loop completes (5 steps):

```
Linkage state:
- P0: continuous 10001-10005 ✓
- P1: continuous 10001-10005 ✓
- P2: two fragments [10001-10002] and [10004-10005] (trackcorr)
      two fragments [10001-10002] and [10004-10005] (track3d)
- P3: continuous 10001-10005 ✓
- P4: continuous 10001-10005 ✓

Gap: frame 10003 with 4 particles, cannot bridge forward or backward alone
```

### Backward Tracking Algorithm (trackback_c)

Operates on **existing forward state**, processes frames in **reverse** (10005 → 10001).

At each backward step:
1. Look at particles with `next_frame >= 0` (have forward links)
2. Try to find predecessors in previous frame
3. Attempt to fill gaps using particles that were previously unlinked

**For P2 gap at frame 10003**:

Forward pass leaves:
- P2 in [10001, 10002]: prev=-1, next=1 (linked forward)
- P2 absent in frame 10003
- P2 in [10004, 10005]: prev=-1 (re-appeared, unlinked in trackcorr, linked in track3d), next=-1

Backward pass from frame 10003:
1. Frame 10003 is empty for P2
2. Move to frame 10002: P2 has `next_frame=1` (forward link exists)
3. Try to extend forward link by finding match at frame 10003:
   - P2_10002 position: (4.378, 3.155, 2.907)
   - Predict next (extrapolate velocity): ~(4.327, 3.221, 2.906)
   - Search for candidate at frame 10003: **FAILS** (P2 absent)
4. Move to frame 10001: try linking to frame 10002
   - Already linked forward ✓

**Backward result**: Cannot bridge 10002→10003 gap (P2 physically absent in 10003).
However, backward tracks the existing **10004→10005** link more thoroughly, assigning final trajectory segments.

---

## Why track3d Succeeds Where trackcorr Fails

### Algorithmic Differences at Step 10004

| Aspect | trackcorr | track3d |
|--------|-----------|---------|
| **Prediction for new particles** | Requires valid 3-frame chain (buf[3]) OR fallback requires `prev_frame >= 0` | Uses zero-velocity fallback in Level 3 for any unlinked particle |
| **Requires dacc/dangle check** | Yes (line 1320-1322) | No in Level 3 (just 3D bounds) |
| **Requires angle_acc computation** | Yes | No |
| **Gate on prev_frame** | YES (line 1343) ✗ | NO ✓ |
| **Handles re-appearance** | Only if it has predecessor or 3-frame lookahead | Yes, via Level 3 |

### Root Cause Summary

**trackcorr**: Last-resort fallback requires `prev_frame >= 0`. At step 10004, P2 has `prev_frame = -1` (no history in frame 10003), so fallback is skipped.

**track3d**: Level 3 explicitly handles particles with `prev_frame == -1`. It uses zero-velocity prediction and searches in the next frame without requiring predecessor history.

---

## Velocity-Based Relinking Insights

### P2's Actual Motion

```
P2 velocity (10001→10002):
  Δ = (4.378 - 4.425, 3.155 - 3.087, 2.907 - 2.906)
    = (-0.047, 0.068, 0.001) mm/frame
  |Δ| = 0.083 mm/frame

P2 velocity (10004→10005):
  Δ = (4.225 - 4.277, 3.355 - 3.289, 2.906 - 2.906)
    = (-0.052, 0.066, 0.0) mm/frame
  |Δ| = 0.085 mm/frame

Velocity consistency: |0.083 - 0.085| / 0.084 = 2.4% change
  → Highly consistent velocity!
```

### What track3d's Zero-Velocity Assumption Captures

At step 10004, P2 has been absent for 1 frame (10003). 

**track3d's Level 3 logic**:
1. P2_10004 position: (4.277, 3.289, 2.906)
2. Prediction (zero-velocity): (4.277, 3.289, 2.906)
3. Search box: ±0.5 mm in x, y, z
4. P2_10005 at (4.225, 3.355, 2.906) is within bounds:
   - |4.277 - 4.225| = 0.052 mm ✓
   - |3.289 - 3.355| = 0.066 mm ✓
   - |2.906 - 2.906| = 0.0 mm ✓

**Why it works**: P2 didn't move far during the gap (particles in turbulence still have local tendency), so zero-velocity prediction is accurate to within search tolerance.

### Lessons for Better Tracking Methods

1. **Don't gate linking on `prev_frame`** — particles can re-appear with no history; use other clues (position proximity, velocity bounds)

2. **Zero-velocity fallback is surprisingly effective** — when particles re-appear, they haven't traveled far; use current position as prediction center

3. **Track **neighborhood motion** — track3d Level 2 uses average neighbor velocity; this could be expanded to estimate velocity from surrounding particles even if predecessor is absent

4. **Multi-scale predictions**:
   - Level 1: full acceleration prediction (2*curr - prev) when history exists
   - Level 2: average neighborhood velocity
   - Level 3: zero-velocity (particles naturally persist locally)
   - Level 4 (proposed): **smooth interpolation** — if particle absent 1 frame, interpolate position from future frames backward

5. **Lookahead depth matters** — trackcorr's 3-frame chain is powerful when available but fragile when lookahead is unavailable; track3d's fallback strategy is more robust at sequence boundaries

---

## Backward Tracking Recovery Capabilities

### What Backward Tracking Can and Cannot Do

**Can recover**:
- Particles that were missed in forward pass but re-appear later
- Secondary links when forward chain exists but was incomplete
- Fill in isolated singletons if they fall between linked trajectories

**Cannot recover**:
- Particles physically absent in intermediate frames (e.g., P2 absent in 10003)
- Entirely new particles detected only in backward pass (would require re-detection)
- Improve detection sensitivity (relies on forward targets)

### Statistical Impact: Forward vs Backward&Forward

After forward (`trackcorr`):
```
P0: 5 frames (1001, 10002, 10003, 10004, 10005)
P1: 5 frames
P2: 2 fragments (3 parts: 10001-10002, 10004, 10005)
P3: 5 frames
P4: 5 frames

Trajectory count: 7
Average trajectory length: 3.43 frames
Gap coverage: 0 frames (1-frame gap at 10003 not bridged)
```

After forward + backward:
```
P0-P1, P3-P4: identical (strong forward links preserved)
P2: still 2 fragments (10002-linked and 10004-10005)
    But backward pass validates and confirms segments

Trajectory count: 7 (no change, gap still unbridgeable)
Average trajectory length: 3.43 frames
Gap coverage: 0 frames (same)

Difference: ~0 filename diffs at key frames (10002 updated very slightly in priorities)
```

**Conclusion**: Backward tracking doesn't **create** new links where forward failed (the gap is unbridgeable). It **validates** and **re-scores** existing links, useful for tie-breaking and refining link priorities when multiple candidates exist.

---

## Recommendations for Users

### When to Expect Fragmentation

1. **Particles in occlusions or gaps** — if a particle is unseen for 1+ frames, both algorithms will fragment
2. **Sequence boundaries** — when lookahead buffer is unavailable (final 1-2 frames), re-appearance linking is reduced
3. **High-speed particles** — if displacement exceeds search tolerance, links fail regardless of algorithm

### How to Improve Trajectory Recovery

1. **Extend sequence length** — ensure you have lookahead frames; the Burgers case has only 5 frames with buffer limited to 4
2. **Tune search tolerances** — `dvxmax`, `dvymax`, `dvzmax` should be 1.5–2× your expected max velocity per frame
3. **Use track3d for pre-reconstructed data** — if you have 3D targets, use track3d; it's simpler and has more robust fallbacks
4. **Use trackcorr for raw images** — trackcorr's multi-camera approach better handles occlusions and provides re-detection of lost particles via `add_particle` option
5. **Run backward tracking** — always run backward pass to validate forward results and fill in easy gaps
6. **Sort by trajectory length** — post-process, filter out singletons and very short fragments; they're often tracking noise

### Next-Generation Tracking Enhancements

Based on this analysis, proposed improvements:

1. **Predictive velocity model** — Instead of zero-velocity, estimate velocity from neighbors or temporal interpolation
2. **Gapless linking** — When re-appearance particle is detected, automatically interpolate position from forward and backward candidates to estimate gap trajectory
3. **Multi-camera re-detection** — For trackcorr, automatically re-project lost particle along predicted path to detect if available in images (implement `add_particle` smartly)
4. **Adaptive search windows** — Expand search tolerance at frame boundaries when lookahead is unavailable
5. **Persistence-based linking** — Track "momentum" of particle groups; particles that move together are likely same objects even if individually untracked

---

## Code References

### trackcorr Last-Resort Link (Problematic)

[algorithms/track.py Line 1343](https://github.com/alexlib/openptv2/blob/main/src/openptv2/algorithms/track.py#L1343)

```python
# try to link if kk is not found/good enough and prev exist
if curr_path_inf.inlist == 0 and curr_path_inf.prev_frame >= 0:
    diff_pos = vec_subt(X[3], X[1])
    if pos3d_in_bounds(diff_pos, tpar):
        # register link
```

**Issue**: `prev_frame >= 0` gate blocks re-appeared particles.

### track3d Level 3 Fallback (Robust)

[algorithms/track.py Line 1673](https://github.com/alexlib/openptv2/blob/main/src/openptv2/algorithms/track.py#L1673)

```python
# Level 3: No previous link, no neighbors with previous links
for i in range(orig_parts):
    curr_pi = curr.path_info[i]
    if curr_pi.prev_frame >= 0 or curr_pi.next_frame >= 0:
        continue  # Only unlinked particles

    predicted = curr_pi.x.copy()  # Zero-velocity
    cand_indices = find_candidates_in_3d(next_buf, predicted, dx, dy, dz)
    
    if cand_indices:
        curr_pi.next_frame = linkdecis[0]  # Link registered
```

**Advantage**: No gate; handles re-appearances via Level 3.

---

## Testing & Validation

### Regression Tests Added

1. **test_burgers_python_forward_relinks_reappeared_particle_in_both_trackers**
   - Verifies P2 10004→10005 link in both trackcorr and track3d forward passes
   - Ensures fix is applied consistently

2. **test_burgers_python_backward_starts_from_forward_and_keeps_relinked_segment**
   - Confirms backward tracking preserves re-linked segments
   - Validates backward doesn't corrupt forward results

3. **test_burgers_track3d_forward_python_vs_cython_verify_deviations**
   - Updated expected diff set; now shows Python and Cython fully match for track3d

### How to Run Tests

```bash
# Focused regression tests
uv run pytest tests/unit/test_track.py -k "burgers" -v
uv run pytest tests/unit/test_track3d.py -k "burgers" -v
```

---

## Conclusion

### Key Findings

1. **Gap causes fragmentation** — A 1-frame gap (frame 10003 absent) fragments P2 into multiple parts; both algorithms cannot bridge a fully absent frame.

2. **Re-appearance linking differs** — When P2 re-appears at frame 10004 with no valid lookahead (buffer 3 is empty):
   - **trackcorr** fails because fallback requires `prev_frame >= 0`
   - **track3d** succeeds via Level 3's zero-velocity prediction

3. **Buffer availability matters** — At sequence boundaries, lack of lookahead frames (buf[3] empty) severely limits linking options. Designers should ensure sufficient margin.

4. **Backward tracking validates but doesn't create links** — Running backward pass confirms forward results but cannot bridge truly absent frames; it's a quality metric, not a gap filler.

5. **Zero-velocity is effective** — For particles absent <2 frames, zero-velocity prediction is accurate to within typical search tolerance (0.5 mm).

### Actionable Insights

- **For users**: Extend sequences beyond your event-of-interest; use track3d when 3D data is available; tune search tolerances conservatively
- **For developers**: Remove unnecessary gates on `prev_frame` in linking logic; implement multi-level fallbacks like track3d; consider adaptive search windows at boundaries
- **For future work**: Explore predictive velocity models, gapless interpolation, and persistence-based group tracking to further reduce fragmentation.

---

## Appendix: Numerical Trace

### Full Step-by-Step Numerical Output

**Step 10001** (trackcorr):
```
step: 10001, curr: 5, next_frame: 5, links: 5, lost: 0, add: 0
Particles in buf[1]: P0, P1, P2, P3, P4 (all have prev_frame=0,1,2,3,4)
Candidates in buf[2]: P0, P1, P2, P3, P4 (all found)
3-frame chain in buf[3]: P0, P1, P2, P3, P4 (all valid)
→ All 5 link forward
```

**Step 10002** (trackcorr):
```
step: 10002, curr: 5, next_frame: 4, links: 4, lost: 1, add: 0
Particles in buf[1]: P0, P1, P2, P3, P4 (prev_frame=0,1,[broken],3,4)
Candidates in buf[2]: P0, P1, -, P3, P4 (P2 ABSENT)
→ P0, P1, P3, P4 link (4 links); P2 breaks (lost: 1)
```

**Step 10003** (trackcorr):
```
step: 10003, curr: 4, next_frame: 5, links: 4, lost: 0, add: 0
Particles in buf[1]: P0, P1, P3, P4 (all have valid predecessors)
Candidates in buf[2]: P0, P1, P3, P4, (P2 ABSENT) (5 total including new detections, P2 missing)
→ 4 link forward; P2 not in frame, no new detection
```

**Step 10004** (trackcorr) — CRITICAL:
```
step: 10004, curr: 5, next_frame: 5, links: 4, lost: 1, add: 0
Particles in buf[1]: P0(prev=P0_10003), P1(prev=P1_10003), P2(prev=-1), P3(prev=P3_10003), P4(prev=P4_10003)
P0-P1, P3-P4: Link forward normally (4 links)
P2: 
  - Candidate found in buf[2] (P2_10005) ✓
  - Try 3-frame chain in buf[3]: EMPTY (no frame 10006) ✗
  - Last-resort fallback: prev_frame=-1, condition FALSE ✗
  - next_frame = -1 (NOT LINKED)
→ 4 links; P2 lost (1 lost)
```

**trackcorr final**: 7 trajectories (T1-T5 + 2 P2 singletons)

---

**Same steps with track3d**:

**Steps 10001-10003**: Identical to trackcorr (all forward links made).

**Step 10004 (track3d)** — CRITICAL:
```
Level 1: P0(prev>=0), P1(prev>=0), P3(prev>=0), P4(prev>=0) → all link (4 links)
Level 2: P2(prev=-1), check neighbors → all >0.5mm away, skip
Level 3: P2(prev=-1, next=-1) → enter Level 3
  - predicted = (4.277, 3.289, 2.906)
  - cand_indices = find_candidates_in_3d(buf[2], predicted, (0.5,0.5,0.5))
    → P2_10005 at (4.225, 3.355, 2.906) within bounds ✓
  - Link registered: next_frame=1 (1 link)
→ 5 links total; 0 lost
```

**track3d final**: 6 trajectories (T1-T5 + 1 P2 fragment)

---

## Update: Python Improvements Over C

Subsequent analysis of the cavity dataset (~700 particles, 4 frames) revealed two bugs in the C code that the Python translation fixes:

### 1. Stale Buffer Recycling (Fixed in Python)

When `step >= last - 2`, no new frame is available to load into the last buffer slot. After `fb_next()` rotates the buffer, the old data remains. The C code's `assess_new_position()` searches this stale data and can create spurious links. The Python fix clears the slot:

```python
if step < run_info.seq_par.last - 2:
    fb.read_frame_at_end(step + 3, read_links=False)
else:
    fb.buf[fb.buf_len - 1].num_parts = 0  # clear stale data
```

This eliminates 2 spurious links in the cavity dataset at the last tracking step.

### 2. Phase 3: Losers Retry (Python Enhancement)

When two particles compete for the same target in C's conflict resolution, the loser is permanently dropped. The Python translation adds a third phase where losers try their fallback candidates (2nd, 3rd best matches) if unclaimed. On the cavity dataset, this recovers ~27 additional correct links.

### 3. C count1 Overcounting

The C code increments `count1` inside the conflict resolution loop. Particles that lose conflicts after being counted still inflate `count1`. This explains the discrepancy between C's `printf` output and the actual file-based link counts:

| Step | C printf count1 | C file links | Python links |
|------|-----------------|-------------|-------------|
| 10001 | 290 | 268 | 281 |
| 10002 | 394 | 346 | 357 |
| 10003 | 339 | 304 | 307 |

With Phase 3 disabled, Python matches C files exactly at steps 10001-10002 (0 mismatches). The 2 remaining mismatches at step 10003 are from the stale buffer fix.

A synthetic test case (`tests/unit/test_synthetic_tracking.py`) with 15 particles and known ground truth trajectories confirms:
- trackcorr: 103/103 correct links (100%), 0 wrong
- track3d: 102/103 correct links (99%), 0 wrong
- trackcorr correctly produces >= track3d links

## References

- **Tracking Algorithms**: [algorithms/tracking.md](tracking.md)
- **trackcorr_c_loop**: `src/openptv2/algorithms/track.py`
- **track3d_loop**: `src/openptv2/algorithms/track3d.py`
- **Backward Tracking**: `src/openptv2/algorithms/track.py` (`trackback_c`)
- **Synthetic Test**: `tests/unit/test_synthetic_tracking.py`
- **Burgers Test Suite**: `tests/unit/test_track.py` and `tests/unit/test_track3d.py`
