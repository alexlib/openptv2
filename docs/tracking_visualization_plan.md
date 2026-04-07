# Tracking Debugging Visualization Plan

## Overview

Extend the existing "Debugging with display" button to provide interactive tracking parameter exploration with visual feedback.

---

## Architecture

```
"Debugging with display" button
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  TrackingDebugPanel (new TraitsUI panel)                        │
├─────────────────────────────────────────────────────────────────┤
│  • Parameter sliders (dvxmin, dvxmax, dvymin, dvymax, dacc,     │
│    dangle)                                                      │
│  • Frame selector (which frame to click on)                     │
│  • Visualization canvas (4 camera views + overlay)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

1. **Init**: Load 4 frames into memory from `tracking_preview.py` → extract_frame_details()
2. **Click**: User clicks on particle in any camera view
3. **Process**: For that particle, compute:
   - Predicted position in frames t+1, t+2, t+3 using velocity model
   - Search volume boundaries in each camera
   - Candidate particles in each candidate frame
4. **Visualize**: Draw all results on the 4 camera views

---

## Key Components to Use

### 1. Parameters Access (existing)
```python
# From main_gui
track_params = main_gui.get_parameter("tracking")  # TrackPar
# Access: track_params.dvxmin, track_params.dvxmax, track_params.dacc, track_params.dangle
```

### 2. Epipolar Lines (existing - pyptv_gui.py:1367)
```python
from optv.epipolar import epipolar_curve
pts = epipolar_curve(point, cal_from, cal_to, num_points, cpar, vpar)
```

### 3. Search Volume Computation
The function `sorted_candidates_in_volume()` in `algorithms/track.py:620` computes the search rectangle. We need a simplified version that just returns the bounds:

```python
def compute_search_bounds_2d(
    center_proj: np.ndarray,  # 2D projected position in each camera
    velocity: np.ndarray,    # [vx, vy, vz] - predicted velocity
    dvxmin: float, dvxmax: float,
    dvymin: float, dvymax: float,
    accel: float,            # dacc - acceleration factor
    frame_offset: int,       # 1, 2, or 3 frames ahead
) -> List[Tuple[float, float, float, float]]:  # (left, right, up, down) per camera
```

### 4. Frame Buffer Access (existing)
```python
# After running preview with 4+ frames:
tracker = Tracker(cpar, vpar, tpar, spar, cals)
tracker.restart()
# Frame buffer: fb = tracker.run_info.fb
# fb.buf[0] = frame t-1, fb.buf[1] = frame t, fb.buf[2] = t+1, fb.buf[3] = t+2
```

### 5. Particle Display (existing)
```python
# From pyptv_gui.py drawcross method:
camera_list[i].drawcross(str_x, str_y, x, y, color, size, marker)
```

---

## Implementation Plan

### Phase 1: Panel Setup

Add new class `TrackingDebugPanel` with:
- **Sliders for parameters**: dvxmin, dvxmax, dvymin, dvymax, dacc, dangle (with live update)
- **Frame indicator**: Show which frame is "current" (frame N where click happened)
- **4 camera canvas**: Use existing CameraWindow components or create 4-views layout

### Phase 2: Click Handler

When user clicks on camera view:
1. Find nearest particle in `frame.targets[cam]` within 5 pixels
2. Get particle's 3D position from `frame.path_info` (if linked) or triangulate from 2D
3. Compute predicted positions for frames t+1, t+2, t+3

### Phase 3: Search Volume Visualization

For each future frame (t+1, t+2, t+3):
1. Compute search rectangle bounds using velocity + acceleration limits
2. Project bounds to each camera using calibration
3. Draw rectangle boundary with distinct colors per frame:
   - Frame t+1: Green
   - Frame t+2: Yellow  
   - Frame t+3: Orange

### Phase 4: Candidate Visualization

In frame t+1 (next frame):
1. Find all targets within search rectangle
2. Compute 3D distance from predicted position
3. Color-code by distance:
   - Green: within dvxmin (close)
   - Yellow: within dvxmax (acceptable)
   - Red: outside (rejected)
4. Draw lines from clicked particle to candidates

### Phase 5: Statistics Output

Print to console/status panel:
```
Clicked particle ID: X in frame N
Position: (x, y, z) in camera C
Candidates in frame N+1: M particles
  - Candidate 1: dist=0.5mm, picked=Yes/No
  - Candidate 2: dist=1.2mm, picked=No
...
Search volume in frame N+1: x in [min,max], y in [min,max]
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `gui/pyptv/tracking_viz_panel.py` | Add `TrackingDebugPanel` class |
| `gui/pyptv/pyptv_gui.py` | Modify `track_debug_with_display_action` to use new panel |
| `gui/pyptv/tracking_preview.py` | Add `compute_search_bounds()` helper function |
| `gui/pyptv/tracking_debug_utils.py` (new) | Search volume computation, candidate finding |

---

## Parameter Defaults

- dvxmin: 0-10 pixels (default 0)
- dvxmax: 0-50 pixels (default 20)
- dvymin: 0-10 pixels (default 0)
- dvymax: 0-50 pixels (default 20)
- dacc: 0-20 (default 5)
- dangle: 0-30 degrees (default 10)

---

## Visual Design

- 4 camera views arranged in 2x2 grid
- Clicked particle: Cyan circle in all visible cameras
- Epipolar lines: Cyan dashed lines to other cameras
- Search volume t+1: Green rectangle outline
- Search volume t+2: Yellow rectangle outline
- Search volume t+3: Orange rectangle outline
- Candidates: Colored by distance (green/yellow/red crosses)
- Status panel: Bottom of window with statistics output