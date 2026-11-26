# Changelog

All notable user-facing changes and algorithmic improvements to **OpenPTV2** are documented here in plain language.

---

## Unreleased (2026-11-25)

### Zero-Gauge-Drift 4-Camera Alignment, Unbiased Merge Refinement & Shared-Blob Ghost Fix

We discovered and fixed three subtle, interacting issues in multi-camera 3D-PTV that previously limited velocity/acceleration accuracy and track continuity whenever particles overlapped or dropped out of a camera view.

#### 1. Eliminating the "Camera-Switching Jump" (`merge_refine.py`)
- **What we found**: In real 4-camera PTV setups, camera calibration is never 100% exact—camera rays typically miss each other by ~0.2–0.3 pixels. When a particle is seen by all 4 cameras in every frame, that tiny static offset cancels out when computing velocity. **However**, whenever 1 camera drops out (a 3-camera point—30–40% of points in real experiments) or when merge refinement down-weights 1 overlapping camera, the 3D position suddenly jumps sideways by ~0.05 mm (often half a frame's motion!) because 3 rays intersect in a slightly different spot than 4 rays. That artificial zig-zag severely corrupts velocity and acceleration.
- **How it is fixed**: `refine_merged` now fits a smooth, per-camera 3D residual map (`fit_4cam_residual_field`, enabled by default via `align_4cam=True`) to the leftover pixel errors of clean 4-camera points and subtracts it from the 2D observations before refining positions. Because 4-camera least-squares residuals sum to zero in 3D Jacobian space, this **leaves the 4-camera 3D volume and scale 100% unchanged (0.000% scale drift)** while making all 4 camera rays intersect at the exact same 3D point in `< 0.05s`. Switching between 4 and 3 cameras—or down-weighting a merged view—no longer causes a jump.
- **Real-data check (`CompleteTest wp2`, held-out unseen frames)**: Cuts 3-camera leave-one-camera-out (LOCO) error by **-43.6%** (`0.347 px -> 0.196 px`) with zero scale drift, in `< 0.05s`.

#### 2. Removing Quadratic Path-Prior Bias (`merge_refine.py`)
- **What we found**: Previously, `refine_merged` defaulted to `prior=True`, pulling refined 3D points toward a 9-frame quadratic curve. That was originally needed to damp the camera-switching jumps above, but it biased curved turbulent trajectories toward a parabola and doubled the triangulation workload.
- **How it is fixed**: With 4-camera alignment active, `refine_merged` now defaults to **`prior=False`**. The 9-frame trajectory fit is used strictly to split overlapping blobs and reject outlier views, while final 3D positions come directly from the aligned camera measurements.

#### 3. Stopping the Ghost Filter from Blocking Recovered Merged Particles (`point_quality.py`)
- **What we found**: When `ptv.shared_pass` recovers a particle that overlapped with another particle in 1 camera, that shared blob is naturally ~2× brighter than the particle's other 3 views. Previously, the ghost detector (`rcm_blob`) included that shared blob when checking brightness consistency across cameras, flagged **79%** of recovered particles as likely ghosts (`ghost_prob > 0.2`), and blocked the tracker from linking them.
- **How it is fixed**: `log_brightness_from_arrays(..., mask_shared=True)` now ignores shared camera views when computing cross-camera brightness spread whenever a point has at least 2 unshared cameras.

#### Measured Impact

| Benchmark | Mode | Velocity Error (Before → After) | Acceleration Error (Before → After) | Yield (Before → After) |
|---|---|---|---|---|
| **`syn1`** (40 frames, rendered + cal offset) | `shared_correction` | `0.1179` → **`0.0930`** (-21.1%) | `7.93` → **`6.44`** (-18.8%) | `0.7087` → **`0.7099`** |
| **`L2`** (200 frames, rendered) | `shared_correction` | `0.1060` → **`0.0913`** (-13.9%) | `6.16` → **`5.30`** (-14.0%) | `0.6936` → **`0.6947`** |
| **`L2`** (200 frames, rendered) | `shared_pass + corr` | `0.1050` → **`0.0891`** (-15.1%) | `6.85` → **`5.01`** (-26.9%) | `0.7139` → **`0.7158`** |
| **`L1 m0.75`** (200 frames, real-like) | `shared_pass + corr` | `0.0876` → **`0.0824`** (-5.9%) | `5.61` → **`5.18`** (-7.7%) | `0.7725` → **`0.7746`** |
| **`L1 m0.75`** (200 frames, real-like) | `holistic_rounds=2` | `0.0843` → **`0.0824`** (-2.3%) | `5.06` → **`4.96`** (-2.0%) | `0.7905` → **`0.7957`** |

See [Camera-Subset Alignment & Merge Refinement](docs/camera-subset-alignment.md) for the full user guide.
