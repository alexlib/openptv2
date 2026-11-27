# OpenPTV2 Tracking Pipeline & Results Guide

This guide explains how particle tracking works in **OpenPTV2**, how to configure tracking parameters in the GUI or YAML, how the multi-pass pipeline and `seal` step operate, and how to interpret trajectory results in `res/run.zarr`.

---

## 1. Overview of the Tracking Pipeline

Tracking in OpenPTV2 links 3D particle positions across consecutive frames to reconstruct Lagrangian trajectories. The store-native pipeline writes all results to `res/run.zarr`.

```
                    ┌───────────────────────────────┐
                    │  3D Particles (zarr)          │
                    │  correspondences/frame_*       │
                    │  targets/cam_*/frame_*         │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ PASS 1: Forward Tracking (full_forward)                  │
       │ Predicts velocity/angle over 4 frames                    │
       └────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ PASS 2: Backward Tracking (full_backward)                │
       │ Reverse scan for cold-start seeds                       │
       └────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ PASS 3: Link Pruning & Post-Processing (postprocess)    │
       │ Reciprocity check, fragment merge                        │
       └────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Linkage (zarr)                │
                    │ linkage/ptv_is/frame_*/{prev, │
                    │  next,pos}                    │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Seal: linkage → trajectories  │
                    │ trajectories/{pos,time,trajid}│
                    │ traj/{trajid,length,first_row}│
                    └───────────────────────────────┘
```

> **Legacy `res/ptv_is.#` text files no longer exist.** All tracking reads/writes `res/run.zarr` via `RunStore` (`src/openptv2/storage/run_store.py:364` `write_linkage`). Use `zarr.open_group("res/run.zarr")` or `ZarrFrameStore` for inspection (`docs/zarr-hdf5-storage.md`).

---

## 2. Parameter Reference (`parameters.yaml`)

```yaml
track:
  tracker: two_phase      # two_phase | trackcorr | fast_3d (default two_phase)
  direction: both         # forward | both (default both)
  shared_correction: false  # two_phase add-on: 4-cam alignment + merged-blob refinement
  holistic_rounds: 0      # two_phase add-on: full merged-blob loop with births, 0 = off
  dvxmin: -10.0
  dvxmax: 10.0
  dvymin: -10.0
  dvymax: 10.0
  dvzmin: -10.0
  dvzmax: 10.0
  angle: 120.0       # gon (400 gon = 360°)
  dacc: 5.0          # [mm/frame²]
  flagNewParticles: true
  track_mode: 0      # 0=Standard, 1=3D Segment
  postprocess: true
  leaf_weight: 1.0   # two_phase only: 2D leaf weight
  min_trajectory_length: 5  # seal filter: drop <5-frame traj

plugins:
  selected_tracking: default  # default = track.tracker; or a custom plugin name
```

### Trackers and switches

| Setting | Values | Default |
| :--- | :--- | :--- |
| `tracker` | `two_phase`, `trackcorr` (legacy algorithm), `fast_3d` (3D only) | `two_phase` |
| `direction` | `forward`, `both` (forward + backward) | `both` |
| `postprocess` | reciprocity + gap bridging, for `trackcorr` and `fast_3d` | `true` |
| `holistic_rounds` | merged-blob loop, `two_phase` only; 0 = off | `0` |

The old presets (`full_multipass`, `standard_forward`, `priority_segment_3d`, ...) are translated and logged; see
[Particle trackers](trackers.md) for the table of old names and for how to choose.

### Detailed Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dvxmin/dvxmax` etc. | float | ±10.0 | Velocity search box [mm/frame] |
| `angle` | float | 120.0 | Max direction change [gon] |
| `dacc` | float | 5.0 | Max acceleration [mm/frame²] |
| `flagNewParticles` | bool | true | Seed new tracks mid-sequence |
| `track_mode` | int | 0 | 0=Standard, 1=3D Segment |
| `postprocess` | bool | true | Pass 3 reciprocity |
| `leaf_weight` | float | 1.0 | `two_phase` 2D ranking weight (0=3D-only) |
| `min_trajectory_length` | int | 5 | `seal` discards shorter traj (`src/openptv2/storage/seal.py:73`) |
| `selected_tracking` | str | `default` | `default` (reads `track.tracker`, defaulting to `two_phase`), `two_phase`, `trackcorr`, `fast_3d`, or a custom plugin name |

---

## 3. The 3-Pass Tracking Pipeline

### Pass 1: Forward (`full_forward`)
Frame-by-frame prediction `2*curr - prev`, 3D search box (`dv`), angle/acc tests.

### Pass 2: Backward (`full_backward`)
Reverse scan for cold-start seeds missed forward.

### Pass 3: Post-Processing (`postprocess`)
Reciprocity: `A→B` at `t→t+1` requires `B→A` at `t+1→t`. Merges backward links.

### Seal — Linkage to Flat Trajectories
After tracking, `seal()` (`src/openptv2/storage/seal.py:73`, called from `src/openptv2/batch/pyptv_batch.py:246`) walks `linkage/ptv_is` to assign `trajid`:

```python
# seal builds flat cache
store.write_trajectories(pos, vel, accel, time, trajid)  # run_store.py:530
store.write_traj_index(trajid, first, last, length, first_row)  # run_store.py:480
```

- `trajectories/{pos,time,trajid}` — sorted by `(trajid,time)`, `pos` in **meters** (mm→m `*1e-3`)
- `traj/{length,first_row}` — per-trajectory index for O(1) `pos[lo:hi]` without loading 66 MB `trajid` (`notebooks/marimo_trajectory_viewer.py:33`)
- `min_trajectory_length` filters short traj *before* writing; `n_dropped` in return dict
- `source_hash` memoizes — skips if linkage unchanged unless `force=True`

> See `docs/zarr-hdf5-storage.md` and `docs/algorithms/tracking.md` §Seal.

---

## 4. How to Choose a Tracker for Your Dataset

Selecting the optimal tracking algorithm depends on your experimental setup, particle density, camera calibration, optical noise characteristics (especially $Z$-axis depth uncertainty), and processing speed requirements.

```
                               ┌─────────────────────────────┐
                               │ What is your primary need?  │
                               └──────────────┬──────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        ⚡ Maximum Throughput / Speed                     🎯 Maximum Precision & Quality
   ┌───────────────────────────────────┐             ┌───────────────────────────────────┐
   │ Fast 3D Space Tracker             │             │ What is the main challenge?       │
   │ (`priority_segment_3d`)           │             └─────────────────┬─────────────────┘
   │ >800,000 p/s (~3,800 FPS)         │                               │
   └───────────────────────────────────┘       ┌───────────────────────┼───────────────────────┐
                                               ▼                       ▼                       ▼
                                      🔬 High Z-Depth Noise    👥 High Particle Density   🌊 Complex / Noisy
                                      & Optical Ambiguity     & Track Crossings          Trajectories / Dropouts
                                      ┌──────────────────────┐┌─────────────────────────┐┌──────────────────────┐
                                      │ Multi-Cam Epipolar   ││ 3D Kalman-Hungarian     ││ 3D Smooth / proPTV   │
                                      │ (`trackcorr` /       ││ (`kalman_hungarian_3d`) ││ (`fast_3d_smooth` /  │
                                      │  `full_multipass`)   ││ Multi-term cost solver ││  `predictive_gmm_3d`)│
                                      └──────────────────────┘└─────────────────────────┘└──────────────────────┘
```

---

### A. Quick Selection Decision Matrix

| Dataset Condition / Goal | Recommended Tracker Key | Preset / Plugin Name | Key Strengths & Why It Works |
| :--- | :--- | :--- | :--- |
| **Ultra-fast batch processing / Quick previews** | `priority_segment_3d` | `priority_segment_3d` | Fastest engine in OpenPTV2 (>800,000 particles/sec). Uses 4-level Cython segment linking. |
| **High $Z$-depth noise / Stereo optical uncertainty** | `cython_epipolar_tracking` | `full_multipass` | Projects 3D search volumes to 2D camera images. Uses multi-camera visibility consensus to resolve $Z$-axis ambiguity. |
| **High particle density ($>1,000$ particles/frame)** | `kalman_hungarian_3d` | `kalman_hungarian_3d` | 9D Kalman filter + cKDTree Hungarian assignment. Global bipartite matching prevents "track swapping" at crossings. |
| **Noisy 3D triangulations / Single-frame velocity jitter** | `fast_3d_smooth` | `fast_3d_smooth` | Savitzky-Golay order-3 polynomial velocity filter. Cuts single-frame velocity noise amplification in half. |
| **Smooth velocity/acceleration fields required** | `predictive_gmm_3d` | `proptv` / `custom_plugin` | Continuous spatial-temporal Gaussian Mixture Model (GMM) fitting for analytic derivative estimation. |
| **Custom multi-term cost weights (e.g. intensity)** | `nearest_hungarian_3d` | `nearest_hungarian_3d` | Configurable physical cost matrix $\{w_{\text{dist}}, w_{\text{vel}}, w_{\text{acc}}, w_{\text{intensity}}\}$ with gap recovery (`max_gap`). |
| **Poor 3D calibration / Heavy $Z$-reconstruction ghosts** | `myptv_2d_tracking` | `myptv_2d_tracking` | Tracks 2D targets in pixel space per camera view first, then triangulates matched 2D trajectories. |

---

### B. Speed vs. Accuracy Comparison

Tracking algorithms in OpenPTV2 span a wide trade-off spectrum from high-throughput Cython engines to sophisticated multi-term optimization solvers:

| Tracker Engine | Implementation | Processing Speed (FPS) | Throughput (particles/sec) | Accuracy & Precision | Trajectory Recovery |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`priority_segment_3d`** | Cython C-Memoryview | **$>3,800\text{ FPS}$** | **$>800,000\text{ p/s}$** | High ($99.1\%\text{--}99.5\%$) | Standard (Forward only) |
| **`fast_3d_smooth`** | Python / NumPy | **$\sim 1,000\text{ FPS}$** | **$\sim 200,000\text{ p/s}$** | High ($98.5\%$) | High (GAP relinking) |
| **`kalman_hungarian_3d`** | Python / cKDTree | **$\sim 300\text{--}500\text{ FPS}$** | **$\sim 80,000\text{--}100,000\text{ p/s}$** | Highest (**$98.0\%\text{--}99.0\%$**) | Highest (Dynamic innovation gating) |
| **`cython_epipolar_tracking` (`trackcorr`)** | Cython / C Loop | **$\sim 250\text{--}800\text{ FPS}$** | **$\sim 50,000\text{--}160,000\text{ p/s}$** | Highest (**2D+3D Reciprocity**) | Highest (3-Pass Forward + Backward + Pass 3) |
| **`predictive_gmm_3d` (proPTV)** | Python / GMM | **$\sim 100\text{--}300\text{ FPS}$** | **$\sim 20,000\text{--}60,000\text{ p/s}$** | Highest (Analytic derivatives) | High (Backtracking & gap tracking) |
| **`nearest_hungarian_3d` (MyPTV)** | Pure Python / SciPy | **$\sim 50\text{--}400\text{ FPS}$** | **$\sim 10,000\text{--}80,000\text{ p/s}$** | Standard ($91.7\%$) | Highest (Longest tracks: $14.3\text{ fr}$) |
| **`myptv_2d_tracking`** | Pure Python | **$\sim 10\text{--}50\text{ FPS}$** | **$\sim 2,000\text{--}10,000\text{ p/s}$** | Moderate | Moderate (Per-camera 2D space) |

---

### C. Matching Trackers to Dataset Conditions

#### 1. Handling $Z$-Axis Depth Noise & Optical Ambiguity

> [!IMPORTANT]
> **Understanding $Z$-Noise in 3D PTV:**
> In stereoscopic and tomographic multi-camera setups, camera optical axes typically form narrow viewing angles ($< 30^\circ\text{--}45^\circ$). Consequently, reconstructed 3D particle positions exhibit $Z$-axis depth uncertainty that is **$3\times\text{ to }10\times$ higher** than in-plane $X,Y$ noise ($\sigma_Z \gg \sigma_X, \sigma_Y$).

* **Why pure 3D distance trackers fail under high $Z$-noise:**
  Trackers that search purely in 3D Euclidean distance ($\Delta r = \sqrt{\Delta X^2 + \Delta Y^2 + \Delta Z^2}$) can mistakenly jump between different nearby particles along the optical $Z$-axis because random $Z$-jitter exceeds the true inter-frame physical displacement.
* **How to overcome high $Z$-noise:**
  * **Option A: Multi-Camera Epipolar Re-projection (`cython_epipolar_tracking` / `full_multipass`)**
    Instead of relying solely on 3D distance, `trackcorr` projects a 3D candidate search cuboid back onto each camera's 2D image plane ($x_i, y_i$). Because $Z$-uncertainty projects along epipolar lines on different cameras, requiring cross-camera target consensus eliminates fake $Z$-jumps.
  * **Option B: Savitzky-Golay Velocity Filtering (`fast_3d_smooth`)**
    Applies an order-3 Savitzky-Golay polynomial filter over a 5-frame moving window. This smooths out high-frequency $Z$-jitter while preserving physical flow acceleration, reducing single-frame velocity noise amplification by $\sim 50\%$.
  * **Option C: 2D Image-Space Tracking (`myptv_2d_tracking`)**
    If 3D triangulation is severely degraded by calibration errors, track in 2D pixel space per camera first, then perform multi-camera triangulation on the resulting 2D trajectories.

#### 2. High Particle Density & Track Crossings

* **The Problem:** As seeding density increases ($>1,000$ particles/frame), the mean distance between neighboring particles approaches the frame-to-frame displacement magnitude. Greedy nearest-neighbor scanners suffer from **order-dependent claim bias** and "track swapping" when two particles pass close to each other.
* **The Solution:**
  * Use **`kalman_hungarian_3d`** or **`nearest_hungarian_3d`**.
  * Both trackers construct a bipartite candidate graph and solve global assignment using SciPy's Hungarian algorithm (`linear_sum_assignment`).
  * `kalman_hungarian_3d` incorporates a multi-term physical cost function ($w_{\text{distance}} + w_{\text{velocity\_continuity}} + w_{\text{acceleration\_penalty}}$) so that candidates maintaining momentum win over closer candidates that require unphysical trajectory bends.

#### 3. Optical Occlusions, Missing Frames & Detection Dropouts

* **The Problem:** In physical experiments, particles frequently disappear for 1–2 frames due to laser sheet non-uniformity, bubble shadows, or field-of-view edges.
* **The Solution:**
  * Use **`full_multipass`** (Pass 3 post-processing bridges gaps and uses backward pass to discover cold-start seeds).
  * Use **`kalman_hungarian_3d`** or **`nearest_hungarian_3d`** with `max_gap = 2` (or higher). The tracker extrapolates track position through invisible frames using established Kalman/polynomial velocity vectors, reconnecting when the particle reappears.

#### 4. Complex Fluid Dynamics & High Acceleration

* **The Problem:** In turbulent flows, vortices, or high-shear boundary layers, linear constant-velocity extrapolation ($\mathbf{X}_{t+1} = \mathbf{X}_t + \mathbf{V}_t \Delta t$) fails near sharp streamline curvatures.
* **The Solution:**
  * Use **`kalman_hungarian_3d`** (uses a 9D Constant-Acceleration state model $\begin{bmatrix}\mathbf{X} & \mathbf{V} & \mathbf{A}\end{bmatrix}^T$ that dynamically adapts innovation bounds).
  * Use **`predictive_gmm_3d`** (proPTV) for continuous spatial-temporal Gaussian Mixture Model smoothing, providing analytical velocity and acceleration derivatives.

---

## 5. Tracking Algorithms & Plugins Reference

| Plugin | Description |
| :--- | :--- |
| `default` (trackcorr) | Cython 3 `track3d_loop_fast` — 3D box search, angle+acc |
| `two_phase` | **New** — Phase 1: 3D KD-tree candidates within `v_max`; Phase 2: per-camera 2D leaf mean distance → Hungarian assignment (`src/openptv2/plugins/two_phase_tracking.py`). `leaf_weight=0` ≡ 3D-only. **74% more multi-frame traj** on aorta phantom (poorly-conditioned). |
| `hybrid_deltat_3d` | Link every N-th frame first, then fill the frames between |
| `cython_epipolar` etc. | Epipolar variants |

Select via GUI **Plugins** or `plugins.selected_tracking`. Custom plugins implement `BaseTrackingPlugin` (`docs/developer_guide/custom_tracking_plugins.md`).

---

## 5. Understanding Results in `res/run.zarr`

### Reading Linkage & Trajectories

```python
import zarr, numpy as np

root = zarr.open_group("res/run.zarr", mode="r")
# Linkage per frame
prev, nxt, pos = (
    root["linkage/ptv_is/frame_000001/prev"][:],
    root["linkage/ptv_is/frame_000001/next"][:],
    root["linkage/ptv_is/frame_000001/pos"][:],
)
# Flat trajectories (sealed)
traj = root["traj"]
idx_tid, idx_len, idx_row = (
    np.asarray(traj["trajid"]),
    np.asarray(traj["length"]),
    np.asarray(traj["first_row"]),
)
# Top 100 longest
order = np.argsort(idx_len)[::-1][:100]
for tid, fr, ln in zip(idx_tid[order], idx_row[order], idx_len[order]):
    pts = np.asarray(root["trajectories/pos"][fr : fr + ln])  # [m]
```

### Copying to Dropbox

```bash
uv run python copy_trajectories.py --include-traj --overwrite  # creates trajectories.zarr + traj.zarr (137 MB vs 1637 MB)
```

See `C:\Users\alex\Downloads\aorta_phantom\wp1\copy_trajectories.py` and `docs/zarr-hdf5-storage.md`.

---

## 6. GUI to Batch Workflow

1. **GUI tuning**: `uv run pyptv <exp>` → Parameters → Tracking → OK (writes `parameters.yaml`)
2. **Batch**:
   ```bash
   uv run openptv2-batch <exp_or_yaml> <first> <last> --mode both
   # Benchmark without overwriting:
   uv run openptv2-batch <exp> 1 50 --mode tracking --tracking-plugin two_phase --output bench_two_phase.zarr
   ```
   API: `from openptv2.batch.pyptv_batch import main; main(yaml, 1, 5005, mode="tracking", output="bench.zarr")` (`src/openptv2/batch/pyptv_batch.py:264`)

---

## 6b. Per-step parameters for transient or periodic flows (opt-in)

One search window for the whole run is wrong when the flow changes a lot in time (a pulse, a cycle). Per-step parameters let some frame transitions use wider bounds. **It works with the classic `trackcorr` tracker only** (`plugins.selected_tracking: trackcorr`); the default `two_phase` tracker builds its own tracker and does not read them.

1. Let the program suggest the steps: `uv run python -m openptv2.cli tune-dynamic parameters_Run1.yaml`. It runs one tracking pass, flags frames where unusually few particles got a forward link (a real particle that failed to link is what a wider window fixes; no truth is needed), and writes `dynamic_track.yaml` next to the YAML file.
2. Read it and edit it. It lists only the flagged steps:
   ```yaml
   steps:
     117: {dvxmin: -25.0, dvxmax: 25.0, dacc: 12.0}
   ```
3. Turn it on in the YAML file: `track: {dynamic_tracking: true}`. A step with no entry uses the normal parameters. `track.dynamic_params_file` gives another file name.
4. Track again. The log shows `[dynamic tracking] per-step overrides loaded from ...`. If the file is missing, the run prints a warning and uses the normal parameters for every step.

Without the `dynamic_tracking` key nothing changes.

---

## 7. Interpreting Statistics

`seal` reports `n_trajectories`, `n_rows`, `n_dropped`. With `z-noise/motion≈19` (aorta phantom) expect short fragments (median ~10 with `min_length=5`, median 1 without filter). For Eulerian `velocity` fields use `flowtracks`/`postptv`, not long Lagrangian `trajid`.

---

## 8. Developing Custom Plugins

See `docs/developer_guide/custom_tracking_plugins.md`.
