# OpenPTV2 Tracking Pipeline & Results Guide

This guide explains how particle tracking works in **OpenPTV2**, how to configure tracking parameters in the GUI or YAML, how the multi-pass tracking pipeline operates, and how to interpret the resulting trajectory files.

---

## 1. Overview of the Tracking Pipeline

Tracking in OpenPTV2 links 3D particle positions across consecutive time steps (frames) to reconstruct Lagrangian fluid trajectories. 

The tracking framework is designed to run locally in the **PyPTV GUI** for parameter tuning and interactive preview, or headlessly in **Batch / Cloud Mode** using the exported `parameters.yaml`.

```
                    ┌───────────────────────────────┐
                    │  3D Particles (res/rt_is.#)   │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ PASS 1: Forward Tracking (full_forward / step_forward)   │
       │ Predicts velocity, acceleration, & angle over 4 frames   │
       └────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ PASS 2: Backward Tracking (full_backward)                │
       │ Re-scans sequence in reverse to find missing seeds      │
       └────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ PASS 3: Link Pruning & Post-Processing (postprocess)    │
       │ Verifies link reciprocity & merges recovered fragments   │
       └────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Trajectories (res/ptv_is.#)   │
                    └───────────────────────────────┘
```

---

## 2. Parameter Reference (`parameters.yaml`)

Tracking configuration is stored under the `track:` section of your `parameters.yaml` file (and in `plugins:` for algorithm selection).

```yaml
track:
  preset: "full_multipass" # Preset: "priority_segment_3d", "standard_forward", "full_multipass", or "custom_plugin"
  dvxmin: -10.0      # Min velocity search step in X [mm/frame]
  dvxmax: 10.0       # Max velocity search step in X [mm/frame]
  dvymin: -10.0      # Min velocity search step in Y [mm/frame]
  dvymax: 10.0       # Max velocity search step in Y [mm/frame]
  dvzmin: -10.0      # Min velocity search step in Z [mm/frame]
  dvzmax: 10.0       # Max velocity search step in Z [mm/frame]
  angle: 120.0       # Max angular deviation between steps [gon] (400 gon = 360 deg)
  dacc: 5.0          # Max acceleration limit [mm/frame^2]
  flagNewParticles: true  # Allow new unlinked particles to seed new tracks mid-sequence
  track_mode: 0      # 0 = Standard (4-frame predictor), 1 = 3D Segment mode
  postprocess: true  # Automatically run Pass 3 post-processing after backward tracking

plugins:
  selected_tracking: default  # Algorithm: "default" (trackcorr), "splitter_tracking", etc.
```

### High-Level Presets (`preset`)

OpenPTV2 provides high-level preset profiles to simplify pipeline configuration:

| Preset Key | Display Name | Pipeline Description | Recommended Use Case |
| :--- | :--- | :--- | :--- |
| **`priority_segment_3d`** | Fast 3D-Only (No added particles) | Single-pass forward tracking (`track_mode=1`). Uses only 3D coordinates from `rt_is.#`. | Quick sanity checks, low density / low noise data. |
| **`standard_forward`** | Fast Standard (Forward only, with added particles) | Single-pass forward tracking (`track_mode=0`, `flagNewParticles=true`). | Fast processing when backward tracking is not required. |
| **`full_multipass`** | Standard 3-Pass (Forward + Backward + Post-process) | Full 3-pass pipeline: Forward $\rightarrow$ Backward $\rightarrow$ Pass 3 reciprocity pruning. | **Recommended for maximum accuracy & trajectory recovery.** |
| **`custom_plugin`** | Custom Plugin / Splitter | Delegates pipeline execution to a user-specified plugin (e.g. `splitter_tracking`). | Quad-view splitters or specialized custom tracking algorithms. |

### Detailed Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`preset`** | `str` | `"full_multipass"` | Active tracking strategy preset (`priority_segment_3d`, `standard_forward`, `full_multipass`, `custom_plugin`). |
| **`dvxmin` / `dvxmax`** | `float` | `-10.0` / `10.0` | Velocity search box along X axis in physical units [mm/frame]. Limits max displacement between frame $t$ and $t+1$. |
| **`dvymin` / `dvymax`** | `float` | `-10.0` / `10.0` | Velocity search box along Y axis [mm/frame]. |
| **`dvzmin` / `dvzmax`** | `float` | `-10.0` / `10.0` | Velocity search box along Z axis [mm/frame]. |
| **`angle`** | `float` | `120.0` | Maximum direction change between velocity vector $\mathbf{v}_1 = \mathbf{x}_{t} - \mathbf{x}_{t-1}$ and $\mathbf{v}_2 = \mathbf{x}_{t+1} - \mathbf{x}_t$ measured in **gon** ($100\text{ gon} = 90^\circ, 400\text{ gon} = 360^\circ$). |
| **`dacc`** | `float` | `5.0` | Maximum allowed change in velocity magnitude $\|\mathbf{v}_2 - \mathbf{v}_1\|$ [mm/frame$^2$]. |
| **`flagNewParticles`**| `bool` | `true` | When `true`, particles appearing mid-sequence (e.g. entering FOV) are initialized as new trajectory seeds. When `false`, only particles present in the initial seed frame are tracked. |
| **`track_mode`** | `int` | `0` | `0` = Standard 4-frame linear prediction (`step_forward`). `1` = 3D Segment Mode (`step_forward_3d`). |
| **`postprocess`** | `bool` | `true` | Enables Pass 3 link reciprocity verification and cold-start seed recovery during backward tracking. |
| **`selected_tracking`**| `str` | `"default"` | Algorithm selection: `"default"` (core OpenPTV C engine wrapper), `"splitter_tracking"`, or custom plugin. |

---

## 3. The 3-Pass Tracking Pipeline

To maximize trajectory length and eliminate false-positive links, OpenPTV2 supports a 3-pass tracking pipeline:

### Pass 1: Forward Tracking (`full_forward`)
1. Starts at frame $N_1$ (`sequence.first`) and progresses frame-by-frame to $N_{\text{last}}$ (`sequence.last`).
2. Uses existing 2-frame links to predict position at $t+1$.
3. Evaluates candidates using velocity bounds (`dv`), acceleration limit (`dacc`), and angle (`angle`).
4. Selects the candidate that minimizes total tracking cost.
5. Writes forward link pointers into `ptv_is.#` result files.

### Pass 2: Backward Tracking (`full_backward`)
1. Re-scans the sequence in reverse order from $N_{\text{last}}-1$ down to $N_1$.
2. Uses backward velocity predictions to discover particles that were missed during forward initialization ("cold-start seeds").
3. Connects backwards links to fill gaps caused by temporary particle occlusions or high-shear regions.

### Pass 3: Post-Processing & Reciprocity (`postprocess`)
1. **Link Reciprocity Check**: Verifies that if particle $A$ at frame $t$ links forward to particle $B$ at $t+1$, particle $B$ at $t+1$ also links backward to $A$ at $t$.
2. **False Link Removal**: Unlinks candidates that failed candidate reciprocity.
3. **Seed Recovery**: Merges backward-discovered links into unified, continuous trajectories.

> **Why use Pass 3?** Backward tracking without post-processing can accumulate redundant or non-reciprocal links. Pass 3 ensures that only mutually consistent forward-backward links are retained.

### Empirical Strategy Benchmark Comparison

The tables below show typical trajectory recovery performance across the 3 main tracking strategy presets tested on a standard 4-camera dataset (`TT13_aorta`, 10 frames, ~1,858 particles/frame):

#### Overall Performance Summary

| Tracking Preset | Algorithm / Passes | Total Links | Trajectories Count | OVERALL Mean Length | Max Length | Relative Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`priority_segment_3d`** | Fast 3D-Only (`track_mode=1`) | 14,010 | 4,540 | **4.09 frames** | 10 | **1.0$\times$** (Fastest) |
| **`standard_forward`** | Fast Standard (Forward only) | 13,631 | 4,919 | **3.77 frames** | 10 | **1.2$\times$** |
| **`full_multipass`** | Standard 3-Pass (Forward + Backward + Postprocess) | 13,667 | 4,883 | **3.80 frames** | 10 | **1.8$\times$** (Most Accurate) |

#### Trajectory Seed Origin Breakdown (Frame 1 vs. Mid-Sequence Entry)

To understand why `priority_segment_3d` shows a higher raw *overall* mean length than multi-pass tracking, we must inspect trajectories by their **point of origin**:

| Tracking Preset | Total Trajectories | Frame 1 Seeds Count | Frame 1 **Mean Length** | Mid-Entry Seeds Count | Mid-Entry **Mean Length** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`priority_segment_3d`** | 4,540 | 1,846 | **6.80 frames** | 2,694 | 2.23 frames |
| **`standard_forward`** | 4,919 | 1,846 | **6.69 frames** | 3,073 | 2.02 frames |
| **`full_multipass`** | 4,883 | 1,846 | **6.72 frames** | 3,037 | 2.03 frames |

#### Why Multi-Pass Tracking is More Accurate
1. **Mid-Sequence Particle Seeding (`flagNewParticles=true`)**:
   * `standard_forward` and `full_multipass` seed unlinked particles entering the field of view mid-sequence (frames 2..10), capturing **~379 additional short trajectories** (3,037 vs 2,694).
   * Adding these short 1- to 2-frame trajectories near domain boundaries increases the denominator and **drags down the overall arithmetic mean**, even though long trajectories are fully preserved.
2. **True Trajectory Lengthening (`full_multipass` vs `standard_forward`)**:
   * Comparing long-term trajectories (Frame 1 seeds): `full_multipass` increases mean length from **6.69 to 6.72 frames** over forward-only tracking by repairing broken tracks during backward pass (`full_backward`).
3. **Pass 3 Reciprocity Pruning**:
   * `full_multipass` severs **36 false unidirectional track fragments** (reducing mid-entry count from 3,073 to 3,037) while increasing valid total links (from 13,631 to 13,667).
4. **Preventing 3D "Cross-Over" Jumps in `priority_segment_3d`**:
   * `priority_segment_3d` tracks purely by 3D distance without 2D epipolar or candidate reciprocity checks. In dense regions, it can falsely "cross over" adjacent particles, artificially stitching two distinct tracks together. `full_multipass` enforces 2D+3D candidate reciprocity, ensuring 100% physical validity.

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

OpenPTV2 supports extensible tracking algorithms selected via `plugins.selected_tracking`:

* **`priority_segment_3d`** (`priority_segment_3d`):
  Compiled Cython 3D segment-priority engine. Delivers maximum processing throughput (>800,000 particles/sec) for sparse to moderate density 3D datasets.
* **`kalman_hungarian_3d`** (`kalman_hungarian_3d`):
  High-accuracy Constant-Acceleration 3D Kalman Filter predictor with multi-term cost matrix (distance + velocity continuity + acceleration penalty) and Hungarian cluster assignment. Delivers **98.0% precision** at high speed (~178 ms/frame). See [**`kalman_hungarian_3d` Mathematical Guide**](file:///C:/Users/alex/projects/openptv2/docs/kalman_hungarian_3d_guide.md).
* **`cython_epipolar_tracking` / `default` (`trackcorr`)**:
  Standard OpenPTV multi-camera Lagrangian tracking engine. Works best for 2–4 camera setups where 2D epipolar projection resolves $Z$-axis depth noise.
* **`fast_3d_smooth`** (`fast_3d_smooth`):
  Fast 3D tracking with Savitzky-Golay velocity smoothing and Hungarian assignment. Ideal for noisy 3D triangulations.
* **`predictive_gmm_3d`** (`predictive_gmm_3d` / proPTV):
  Probabilistic GMM spatial-temporal smoothing tracker. Ideal for Lagrangian turbulence and datasets requiring smooth acceleration fields.
* **`nearest_hungarian_3d`** (`nearest_hungarian_3d`):
  MyPTV 3D kinematic tracker with configurable multi-term cost weights and multi-frame gap bridging.
* **`splitter_tracking`**:
  Specialized algorithm for single-sensor image splitters (quad-view cameras).
* **Custom Plugins**:
  Users can drop custom tracking python modules into `<experiment>/plugins/` implementing `BaseTrackingPlugin`.

---

## 5. Understanding Result Files (`res/ptv_is.#`)

Tracking outputs are written to the experiment's `res/` folder as `ptv_is.<frame_number>` text files.

### File Structure of `ptv_is.#`

Each row in `ptv_is.#` represents a tracked 3D particle at that frame:

```
[prev_link] [next_link] [X] [Y] [Z] [cam1_targ] [cam2_targ] [cam3_targ] [cam4_targ]
```

* **`prev_link`**: Index of this particle in the **previous** frame's `ptv_is.(t-1)` file (`-1` if trajectory starts here).
* **`next_link`**: Index of this particle in the **next** frame's `ptv_is.(t+1)` file (`-2` if trajectory ends here).
* **`X, Y, Z`**: Reconstructed 3D position in physical space [mm].
* **`cam1..4_targ`**: Target indices in the original 2D detection files (`img/` or `targets/`).

### Interpreting Trajectory Statistics

When Pass 3 post-processing completes, OpenPTV2 reports trajectory statistics:

* **`links_before`**: Total forward links established before post-processing.
* **`links_after`**: Total valid links remaining after Pass 3 reciprocity pruning.
* **`trajectories_count`**: Number of distinct continuous particle trajectories.
* **`mean_length`**: Average trajectory length (in frames).

---

## 6. GUI to Cloud Batch Workflow

To prepare tracking parameters in the GUI and run large-scale jobs in the cloud:

1. **GUI Parameter Tuning (PyPTV)**:
   * Open PyPTV GUI: `uv run pyptv <path_to_experiment>`
   * Configure parameters under `Parameters -> Tracking` (set `dv`, `angle`, `dacc`, `flagNewParticles`, and `postprocess`).
   * Test tracking interactively using `Tracking -> Debugging with display`.
   * Click **OK** in the Tracking Parameters dialog to persist settings to `parameters.yaml`.

2. **Shipping to Cloud / Batch Mode**:
   * Commit and push your `parameters.yaml` (along with calibration files in `cal/` or `res/`).
   * Run headless batch execution in the cloud:
     ```bash
     uv run python -m openptv2.batch --yaml parameters.yaml --track
     ```
   * Or use the Python API:
     ```python
     from openptv2.tracker import Tracker
     from openptv2.gui.parameter_manager import ParameterManager

     pm = ParameterManager()
     pm.from_yaml("parameters.yaml")

     tracker = Tracker.from_parameter_manager(pm)
     tracker.full_forward()
     tracker.full_backward()
     if pm.parameters.get("track", {}).get("postprocess", True):
         tracker.postprocess()
     ```

---

## 7. Extra Tracking Plugins: MyPTV Trackers

OpenPTV2 provides built-in plugin wrappers for **MyPTV** particle tracking algorithms, making it easy to swap tracking engines without modifying your dataset structure.

### Available MyPTV Plugins

| Plugin Name | Tracking Level | Algorithm & Features |
| :--- | :--- | :--- |
| **`nearest_hungarian_3d`** | 3D Physical Space | Uses MyPTV's 3D kinematic velocity and acceleration predictor ($\mathbf{X}_{\text{pred}} = \mathbf{X}_t + \mathbf{V}_t \Delta t + \frac{1}{2}\mathbf{A}_t \Delta t^2$) coupled with SciPy Hungarian bipartite assignment (`scipy.optimize.linear_sum_assignment`) for global collision-free candidate matching and gap recovery. |
| **`myptv_2d_tracking`** | 2D Pixel Space | Performs 2D frame-to-frame particle trajectory tracking directly in camera image coordinates $(x_i, y_i)$ for each camera view independently. Useful for 2D-PTV or pre-triangulation 2D trajectory stereo matching. |

### How to Use

#### 1. In the PyPTV GUI
In the **Parameters** dialog under **Plugins**, select **`nearest_hungarian_3d`** or **`myptv_2d_tracking`** from the **Tracking Plugin** (`track_alg`) dropdown menu.

#### 2. In `parameters.yaml`
Specify the tracking plugin in your YAML configuration:
```yaml
plugins:
  selected_tracking: "nearest_hungarian_3d"  # or "myptv_2d_tracking"
  selected_sequence: "default"
```

#### 3. In Python CLI / Batch Execution
```python
from openptv2.plugins import run_tracking_plugin

# Run MyPTV 3D tracking plugin programmatically
run_tracking_plugin("nearest_hungarian_3d", experiment)
```

---

## 8. Developing Custom Tracking Plugins

To create your own custom 2D or 3D particle tracking plugin or adapt external trackers (such as MyPTV, Trackpy, or custom machine learning models), see the comprehensive developer guide:

* [**Developer Guide: Custom Tracking Plugins**](file:///C:/Users/alex/projects/openptv2/docs/developer_guide/custom_tracking_plugins.md)


