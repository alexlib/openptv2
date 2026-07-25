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

### Parameter Explanations

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
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

---

## 4. Tracking Algorithms & Plugins

OpenPTV2 supports extensible tracking algorithms selected via `plugins.selected_tracking`:

* **`default` (`trackcorr`)**:
  Standard OpenPTV Lagrangian tracking engine. Works best for 3D PTV setups with 2-4 cameras.
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
