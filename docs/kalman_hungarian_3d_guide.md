# OpenPTV2 `kalman_hungarian_3d` Tracking Engine & Mathematical Guide

This document provides a comprehensive mathematical formulation, parameter selection guide, and benchmark verification instructions for the **`kalman_hungarian_3d`** particle tracking engine in OpenPTV2 ([`src/openptv2/plugins/kalman_hungarian_3d.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/plugins/kalman_hungarian_3d.py) and [`src/openptv2/tracking_kalman.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/tracking_kalman.py)).

---

## 1. Overview & Key Architecture

`kalman_hungarian_3d` is OpenPTV2's high-accuracy 3D particle tracking algorithm designed for turbulent flows, high particle densities, and noisy 3D observations. It combines:

1. **3D Constant-Acceleration Kalman Filter Prediction**: Maintains a full 9D kinematic state vector $(x, y, z, v_x, v_y, v_z, a_x, a_y, a_z)$ and error covariance $\mathbf{P}$ per trajectory.
2. **Dynamic Innovation & Search Window Gating**: Adapts candidate search radius based on track history and kinematic velocity bounds ($v_{\text{max}}, a_{\text{max}}$).
3. **Multi-Term Physical Cost Matrix**: Evaluates candidates using position displacement, velocity vector continuity, and acceleration change penalties.
4. **Cluster-Local Optimal Assignment**: Decomposes candidate graphs into isolated spatial clusters and solves collision-free optimal assignments using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`).
5. **Gap Persistence Bridge**: Preserves active tracks through temporary particle occlusions across $N$ missing frames (`max_gap`).

---

## 2. Mathematical Formulation

### 2.1 State Vector & Kinematic Motion Model

For each active trajectory, the particle state at time $t$ is represented by a 9D state vector $\mathbf{x}_t \in \mathbb{R}^9$:

$$\mathbf{x}_t = \begin{bmatrix} x & y & z & v_x & v_y & v_z & a_x & a_y & a_z \end{bmatrix}^T$$

Assuming constant acceleration over a time step $\Delta t$, the discrete-time state transition equation is:

$$\mathbf{x}_{t+\Delta t} = \mathbf{F}(\Delta t) \mathbf{x}_t + \mathbf{w}_t$$

where the state transition matrix $\mathbf{F}(\Delta t) \in \mathbb{R}^{9 \times 9}$ is defined as:

$$\mathbf{F}(\Delta t) = \begin{bmatrix} 
\mathbf{I}_3 & \Delta t \cdot \mathbf{I}_3 & \frac{1}{2}\Delta t^2 \cdot \mathbf{I}_3 \\ 
\mathbf{0}_3 & \mathbf{I}_3 & \Delta t \cdot \mathbf{I}_3 \\ 
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{I}_3 
\end{bmatrix}$$

and $\mathbf{I}_3$ is the $3 \times 3$ identity matrix.

### 2.2 Process & Measurement Covariance

The process noise vector $\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$ accounts for unmodeled turbulent acceleration fluctuations (jerk). For process noise spectral density $q = \sigma_a^2$:

$$\mathbf{Q}(\Delta t) = q \cdot \begin{bmatrix} 
\frac{\Delta t^5}{20}\mathbf{I}_3 & \frac{\Delta t^4}{8}\mathbf{I}_3 & \frac{\Delta t^3}{6}\mathbf{I}_3 \\ 
\frac{\Delta t^4}{8}\mathbf{I}_3 & \frac{\Delta t^3}{3}\mathbf{I}_3 & \frac{\Delta t^2}{2}\mathbf{I}_3 \\ 
\frac{\Delta t^3}{6}\mathbf{I}_3 & \frac{\Delta t^2}{2}\mathbf{I}_3 & \Delta t \cdot \mathbf{I}_3 
\end{bmatrix}$$

3D position measurements $\mathbf{z}_t = [x_m, y_m, z_m]^T \in \mathbb{R}^3$ are related to the state vector via measurement matrix $\mathbf{H} \in \mathbb{R}^{3 \times 9}$:

$$\mathbf{H} = \begin{bmatrix} \mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 \end{bmatrix}$$

The measurement error covariance $\mathbf{R} \in \mathbb{R}^{3 \times 3}$ represents 3D reconstruction accuracy:

$$\mathbf{R} = \sigma_m^2 \cdot \mathbf{I}_3$$

### 2.3 Prediction & Update Steps

#### Prediction Phase
$$\hat{\mathbf{x}}_{t|\text{prev}} = \mathbf{F}(\Delta t) \mathbf{x}_{\text{prev}}$$

$$\mathbf{P}_{t|\text{prev}} = \mathbf{F}(\Delta t) \mathbf{P}_{\text{prev}} \mathbf{F}(\Delta t)^T + \mathbf{Q}(\Delta t)$$

#### Measurement Innovation & Covariance
$$\mathbf{y}_t = \mathbf{z}_t - \mathbf{H} \hat{\mathbf{x}}_{t|\text{prev}}$$

$$\mathbf{S}_t = \mathbf{H} \mathbf{P}_{t|\text{prev}} \mathbf{H}^T + \mathbf{R}$$

#### Kalman Gain & Joseph-Form Covariance Update
$$\mathbf{K}_t = \mathbf{P}_{t|\text{prev}} \mathbf{H}^T \mathbf{S}_t^{-1}$$

$$\mathbf{x}_{t|t} = \hat{\mathbf{x}}_{t|\text{prev}} + \mathbf{K}_t \mathbf{y}_t$$

$$\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}) \mathbf{P}_{t|\text{prev}} (\mathbf{I} - \mathbf{K}_t \mathbf{H})^T + \mathbf{K}_t \mathbf{R} \mathbf{K}_t^T$$

The Joseph-form update guarantees numerical positive-definiteness of $\mathbf{P}_{t|t}$ under floating-point operations.

---

## 3. Candidate Matching & Multi-Term Cost Matrix

For each candidate particle $\mathbf{c}_j$ within search radius $r_s$ of predicted position $\hat{\mathbf{p}}_i = \mathbf{H} \hat{\mathbf{x}}_i$, matching cost $C_{i,j}$ is computed from three physical components:

$$C_{i,j} = w_{\text{dist}} \cdot \frac{\|\hat{\mathbf{p}}_i - \mathbf{c}_j\|}{r_s} + w_{\text{vel}} \cdot \frac{\|\hat{\mathbf{v}}_i - \mathbf{v}_{\text{cand}, j}\|}{v_{\text{max}}} + w_{\text{acc}} \cdot \frac{\|\mathbf{a}_i - \mathbf{a}_{\text{cand}, j}\|}{a_{\text{max}}}$$

where:
- $w_{\text{dist}}, w_{\text{vel}}, w_{\text{acc}}$ are relative term weights (default: $1.0, 0.5, 0.2$).
- $\mathbf{v}_{\text{cand}, j} = (\mathbf{c}_j - \mathbf{p}_{i, \text{prev}}) / \Delta t$ is candidate implied velocity.
- $\mathbf{a}_{\text{cand}, j} = (\mathbf{v}_{\text{cand}, j} - \mathbf{v}_{i, \text{prev}}) / \Delta t$ is candidate implied acceleration.

Candidate matching is solved per spatial cluster via bipartite graph optimal assignment (`scipy.optimize.linear_sum_assignment`), ensuring global minimum total cost and zero duplicate particle claims.

---

## 4. Parameter Settings & Configuration

`kalman_hungarian_3d` parameters can be specified in `parameters.yaml` or when instantiating `Quality3DTracker`.

### 4.1 Example `parameters.yaml` Configuration

```yaml
track:
  preset: "kalman_hungarian_3d"
  v_max: 6.0             # Maximum expected velocity displacement [mm/frame]
  a_max: 6.0             # Maximum expected acceleration displacement [mm/frame^2]
  max_gap: 1             # Max missing frames bridged during occlusions
  process_noise: 0.1     # Spectral process noise density [mm/frame^2.5]
  measurement_noise: 0.05# 3D position reconstruction noise [mm]
  cost_weights: [1.0, 0.5, 0.2]  # Position, velocity, and acceleration weights

plugins:
  selected_tracking: "kalman_hungarian_3d"
```

### 4.2 Parameter Selection Guide

| Parameter | Recommended Value | Selection Logic / Physics Basis |
| :--- | :--- | :--- |
| **`v_max`** | $1.2 \times U_{\text{max}} \cdot \Delta t$ | Set slightly higher than max expected particle displacement per frame [mm]. Cold start tracks use $v_{\text{max}}$ as initial search ball. |
| **`a_max`** | $1.5 \times \left|\frac{dU}{dt}\right|_{\text{max}} \cdot \Delta t^2$ | Maximum acceleration step [mm/frame$^2$]. Established tracks ($N \ge 2$ points) use $a_{\text{max}}$ for candidate search. |
| **`max_gap`** | `1` (or `2` for high noise) | Maximum consecutive unobserved frames to persist Kalman prediction before closing track. |
| **`process_noise`** | $0.05 \text{ to } 0.5$ | Increase for highly turbulent or curving flows; decrease for laminar linear flows. |
| **`measurement_noise`**| $0.02 \text{ to } 0.1$ mm | Equals 3D stereoscopic calibration residual RMS error. |
| **`cost_weights`** | `[1.0, 0.5, 0.2]` | Position weight $w_1=1.0$, velocity continuity $w_2=0.5$, acceleration penalty $w_3=0.2$. |

---

## 5. How to Run Benchmarks & Observe High-Quality Results

### 5.1 Running the Benchmark Command

Execute the multi-tracker benchmark suite across standard datasets using `uv`:

```bash
# Run multi-tracker benchmark on standard dataset
uv run python scripts/bench_trackers.py

# Or run via OpenPTV2 CLI tool
uv run openptv benchmark-tracking --flow vortex --particles 1000 --frames 20 --noise 0.05
```

### 5.2 Expected Quality Metrics

When evaluating `kalman_hungarian_3d` on `test_data/synthetic_turbulent` (225 particles/frame, 30 frames), you should observe results matching or exceeding:

| Metric | Target / Measured | Meaning & Physical Interpretation |
| :--- | :---: | :--- |
| **Precision** | **`0.980` (98.0%)** | Fraction of reconstructed trajectory links that match true physical particle motion. |
| **Recall** | **`0.893` (89.3%)** | Fraction of all true physical links successfully recovered by the tracker. |
| **Ghost Capture %** | **`3.56%`** | Rate of false candidate claims from spurious 3D ghost points. |
| **Track Fragmentation ($F$)** | **`3.83`** | Average number of track splits per physical particle trajectory. Lower is better. |
| **Continuum ($C$)** | **`1.00`** | Ratio of continuous steps without artificial frame skips. |
| **Track Purity** | **`0.971` (97.1%)** | Identity consistency along continuous trajectories (no ID swaps). |
| **PMT%** | **`75.4%`** | Perfect Match Trajectories (fraction of tracks recovered 100% unbroken). |
| **Speed** | **`178.7 ms/frame`** | Processing execution throughput time. |

---

## 6. Python API Usage Example

```python
from openptv2.plugins.kalman_hungarian_3d import Quality3DTracker
from openptv2.benchmarking.runner import read_trajectories
from pathlib import Path

# Instantiate Quality3DTracker with custom parameters
tracker = Quality3DTracker(
    v_max=6.0,
    a_max=6.0,
    max_gap=1,
    process_noise=0.1,
    measurement_noise=0.05,
    cost_weights=(1.0, 0.5, 0.2)
)

# Run tracker on an experiment directory reading res/rt_is.# and writing res/ptv_is.#
experiment_dir = Path("path/to/experiment")
tracker.track_directory(experiment_dir)

# Read reconstructed trajectories
trajectories = read_trajectories(experiment_dir / "res", first=10001, last=10030)
print(f"Successfully reconstructed {len(trajectories)} trajectories.")
```
