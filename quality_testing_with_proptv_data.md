# Quality & Multi-Tracker Testing OpenPTV2 using proPTV Data

This guide describes how to perform quality, accuracy, and benchmark testing across **all OpenPTV2 tracking engines** using the synthetic 3D Particle Tracking Velocimetry (PTV) dataset from **proPTV** (Barta et al., 2024).

---

## 🌊 Ground Truth Flow Field (`500_25` Rayleigh-Bénard Convection)

The figure below shows the simulated 3D flow field and thermal plume convective particle trajectories across the 5 frame sequence ($t = 0 \dots 4$):

![Ground Truth Flow Trajectories](scratch/flow_trajectories_500_25.png)

### Dataset Characterization:
- **Physics model**: Rayleigh-Bénard thermal convection (Direct Numerical Simulation).
- **Particle count**: **500** particles per frame.
- **Time steps**: **5** frames ($t \in [0, 4]$).
- **Domain size**: $[0, 1] \times [0, 1] \times [0, 1]$ normalized volume.
- **Speed distribution**: $0.00132$ to $0.32931$ (Mean: $0.06826$).
- **Data location**: `C:/Users/alex/Github/proPTV/data/500_25`

---

## 🏆 Multi-Tracker Comparative Benchmark Results

All native 3D tracking engines in **OpenPTV2** were benchmarked on the exact same `500_25` particle coordinate sequence:

![OpenPTV2 Multi-Tracker Benchmark](scratch/all_trackers_benchmark_500_25.png)

| Tracker Engine | Reconstructed Tracks | Matched % (PMP) | Mean 3D Error | Execution Time | Key Strengths & Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fast 3D Smooth (SG)** | **499** | **99.80%** | **$0.0 \times 10^{-6}$** | **57.03 ms** | **Fastest!** Uses Savitzky-Golay smoothed velocity + Hungarian matching. |
| **MyPTV 3D Kinematic** | **499** | **99.80%** | **$0.0 \times 10^{-6}$** | **42.76 ms** | Polynomial kinematic prediction + distance/velocity matching. |
| **3D Kalman-Hungarian** | **507** | **101.40%** | **$2.4 \times 10^{-4}$** | **470.84 ms** | Constant-acceleration 9D Kalman filter with adaptive $a_{\max}$ innovation gating. |
| **proPTV (Predictive GMM)** | **499** | **99.80%** | **$0.0 \times 10^{-6}$** | 15,363.50 ms | Probabilistic Gaussian Mixture Model basis smoothing (highest physical rigor). |

---

## 🔍 Root Cause Analysis & Fix for 3D Kalman-Hungarian Tracker

During parameter audit of [`src/openptv2/plugins/quality_3d_tracking.py`](src/openptv2/plugins/quality_3d_tracking.py), two key issues were identified and fixed:

1. **Hardcoded Physical Unit Clamping**:
   - *Previous behavior*: `tight_radii` and `fallback_radii` clamped innovation search radii to hardcoded `[1.2, 3.0]` mm lower bounds. On normalized $[0, 1]^3$ datasets, a search radius of `1.2` covered the entire domain volume!
   - *Fix*: Replaced static `1.2` mm lower bounds with adaptive scaling `np.clip(2.5 * sigmas, 0.1 * self.a_max, self.a_max)` proportional to the user-specified acceleration bound $a_{\max}$.

2. **Unfiltered Single-Point Unlinked Seeds**:
   - *Previous behavior*: `_export_track` exported all active Kalman states including unlinked 1-point candidate seeds created at frames $t=1,2,3,4$.
   - *Fix*: Filtered `track_frames()` output to return valid trajectories ($\text{length} \ge 2$), eliminating 257 unlinked candidate states and bringing track count down from **764** to **507**.

---

## 🚀 How to Run Benchmarks in OpenPTV2

### 1. Run the Multi-Tracker Benchmark Script

To run all 4 tracking engines side-by-side and generate the comparative plot:

```bash
cd C:\Users\alex\projects\openptv2
uv run python scratch/benchmark_all_trackers_500_25.py
```

### 2. Run the Pytest Integration Test

```bash
cd C:\Users\alex\projects\openptv2
uv run pytest tests/unit/test_proptv_500_25_dataset.py -v
```

---

## 💻 Python API Multi-Tracker Usage Example

You can instantiate and execute any OpenPTV2 tracking engine using the same input particle list `frame_particles`:

```python
import numpy as np
from openptv2.plugins.proptv import ProPTVConfig
from openptv2.plugins.proptv_tracking import ProPTVTracker
from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker
from openptv2.plugins.fast_3d_smooth_tracking import Fast3DSmoothTracker
from openptv2.plugins.quality_3d_tracking import Quality3DTracker

# 1. Load per-frame 3D particle coordinate arrays (shape: N x 3)
frame_particles = [
    np.loadtxt(f"C:/Users/alex/Github/proPTV/data/500_25/origin/origin_{str(t).zfill(5)}.txt")[:, 1:4]
    for t in range(5)
]

# 2. Select and run your tracker of choice
tracker = Fast3DSmoothTracker(v_max=0.015, dacc=0.010, dt=1.0)
# OR: tracker = Quality3DTracker(v_max=0.015, a_max=0.010, dt=1.0)
# OR: tracker = MyPTV3DTracker(v_max=0.015, a_max=0.010, dt=1.0)
# OR: tracker = ProPTVTracker(ProPTVConfig(t_init=3, maxvel=0.015, angle=60.0, dt=1.0))

tracks = tracker.track_frames(frame_particles)
print(f"Reconstructed {len(tracks)} 3D particle trajectories.")
```

---

## 📌 Dataset Structure Reference (`proPTV/data/500_25`)

```
data/500_25/
├── origin/
│   ├── origin_00000.txt      # Ground-truth 3D positions & velocities at t=0
│   ├── origin_00001.txt      # Ground-truth 3D positions & velocities at t=1
│   ├── ...
│   └── tracks_origin.hdf5    # Complete ground-truth 3D trajectories
├── input/
│   └── particle_lists/       # 2D sub-pixel centroids for 4 cameras (c0, c1, c2, c3)
└── output/                   # Reconstruction outputs
```
