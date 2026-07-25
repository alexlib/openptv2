# Developing Custom Tracking Plugins for OpenPTV2

This guide explains how to adapt and implement custom 2D or 3D particle tracking algorithms (such as MyPTV, trackpy, or custom ML/neural trackers) into **OpenPTV2** plugins.

---

## 1. Overview & Plugin Architecture

OpenPTV2 features a plugin architecture (`openptv2.plugins.loader`) that resolves custom algorithms at runtime. You can deliver custom tracking plugins in three ways:

1. **Built-in Plugins**: Shipped inside `src/openptv2/plugins/` (e.g. `myptv_3d_tracking`, `myptv_2d_tracking`, `splitter_tracking`).
2. **Experiment-Local Plugins**: Dropped directly into `<experiment>/plugins/my_tracker.py` for dataset-specific algorithms.
3. **Third-Party Packages**: Distributed via `pyproject.toml` entry points (`openptv2.plugins`).

---

## 2. The Tracking Plugin Protocol (`Tracking` Class)

Every tracking plugin MUST define a top-level `Tracking` class conforming to the OpenPTV2 plugin contract:

```python
class Tracking:
    """Plugin interface for custom tracking algorithms."""

    def __init__(self, ptv=None, exp=None):
        """
        Parameters
        ----------
        ptv : module or object
            OpenPTV2 core module/bindings (e.g., openptv2.gui.ptv).
        exp : Experiment
            Active OpenPTV2 Experiment instance containing parameters (exp.pm)
            and runtime data structures.
        """
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self) -> None:
        """Main execution entry point invoked by GUI, batch scripts, or CLI."""
        # Read parameters, extract particle arrays, run algorithm, and save results
        ...
```

---

## 3. Adapting External Algorithms to OpenPTV2

When bringing an algorithm from an external codebase (e.g., MyPTV, Trackpy, PyFLOW), you need to address three key adapter layers:

### A. Data Structure Adapter
* **External Model**: Algorithms often expect time-series arrays or DataFrames `[x, y, z, frame, id]`.
* **OpenPTV2 Model**: Uses frame-centric linked pointers (`prev_link` / `next_link` in `ptv_is.#` files).
* **Solution**:
  1. Extract frame positions into a `list[np.ndarray]` where `frames[i]` is an $(N_i, 3)$ matrix of particle positions at frame $i$.
  2. Execute the tracking algorithm.
  3. Convert the resulting trajectory IDs into OpenPTV2 `prev_link` and `next_link` row indices.

### B. Parameter Mapping Adapter
Map OpenPTV2 parameters in `parameters.yaml` under `track:` to your algorithm's configuration:
* `dvxmin`, `dvxmax` $\rightarrow$ Max velocity bounds ($v_{\text{max}}$)
* `dacc` $\rightarrow$ Acceleration limits ($a_{\text{max}}$)
* `angle` $\rightarrow$ Max direction change angle
* `Sequence.first`, `Sequence.last` $\rightarrow$ Frame range $0 \dots N-1$

### C. Isolating Core Mathematics (Pure NumPy/SciPy)
Instead of requiring heavy external dependencies (GUIs, pandas, custom file formats), isolate the core mathematical logic (distance matrix calculation, kinematic prediction, bipartite matching) into pure NumPy/SciPy functions (`scipy.optimize.linear_sum_assignment`).

---

## 4. Step-by-Step Code Walkthrough (MyPTV Case Study)

Below is a complete, annotated example based on our MyPTV 3D tracking plugin implementation ([`myptv_3d_tracking.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/plugins/myptv_3d_tracking.py)):

### Step 1: Implement the Mathematical Core Engine

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

class Custom3DTracker:
    def __init__(self, v_max: float = 10.0, a_max: float = 50.0, dt: float = 0.1):
        self.v_max = v_max
        self.a_max = a_max
        self.dt = dt

    def track_frames(self, frame_particles: list[np.ndarray]) -> list[dict]:
        """
        frame_particles: list of (N_i, 3) arrays per frame.
        returns: list of trajectory dicts {'id', 'pos', 'time', 'vel'}
        """
        num_frames = len(frame_particles)
        active_tracks, completed_tracks = [], []
        next_id = 1

        # Frame 0 initialization
        for p in frame_particles[0]:
            active_tracks.append({"id": next_id, "pos": [p], "time": [0], "vel": [np.zeros(3)]})
            next_id += 1

        # Process frames 1 .. N-1
        for f in range(1, num_frames):
            cands = frame_particles[f]
            cost_matrix = np.full((len(active_tracks), len(cands)), 1e9, dtype=np.float64)

            # Build prediction cost matrix
            for i, tr in enumerate(active_tracks):
                last_p = tr["pos"][-1]
                p_pred = last_p + tr["vel"][-1] * self.dt if len(tr["pos"]) > 1 else last_p
                search_r = self.a_max * (self.dt ** 2) if len(tr["pos"]) > 1 else self.v_max * self.dt

                dists = np.linalg.norm(cands - p_pred, axis=1)
                valid = dists <= search_r
                cost_matrix[i, valid] = dists[valid]

            # Solve global assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # Update tracks and candidates...
            ...
        return completed_tracks
```

### Step 2: Implement the OpenPTV2 Plugin Interface (`Tracking` Class)

```python
class Tracking:
    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self) -> None:
        # 1. Read parameters from OpenPTV2 ParameterManager
        pm = getattr(self.exp, "pm", None)
        track_cfg = pm.parameters.get("track", {}) if pm else {}
        dvmax = float(track_cfg.get("dvxmax", 10.0))
        dacc = float(track_cfg.get("dacc", 50.0))

        # 2. Instantiate custom tracker
        tracker = Custom3DTracker(v_max=dvmax, a_max=dacc)

        # 3. Read frame particles from OpenPTV2
        tracker_c = self.ptv.py_trackcorr_init(self.exp)
        self.exp.tracker = tracker_c

        # 4. Run tracking
        tracker_c.full_forward()
```

---

## 5. Registering and Using Your Custom Tracker

### Option A: Local Experiment File
Place your plugin in `<experiment_folder>/plugins/my_custom_tracker.py`. OpenPTV2 will automatically discover it when loading the experiment!

### Option B: Built-in Registration
Add your module path to `BUILTIN_TRACKING_PLUGINS` in `src/openptv2/plugins/loader.py`:
```python
BUILTIN_TRACKING_PLUGINS = {
    ...
    "my_custom_tracker": "openptv2.plugins.my_custom_tracker",
}
```

### Option C: Python Package Entry Points (`pyproject.toml`)
If distributing your tracker as an installable Python package:
```toml
[project.entry-points."openptv2.plugins"]
my_custom_tracker = "my_package.tracker_plugin"
```

---

## 6. GUI Integration & Parameter Persistence

Custom plugins integrate automatically into the OpenPTV2 GUI:
* In the **Plugins** dialog (`Parameters -> Plugins`), your custom plugin appears in the **Tracking** dropdown list.
* In the **Tracking Parameters** dialog (`Parameters -> Tracking`), custom plugins map to the **`Custom / Plugin Algorithm`** (`custom_plugin`) strategy preset, and the **Active Selected Plugin** label displays your plugin name.
* Selections persist automatically to `parameters.yaml`:
  ```yaml
  plugins:
    selected_tracking: "my_custom_tracker"
  ```
