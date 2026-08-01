# Zarr + HDF5 Storage Guide in OpenPTV2

OpenPTV2 includes a high-performance **Zarr + HDF5 Storage System** (`openptv2.storage.ZarrFrameStore`). This system replaces legacy per-frame ASCII text files (`*_targets`, `rt_is.*`, `ptv_is.*`, `added.*`) with a single, highly compressed, cloud-native Zarr dataset (`res/run.zarr`).

---

## Why Zarr + HDF5?

- **Eliminates File System Bottlenecks**: High-frame-rate, multi-camera PTV experiments typically produce tens of thousands of small text files. Zarr consolidates everything into a clean chunked store.
- **Cloud & Cluster Parallelism**: Multiple parallel worker processes or cloud nodes (e.g., AWS Lambda, GCP Cloud Run) can process different frames and write to distinct chunk keys simultaneously without file locking or storage contention.
- **Flowtracks & HDF5 Native Bridge**: Direct export to Flowtracks-compliant HDF5 files (`.h5`) for 3D Lagrangian trajectory analysis, turbulence statistics, and ParaView exports.
- **Transparent ASCII Inspection**: Built-in CLI and Python inspectors allow humans to view binary Zarr frame data formatted as traditional ASCII text at any time.

---

## Data Architecture inside `res/run.zarr`

Inside a `.zarr` directory (e.g., `res/run.zarr`), data is organized into logical subgroups:

| Zarr Path / Group | Legacy Text Equivalent | Description |
| :--- | :--- | :--- |
| `targets/cam_<idx>/frame_<num>` | `cam1.10000_targets` | 2D detected targets per camera ($x, y, n, nx, ny, \text{sumg}, \text{tnr}$). |
| `correspondences/frame_<num>` | `res/rt_is.10000` | 3D stereo-matched particle coordinates ($x, y, z$ in mm) and camera target IDs. |
| `linkage/ptv_is/frame_<num>` | `res/ptv_is.10000` | 3D particle tracking linkage links ($\text{prev}, \text{next}, x, y, z$). |
| `linkage/added/frame_<num>` | `res/added.10000` | Multi-pass tracking added particle linkages ($\text{prev}, \text{next}, x, y, z$). |
| `trajectories/` | Flowtracks dataset | Consolidated particle tracks ($\text{pos}, \text{vel}, \text{acc}, \text{frame}, \text{trajid}$). |

---

## Storage Modes (`OPENPTV_STORAGE`)

You can control storage behavior using the `OPENPTV_STORAGE` environment variable:

```bash
export OPENPTV_STORAGE=zarr        # DEFAULT mode
export OPENPTV_STORAGE=zarr_only   # High-performance cloud mode
export OPENPTV_STORAGE=legacy      # Legacy text-file-only mode
```

### Mode Comparison

1. **`zarr` (DEFAULT)**:
   - Writes all targets, correspondences, and tracking linkages to `res/run.zarr`.
   - **Dual-writes** legacy text files to disk so existing legacy tools, scripts, and tests continue to work without modification.

2. **`zarr_only` (Cloud Mode)**:
   - Writes **ONLY** to `res/run.zarr`.
   - **Creates 0 text files on disk**, completely eliminating disk write I/O overhead. Ideal for distributed cloud workflows (`openptv-cloud`).

3. **`legacy`**:
   - Writes only legacy ASCII text files (`*_targets`, `rt_is.*`, `ptv_is.*`).

---

## How to Inspect & "Peek In" Data

### 1. Terminal Inspector Tool (CLI)

You do not need to write Python code to view frame data inside `res/run.zarr`. Use the built-in CLI inspector:

```bash
# View 2D targets for Camera 0 at Frame 10000
uv run python -m openptv2.storage.zarr_store res/run.zarr --frame 10000 --type targets --cam 0

# View 3D correspondences (rt_is) at Frame 10000
uv run python -m openptv2.storage.zarr_store res/run.zarr --frame 10000 --type rt_is

# View 3D tracking links (ptv_is) at Frame 10000
uv run python -m openptv2.storage.zarr_store res/run.zarr --frame 10000 --type ptv_is

# View added particles from multi-pass tracking at Frame 10000
uv run python -m openptv2.storage.zarr_store res/run.zarr --frame 10000 --type added
```

#### Example Output (`rt_is` inspection):
```
12
   0   -12.450    34.120   120.450    1    3    0   -1
   1    45.120   -10.330   118.900    2    1    5    4
...
```

---

### 2. Python Inspection API (`ZarrFrameStore`)

In Python, print or format any frame directly:

```python
from openptv2.storage import ZarrFrameStore

store = ZarrFrameStore("res/run.zarr", mode="r")

# Print 3D correspondences directly to console in legacy ASCII format
store.dump_frame_text(frame=10000, dataset_type="rt_is")

# Print 2D targets for camera 0
store.dump_frame_text(frame=10000, dataset_type="targets", cam_idx=0)

# Get the formatted text as a string
text_str = store.export_frame_text(frame=10000, dataset_type="ptv_is")
```

---

### 3. Direct NumPy / Zarr Tree Access

Access raw arrays directly for custom analysis or inspect the tree structure:

```python
import zarr

root = zarr.open("res/run.zarr", mode="r")

# Print directory hierarchy
print(root.tree())

# Read 3D particle positions as a NumPy array (N, 3)
pos_3d = root["correspondences/frame_10000/pos"][:]
print("Frame 10000 3D positions shape:", pos_3d.shape)
```

---

### 4. Flowtracks HDF5 Export

Export full trajectory datasets directly to Flowtracks-compliant HDF5 files (converting coordinates from mm to meters):

```python
from openptv2.storage import ZarrFrameStore

store = ZarrFrameStore("res/run.zarr", mode="r")
store.to_flowtracks_h5("res/trajectories.h5")
```

---

## Running Batch Processing in Zarr Mode

To run batch sequence processing in Zarr mode from Python or CLI:

```bash
# Enable multi-process target detection + Zarr storage
export OPENPTV_PARALLEL_PREPROCESS=True
export OPENPTV_STORAGE=zarr_only

# Run batch sequence
pyptv_batch tests/test_cavity/parameters_Run1.yaml 10000 10004
```

Or from Python API:

```python
import os
from openptv2.batch.pyptv_batch import main

os.environ["OPENPTV_STORAGE"] = "zarr_only"
main("tests/test_cavity/parameters_Run1.yaml", 10000, 10004)
```
