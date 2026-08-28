# Tutorial: Command-Line Batch Processing with OpenPTV2

Batch processing runs the Cython 3 engine headlessly — ideal for large datasets, overnight runs, or clusters.

---

## CLI Command Structure

```bash
uv run openptv2-batch <experiment_directory_or_yaml> <first_frame> <last_frame> [options]
```

### Key Positional Arguments:
* `<experiment_directory_or_yaml>`: Directory or `.yaml` (e.g., `parameters_Run1.yaml`). Directory auto-selects first YAML.
* `<first_frame>`, `<last_frame>`: Inclusive frame range.

### Major Optional Flags:
* `--mode <both|sequence|tracking>`:
  * `both` (Default): Full pipeline → `res/run.zarr` (targets, correspondences, linkage, trajectories)
  * `sequence`: Detection + correspondence only → `correspondences/` + `targets/`
  * `tracking`: Tracking only → reads existing `correspondences/`/`targets/` → writes `linkage/` + `trajectories/`/`traj` via `seal`
* `--track3d`: 3D segment tracking.
* `--sequence-plugin <name>` / `--tracking-plugin <name>`: Alternate strategy (`default`, `two_phase`, `myptv_3d_tracking`, …). Example for splitter:
  ```bash
  uv run openptv2-batch test_data/test_splitter 1000001 1000002 \
    --sequence-plugin splitter_sequence --tracking-plugin splitter_tracking
  ```
  See [Plugins tutorial](plugins.md).
* `--output <name>` (**New**): Copy result to `res/<name>` without overwriting `res/run.zarr`. Example benchmark-safe runs:
  ```bash
  uv run openptv2-batch <exp> 1 50 --mode tracking --tracking-plugin default --output bench_default.zarr
  uv run openptv2-batch <exp> 1 50 --mode tracking --tracking-plugin two_phase --output bench_two_phase.zarr
  # res/run.zarr preserved, outputs in bench_*.zarr
  ```
  Python API: `main(yaml, 1, 50, mode="tracking", tracking_plugin="two_phase", output="bench.zarr")` (`src/openptv2/batch/pyptv_batch.py:264`).

---

## Guided Walkthrough: Cavity Flow Dataset

### 1. Identifying the Frame Range

`test_data/test_cavity/parameters_Run1.yaml`:
```yaml
sequence:
  first: 10001
  last: 10004
```

### 2. Standard Tracking Mode (`--mode tracking`)

Run tracking only using pre-existing zarr correspondences:

```bash
uv run openptv2-batch test_data/test_cavity 10001 10004 --mode tracking
```

#### Expected Output:
```text
Starting batch processing with YAML file: .../parameters_Run1.yaml
Frame range: 10001 to 10004
Running tracking plugin: default
track3d step: 1, curr: 672, next: 699, links: 447
...
Sealed: {'n_trajectories': 1148, 'n_rows': 3121, 'n_dropped': 0}
Batch processing completed successfully
```

All results in `res/run.zarr` (`correspondences/`, `linkage/ptv_is/`, `trajectories/`, `traj/`). Inspect via `zarr.open_group("res/run.zarr")` (`docs/zarr-hdf5-storage.md`). `seal` (`src/openptv2/storage/seal.py:73`) builds flat `trajectories` + index `traj` with `min_trajectory_length` filtering (default 5, set in `track:`).

### 3. Full Pipeline Mode (`--mode both`)

```bash
uv run openptv2-batch test_data/test_cavity 10001 10004 --mode both
```

> Requires tuned `targ_rec`/`detect_plate` thresholds — use GUI (`uv run pyptv`) first.

---

## Benefits of Single-Engine Batch Mode
1. **Headless** — no GUI/X11, runs over SSH/Docker.
2. **C-compiled** — Cython 3 loops at C speed.
3. **Reproducible** — YAML is the sole provenance; `res/run.zarr` + `seal` with `source_hash` memoization.

## Viewing Trajectories

```python
import zarr, numpy as np
root = zarr.open_group("res/run.zarr", mode="r")
traj = root["traj"]; ln = np.asarray(traj["length"]); fr = np.asarray(traj["first_row"])
top = np.argsort(ln)[::-1][:100]
for tid, fr_i, ln_i in zip(np.asarray(traj["trajid"])[top], fr[top], ln[top]):
    pts = np.asarray(root["trajectories/pos"][fr_i:fr_i+ln_i])  # [m]
```

Or `notebooks/marimo_trajectory_viewer.py` (plotly, top-N longest).

See also `docs/zarr-hdf5-storage.md` (copying to Dropbox) and `docs/tracking_guide.md`.
