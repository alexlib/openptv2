# Zarr Storage Guide in OpenPTV2

OpenPTV2 uses a single **Zarr v3 store** `res/run.zarr` (`openptv2.storage.RunStore`, `openptv2.storage.ZarrFrameStore`) as the sole database for 2D targets, 3D correspondences, linkage, and trajectories. Legacy per-frame ASCII files (`*_targets`, `rt_is.*`, `ptv_is.*`) are **not written**.

---

## Why Zarr?

- **Single store** replaces tens of thousands of text files — chunked, compressed, cloud-native.
- **Parallelism** — workers write distinct `frame_*` keys without file locking.
- **Flowtracks bridge** — `trajectories/` layout matches `flowtracks.ZarrScene` / `openptv2.storage.seal` output.

---

## Data Architecture inside `res/run.zarr`

| Zarr Path | Legacy Equivalent | Description |
| :--- | :--- | :--- |
| `targets/cam_<c>/frame_<n>` | `cam1.10000_targets` | 2D targets per camera `(x,y,n,nx,ny,sumg)` |
| `correspondences/frame_<n>` | `res/rt_is.10000` | 3D positions `(x,y,z [mm], cam_ids)` — `(N,3+C)` |
| `linkage/ptv_is/frame_<n>/{prev,next,pos}` | `res/ptv_is.10000` | Linkage: `prev/next` (int32, -1=none) + `pos` (m) per particle |
| `linkage/added/frame_<n>/{prev,next,pos}` | `res/added.10000` | Second pass (unused by default) |
| `trajectories/{pos,vel,accel,time,trajid}` | Flowtracks | **Flat cache** — `pos` in **meters** (`mm*1e-3`), sorted by `(trajid,time)` (`run_store.py:530`) |
| `traj/{trajid,first,last,length,first_row}` | — | **Index** — per-trajectory summary, `first_row` = row offset in `trajectories/pos` (`run_store.py:480`) |
| `meta/{sealed,source_hash}` | — | Seal provenance; `sealed=False` before `seal()` |

Example (`run.zarr` with `seal(min_length=5)`): 5005 frames → `trajectories` 5,791,200 rows, `traj` 301,733 entries (median 10, mean 19.2).

### Seal: Linkage → Flat Cache

```python
from openptv2.storage import RunStore
from openptv2.storage.seal import seal  # seal.py:73

store = RunStore("res/run.zarr", mode="a")
seal(store, min_length=5)  # walks linkage, assigns trajid, writes traj+trajectories
# Batch does this automatically after tracking: batch/pyptv_batch.py:246
```

- Filters `length < min_length` *before* writing; `n_dropped` in return.
- `source_hash` memoizes — skips if linkage unchanged.
- `first_row` enables `pos[lo:hi]` without loading 66 MB `trajid` + `searchsorted` (`notebooks/marimo_trajectory_viewer.py:33`).

---

## Inspecting Data

### 1. Python `zarr` API (recommended)

```python
import zarr, numpy as np

root = zarr.open_group("res/run.zarr", mode="r")
print(root.tree())

# Flat trajectories — top 100 longest without loading 135 MB
traj = root["traj"]
tid = np.asarray(traj["trajid"])
ln = np.asarray(traj["length"])
fr = np.asarray(traj["first_row"])
order = np.argsort(ln)[::-1][:100]
for tid_i, fr_i, ln_i in zip(tid[order], fr[order], ln[order]):
    pts = np.asarray(root["trajectories/pos"][fr_i : fr_i + ln_i])  # [m]

# Per-frame linkage
prev, nxt, pos = (
    np.asarray(root["linkage/ptv_is/frame_000001/prev"]),
    np.asarray(root["linkage/ptv_is/frame_000001/next"]),
    np.asarray(root["linkage/ptv_is/frame_000001/pos"]),
)
```

### 2. `ZarrFrameStore` helpers

```python
from openptv2.storage import ZarrFrameStore

store = ZarrFrameStore("res/run.zarr", mode="r")
store.dump_frame_text(frame=10000, dataset_type="rt_is")  # legacy ASCII view
```

### 3. CLI (if installed)

```bash
uv run python -m openptv2.storage.zarr_store res/run.zarr --frame 10000 --type rt_is
```

### 4. Copying Trajectories Only (Dropbox)

```bash
# Standalone zarrs (137 MB vs 1637 MB full store)
uv run python C:/Users/alex/Downloads/TT13_aorta/wp1/copy_trajectories.py --include-traj --overwrite --verify
# Creates trajectories.zarr (135 MB) + traj.zarr (1.9 MB) — filesystem copy of run.zarr subgroups
```

---

## Batch Processing with Zarr

```bash
uv run openptv2-batch <exp_or_yaml> <first> <last> --mode both
uv run openptv2-batch <exp> 1 50 --mode tracking --tracking-plugin two_phase --output bench_two_phase.zarr  # preserves run.zarr
```

From Python: `from openptv2.batch.pyptv_batch import main; main(yaml, 1, 5005, mode="tracking", output="bench.zarr")` (`batch/pyptv_batch.py:264`)

---

## Chunked Images (`res/images.zarr`) — Optional

When `res/images.zarr` exists (Zstd, ~1.2 GB vs 5 GB TIFF), batch reads frames from zarr chunks instead of `img/*.tif` — zero TIFF I/O, lower RAM. GUI falls back to `img/` if absent.

---

## Migration Notes

- **HDF5 / `OPENPTV_STORAGE` env** removed — single `res/run.zarr` is the only store.
- **Text files** not written; use `zarr` API or `ZarrFrameStore.export_frame_text` for ASCII.
- **Units**: `correspondences/linkage pos` in mm; `trajectories/pos` in meters (flowtracks convention).
