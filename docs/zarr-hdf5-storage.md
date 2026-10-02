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
| `linkage/ptv_is/block_<gen>_<first>_<last>/{frames,off,prev,next,pos[,prio,trajid]}` | `res/ptv_is.*` | Same linkage as **blocks** (tracker output, openptv2 ≥ 0.5.15): flat arrays + row offsets for many frames; see below |
| `quality/{frame_<n> or block_<gen>_<first>_<last>/{frames,off,ghost}}` | — | Per-point ghost probability from ray convergence |
| `linkage/added/frame_<n>/{prev,next,pos}` | `res/added.10000` | Second pass (unused by default) |
| `trajectories/{pos,vel,accel,time,trajid}` | Flowtracks | **Flat cache** — `pos` in **meters** (`mm*1e-3`), sorted by `(trajid,time)` (`run_store.py:530`) |
| `traj/{trajid,first,last,length,first_row}` | — | **Index** — per-trajectory summary, `first_row` = row offset in `trajectories/pos` (`run_store.py:480`) |
| `meta/{sealed,source_hash}` | — | Seal provenance; `sealed=False` before `seal()` |

Example (`run.zarr` with `seal(min_length=5)`): 5005 frames → `trajectories` 5,791,200 rows, `traj` 301,733 entries (median 10, mean 19.2).

### Linkage blocks (tracker output)

A per-frame zarr array costs ~1.2 ms of CPU locally and one request on an object
store, so per-frame linkage is thousands of tiny objects (measured on 1072 real
frames: 6.2 s to write, 4.0 s even with concurrent writes). The tracker writes its whole
result at once, so `RunStore.write_linkage_many` / `write_point_quality_many` store
it as **blocks** of up to 256 frames: `0.04 s` for the same data, ~40 objects instead
of ~3,200.

- **Commit.** A write creates blocks of a new generation and then publishes them with
  ONE attribute write (`linkage/<name>.attrs["blocks_gen"]`). Blocks above that
  generation (a halted write) are invisible and are removed by the next write, so a
  crash leaves the previous complete result.
- **Precedence.** For a frame in several committed blocks the newest generation wins
  (post-processing that touches a few frames writes one small block). A frame in no
  block falls back to its legacy `frame_<n>` group, so old stores read unchanged.
  Once a name has blocks, `write_linkage` for a single frame also writes a block.
- **One writer** per group at a time. Many concurrent writers (distributed image
  processing / stereomatching) keep the per-frame layout; `targets/` and
  `correspondences/` are unchanged.
- **Readers.** Use `RunStore.read_linkage` / `linkage_frames`, or for a raw zarr group
  `openptv2.storage.linkage_blocks.iter_linkage(group)` and `read_all_quality`.
  Don't list `linkage/<name>` keys yourself.

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

# Per-frame linkage (works for both the block and the legacy per-frame layout)
from openptv2.storage import RunStore
prev, nxt, pos = RunStore("res/run.zarr", mode="r").read_linkage(1)
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
uv run python C:/Users/alex/Downloads/aorta_phantom/wp1/copy_trajectories.py --include-traj --overwrite --verify
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
