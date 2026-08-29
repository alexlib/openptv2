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
  preset: "full_multipass"
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
  selected_tracking: default  # default, two_phase, myptv_3d_tracking, ...
```

### Presets

| Preset | Passes | Recommended |
| :--- | :--- | :--- |
| `standard_forward` | Forward only | Fast preview |
| `full_multipass` | Forward→Backward→Postprocess | **Recommended** |

> `priority_segment_3d` was removed. Use `default` or `two_phase`.

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
| `selected_tracking` | str | `default` | `default` (trackcorr), `two_phase` (3D→2D Hungarian), `myptv_3d_tracking`, … |

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

## 4. Tracking Algorithms & Plugins

| Plugin | Description |
| :--- | :--- |
| `default` (trackcorr) | Cython 3 `track3d_loop_fast` — 3D box search, angle+acc |
| `two_phase` | **New** — Phase 1: 3D KD-tree candidates within `v_max`; Phase 2: per-camera 2D leaf mean distance → Hungarian assignment (`src/openptv2/plugins/two_phase_tracking.py`). `leaf_weight=0` ≡ 3D-only. **74% more multi-frame traj** on TT13 aorta (poorly-conditioned). |
| `myptv_3d_tracking` | MyPTV kinematic predictor |
| `cython_epipolar` etc. | Epipolar variants |

Select via GUI **Plugins** or `plugins.selected_tracking`. Custom plugins implement `BaseTrackingPlugin` (`docs/developer_guide/custom_tracking_plugins.md`).

---

## 5. Understanding Results in `res/run.zarr`

### Reading Linkage & Trajectories

```python
import zarr, numpy as np
root = zarr.open_group("res/run.zarr", mode="r")
# Linkage per frame
prev, nxt, pos = root["linkage/ptv_is/frame_000001/prev"][:], root["linkage/ptv_is/frame_000001/next"][:], root["linkage/ptv_is/frame_000001/pos"][:]
# Flat trajectories (sealed)
traj = root["traj"]; idx_tid, idx_len, idx_row = np.asarray(traj["trajid"]), np.asarray(traj["length"]), np.asarray(traj["first_row"])
# Top 100 longest
order = np.argsort(idx_len)[::-1][:100]
for tid, fr, ln in zip(idx_tid[order], idx_row[order], idx_len[order]):
    pts = np.asarray(root["trajectories/pos"][fr:fr+ln])  # [m]
```

### Copying to Dropbox

```bash
uv run python copy_trajectories.py --include-traj --overwrite  # creates trajectories.zarr + traj.zarr (137 MB vs 1637 MB)
```

See `C:\Users\alex\Downloads\TT13_aorta\wp1\copy_trajectories.py` and `docs/zarr-hdf5-storage.md`.

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

## 7. Interpreting Statistics

`seal` reports `n_trajectories`, `n_rows`, `n_dropped`. With `z-noise/motion≈19` (TT13 aorta) expect short fragments (median ~10 with `min_length=5`, median 1 without filter). For Eulerian `velocity` fields use `flowtracks`/`postptv`, not long Lagrangian `trajid`.

---

## 8. Developing Custom Plugins

See `docs/developer_guide/custom_tracking_plugins.md`.
