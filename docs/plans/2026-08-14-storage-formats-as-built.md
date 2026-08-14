# Storage formats as built (2026-08-14) — backward-compatibility reference

Snapshot of how openptv2 stores data **today**, captured before redesigning the
storage layer. Its purpose is backward compatibility: any new format must be
able to import from, and (on request) export to, everything described here.

Verified against `main` @ `74c9604`.

---

## Layer 1 — legacy per-frame ASCII (the canonical format; everything else is derived)

| File | Written by | Format |
|---|---|---|
| `img/cam<N>.<frame>_targets` | `algorithms/tracking_frame_buf.py:215` `write_targets` | count line, then `pnr x y n nx ny sumg tnr` |
| `res/rt_is.<frame>` | `gui/ptv.py:763` (`py_determination_proc_c`), `tracking_frame_buf.py:574` (`write_path_frame`) | count, then `nr X Y Z p0 p1 p2 p3` — mm; `p<i>` = target id in camera i, `-1` = not seen |
| `res/ptv_is.<frame>` | `tracking_frame_buf.py:569` | count, then `prev next X Y Z` — **frame-local** 0-based indices; `prev == -1`, `next == -2` mean no link |
| `res/added.<frame>` | same writer, different `linkage_file_base` | identical 5 columns; second tracking pass |
| prio file (optional) | `tracking_frame_buf.py:579` | ptv_is columns + `prio` |

Exact format strings (must be reproducible by any exporter):

```python
# targets
f"{t.pnr:4d} {t.x:9.4f} {t.y:9.4f} {t.n:5d} {t.nx:5d} {t.ny:5d} {t.sumg:5d} {t.tnr:5d}\n"
# rt_is
"%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n"
# ptv_is
f"{p.prev:4d} {p.next_idx:4d} {p.x[0]:10.3f} {p.x[1]:10.3f} {p.x[2]:10.3f}\n"
```

Filenames are positional-by-convention. `_resolve_file_base` (`:156`) supports
`%d` templates, a `{frame:04d}` fallback, and — on a read miss — a regex that
guesses the camera index back out of the filename (`:195`).

`Frame` / `FrameBuf` (`tracking_frame_buf.py:630`, `:936`) read and write these
directly; this is the format the Cython tracker actually sees.

## Layer 2 — Zarr mirror (`res/run.zarr`), gated by `OPENPTV_STORAGE`

`src/openptv2/storage/zarr_store.py`, class `ZarrFrameStore`.

| Mode | Behaviour |
|---|---|
| `zarr` (**default**) | dual-write: ASCII **and** Zarr |
| `zarr_only` | writes both, then `os.remove`s the text files (`write_path_frame:616`) |
| `legacy` | ASCII only |

Reads prefer ASCII; Zarr is a fallback when the file is missing or zero-length
(`read_targets:190`, `read_path_frame:360`).

```
run.zarr/
  targets/cam_<i>/frame_<n>              (N,8) float64      # unpadded frame key
  correspondences/frame_<n>              (N, 3+ncams) float64, xyz hstacked with cam ids cast to float
  linkage/<ptv_is|added>/frame_<n:05d>/  {prev:int32, next:int32, pos:float64}   # zero-padded key
  trajectories/                          {pos, vel, acc, frame, trajid}
  metadata/                              created empty, never written
```

## Layer 3 — HDF5 / flowtracks (export end)

- `ZarrFrameStore.to_flowtracks_h5()` (`:303`) — flat `pos/vel/acc/frame/trajid`, mm→m.
- `read_zarr_trajectories()` (`:538`) — builds flowtracks `Trajectory` objects.
  Case 1 reads `trajectories/`; Case 2 **walks the linkage graph** and is the
  path actually taken, because `trajectories/` is a stale cache written by
  openptv-cloud post-processing and not refreshed by a re-track (comment at `:560`).
- `gui/flowtracks_utils.py` falls back to `flowtracks.io.trajectories_ptvis("res/ptv_is.%d")`.
- The real trajectory database (`trajectories.h5`) is produced downstream by
  **postptv/flowtracks** via openptv-cloud's `post_process.py`, not here.

## Layer 4 — `res/images.zarr` (input images)

Read-only in this repo (`gui/ptv.py:825`); expects a `raw_images` array, shape
`(N,h,w)` or `(N,cams,h,w)`, 50-frame Blosc-Zstd chunks. The writer
(`convert_to_zarr.py`) lives in **openptv-cloud/docker** — it does not exist in
openptv2, despite `docs/zarr-hdf5-storage.md:173` implying otherwise.

## Pipeline graph

```
img/*.tif ──┬─→ (openptv-cloud) convert_to_zarr.py → res/images.zarr ─┐
            └──────────────────────────────────────────────────────┬─┘
                                                                   ↓
                          detection   → img/*_targets  +  targets/cam_i/frame_n
                          correspond  → res/rt_is.*    +  correspondences/frame_n
                          tracking    → res/ptv_is.*   +  linkage/ptv_is/frame_n
                                                                   ↓
                                   read_zarr_trajectories (walks linkage)
                                                                   ↓
                                   flowtracks Trajectory → trajectories.h5 (postptv)
                                                                   ↓
                                   matlab_to_python_3dptv → Eulerian grid → VTK
```

---

## Known defects in the as-built design

Recorded so the redesign does not reproduce them.

1. **Two sources of truth that diverge silently.** `write_path_frame:607` wraps
   the whole Zarr write in a bare `except: pass`. A run can produce a store with
   missing frames and report success.
2. **Inconsistent frame keys** — `frame_10000` for targets/correspondences,
   `frame_{n:05d}` for linkage. Six-digit frame numbers break the lexical sorts
   in `inspect_zarr_store` and `read_zarr_trajectories`.
3. **`inspect_zarr_store:474` reads `trajectories/time`; `write_trajectories:246`
   writes `frame`.** It raises. It also never reports the `linkage/` group — the
   one group holding the tracking result.
4. **`write_trajectories()` has no caller in this repo.** Only openptv-cloud
   writes `trajectories/`, which is why `read_zarr_trajectories` distrusts it.
5. **Integer target ids stored as float64** (`write_correspondences:188` hstack),
   forcing a cast on every read.
6. **One Zarr group per frame per camera** — the small-files problem, moved
   inside a directory store. `_get_or_create_group` needs a 10-attempt retry loop
   with sleeps to survive concurrent workers; that retry loop is the symptom.
7. **`Path("res/run.zarr")` is CWD-relative** (`gui/ptv.py:745`, the plugins).
   Only works because the pipeline chdirs into the experiment folder.
8. **mm↔m conversion scattered across call sites** (`zarr_store.py:314`, `:572`,
   `:637`; `flowtracks_utils.py:49`). No units are recorded anywhere.
9. **`prev`/`next` are frame-local indices**, so every consumer must re-derive
   "which frame does this index address". `read_zarr_trajectories:640-650` has
   three separate guard clauses for the ways that goes wrong (frame gap, row-count
   mismatch, multi-claim ambiguity).
10. **No run statistics are persisted.** `correspondences()` returns
    `match_counts = [quads, triplets, pairs, total]`
    (`algorithms/correspondences.py:901`); `gui/ptv.py:1381` and `:707` print it
    to stdout and drop it. Per-frame target counts, quad/triplet/pair counts,
    per-camera participation, and link counts are all recoverable only by
    re-reading the entire run.
