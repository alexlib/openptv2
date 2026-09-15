# Zarr Performance Plan — end-to-end pipeline (openptv2 / openptv-cloud / postptv)

Date: 2026-09-15
Source guide: https://zarr.readthedocs.io/en/v3.1.0/user-guide/performance.html
Scope: zarr across the full pipeline **except images** (`images.zarr` transport excluded, used here only as reference).
Status: audit + phased plan. Not yet implemented.

## 1. Context / problem

`res/run.zarr` is the database of record (per-frame `targets/`, `correspondences/`, `linkage/`, sealed `traj/` + `trajectories/`, `stats/`, `meta/`), produced by openptv2 `RunStore`, merged by openptv-cloud `merge.py`, converted by `openptv-cloud/post.py`, and read by postptv `flowtracks/io.py`, `zarr_scene.py`, `eulerian.py`.

Observed: performance issues. Audit finding: **exactly one place follows the perf guide** (`openptv-cloud/images.py` — explicit chunks + `BloscCodec(zstd,3)`). Everything else takes zarr defaults:

- All per-frame writes (`run_store.py`, `zarr_store.py`, `io.save_zarr_trajectories`) call `create_array(data=...)` with no `chunks`, `shards`, `compressors`, `write_empty_chunks`, or `order`.
- All large flat-table writes (`post.py:save_trajectories_zarr`, `phase_io.py`) use `chunks=arr.shape` (single chunk).
- All large flat-table readers (`RunStore.trajectories()`, `post.read_zarr_trajectories`, `ZarrScene.__init__`, `phase_io.read_samples`) do `np.asarray(whole array)` then mask in memory.

Consequence A (too many tiny files): each per-frame array is a few KB (N≈10²–10³ × 3–8 cols), far below the guide's ≥1 MB chunk guideline, yet costs 1+ data files + metadata. A 4-cam × 5k-frame run ≈ 20k targets + 5k correspondences + ~15k linkage arrays ≈ 40–50k objects. Metadata ops dominate; consistent with the measured `~1.5s/op over gcsfs` (`openptv-cloud/entrypoint.py:31`, `merge.py:1-23`) and the `zarr.json` race retries (`run_store.py:35-50,148-154`).
Consequence B (too few huge chunks): the sealed `trajectories/` table is one giant chunk → one giant decompress, memory spike, no partial `first/last` reads, hostile to dask/xarray.

## 2. Guide section → current state → decision

### 2.1 Chunk size/shape (≥1 MB uncompressed; shape matches access)

- Per-frame arrays: default `chunks='auto'` → single tiny chunk each. **Decision: make it explicit** — `chunks=shape`, no compressor (or fast lz4), see Phase 1. Keeps lock-free per-frame parallel writes; removes auto-guess + wasted compression of KBs.
- Sealed `trajectories/{pos,vel,accel,time,trajid}`: `post.py:417` forces single chunk; `run_store.py:559-581` relies on auto. Readers always full-load. **Decision: row-chunk at 1–4 MB**, e.g. `(65536,3)` float64 ≈ 1.5 MB, and change readers to slice (`oindex`/block loop) instead of load-then-mask. See Phase 2.
- `phase_io.py:59,77` tiny `(sum,sumsq,count)` arrays with `chunks=shape`: fine as-is (always read whole, few KB). Keep explicit.
- `eulerian.py:813-816` `ds.to_zarr(mode="w")`: defaults. Grids are small/dense, usually read whole for ParaView. **Decision: pass explicit `chunks` + zstd encoding** (Phase 4).
- `images.py:56-62` `chunks=(50,H,W)`: ≈50 MB/chunk at 1 MP uint8, and `unpack_range():133-146` reads one row per image → 50× read amplification. **Decision: Phase 3.**

### 2.2 Sharding (many chunks → few files)

Unused anywhere. Natural target: sealed flat `trajectories/` table — `shard ≈ 32–128 MB`, `chunk ≈ 1–4 MB` → independent row reads + efficient writes. This is the long-term fix for the file-count problem on the **read path**.

Explicit non-goal: do NOT shard per-frame `targets/cam_X/frame_N` groups. A shard is the write unit; concurrent workers writing disjoint frames into one shard needs coordination and breaks the lock-free design (`run_store.py:16-21`). Keep per-frame layout for writes, seal into a sharded flat table for reads. Note: `merge.py:38-78` file-copy merge assumes one-file-per-frame and must become a Zarr-level copy once sharding lands. Requires `zarr>=3` everywhere (see §3).

### 2.3 Chunk memory layout (`order` C/F)

Default everywhere; correctly C-order for `(N,3)`/`(N,8)` particle-major tables. No change. Optional one-line experiment (C vs F on `pos` with Zstd) to close out, then leave as C.

### 2.4 Empty chunks (`write_empty_chunks=False`)

Default (`True`) everywhere. Zero-particle frames exist (`run_store.py:359-367` handles flat `(0,)`). **Decision: `write_empty_chunks=False` on per-frame `targets/correspondences/linkage` arrays** — empty frames store nothing, read back as fill; fewer objects, faster reads, negligible write-side fill check on KB arrays. Keep `True` for dense sealed tables (guide: check is pure overhead when chunks are almost always non-empty).

### 2.5 Parallel / Pickle / Blosc (guide stubs — "Coming soon")

- `post.py:115` already reads linkage frames via `ThreadPoolExecutor(16)` (I/O+decompress bound) — correct, keep. Extend the pattern to `seal.py:compute_source_hash` / `seal()` frame walk (`seal.py:63-68,111-112`), currently serial.
- `RunStore` holds a `threading.RLock` + live zarr group (`run_store.py:155,202-223`) — not picklable; `MemoryStore` isn't either per guide. Workers must keep reopening by path (current behavior, e.g. `tracking_chunked.py:452`) — do not pass store objects to `ProcessPoolExecutor`.
- Compressor hygiene: v3 default is `Zstd(level=0)` ≈ no compression, except `images.py` (`zstd-3`). Centralize: Zstd-3/5 + shuffle for float `pos/vel`, none/fast for int `prev/next/trajid`, keep Blosc-Zstd for images.

## 3. Preconditions / risks

1. **Pin `zarr>=3` everywhere** before sharding. Today `openptv2/pyproject.toml:54` allows `zarr>=2.16` (no `shards=` kwarg, different `create_array` API); postptv already requires `>=3.0`. Unify on v3 first.
2. **Format migration**: chunking/compressor changes alter `zarr.json` layout. Old stores stay readable (chunking is per-array metadata), but `merge.py` file-copy only works between identically-structured chunks — version-gate or re-seal path needed during rollout.
3. **Benchmark before/after**: add a `pytest-benchmark` (postptv convention, `SPEEDUP_PLAN.md`) or timing harness measuring: seal time, `to_flowtracks_trajectories` time, `post.convert` time, file count (`*.zarr` objects), store bytes. Use a mid-size fixture (e.g. 500–1000 frames), not just unit fixtures.

## 4. Phased implementation

### Phase 0 — pin + harness (prereq)
- Pin `zarr>=3,<4` in `openptv2/pyproject.toml`, `openptv-cloud/pyproject.toml`, `postptv/pyproject.toml`; CI matrix on zarr v3 only.
- Add bench: N files under `run.zarr`, wall time for `seal()`, `read_zarr_linkage_arrays()`, `ZarrScene` load, `post.convert()`.
- Files: 3× `pyproject.toml`, new `benchmarks/test_zarr_layout.py` (postptv) or equivalent.

### Phase 1 — per-frame writes: explicit single-chunk, no compress, skip empties (1 day, no format break for readers)
- `openptv2/storage/run_store.py`: `write_targets`, `write_correspondences`, `write_linkage` (prev/next/pos/prio), `write_stats`, `write_mmlut`, `write_unified_table` → `chunks=shape`, `compressors=None` (or fast codec), `write_empty_chunks=False`.
- Same for legacy `zarr_store.py` if still written in the tested path.
- Acceptance: identical arrays back (`np.array_equal`), fewer stored objects on sparse sequences, faster write microbench.

### Phase 2 — sealed `trajectories/` table: row-chunk + slice readers (the end-to-end win)
- Writers (`run_store.write_trajectories`, `openptv-cloud/post.save_trajectories_zarr`, `flowtracks/io.save_zarr_trajectories`): chunk rows at 1–4 MB (e.g. `pos/vel/accel: (65536,3)`, `time/trajid: (65536,)`), `Zstd(level=3, shuffle)` for floats, none/fast for ints.
- Readers (`RunStore.trajectory/trajectories/to_flowtracks_trajectories`, `post.read_zarr_trajectories`, `ZarrScene.__init__`, `phase_io.read_samples`, `zarr_store.read_zarr_trajectories`): replace `np.asarray(whole)` + boolean mask with range slices / `oindex` / block iteration honoring `first/last`.
- Acceptance: `first/last` sub-read touches O(range) chunks; full-load time ≤ before; peak RAM down on large stores.

### Phase 3 — `images.zarr` chunk rows vs. selective read
- `openptv-cloud/images.py`: reduce `CHUNK_FRAMES` to 1–10, OR keep 50 for pack throughput and batch `unpack_range` reads per chunk instead of per row.
- Acceptance: chunk-worker unpack reads ≈1 chunk perOWNED range, not 50× amplification.

### Phase 4 — Eulerian + phase-binned outputs
- `postptv/eulerian.save_zarr`: explicit `chunks` + zstd encoding + consolidated metadata for cloud reads.
- `phase_io.write_binned/write_averaged`: keep single-chunk (tiny, always whole-read); add comment stating intent.
- Acceptance: `export.run_stage4` output identical values, documented chunking.

### Phase 5 — shard the sealed table (deferred, needs Phase 0–2 + merge rework)
- `trajectories/` arrays: `shards ≈ 32–128 MB` over `chunks ≈ 1–4 MB`.
- Rewrite `merge.merge_chunks` as Zarr-level copy (file-copy breaks under sharding); re-test `tracking-shard` stitch path.
- Acceptance: object count collapses ~10–50× on large runs with no read regression.

## 5. File index (where to cut)

| Area | Files |
|---|---|
| Per-frame writes | `openptv2/src/openptv2/storage/run_store.py:285-414,493-526,559-581,681-706,734-767`, `openptv2/src/openptv2/storage/zarr_store.py:64-301` |
| Seal (serial walk → parallelize) | `openptv2/src/openptv2/storage/seal.py:53-112` |
| Cloud merge (file-copy assumption) | `openptv-cloud/src/openptv_cloud/merge.py:38-78`, `entrypoint.py:199-240` |
| Cloud convert (good read pattern + single-chunk write) | `openptv-cloud/src/openptv_cloud/post.py:64-154,212-266,340-420` |
| Phase outputs (single-chunk, fine) | `openptv-cloud/src/openptv_cloud/phase_io.py:53-78` |
| Images (only guide-conformant code; fix amplification) | `openptv-cloud/src/openptv_cloud/images.py:42-82,118-149` |
| Flowtracks Zarr I/O | `postptv/flowtracks/io.py:957-1240`, `postptv/flowtracks/zarr_scene.py:47-68`, `postptv/flowtracks/eulerian.py:813-826` |

## 6. What was deliberately NOT adopted

- Sharding per-frame groups (breaks lock-free parallel writes).
- F-order chunk layout (wrong for particle-major access).
- `write_empty_chunks=False` on dense sealed tables (pure overhead per guide).
- Passing live store objects to process pools (not picklable; reopen by path).
