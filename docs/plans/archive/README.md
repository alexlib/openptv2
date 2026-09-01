# Archived Plans

These plans are **historical** — kept for reference but superseded by the zarr-only `v0.5.6` implementation (`docs/zarr-hdf5-storage.md`, `docs/tracking_guide.md`).

| Plan | Why archived |
| :--- | :--- |
| `2026-08-14-storage-formats-as-built` | Superseded by `docs/zarr-hdf5-storage.md` (run.zarr is sole store) |
| `2026-08-15-*`, `2026-08-16-*`, `2026-08-24-*`, `2026-08-25-*`, `2026-08-26-*` | Transition/parallelization/CI plans — implemented, see `docs/benchmarking_guide.md` |
| `2026-08-17-lagrangian-accuracy` | Early accuracy program — replaced by `tracking-as-joint-tree-forest` + `unified-particle-table` |
| `2026-08-27-track3d-vs-liboptv` / `track3d-vs-3dptv` | Reference comparisons — no phases to execute, kept as docs |
| `2026-08-27-quad-uniqueness-pass-study` | Study done 2026-08-27 — superseded by `verified-pipeline` Phase2 FEATURE verdict |
| `two-subrig-calibration` | Negative result on 4-frame test_cavity — plateau is data quantity, not parametrisation |
| `2026-09-01-zarr-only-final-cutover` | Follow-through of `2026-08-15`/`2026-08-26` — `res/run.zarr` sole DB, PR #36 + `5d44a0c4` |

Current active plans (in `docs/plans/`):
- `2026-08-27-unified-particle-table.md` — tree-forest particle table + KD-tree (research, conflicts with current RunStore schema)
- `2026-08-27-tracking-as-joint-tree-forest.md` — two_phase 3D→2D tracking (north star)
- `2026-08-27-track3d-beat-gt-plan.md` — `track3d` cascade vs GT (next: losers-retry + adaptive gate)
- `2026-08-27-backward-postprocess-double-claim-bug-plan.md` — 185 double-claims via postprocess
- `2026-08-27-verified-pipeline-ghost-particle-study-plan.md` — Phase1 rerun with highpass + Phase3 gated implementation
- `2026-08-27-eps0-dynamic-band-study-plan.md` — per-particle epipolar band (deferred)
- `2026-08-30-illmenau-dots-plate-pipeline.md` — 1–4 done via PR #36, §8 cams 5–8 remains
- `2026-08-30-calibration-hub-multi-source.md` — Doors A/B done via PR #36, Doors C/D + DLT/rig.yaml/eps remain
- `2026-08-31-16bit-image-handling.md` — 16→8 scaling (`to_uint8` + `targ_rec_scaled`)
- `differentiable_ptv_nextgen_plan.md` — nextgen differentiable PTV (long-term vision)
