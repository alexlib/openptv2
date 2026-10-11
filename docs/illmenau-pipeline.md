# Illmenau: from bubble images to merged barrel trajectories

The processing chain of the Illmenau barrel dataset (two rigs of four cameras,
helium-filled soap bubbles, 10 fps), after calibration
([Illmenau 4-camera case study](illmenau-4cam-calibration.md)). Each step is a
generic library function; the scripts in `scripts/illmenau/` only add the
Illmenau specifics (folders, camera groups, shards, barrel geometry).

| step | generic code | Illmenau script | settings (parameter file of the rig) |
|---|---|---|---|
| 1. bubble detection + correspondences | `openptv2-batch --mode sequence`, plugin `bubble_sequence` | — | `bubble_detection:`, `background:`, `criteria:` |
| 2. re-correspondence with the current calibration, from the stored detections | `openptv2.recorrespond` (`load_parameters`, `stored_frames`, `correspond_frame`, `write_store`) | `correspond_fulldiam.py` (one process per shard) | `ptv:`, `criteria:`, `cal_ori:` |
| 3. tracking | `openptv2.plugins.two_phase_tracking.track_store` (what the `two_phase` plugin runs) | `assemble_track.py` (merges the shards into `res/run_fulldiam.zarr`) | `track:` |
| 4. cleaning | `openptv2.storage.linkage_arrays.read_linkage_arrays`, `openptv2.reconnect.reconnect_pieces`, then `flowtracks.clean.clean_trajectories` (postptv) | `build_postptv_clean.py` | `postptv_clean:` (every value printed with its source) |
| 5. merging the two rigs | `flowtracks.merge` (postptv): rigid map from common bubbles, overlaps, duplicates | `merge_dual_rig.py` (rotations to try, barrel frame and walls, output layout) | command line (`--dup-mm`, `--keep-duplicates`) |
| 6. movie | `flowtracks.graphics.animate_trajectories_3d` (postptv) | `make_barrel_movie.py` (view, colour bar, barrel rims) | `plate.yaml` (barrel size) |

Nothing is set inside the scripts: the camera model, the tracker gates and
every cleaning value come from the rig's `parameters_Run1.yaml`.

## Running it

One rig, steps 1-4, with the runner kept in the dataset folder (it refuses to
overwrite existing results), then the merge:

```bash
bash claude_work_2026-10-10/pipeline/full_pipeline_tuned.sh 14
bash claude_work_2026-10-10/pipeline/full_pipeline_tuned.sh 58
uv run --with-editable /path/to/postptv python scripts/illmenau/merge_dual_rig.py \
    --clean run_fulldiam_clean.zarr --out merged_fulldiam.zarr
```

`--with-editable` supplies `flowtracks.clean` and `flowtracks.merge` until a
postptv release carries them.

## The two rigs

Each rig is calibrated in its own frame (the front and back faces of the
plate). The merge measures the map from rig 5-8 to rig 1-4 from bubbles both
see in the same frame: a 180° turn about the vertical and an offset of about
(−1.1, 0.6, 8.9) mm, from 12,364 bubble pairs. Rig 5-8 trajectories that share
same-frame points (closer than 20 mm) with an at least as long rig 1-4
trajectory are duplicates; the merged store keeps the longer copy, or with
`--keep-duplicates` keeps both and marks them in `track_info`.
