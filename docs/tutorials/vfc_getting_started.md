# Getting Started with the Spatial Consensus (VFC)

This guide shows how to switch on the spatial consensus in the `two_phase`
tracker, how to compare it with the tracker without the consensus on the same
data, and how to use it from the GUI.

The method is described in [Spatial Consensus and the Cold-Start
Bootstrap](../two-phase-spatial-consensus.md). This guide only shows the
practice.

This document uses the ASD-STE100 style (Simplified Technical English).

## 1. What you get

The consensus compares each link of the tracker with the links of the
neighbouring particles. The tracker then removes the links that do not fit the
neighbourhood. Most of these links go to ghost points.

The result on synthetic data with ghosts (800 real particles, 6 frames, the
measured output of `scripts/bench_two_phase_vfc.py`):

| Ghost ratio | Variant | Precision | Recall | False links | Time |
|---|---|---|---|---|---|
| 0 % | off | 100.0 % | 100.0 % | 0 | 2.6 ms/frame |
| 0 % | **on** | **100.0 %** | **100.0 %** | **0** | 44.9 ms/frame |
| 50 % | off | 91.5 % | 99.2 % | 367 | 4.5 ms/frame |
| 50 % | **on** | **99.6 %** | **100.0 %** | **18** | 65.1 ms/frame |
| 100 % | off | 75.7 % | 97.3 % | 1249 | 6.4 ms/frame |
| 100 % | **on** | **98.2 %** | **100.0 %** | **72** | 87.9 ms/frame |
| 200 % | off | 49.1 % | 92.8 % | 3850 | 10.3 ms/frame |
| 200 % | **on** | **91.6 %** | **99.8 %** | **364** | 136.2 ms/frame |

The "on" row uses the gate, the filter and the bootstrap. The number of false
links goes down by a factor of 10 to 20. The tracker is much slower, because
the bootstrap runs the tracker twice and the gate checks every candidate.

The affine fit (`vfc_affine: true`) lowers the number of false links a little
more. In the same test it gives 14 false links at a ghost ratio of 50 %, 48 at
100 % and 131 at 200 %.

## 2. Step 1: measure your data

The consensus needs three numbers from your data.

| Number | Meaning | How to get it |
|---|---|---|
| Spacing | The median distance from a particle to its nearest neighbour | Estimate it from the density that `scripts/tracking_advice.py` prints (the number of neighbours within 5 mm) |
| Step | The median length of the links of a good run | `scripts/tracking_advice.py` prints "median step per frame" |
| Noise | The change of the step from frame to frame | `scripts/tracking_advice.py` prints "median kink" |

```bash
uv run python scripts/tracking_advice.py PATH_TO_RUN_FOLDER
```

Use these numbers for the values of section 3. Do not copy the values of
another experiment.

## 3. Step 2: add the keys to the YAML

The tracker reads the keys from the `track:` section of the experiment YAML.
All features are off by default. Add the block below to the `track:` section.

```yaml
plugins:
  selected_tracking: two_phase

track:
  # --- spatial consensus ---
  vfc_mode: both                # off | pre | post | both
  vfc_radius: 10.0              # mm
  vfc_min_neighbors: 3
  vfc_gate: 0.5                 # mm/frame
  vfc_weight: 0.0
  vfc_post_k: 3.0
  vfc_post_min_tol: 0.5         # mm/frame
  vfc_post_reject_cold_start: true
  # --- cold-start bootstrap ---
  vfc_bootstrap: true
  vfc_bootstrap_passes: 2
  vfc_bootstrap_shift_ratio: 0.5
  vfc_bootstrap_grid_size: 20.0 # mm
  # --- affine consensus ---
  vfc_affine: true
  vfc_affine_min_neighbors: 6
```

### 3.1 How to choose each value

| Key | Default | How to choose it |
|---|---|---|
| `vfc_mode` | `off` | `post` at the start. It is the safe option. Then try `both`. |
| `vfc_radius` | 10.0 mm | About 3 times the spacing. Large enough for neighbours, small enough for one local flow. |
| `vfc_min_neighbors` | 3 | 3 or 4. More in sparse data. |
| `vfc_gate` | `None` | 2 to 3 times the noise. `None` switches the gate off. |
| `vfc_weight` | 0.0 | Leave at 0. The soft penalty had almost no effect in the tests. |
| `vfc_post_k` | 3.0 | Leave at 3. |
| `vfc_post_min_tol` | 0.5 mm/frame | 2 to 3 times the noise. |
| `vfc_post_reject_cold_start` | true | Leave on. It stops a removed ghost from starting a track. |
| `vfc_bootstrap` | false | On, if the data has ghosts. The tracker then runs twice. |
| `vfc_bootstrap_passes` | 2 | Leave at 2. The value 1 uses the old, weaker estimate. |
| `vfc_bootstrap_shift_ratio` | 0.5 | Leave at 0.5. The tracker uses this to switch the bootstrap off by itself. |
| `vfc_bootstrap_grid_size` | 20.0 mm | Only for `vfc_bootstrap_passes: 1`. Leave at 20. |
| `vfc_affine` | false | On, if the flow has a shear or a rotation. Use it with `vfc_bootstrap`. |
| `vfc_affine_min_neighbors` | 6 | 6, or 4 in sparse data. |

### 3.2 A safe first run

Use this block for the first test. It is the safest combination and it is
close to the values of the table in section 1.

```yaml
track:
  vfc_mode: post
  vfc_radius: 10.0
  vfc_min_neighbors: 3
  vfc_post_k: 3.0
  vfc_post_min_tol: 0.5
```

Run the tracker and compare the result with the run without the consensus
(section 4). Then add the gate and the bootstrap.

## 4. Step 3: compare the consensus off and on

Both runs must use the same data. Use the same YAML for everything except the
`vfc_*` keys.

### 4.1 Make two YAML files

```bash
cd PATH_TO_EXPERIMENT
cp parameters_Run1.yaml parameters_vfc_off.yaml
cp parameters_Run1.yaml parameters_vfc_on.yaml
```

Add the block of section 3 to `parameters_vfc_on.yaml`. Leave
`parameters_vfc_off.yaml` unchanged.

### 4.2 Run the tracker twice

The `--output` option keeps the result of `res/run.zarr` and writes the new
result to `res/<name>.zarr`.

```bash
uv run openptv2-batch parameters_vfc_off.yaml 10000 10004 \
    --mode tracking --tracking-plugin two_phase --output vfc_off.zarr

uv run openptv2-batch parameters_vfc_on.yaml 10000 10004 \
    --mode tracking --tracking-plugin two_phase --output vfc_on.zarr
```

The frame range must be the same in both runs. The `--mode tracking` mode
uses the correspondences and the targets of the store, so the detection step
does not run again.

### 4.3 Compare the two runs

```bash
uv run python scripts/compare_vfc_runs.py \
    off=res/vfc_off.zarr on=res/vfc_on.zarr
```

The script prints one row per run and the change:

```text
Run                Tracks    Points  Mean len  Median len  Max len    Long  Points/frame
----------------------------------------------------------------------------------------
off                  320      9600    30.00        28.0      142      180         96.0
on                   295      9450    32.03        30.0      151      205         94.5

off -> on:
  points                    9600.0 ->      9450.0    -1.6 %
  mean track length           30.0 ->        32.0    +6.8 %  (better)
  long tracks                 180.0 ->        205.0   +13.9 %  (better)
  points per frame            96.0 ->         94.5    -1.6 %
```

(The numbers above are an example. Use the numbers of your own data.)

### 4.4 What to look at

| Result | Meaning |
|---|---|
| More long tracks, same number of points | Good. The tracker removed the wrong links and kept the true ones. |
| Fewer points and fewer tracks | Bad. The tolerance is too tight. Increase `vfc_gate` and `vfc_post_min_tol`, or set `vfc_mode: post`. |
| The same number of tracks and the same points | The consensus did nothing. Check `vfc_min_neighbors` and `vfc_radius`. The data is too sparse. |
| The run is much slower | The bootstrap runs the tracker twice. Check the time before you keep it. |

### 4.5 Compare on synthetic data first

The script `scripts/bench_two_phase_vfc.py` makes the same comparison on
synthetic data with ghosts. Use it before you touch the real data.

```bash
uv run python scripts/bench_two_phase_vfc.py --n-particles 800 --n-frames 6
```

The script prints the precision, the recall, the number of false links and
the time for every combination of the options. Use it to find the values that
fit your noise level.

## 5. Step 4: use the consensus from the GUI

### 5.1 What works today

The GUI has no fields for the `vfc_*` keys. The GUI still uses them.

- The GUI reads the `track:` section of the experiment YAML.
- The tracking panel of the GUI only writes its own keys. It does not remove
  the other keys (`src/openptv2/gui/parameter_gui.py:403`).
- The `two_phase` plugin reads the `track:` section when it starts
  (`src/openptv2/plugins/two_phase_tracking.py:1954`).

So this works:

1. Add the block of section 3 to the `track:` section of the experiment YAML.
2. Open the folder in the GUI.
3. On the Plugins page, select the `two_phase` tracker.
4. Press Track.

The GUI uses the values of the YAML. The Tracking Parameters panel does not
show them.

### 5.2 How to add the fields to the GUI

This is a task for a developer. Four places must change.

1. **The model.** Add the keys with their defaults to the `TrackParams` class
   in `src/openptv2/gui/parameter_models.py`.

   ```python
   class TrackParams(SectionModel):
       ...
       vfc_mode: str = "off"
       vfc_radius: float = 10.0
       vfc_min_neighbors: int = 3
       vfc_gate: float | None = None
       vfc_post_k: float = 3.0
       vfc_post_min_tol: float = 0.5
       vfc_bootstrap: bool = False
       vfc_affine: bool = False
   ```

2. **The panel.** Add the traits to the `Tracking_Params` class in
   `src/openptv2/gui/parameter_gui.py`. Read the values from the section in
   `__init__`, in the same way as `leaf_weight`.

   ```python
   self.vfc_mode = str(tracking_params.get("vfc_mode", "off"))
   self.vfc_radius = float(tracking_params.get("vfc_radius", 10.0))
   ```

3. **The layout.** Add the items to `Tracking_Params_View` in the same file.
   Use an `EnumEditor` for `vfc_mode` with the values `off`, `pre`, `post`
   and `both`.

4. **The save.** Add the new keys to the dictionary that
   `TrackHandler.closed` passes to `apply_tracker`
   (`src/openptv2/gui/parameter_gui.py:386`). The handler merges this
   dictionary into the `track:` section, so the new keys reach the YAML.

5. **The test.** Add a test in `tests/gui/`. The tests of the tracking panel
   are in `tests/gui/test_tracking_parameters.py`.

Keep the panel simple. The most useful fields are `vfc_mode`, `vfc_radius`,
`vfc_gate` and `vfc_bootstrap`. The other keys can stay in the YAML.

## 6. Problems and what to do

| Problem | Cause | What to do |
|---|---|---|
| The tracker finds fewer tracks than before | The tolerance is too tight | Increase `vfc_post_min_tol` and `vfc_gate`. Use `vfc_mode: post` only. |
| The consensus does nothing | Too few neighbours | Increase `vfc_radius` or decrease `vfc_min_neighbors`. Check the seeding density. |
| The bootstrap switches itself off | The particles move too far per frame | The data is too sparse or too fast. The tracker cannot work on this data. |
| The affine fit fails | Too few neighbours for the fit | Decrease `vfc_affine_min_neighbors` to 4, or increase `vfc_radius`. |
| The run is twice as slow | The bootstrap runs the tracker twice | Keep the bootstrap only if the data has ghosts. |
| The GUI shows no new fields | The fields do not exist yet | Use the YAML (section 5.1), or add the fields (section 5.2). |

## 7. Limits

- The features help only when the flow is slow enough compared to the
  distance between the particles. If the particles move farther per frame than
  their distance from each other, every tracker fails. Check this first.
- The consensus needs a dense enough seeding. In sparse data the tracker has
  no neighbours, and the features do nothing.
- The bootstrap doubles the tracking time.
- The two-pass field uses the tracks of the first pass. It describes the flow
  of the later frames. It is wrong for a flow that changes quickly in time.
- The affine fit assumes a flow with constant density. A compressible flow
  gives a small bias.

## 8. Where to go next

- The method: [Spatial Consensus and the Cold-Start Bootstrap](../two-phase-spatial-consensus.md)
- The tracker guide: [The two_phase tracker](../two-phase-tracking.md)
- The parameters of all trackers: [Tracking parameters](../tracking_parameters_guide.md)
- The measure tool: `scripts/tracking_advice.py`
- The synthetic comparison: `scripts/bench_two_phase_vfc.py`
- The run comparison: `scripts/compare_vfc_runs.py`
- The picture of the ghost removal: `scripts/visualize_vfc_ghosts.py`
