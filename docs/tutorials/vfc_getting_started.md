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
  vfc_radius: 10.0              # mm, about 3x the distance between the tracked particles
  vfc_min_neighbors: 3
  vfc_gate: 0.5                 # mm/frame, 6 to 8x the noise (see 3.1)
  vfc_weight: 0.0
  vfc_post_k: 3.0
  vfc_post_min_tol: 0.5         # mm/frame, the same scale as the gate
  vfc_post_reject_cold_start: true
  # --- cold-start bootstrap ---
  vfc_bootstrap: true
  vfc_bootstrap_passes: 2
  vfc_bootstrap_shift_ratio: 0.5
  vfc_bootstrap_grid_size: 20.0 # mm
  vfc_bootstrap_max_neighbors: 32
  # --- affine consensus ---
  vfc_affine: true
  vfc_affine_min_neighbors: 6
```

### 3.1 How to choose each value

| Key | Default | How to choose it |
|---|---|---|
| `vfc_mode` | `off` | `post` at the start. It is the safe option. Then try `both`. |
| `vfc_radius` | 10.0 mm | About 3 times the distance between the **tracked** particles. The correspondences are much further apart (they hold the ghost points), so use the tracked particles here. |
| `vfc_min_neighbors` | 3 | 3 or 4. More in sparse data. |
| `vfc_gate` | `None` | **6 to 8 times** the noise (see the warning below). `None` switches the gate off. |
| `vfc_weight` | 0.0 | Leave at 0. The soft penalty had almost no effect in the tests. |
| `vfc_post_k` | 3.0 | Leave at 3. |
| `vfc_post_min_tol` | 0.5 mm/frame | The same scale as the gate. |
| `vfc_post_reject_cold_start` | true | Leave on. It stops a removed ghost from starting a track. |
| `vfc_bootstrap` | false | On, if the data has ghosts. The tracker then runs twice. |
| `vfc_bootstrap_passes` | 2 | Leave at 2. The value 1 uses the old, weaker estimate. |
| `vfc_bootstrap_shift_ratio` | 0.5 | Leave at 0.5. The tracker uses this to switch the bootstrap off by itself. |
| `vfc_bootstrap_grid_size` | 20.0 mm | Only for `vfc_bootstrap_passes: 1`. Leave at 20. |
| `vfc_bootstrap_max_neighbors` | 32 | Leave at 32. |
| `vfc_affine` | false | On, if the flow has a shear or a rotation. Use it with `vfc_bootstrap`. |
| `vfc_affine_min_neighbors` | 6 | 6, or 4 in sparse data. |

> **Warning: the gate prunes candidates before the assignment.** When the gate
> removes the true candidate, the solver picks another one, and that one is
> often worse than the one it removed. A tight gate therefore does not only
> remove the bad links, it also replaces the good ones by worse ones, and the
> trajectories get shorter. Set the gate **well above the noise** (6 to 8
> times), so that it prunes the wild candidates only.
>
> Measured on a real recording (5005 frames, the noise 0.063 mm/frame):
>
> | Gate | Points | Mean track length | Tracks over 100 frames | Wrong links removed |
> |---|---|---|---|---|
> | off (no VFC) | 4 859 718 | 32.7 | 9 659 | - |
> | 0.2 mm/frame (3 x the noise) | 4 782 355 | 31.3 | 9 151 | 23 % of the marginal links |
> | **0.4 mm/frame (6 x the noise)** | 4 856 759 | **32.8** | **9 656** | the wild links |
>
> With the gate at 3 times the noise the mean track length falls by 4 % and the
> long tracks by 5 %. With the gate at 6 times the noise nothing is lost and the
> wrong links are still removed.

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
| Fewer points **and** fewer tracks | Bad. The tolerance is too tight, and the gate has replaced the good links by worse ones. Increase `vfc_gate` and `vfc_post_min_tol`, or set `vfc_mode: post`. |
| The same number of tracks and the same points | The consensus did nothing. Check `vfc_min_neighbors` and `vfc_radius`. The data is too sparse. |
| The run is much slower | The bootstrap runs the tracker twice. Check the time before you keep it. |

The mean track length alone is not a quality measure: a wrong link makes a
track *longer*. Look at the number of tracks over 30, 100 and 300 frames, and at
the acceleration kurtosis, together with the number of points.

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

### 5.1 Through the Tracking Parameters dialog

The **Tracking Parameters** dialog (`src/openptv2/gui/parameter_gui.py`) has a
**Spatial Consensus (VFC — Two-Phase Tracker)** group:

1. Launch the GUI (`uv run openptv2-gui`) and open your experiment folder.
2. Open **Parameters** → **Tracking Parameters**.
3. Under **Pipeline**, set **Tracker** to **Two-Phase (3D search + 2D rank)**.
4. Under **Spatial Consensus (VFC — Two-Phase Tracker)**, set:
   - **VFC mode**: `off`, `pre`, `post`, or `both`
   - **Radius (mm)**: neighbourhood size (e.g. `10.0` or `12.0`)
   - **Min neighbors**: minimum neighbours for a consensus (default `3`)
   - **Pre-gate (mm/f, 0=off)**: Option A hard gate (`0.0` saves `None` = off;
     set to `6–8x` the position noise, e.g. `0.4`)
   - **Post min tol (mm/f)**: Option B minimum tolerance (e.g. `0.4`)
   - **Cold-start bootstrap**: switch on the two-pass frame-0 velocity field
   - **Affine consensus (tr(A)=0)**: fit the local incompressible velocity gradient
5. Click **OK** (saves to the active YAML) and press **Track**.

### 5.2 Advanced keys in the YAML

Advanced keys (`vfc_weight`, `vfc_post_k`, `vfc_post_reject_cold_start`,
`vfc_bootstrap_passes`, `vfc_bootstrap_shift_ratio`,
`vfc_bootstrap_max_neighbors`, `vfc_affine_min_neighbors`) can be set in the
`track:` section of the experiment YAML. The GUI dialog preserves any extra
keys in `track:` when you click **OK**.

## 6. Problems and what to do

| Problem | Cause | What to do |
|---|---|---|
| The tracker finds fewer tracks than before | The tolerance is too tight, and the gate has replaced the good links by worse ones | Increase `vfc_post_min_tol` and `vfc_gate`. Use `vfc_mode: post` only. |
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
