# Dense seeding: particles that touch in the images

**You have:** images with so many particles that some of them overlap in the cameras and are detected as one blob.
**You want:** more particles and longer tracks than the normal tracker gives.
**Time:** 10 minutes. **Tested:** every command below runs in CI on `test_data/test_cavity` (five frames).

Read [course page 4](../course/04-detect.md) (detection) and [page 6](../course/06-track.md) (tracking) first.

## 1. The problem, in plain words

If two particles are so close in one camera that they make one bright spot, the program sees one blob with a shifted centre.
The matcher gives the blob to one particle. That particle gets a wrong position, and its partner is lost. In our benchmark,
about 27 % of all blob observations were of this kind (about half of that in a real recording).

The *merged-blob loop* uses the tracks to repair this. Per round: it links, finds the blobs that must hold several particles
(a blob brighter than the track's particle is usually), gives each particle its own share of the blob, and triangulates again.
Details: [Holistic tracking](../holistic-tracking.md).

## 2. Prepare a run and turn the loop on

The loop is an option of the `two_phase` tracker. It is off by default.

```bash
cp -r test_data/test_cavity dense_run
```

```bash
openptv2-batch --workdir dense_run --first 10000 --last 10004 --mode sequence
```

Two lines in the YAML file switch it on. This command makes them and prints the result:

```bash
python -c "import re; p='dense_run/parameters_Run1.yaml'; s=open(p).read().replace('selected_tracking: default','selected_tracking: two_phase').replace('\ntrack:\n','\ntrack:\n  holistic_rounds: 2\n',1); open(p,'w').write(s); print([l.strip() for l in s.splitlines() if 'selected_tracking' in l or 'holistic_rounds' in l])"
```

```text
['selected_tracking: two_phase', 'holistic_rounds: 2']
```

(In the file you can edit by hand: `plugins: selected_tracking: two_phase` and `track: holistic_rounds: 2`. In the graphical
interface: Tracking Parameters, *Tracker* and *Merged-blob loop rounds*.)

## 3. Track

```bash
openptv2-batch --workdir dense_run --first 10000 --last 10004 --mode tracking
```

```text
Average over sequence, particles: 1665.2, links: 357.8, lost: 1307.5
Average over sequence, particles: 2020.2, links: 378.2, lost: 1642.0
Average over sequence, particles: 2180.0, links: 394.5, lost: 1785.5
Average over sequence, particles: 1713.8, links: 376.8, lost: 1337.0
Sealed: {'n_trajectories': 170, 'n_rows': 850, 'n_dropped': 0}
Batch processing completed successfully
```

One line per pass. The particle count per frame rises (2020, 2180) while the loop adds particles that were hidden in other
particles' blobs. At the end the added particles that no track took are removed (1714 left).

## 4. What you should see

| | plain tracker | with the loop |
|---|---|---|
| trajectories (5 frames) | 153 | **170** |
| points in trajectories | 765 | **850** |

About 11 % more points in tracks. This is a short example: on a longer, denser recording the measured gain was larger in
track length (real recording, 300 frames: mean track length 20.7 → 28.2, jumps 1.44 % → 0.47 %; [measurements](../holistic-tracking.md)).

## 5. Check that it is real, not invented

More tracks are only good if the new particles are real. Two checks:

* **Look at the smoothness.** Page 7 of the course: the new tracks should not be rougher than the old ones.
* **Leave-one-camera-out:** project each reconstructed particle into a camera that did not see it, and measure the miss
  (`openptv2.loco`, [Holistic tracking, limits](../holistic-tracking.md)). A real particle lands on a blob.

## If it did not work

| You see | Cause | What to do |
|---|---|---|
| no gain | few overlaps in your images (sparse seeding) | keep `holistic_rounds: 0`; it costs about 0.6 s per frame per round |
| very dense data, small gain | at four times our benchmark density the yield stays far below the ceiling | lower the seeding, or accept it |
| the first tracks were poor | the loop starts from the `two_phase` tracks | fix detection and calibration first (pages 4, 8) |

## Next

[Holistic tracking: all keys and measurements](../holistic-tracking.md), [Particle trackers](../trackers.md).
