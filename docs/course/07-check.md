# 7. Is the result good?

**Time:** 15 minutes. **You need:** the results of pages 4 to 6.

## What this step does

The program finishes without an error even when the result is wrong. This page gives you four checks that you can do on every
run, with numbers. Each check looks at one step of the chain.

One thing to know before you read the code: matched points are stored in millimetres, but trajectories are stored in
**metres**. The code multiplies the trajectory positions by 1000 to get millimetres.

## Do it

```bash
python -c "
import numpy as np
from openptv2.storage.run_store import RunStore

store = RunStore('my_run/res/run.zarr', mode='r')
frames = range(10000, 10005)

det = [[len(store.read_targets(cam, f)) for cam in range(4)] for f in frames]
print('1. targets per image: smallest', np.min(det), ' largest', np.max(det))

n4 = n3 = 0
for f in frames:
    pos, cams = store.read_correspondences(f)
    seen = (cams >= 0).sum(axis=1)
    n4 += int((seen == 4).sum())
    n3 += int((seen == 3).sum())
print(f'2. matched points in 5 frames: {n4} seen by 4 cameras, {n3} by 3 cameras')

tr = store.trajectories()
ids = np.unique(tr['trajid'])
lengths = [int((tr['trajid'] == i).sum()) for i in ids]
print(f'3. trajectories: {len(ids)}, length from {min(lengths)} to {max(lengths)} frames')

steps, bends = [], []
for i in ids:
    m = tr['trajid'] == i
    p = 1000 * tr['pos'][m][np.argsort(tr['time'][m])]
    d = np.diff(p, axis=0)
    steps.append(np.linalg.norm(d, axis=1))
    bends.append(np.linalg.norm(np.diff(d, axis=0), axis=1))
step, bend = np.median(np.concatenate(steps)), np.median(np.concatenate(bends))
print(f'4. smoothness: median step {step:.1f} mm, median change of the step {bend:.1f} mm, ratio {bend / step:.2f}')
"
```

## What you should see

```text
1. targets per image: smallest 92  largest 148
2. matched points in 5 frames: 13 seen by 4 cameras, 202 by 3 cameras
3. trajectories: 8, length from 5 to 5 frames
4. smoothness: median step 3.9 mm, median change of the step 8.2 mm, ratio 2.10
```

## How to read the four numbers

| Check | What it tells you | Good | Bad |
|---|---|---|---|
| 1. targets per image | detection is steady (page 4) | similar numbers in all images | one camera or frame far lower than the rest |
| 2. points by 4 and 3 cameras | matching works and the cameras agree (page 5) | a large share seen by 4 cameras | almost none seen by 4 |
| 3. trajectories and their length | tracking follows particles (page 6) | many long trajectories | many very short ones |
| 4. smoothness ratio | the tracks are physical | below 1, better below 0.5 | 1 or more |

**The smoothness ratio** compares the change of a particle's step from one frame to the next with the step itself. A real
particle in a smooth flow changes its step only a little. If the change is as large as the step, the track zigzags. That
means wrong links, or position errors as large as the movement.

## What this example tells you

This small example **fails check 4**: the ratio is 2.1. The tracks in the picture of page 6 zigzag. This is normal for such a
small crop: only 43 points per frame, and few of them are seen by 4 cameras. For a real measurement, you would not use these
tracks.

For comparison, the bigger example `test_data/test_cavity` (same experiment, full images, 5 frames, 1600 points per frame)
gives a ratio of 0.57. The step there is 0.3 mm, about as large as the position noise, so even that ratio is limited by noise.
The lesson: check 4 depends on the particle speed. Slow particles, with a step close to the noise, always give a high ratio.
Then use more frames and a smoothing fit, not the raw steps.

The four checks are a quick filter, not a proof. [Page 8](08-calibration.md) adds a check of the calibration itself.

## Is the calibration good?

The four checks above look at the result. They cannot say whether the cameras agree with each other. That is the job of
[page 8](08-calibration.md): the reprojection error looks at one camera at a time, and the camera-switching jump looks at all of
them together.

## If it did not work

| You see | Cause | What to do |
|---|---|---|
| check 1 differs a lot between cameras | the images are different (light, focus) | look at the images (page 2); set a threshold per camera |
| check 2 shows almost no 4-camera points | one camera is badly calibrated or not synchronised | check the camera files (page 3), and that all cameras show the same frame |
| check 3 shows only short trajectories | the allowed step is too small, or the seeding is too dense | change `dvxmax` and the others (page 6) |
| check 4 is 1 or more | wrong links, or a step as small as the noise | check page 5 for ghosts; use more frames |

[Next: 8. Is the calibration good? →](08-calibration.md){ .md-button .md-button--primary }
