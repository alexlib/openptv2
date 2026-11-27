# Many runs with one calibration, and running on a server

**You have:** several recordings made with the same cameras (repeated trials, several samples), or a computer without a screen.
**You want:** to process them all the same way, without the graphical interface.
**Time:** 15 minutes. **Tested:** the commands in sections 2 and 3 run in CI on `test_data/test_cavity_small`. Section 4 and the server
install are not run in CI.

Course page [2](../course/02-your-data.md) explains the folder and the YAML file.

## 1. The one rule

**Every run has its own folder.** The program writes its results to `res/` inside the folder of the YAML file, and the name
is fixed. Two runs in one folder overwrite each other without any warning. Separate folders can run at the same time.

```text
experiment/
  shared_calibration/     one calibration for all runs (the .ori and .addpar files, the calibration images)
  run1/  img/  parameters_Run1.yaml   res/ (made by the program)
  run2/  img/  parameters_Run1.yaml   res/
```

## 2. Two runs

Two folders from the bundled example stand in for two trials:

```bash
cp -r test_data/test_cavity_small trial_a
```

```bash
cp -r test_data/test_cavity_small trial_b
```

One command per run (a loop in the shell does the same):

```bash
openptv2-batch --workdir trial_a --first 10000 --last 10004 --mode both
```

```text
Average over sequence, particles: 43.0, links: 20.8, lost: 22.2
Sealed: {'n_trajectories': 8, 'n_rows': 40, 'n_dropped': 0}
Batch processing completed successfully
```

```bash
openptv2-batch --workdir trial_b --first 10000 --last 10004 --mode both
```

```text
Average over sequence, particles: 43.0, links: 20.8, lost: 22.2
Sealed: {'n_trajectories': 8, 'n_rows': 40, 'n_dropped': 0}
Batch processing completed successfully
```

## 3. What you should see

Each folder now has its own `res/run.zarr`. This command lists them:

```bash
python -c "import pathlib; print(sorted(str(p) for p in pathlib.Path('.').glob('trial_*/res/run.zarr')))"
```

```text
['trial_a/res/run.zarr', 'trial_b/res/run.zarr']
```

For many runs, the script `scripts/run_pipeline_multi.sh <experiment_root>` finds all folders that contain a YAML file and runs
them one after another, or at the same time with `--parallel`. It writes one log per run. Check that every log ends with
`Batch processing completed successfully`, and that the correspondences per frame are in the hundreds (near zero in every
frame means the calibration or the detection is wrong, not noise).

## 4. Sharing one calibration: the checklist

Each point cost someone a day.

1. **Is it really the same rig?** Compare the first line of `cam1.tif.ori` (the camera position in mm) of the shared calibration
   with the calibration of every run. If a camera differs by tens of mm, the rig moved: do not share. A wrong calibration loads
   without any error and still gives plausible counts. Nothing in the pipeline warns you.
2. **Same way of recording.** A calibration made from four separate images is not automatically valid for a run that puts four
   views on one sensor (`ptv.splitter: true`). The camera numbers must mean the same cameras.
3. **Relative paths.** Paths in the YAML are relative to the YAML file. From `run1/` the shared files are `../shared_calibration/cam1.tif.ori`.
4. **Frame numbers.** `sequence.first` and `sequence.last` must be the numbers in your real file names. Check with
   `ls run1/img | sort | sed -n '1p;$p'`. A wrong number fails on the first frame.
5. **One YAML per purpose.** Keep a `_sample` file (ten frames, for a quick look) and a `_batch` file (all frames), and nothing else.
   Run the sample first, in the graphical interface and in the command line, and look at it before you run the whole batch.
6. **Check the calibration once, for all runs:** [page 8](../course/08-calibration.md) on one of the runs, and
   `scripts/calibration_diagnostics.py` on the shared folder (it checks the camera positions and viewing directions, and exits with an
   error when something is off).

## 5. On a server (no screen)

Install without the graphical interface. This is the default install:

```bash
uv venv /opt/venv
VIRTUAL_ENV=/opt/venv uv pip install .
```

Then check it, as on [page 1](../course/01-install.md), and run `openptv2-batch /data/exp1/parameters_Run1.yaml` (the frame range
is taken from the YAML), or give a range after the file name. The program prints a line per frame. Full details:
[Cloud-like batch](../cloud-batch.md) and [Multi-folder runs](../multi-folder-runs.md).

## If it did not work

| You see | Cause | What to do |
|---|---|---|
| `FileNotFoundError` on the first frame | `sequence.base_name` or `first` do not match the file names | check point 4 |
| results of run 2 look like run 1 | both used one folder | give every run its own folder |
| hundreds of frames with near-zero matches | wrong shared calibration, or thresholds | check points 1 and 6, then [page 4](../course/04-detect.md) |

## Next

[Cloud-like batch](../cloud-batch.md), [Multi-folder runs](../multi-folder-runs.md).
