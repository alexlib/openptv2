# Tutorial: Command-Line Batch Processing with OpenPTV2

In this tutorial, you will learn how to run Particle Tracking Velocimetry (PTV) on image sequences using the `openptv2-batch` command-line interface (CLI). 

The batch processing utility is designed for high-performance automation, running the unified **Cython 3 Pure Python** engine headlessly. This is ideal for large datasets, overnight runs, or remote cluster deployments where a graphical user interface (GUI) is not available.

---

## Prerequisites

Ensure that you have installed OpenPTV2 using `uv` (as managed in this project):
```bash
# Setup development dependencies if not already done
uv sync --extra dev
```

To run the commands in this tutorial, we will use the `test_cavity` dataset included in the repository's test suite under `test_data/test_cavity`.

---

## CLI Command Structure

The general syntax of the batch processor command is:
```bash
uv run openptv2-batch <experiment_directory_or_yaml> <first_frame> <last_frame> [options]
```

### Key Positional Arguments:
* `<experiment_directory_or_yaml>`: The absolute or relative path to either the directory containing your experiment files, or a specific `.yaml` configuration file (e.g., `parameters_Run1.yaml`). If a directory is specified, the CLI will automatically search for and select the first available YAML parameter file.
* `<first_frame>`: The first frame index of the sequence to process.
* `<last_frame>`: The last frame index of the sequence to process.

### Major Optional Flags:
* `--mode <both|sequence|tracking>`:
  * `both` (Default): Runs the entire 3D-PTV pipeline (image preprocessing, target detection, stereo-correspondence, and trajectory tracking).
  * `sequence`: Runs the sequence loop only (preprocessing, detection, and stereo-correspondence), writing output files into `res/rt_is.<frame>`.
  * `tracking`: Runs the tracking engine only, using pre-existing 3D correspondence files (`res/rt_is.<frame>`) and 2D target files (`img/cam*.<frame>_targets`).
* `--track3d`: Enables 3D segment tracking instead of standard epipolar tracking.
* `--sequence-plugin <name>` / `--tracking-plugin <name>`: Use an alternate
  processing strategy instead of the core pipeline (`default`) — for example
  an image-splitter dataset:
  ```bash
  uv run openptv2-batch test_data/test_splitter 1000001 1000002 \
    --sequence-plugin splitter_sequence --tracking-plugin splitter_tracking
  ```
  See the [Plugins tutorial](plugins.md) for the full list of built-ins and
  how to write your own.

---

## Guided Walkthrough: Cavity Flow Dataset

Let's walk through how to run batch tracking on the `test_cavity` dataset located in `test_data/test_cavity`.

### 1. Identifying the Frame Range
First, check the sequence parameters configured for the dataset. Viewing `test_data/test_cavity/parameters/sequence.par` or `test_data/test_cavity/parameters_Run1.yaml` reveals:
```yaml
sequence:
  first: 10001
  last: 10004
```
Our sequence spans from frame **10001** to **10004**.

### 2. Standard Tracking Mode (`--mode tracking`)
Experimental PTV datasets often contain fine-tuned 2D target and 3D correspondence files generated or corrected manually. When you have high-quality pre-existing `rt_is` correspondence files and wish to run/re-run only the tracking step, use `--mode tracking`.

Let's clean and restore the original `test_cavity` sequence data, then run the tracking batch processor:

```bash
# Clean up any previously generated targets and results
rm -rf test_data/test_cavity/img test_data/test_cavity/res

# Restore original targets and correspondence files
cp -rf test_data/test_cavity/img_orig test_data/test_cavity/img
cp -rf test_data/test_cavity/res_orig test_data/test_cavity/res

# Run the batch tracker
uv run openptv2-batch test_data/test_cavity 10001 10004 --mode tracking
```

#### Expected Output:
```text
Starting batch processing
Using tracking engine: cython3-pure-python
Directory provided. Selected parameter file: /home/user/Documents/GitHub/openptv2/test_data/test_cavity/parameters_Run1.yaml
Starting batch processing with YAML file: /home/user/Documents/GitHub/openptv2/test_data/test_cavity/parameters_Run1.yaml
Frame range: 10001 to 10004
Repetitions: 1
Experiment directory: /home/user/Documents/GitHub/openptv2/test_data/test_cavity

Loaded calibration for camera 1 from cal/cam1.tif.ori
Loaded calibration for camera 2 from cal/cam2.tif.ori
Loaded calibration for camera 3 from cal/cam3.tif.ori
Loaded calibration for camera 4 from cal/cam4.tif.ori

Initializing tracker only (skipping sequence)...
Initializing Tracker with parameters:
[ENGINE] Using single Cython 3 tracker runtime
Running Standard Epipolar Tracking only...
step: 10001, curr: 672, next: 699, links: 447, lost: 225, add: 0
step: 10002, curr: 699, next: 711, links: 492, lost: 207, add: 0
step: 10003, curr: 711, next: 692, links: 439, lost: 272, add: 0
step: 10004, curr: 692, next: 0, links: 426, lost: 266, add: 0
Average over sequence, particles: 924.7, links: 601.3, lost: 323.3
Batch processing completed successfully
Total processing time: 1.41 seconds
```

The output indicates that:
* At step 10001, out of 672 active particles, **447** successful forward links (trajectories) were established.
* At step 10002, **492** links were found.
* At step 10003, **439** links were found.
* A total of **1,378 unique trajectory links** were established across the sequence, matching the standard benchmark exactly.

---

### 3. Full Pipeline Mode (`--mode both`)
If you want to perform particle detection, stereomatching, and tracking from raw images entirely from scratch:
```bash
uv run openptv2-batch test_data/test_cavity 10001 10004 --mode both
```

> [!IMPORTANT]
> Running the full pipeline from scratch requires highly accurate detection thresholds (e.g., grey-value thresholds, pixel size limits) configured in `parameters_Run1.yaml` (`targ_rec` and `detect_plate` sections). If the default values are not tuned, particle detection may yield too few or noisy targets, leading to low stereomatching quality and poor tracking links. When processing new raw images, use the `openptv2-gui` first to fine-tune your detection parameters visually before deploying batch runs!

---

## Benefits of the Cython 3 Single-Engine Batch Mode
1. **Headless Execution**: No graphics context, display server (X11/Wayland), or GUI libraries are loaded. It runs perfectly over SSH and within Docker containers.
2. **Compiled Execution Speed**: By utilizing the unified Cython 3 engine, the mathematical loops run compiled at C-speeds, processing hundreds of particles per frame in milliseconds.
3. **Reproducibility**: Parameter configs are completely standardized in human-readable YAML files, ensuring your scientific tracking experiments are 100% reproducible across different systems.
