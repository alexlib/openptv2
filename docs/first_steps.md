# First Steps with OpenPTV2

This guide gets you started using OpenPTV2 both via Python scripting and through the graphical user interface.

---

## 1. Obtaining Sample Data

For a complete, real-world sample dataset to explore, it is highly recommended to download the official `test_cavity` case from Git using:
```bash
git clone https://github.com/openptv/test_cavity
```

You can then launch the OpenPTV2 GUI directly pointing to this folder:
```bash
uv run openptv2-gui ./test_cavity
```

*(Note: A small mock version of this dataset is also pre-included inside your cloned repository root under `test_data/test_cavity` for running internal unit tests.)*

---

## 2. Programmatic Scripting Tutorial

Here is a step-by-step walk-through of loading calibration parameters, performing particle detection, matching multi-camera correspondences, and tracking particles in 3D using Python.

### Step 1: Import OpenPTV2 and Check Runtime Info
```python
import openptv2

# Verify that we are running the compiled Cython binary library
info = openptv2.get_runtime_info()
print(f"Engine: {info['engine']}")
print(f"Compiled (Cython): {info['compiled']}")

assert info['compiled'] is True, "Warning: Running in uncompiled mode!"
```

### Step 2: Load Calibration Parameters
To map 2D camera pixels to 3D physical coordinates, we load the calibration parameters.

```python
from openptv2.calibration import Calibration
import numpy as np

# Load calibration files for a camera
cal = Calibration()
cal.from_file(
    ori_file="test_data/test_cavity/cal/cam1.ori",
    addpar_file="test_data/test_cavity/cal/cam1.addpar"
)

print("Camera Position (3D physical):", cal.get_pos())
print("Camera Angles (rad):", cal.get_angles())
```

### Step 3: Detect Targets (2D Segmentation)
Identify particles in each camera's raw TIFF images.

```python
from openptv2.segmentation import detect_targets
import skimage.io

# Load camera frame
img = skimage.io.imread("test_data/test_cavity/img/cam1_10000.tif")

# Detect particles with a threshold of 10
targets = detect_targets(img, threshold=10)

print(f"Detected {len(targets)} particle targets.")
for i, t in enumerate(targets[:3]):
    print(f"Target {i}: Pixel x={t.x:.2f}, y={t.y:.2f}, Area={t.p_n}px")
```

### Step 4: Multi-Camera Correspondence (Epipolar Matching)
Combine 2D targets from all 4 cameras to find 3D physical coordinate correspondences.

```python
from openptv2.correspondence import establish_correspondences

# Assume we have target lists for all four cameras: targets1, targets2, targets3, targets4
# and their respective calibrations loaded in list: cals

# correspondences = establish_correspondences(
#     [targets1, targets2, targets3, targets4],
#     cals,
#     ptv_params
# )
```

### Step 5: Run 3D Tracking Sequence
Using the `Tracker` object, we track particles across sequential frames.

```python
from openptv2 import Tracker

# Initialize the tracker with parameter file
tracker = Tracker(parameter_file="test_data/test_cavity/parameters_Run1.yaml")

# Run sequence tracking on a small frame range
tracks = tracker.track(first_frame=10000, last_frame=10005)

print(f"Completed tracking! Found {len(tracks)} continuous trajectories.")
```

---

## 3. First Steps with the GUI

The modern OpenPTV2 GUI provides interactive visual parameter configuration, target threshold adjustments, calibration optimization, and live tracking preview.

### Step 1: Launch the GUI
Run the command-line shortcut in your terminal (with activated virtual environment or prepended by `uv run`):

```bash
# Launch the main Tkinter/ttkbootstrap interface
uv run openptv2-gui -w ./test_data/test_cavity
```

### Step 2: Initialize parameters
1. Once the GUI opens, select **Start → Init / Reload** from the top menu bar.
2. This loads the calibrations, image paths, and tracking settings specified in the configuration directory.
3. Check the command-line log output to ensure parameters load cleanly:
   ```text
   Read all the parameters and calibrations successfully
   ```

### Step 3: Run Particle Detection Preview
1. Select **Preprocess → Image coord**.
2. The GUI executes particle detection algorithms on the current frame.
3. Camera display tabs will populate with blue cross markers representing successfully segmented particles.

### Step 4: Run Tracking Sequence
1. Select **Tracking → Track Sequence**.
2. The tracker runs multi-camera tracking.
3. You can watch the real-time link counter updating in the status panel.

---

## 4. First Steps with Command-Line Batch Processing

If you are running large datasets and do not want to use the GUI, run the batch processing script `pyptv_batch`:

```bash
# Process frames 10000 to 10005 using the YAML configuration
uv run pyptv_batch --workdir=./test_data/test_cavity --first=10000 --last=10005
```

This runs detection, correspondences, and tracking, exporting the resulting tracked trajectories to the experimental output directory.
