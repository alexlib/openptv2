# GUI Testing Checklist

This checklist is designed to verify the stability, accuracy, and performance of the modern Tkinter-based GUI using the unified Cython 3+ compiled engine runtime.

---

## 🛠️ Pre-Test Setup

Ensure you are in the repository root and have the correct virtual environment activated:

```bash
cd /home/user/Documents/GitHub/openptv2
# Check git status to ensure you have a clean workspace
git status
```

Verify that the application packages import successfully and point to the consolidated engine:

```bash
uv run python -c "import openptv2; print('OpenPTV2 unified engine: OK')"
```

---

## 🚀 Step-by-Step GUI Workflow Verification

Follow this checklist to verify that all major GUI workflows complete without any C-level crashes or segmentation faults.

### Test 1: Launch the GUI

Start the GUI using the standard execution command:

```bash
uv run pyptv_gui
```

*Or via the explicit module path:*
```bash
uv run python -m openptv2.gui.pyptv.pyptv_gui
```

**Expected Results:**
- [ ] The Tkinter GUI window opens successfully.
- [ ] No module-import errors or startup tracebacks appear in the terminal.

---

### Test 2: Load an Existing Project

With the GUI window open:

1. Go to **File** ➔ **Open Project**.
2. Navigate to one of the following directories:
   - `test_data/synthetic/`
   - `test_data/burgers/`
   - `test_data/test_cavity/`
3. Select the `parameters/` directory and click **OK**.

**Expected Results:**
- [ ] The project loads without raising any validation exceptions.
- [ ] The status bar successfully indicates loaded parameters.
- [ ] The camera panels display the correct number of cameras configured in the parameters.

---

### Test 3: Run Particle Detection (Segmentation)

1. Go to **Segmentation** ➔ **Detection**.
2. Keep or set the target frame number (e.g., `10001` or `10000`).
3. Click **Detect**.

**Expected Results:**
- [ ] Detection runs smoothly, showing a progress bar.
- [ ] Detected targets are displayed and highlighted in the camera views.
- [ ] The target counts are updated in the status bar and match expected targets.

---

### Test 4: Find Correspondences (3D Reconstruction)

1. Go to **Tracking** ➔ **Correspondences**.
2. Click **Find Correspondences**.

**Expected Results:**
- [ ] Correspondences are successfully calculated from the multi-camera inputs.
- [ ] The 3D coordinates are plotted and visible.
- [ ] No array transformation errors or C-level casting exceptions occur.

---

### Test 5: Run Track Forward (Sequence Tracking)

1. Go to **Tracking** ➔ **Track Forward**.
2. Verify the frame range is correct for the dataset (e.g., `10000` to `10002` or similar short sequence).
3. Click **Start Tracking**.

**Expected Results:**
- [ ] The tracking sequence completes successfully without any crashes.
- [ ] A progress bar monitors the tracking frames.
- [ ] All trajectory and particle files are created inside the project's `res/` directory (e.g., `res/ptv_is.*`, `res/rt_is.*`).

---

## 📊 Verification of Outputs

After finishing the sequence run, verify that the files are properly generated in the terminal:

```bash
# Check the generated result files (example for synthetic dataset)
ls -lh test_data/synthetic/res/

# Confirm files are not empty and have generated lines
wc -l test_data/synthetic/res/rt_is.10001
```

**Expected Results:**
- [ ] Directory `res/` is populated.
- [ ] Result files have reasonable sizes and contain valid coordinate rows.

---

## 📓 Notebook Verification (Optional)

If using interactive Marimo or Jupyter notebooks to inspect parameters and tracking sequences:

```bash
uv run jupyter notebook notebooks/
```

Open a notebook and confirm that imports and basic operations run directly from the consolidated namespace:

```python
from openptv2 import Calibration, Tracker
import openptv2
# Verify tracking calibration can be loaded directly
```

---

## 🛠️ Reporting Issues

If any step raises an error or crashes:
1. **Identify the exact step** (Test 1, 2, 3, etc.) where the issue occurred.
2. **Copy the full traceback** from the terminal.
3. **Describe the action** (e.g., loaded specific file `Y` or clicked button `X`).
