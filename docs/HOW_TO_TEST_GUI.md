# How to Test the GUI

This document explains how to test that the OpenPTV2 GUI works correctly under the restructured directory format.

---

## 1. Quick Launch

To launch the desktop GUI from the terminal, run:

```bash
# Using the command-line shortcut
uv run openptv2-gui -w ./test_data/test_cavity

# Or directly running the module
uv run python -m openptv2.gui.pyptv_gui -w ./test_data/test_cavity
```

---

## 2. Automated Tests

All GUI-related tests are located in `tests/gui/` and run seamlessly using pytest.

### Running GUI Tests
```bash
# Run the entire GUI test suite (60+ test files, 230+ test cases)
uv run pytest tests/gui/ -v
```

### Running Specific GUI Tests
```bash
# Test general core interface functionality
uv run pytest tests/gui/test_gui_functionality.py -v

# Test parameter manager loading and YAML transitions
uv run pytest tests/gui/test_parameter_manager_structure.py -v

# Test image path resolution across working directories
uv run pytest tests/gui/test_image_path_resolution_fixed.py -v
```

---

## 3. Scope of Automated GUI Testing

The GUI test suite covers a wide range of components:
- **Module Imports**: Verifies that standard packages import correctly under the unified single-engine runtime.
- **Path Resolution**: Asserts that relative and absolute path conversions resolve successfully across changing working directories.
- **Parameter Translation**: Validates the translation pipeline from human-readable `.yaml` parameter files to structured parameter models.
- **Class Instantiation**: Smoke-tests panel and dialog windows to prevent crash-on-launch regressions.
- **Workflow Simulation**: Simulates the target detection, calibration optimization, and tracking pipelines end-to-end.

---

## 4. Manual Verification Workflow

If you are modifying interactive graphics, UI components, or Matplotlib figures, perform this manual checklist:

1. **Launch the GUI** with the mock dataset:
   ```bash
   uv run openptv2-gui -w ./test_data/test_cavity
   ```
2. **Reload parameters**: Click **Start ➔ Init / Reload** on the top menu bar. Check that the console logs successful loading of cameras and parameters.
3. **Run Segmentation**: Click **Preprocess ➔ Image coord**. Verify that camera image panels render successfully and display blue coordinate crosses marking detected targets.
4. **Run Tracking**: Click **Tracking ➔ Track Sequence**. Check that the status panel updates the tracked links in real-time.
