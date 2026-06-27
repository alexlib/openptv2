# Running and Managing Tests

OpenPTV2 uses `pytest` for all tests. Since migrating to the Cython 3 Pure Python architecture, the test suites have been completely streamlined.

---

## 1. Test Suite Structure

The tests are organized under the main `/tests/` directory:
- **`tests/unit/`**: Light and fast unit tests for each individual algorithmic module (trafo, imgcoord, correspondences, tracker compatibility, etc.).
- **`tests/batch/`**: Functional and integration tests for CLI batch sequences, parallelization commands, and tracking parameters optimizations.
- **`tests/gui/`**: End-to-end and functionality tests for the user interface, parameter managers, dialog frames, and window states (headless compatible).
- **`tests/parity/`**: Validates value-level parity, C-comparisons, and runtime indicators.

---

## 2. Running Tests with uv

We recommend utilizing `uv` for python testing to automatically resolve dependencies in the active virtual environment context.

### Running the Entire Test Suite
To run all tests (excluding slow tracking integration runs):
```bash
uv run pytest -v
```

### Excluding Slow Integration Tests
Several integration tests process multi-megabyte raw TIFF files and perform complex multi-frame particle matching. These are marked as slow. To exclude them for instant feedback under ~20 seconds:
```bash
uv run pytest -m "not slow" -v
```

### Running Specific Test Categories

=== "Unit Tests"
    ```bash
    uv run pytest tests/unit/ -v
    ```

=== "Batch Processing Tests"
    ```bash
    uv run pytest tests/batch/ -v
    ```

=== "GUI Interface Tests"
    To run the GUI tests without physically rendering a window on your desktop screen:
    ```bash
    uv run pytest tests/gui/ -v
    ```

=== "Parity Verification"
    ```bash
    uv run pytest tests/parity/ -v
    ```

---

## 3. Test Suite Performance Optimizations

Our batch and optimization tests are highly optimized:
- **Redundancy Removal**: We eliminated redundant sequence-mode file pre-generations in tracking perturbation runs. Because tracking parameter updates do not change camera detection results, skipping these steps optimized execution time down from **178s to 33s**.
- **Unified Pathing**: Integrated automated environment helpers inside subprocess tests to inject correct relative workspace paths, preventing module-not-found errors during child process runs.
- **Parallel Workers Context**: Configured multiprocessing to select `'fork'` over `'spawn'` on Linux/macOS, entirely avoiding Python pytest-sandbox deadlocks.
