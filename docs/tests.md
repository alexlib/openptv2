# Running and Managing Tests

OpenPTV2 uses `pytest` for all Python-level tests, `ctest` for the underlying native C library, and automated build pipelines for wheel verification.

---

## 1. Test Suite Structure

The test suite is structured as follows:
- **`algorithms/tests/`**: Unit tests for the core algorithms (segmentation, calibration, tracking).
- **`bindings/tests/`**: Verifies that the legacy Cython binding APIs operate exactly like the new unified algorithms.
- **`gui/tests/`**: End-to-end and unit tests verifying interface parameters, window rendering, and visualization layouts.
- **`tests/integration/`**: Verification of full tracking sequences on actual experimental datasets.
- **`tests/engine_comparison/`**: Asserts mathematical output equivalence (within `1e-7`) between the Cython pure-python algorithms and the original C core.

---

## 2. Command Line Execution

### Running the Standard Test Suite (Fast)
To run all standard unit tests and quickly verify your environment, use:
```bash
# Runs the full test suite while automatically bypassing heavy/slow integration tests
uv run pytest -v
```

### Excluding Slow Integration Tests
Some tracking integration tests read multi-megabyte TIFF files and perform complex search calculations. These take around 1.5 minutes to complete and are decorated with `@pytest.mark.slow`.

To guarantee a fast test run (under ~70 seconds), explicitly exclude slow tests:
```bash
uv run pytest -m "not slow" -v
```

To run *only* those heavy integration tests:
```bash
uv run pytest -m "slow" -v
```

### Running Specific Test Categories

=== "Algorithms Unit Tests"
    ```bash
    uv run pytest algorithms/tests/ -v
    ```

=== "Bindings Compatibility Tests"
    ```bash
    uv run pytest bindings/tests/ -v
    ```

=== "Headless GUI Tests"
    To run the GUI test suite without physically launching a Tkinter screen (using virtual framebuffers):
    ```bash
    uv run pytest gui/tests/ --headless
    ```

=== "Engine Comparison Tests"
    ```bash
    uv run pytest tests/engine_comparison/ --validate-engine
    ```

---

## 3. Running Native C Core Tests (CMake + CTest)

If you have modified the underlying C code inside `lib/` and want to compile and verify the native C tests:

```bash
cd lib
mkdir -p build && cd build
cmake ..
make
ctest -V
```

This compiles and runs the raw C assertions independently from the Python/Cython binding wrappers.

---

## 4. Binary Wheel Build & Verification Pipeline

We maintain a dedicated pipeline to compile binary wheels using `cibuildwheel` locally and verify that they install cleanly and run compiled native code in clean virtual environments.

### Complete Build and Test Pipeline
To compile binary wheels for your active platform and run verification tests on it:
```bash
uv run python scripts/wheel_test_pipeline.py
```
This script will:
1. Initialize `cibuildwheel` to target the active Python version (e.g. `cp313` inside Docker on Linux).
2. Save compiled `.whl` files into the `wheelhouse/` directory.
3. Build a temporary clean virtual environment completely isolated from your project directory.
4. Install the precompiled binary wheel inside the isolated venv.
5. Run Step 4 imports to assert `is_compiled() is True` (verifying optimized Cython execution).
6. Run the fast unit test suite inside that isolated venv.

### Testing Already Built Wheels
If you have already built wheels inside `wheelhouse/` and want to skip compilation, verifying only installation and test behaviors:
```bash
uv run python scripts/wheel_test_pipeline.py --skip-build
```
This is an incredibly fast way to verify packaging modifications.
