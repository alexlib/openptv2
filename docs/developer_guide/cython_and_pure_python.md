# OpenPTV2: Developer Guide to Pure Python & Cython 3+ Modes

This guide explains how to leverage OpenPTV2's **Pure Python Cython architecture** to switch seamlessly between two modes:
1.  **Pure Python (Interpreted) Mode:** Perfect for debugging, stepping through algorithms with a standard IDE debugger, quickly testing new mathematical ideas, and inspecting code behavior.
2.  **Cython 3+ (Compiled) Mode:** Translates the code to raw C and compiles it to native machine binary extensions, achieving **up to 100x speedups** for production runs.

---

## 1. The Core Architecture: Pure Python Cython

OpenPTV2 uses Cython 3's **Pure Python mode**. This means:
*   All source code is written in standard `.py` files with regular Python syntax.
*   Type-hints and directives are imported from the `cython` module (e.g., `cython.int`, `cython.double`, `@cython.cclass`).
*   **Double Nature:**
    *   When imported by CPython, the `cython` imports are treated as no-ops, and the `.py` files run as regular, interpreted Python code.
    *   When processed by Cython, the type-hints and decorators are parsed to generate highly optimized, typed C code.

---

## 2. Running in Pure Python Mode (For Debugging & Inspection)

In Pure Python mode, you can use any Python debugger (like standard `pdb`, or the visual debuggers in VS Code, PyCharm, or Cursor) to set breakpoints, step through the math line-by-line, and inspect the values of variables.

### Step 1: Clean Compiled Binary Extensions
Python's import system always prioritizes compiled shared libraries (`.so` or `.pyd` files) over `.py` files. To switch back to Pure Python mode, you must delete any pre-compiled binaries from the source folder:

```bash
# Clean all compiled .so, .pyd, and generated C files
rm src/openptv2/algorithms/*.so
rm src/openptv2/algorithms/*.c
```

Alternatively, you can let Git clean the untracked build artifacts for you:
```bash
git clean -fdx src/openptv2/algorithms/
```

### Step 2: Running and Testing Directly from Source
Once the compiled binaries are gone, any Python process will import and interpret the `.py` source files directly:

```bash
# Run pytest directly on the interpreted Python files
uv run pytest tests/unit/test_epi.py -v
```

### Step 3: Debugging with Standard Breakpoints
You can now open any `.py` file (e.g., `src/openptv2/algorithms/trafo.py`), place a standard Python breakpoint anywhere in the code, and run your script:

```python
def flat_to_dist(x, y, cal):
    # Add a debugger breakpoint here
    breakpoint()
    
    # ... your mathematical logic ...
```

When you run your tests or code, the execution will pause at this line, allowing you to step through the code and print local variable values.

---

## 3. Running in Compiled Mode (For 100x Production Speeds)

To achieve maximum performance, you compile the `.py` files into native compiled binary extensions. 

### Step 1: Compile and Install the Extensions
Our unified `setup.py` build script handles parallel compiling using multiple CPU cores, and compiles the files with high C optimization flags (`-O3` on Linux/macOS, `/O2` on Windows).

To compile the extensions **in-place** (in the source directory) without doing a global installation, run:
```bash
uv run python setup.py build_ext --inplace
```

To compile and install the package into your active virtual environment in **editable development mode**, run:
```bash
uv pip install -e .
```

### What happens under the hood during compilation?
1.  **Cythonization:** `cythonize` translates the annotated `.py` files into optimized C source code files (`.c` files).
2.  **C-Compilation:** Your system's C compiler (GCC/Clang on Linux/macOS, MSVC on Windows) compiles the `.c` files into native compiled binary shared libraries (`.so` files on Linux/macOS, `.pyd` files on Windows) with full `-O3` vectorizations.

---

## 4. How to Programmatically Verify the Active Mode

You can verify whether a module is currently running as interpreted Pure Python or as a compiled Cython extension using the `cython.compiled` property:

```python
import cython
from openptv2.algorithms import trafo

# Verify status
if cython.compiled:
    print("🚀 Running in High-Speed Compiled Cython Mode!")
else:
    print("🔍 Running in Interpreted Pure Python Mode (Perfect for Debugging)")
```

---

## 5. Architectural Comparison

| Feature | Interpreted Pure Python Mode 🔍 | Compiled Cython 3+ Mode 🚀 |
| :--- | :--- | :--- |
| **Execution Engine** | Standard CPython Interpreter | Native Machine Binary |
| **Compilation Required?**| No (just run the `.py` files directly) | Yes (`build_ext` or `pip install -e .`) |
| **Debugging Capabilities**| Line-by-line debugging via `pdb`, VS Code, etc. | Native debuggers (`gdb`/`lldb`) only |
| **Execution Speed** | Standard Python interpreter speed | Up to **100x faster** (C-level performance) |
| **Typical Use-Case** | Finding bugs, prototyping new algorithms. | Production multi-camera tracking runs. |
