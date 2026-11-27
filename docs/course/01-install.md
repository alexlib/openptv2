# 1. Install and check that it works

**Time:** 5 minutes. **You need:** Python 3.11, 3.12 or 3.13, and a terminal.

## What this step does

Installs the program and runs a built-in test that tells you whether everything it needs is present, and whether the fast
(compiled) version is in use.

## Do it

=== "Command line"

    ```bash
    pip install "openptv2[gui]"
    ```

    Leave out `[gui]` on a server that has no screen. If you work on the source code instead, see
    [building from source](../developer_guide/building.md).

    Get the example data (it is in the source repository, not in the pip package):

    ```bash
    git clone https://github.com/alexlib/openptv2
    cd openptv2
    ```

    Everything in the course is run from this folder.

=== "Graphical interface"

    After the same `pip install`, start the window:

    ```bash
    openptv2-gui
    ```

    A window with a menu bar opens (Start, Preprocess, Sequence, Tracking, ...). Close it again; the next pages use it.

## What you should see

```bash
openptv self-check
```

ends with a summary line like

```text
Summary: 18 passed, 0 failed, 2 warnings (Total: 20)
```

and, higher up, `Engine: cython3-pure-python (compiled)`.

## Check that it worked

* **0 failed** is what matters. Warnings about *Numba* or *PyTorch* are about optional extras that this course does not use.
* `cython_compiled ... (compiled)` means you have the fast version. If it says *interpreted*, everything still works but
  runs 50 to 100 times slower on real data; reinstall from a wheel for your platform.

## If it did not work

| You see | Cause | Do |
|---|---|---|
| `command not found: openptv` | the folder of pip's scripts is not on your PATH | run `python -m openptv2.cli self-check` instead |
| `No module named PySide6` when opening the window | installed without `[gui]` | `pip install "openptv2[gui]"` |
| a failed line starting `dep_` | one dependency is missing or too old | the line names it; `pip install` that package |

[Next: look at your data →](02-your-data.md){ .md-button .md-button--primary }
