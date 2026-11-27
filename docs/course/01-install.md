# 1. Install and check that it works

**Time:** 5 minutes. **You need:** Python 3.11, 3.12 or 3.13, and a terminal.

## What this step does

It installs the program. Then a built-in test checks that everything is in place, and that the fast (compiled) version is used.

## Do it

=== "Command line"

    ```bash
    pip install "openptv2[gui]"
    ```

    On a server without a screen, leave out `[gui]`. To work on the source code, see
    [building from source](../developer_guide/building.md).

    The example data is in the source repository, not in the pip package. Download it:

    ```bash
    git clone https://github.com/alexlib/openptv2
    cd openptv2
    ```

    Run every command of the course from this folder.

=== "Graphical interface"

    Install the same way, then start the window:

    ```bash
    openptv2-gui
    ```

    A window with a menu bar opens (Start, Preprocess, Sequence, Tracking and more). Close it again. You will use it from
    page 2 on.

## What you should see

The program is called **openptv2** (the package name) and its command is `openptv`. Both names mean the same software.

```bash
openptv self-check
```

ends with a line like this:

```text
Summary: 18 passed, 0 failed, 2 warnings (Total: 20)
```

A few lines above it you should see `Engine: cython3-pure-python (compiled)`.

## Check that it worked

* **"0 failed"** is what matters. A warning about *Numba* or *PyTorch* is about optional parts. The course does not use them.
* **"(compiled)"** means you have the fast version. If it says *interpreted*, everything works but real data runs 50 to 100
  times slower. The compiled parts were not built for your platform. Try
  `pip install --upgrade --force-reinstall "openptv2[gui]"`: pip takes a ready-made build when one exists for your system. If it
  has to build from the source, it needs a C compiler; see [building from source](../developer_guide/building.md).

## If it did not work

| You see | Cause | What to do |
|---|---|---|
| `command not found: openptv` | pip's script folder is not on your PATH | run `python -m openptv2.cli self-check` |
| `No module named PySide6` | installed without `[gui]` | `pip install "openptv2[gui]"` |
| a failed line that starts with `dep_` | a package is missing or too old | the line names it. Install it with pip. |

[Next: 2. Your data →](02-your-data.md){ .md-button .md-button--primary }
