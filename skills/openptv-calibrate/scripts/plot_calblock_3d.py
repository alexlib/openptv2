# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.5.1",
#     "matplotlib",
#     "pyyaml==6.0.3",
# ]
# ///
"""Interactive 3D view of a calibration body (fixp/calblock), IDs labeled.

Runs directly from this skills/ checkout -- no per-dataset copy needed. Pass
the target via --target after `--` (marimo's CLI-args convention): a
calibration folder or dataset dir, a parameters_*.yaml, or a calblock .txt
directly. Falls back to `mo.notebook_dir()` if omitted.

Unlike visualize_calibration_setup.py (which plots the calblock *alongside*
camera poses and therefore needs existing cam_N.tif.ori files), this one
only needs the calblock -- useful on a dataset that has no calibration yet.

    uv run marimo edit --sandbox --no-token \\
        skills/openptv-calibrate/scripts/plot_calblock_3d.py \\
        -- --target "<dataset-or-yaml-or-calblock.txt>"

Drag directly on the 3D plot with the mouse to rotate (mo.mpl.interactive,
backed by matplotlib's real WebAgg backend -- not slider-driven).
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    return Path, mo, np, plt


@app.cell
def _(Path, mo):
    def resolve_calblock(arg: str) -> Path:
        """Accept a dataset dir, a parameters_*.yaml, or a calblock .txt directly."""
        from openptv2.autocalibration import resolve_calblock as _resolve

        p = Path(arg).resolve()
        if p.is_file() and p.suffix in (".yaml", ".yml"):
            return _resolve(p.parent)
        if p.is_file():
            return p  # already a calblock file
        return _resolve(p)

    _target_arg = mo.cli_args().get("target")
    target_dir = Path(_target_arg).resolve() if _target_arg else mo.notebook_dir()
    calblock_path = resolve_calblock(str(target_dir))
    return (calblock_path,)


@app.cell
def _(calblock_path, np):
    def read_calblock(path):
        """Parse 'id x y z' lines -> (ids[n], xyz[n,3]), skipping blank/short lines."""
        ids, pts = [], []
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 4:
                ids.append(int(float(parts[0])))
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(ids), np.array(pts)

    ids, body = read_calblock(calblock_path)
    return body, ids


@app.cell
def _(body, calblock_path, ids, mo, np, plt):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(body[:, 0], body[:, 1], body[:, 2], s=25, c=body[:, 2], cmap="viridis")
    for pid, (bx, by, bz) in zip(ids, body):
        ax.text(bx, by, bz, str(pid), fontsize=9)

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title(f"{calblock_path.name} -- {len(ids)} points  (drag to rotate)")
    try:
        ax.set_box_aspect(
            (
                float(np.ptp(body[:, 0])),
                float(np.ptp(body[:, 1])),
                float(np.ptp(body[:, 2])),
            )
        )
    except (AttributeError, ValueError):
        pass
    plt.tight_layout()
    mo.mpl.interactive(fig)
    return


if __name__ == "__main__":
    app.run()
