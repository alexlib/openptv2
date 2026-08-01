# /// script
# dependencies = [
#     "marimo",
#     "numpy"
# ]
# ///

# ruff: noqa: E501
import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys

    import numpy as np

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from openptv2.gui.pyptv.ptv import read_rt_is_file

    return np, os, read_rt_is_file


@app.cell
def _(os):
    def list_files_in_and_above(current_dir: str = ".") -> dict[str, list[str]]:
        """Lists files in the current directory and its parent directory."""
        files_map = {}

        # Current directory
        files_map["current_dir"] = [
            f
            for f in os.listdir(current_dir)
            if os.path.isfile(os.path.join(current_dir, f))
        ]

        # Parent directory
        parent_dir = os.path.dirname(os.path.abspath(current_dir))
        files_map["parent_dir"] = [
            f
            for f in os.listdir(parent_dir)
            if os.path.isfile(os.path.join(parent_dir, f))
        ]

        return files_map

    file_listings = list_files_in_and_above(".")
    print(f"Files in current directory: {file_listings['current_dir']}")
    # print(f"Files in parent directory: {file_listings['parent_dir']}")
    return


@app.cell
def _():
    data_dir = "./test_data/burgers/res_orig"
    return (data_dir,)


@app.cell
def _(data_dir, np, os, read_rt_is_file):
    # Load frames
    frames = []
    for i in range(10001, 10006):
        fname = os.path.join(data_dir, f"rt_is.{i:05d}")
        arr = read_rt_is_file(fname)
        arr = np.array(arr)  # shape (N, 7)
        frames.append(arr[:, :3])  # only x, y, z
    print(
        f"Loaded {len(frames)} frames, each with {[f.shape[0] for f in frames]} particles"
    )

    # Per-frame interparticle distance
    for idx, arr in enumerate(frames):
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        vol = np.prod(maxs - mins)
        n = arr.shape[0]
        ipd = (vol / n) ** (1 / 3) if n > 0 else np.nan
        print(
            f"Frame {idx}: Volume={vol:.2f} mm^3, N={n}, Interparticle dist~{ipd:.2f} mm"
        )

    # Merge 4 frames for fictitious density
    merged = np.vstack(frames[:4])
    mins = merged.min(axis=0)
    maxs = merged.max(axis=0)
    vol = np.prod(maxs - mins)
    n = merged.shape[0]
    ipd_merged = (vol / n) ** (1 / 3) if n > 0 else np.nan
    print(
        f"Merged 4 frames: Volume={vol:.2f} mm^3, N={n}, Interparticle dist~{ipd_merged:.2f} mm"
    )

    # Inter-frame displacement
    all_disp = []
    for i in range(4):
        arr1, arr2 = frames[i], frames[i + 1]
        if arr1.shape == arr2.shape:
            d = np.linalg.norm(arr2 - arr1, axis=1)
            all_disp.append(d)
        else:
            print(
                f"Frame {i} and {i + 1} have different particle counts; skipping displacement calc."
            )
    if all_disp:
        all_disp = np.concatenate(all_disp)
        print(
            f"Inter-frame displacement: min={all_disp.min():.2f}, max={all_disp.max():.2f}, mean={all_disp.mean():.2f} mm"
        )
    else:
        print("Inter-frame displacement: not computed (mismatched particle counts)")
    return (frames,)


@app.cell
def _(frames, np):
    import matplotlib.pyplot as plt

    def create_3d_particle_plot(frames: list[np.ndarray]) -> plt.Figure:
        """Creates an interactive 3D scatter plot of particle data frames using matplotlib."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for idx, arr in enumerate(frames):
            # Sample if too many points to keep it responsive
            if arr.shape[0] > 5000:
                indices = np.random.choice(arr.shape[0], 5000, replace=False)
                subset = arr[indices]
            else:
                subset = arr

            ax.scatter(
                subset[:, 0],
                subset[:, 1],
                subset[:, 2],
                s=3,
                alpha=0.7,
                label=f"Frame {idx + 10001}",
            )

        ax.set_title("Particle Positions in 3D")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.legend()

        return fig

    # Generate plot
    particle_plot = create_3d_particle_plot(frames)

    import marimo as mo

    mo.mpl.interactive(particle_plot)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
