# /// script
# dependencies = [
#     "marimo",
#     "numpy"
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import os

    return np, os


@app.cell
def _(np, os):
    data_dir = "../../test_data/burgers/res_orig"
    def load_frame_xyz(frame_idx):
        fname = os.path.join(data_dir, f"rt_is.{frame_idx:04d}")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Missing file: {fname}")
        arr = np.loadtxt(fname, usecols=(1,2,3))
        return arr
 

    return (load_frame_xyz,)


@app.cell
def _(load_frame_xyz):
   
    frames = []
    for i in range(10001, 10006):
        frames.append(load_frame_xyz(i))
    
    print(f"Loaded {len(frames)} frames, each with {[f.shape[0] for f in frames]} particles")
    return (frames,)


@app.cell
def _(frames, np):
    for idx, arr in enumerate(frames):
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        vol = np.prod(maxs - mins)
        n = arr.shape[0]
        ipd = (vol / n) ** (1/3) if n > 0 else np.nan
        print(f"Frame {idx}: Volume={vol:.2f} mm^3, N={n}, Interparticle dist~{ipd:.2f} mm")
    return


@app.cell
def _(frames, np):
    merged = np.vstack(frames[:4])
    mins = merged.min(axis=0)
    maxs = merged.max(axis=0)
    vol = np.prod(maxs - mins)
    n = merged.shape[0]
    ipd_merged = (vol / n) ** (1/3) if n > 0 else np.nan
    print(f"Merged 4 frames: Volume={vol:.2f} mm^3, N={n}, Interparticle dist~{ipd_merged:.2f} mm")
    return


@app.cell
def _(frames, np):
    disps = []
    for i in range(4):
        arr1, arr2 = frames[i], frames[i+1]
        if arr1.shape == arr2.shape:
            d = np.linalg.norm(arr2 - arr1, axis=1)
            disps.append(d)
        else:
            disps.append(np.array([]))
    all_disp = np.concatenate(disps) if disps else np.array([])
    if all_disp.size > 0:
        print(f"Inter-frame displacement: min={all_disp.min():.2f}, max={all_disp.max():.2f}, mean={all_disp.mean():.2f} mm")
    else:
        print("Inter-frame displacement: not computed (mismatched particle counts)")
    return


@app.cell
def _():
    """
    # Quick Probe: Interparticle and Inter-frame Distance (Burgers Case)
    This notebook computes quick estimates of interparticle distance and inter-frame displacement using the first Burgers dataset.
    """
    return


if __name__ == "__main__":
    app.run()
